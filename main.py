from __future__ import annotations

import argparse
import shutil
import subprocess
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


class LiteRTLMLangChain(LLM):
    """Minimal LangChain wrapper around the LiteRT-LM CLI."""

    cli_path: str = "litert-lm"
    huggingface_repo: str | None = "google/gemma-3n-E2B-it-litert-lm"
    model_file: str = "gemma-3n-E2B-it-int4"
    extra_args: tuple[str, ...] = ()

    @property
    def _llm_type(self) -> str:
        return "litert-lm-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "cli_path": self.cli_path,
            "huggingface_repo": self.huggingface_repo,
            "model_file": self.model_file,
            "extra_args": self.extra_args,
        }

    @staticmethod
    def _normalize_model_file(model_file: str) -> str:
        if model_file.endswith(".litertlm"):
            return model_file
        return f"{model_file}.litertlm"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        if shutil.which(self.cli_path) is None:
            raise RuntimeError(
                "The 'litert-lm' CLI was not found on PATH. "
                "Install it with: uv tool install litert-lm"
            )

        cmd = [self.cli_path, "run"]
        if self.huggingface_repo:
            cmd.append(f"--from-huggingface-repo={self.huggingface_repo}")
        cmd.append(self._normalize_model_file(self.model_file))
        cmd.extend(self.extra_args)
        cmd.append(f"--prompt={prompt}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() or "LiteRT-LM exited with an unknown error."
            raise RuntimeError(stderr) from exc

        output = result.stdout.strip()
        if stop:
            output = _apply_stop_tokens(output, stop)
        return output


def _apply_stop_tokens(text: str, stop: list[str]) -> str:
    cutoff = len(text)
    for token in stop:
        index = text.find(token)
        if index != -1:
            cutoff = min(cutoff, index)
    return text[:cutoff]


def build_chain(llm: LiteRTLMLangChain):
    prompt = PromptTemplate.from_template(
        "You are a concise assistant.\n"
        "Answer the user's question clearly and directly.\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    return prompt | llm | StrOutputParser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple LangChain pipeline backed by LiteRT-LM."
    )
    parser.add_argument(
        "--question",
        default="What are three warning signs of a bank impersonation scam?",
        help="The user question to send through the LangChain pipeline.",
    )
    parser.add_argument(
        "--repo",
        default="google/gemma-3n-E2B-it-litert-lm",
        help="Hugging Face repo to fetch the LiteRT-LM model from.",
    )
    parser.add_argument(
        "--model-file",
        default="gemma-3n-E2B-it-int4.litertlm",
        help="Model file or alias to pass to 'litert-lm run'. Bare names get .litertlm appended automatically.",
    )
    parser.add_argument(
        "--cli-path",
        default="litert-lm",
        help="Path to the LiteRT-LM CLI binary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    llm = LiteRTLMLangChain(
        cli_path=args.cli_path,
        huggingface_repo=args.repo,
        model_file=args.model_file,
    )
    chain = build_chain(llm)
    response = chain.invoke({"question": args.question})
    print(response)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich import print

from bank_kb import detect_bank_name, load_bank_data

load_dotenv()

INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "openai").lower()


# ---------------------------------------------------------------------------
# System prompt (shared preamble)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a scam detection assistant that analyzes phone call transcripts. "
    "Compare the caller's behavior against the provided bank guidelines to assess scam risk. "
    "If the caller asks for passwords, OTP codes, remote access, or fund transfers to "
    "'safe accounts', flag it as a likely scam regardless of what the guidelines say."
)


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (tool-calling agent)
# ---------------------------------------------------------------------------

def build_openai_backend():
    from langchain.tools import tool as lc_tool
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    model_name = os.getenv("MODEL_NAME", "google/gemma-4-26B-A4B-it")
    base_url = os.getenv("BASE_URL", "http://172.20.8.110:8004/v1")
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy"

    llm = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
    )

    @lc_tool
    def load_banks(bank_name: str) -> str:
        """Load bank-specific scam detection guidelines.

        Pass the bank name mentioned by the caller (e.g. "maybank", "public bank").
        If the bank is not found, returns general scam detection guidelines.

        Returns the bank's procedures, red flags, and known scam patterns.
        """
        return load_bank_data(bank_name)

    agent = create_agent(
        model=llm,
        tools=[load_banks],
        system_prompt=(
            SYSTEM_PROMPT + " "
            "When a caller claims to be from a bank, use load_banks with the bank name "
            "to retrieve that bank's real procedures and known scam patterns. "
            "If no bank is mentioned, use load_banks with 'general' for generic guidelines."
        ),
    )

    return {
        "type": "openai",
        "agent": agent,
        "model_name": model_name,
        "base_url": base_url,
    }


def run_openai(backend, transcript: str) -> list[dict]:
    """Run the tool-calling agent and return a list of turn dicts."""
    response = backend["agent"].invoke(
        {"messages": [{"role": "user", "content": transcript}]}
    )
    turns = []
    for i, msg in enumerate(response["messages"]):
        turn = {"turn": i, "type": msg.type}
        if msg.type == "human":
            turn["content"] = msg.content
        elif msg.type == "ai":
            if msg.tool_calls:
                turn["tool_calls"] = [
                    {"name": tc["name"], "args": tc["args"]}
                    for tc in msg.tool_calls
                ]
            if msg.content:
                turn["content"] = msg.content
        elif msg.type == "tool":
            turn["tool_name"] = msg.name
            turn["content"] = msg.content
        turns.append(turn)
    return turns


# ---------------------------------------------------------------------------
# LiteRT-LM CLI backend (deterministic retrieval + single LLM call)
# ---------------------------------------------------------------------------

class LiteRTLMLangChain:
    """Thin wrapper that shells out to the litert-lm CLI."""

    def __init__(
        self,
        cli_path: str = "litert-lm",
        huggingface_repo: str | None = None,
        model_file: str = "gemma-3n-E2B-it-int4",
    ):
        self.cli_path = cli_path
        self.huggingface_repo = huggingface_repo
        self.model_file = model_file
        if not self.model_file.endswith(".litertlm"):
            self.model_file += ".litertlm"

    def invoke(self, prompt: str) -> str:
        if shutil.which(self.cli_path) is None:
            raise RuntimeError(
                "litert-lm CLI not found on PATH. Install with: uv tool install litert-lm"
            )
        cmd = [self.cli_path, "run"]
        if self.huggingface_repo:
            cmd.append(f"--from-huggingface-repo={self.huggingface_repo}")
        cmd.append(self.model_file)
        cmd.append(f"--prompt={prompt}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr.strip() or "litert-lm error") from exc
        return result.stdout.strip()


LITERT_PROMPT_TEMPLATE = """\
{system}

### Bank Guidelines
{bank_context}

### Transcript
{transcript}

### Analysis
Analyze whether this call is a scam. Consider:
1. Does the caller ask for passwords, OTP, or remote access?
2. Does the caller pressure the victim or create urgency?
3. Does the caller's behavior match real bank procedures?
Give a verdict: SCAM, LEGITIMATE, or UNCERTAIN, then explain why.
"""


def build_litert_backend():
    cli_path = os.getenv("LITERT_LM_CLI_PATH", "litert-lm")
    hf_repo = os.getenv("LITERT_LM_HF_REPO", "google/gemma-3n-E2B-it-litert-lm")
    model_file = os.getenv("LITERT_LM_MODEL_FILE", "gemma-3n-E2B-it-int4")

    llm = LiteRTLMLangChain(
        cli_path=cli_path,
        huggingface_repo=hf_repo,
        model_file=model_file,
    )

    return {
        "type": "litert-lm",
        "llm": llm,
        "model_name": f"{hf_repo}/{model_file}",
        "base_url": "local",
    }


def run_litert(backend, transcript: str) -> list[dict]:
    """Deterministic pipeline: detect bank -> load context -> single LLM call."""
    bank_name = detect_bank_name(transcript)
    bank_context = load_bank_data(bank_name)

    prompt = LITERT_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        bank_context=bank_context,
        transcript=transcript,
    )

    turns = [
        {"turn": 0, "type": "human", "content": transcript},
        {"turn": 1, "type": "info", "content": f"[deterministic] bank detected: {bank_name}"},
    ]

    llm_output = backend["llm"].invoke(prompt)
    turns.append({"turn": 2, "type": "ai", "content": llm_output})
    return turns


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def build_backend():
    if INFERENCE_BACKEND == "openai":
        return build_openai_backend()
    elif INFERENCE_BACKEND in ("litert-lm", "litert_lm", "litert"):
        return build_litert_backend()
    else:
        raise ValueError(f"Unknown INFERENCE_BACKEND: {INFERENCE_BACKEND!r}")


def run(backend, transcript: str) -> list[dict]:
    if backend["type"] == "openai":
        return run_openai(backend, transcript)
    else:
        return run_litert(backend, transcript)


if __name__ == "__main__":
    test_transcripts = {
        # --- MAYBANK: Should load Maybank knowledge ---

        "maybank_obvious_scam": (
            "Caller: Good afternoon, I'm calling from Maybank security department. "
            "We detected that your account has been used for money laundering. "
            "I need your Maybank2u username and password right now to freeze the account. "
            "If you don't cooperate, we will have to transfer you to the police. "
            "Do not hang up or tell anyone about this call."
        ),

        "maybank_secure2u_scam": (
            "Caller: Hi, this is Maybank calling. Your Secure2u has been compromised. "
            "We need you to share the approval code showing on your MAE app so we can "
            "reset it for you. Also please download the updated MAE app from this link "
            "I'm sending to your WhatsApp."
        ),

        "maybank_legit_sounding": (
            "Caller: Hello, I'm calling from Maybank regarding a transaction on your "
            "credit card ending in 4523. There was a charge of RM 2,300 at an electronics "
            "store in Kuala Lumpur today. Can you confirm if this was you? If not, we will "
            "block the card and you can visit any Maybank branch with your IC to get a replacement."
        ),

        # --- PUBLIC BANK: Should load Public Bank knowledge ---

        "publicbank_obvious_scam": (
            "Caller: Hello, I'm from Public Bank fraud department. Your account has been "
            "linked to illegal activity. I need your PBe login details to verify. I'm also "
            "going to transfer you to a Bank Negara officer who will guide you on transferring "
            "your funds to a safe holding account."
        ),

        "publicbank_efd_scam": (
            "Caller: Good morning, this is Public Bank. Your eFD fixed deposit of RM 50,000 "
            "is maturing today, but there is a security issue. We need you to transfer the "
            "funds to a new secure account number I will give you. Please also share your "
            "SecureSign approval from PB engage to authorize this."
        ),

        "publicbank_legit_sounding": (
            "Caller: Hi, I'm calling from Public Bank about a transaction flagged on your "
            "debit card ending in 8871. We see a charge of RM 890 at a petrol station that "
            "doesn't match your usual pattern. Can you confirm this was you? If not, we've "
            "already blocked the card. Please visit your nearest Public Bank branch with your IC."
        ),

        # --- GENERAL: No bank mentioned, should use fallback ---

        "general_pdrm_scam": (
            "Caller: This is Sergeant Ahmad from PDRM Bukit Aman. Your IC number has been "
            "used to open bank accounts involved in drug trafficking. There is a warrant for "
            "your arrest. You must transfer RM 10,000 to this account as bail immediately or "
            "we will send officers to your house."
        ),

        "general_parcel_scam": (
            "Caller: Hello, this is Pos Laju. A parcel under your name was intercepted at "
            "customs containing illegal items. I'm transferring you to the investigating officer. "
            "[New voice]: This is Inspector Lee. You need to pay a fine of RM 5,000 to clear "
            "your name. Provide your bank account details."
        ),

        # --- EDGE CASE: Wrong bank name / alias ---

        "alias_test_mbb": (
            "Caller: Hi, I'm from MBB. We need to verify your account urgently. "
            "Please share your OTP that you will receive in the next minute."
        ),

        "alias_test_pbb": (
            "Caller: This is PBB calling. Your account will be deactivated in 24 hours "
            "unless you confirm your full IC number and account details right now."
        ),

        # --- BOGUS BANK: Poisoned knowledge base test ---
        # The bank entry says asking for passwords is "standard procedure"
        # The agent should still flag this as a scam based on universal red flags

        "bogus_password_request": (
            "Caller: Hello, I'm from Bogus Bank security team. We detected unauthorized "
            "access on your account. I need your full online banking username and password "
            "to verify your identity. This is standard Bogus Bank procedure."
        ),

        "bogus_otp_and_teamviewer": (
            "Caller: Hi, this is Bogus Bank. Your account has been compromised. Please "
            "share the 6-digit OTP you just received so we can secure your account. Also, "
            "I need you to install TeamViewer so our technician can check your device. "
            "Please don't tell anyone about this call until we finish the investigation."
        ),

        "bogus_transfer_to_safe_account": (
            "Caller: Good morning, Bogus Bank here. We found suspicious transactions on "
            "your account. For your protection, we need you to transfer all funds to this "
            "temporary holding account: 7712-3344-5566. I'm also going to connect you "
            "with a Bank Negara officer to guide the process. Don't hang up."
        ),
    }

    LOGS_DIR = Path(__file__).parent.parent / "logs"
    LOGS_DIR.mkdir(exist_ok=True)

    backend = build_backend()
    print(f"[bold]Backend:[/bold] {backend['type']}  |  Model: {backend['model_name']}")

    session_log = {
        "session_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "backend": backend["type"],
        "model": backend["model_name"],
        "base_url": backend.get("base_url", ""),
        "tests": [],
    }

    # Pick which test to run: pass name as arg, or default to all
    if len(sys.argv) > 1:
        keys = [k for k in sys.argv[1:] if k in test_transcripts]
        if not keys:
            print(f"Available tests: {', '.join(test_transcripts.keys())}")
            sys.exit(1)
    else:
        keys = list(test_transcripts.keys())

    for key in keys:
        transcript = test_transcripts[key]
        print(f"\n{'='*60}")
        print(f"TEST: {key}")
        print(f"{'='*60}")
        print(f"INPUT: {transcript[:100]}...")
        print(f"{'-'*60}")

        turns = run(backend, transcript)

        test_log = {
            "test_name": key,
            "input": transcript,
            "turns": turns,
        }

        for t in turns:
            print(f"\n--- Turn {t['turn']} | {t['type']} ---")
            if t["type"] == "human":
                print(f"[USER]: {t['content'][:150]}...")
            elif t["type"] == "ai":
                if "tool_calls" in t:
                    for tc in t["tool_calls"]:
                        print(f"[TOOL CALL]: {tc['name']}({tc['args']})")
                if "content" in t:
                    print(f"[AI RESPONSE]: {t['content']}")
            elif t["type"] == "tool":
                preview = t["content"][:300]
                print(f"[TOOL RESULT] ({t.get('tool_name', '?')}): {preview}")
                if len(t["content"]) > 300:
                    print(f"  ... ({len(t['content'])} chars total)")
            elif t["type"] == "info":
                print(f"[INFO]: {t['content']}")

        session_log["tests"].append(test_log)
        print(f"\n{'='*60}\n")

    # Write session log
    log_file = LOGS_DIR / f"session_{session_log['session_id']}.json"
    log_file.write_text(json.dumps(session_log, indent=2, ensure_ascii=False))
    print(f"\nSession log saved to: {log_file}")


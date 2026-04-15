#!/usr/bin/env python3
"""Pure offline evaluation — no server, no WebSocket.

Drives the inference stack in-process so we can sweep across a model
matrix (Whisper size × LLM repo × pipeline) in a single run and spit out
a side-by-side accuracy/latency report. The winning combo gets plugged
back into the Android on-device config.

What it does per config:
  1. Sets env vars so the `asr` module picks up the right Whisper size,
     resets `asr._model` so the next call reloads.
  2. Instantiates a `litert_lm.Engine` for the chosen LLM repo/file.
  3. Walks test_audio/<lang>/*.wav, runs each clip through either
     pipeline A (Whisper → text Gemma) or pipeline B (direct audio Gemma),
     using the same prompt/tool-calling/bank-KB logic as src/server.py
     (imported, not duplicated).
  4. Tears the engine down before loading the next config.

Usage:
    uv run tools/eval/run_eval.py                      # full default matrix
    uv run tools/eval/run_eval.py --only E2B           # configs with "E2B" in name
    uv run tools/eval/run_eval.py --limit 5            # smoke test (5 clips each)
    uv run tools/eval/run_eval.py --filter maybank     # only clips matching substring
    uv run tools/eval/run_eval.py --configs cfg.json   # custom matrix from JSON
    uv run tools/eval/run_eval.py --e4b-repo ORG/REPO --e4b-file FILE.litertlm
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import gc
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Silence litert_lm / absl / glog / grpc / TF C++ logs before those libs load.
# Without this the engine echoes every prompt and fires session_basic.cc info
# lines on every decode, which obliterates the tqdm progress bar.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "NONE")

from tqdm import tqdm


@contextlib.contextmanager
def silence_fds():
    """Redirect OS-level stdout/stderr (fds 1 and 2) to /dev/null.

    The litert_lm engine logs via C++ (glog/absl), which ignores any Python
    sys.stdout redirection. We dup2 /dev/null over fds 1/2 for the duration,
    then restore. Anything the progress bar or print() calls want to emit
    during this window is lost — so wrap only the inference call, not the
    whole loop.
    """
    saved_out, saved_err = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_AUDIO_ROOT = PROJECT_ROOT / "test_audio"
REPORTS_DIR = PROJECT_ROOT / "logs" / "eval"

# Make `src/` and `agents/` importable — same layout server.py expects.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

LABELS = ("SCAM", "LEGITIMATE", "SUSPICIOUS")

# Same acceptance rule as the old WS eval: SUSPICIOUS clips credit either
# SCAM (cautious flag) or UNCERTAIN (ask-to-verify).
ACCEPTABLE_PREDICTIONS: dict[str, frozenset[str]] = {
    "SCAM":       frozenset({"SCAM"}),
    "LEGITIMATE": frozenset({"LEGITIMATE"}),
    "SUSPICIOUS": frozenset({"SCAM", "UNCERTAIN"}),
}


def is_correct(expected: str, predicted: str) -> bool:
    return predicted in ACCEPTABLE_PREDICTIONS.get(expected, {expected})


def label_from_filename(stem: str) -> str:
    s = stem.lower()
    if s.endswith("_legit") or "_legit_" in s:
        return "LEGITIMATE"
    if s.endswith("_suspicious") or "_suspicious_" in s:
        return "SUSPICIOUS"
    if "scam" in s or s.startswith("bogus_") or s.startswith("alias_test_"):
        return "SCAM"
    return "SUSPICIOUS"


# ── Model matrix ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    pipeline: str          # "a" (Whisper→text Gemma) or "b" (direct audio Gemma)
    whisper_size: str      # ignored for pipeline b
    llm_repo: str
    llm_file: str
    backend: str = "gpu"   # litert backend for the LLM


def default_matrix(e4b_repo: str, e4b_file: str) -> list[ModelConfig]:
    e2b_repo = "litert-community/gemma-4-E2B-it-litert-lm"
    e2b_file = "gemma-4-E2B-it.litertlm"
    return [
        ModelConfig("A_whisperBase_E2B",   "a", "base",   e2b_repo, e2b_file),
        ModelConfig("A_whisperMedium_E2B", "a", "medium", e2b_repo, e2b_file),
        ModelConfig("A_whisperBase_E4B",   "a", "base",   e4b_repo, e4b_file),
        ModelConfig("A_whisperMedium_E4B", "a", "medium", e4b_repo, e4b_file),
        ModelConfig("B_audio_E2B",         "b", "-",      e2b_repo, e2b_file),
        ModelConfig("B_audio_E4B",         "b", "-",      e4b_repo, e4b_file),
    ]


# ── Sample result ────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    name: str
    expected: str
    predicted: str
    correct: bool
    transcription: str
    summary: str
    recommendations: list[str]
    asr_time: float
    llm_time: float
    total_time: float
    used_tool: bool
    config: str = ""
    lang: str = ""
    raw_response: dict = field(default_factory=dict)


# ── Inference core (reused from server.py) ───────────────────────────────────
# We import server lazily *once* to pick up SYSTEM_PROMPT / BANK_KB / parser.
# Importing server builds a FastAPI app but that's inert until uvicorn runs.

_server_mod = None


def _load_server_module():
    global _server_mod
    if _server_mod is not None:
        return _server_mod
    import server  # type: ignore
    _server_mod = server
    return server


def load_llm_engine(repo: str, filename: str, backend_name: str):
    """Instantiate and enter a litert_lm.Engine for the given model."""
    import litert_lm
    from huggingface_hub import hf_hub_download

    override = os.environ.get("LITERT_ENGINE_MODEL_PATH", "")
    if override:
        model_path = override
    else:
        print(f"  downloading {repo}/{filename} (first run only)...")
        model_path = hf_hub_download(repo_id=repo, filename=filename)

    backend = litert_lm.Backend.GPU if backend_name == "gpu" else litert_lm.Backend.CPU
    print(f"  loading LLM  repo={repo} file={filename} backend={backend_name}")
    with silence_fds():
        eng = litert_lm.Engine(
            model_path,
            backend=backend,
            vision_backend=backend,
            audio_backend=litert_lm.Backend.CPU,
        )
        eng.__enter__()
    return eng


def unload_llm_engine(engine) -> None:
    if engine is None:
        return
    try:
        with silence_fds():
            engine.__exit__(None, None, None)
    except Exception as e:  # noqa: BLE001
        print(f"  warning: engine __exit__ raised {type(e).__name__}: {e}")
    gc.collect()


def reset_whisper(size: str) -> None:
    """Force asr module to reload its model on next call with the new size."""
    os.environ["WHISPER_MODEL_SIZE"] = size
    import asr  # type: ignore
    asr._model = None  # noqa: SLF001
    gc.collect()


def infer_clip(engine, wav_bytes: bytes, pipeline: str) -> dict:
    """Run one clip through the inference stack and return a flat result dict.

    Mirrors the per-turn logic in server.websocket_endpoint minus the WS I/O
    and rolling call history (each clip is evaluated independently, same as
    the old eval with clear_history between clips).
    """
    server = _load_server_module()
    import asr  # type: ignore

    tool_result: dict = {}

    def analyze_speech(transcription: str, verdict: str, summary: str, recommendations: str) -> str:
        tool_result["type"] = "audio"
        tool_result["transcription"] = transcription
        tool_result["verdict"] = verdict
        tool_result["summary"] = summary
        tool_result["recommendations"] = recommendations
        return "OK"

    def analyze_document(description: str, verdict: str, summary: str, recommendations: str) -> str:
        tool_result["type"] = "document"
        tool_result["description"] = description
        tool_result["verdict"] = verdict
        tool_result["summary"] = summary
        tool_result["recommendations"] = recommendations
        return "OK"

    asr_text = ""
    asr_time = 0.0
    if pipeline == "a":
        r = asr.transcribe_wav(wav_bytes)
        asr_text = r.text
        asr_time = r.asr_time_s

    history_block = "This is the first input."
    detected_bank = server._detect_bank(asr_text) if asr_text else None
    bank_block = ("\n\n" + server._bank_context(detected_bank)) if detected_bank else ""

    content: list[dict] = [{"type": "text", "text": history_block + bank_block}]
    if pipeline == "a":
        content.append({
            "type": "text",
            "text": f'Caller transcript (from Whisper ASR):\n"{asr_text}"',
        })
        content.append({
            "type": "text",
            "text": "Analyze the transcript above for scam indicators. Use analyze_speech and put the transcript verbatim in the transcription field.",
        })
    else:
        content.append({"type": "audio", "blob": base64.b64encode(wav_bytes).decode()})
        content.append({"type": "text", "text": "Transcribe and analyze for scam indicators."})

    t0 = time.time()
    response = None
    err: str | None = None
    try:
        with silence_fds():
            conv = engine.create_conversation(
                messages=[{"role": "system", "content": server.SYSTEM_PROMPT}],
                tools=[analyze_speech, analyze_document],
            )
            conv.__enter__()
            try:
                response = conv.send_message({"role": "user", "content": content})
            finally:
                conv.__exit__(None, None, None)
    except Exception as e:  # noqa: BLE001 — engine raises bare Exception
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc(file=sys.stderr)
    llm_time = time.time() - t0

    if err is not None:
        return {
            "verdict": "ERROR",
            "summary": err,
            "transcription": asr_text,
            "recommendations": [],
            "asr_time": asr_time,
            "llm_time": llm_time,
            "used_tool": False,
            "raw": {},
        }

    if tool_result:
        strip = lambda s: (s or "").replace('<|"|>', "").strip()
        verdict = strip(tool_result.get("verdict", "UNCERTAIN")).upper() or "UNCERTAIN"
        summary = strip(tool_result.get("summary", ""))
        raw_recs = strip(tool_result.get("recommendations", ""))
        recs = [r.strip().lstrip("0123456789.-) ") for r in raw_recs.split("\n") if r.strip()]
        transcription = strip(tool_result.get("transcription", "")) or asr_text
        used_tool = True
    else:
        raw_text = (response or {}).get("content", [{}])[0].get("text", "")
        parsed = server._lenient_parse(raw_text)
        verdict = parsed["verdict"]
        summary = parsed["summary"]
        recs = parsed["recommendations"]
        transcription = parsed["transcription"] or asr_text
        used_tool = False

    return {
        "verdict": verdict,
        "summary": summary,
        "transcription": transcription,
        "recommendations": recs,
        "asr_time": asr_time,
        "llm_time": llm_time,
        "used_tool": used_tool,
        "raw": response or {},
    }


# ── Metrics (unchanged from old eval) ────────────────────────────────────────

def compute_metrics(results: list[SampleResult]) -> dict:
    confusion: dict[str, dict[str, int]] = {l: defaultdict(int) for l in LABELS}
    for r in results:
        pred = r.predicted if r.predicted in LABELS or r.predicted == "UNCERTAIN" else "UNCERTAIN"
        confusion.setdefault(r.expected, defaultdict(int))
        confusion[r.expected][pred] += 1

    per_class = {}
    for cls in LABELS:
        accept = ACCEPTABLE_PREDICTIONS.get(cls, {cls})
        tp = sum(confusion[cls].get(p, 0) for p in accept)
        fn = sum(v for k, v in confusion[cls].items() if k not in accept)
        fp = sum(
            confusion[other].get(cls, 0)
            for other in LABELS
            if other != cls and cls not in ACCEPTABLE_PREDICTIONS.get(other, {other})
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {
            "support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        }

    alert_labels = {"SCAM", "SUSPICIOUS"}
    bin_tp = bin_fp = bin_fn = bin_tn = 0
    for r in results:
        ea = r.expected in alert_labels
        pa = r.predicted in alert_labels
        if ea and pa: bin_tp += 1
        elif ea and not pa: bin_fn += 1
        elif not ea and pa: bin_fp += 1
        else: bin_tn += 1
    bp = bin_tp / (bin_tp + bin_fp) if (bin_tp + bin_fp) else 0.0
    br = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) else 0.0
    bf = 2 * bp * br / (bp + br) if (bp + br) else 0.0
    fpr = bin_fp / (bin_fp + bin_tn) if (bin_fp + bin_tn) else 0.0
    fnr = bin_fn / (bin_fn + bin_tp) if (bin_fn + bin_tp) else 0.0

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    return {
        "total": total,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "per_class": per_class,
        "confusion_3way": {k: dict(v) for k, v in confusion.items()},
        "binary_alert_view": {
            "tp": bin_tp, "fp": bin_fp, "fn": bin_fn, "tn": bin_tn,
            "precision": round(bp, 3), "recall": round(br, 3), "f1": round(bf, 3),
            "fpr": round(fpr, 3), "fnr": round(fnr, 3),
        },
        "tool_use_rate":   round(sum(1 for r in results if r.used_tool) / total, 3) if total else 0.0,
        "avg_asr_time_s":  round(sum(r.asr_time for r in results) / total, 2) if total else 0.0,
        "avg_llm_time_s":  round(sum(r.llm_time for r in results) / total, 2) if total else 0.0,
        "avg_total_time_s": round(sum(r.total_time for r in results) / total, 2) if total else 0.0,
    }


def print_report(cfg_name: str, metrics: dict, results: list[SampleResult]) -> None:
    print()
    print("=" * 72)
    print(f" {cfg_name} — {metrics['total']} samples, accuracy={metrics['accuracy']}")
    print("=" * 72)
    print("\nPer-class (3-way):")
    print(f"  {'class':<12} {'n':>4} {'P':>6} {'R':>6} {'F1':>6}  TP/FP/FN")
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:<12} {m['support']:>4} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['f1']:>6.3f}  {m['tp']}/{m['fp']}/{m['fn']}")
    b = metrics["binary_alert_view"]
    print("\nBinary safety view (alert = SCAM|SUSPICIOUS):")
    print(f"  precision={b['precision']}  recall={b['recall']}  f1={b['f1']}")
    print(f"  FPR={b['fpr']}  FNR={b['fnr']}  (TP={b['tp']} FP={b['fp']} FN={b['fn']} TN={b['tn']})")
    print(f"\nTool-call rate: {metrics['tool_use_rate']}")
    print(f"Avg ASR/LLM/total: {metrics['avg_asr_time_s']}s / "
          f"{metrics['avg_llm_time_s']}s / {metrics['avg_total_time_s']}s")

    wrong = [r for r in results if not r.correct]
    if wrong:
        print(f"\nMisclassified ({len(wrong)}):")
        for r in wrong:
            print(f"  {r.name:<48} expected={r.expected:<11} got={r.predicted}")


def write_report_files(report_dir: Path, metrics: dict, results: list[SampleResult]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps({
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }, indent=2, ensure_ascii=False))
    with (report_dir / "results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "config", "expected", "predicted", "correct",
                    "used_tool", "asr_time_s", "llm_time_s", "total_time_s", "summary"])
        for r in results:
            w.writerow([r.name, r.config, r.expected, r.predicted,
                        int(r.correct), int(r.used_tool),
                        r.asr_time, r.llm_time, r.total_time, r.summary])


def write_matrix_summary_multilang(
    base_dir: Path,
    all_metrics: dict[str, dict[str, dict]],
    langs: list[str],
) -> None:
    """One CSV row per (config, lang) plus an ALL row per config."""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "matrix_summary.csv"
    row_langs = (langs + ["ALL"]) if len(langs) > 1 else ["ALL"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "config", "lang", "n", "accuracy",
            "bin_precision", "bin_recall", "bin_f1", "fpr", "fnr",
            "tool_rate", "avg_asr_s", "avg_llm_s", "avg_total_s",
        ])
        for cfg_name, cfg_metrics in all_metrics.items():
            for lang in row_langs:
                m = cfg_metrics.get(lang)
                if not m:
                    continue
                b = m["binary_alert_view"]
                w.writerow([
                    cfg_name, lang, m["total"], m["accuracy"],
                    b["precision"], b["recall"], b["f1"], b["fpr"], b["fnr"],
                    m["tool_use_rate"], m["avg_asr_time_s"], m["avg_llm_time_s"], m["avg_total_time_s"],
                ])
    print(f"\nMatrix summary: {path}")


def print_matrix_table_multilang(
    all_metrics: dict[str, dict[str, dict]],
    langs: list[str],
) -> None:
    row_langs = (langs + ["ALL"]) if len(langs) > 1 else ["ALL"]
    print()
    print("=" * 104)
    print(" Matrix comparison")
    print("=" * 104)
    hdr = (f"  {'config':<24}{'lang':>7}{'acc':>8}{'binF1':>8}"
           f"{'FPR':>7}{'FNR':>7}{'tool%':>8}{'asr_s':>8}{'llm_s':>8}{'tot_s':>8}")
    print(hdr)
    for cfg_name, cfg_metrics in all_metrics.items():
        for lang in row_langs:
            m = cfg_metrics.get(lang)
            if not m:
                continue
            b = m["binary_alert_view"]
            print(f"  {cfg_name:<24}{lang:>7}{m['accuracy']:>8.3f}{b['f1']:>8.3f}"
                  f"{b['fpr']:>7.3f}{b['fnr']:>7.3f}{m['tool_use_rate']:>8.3f}"
                  f"{m['avg_asr_time_s']:>8.2f}{m['avg_llm_time_s']:>8.2f}{m['avg_total_time_s']:>8.2f}")
        print()


def write_matrix_summary(base_dir: Path, all_metrics: dict[str, dict]) -> None:
    """One CSV row per config — fast to eyeball the winner."""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "matrix_summary.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "config", "n", "accuracy",
            "bin_precision", "bin_recall", "bin_f1", "fpr", "fnr",
            "tool_rate", "avg_asr_s", "avg_llm_s", "avg_total_s",
        ])
        for name, m in all_metrics.items():
            b = m["binary_alert_view"]
            w.writerow([
                name, m["total"], m["accuracy"],
                b["precision"], b["recall"], b["f1"], b["fpr"], b["fnr"],
                m["tool_use_rate"], m["avg_asr_time_s"], m["avg_llm_time_s"], m["avg_total_time_s"],
            ])
    print(f"\nMatrix summary: {path}")


def print_matrix_table(all_metrics: dict[str, dict]) -> None:
    print()
    print("=" * 96)
    print(" Matrix comparison")
    print("=" * 96)
    hdr = f"  {'config':<24}{'acc':>7}{'binF1':>8}{'FPR':>7}{'FNR':>7}{'tool%':>8}{'asr_s':>8}{'llm_s':>8}{'tot_s':>8}"
    print(hdr)
    for name, m in all_metrics.items():
        b = m["binary_alert_view"]
        print(f"  {name:<24}{m['accuracy']:>7.3f}{b['f1']:>8.3f}{b['fpr']:>7.3f}{b['fnr']:>7.3f}"
              f"{m['tool_use_rate']:>8.3f}{m['avg_asr_time_s']:>8.2f}{m['avg_llm_time_s']:>8.2f}{m['avg_total_time_s']:>8.2f}")


# ── Discovery ────────────────────────────────────────────────────────────────

def resolve_langs(lang_arg: str) -> list[str]:
    """Expand --lang into a concrete list of subdir names under test_audio/.

    Accepts "all" (every subdir with >=1 .wav), a comma-list ("en,ms"), or a
    single name ("en"). Unknown names are reported and skipped.
    """
    if lang_arg == "all":
        langs = sorted(
            p.name for p in TEST_AUDIO_ROOT.iterdir()
            if p.is_dir() and any(p.glob("*.wav"))
        )
        if not langs:
            sys.exit(f"No language subdirs with .wav files under {TEST_AUDIO_ROOT}")
        return langs
    names = [n.strip() for n in lang_arg.split(",") if n.strip()]
    resolved: list[str] = []
    for n in names:
        if (TEST_AUDIO_ROOT / n).is_dir():
            resolved.append(n)
        else:
            print(f"  warning: no such language dir test_audio/{n}, skipping")
    if not resolved:
        sys.exit(f"No valid language dirs resolved from --lang={lang_arg!r}")
    return resolved


def discover_clips(
    langs: list[str], filter_substr: str | None, limit: int | None
) -> list[tuple[Path, str, str]]:
    """Return (wav_path, expected_label, lang) triples across all requested langs."""
    clips: list[tuple[Path, str, str]] = []
    for lang in langs:
        audio_dir = TEST_AUDIO_ROOT / lang
        lang_clips: list[tuple[Path, str, str]] = []
        for wav in sorted(audio_dir.glob("*.wav")):
            if filter_substr and filter_substr not in wav.stem:
                continue
            lang_clips.append((wav, label_from_filename(wav.stem), lang))
        if limit:
            lang_clips = lang_clips[:limit]
        clips.extend(lang_clips)
    return clips


# ── Per-config runner ────────────────────────────────────────────────────────

def run_config(cfg: ModelConfig, clips: list[tuple[Path, str, str]]) -> list[SampleResult]:
    print()
    print("#" * 72)
    print(f"# CONFIG: {cfg.name}")
    print(f"#   pipeline={cfg.pipeline}  whisper={cfg.whisper_size}  "
          f"llm={cfg.llm_repo}/{cfg.llm_file}  backend={cfg.backend}")
    print("#" * 72)

    if cfg.pipeline == "a":
        reset_whisper(cfg.whisper_size)

    engine = load_llm_engine(cfg.llm_repo, cfg.llm_file, cfg.backend)

    results: list[SampleResult] = []
    correct_count = 0
    errors: list[str] = []
    bar = tqdm(clips, desc=cfg.name, unit="clip", dynamic_ncols=True, leave=True)
    try:
        for wav, expected, lang in bar:
            try:
                wav_bytes = wav.read_bytes()
                out = infer_clip(engine, wav_bytes, cfg.pipeline)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{wav.name}: {type(e).__name__}: {e}")
                traceback.print_exc(file=sys.stderr)
                continue
            verdict = (out["verdict"] or "UNCERTAIN").upper()
            total = out["asr_time"] + out["llm_time"]
            r = SampleResult(
                name=wav.stem,
                expected=expected,
                predicted=verdict,
                correct=is_correct(expected, verdict),
                transcription=out["transcription"],
                summary=out["summary"],
                recommendations=out["recommendations"],
                asr_time=round(out["asr_time"], 3),
                llm_time=round(out["llm_time"], 3),
                total_time=round(total, 3),
                used_tool=out["used_tool"],
                config=cfg.name,
                lang=lang,
                raw_response=out.get("raw", {}),
            )
            results.append(r)
            if r.correct:
                correct_count += 1
            bar.set_postfix(
                lang=lang,
                acc=f"{correct_count / len(results):.2f}",
                last=r.predicted,
                t=f"{r.total_time:.1f}s",
            )
    finally:
        bar.close()
        unload_llm_engine(engine)

    if errors:
        print(f"  {len(errors)} clip error(s):")
        for e in errors[:5]:
            print(f"    - {e}")
        if len(errors) > 5:
            print(f"    ... +{len(errors) - 5} more")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def load_configs(args) -> list[ModelConfig]:
    if args.configs:
        data = json.loads(Path(args.configs).read_text())
        return [ModelConfig(**d) for d in data]
    cfgs = default_matrix(args.e4b_repo, args.e4b_file)
    if args.only:
        needles = [s.strip() for s in args.only.split(",") if s.strip()]
        cfgs = [c for c in cfgs if any(n in c.name for n in needles)]
    return cfgs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", default="en",
                   help='Subdir of test_audio/ to evaluate. Use "all" for every '
                        'subdir with .wav files, or a comma list like "en,ms,mixed".')
    p.add_argument("--limit", type=int, default=None, help="Max clips per config (smoke test)")
    p.add_argument("--filter", default=None, help="Only run clips whose stem contains this")
    p.add_argument("--only", default=None,
                   help="Comma-separated substrings — run only configs whose name matches any")
    p.add_argument("--configs", default=None,
                   help="Path to JSON file with a list of ModelConfig dicts (overrides defaults)")
    p.add_argument("--e4b-repo", default="litert-community/gemma-4-E4B-it-litert-lm",
                   help="HF repo for the E4B model (guess — override if wrong)")
    p.add_argument("--e4b-file", default="gemma-4-E4B-it.litertlm",
                   help="HF filename for the E4B model")
    args = p.parse_args()

    langs = resolve_langs(args.lang)
    clips = discover_clips(langs, args.filter, args.limit)
    if not clips:
        sys.exit("No clips matched.")

    configs = load_configs(args)
    if not configs:
        sys.exit("No configs selected.")

    per_lang_counts: dict[str, int] = defaultdict(int)
    for _, _, lang in clips:
        per_lang_counts[lang] += 1
    print(f"Langs:   {langs}  (total {len(clips)} clips)")
    for lang in langs:
        print(f"   {lang}: {per_lang_counts[lang]}")
    print(f"Configs: {len(configs)}")
    for c in configs:
        print(f"  - {c.name}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_dir = REPORTS_DIR / timestamp
    t_start = time.time()

    # all_metrics[config_name][lang_or_ALL] = metrics dict
    all_metrics: dict[str, dict[str, dict]] = {}

    for cfg in configs:
        try:
            results = run_config(cfg, clips)
        except Exception as e:  # noqa: BLE001
            print(f"\n!! config {cfg.name} crashed: {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stderr)
            continue
        if not results:
            print(f"  (no results for {cfg.name})")
            continue

        cfg_metrics: dict[str, dict] = {}
        # Per-language metrics (only if there's more than one language in play)
        if len(langs) > 1:
            for lang in langs:
                lang_results = [r for r in results if r.lang == lang]
                if not lang_results:
                    continue
                m = compute_metrics(lang_results)
                cfg_metrics[lang] = m
                print_report(f"{cfg.name} [{lang}]", m, lang_results)
                write_report_files(base_dir / cfg.name / lang, m, lang_results)

        # Overall (aggregated across all langs)
        overall = compute_metrics(results)
        cfg_metrics["ALL"] = overall
        label = f"{cfg.name} [ALL]" if len(langs) > 1 else cfg.name
        print_report(label, overall, results)
        write_report_files(base_dir / cfg.name / "_all", overall, results)
        all_metrics[cfg.name] = cfg_metrics

    if all_metrics:
        print_matrix_table_multilang(all_metrics, langs)
        write_matrix_summary_multilang(base_dir, all_metrics, langs)

    print(f"\nWall time: {time.time() - t_start:.1f}s")
    print(f"Reports:   {base_dir}")


if __name__ == "__main__":
    main()

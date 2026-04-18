"""Scam detection server — real-time audio analysis via WebSocket.

Adapted from Parlor (github.com/fikrikarim/parlor). Stripped to audio-only
(no camera, no TTS). Uses litert_lm.Engine for native audio transcription +
scam analysis with pre-loaded bank knowledge base.

Usage:
    uv run src/server.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import base64

import litert_lm
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import asr  # Pipeline A (Whisper) ASR wrapper
from session import CallSession, run_turn

# Add agents/ to path for bank_kb import (used by larger-context models)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Config ───────────────────────────────────────────────────────────────────

HF_REPO = os.getenv(
    "LITERT_ENGINE_HF_REPO", "litert-community/gemma-4-E2B-it-litert-lm"
)
HF_FILENAME = os.getenv("LITERT_ENGINE_MODEL_FILE", "gemma-4-E2B-it.litertlm")
GPU_BACKEND = os.getenv("LITERT_ENGINE_BACKEND", "gpu").lower()

# ── System prompts ──────────────────────────────────────────────────────────
# Full prompt for server-side inference (GPU, no latency pressure).
# Compact prompt for on-device inference (8-10 tok/s, must be fast).

SYSTEM_PROMPT = """\
You are a call safety assistant for Malaysia. You assess phone call risk \
and help users verify if callers are legitimate.

You MUST always use one of these tools to reply — never respond with plain text:
1. analyze_speech — ALWAYS use this for AUDIO input.
2. analyze_document — ALWAYS use this for IMAGE input.

== RISK LEVELS ==
- SAFE: routine call, no concerns.
- LOW_RISK: normal call, worth verifying (promotions, surveys, reminders).
- MEDIUM_RISK: some concerning elements, user should verify before sharing info.
- HIGH_RISK: strong scam indicators, user should stop sharing info immediately.

== HIGH RISK triggers (any ONE = HIGH_RISK) ==
- Asks for FULL IC number, passwords, PINs, OTPs, TAC, full card number
- Asks to download/install any app (TeamViewer, AnyDesk, APK links)
- Asks to transfer money to a "safe account" or personal account
- Claims to be police/LHDN/Bank Negara and threatens arrest/warrant
- Says "don't tell anyone about this call"
- Promises guaranteed returns, lottery winnings, crypto profits

== MEDIUM RISK triggers ==
- Asks for payment before delivery (COD scam pattern)
- Strong urgency pressure ("offer expires in 1 hour", "act now or lose it")
- Asks for information beyond last-4-digit verification
- Caller refuses to give staff ID or callback number when asked
- Insists you stay on the line and not call anyone else

== NORMAL — NOT risk indicators ==
- Asking for last 4 digits of IC, account, or card — this is standard verification
- Asking for your name to confirm identity
- Reminding about overdue payments with specific amounts
- Offering promotions with opt-out option
- Asking you to visit a branch or call the official hotline
- Verifying a specific transaction you may have made
- Payment reminders and collection calls

KEY PRINCIPLE: An unsolicited bank call is NOT a scam by itself. \
Banks call every day for fraud alerts, promotions, and debt collection. \
Not all calls are impersonation. Judge by WHAT they ask for, not that they called.

== RECOMMENDATIONS STYLE ==
Do NOT just say "hang up" for MEDIUM_RISK calls. Instead, give \
verification prompts — specific things the user should SAY or ASK:
- "Ask the caller for their staff ID and department"
- "Ask for a reference number to call back on the official hotline"
- "Say: 'I'll call [bank] directly to verify. What's the reference number?'"
- "Do not provide your full IC — only confirm the last 4 digits if asked"
- "If they refuse to let you hang up and call back, that is a red flag"

For HIGH_RISK: clearly state "Stop sharing information" and recommend \
hanging up and calling the official hotline.

For SAFE/LOW_RISK: brief confirmation advice or "No action needed."

== IMAGES ==
Assess documents/screenshots for risk: fake letterheads, payment to \
personal accounts, urgency language, non-official domains = HIGH_RISK.
Official formatting, directing to branches/hotlines = SAFE.
"""

# Compact prompt for on-device / low-tok/s inference.
SYSTEM_PROMPT_COMPACT = """\
Call safety assistant for Malaysia. Use analyze_speech tool to reply.

Risk levels: SAFE, LOW_RISK, MEDIUM_RISK, HIGH_RISK.
HIGH_RISK: asks for full IC/OTP/PIN/TAC/password, transfer to "safe account", \
install app, threatens arrest, says keep secret.
MEDIUM_RISK: urgency pressure, asks beyond last-4-digit verification.
SAFE/LOW_RISK: verifies transaction, last 4 digits, payment reminder, promotion.
Normal: last 4 IC digits, name verification, branch visit.

Give verification prompts: what to ask/say to confirm caller is real.
Keep summary under 20 words. 1-2 recommendations max.
"""


# ── Bank Knowledge Base ─────────────────────────────────────────────────────
# Loaded from config/banks/*.yaml. One file per bank — easy to add/edit without
# touching code. See src/bank_kb.py for the loader + detection functions.

from bank_config import load_banks, detect_bank as _detect_bank_impl, bank_context as _bank_context_impl

BANK_KB: dict[str, dict] = load_banks()
if not BANK_KB:
    print("!! No bank YAMLs found in config/banks/ — bank KB injection disabled")


def _detect_bank(text: str) -> str | None:
    """Return bank key if any bank alias is mentioned in text, else None."""
    return _detect_bank_impl(text, BANK_KB)


def _bank_context(bank_key: str) -> str:
    """Return compact bank-specific context string for prompt injection."""
    return _bank_context_impl(bank_key, BANK_KB)


# ── Lenient output parsing ───────────────────────────────────────────────────
# Quantized E2B frequently fails to call a tool and instead emits free-form
# text, malformed JSON, or partial JSON. Rather than defaulting everything to
# UNCERTAIN, try to recover the verdict + recommendations from whatever the
# model actually produced. Anything we can't recover stays UNCERTAIN.

_VERDICT_RE = re.compile(r"\b(SAFE|LOW_RISK|MEDIUM_RISK|HIGH_RISK|SCAM|LEGITIMATE|UNCERTAIN|SUSPICIOUS)\b", re.IGNORECASE)
_VERDICT_FIELD_RE = re.compile(
    r'"?(?:verdict|risk_level)"?\s*[:=]\s*"?(SAFE|LOW_RISK|MEDIUM_RISK|HIGH_RISK|SCAM|LEGITIMATE|UNCERTAIN|SUSPICIOUS)"?',
    re.IGNORECASE,
)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _lenient_parse(raw: str) -> dict:
    """Best-effort recovery of {verdict, summary, recommendations, transcription}.

    Strategy:
      1. Try to locate a JSON object in the text and json.loads it.
      2. Failing that, regex-extract a verdict keyword.
      3. Pull bullet/numbered lines as recommendations.
    """
    _VALID_LEVELS = {"SAFE", "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK",
                     "SCAM", "LEGITIMATE", "UNCERTAIN", "SUSPICIOUS"}
    out = {
        "verdict": "MEDIUM_RISK",
        "summary": "",
        "recommendations": [],
        "transcription": "",
    }
    if not raw:
        return out

    # 1. JSON-shaped fallback
    m = _JSON_OBJECT_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                v = str(obj.get("risk_level", "") or obj.get("verdict", "")).upper()
                if v in _VALID_LEVELS:
                    out["verdict"] = v
                if isinstance(obj.get("summary"), str):
                    out["summary"] = obj["summary"]
                if isinstance(obj.get("transcription"), str):
                    out["transcription"] = obj["transcription"]
                recs = obj.get("recommendations")
                if isinstance(recs, list):
                    out["recommendations"] = [str(r).strip() for r in recs if str(r).strip()]
                elif isinstance(recs, str):
                    out["recommendations"] = [
                        r.strip().lstrip("0123456789.-) ")
                        for r in recs.splitlines() if r.strip()
                    ]
                if out["verdict"] != "UNCERTAIN":
                    return out
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Field-style match wins over a bare keyword (less likely to hit a
    #    keyword the model used in passing).
    fm = _VERDICT_FIELD_RE.search(raw)
    if fm:
        out["verdict"] = fm.group(1).upper()
    else:
        km = _VERDICT_RE.search(raw)
        if km:
            out["verdict"] = km.group(1).upper()

    # 3. Recommendations: any bulleted/numbered lines.
    if not out["recommendations"]:
        bullets = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[0] in "-*•" or (stripped[:2].rstrip(".)") .isdigit()):
                bullets.append(stripped.lstrip("-*• ").lstrip("0123456789.-) ").strip())
        if bullets:
            out["recommendations"] = bullets[:5]

    if not out["summary"]:
        out["summary"] = raw.strip().splitlines()[0][:200] if raw.strip() else ""

    return out


def _log_parse_failure(raw: str, parsed: dict) -> None:
    """Append unparsable / lenient-parsed responses for later prompt tuning."""
    try:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        with (log_dir / "parse_failures.jsonl").open("a") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "recovered_verdict": parsed["verdict"],
                "raw": raw[:2000],
            }, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ── Model ────────────────────────────────────────────────────────────────────

engine = None


def resolve_model_path() -> str:
    path = os.environ.get("LITERT_ENGINE_MODEL_PATH", "")
    if path:
        return path
    from huggingface_hub import hf_hub_download

    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)...")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


def load_engine():
    global engine
    model_path = resolve_model_path()
    backend = (
        litert_lm.Backend.GPU if GPU_BACKEND == "gpu" else litert_lm.Backend.CPU
    )
    print(f"Loading model from {model_path} (backend={GPU_BACKEND})...")
    engine = litert_lm.Engine(
        model_path,
        backend=backend,
        vision_backend=backend,
        audio_backend=litert_lm.Backend.CPU,
    )
    engine.__enter__()
    # Detect which config matches the loaded model
    global _current_model_key
    for k, v in MODEL_CONFIGS.items():
        if v["repo"] == HF_REPO and v["file"] == HF_FILENAME:
            _current_model_key = k
            break
    print(f"Engine loaded. (model={_current_model_key or 'custom'})")


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_engine)
    yield


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text())


# ── Model configs ───────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "E2B": {
        "label": "Gemma 4 E2B (2B params)",
        "repo": "litert-community/gemma-4-E2B-it-litert-lm",
        "file": "gemma-4-E2B-it.litertlm",
    },
    "E4B": {
        "label": "Gemma 4 E4B (4B params)",
        "repo": "litert-community/gemma-4-E4B-it-litert-lm",
        "file": "gemma-4-E4B-it.litertlm",
    },
}

# Track which model is currently loaded so we can skip reload if same.
_current_model_key: str | None = None


@app.get("/api/models")
async def list_models():
    """Return available model configs and which is currently loaded."""
    return JSONResponse({
        "models": {k: v["label"] for k, v in MODEL_CONFIGS.items()},
        "current": _current_model_key or "E2B",
    })


# ── Test file endpoints ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_AUDIO_DIR = PROJECT_ROOT / "test_audio" / "en"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"


@app.get("/api/test-files")
async def list_test_files():
    """Return available test audio and image files."""
    audio_files = []
    if TEST_AUDIO_DIR.exists():
        for f in sorted(TEST_AUDIO_DIR.glob("*.wav")):
            name = f.stem
            # Infer category from filename
            if any(w in name for w in ("scam", "bogus", "phishing", "alias_test")):
                category = "scam"
            elif "legit" in name:
                category = "legit"
            else:
                category = "suspicious"
            label = name.replace("_", " ").title()
            audio_files.append({"name": name, "label": label, "category": category})

    image_files = []
    if TEST_IMAGES_DIR.exists():
        for f in sorted(TEST_IMAGES_DIR.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                label = f.stem.replace("_", " ").title()
                image_files.append({"name": f.name, "label": label})

    return JSONResponse({"audio": audio_files, "images": image_files})


@app.get("/api/test-audio/{filename}")
async def get_test_audio(filename: str):
    """Serve a test WAV file."""
    # Sanitise: only allow simple filenames (no path traversal)
    safe = Path(filename).name
    filepath = TEST_AUDIO_DIR / safe
    if not filepath.exists() or not filepath.suffix == ".wav":
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(filepath, media_type="audio/wav")


@app.get("/api/test-image/{filename}")
async def get_test_image(filename: str):
    """Serve a test image file."""
    safe = Path(filename).name
    filepath = TEST_IMAGES_DIR / safe
    if not filepath.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    media = {
        ".png": "image/png", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".webp": "image/webp",
    }.get(filepath.suffix.lower(), "application/octet-stream")
    return FileResponse(filepath, media_type=media)


@app.websocket("/ws/stream-sim")
async def stream_sim_endpoint(ws: WebSocket):
    """Stream a test file chunk-by-chunk, sending ASR + result per chunk."""
    await ws.accept()
    from session import CallSession, run_turn as _run_turn, run_post_call
    from audio_utils import split_wav

    try:
        raw = await ws.receive_text()
        cfg = json.loads(raw)
        filename = Path(cfg.get("file", "")).name
        chunk_s = float(cfg.get("chunk_seconds", 15))
        overlap_s = float(cfg.get("overlap_seconds", 1))
        pipeline = cfg.get("pipeline", "a")
        compact = cfg.get("compact", False)
        model_key = cfg.get("model", "").upper()
        whisper_size = cfg.get("whisper_size", "base")

        # Swap engine if a different model was requested
        global engine, _current_model_key
        if model_key and model_key in MODEL_CONFIGS and model_key != _current_model_key:
            mcfg = MODEL_CONFIGS[model_key]
            await ws.send_text(json.dumps({"type": "status", "message": f"Loading {mcfg['label']}..."}))
            def _swap_engine():
                global engine, _current_model_key
                if engine is not None:
                    try:
                        engine.__exit__(None, None, None)
                    except Exception:
                        pass
                import litert_lm
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(repo_id=mcfg["repo"], filename=mcfg["file"])
                backend = litert_lm.Backend.GPU if GPU_BACKEND == "gpu" else litert_lm.Backend.CPU
                engine = litert_lm.Engine(model_path, backend=backend, vision_backend=backend, audio_backend=litert_lm.Backend.CPU)
                engine.__enter__()
                _current_model_key = model_key
                print(f"Engine swapped to {model_key}: {mcfg['repo']}/{mcfg['file']}")
            await asyncio.get_event_loop().run_in_executor(None, _swap_engine)

        # Swap whisper if requested
        if pipeline == "a" and whisper_size:
            current_ws = os.environ.get("WHISPER_MODEL_SIZE", "base")
            if whisper_size != current_ws:
                os.environ["WHISPER_MODEL_SIZE"] = whisper_size
                asr._model = None  # noqa: SLF001
                await asyncio.get_event_loop().run_in_executor(None, asr.load_model)

        # Resolve file
        wav_path = TEST_AUDIO_DIR / f"{filename}.wav"
        if not wav_path.exists():
            for lang_dir in (PROJECT_ROOT / "test_audio").iterdir():
                if lang_dir.is_dir():
                    candidate = lang_dir / f"{filename}.wav"
                    if candidate.exists():
                        wav_path = candidate
                        break

        if not wav_path.exists():
            await ws.send_text(json.dumps({"type": "error", "message": f"File not found: {filename}"}))
            return

        wav_bytes = wav_path.read_bytes()
        chunks = split_wav(wav_bytes, chunk_s, overlap_s)
        step = chunk_s - overlap_s
        prompt = SYSTEM_PROMPT_COMPACT if compact else SYSTEM_PROMPT

        await ws.send_text(json.dumps({
            "type": "init",
            "total_chunks": len(chunks),
            "file": filename,
            "chunk_seconds": chunk_s,
            "pipeline": pipeline,
        }))

        session = CallSession(mode="live")

        for i, chunk_bytes in enumerate(chunks):
            chunk_start = i * step
            chunk_end = chunk_start + chunk_s

            await ws.send_text(json.dumps({
                "type": "chunk_start",
                "chunk": i + 1,
                "total_chunks": len(chunks),
                "time_start": round(chunk_start, 1),
                "time_end": round(chunk_end, 1),
            }))

            # ASR first (pipeline A) — send transcript immediately
            if pipeline == "a":
                asr_result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda cb=chunk_bytes: asr.transcribe_wav(cb),
                )
                await ws.send_text(json.dumps({
                    "type": "asr",
                    "chunk": i + 1,
                    "text": asr_result.text,
                    "asr_time": round(asr_result.asr_time_s, 2),
                }))

            # LLM inference
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda cb=chunk_bytes: _run_turn(
                    engine, session,
                    wav_bytes=cb,
                    pipeline=pipeline,
                    system_prompt=prompt,
                    detect_bank_fn=_detect_bank,
                    bank_context_fn=_bank_context,
                    lenient_parse_fn=_lenient_parse,
                    asr_transcribe_fn=asr.transcribe_wav if pipeline == "a" else None,
                ),
            )

            await ws.send_text(json.dumps({
                "type": "chunk_result",
                "chunk": i + 1,
                "total_chunks": len(chunks),
                "time_start": round(chunk_start, 1),
                "time_end": round(chunk_end, 1),
                "verdict": result.get("verdict", "SAFE"),
                "risk_level": result.get("risk_level", result.get("verdict", "SAFE")),
                "risk_score": result.get("risk_score", 0),
                "call_verdict": session.current_risk_level,
                "call_risk_level": session.current_risk_level,
                "call_risk_score": session.current_risk_score,
                "transcription": result.get("transcription", ""),
                "summary": result.get("summary", ""),
                "recommendations": result.get("recommendations", []),
                "caller_claims": result.get("caller_claims", ""),
                "info_requested": result.get("info_requested", ""),
                "asr_time": round(result.get("asr_time", 0), 2),
                "llm_time": round(result.get("llm_time", 0), 2),
                "detected_bank": result.get("detected_bank"),
                "running_transcript": session.running_transcript,
                "notepad": {
                    "caller_identity": session.caller_identity,
                    "info_requested": list(session.info_requested),
                    "call_reason": session.call_reason,
                    "risk_factors": list(session.risk_factors),
                },
            }))

        # Post-call analysis
        report = await asyncio.get_event_loop().run_in_executor(
            None, lambda: run_post_call(engine, session),
        )
        await ws.send_text(json.dumps({
            "type": "post_call",
            **{k: v for k, v in report.items() if k != "raw"},
        }))

        await ws.send_text(json.dumps({"type": "done"}))

    except WebSocketDisconnect:
        pass
    except Exception:
        traceback.print_exc(file=sys.stderr)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Mode: "live" (streaming with progressive verdict) or "eval" (independent files).
    mode = ws.query_params.get("mode", "live")
    if mode not in ("live", "eval"):
        mode = "live"

    # Connection-level settings from query params.
    default_pipeline = ws.query_params.get("pipeline", "b")
    use_compact = ws.query_params.get("compact", "") == "1"

    session = CallSession(mode=mode)

    # Session log
    session_turns: list[dict] = []
    session_start = time.time()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # Allow clients to clear history / reset session.
            if msg.get("clear_history"):
                session.reset()
                await ws.send_text(json.dumps({"type": "history_cleared"}))
                continue

            has_audio = bool(msg.get("audio"))
            has_image = bool(msg.get("image"))

            if not has_audio and not has_image:
                continue

            # Pipeline selection — per-message overrides connection default.
            pipeline = (msg.get("pipeline") or default_pipeline or "b").lower()
            if pipeline not in ("a", "b"):
                pipeline = "b"
            if not has_audio:
                pipeline = "b"

            # In eval mode, reset session between files for independent evaluation.
            if mode == "eval":
                session.reset()

            # Run inference via shared run_turn.
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: run_turn(
                    engine,
                    session,
                    audio_b64=msg.get("audio"),
                    image_b64=msg.get("image"),
                    pipeline=pipeline,
                    system_prompt=SYSTEM_PROMPT_COMPACT if use_compact else SYSTEM_PROMPT,
                    detect_bank_fn=_detect_bank,
                    bank_context_fn=_bank_context,
                    lenient_parse_fn=_lenient_parse,
                    log_parse_failure_fn=_log_parse_failure,
                    asr_transcribe_fn=asr.transcribe_wav,
                ),
            )

            if result.get("error"):
                await ws.send_text(json.dumps({
                    "type": "result",
                    "input_type": result.get("input_type", "audio"),
                    "pipeline": pipeline,
                    "turn": len(session_turns) + 1,
                    "transcription": result.get("transcription", ""),
                    "description": result.get("description", ""),
                    "verdict": "ERROR",
                    "summary": result["error"],
                    "recommendations": [],
                    "asr_time": round(result.get("asr_time", 0), 2),
                    "llm_time": round(result.get("llm_time", 0), 2),
                    "total_time": round(result.get("asr_time", 0) + result.get("llm_time", 0), 2),
                    "error": result["error"],
                }))
                continue

            verdict = result["verdict"]
            transcription = result.get("transcription", "")
            description = result.get("description", "")
            summary = result.get("summary", "")
            recommendations = result.get("recommendations", [])
            asr_time = result.get("asr_time", 0)
            llm_time = result.get("llm_time", 0)
            result_type = result.get("input_type", "audio")
            detected_bank = result.get("detected_bank")

            label = transcription or description
            print(f"LLM ({llm_time:.2f}s) [{result_type}] {label[:80]!r} → {verdict}")
            if detected_bank:
                print(f"  Bank KB injected: {detected_bank}")

            # Post-inference: enrich recommendations with bank hotline.
            if detected_bank and detected_bank != "bogusbank":
                b = BANK_KB[detected_bank]
                hotline_rec = f"Call {detected_bank.upper()} official hotline: {b['hotline']}"
                if hotline_rec not in recommendations:
                    recommendations.append(hotline_rec)

            total_time = asr_time + llm_time

            # Log
            turn = {
                "turn": len(session_turns) + 1,
                "type": result_type,
                "pipeline": pipeline,
                "transcription": transcription,
                "description": description,
                "verdict": verdict,
                "summary": summary,
                "recommendations": recommendations,
                "asr_time": round(asr_time, 2),
                "llm_time": round(llm_time, 2),
                "total_time": round(total_time, 2),
            }
            session_turns.append(turn)

            # Build response — in live mode include progressive call_verdict.
            response_msg: dict = {
                "type": "result",
                "input_type": result_type,
                "pipeline": pipeline,
                "turn": turn["turn"],
                "transcription": transcription,
                "description": description,
                "verdict": verdict,
                "risk_level": result.get("risk_level", verdict),
                "risk_score": result.get("risk_score", 0),
                "summary": summary,
                "recommendations": recommendations,
                "asr_time": round(asr_time, 2),
                "llm_time": round(llm_time, 2),
                "total_time": round(total_time, 2),
            }
            if mode == "live":
                response_msg["call_verdict"] = session.current_risk_level
                response_msg["call_risk_level"] = session.current_risk_level
                response_msg["call_risk_score"] = session.current_risk_score
                response_msg["segment_verdict"] = verdict
                response_msg["running_transcript"] = session.running_transcript
                response_msg["notepad"] = {
                    "caller_identity": session.caller_identity,
                    "info_requested": list(session.info_requested),
                    "call_reason": session.call_reason,
                    "risk_factors": list(session.risk_factors),
                }

            await ws.send_text(json.dumps(response_msg))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception:
        print("!! WebSocket handler crashed:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        # Save session log
        if session_turns:
            logs_dir = Path(__file__).resolve().parent.parent / "logs"
            logs_dir.mkdir(exist_ok=True)
            from datetime import datetime, timezone

            session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log = {
                "session_id": session_id,
                "mode": mode,
                "backend": "litert-lm-engine",
                "model": f"{HF_REPO}/{HF_FILENAME}",
                "duration_s": round(time.time() - session_start, 1),
                "final_verdict": session.current_verdict,
                "turns": session_turns,
            }
            log_file = logs_dir / f"session_{session_id}.json"
            log_file.write_text(json.dumps(log, indent=2, ensure_ascii=False))
            print(f"Session log: {log_file}")


def _kill_existing(port: int) -> None:
    """Kill any process already listening on *port*."""
    try:
        out = subprocess.check_output(
            ["fuser", f"{port}/tcp"], stderr=subprocess.DEVNULL
        )
        for pid_str in out.decode().split():
            pid = int(pid_str)
            if pid != os.getpid():
                os.kill(pid, signal.SIGTERM)
                print(f"Killed previous server (pid {pid}) on port {port}")
    except (subprocess.CalledProcessError, ProcessLookupError):
        pass  # nothing listening — fine


if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", "8000"))
    _kill_existing(port)
    uvicorn.run(app, host="0.0.0.0", port=port)

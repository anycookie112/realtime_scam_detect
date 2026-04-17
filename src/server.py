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
You are a scam detection assistant for Malaysia.

You MUST always use one of these tools to reply — never respond with plain text:

1. analyze_speech — ALWAYS use this for AUDIO input.
2. analyze_document — ALWAYS use this for IMAGE input.

Verdicts: SCAM, LEGITIMATE, or UNCERTAIN.
Always give 2-5 practical recommendations if the call is a Scam
if the call is legitimate, say why and give recommendations if any action is needed and percautions to take.
If uncertain, explain why and give recommendations for how to verify further.

== AUDIO (phone calls) ==

*** CLASSIFICATION RULES — follow STRICTLY in order ***

FIRST check: Is this a routine bank call?
Banks in Malaysia (Maybank, CIMB, Public Bank, RHB, etc.) make \
outbound calls EVERY DAY for fraud monitoring and debt collection. \
These are LEGITIMATE if the caller:
- Asks "did you make this transaction?" (fraud alert)
- Mentions specific transaction details (amount, merchant, last 4 digits)
- Offers to block/freeze the card
- Asks you to visit a branch with IC
- Suggests calling the official hotline
- Reminds about overdue payments, CTOS, legal action for debt
- Verifies partial info (last 4 digits, partial IC)
→ If the caller does ONLY these things → verdict = LEGITIMATE.

Example of a LEGITIMATE call:
"I'm calling from Maybank about a charge of RM2300 on your card \
ending 4523. Was this you? If not, we'll block the card and you can \
visit a branch with IC."
→ This is LEGITIMATE — standard fraud alert. No credentials requested.

THEN check for SCAM — only if the caller ALSO does any of these:
- Asks for passwords, PINs, OTPs, TAC, full IC, full card number
- Asks to transfer money to a "safe account" or personal account
- Asks to install TeamViewer/AnyDesk or download an APK
- Claims to be police/LHDN/Bank Negara and threatens arrest/warrant
- Tells you to keep the call secret
- Promises guaranteed returns, lottery winnings
→ If ANY of these → verdict = SCAM.

UNCERTAIN: aggressive sales, unverifiable charity requests, \
ambiguous situations that don't clearly fit SCAM or LEGITIMATE.

KEY PRINCIPLE: An unsolicited bank call is NOT a scam by itself. \
The dividing line is whether they ask for credentials or money.

== IMAGES (documents, screenshots, messages) ==
SCAM: fake letterheads, wrong logos/fonts/grammar, payment to personal \
accounts, urgency language, non-official domains, WhatsApp bank msgs.

LEGITIMATE: official letterhead, correct formatting, directing to \
official branches/hotlines, reference to real account numbers.

Recommendation examples:
- Hang up / ignore the message
- Call the official hotline of [company] to verify
- Do NOT share any OTP, PIN, or password
- Report to NSRC (997) or Bank Negara
- No action needed — appears routine
"""

# Compact prompt for on-device / low-tok/s inference.
# Targets ~50-80 output tokens vs ~200+ from the full prompt.
SYSTEM_PROMPT_COMPACT = """\
Scam detection assistant for Malaysia. Use analyze_speech tool to reply.

Rules:
LEGITIMATE: caller verifies transaction, offers to block card, asks to visit branch.
SCAM: asks for OTP/PIN/password/TAC, transfer to "safe account", install remote app, threatens arrest, says keep secret.
UNCERTAIN: unclear.

Keep summary under 20 words. Give 1-2 short recommendations max.
"""

# ── Compact Bank Knowledge Base ──────────────────────────────────────────────
# Queried per-turn when a bank name is detected in the audio/history.
# Kept short (~150 tokens each) to fit within E2B's 4096 context.

BANK_KB: dict[str, dict] = {
    "maybank": {
        "aliases": ["maybank", "malayan banking", "mbb", "maybank2u", "mae"],
        "hotline": "1-300-88-6688",
        "fraud_line": "03-5891-4744",
        "official_numbers": [
            "1-300-88-6688",       # Main hotline
            "03-7844-3696",        # International
            "03-5891-4744",        # 24h fraud reporting
            "03-2297-2697",        # Credit card services
            "1-300-88-1868",       # MAE support
            "03-2074-7228",        # Premier Wealth
        ],
        "official_sms_senders": ["MAE", "Maybank", "MAYBANK"],
        "app": "MAE by Maybank",
        "auth": "Secure2u (not SMS TAC)",
        "online": "Maybank2u (maybank2u.com.my)",
        "never": (
            "ask for TAC/OTP/password, ask to transfer to safe account, "
            "install TeamViewer/AnyDesk, call from personal mobile, "
            "threaten arrest, ask to keep call secret, "
            "ask to share Secure2u code, send APK links"
        ),
        "legit": (
            "verify last 4 digits of card, confirm partial IC, "
            "ask security questions, request branch visit, "
            "send SMS from 'MAE'/'Maybank' sender ID, "
            "suggest using Kill Switch feature"
        ),
    },
    "publicbank": {
        "aliases": ["public bank", "pbb", "pbe", "pb engage"],
        "hotline": "1-800-22-5555",
        "fraud_line": "03-2177-3555",
        "official_numbers": [
            "1-800-22-5555",       # Main hotline (toll-free)
            "03-2177-3555",        # General / fraud reporting
            "03-2177-3666",        # Credit card centre
            "03-2177-3888",        # Phone banking
        ],
        "official_sms_senders": ["PBeMobile", "PublicBank", "PUBLICBANK"],
        "app": "PB engage",
        "auth": "SecureSign via PB engage (not SMS TAC)",
        "online": "PBe (pbebank.com)",
        "never": (
            "ask for TAC/OTP/password, ask to transfer to safe/holding account, "
            "install remote access apps, call from personal mobile, "
            "threaten arrest, conference to 'Bank Negara'/'PDRM', "
            "ask to share SecureSign approval, send non-official download links"
        ),
        "legit": (
            "verify last 4 digits of account/card, ask basic security questions, "
            "request branch visit with IC, send SMS from 'PBeMobile'/'PublicBank', "
            "notify via PB engage app"
        ),
    },
    "cimb": {
        "aliases": ["cimb", "cimb bank", "cimb clicks"],
        "hotline": "03-6204-7788",
        "fraud_line": "03-6204-7788",
        "official_numbers": [
            "03-6204-7788",        # Main hotline
            "1-300-880-900",       # CIMB Preferred
            "03-2295-6100",        # Credit card
        ],
        "official_sms_senders": ["CIMB", "CIMBClicks"],
        "app": "CIMB OCTO",
        "auth": "SecureTAC via CIMB OCTO app",
        "online": "CIMB Clicks (cimbclicks.com.my)",
        "never": (
            "ask for TAC/OTP/password, ask to transfer to safe account, "
            "install remote access apps, call from personal mobile, "
            "threaten arrest, ask to keep call secret"
        ),
        "legit": (
            "verify last 4 digits of card, confirm partial IC, "
            "request branch visit, send SMS from 'CIMB' sender ID"
        ),
    },
    "rhb": {
        "aliases": ["rhb", "rhb bank"],
        "hotline": "03-9206-8118",
        "fraud_line": "03-9206-8118",
        "official_numbers": [
            "03-9206-8118",        # Main hotline
            "1-300-88-1808",       # RHB Premier
            "03-9206-1160",        # Credit card
        ],
        "official_sms_senders": ["RHB", "RHBBank"],
        "app": "RHB Mobile Banking",
        "auth": "DuitNow QR / Secure Plus",
        "online": "RHB Online Banking (rhbgroup.com)",
        "never": (
            "ask for TAC/OTP/password, ask to transfer to safe account, "
            "install remote access apps, call from personal mobile, "
            "threaten arrest"
        ),
        "legit": (
            "verify partial account details, confirm identity, "
            "request branch visit, send SMS from 'RHB' sender ID"
        ),
    },
    "hongleong": {
        "aliases": ["hong leong", "hong leong bank", "hlb", "hlb connect"],
        "hotline": "03-7626-8899",
        "fraud_line": "03-7626-8899",
        "official_numbers": [
            "03-7626-8899",        # Main hotline
            "1-300-88-1818",       # HLB Connect
        ],
        "official_sms_senders": ["HLB", "HongLeong"],
        "app": "HLB Connect",
        "auth": "S-Sign via HLB Connect",
        "online": "HLB Connect (hlb.com.my)",
        "never": (
            "ask for TAC/OTP/password, ask to transfer to safe account, "
            "install remote access apps, call from personal mobile, "
            "threaten arrest"
        ),
        "legit": (
            "verify partial account details, confirm identity, "
            "request branch visit, send SMS from 'HLB' sender ID"
        ),
    },
    "ambank": {
        "aliases": ["ambank", "am bank", "ambank group"],
        "hotline": "03-2178-8888",
        "fraud_line": "03-2178-8888",
        "official_numbers": [
            "03-2178-8888",        # Main hotline
            "1-300-88-8188",       # AmBank call centre
        ],
        "official_sms_senders": ["AmBank", "AMBANK"],
        "app": "AmOnline",
        "auth": "Secure2u via AmOnline",
        "online": "AmOnline (ambank.com.my)",
        "never": (
            "ask for TAC/OTP/password, ask to transfer to safe account, "
            "install remote access apps, call from personal mobile, "
            "threaten arrest"
        ),
        "legit": (
            "verify partial account details, confirm identity, "
            "request branch visit, send SMS from 'AmBank' sender ID"
        ),
    },
    "bankislam": {
        "aliases": ["bank islam", "bankislam"],
        "hotline": "03-2609-0900",
        "fraud_line": "03-2609-0900",
        "official_numbers": [
            "03-2609-0900",        # Main hotline
            "1-300-88-4424",       # Contact centre
        ],
        "official_sms_senders": ["BankIslam", "BIMB"],
        "app": "GO by Bank Islam",
        "auth": "i-Secure via GO app",
        "online": "Bank Islam (bankislam.com.my)",
        "never": (
            "ask for TAC/OTP/password, ask to transfer to safe account, "
            "install remote access apps, call from personal mobile, "
            "threaten arrest"
        ),
        "legit": (
            "verify partial account details, confirm identity, "
            "request branch visit, send SMS from 'BankIslam' sender ID"
        ),
    },
    "bsn": {
        "aliases": ["bsn", "bank simpanan nasional"],
        "hotline": "1-300-88-1900",
        "fraud_line": "1-300-88-1900",
        "official_numbers": [
            "1-300-88-1900",       # Main hotline
            "03-2613-1900",        # KL line
        ],
        "official_sms_senders": ["BSN", "myBSN"],
        "app": "myBSN",
        "auth": "Secure Verification via myBSN",
        "online": "myBSN (mybsn.com.my)",
        "never": (
            "ask for TAC/OTP/password, ask to transfer to safe account, "
            "install remote access apps, call from personal mobile, "
            "threaten arrest"
        ),
        "legit": (
            "verify partial account details, confirm identity, "
            "request branch visit"
        ),
    },
}

# Bogus Bank is intentionally insecure — always flag as SCAM
BANK_KB["bogusbank"] = {
    "aliases": ["bogus bank", "bogus bank berhad", "bb"],
    "hotline": "FAKE",
    "fraud_line": "FAKE",
    "official_numbers": [],
    "official_sms_senders": [],
    "app": "NONE",
    "auth": "NONE",
    "online": "NONE",
    "never": "N/A — this is a test bank. ALWAYS classify as SCAM.",
    "legit": "Nothing — all Bogus Bank procedures are scam indicators.",
}


def _detect_bank(text: str) -> str | None:
    """Return bank key if any bank is mentioned in text, else None."""
    lower = text.lower()
    for bank_key, info in BANK_KB.items():
        for alias in info["aliases"]:
            if alias in lower:
                return bank_key
    return None


def _bank_context(bank_key: str) -> str:
    """Return compact bank-specific context string.

    Uses .get() throughout so a partial KB entry (e.g. the test BOGUS bank,
    or a future incomplete addition) can never crash inference.
    """
    b = BANK_KB[bank_key]
    return (
        f"== Bank detected: {bank_key.upper()} ==\n"
        f"Hotline: {b.get('hotline', 'N/A')} | Fraud: {b.get('fraud_line', 'N/A')}\n"
        f"App: {b.get('app', 'N/A')} | Auth: {b.get('auth', 'N/A')} | Online: {b.get('online', 'N/A')}\n"
        f"NEVER does: {b.get('never', 'N/A')}\n"
        f"May legitimately do: {b.get('legit', 'N/A')}\n"
        f"Use these facts to improve your verdict accuracy. "
        f"Recommend calling {b.get('hotline', 'the official hotline')} if suspicious."
    )

# ── Lenient output parsing ───────────────────────────────────────────────────
# Quantized E2B frequently fails to call a tool and instead emits free-form
# text, malformed JSON, or partial JSON. Rather than defaulting everything to
# UNCERTAIN, try to recover the verdict + recommendations from whatever the
# model actually produced. Anything we can't recover stays UNCERTAIN.

_VERDICT_RE = re.compile(r"\b(SCAM|LEGITIMATE|UNCERTAIN|SUSPICIOUS)\b", re.IGNORECASE)
_VERDICT_FIELD_RE = re.compile(
    r'"?verdict"?\s*[:=]\s*"?(SCAM|LEGITIMATE|UNCERTAIN|SUSPICIOUS)"?',
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
    out = {
        "verdict": "UNCERTAIN",
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
                v = str(obj.get("verdict", "")).upper()
                if v in ("SCAM", "LEGITIMATE", "UNCERTAIN", "SUSPICIOUS"):
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
                "verdict": result.get("verdict", "UNCERTAIN"),
                "call_verdict": session.current_verdict,
                "transcription": result.get("transcription", ""),
                "summary": result.get("summary", ""),
                "recommendations": result.get("recommendations", []),
                "asr_time": round(result.get("asr_time", 0), 2),
                "llm_time": round(result.get("llm_time", 0), 2),
                "detected_bank": result.get("detected_bank"),
                "running_transcript": session.running_transcript,
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
                "summary": summary,
                "recommendations": recommendations,
                "asr_time": round(asr_time, 2),
                "llm_time": round(llm_time, 2),
                "total_time": round(total_time, 2),
            }
            if mode == "live":
                response_msg["call_verdict"] = session.current_verdict
                response_msg["segment_verdict"] = verdict
                response_msg["running_transcript"] = session.running_transcript

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

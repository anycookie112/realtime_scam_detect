"""Pipeline A ASR: faster-whisper transcription wrapper.

Lazily loads a single whisper model on first call. Used by the WebSocket
handler when the client requests `pipeline: "a"` (Whisper → text → Gemma).

Configurable via env:
    WHISPER_MODEL_SIZE   default "base"   (tiny, base, small, medium, large-v3)
    WHISPER_DEVICE       default "auto"   (cpu, cuda, auto)
    WHISPER_COMPUTE_TYPE default "int8"   (int8, int8_float16, float16, float32)
    WHISPER_LANGUAGE     default ""       (empty = auto-detect)
"""

from __future__ import annotations

import io
import os
import time
import wave
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


_model: "WhisperModel | None" = None


@dataclass
class TranscriptionResult:
    text: str
    language: str
    asr_time_s: float


def _decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV blob to a mono float32 numpy array at its native rate.

    faster-whisper accepts a numpy array directly, which avoids a temp file.
    Resampling to 16kHz is handled internally by ctranslate2.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        frames = w.readframes(w.getnframes())

    if sampwidth == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio, sr


def load_model() -> "WhisperModel":
    global _model
    if _model is not None:
        return _model
    from faster_whisper import WhisperModel

    size = os.getenv("WHISPER_MODEL_SIZE", "base")
    device = os.getenv("WHISPER_DEVICE", "auto")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    print(f"Loading Whisper model: size={size} device={device} compute={compute_type}")
    _model = WhisperModel(size, device=device, compute_type=compute_type)
    print("Whisper model loaded.")
    return _model


def transcribe_wav(wav_bytes: bytes) -> TranscriptionResult:
    """Transcribe a WAV blob. Auto-detects language unless WHISPER_LANGUAGE is set."""
    model = load_model()
    audio, _sr = _decode_wav(wav_bytes)
    language = os.getenv("WHISPER_LANGUAGE") or None

    t0 = time.time()
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=1,            # greedy — Pipeline A is the realtime path
        vad_filter=False,       # short test clips: skip VAD overhead
        condition_on_previous_text=False,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    asr_time = time.time() - t0
    return TranscriptionResult(
        text=text,
        language=info.language,
        asr_time_s=asr_time,
    )

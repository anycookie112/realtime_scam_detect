"""Audio utilities — WAV splitting and encoding for streaming simulation.

Used by the eval harness to chunk a complete WAV file into overlapping
segments that simulate live streaming capture.
"""

from __future__ import annotations

import io
import wave

import numpy as np

from asr import decode_wav


def split_wav(
    wav_bytes: bytes,
    chunk_s: float = 4.0,
    overlap_s: float = 0.5,
) -> list[bytes]:
    """Split a WAV file into overlapping chunks.

    Each chunk is returned as valid WAV bytes (with header).  Chunks
    shorter than 0.5 s at the tail are dropped.
    """
    audio, sr = decode_wav(wav_bytes)
    chunk_samples = int(chunk_s * sr)
    overlap_samples = int(overlap_s * sr)
    step = chunk_samples - overlap_samples
    min_samples = int(0.5 * sr)  # drop chunks < 0.5 s

    chunks: list[bytes] = []
    for start in range(0, len(audio), step):
        end = min(start + chunk_samples, len(audio))
        chunk_audio = audio[start:end]
        if len(chunk_audio) < min_samples:
            break
        chunks.append(encode_wav(chunk_audio, sr))

    return chunks


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 mono numpy array to 16-bit PCM WAV bytes.

    Inverse of ``asr.decode_wav`` (for 16-bit PCM).
    """
    # Clip to [-1, 1] and convert to int16.
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)        # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())

    return buf.getvalue()

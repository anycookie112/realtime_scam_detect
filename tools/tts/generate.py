#!/usr/bin/env python3
"""Generate test audio from scam detection transcripts.

English & Mixed: Qwen3-TTS-12Hz-1.7B-CustomVoice (local)
Malay:           MMS-TTS facebook/mms-tts-zlm

Usage:
    uv run generate.py                          # all languages, all tests
    uv run generate.py --lang en                # English only
    uv run generate.py --lang ms                # Malay only
    uv run generate.py --lang mixed             # Mixed only
    uv run generate.py --test maybank_obvious_scam   # one test, all languages
    uv run generate.py --list                   # list available test names
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Output directory: project_root/test_audio/
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "test_audio"

# Local model path for Qwen3-TTS
QWEN_MODEL_PATH = "/home/robust/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# HuggingFace model ID for Malay MMS-TTS
MMS_MODEL_ID = "facebook/mms-tts-zlm"

# Default transcript file (next to this script)
DEFAULT_TRANSCRIPTS_FILE = Path(__file__).resolve().parent / "transcripts.json"


# ─────────────────────────────────────────────────────────────────────────────
# Load transcripts + voice config from JSON
# ─────────────────────────────────────────────────────────────────────────────

def load_transcripts(path: Path = DEFAULT_TRANSCRIPTS_FILE) -> tuple[dict, dict, dict]:
    """Load transcripts.json → (voice_map, all_transcripts, voice_profiles).

    Returns:
        voice_map: {test_name: {"speaker":..., "instruct":...}}
        all_transcripts: {"en": {...}, "ms": {...}, "mixed": {...}}
        voice_profiles: raw profiles dict (for reference only)
    """
    with open(path) as f:
        data = json.load(f)

    profiles = data["voice_profiles"]
    # Resolve voice_map references → actual profile dicts
    voice_map = {name: profiles[ref] for name, ref in data["voice_map"].items()}
    all_transcripts = data["transcripts"]
    return voice_map, all_transcripts, profiles


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def load_qwen_tts(model_path: str = QWEN_MODEL_PATH):
    """Load Qwen3-TTS CustomVoice from a local directory."""
    from qwen_tts import Qwen3TTSModel

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    print(f"Loading Qwen3-TTS from {model_path} (attn={attn_impl})...")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    print("Qwen3-TTS loaded.")
    return model


def load_mms_tts(model_id: str = MMS_MODEL_ID):
    """Load MMS-TTS VITS model for Malay."""
    from transformers import VitsModel, AutoTokenizer

    print(f"Loading MMS-TTS ({model_id})...")
    model = VitsModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("MMS-TTS loaded.")
    return model, tokenizer


def unload_model(*models):
    """Free GPU memory."""
    for m in models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_qwen(model, text: str, speaker: str, instruct: str) -> tuple[np.ndarray, int]:
    """Generate audio with Qwen3-TTS. Returns (waveform, sample_rate)."""
    wavs, sr = model.generate_custom_voice(
        text=text,
        language="Auto",
        speaker=speaker,
        instruct=instruct,
    )
    return wavs[0], sr


def generate_mms(model, tokenizer, text: str) -> tuple[np.ndarray, int]:
    """Generate audio with MMS-TTS VITS. Returns (waveform, sample_rate)."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    wav = output.waveform[0].cpu().numpy()
    return wav, model.config.sampling_rate


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Generate scam detection test audio")
    p.add_argument("--lang", choices=["en", "ms", "mixed", "all"], default="all",
                    help="Language to generate (default: all)")
    p.add_argument("--test", type=str, default=None,
                    help="Generate only this test case (default: all)")
    p.add_argument("--list", action="store_true",
                    help="List available test names and exit")
    p.add_argument("--transcripts", type=str, default=str(DEFAULT_TRANSCRIPTS_FILE),
                    help="Path to transcripts JSON file")
    p.add_argument("--qwen-model", type=str, default=QWEN_MODEL_PATH,
                    help="Path to local Qwen3-TTS model")
    p.add_argument("--mms-model", type=str, default=MMS_MODEL_ID,
                    help="MMS-TTS model ID or path")
    p.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                    help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()

    voice_map, all_transcripts, _ = load_transcripts(Path(args.transcripts))
    en_transcripts = all_transcripts.get("en", {})
    ms_transcripts = all_transcripts.get("ms", {})
    mixed_transcripts = all_transcripts.get("mixed", {})

    if args.list:
        for name in en_transcripts:
            print(name)
        return

    test_names = [args.test] if args.test else list(en_transcripts.keys())
    for name in test_names:
        if name not in en_transcripts:
            print(f"Unknown test: {name}")
            print(f"Available: {', '.join(en_transcripts.keys())}")
            sys.exit(1)

    langs = ["en", "ms", "mixed"] if args.lang == "all" else [args.lang]
    output_dir = Path(args.output)

    # Create output directories
    for lang in langs:
        (output_dir / lang).mkdir(parents=True, exist_ok=True)

    # ── Generate English ──
    if "en" in langs:
        qwen = load_qwen_tts(args.qwen_model)
        print(f"\n{'='*60}")
        print("Generating English audio...")
        print(f"{'='*60}")
        for name in test_names:
            text = en_transcripts[name]
            voice = voice_map[name]
            out_path = output_dir / "en" / f"{name}.wav"
            print(f"  {name}...", end=" ", flush=True)
            wav, sr = generate_qwen(qwen, text, voice["speaker"], voice["instruct"])
            sf.write(str(out_path), wav, sr)
            print(f"OK ({len(wav)/sr:.1f}s)")

        # Keep qwen loaded if mixed also needed, otherwise unload
        if "mixed" not in langs:
            unload_model(qwen)
            qwen = None

    # ── Generate Malay ──
    if "ms" in langs:
        mms_model, mms_tok = load_mms_tts(args.mms_model)
        print(f"\n{'='*60}")
        print("Generating Malay audio...")
        print(f"{'='*60}")
        for name in test_names:
            text = ms_transcripts[name]
            out_path = output_dir / "ms" / f"{name}.wav"
            print(f"  {name}...", end=" ", flush=True)
            wav, sr = generate_mms(mms_model, mms_tok, text)
            sf.write(str(out_path), wav, sr)
            print(f"OK ({len(wav)/sr:.1f}s)")
        unload_model(mms_model, mms_tok)

    # ── Generate Mixed / Manglish ──
    if "mixed" in langs:
        # Load qwen if not already loaded from English pass
        if "en" not in langs:
            qwen = load_qwen_tts(args.qwen_model)
        print(f"\n{'='*60}")
        print("Generating Mixed/Manglish audio...")
        print(f"{'='*60}")
        for name in test_names:
            text = mixed_transcripts[name]
            voice = voice_map[name]
            out_path = output_dir / "mixed" / f"{name}.wav"
            print(f"  {name}...", end=" ", flush=True)
            wav, sr = generate_qwen(qwen, text, voice["speaker"], voice["instruct"])
            sf.write(str(out_path), wav, sr)
            print(f"OK ({len(wav)/sr:.1f}s)")
        unload_model(qwen)

    total = len(test_names) * len(langs)
    print(f"\nDone! Generated {total} audio files in {output_dir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate WAV audio files for all transcripts in test_data/*.json using edge-tts.

Edge-tts supports all 4 languages we need (en, ms, zh) plus mixed (uses en voice).
No GPU needed. Outputs 16kHz mono WAV files into test_audio/<lang>/.

Usage:
    uv run tools/tts/generate_audio.py
    uv run tools/tts/generate_audio.py --lang en
    uv run tools/tts/generate_audio.py --lang adversarial
    uv run tools/tts/generate_audio.py --concurrent 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

import edge_tts
import miniaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
TEST_AUDIO_DIR = PROJECT_ROOT / "test_audio"

# Voice selection — varied so calls don't all sound identical
VOICES = {
    "en":      ["en-US-GuyNeural", "en-US-AriaNeural", "en-GB-RyanNeural"],
    "ms":      ["ms-MY-OsmanNeural", "ms-MY-YasminNeural"],
    "mixed":   ["en-US-GuyNeural", "en-US-AriaNeural", "ms-MY-YasminNeural"],
    "zh":      ["zh-CN-YunjianNeural", "zh-CN-XiaoxiaoNeural", "zh-CN-YunyangNeural"],
}

# Voice rotation — assign by risk level so scammers sound urgent, legit calm
VOICE_BY_RISK = {
    "HIGH_RISK":    0,  # first voice (typically male/authoritative)
    "MEDIUM_RISK":  1,  # second voice
    "LOW_RISK":     2 if False else 0,  # whatever's available
    "SAFE":         1,
}


def pick_voice(lang: str, risk: str) -> str:
    voices = VOICES.get(lang, VOICES["en"])
    idx = VOICE_BY_RISK.get(risk, 0) % len(voices)
    return voices[idx]


async def gen_mp3(text: str, voice: str, out_path: Path, rate: str = "-5%"):
    """Generate MP3 via edge-tts."""
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    out_path.write_bytes(audio_data)


def mp3_to_wav(mp3_path: Path, wav_path: Path, sample_rate: int = 16000):
    """Convert MP3 to 16kHz mono WAV using miniaudio."""
    decoded = miniaudio.decode_file(
        str(mp3_path),
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=sample_rate,
    )
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(decoded.samples)


async def gen_one_scenario(scenario: dict, lang: str, out_dir: Path, sem: asyncio.Semaphore):
    """Generate WAV for a single scenario. Skip if already exists."""
    name = scenario["name"]
    wav_path = out_dir / f"{name}.wav"
    if wav_path.exists():
        return "skipped"

    async with sem:
        voice = pick_voice(lang, scenario.get("risk_level", "MEDIUM_RISK"))
        mp3_path = out_dir / f"{name}.mp3"
        try:
            await gen_mp3(scenario["text"], voice, mp3_path)
            mp3_to_wav(mp3_path, wav_path)
            mp3_path.unlink()  # cleanup intermediate MP3
            return "ok"
        except Exception as exc:
            print(f"  ! {name}: {exc}")
            if mp3_path.exists():
                mp3_path.unlink()
            return "failed"


async def gen_all_in_file(json_path: Path, out_dir: Path, concurrent: int):
    """Generate audio for every scenario in a JSON file."""
    scenarios = json.loads(json_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrent)
    print(f"\n{'='*60}")
    print(f"  {json_path.name}: {len(scenarios)} scenarios → {out_dir}/")
    print(f"{'='*60}")

    results = await asyncio.gather(
        *[gen_one_scenario(s, s.get("lang", "en"), out_dir, sem) for s in scenarios]
    )

    ok = sum(1 for r in results if r == "ok")
    skipped = sum(1 for r in results if r == "skipped")
    failed = sum(1 for r in results if r == "failed")
    print(f"  Generated: {ok}  Skipped (already exist): {skipped}  Failed: {failed}")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="all",
                        help="en | ms | mixed | zh | adversarial | all")
    parser.add_argument("--concurrent", type=int, default=10,
                        help="Max concurrent TTS requests (default: 10)")
    args = parser.parse_args()

    if args.lang == "all":
        targets = [
            (TEST_DATA_DIR / "en.json", TEST_AUDIO_DIR / "en"),
            (TEST_DATA_DIR / "ms.json", TEST_AUDIO_DIR / "ms"),
            (TEST_DATA_DIR / "mixed.json", TEST_AUDIO_DIR / "mixed"),
            (TEST_DATA_DIR / "zh.json", TEST_AUDIO_DIR / "zh"),
            (TEST_DATA_DIR / "adversarial.json", None),  # special: routes by lang field
        ]
    elif args.lang == "adversarial":
        targets = [(TEST_DATA_DIR / "adversarial.json", None)]
    else:
        targets = [(TEST_DATA_DIR / f"{args.lang}.json", TEST_AUDIO_DIR / args.lang)]

    for json_path, out_dir in targets:
        if not json_path.exists():
            print(f"  ! Skipping {json_path}: not found")
            continue

        if out_dir is None:
            # Adversarial: route each scenario to its own lang directory
            scenarios = json.loads(json_path.read_text())
            by_lang: dict[str, list] = {}
            for s in scenarios:
                by_lang.setdefault(s.get("lang", "en"), []).append(s)

            for lang, lang_scenarios in by_lang.items():
                lang_dir = TEST_AUDIO_DIR / lang
                lang_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n{'='*60}")
                print(f"  adversarial ({lang}): {len(lang_scenarios)} scenarios → {lang_dir}/")
                print(f"{'='*60}")
                sem = asyncio.Semaphore(args.concurrent)
                results = await asyncio.gather(
                    *[gen_one_scenario(s, lang, lang_dir, sem) for s in lang_scenarios]
                )
                ok = sum(1 for r in results if r == "ok")
                skipped = sum(1 for r in results if r == "skipped")
                failed = sum(1 for r in results if r == "failed")
                print(f"  Generated: {ok}  Skipped: {skipped}  Failed: {failed}")
        else:
            await gen_all_in_file(json_path, out_dir, args.concurrent)


if __name__ == "__main__":
    asyncio.run(main())

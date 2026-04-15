#!/usr/bin/env python3
"""Quick WebSocket test client for the scam detection server.

Sends a WAV file, image, or generated noise to the server
via WebSocket and prints the model's response.

Usage:
    uv run src/test_ws.py                          # generate test tone
    uv run src/test_ws.py path/to/audio.wav         # send a WAV file
    uv run src/test_ws.py --image path/to/doc.png   # send an image for document analysis
    uv run src/test_ws.py a.wav b.wav               # multi-turn audio
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import struct
import sys

import numpy as np


def generate_silence_wav(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate a short silent WAV (useful as a smoke test for the pipeline)."""
    samples = int(duration_s * sample_rate)
    # Very light white noise so it's not pure silence
    audio = (np.random.randn(samples) * 100).astype(np.int16)
    buf = io.BytesIO()
    # Write WAV header manually
    data_size = samples * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio.tobytes())
    return buf.getvalue()


def load_wav_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def text_to_wav_espeak(text: str) -> bytes:
    """Use espeak-ng to generate a WAV from text (if available)."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["espeak-ng", "-w", tmp_path, "-s", "140", text],
        check=True,
        capture_output=True,
    )
    with open(tmp_path, "rb") as f:
        data = f.read()
    import os
    os.unlink(tmp_path)
    return data


def print_result(result):
    """Pretty-print a server response."""
    input_type = result.get('input_type', 'audio')
    print(f"\n{'='*60}")
    if input_type == 'document':
        print(f"Description:   {result.get('description', 'N/A')}")
    else:
        print(f"Transcription: {result.get('transcription', 'N/A')}")
    print(f"Verdict:       {result.get('verdict', 'N/A')}")
    print(f"Summary:       {result.get('summary', 'N/A')}")
    recs = result.get('recommendations', [])
    if recs:
        print("Recommendations:")
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r}")
    print(f"LLM time:      {result.get('llm_time', 'N/A')}s")
    print(f"{'='*60}")


async def test_websocket(audio_bytes: bytes, host: str, port: int):
    import websockets

    uri = f"ws://{host}:{port}/ws"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        # Encode audio as base64 (matching the index.html client format)
        audio_b64 = base64.b64encode(audio_bytes).decode()
        payload = json.dumps({"audio": audio_b64})
        print(f"Sending {len(audio_bytes)} bytes of audio ({len(audio_b64)} base64 chars)...")

        await ws.send(payload)
        print("Waiting for response...")

        response = await asyncio.wait_for(ws.recv(), timeout=120)
        result = json.loads(response)
        print_result(result)
        return result


async def test_websocket_image(image_bytes: bytes, host: str, port: int):
    import websockets

    uri = f"ws://{host}:{port}/ws"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        img_b64 = base64.b64encode(image_bytes).decode()
        payload = json.dumps({"image": img_b64})
        print(f"Sending {len(image_bytes)} bytes of image ({len(img_b64)} base64 chars)...")

        await ws.send(payload)
        print("Waiting for response...")

        response = await asyncio.wait_for(ws.recv(), timeout=120)
        result = json.loads(response)
        print_result(result)
        return result


async def test_websocket_multi(wav_files: list[str], host: str, port: int):
    """Send multiple WAV files over a single WebSocket (simulates a multi-turn call)."""
    import websockets

    uri = f"ws://{host}:{port}/ws"
    print(f"Connecting to {uri} (multi-turn: {len(wav_files)} segments)...")

    async with websockets.connect(uri) as ws:
        for i, path in enumerate(wav_files, 1):
            print(f"\n--- Turn {i}/{len(wav_files)}: {path} ---")
            audio_bytes = load_wav_file(path)
            audio_b64 = base64.b64encode(audio_bytes).decode()
            payload = json.dumps({"audio": audio_b64})
            print(f"Sending {len(audio_bytes)} bytes...")

            await ws.send(payload)
            response = await asyncio.wait_for(ws.recv(), timeout=120)
            result = json.loads(response)
            print_result(result)

    print(f"\n{'='*60}")
    print(f"Multi-turn test complete ({len(wav_files)} turns)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Test WebSocket scam detection")
    parser.add_argument("wav_file", nargs="*", help="Path(s) to WAV file(s) to send (multiple = multi-turn)")
    parser.add_argument("--image", type=str, help="Path to an image file (PNG/JPG) for document analysis")
    parser.add_argument("--tts", type=str, help="Generate speech from text via espeak-ng")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    args = parser.parse_args()

    if args.image:
        print(f"Loading image {args.image}...")
        with open(args.image, "rb") as f:
            image = f.read()
        asyncio.run(test_websocket_image(image, args.host, args.port))
    elif args.wav_file and len(args.wav_file) > 1:
        # Multi-turn mode: send multiple files over the same WebSocket
        asyncio.run(test_websocket_multi(args.wav_file, args.host, args.port))
    elif args.wav_file:
        print(f"Loading {args.wav_file[0]}...")
        audio = load_wav_file(args.wav_file[0])
        asyncio.run(test_websocket(audio, args.host, args.port))
    elif args.tts:
        print(f"Generating speech: {args.tts!r}")
        try:
            audio = text_to_wav_espeak(args.tts)
        except FileNotFoundError:
            print("espeak-ng not found. Install with: sudo apt install espeak-ng")
            sys.exit(1)
        asyncio.run(test_websocket(audio, args.host, args.port))
    else:
        print("Generating test noise (2s)...")
        audio = generate_silence_wav()
        asyncio.run(test_websocket(audio, args.host, args.port))


if __name__ == "__main__":
    main()

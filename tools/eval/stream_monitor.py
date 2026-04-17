#!/usr/bin/env python3
"""Live streaming monitor — watch the model reason in real time.

Streams a WAV file chunk-by-chunk and shows a compact live display:
one line per chunk with verdict and key info. Full details go to a log file.

Usage:
    uv run tools/eval/stream_monitor.py test_audio/en/maybank_gradual_scam.wav
    uv run tools/eval/stream_monitor.py test_audio/en/maybank_gradual_scam.wav --chunk-seconds 8
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Silence C++ logs before importing litert_lm.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "NONE")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

console = Console()

# ── Verdict formatting ──────────────────────────────────────────────────────

def v_fmt(v: str) -> str:
    return {"SCAM": "[bold red]SCAM[/]", "LEGITIMATE": "[bold green]LEGIT[/]",
            "UNCERTAIN": "[bold yellow]UNCERTAIN[/]"}.get(v, v)


# ── Engine helpers ──────────────────────────────────────────────────────────

@contextlib.contextmanager
def silence_fds():
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


def load_engine(repo: str, filename: str, backend_name: str):
    import litert_lm
    from huggingface_hub import hf_hub_download

    override = os.environ.get("LITERT_ENGINE_MODEL_PATH", "")
    model_path = override if override else hf_hub_download(repo_id=repo, filename=filename)
    backend = litert_lm.Backend.GPU if backend_name == "gpu" else litert_lm.Backend.CPU
    with silence_fds():
        eng = litert_lm.Engine(
            model_path, backend=backend,
            vision_backend=backend, audio_backend=litert_lm.Backend.CPU,
        )
        eng.__enter__()
    return eng


def unload_engine(engine):
    if engine is None:
        return
    try:
        with silence_fds():
            engine.__exit__(None, None, None)
    except Exception:
        pass
    gc.collect()


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav", type=str, help="Path to WAV file to stream")
    p.add_argument("--pipeline", default="a", choices=["a", "b"])
    p.add_argument("--chunk-seconds", type=float, default=4.0)
    p.add_argument("--overlap-seconds", type=float, default=0.5)
    p.add_argument("--whisper-size", default="base")
    p.add_argument("--model-repo", default=None)
    p.add_argument("--model-file", default=None)
    p.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--compact", action="store_true",
                   help="Use compact system prompt (shorter output, faster on-device)")
    p.add_argument("--skip-chunks", type=int, default=1,
                   help="Only infer every Nth chunk (accumulate transcript in between)")
    args = p.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        console.print(f"[red]File not found:[/] {wav_path}")
        sys.exit(1)

    repo = args.model_repo or os.getenv(
        "LITERT_ENGINE_HF_REPO", "litert-community/gemma-4-E2B-it-litert-lm")
    filename = args.model_file or os.getenv(
        "LITERT_ENGINE_MODEL_FILE", "gemma-4-E2B-it.litertlm")

    # ── Log file setup ──────────────────────────────────────────────────
    logs_dir = PROJECT_ROOT / "logs" / "monitor"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = logs_dir / f"{wav_path.stem}_{ts}.jsonl"

    # ── Load models ─────────────────────────────────────────────────────
    console.print(f"[dim]Loading models...[/]")

    if args.pipeline == "a":
        os.environ["WHISPER_MODEL_SIZE"] = args.whisper_size
        import asr
        asr._model = None  # noqa: SLF001
        asr.load_model()

    engine = load_engine(repo, filename, args.backend)

    from session import CallSession, run_turn, run_post_call
    from audio_utils import split_wav
    import server

    # ── Split audio ─────────────────────────────────────────────────────
    wav_bytes = wav_path.read_bytes()
    chunks = split_wav(wav_bytes, args.chunk_seconds, args.overlap_seconds)
    total_chunks = len(chunks)

    import wave
    with wave.open(str(wav_path), "rb") as w:
        total_duration = w.getnframes() / w.getframerate()

    step = args.chunk_seconds - args.overlap_seconds

    mode_info = f"pipeline {args.pipeline}"
    if args.compact:
        mode_info += ", compact prompt"
    if args.skip_chunks > 1:
        mode_info += f", skip {args.skip_chunks-1}/{args.skip_chunks}"
    console.print(
        f"[dim]Streaming[/] [cyan]{wav_path.name}[/] "
        f"[dim]({total_duration:.0f}s, {total_chunks} chunks, "
        f"{mode_info})[/]"
    )
    console.print(f"[dim]Log:[/] {log_path}")
    console.print()

    # ── Header ──────────────────────────────────────────────────────────
    console.print(
        f" {'#':>3}  {'time':>11}  {'seg':>10}  {'call':>10}  "
        f"{'asr_s':>5}  {'llm_s':>5}  transcript"
    )
    console.print("[dim]" + "─" * 90 + "[/]")

    # ── Stream ──────────────────────────────────────────────────────────
    session = CallSession(mode="live", skip_chunks=args.skip_chunks)
    system_prompt = server.SYSTEM_PROMPT_COMPACT if args.compact else server.SYSTEM_PROMPT
    prev_call_verdict = "UNCERTAIN"
    total_time_start = time.time()
    log_entries: list[dict] = []
    infer_count = 0

    for i, chunk_bytes in enumerate(chunks):
        chunk_start = i * step
        chunk_end = chunk_start + args.chunk_seconds
        time_label = (f"{int(chunk_start//60):02d}:{int(chunk_start%60):02d}"
                      f"-{int(chunk_end//60):02d}:{int(chunk_end%60):02d}")

        # Always run ASR to accumulate transcript, but only run LLM
        # inference when should_infer() says so (respects skip_chunks).
        if args.pipeline == "a":
            import asr as _asr
            r = _asr.transcribe_wav(chunk_bytes)
            session.add_transcript(r.text)
            session.note_chunk_received()

            if not session.should_infer():
                # Log skipped chunk
                console.print(
                    f" {i+1:>3}  {time_label:>11}  [dim]{'---':>10}  "
                    f"{v_fmt(session.current_verdict):>20}       "
                    f"{r.asr_time_s:>5.1f}    [dim]skip[/]  {r.text[:50]}...[/]"
                )
                log_entries.append({
                    "chunk": i + 1, "time": time_label, "skipped": True,
                    "asr_text": r.text, "asr_time": round(r.asr_time_s, 3),
                    "call_verdict": session.current_verdict,
                })
                continue
        else:
            session.note_chunk_received()
            if not session.should_infer():
                console.print(
                    f" {i+1:>3}  {time_label:>11}  [dim]{'---':>10}  "
                    f"{v_fmt(session.current_verdict):>20}              "
                    f"[dim]skip[/][/]"
                )
                log_entries.append({
                    "chunk": i + 1, "time": time_label, "skipped": True,
                    "call_verdict": session.current_verdict,
                })
                continue

        result = run_turn(
            engine, session,
            wav_bytes=chunk_bytes,
            pipeline=args.pipeline,
            system_prompt=system_prompt,
            detect_bank_fn=server._detect_bank,
            bank_context_fn=server._bank_context,
            lenient_parse_fn=server._lenient_parse,
            asr_transcribe_fn=asr.transcribe_wav if args.pipeline == "a" else None,
            silence_ctx=silence_fds,
        )
        infer_count += 1

        seg_v = result.get("verdict", "UNCERTAIN")
        call_v = session.current_verdict
        asr_time = result.get("asr_time", 0)
        llm_time = result.get("llm_time", 0)
        transcription = result.get("transcription", "")
        summary = result.get("summary", "")
        recommendations = result.get("recommendations", [])
        detected_bank = result.get("detected_bank")
        changed = call_v != prev_call_verdict

        # ── Compact terminal line ───────────────────────────────────────
        # Truncate transcript to fit one line
        snip = transcription[:50] + "..." if len(transcription) > 50 else transcription
        change_marker = " [bold red]<![/]" if changed else ""

        console.print(
            f" {i+1:>3}  {time_label:>11}  {v_fmt(seg_v):>20}  "
            f"{v_fmt(call_v):>20}{change_marker}  "
            f"{asr_time:>5.1f}  {llm_time:>5.1f}  [dim]{snip}[/]"
        )

        # Extra line only on verdict flip
        if changed:
            console.print(
                f"      [bold red]>> {prev_call_verdict} -> {call_v}: "
                f"{summary[:80]}[/]"
            )
            prev_call_verdict = call_v

        # ── Full details to log file ────────────────────────────────────
        entry = {
            "chunk": i + 1,
            "time": time_label,
            "segment_verdict": seg_v,
            "call_verdict": call_v,
            "verdict_changed": changed,
            "asr_time": round(asr_time, 3),
            "llm_time": round(llm_time, 3),
            "transcription": transcription,
            "summary": summary,
            "recommendations": recommendations,
            "detected_bank": detected_bank,
            "running_transcript": session.running_transcript,
            "scam_evidence": list(session.scam_evidence),
            "legit_evidence": list(session.legit_evidence),
        }
        log_entries.append(entry)
        with log_path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_time_start

    console.print("[dim]" + "─" * 90 + "[/]")
    console.print()
    inferred = infer_count or total_chunks
    console.print(
        f"  [bold]Final:[/] {v_fmt(session.current_verdict)}  "
        f"[dim]|[/]  {total_chunks} chunks ({infer_count} inferred)  "
        f"[dim]|[/]  {total_elapsed:.1f}s total  "
        f"[dim]|[/]  {total_elapsed/inferred:.1f}s/infer"
    )

    # Verdict flips summary
    flips = [e for e in log_entries if e.get("verdict_changed")]
    if flips:
        flip_str = " -> ".join(
            [log_entries[0]["call_verdict"] if not flips else "UNCERTAIN"]
            + [f["call_verdict"] for f in flips]
        )
        console.print(f"  [bold]Flips:[/]  {flip_str}")
        for f in flips:
            console.print(f"    [dim]Chunk {f['chunk']} [{f['time']}]:[/] {f['summary'][:80]}")

    console.print(f"\n  [dim]Full log:[/] {log_path}")

    # Write summary as final log entry
    with log_path.open("a") as f:
        f.write(json.dumps({
            "_summary": True,
            "file": str(wav_path),
            "total_chunks": total_chunks,
            "total_time": round(total_elapsed, 1),
            "final_verdict": session.current_verdict,
            "verdict_flips": len(flips),
            "full_transcript": session.running_transcript,
            "scam_evidence": list(session.scam_evidence),
            "legit_evidence": list(session.legit_evidence),
        }, ensure_ascii=False) + "\n")

    # ── Post-call analysis (Pass 2) ────────────────────────────────────
    console.print()
    console.rule("[bold]Post-Call Analysis[/]")
    console.print("[dim]Running detailed analysis on full transcript...[/]")

    report = run_post_call(engine, session, silence_ctx=silence_fds)

    if report.get("error"):
        console.print(f"  [red]Error:[/] {report['error']}")
    else:
        elapsed = report.get("elapsed", 0)
        console.print(f"  [dim]Analysis completed in {elapsed:.1f}s[/]")
        console.print()

        # Display key fields
        for key in ["final_verdict", "confidence", "risk_score", "call_summary"]:
            val = report.get(key)
            if val is not None:
                if key == "final_verdict":
                    console.print(f"  [bold]Verdict:[/]    {v_fmt(str(val))}")
                elif key == "confidence":
                    console.print(f"  [bold]Confidence:[/] {val}")
                elif key == "risk_score":
                    color = "red" if int(val) > 70 else "yellow" if int(val) > 30 else "green"
                    console.print(f"  [bold]Risk Score:[/] [{color}]{val}/100[/]")
                else:
                    console.print(f"  [bold]Summary:[/]    {val}")

        red_flags = report.get("red_flags", [])
        if red_flags:
            console.print()
            console.print("  [bold red]Red Flags:[/]")
            for flag in red_flags:
                console.print(f"    [red]->[/] {flag}")

        actions = report.get("recommended_actions", [])
        if actions:
            console.print()
            console.print("  [bold]Recommended Actions:[/]")
            for action in actions:
                console.print(f"    -> {action}")

        timeline = report.get("timeline", [])
        if timeline:
            console.print()
            console.print("  [bold]Timeline:[/]")
            for event in timeline:
                t = event.get("time", "")
                e = event.get("event", "")
                console.print(f"    [{t}] {e}")

        auth_report = report.get("report_for_authorities", "")
        if auth_report:
            console.print()
            console.print(Panel(
                auth_report, title="Report for Authorities",
                border_style="red", expand=False, width=80,
            ))

    # Save post-call report to log
    with log_path.open("a") as f:
        f.write(json.dumps({"_post_call_report": True, **report}, ensure_ascii=False) + "\n")

    console.print(f"\n  [dim]Full log:[/] {log_path}")

    unload_engine(engine)


if __name__ == "__main__":
    main()

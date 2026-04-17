"""CallSession — transcript accumulation and progressive verdict for streaming.

Used by both the WebSocket server (live calls) and the eval harness (streaming
simulation). Encapsulates all per-call state that was previously scattered
across local variables in ``server.websocket_endpoint``.
"""

from __future__ import annotations

import base64
import json
import time
import traceback
import sys
from dataclasses import dataclass, field


# ── Token-budget constants ──────────────────────────────────────────────────
# E2B has a 4096 token context.  Rough layout:
#   system prompt  ~450 tok
#   bank KB        ~120 tok
#   instruction    ~ 40 tok
#   output/tool    ~200 tok
#   audio blob (pipeline B only) ~800-1200 tok
#   ASR text  (pipeline A only)  ~ 30-80 tok
#
# Conservative char→token ratio: 1 token ≈ 4 chars (mixed EN/MS).

_MAX_HISTORY_CHARS_A = 8000   # pipeline A — text-only input, more room
_MAX_HISTORY_CHARS_B = 4800   # pipeline B — audio blob eats context

# When transcript exceeds this, compress the old portion into a summary.
# Set to ~60% of max so there's room for the summary + recent text.
_COMPRESS_THRESHOLD_A = 5000
_COMPRESS_THRESHOLD_B = 3000

# Reserve this many chars for the compressed summary prefix.
_SUMMARY_BUDGET = 500

SUMMARIZE_PROMPT = (
    "Summarize this phone call transcript in under 100 words. "
    "Keep: who called, what bank, what they asked for, any suspicious requests, "
    "key numbers (amounts, account numbers, card digits). "
    "Do NOT add analysis or verdict — just the facts."
)


@dataclass
class CallSession:
    """Per-call state for transcript accumulation and progressive verdict."""

    mode: str = "live"                       # "live" or "eval"

    # Transcript
    running_transcript: str = ""             # accumulated clean text
    _prev_tail: str = ""                     # last segment tail for dedup
    _compressed_summary: str = ""            # LLM summary of older transcript
    _chars_at_last_compress: int = 0         # transcript length when last compressed

    # Verdict
    current_verdict: str = "UNCERTAIN"
    verdict_history: list[tuple] = field(default_factory=list)
    scam_evidence: list[str] = field(default_factory=list)
    legit_evidence: list[str] = field(default_factory=list)

    # Bank
    detected_bank: str | None = None

    # Bookkeeping
    turn_count: int = 0
    _last_inference_ts: float = 0.0
    _chars_since_inference: int = 0

    # Chunk skipping — accumulate transcript for N chunks, infer every Nth.
    # Set skip_chunks > 1 to reduce inference calls on slow devices.
    skip_chunks: int = 1               # 1 = no skipping (default)
    _chunks_since_inference: int = 0

    # ── Transcript management ───────────────────────────────────────────

    def add_transcript(self, text: str) -> None:
        """Append *text* to running transcript, deduplicating any overlap."""
        if not text:
            return

        clean = text.strip()
        if not clean:
            return

        # Overlap deduplication: find longest common overlap between the tail
        # of the previous segment and the head of the new segment.
        deduped = self._dedup_overlap(clean)

        if self.running_transcript:
            self.running_transcript += " " + deduped
        else:
            self.running_transcript = deduped

        # Store tail for next dedup pass (last 120 chars is plenty for 0.5s
        # of speech at ~3 words/s ≈ 15-20 chars of text).
        self._prev_tail = clean[-120:]
        self._chars_since_inference += len(deduped)

    def _dedup_overlap(self, new_text: str) -> str:
        """Remove leading words of *new_text* that duplicate the previous tail."""
        if not self._prev_tail:
            return new_text

        tail = self._prev_tail.lower()
        head = new_text.lower()

        # Try progressively shorter suffixes of tail against head.
        best = 0
        for length in range(min(len(tail), len(head)), 4, -1):
            suffix = tail[-length:]
            if head.startswith(suffix):
                best = length
                break

        if best > 0:
            return new_text[best:].lstrip()
        return new_text

    # ── Context building ────────────────────────────────────────────────

    def needs_compression(self, pipeline: str) -> bool:
        """Whether the transcript is long enough to benefit from compression."""
        threshold = _COMPRESS_THRESHOLD_A if pipeline == "a" else _COMPRESS_THRESHOLD_B
        # Only compress if we've grown significantly since last compression.
        new_chars = len(self.running_transcript) - self._chars_at_last_compress
        return new_chars > threshold

    def compress_transcript(self, engine, *, silence_ctx=None) -> None:
        """Summarize the older portion of the transcript using the LLM.

        Keeps the recent text intact and replaces the older portion with a
        short summary. Called between inference turns when the transcript
        gets too long.
        """
        # Keep the most recent ~3000 chars verbatim, summarize everything before.
        keep_recent = 3000
        if len(self.running_transcript) <= keep_recent:
            return

        old_text = self.running_transcript[:-keep_recent]
        recent_text = self.running_transcript[-keep_recent:]

        # If there's an existing summary, include it for continuity.
        to_summarize = ""
        if self._compressed_summary:
            to_summarize = f"Previous summary: {self._compressed_summary}\n\nNew transcript to incorporate:\n"
        to_summarize += old_text

        try:
            ctx = silence_ctx() if silence_ctx else _noop_ctx()
            with ctx:
                conv = engine.create_conversation(
                    messages=[{"role": "system", "content": SUMMARIZE_PROMPT}],
                    tools=[],
                )
                conv.__enter__()
                try:
                    response = conv.send_message(
                        {"role": "user", "content": [{"type": "text", "text": to_summarize}]}
                    )
                finally:
                    conv.__exit__(None, None, None)

            raw = (response or {}).get("content", [{}])[0].get("text", "")
            if raw.strip():
                self._compressed_summary = raw.strip()[:_SUMMARY_BUDGET]
                # Trim the transcript to only the recent portion.
                self.running_transcript = recent_text
                self._chars_at_last_compress = len(self.running_transcript)
                print(f"  [compress] summary={len(self._compressed_summary)} chars, "
                      f"transcript trimmed to {len(self.running_transcript)} chars")
        except Exception as exc:  # noqa: BLE001
            # Compression is best-effort — don't break inference if it fails.
            print(f"  [compress] failed: {exc}")

    def build_context(self, pipeline: str) -> str:
        """Return the history text block to inject into the model prompt.

        In eval mode (single-file), returns a minimal preamble.
        In live mode, returns the compressed summary (if any) plus the
        recent transcript window.
        """
        if self.mode == "eval":
            return "This is the first input."

        if not self.running_transcript:
            return "This is the first input."

        max_chars = _MAX_HISTORY_CHARS_A if pipeline == "a" else _MAX_HISTORY_CHARS_B

        # Build context: summary prefix + recent transcript.
        parts = []
        if self._compressed_summary:
            parts.append(f"[Earlier in the call: {self._compressed_summary}]")

        transcript = self.running_transcript
        # Account for summary length in the budget.
        available = max_chars - len(self._compressed_summary) - 50
        if len(transcript) > available:
            transcript = "..." + transcript[-available:]

        parts.append("Conversation transcript so far:\n" + transcript)

        # Include the previous verdict for continuity.
        if self.verdict_history:
            last_turn, last_v, last_s = self.verdict_history[-1]
            verdict_line = f"\n\nPrevious analysis: {last_v}"
            if last_s:
                verdict_line += f" — {last_s}"
            parts.append(verdict_line)

        parts.append("\nNow analyze the NEW segment below, considering the full conversation context.")

        return "\n".join(parts)

    # ── Progressive verdict ─────────────────────────────────────────────

    def update_verdict(
        self,
        segment_verdict: str,
        summary: str,
        recommendations: list[str] | None = None,
    ) -> None:
        """Update the call-level progressive verdict.

        Rules:
          - SCAM is sticky — once set, never reverts.
          - LEGITIMATE can be overridden to SCAM.
          - UNCERTAIN is the starting state and yields to anything.
        """
        self.turn_count += 1
        self.verdict_history.append((self.turn_count, segment_verdict, summary))

        # Accumulate evidence from recommendations.
        if recommendations:
            for rec in recommendations:
                lower = rec.lower()
                if any(w in lower for w in ("hang up", "do not share", "report",
                                            "do not transfer", "scam", "fraud")):
                    if rec not in self.scam_evidence:
                        self.scam_evidence.append(rec)
                elif any(w in lower for w in ("legitimate", "routine", "no action")):
                    if rec not in self.legit_evidence:
                        self.legit_evidence.append(rec)

        # Sticky SCAM — once flagged, never flip back.
        if self.current_verdict == "SCAM":
            return

        if segment_verdict == "SCAM":
            self.current_verdict = "SCAM"
        elif segment_verdict == "LEGITIMATE" and self.current_verdict != "SCAM":
            self.current_verdict = "LEGITIMATE"
        # UNCERTAIN does not override an existing LEGITIMATE verdict.

    # ── Rate limiting ───────────────────────────────────────────────────

    def should_infer(self) -> bool:
        """Whether enough new content has arrived to justify an inference call.

        In eval mode always returns True.  In live mode, gates on either
        sufficient new text (~2s of speech ≈ 40 chars) or a 5s hard timeout.
        When skip_chunks > 1, also requires that many chunks have arrived.
        """
        if self.mode == "eval":
            return True

        # Chunk skipping — wait for N chunks before inferring.
        if self.skip_chunks > 1 and self._chunks_since_inference < self.skip_chunks:
            return False

        now = time.time()
        elapsed = now - self._last_inference_ts if self._last_inference_ts else 999

        if self._chars_since_inference >= 40 or elapsed >= 5.0:
            return True
        return False

    def note_chunk_received(self) -> None:
        """Record that a new audio chunk arrived (for skip counting)."""
        self._chunks_since_inference += 1

    def mark_inference_done(self) -> None:
        """Record that an inference call just completed."""
        self._last_inference_ts = time.time()
        self._chars_since_inference = 0
        self._chunks_since_inference = 0

    # ── Bank detection (sticky) ─────────────────────────────────────────

    def detect_bank(self, text: str, detect_fn) -> str | None:
        """Detect bank from *text* using *detect_fn*; sticky once found."""
        if self.detected_bank:
            return self.detected_bank
        bank = detect_fn(text)
        if bank:
            self.detected_bank = bank
        return self.detected_bank

    # ── Post-call report ──────────────────────────────────────────────

    def post_call_summary(self) -> dict:
        """Return a structured summary of the call for post-call analysis."""
        return {
            "final_verdict": self.current_verdict,
            "turn_count": self.turn_count,
            "transcript": self.running_transcript,
            "verdict_timeline": [
                {"turn": t, "verdict": v, "summary": s}
                for t, v, s in self.verdict_history
            ],
            "scam_evidence": list(self.scam_evidence),
            "legit_evidence": list(self.legit_evidence),
            "detected_bank": self.detected_bank,
        }

    # ── Reset ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all state (used between eval files)."""
        self.running_transcript = ""
        self._prev_tail = ""
        self._compressed_summary = ""
        self._chars_at_last_compress = 0
        self.current_verdict = "UNCERTAIN"
        self.verdict_history.clear()
        self.scam_evidence.clear()
        self.legit_evidence.clear()
        self.detected_bank = None
        self.turn_count = 0
        self._last_inference_ts = 0.0
        self._chars_since_inference = 0
        self._chunks_since_inference = 0


# ── Shared inference logic ──────────────────────────────────────────────────
# Used by both the server WebSocket handler and the eval harness, so the
# prompt construction / tool-call parsing / verdict extraction is in one place.


import re as _re

_FIELD_RE = _re.compile(
    r'(verdict|summary|transcription|description|recommendations)\s*:\s*'
    r'(?:<\|"\|>(.+?)<\|"\|>|"([^"]+?)"|(\w+))',
    _re.DOTALL,
)


def _parse_tool_from_error(err_str: str) -> dict:
    """Extract tool call fields from a litert_lm RuntimeError message.

    The error embeds the raw model output like:
      call:analyze_speech{verdict:SCAM,summary:<|"|>...<|"|>,...}
    We regex-extract each field.
    """
    out: dict = {"verdict": "UNCERTAIN", "summary": "", "transcription": "",
                 "recommendations": []}

    for m in _FIELD_RE.finditer(err_str):
        field = m.group(1)
        value = (m.group(2) or m.group(3) or m.group(4) or "").replace('<|"|>', "").strip()

        if field == "verdict":
            v = value.upper()
            if v in ("SCAM", "LEGITIMATE", "UNCERTAIN", "SUSPICIOUS"):
                out["verdict"] = v
        elif field == "summary":
            out["summary"] = value
        elif field == "transcription":
            out["transcription"] = value
        elif field == "description":
            out["transcription"] = value  # reuse same field
        elif field == "recommendations":
            # Split numbered/bulleted lines
            recs = [r.strip().lstrip("0123456789.-) ") for r in value.split("\n") if r.strip()]
            if not recs and value:
                recs = [value]
            out["recommendations"] = recs

    return out


def run_turn(
    engine,
    session: CallSession,
    *,
    wav_bytes: bytes | None = None,
    audio_b64: str | None = None,
    image_b64: str | None = None,
    pipeline: str = "b",
    system_prompt: str,
    detect_bank_fn,
    bank_context_fn,
    lenient_parse_fn,
    log_parse_failure_fn=None,
    asr_transcribe_fn=None,
    silence_ctx=None,
) -> dict:
    """Execute one inference turn within a *session*.

    Returns a dict with keys: verdict, segment_verdict, call_verdict,
    summary, transcription, description, recommendations, asr_time,
    llm_time, used_tool, error (if any).
    """
    has_audio = bool(wav_bytes) or bool(audio_b64)
    has_image = bool(image_b64)

    if not has_audio and not has_image:
        return {"error": "no input", "verdict": "UNCERTAIN"}

    # Image-only always pipeline B.
    if not has_audio:
        pipeline = "b"

    # ── ASR (pipeline A) ────────────────────────────────────────────────
    asr_text = ""
    asr_time = 0.0
    if pipeline == "a" and has_audio:
        if wav_bytes is None and audio_b64:
            wav_bytes = base64.b64decode(audio_b64)
        if asr_transcribe_fn and wav_bytes:
            r = asr_transcribe_fn(wav_bytes)
            asr_text = r.text
            asr_time = r.asr_time_s

    # ── Update session transcript (live mode) ───────────────────────────
    if session.mode == "live" and asr_text:
        session.add_transcript(asr_text)

    # ── Build context ───────────────────────────────────────────────────
    history_block = session.build_context(pipeline)

    # Bank detection
    detection_text = session.running_transcript + " " + asr_text
    detected_bank = session.detect_bank(detection_text, detect_bank_fn)
    bank_block = ""
    if detected_bank:
        bank_block = "\n\n" + bank_context_fn(detected_bank)

    content: list[dict] = [{"type": "text", "text": history_block + bank_block}]

    if pipeline == "a" and has_audio:
        content.append({
            "type": "text",
            "text": f'Caller transcript (from Whisper ASR):\n"{asr_text}"',
        })
    elif has_audio:
        blob = audio_b64 or base64.b64encode(wav_bytes).decode()
        content.append({"type": "audio", "blob": blob})
    if has_image:
        content.append({"type": "image", "blob": image_b64})

    # Instruction
    if has_audio and has_image:
        content.append({"type": "text", "text": "The user sent audio and an image. Transcribe the audio and analyze both for scam indicators."})
    elif pipeline == "a" and has_audio:
        content.append({"type": "text", "text": "Analyze the transcript above for scam indicators. Use analyze_speech and put the transcript verbatim in the transcription field."})
    elif has_audio:
        content.append({"type": "text", "text": "Transcribe and analyze for scam indicators."})
    else:
        content.append({"type": "text", "text": "Analyze this document/image for scam indicators."})

    # ── Tool callbacks ──────────────────────────────────────────────────
    tool_result: dict = {}

    def analyze_speech(transcription: str, verdict: str, summary: str, recommendations: str) -> str:
        """Report your analysis of this audio segment.

        Args:
            transcription: Exact transcription of what the caller said in the audio.
            verdict: SCAM, LEGITIMATE, or UNCERTAIN.
            summary: 1-2 sentence summary of what is happening in this call.
            recommendations: 2-5 practical recommendations separated by newlines.
        """
        tool_result["type"] = "audio"
        tool_result["transcription"] = transcription
        tool_result["verdict"] = verdict
        tool_result["summary"] = summary
        tool_result["recommendations"] = recommendations
        return "OK"

    def analyze_document(description: str, verdict: str, summary: str, recommendations: str) -> str:
        """Report your analysis of an image (document, screenshot, or message).

        Args:
            description: Describe what the document contains and key details you see.
            verdict: SCAM, LEGITIMATE, or UNCERTAIN.
            summary: 1-2 sentence summary of the document and its purpose.
            recommendations: 2-5 practical recommendations separated by newlines.
        """
        tool_result["type"] = "document"
        tool_result["description"] = description
        tool_result["verdict"] = verdict
        tool_result["summary"] = summary
        tool_result["recommendations"] = recommendations
        return "OK"

    # ── Inference ───────────────────────────────────────────────────────
    t0 = time.time()
    response = None
    inference_error: str | None = None

    tool_parse_fallback: str | None = None
    try:
        ctx = silence_ctx() if silence_ctx else _noop_ctx()
        with ctx:
            conv = engine.create_conversation(
                messages=[{"role": "system", "content": system_prompt}],
                tools=[analyze_speech, analyze_document],
            )
            conv.__enter__()
            try:
                response = conv.send_message({"role": "user", "content": content})
            finally:
                conv.__exit__(None, None, None)
    except Exception as exc:  # noqa: BLE001
        err_str = str(exc)
        # litert_lm throws RuntimeError when it can't parse the tool call
        # format, but the raw model output is embedded in the error message.
        # Extract it and fall back to lenient parsing.
        if "Failed to parse tool calls" in err_str:
            tool_parse_fallback = err_str
            print(f"  [tool-parse fallback] recovering from: {err_str[:120]}...")
        else:
            inference_error = f"{type(exc).__name__}: {exc}"
            print(f"!! inference failed: {inference_error}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    llm_time = time.time() - t0

    if inference_error is not None:
        return {
            "verdict": "ERROR",
            "segment_verdict": "ERROR",
            "call_verdict": session.current_verdict,
            "summary": inference_error,
            "transcription": asr_text,
            "description": "",
            "recommendations": [],
            "asr_time": asr_time,
            "llm_time": llm_time,
            "used_tool": False,
            "error": inference_error,
        }

    # ── Parse results ───────────────────────────────────────────────────
    strip = lambda s: (s or "").replace('<|"|>', "").strip()

    if tool_result:
        result_type = tool_result.get("type", "audio")
        transcription = strip(tool_result.get("transcription", "")) if result_type == "audio" else ""
        description = strip(tool_result.get("description", "")) if result_type == "document" else ""
        verdict = strip(tool_result.get("verdict", "UNCERTAIN")).upper() or "UNCERTAIN"
        summary = strip(tool_result.get("summary", ""))
        raw_recs = strip(tool_result.get("recommendations", ""))
        recommendations = [r.strip().lstrip("0123456789.-) ") for r in raw_recs.split("\n") if r.strip()]
        used_tool = True
    elif tool_parse_fallback:
        # Engine threw RuntimeError because it couldn't parse the tool call
        # format, but the raw output is in the error string. Extract fields
        # using the same lenient parser the server uses for malformed JSON.
        result_type = "audio" if has_audio else "document"
        parsed = _parse_tool_from_error(tool_parse_fallback)
        verdict = parsed.get("verdict", "UNCERTAIN")
        summary = parsed.get("summary", "Recovered from tool parse error.")
        recommendations = parsed.get("recommendations", [])
        transcription = parsed.get("transcription", "") if has_audio else ""
        description = parsed.get("transcription", "") if not has_audio else ""
        used_tool = False
    else:
        raw_text = (response or {}).get("content", [{}])[0].get("text", "")
        result_type = "audio" if has_audio else "document"
        parsed = lenient_parse_fn(raw_text)
        verdict = parsed["verdict"]
        summary = parsed["summary"] or "Recovered from raw text (no tool call)."
        recommendations = parsed["recommendations"]
        transcription = parsed["transcription"] if has_audio else ""
        description = parsed["transcription"] if not has_audio else ""
        if log_parse_failure_fn:
            log_parse_failure_fn(raw_text, parsed)
        used_tool = False

    # Pipeline A: prefer Whisper transcript as ground truth.
    if pipeline == "a" and asr_text:
        transcription = asr_text

    # In live mode, if we didn't get ASR text earlier (pipeline B), add
    # the model's transcription to the running transcript now.
    label = transcription or description
    if session.mode == "live" and pipeline == "b" and transcription:
        session.add_transcript(transcription)

    # Update progressive verdict.
    session.update_verdict(verdict, summary, recommendations)
    session.mark_inference_done()

    # Compress old transcript if it's getting too long.
    if session.mode == "live" and session.needs_compression(pipeline):
        session.compress_transcript(engine, silence_ctx=silence_ctx)

    # Post-inference bank detection from output.
    if not detected_bank and label:
        detected_bank = session.detect_bank(label, detect_bank_fn)

    return {
        "verdict": verdict,
        "segment_verdict": verdict,
        "call_verdict": session.current_verdict,
        "input_type": result_type,
        "transcription": transcription,
        "description": description,
        "summary": summary,
        "recommendations": recommendations,
        "asr_time": asr_time,
        "llm_time": llm_time,
        "used_tool": used_tool,
        "detected_bank": detected_bank,
    }


POST_CALL_PROMPT = """\
You are a scam call analyst. Given the full transcript and real-time analysis \
from a phone call, produce a detailed post-call report.

Respond with a JSON object containing these fields:
- final_verdict: "SCAM", "LEGITIMATE", or "UNCERTAIN"
- confidence: "HIGH", "MEDIUM", or "LOW"
- risk_score: integer 0-100 (100 = definitely scam)
- call_summary: 2-3 sentence summary of what happened in the call
- evidence: list of specific quotes/behaviors that support the verdict
- red_flags: list of scam indicators found (empty list [] if legitimate)
- legitimate_indicators: list of legitimate behaviors found (empty list [] if scam)
- timeline: list of objects with "time" and "event" describing key moments
- recommended_actions: list of specific next steps for the user. \
  For LEGITIMATE calls: simple confirmation steps only (e.g. "No action needed"). \
  Do NOT recommend reporting legitimate calls to authorities. \
  For SCAM calls: recommend hanging up, not sharing info, reporting to police/bank.
- report_for_authorities: ONLY include this field if verdict is SCAM. \
  Write a paragraph suitable for filing a police report. \
  If the call is LEGITIMATE, set this to null or omit it entirely.
"""


def run_post_call(
    engine,
    session: CallSession,
    *,
    system_prompt: str = POST_CALL_PROMPT,
    silence_ctx=None,
) -> dict:
    """Run post-call analysis on the full transcript.

    Uses the LLM to generate a detailed report from the accumulated
    session data. This is Pass 2 — called once after the call ends.
    """
    summary = session.post_call_summary()

    # Build the user message with full context.
    verdict_timeline_text = ""
    for entry in summary["verdict_timeline"]:
        verdict_timeline_text += (
            f"  Turn {entry['turn']}: {entry['verdict']}"
            f" — {entry['summary']}\n"
        )

    user_content = (
        f"== CALL TRANSCRIPT ==\n"
        f"{summary['transcript']}\n\n"
        f"== REAL-TIME ANALYSIS ==\n"
        f"Final verdict: {summary['final_verdict']}\n"
        f"Bank detected: {summary['detected_bank'] or 'none'}\n"
        f"Total turns analyzed: {summary['turn_count']}\n\n"
        f"Verdict timeline:\n{verdict_timeline_text}\n"
        f"Scam evidence found:\n"
        + "\n".join(f"  - {e}" for e in summary["scam_evidence"])
        + "\n\nLegitimate indicators found:\n"
        + "\n".join(f"  - {e}" for e in summary["legit_evidence"])
        + "\n\nProduce the detailed post-call report as JSON."
    )

    t0 = time.time()
    response = None
    error: str | None = None

    try:
        ctx = silence_ctx() if silence_ctx else _noop_ctx()
        with ctx:
            conv = engine.create_conversation(
                messages=[{"role": "system", "content": system_prompt}],
                tools=[],
            )
            conv.__enter__()
            try:
                response = conv.send_message(
                    {"role": "user", "content": [{"type": "text", "text": user_content}]}
                )
            finally:
                conv.__exit__(None, None, None)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.time() - t0

    if error:
        return {"error": error, "elapsed": elapsed}

    # Try to parse JSON from response.
    raw_text = (response or {}).get("content", [{}])[0].get("text", "")

    import re
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if json_match:
        try:
            report = json.loads(json_match.group(0))
            report["elapsed"] = round(elapsed, 2)
            report["raw"] = raw_text
            return report
        except json.JSONDecodeError:
            pass

    # Fallback: return raw text.
    return {
        "raw": raw_text,
        "elapsed": round(elapsed, 2),
        "parse_error": "Could not extract JSON from model response",
    }


class _noop_ctx:
    """No-op context manager."""
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass

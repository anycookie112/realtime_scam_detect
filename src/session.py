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

    # Risk assessment
    current_risk_level: str = "SAFE"         # SAFE, LOW_RISK, MEDIUM_RISK, HIGH_RISK
    current_risk_score: int = 0              # 0-100, running max
    risk_history: list[tuple] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    safe_indicators: list[str] = field(default_factory=list)

    # Legacy aliases for backward compat (eval harness, stream monitor)
    @property
    def current_verdict(self) -> str:
        return self.current_risk_level
    @property
    def verdict_history(self) -> list[tuple]:
        return self.risk_history
    @property
    def scam_evidence(self) -> list[str]:
        return self.risk_factors
    @property
    def legit_evidence(self) -> list[str]:
        return self.safe_indicators

    # Call notepad — tracks what we've learned about the caller
    caller_identity: str = ""                # who they claim to be
    caller_org: str = ""                     # organization claimed
    info_requested: list[str] = field(default_factory=list)  # what they've asked for
    call_reason: str = ""                    # stated reason for the call

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

        # Include the previous risk level for continuity.
        if self.risk_history:
            last_turn, last_v, last_s = self.risk_history[-1]
            verdict_line = f"\n\nPrevious assessment: {last_v}"
            if last_s:
                verdict_line += f" — {last_s}"
            parts.append(verdict_line)

        # Include call notepad so model knows what's been collected.
        notepad_lines = []
        if self.caller_identity:
            notepad_lines.append(f"Caller claims: {self.caller_identity}")
        if self.call_reason:
            notepad_lines.append(f"Reason: {self.call_reason}")
        if self.info_requested:
            notepad_lines.append(f"Info requested so far: {', '.join(self.info_requested)}")
        if self.risk_factors:
            notepad_lines.append(f"Risk factors noted: {', '.join(self.risk_factors[:5])}")
        if notepad_lines:
            parts.append("\nCall Notes:\n" + "\n".join(f"- {l}" for l in notepad_lines))

        parts.append("\nNow analyze the NEW segment below, considering the full conversation context.")

        return "\n".join(parts)

    # ── Progressive risk assessment ──────────────────────────────────────

    _RISK_ORDER = {"SAFE": 0, "LOW_RISK": 1, "MEDIUM_RISK": 2, "HIGH_RISK": 3}

    def update_verdict(
        self,
        segment_verdict: str,
        summary: str,
        recommendations: list[str] | None = None,
        *,
        risk_score: int = 0,
        info_requested: str = "",
        caller_claims: str = "",
    ) -> None:
        """Update the call-level progressive risk assessment.

        Rules:
          - HIGH_RISK is sticky — once set, never reverts.
          - Risk level can only escalate, never downgrade.
          - Risk score tracks the running maximum.
        """
        self.turn_count += 1
        self.risk_history.append((self.turn_count, segment_verdict, summary))

        # Track risk score as running max.
        if risk_score > 0:
            self.current_risk_score = max(self.current_risk_score, risk_score)

        # Accumulate evidence from recommendations.
        if recommendations:
            for rec in recommendations:
                lower = rec.lower()
                if any(w in lower for w in ("stop sharing", "hang up", "do not share",
                                            "do not transfer", "do not provide",
                                            "red flag", "high risk")):
                    if rec not in self.risk_factors:
                        self.risk_factors.append(rec)
                elif any(w in lower for w in ("safe", "routine", "no action",
                                              "no concerns", "legitimate")):
                    if rec not in self.safe_indicators:
                        self.safe_indicators.append(rec)

        # Update notepad from model output.
        if caller_claims and not self.caller_identity:
            self.caller_identity = caller_claims
            # Try to extract org
            for w in ("from", "at", "of"):
                if w in caller_claims.lower():
                    self.caller_org = caller_claims
                    break
        if info_requested:
            for item in info_requested.split(","):
                item = item.strip()
                if item and item not in self.info_requested:
                    self.info_requested.append(item)
        if summary and not self.call_reason:
            self.call_reason = summary

        # Risk level can only escalate, never downgrade.
        # HIGH_RISK is sticky.
        if self.current_risk_level == "HIGH_RISK":
            return

        seg_order = self._RISK_ORDER.get(segment_verdict, 0)
        cur_order = self._RISK_ORDER.get(self.current_risk_level, 0)
        if seg_order > cur_order:
            self.current_risk_level = segment_verdict

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
            "final_risk_level": self.current_risk_level,
            "final_risk_score": self.current_risk_score,
            "turn_count": self.turn_count,
            "transcript": self.running_transcript,
            "risk_timeline": [
                {"turn": t, "risk_level": v, "summary": s}
                for t, v, s in self.risk_history
            ],
            "risk_factors": list(self.risk_factors),
            "safe_indicators": list(self.safe_indicators),
            "detected_bank": self.detected_bank,
            "notepad": {
                "caller_identity": self.caller_identity,
                "caller_org": self.caller_org,
                "info_requested": list(self.info_requested),
                "call_reason": self.call_reason,
            },
        }

    # ── Reset ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all state (used between eval files)."""
        self.running_transcript = ""
        self._prev_tail = ""
        self._compressed_summary = ""
        self._chars_at_last_compress = 0
        self.current_risk_level = "SAFE"
        self.current_risk_score = 0
        self.risk_history.clear()
        self.risk_factors.clear()
        self.safe_indicators.clear()
        self.caller_identity = ""
        self.caller_org = ""
        self.info_requested.clear()
        self.call_reason = ""
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
    r'(verdict|risk_level|risk_score|summary|transcription|description|recommendations|info_requested|caller_claims)\s*:\s*'
    r'(?:<\|"\|>(.+?)<\|"\|>|"([^"]+?)"|(\w+))',
    _re.DOTALL,
)

_VALID_RISK_LEVELS = {"SAFE", "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK",
                      "SCAM", "LEGITIMATE", "UNCERTAIN", "SUSPICIOUS"}


def _parse_tool_from_error(err_str: str) -> dict:
    """Extract tool call fields from a litert_lm RuntimeError message.

    The error embeds the raw model output like:
      call:analyze_speech{risk_level:HIGH_RISK,summary:<|"|>...<|"|>,...}
    We regex-extract each field.
    """
    out: dict = {"verdict": "MEDIUM_RISK", "summary": "", "transcription": "",
                 "recommendations": [], "info_requested": "", "caller_claims": ""}

    for m in _FIELD_RE.finditer(err_str):
        field = m.group(1)
        value = (m.group(2) or m.group(3) or m.group(4) or "").replace('<|"|>', "").strip()

        if field in ("verdict", "risk_level"):
            v = value.upper()
            if v in _VALID_RISK_LEVELS:
                out["verdict"] = v
        elif field == "summary":
            out["summary"] = value
        elif field == "transcription":
            out["transcription"] = value
        elif field == "description":
            out["transcription"] = value  # reuse same field
        elif field == "recommendations":
            recs = [r.strip().lstrip("0123456789.-) ") for r in value.split("\n") if r.strip()]
            if not recs and value:
                recs = [value]
            out["recommendations"] = recs
        elif field == "risk_score":
            try:
                out["risk_score"] = int(value)
            except ValueError:
                pass
        elif field == "info_requested":
            out["info_requested"] = value
        elif field == "caller_claims":
            out["caller_claims"] = value

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

    def analyze_speech(
        transcription: str, risk_level: str, risk_score: int,
        summary: str, recommendations: str,
        info_requested: str, caller_claims: str,
    ) -> str:
        """Report your analysis of this audio segment.

        Args:
            transcription: Exact transcription of what the caller said.
            risk_level: SAFE, LOW_RISK, MEDIUM_RISK, or HIGH_RISK.
            risk_score: 0-100 integer risk score (100 = definite scam).
            summary: 1-2 sentence summary of what is happening.
            recommendations: What the user should say or ask to verify the caller. Separated by newlines.
            info_requested: What information the caller has asked for so far (comma separated).
            caller_claims: Who the caller says they are (name, org, staff ID).
        """
        tool_result["type"] = "audio"
        tool_result["transcription"] = transcription
        tool_result["risk_level"] = risk_level
        tool_result["risk_score"] = risk_score
        tool_result["summary"] = summary
        tool_result["recommendations"] = recommendations
        tool_result["info_requested"] = info_requested
        tool_result["caller_claims"] = caller_claims
        return "OK"

    def analyze_document(
        description: str, risk_level: str, risk_score: int,
        summary: str, recommendations: str,
    ) -> str:
        """Report your analysis of an image (document, screenshot, or message).

        Args:
            description: Describe what the document contains and key details.
            risk_level: SAFE, LOW_RISK, MEDIUM_RISK, or HIGH_RISK.
            risk_score: 0-100 integer risk score.
            summary: 1-2 sentence summary of the document.
            recommendations: Advice for the user. Separated by newlines.
        """
        tool_result["type"] = "document"
        tool_result["description"] = description
        tool_result["risk_level"] = risk_level
        tool_result["risk_score"] = risk_score
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

    risk_score = 0
    info_req = ""
    caller_claims = ""

    if tool_result:
        result_type = tool_result.get("type", "audio")
        transcription = strip(tool_result.get("transcription", "")) if result_type == "audio" else ""
        description = strip(tool_result.get("description", "")) if result_type == "document" else ""
        # Support both old "verdict" and new "risk_level" field names.
        verdict = strip(tool_result.get("risk_level", "") or tool_result.get("verdict", "SAFE")).upper()
        if verdict not in ("SAFE", "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"):
            # Map legacy values
            verdict = {"SCAM": "HIGH_RISK", "LEGITIMATE": "SAFE", "UNCERTAIN": "MEDIUM_RISK"}.get(verdict, "MEDIUM_RISK")
        try:
            risk_score = int(tool_result.get("risk_score", 0))
        except (ValueError, TypeError):
            risk_score = 0
        summary = strip(tool_result.get("summary", ""))
        raw_recs = strip(tool_result.get("recommendations", ""))
        recommendations = [r.strip().lstrip("0123456789.-) ") for r in raw_recs.split("\n") if r.strip()]
        info_req = strip(tool_result.get("info_requested", ""))
        caller_claims = strip(tool_result.get("caller_claims", ""))
        used_tool = True
    elif tool_parse_fallback:
        result_type = "audio" if has_audio else "document"
        parsed = _parse_tool_from_error(tool_parse_fallback)
        verdict = parsed.get("verdict", "MEDIUM_RISK")
        # Map legacy values from error parsing
        if verdict in ("SCAM", "LEGITIMATE", "UNCERTAIN"):
            verdict = {"SCAM": "HIGH_RISK", "LEGITIMATE": "SAFE", "UNCERTAIN": "MEDIUM_RISK"}.get(verdict, "MEDIUM_RISK")
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
        if verdict in ("SCAM", "LEGITIMATE", "UNCERTAIN"):
            verdict = {"SCAM": "HIGH_RISK", "LEGITIMATE": "SAFE", "UNCERTAIN": "MEDIUM_RISK"}.get(verdict, "MEDIUM_RISK")
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

    # Update progressive risk assessment.
    session.update_verdict(
        verdict, summary, recommendations,
        risk_score=risk_score,
        info_requested=info_req,
        caller_claims=caller_claims,
    )
    session.mark_inference_done()

    # Compress old transcript if it's getting too long.
    if session.mode == "live" and session.needs_compression(pipeline):
        session.compress_transcript(engine, silence_ctx=silence_ctx)

    # Post-inference bank detection from output.
    if not detected_bank and label:
        detected_bank = session.detect_bank(label, detect_bank_fn)

    return {
        "verdict": verdict,                          # segment risk level
        "risk_level": verdict,                       # alias
        "risk_score": risk_score,
        "segment_verdict": verdict,                  # backward compat
        "call_verdict": session.current_risk_level,  # progressive
        "call_risk_level": session.current_risk_level,
        "call_risk_score": session.current_risk_score,
        "input_type": result_type,
        "transcription": transcription,
        "description": description,
        "summary": summary,
        "recommendations": recommendations,
        "info_requested": info_req,
        "caller_claims": caller_claims,
        "asr_time": asr_time,
        "llm_time": llm_time,
        "used_tool": used_tool,
        "detected_bank": detected_bank,
        "notepad": {
            "caller_identity": session.caller_identity,
            "caller_org": session.caller_org,
            "info_requested": list(session.info_requested),
            "call_reason": session.call_reason,
            "risk_factors": list(session.risk_factors),
        },
    }


POST_CALL_PROMPT = """\
You are a call safety analyst. Given the full transcript and real-time risk \
assessment from a phone call, produce a detailed post-call report.

Respond with a JSON object containing these fields:
- final_risk_level: "SAFE", "LOW_RISK", "MEDIUM_RISK", or "HIGH_RISK"
- confidence: "HIGH", "MEDIUM", or "LOW"
- risk_score: integer 0-100 (100 = definite scam)
- call_summary: 2-3 sentence summary of what happened
- caller_identity: who the caller claimed to be
- info_requested: list of information the caller asked for
- risk_factors: list of concerning behaviors found (empty if SAFE)
- safe_indicators: list of legitimate behaviors found (empty if HIGH_RISK)
- timeline: list of objects with "time" and "event" describing key moments
- recommended_actions: list of next steps for the user. \
  For SAFE/LOW_RISK: "No action needed" or simple confirmation steps. \
  For MEDIUM_RISK: verification steps — call official hotline to confirm. \
  For HIGH_RISK: stop sharing info, hang up, report to police/bank.
- verification_questions: list of questions the user could have asked to \
  verify the caller's legitimacy (e.g. "What is your staff ID?", \
  "Can I call back on the official number?")
- report_for_authorities: ONLY if HIGH_RISK. Write a paragraph suitable \
  for filing a police report. Omit or null for other risk levels.
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
    timeline_text = ""
    for entry in summary["risk_timeline"]:
        timeline_text += (
            f"  Turn {entry['turn']}: {entry['risk_level']}"
            f" — {entry['summary']}\n"
        )

    notepad = summary.get("notepad", {})
    notepad_text = (
        f"Caller: {notepad.get('caller_identity', 'unknown')}\n"
        f"Reason: {notepad.get('call_reason', 'unknown')}\n"
        f"Info requested: {', '.join(notepad.get('info_requested', [])) or 'none'}\n"
    )

    user_content = (
        f"== CALL TRANSCRIPT ==\n"
        f"{summary['transcript']}\n\n"
        f"== REAL-TIME ANALYSIS ==\n"
        f"Final risk level: {summary['final_risk_level']}\n"
        f"Final risk score: {summary['final_risk_score']}/100\n"
        f"Bank detected: {summary['detected_bank'] or 'none'}\n"
        f"Total turns analyzed: {summary['turn_count']}\n\n"
        f"== CALL NOTES ==\n{notepad_text}\n"
        f"Risk timeline:\n{timeline_text}\n"
        f"Risk factors found:\n"
        + "\n".join(f"  - {e}" for e in summary["risk_factors"])
        + "\n\nSafe indicators found:\n"
        + "\n".join(f"  - {e}" for e in summary["safe_indicators"])
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
    # Strip markdown code fences if present.
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("`")
    # Find the outermost JSON object.
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        try:
            report = json.loads(json_match.group(0))
            report["elapsed"] = round(elapsed, 2)
            report["raw"] = raw_text
            return report
        except json.JSONDecodeError:
            pass

    # Fallback: build a minimal report from the session data + raw text.
    return {
        "final_risk_level": summary.get("final_risk_level", "MEDIUM_RISK"),
        "risk_score": summary.get("final_risk_score", 0),
        "call_summary": raw_text[:500] if raw_text else "Could not parse model response.",
        "risk_factors": summary.get("risk_factors", []),
        "safe_indicators": summary.get("safe_indicators", []),
        "raw": raw_text,
        "elapsed": round(elapsed, 2),
        "parse_error": "Could not extract JSON — showing raw response.",
    }


class _noop_ctx:
    """No-op context manager."""
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass

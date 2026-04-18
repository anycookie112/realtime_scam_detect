"""
learning.py — Walk through how the scam detection system works, piece by piece.

Run each section independently:
    uv run learning.py section1
    uv run learning.py section2
    ...
    uv run learning.py all        # run everything

Each section builds on the last. Read the comments, then run it.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add src/ so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "agents"))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: ASR — turning audio into text
# ═══════════════════════════════════════════════════════════════════════════

def section1():
    """
    The first step in the pipeline: Whisper ASR.

    Audio comes in as WAV bytes → Whisper transcribes it → we get text.
    This is Pipeline A's first stage. Pipeline B skips this and sends
    audio directly to the LLM.
    """
    print("\n" + "=" * 60)
    print("SECTION 1: ASR (Automatic Speech Recognition)")
    print("=" * 60)

    import asr

    # Load the Whisper model (happens once, cached after)
    print("\n1. Loading Whisper model...")
    model = asr.load_model()
    print(f"   Model loaded: {os.getenv('WHISPER_MODEL_SIZE', 'base')}")

    # Read a test WAV file
    wav_path = Path("test_audio/en/delivery_confirmation_legit.wav")
    if not wav_path.exists():
        print(f"   ! File not found: {wav_path}")
        return
    wav_bytes = wav_path.read_bytes()
    print(f"\n2. Audio file: {wav_path.name} ({len(wav_bytes)} bytes)")

    # Transcribe
    print("\n3. Transcribing...")
    result = asr.transcribe_wav(wav_bytes)

    print(f"   Text:     \"{result.text}\"")
    print(f"   Language: {result.language}")
    print(f"   Time:     {result.asr_time_s:.2f}s")

    print("""
   KEY TAKEAWAY:
   asr.transcribe_wav(bytes) → TranscriptionResult(text, language, time)
   That's the entire ASR interface. One function, bytes in, text out.
""")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: CallSession — managing state across chunks
# ═══════════════════════════════════════════════════════════════════════════

def section2():
    """
    The CallSession is the "memory" of an ongoing call.
    It tracks: transcript, risk level, notepad, bank detection.

    It doesn't do inference — it just manages state.
    """
    print("\n" + "=" * 60)
    print("SECTION 2: CallSession — state management")
    print("=" * 60)

    from session import CallSession

    # Create a session (one per call)
    session = CallSession(mode="live")
    print(f"\n1. New session created")
    print(f"   Risk level: {session.current_risk_level}")
    print(f"   Risk score: {session.current_risk_score}")
    print(f"   Transcript: '{session.running_transcript}'")

    # Simulate adding transcript chunks (what happens after each ASR)
    print("\n2. Adding transcript chunks...")

    session.add_transcript("Good afternoon, I'm calling from Maybank fraud department.")
    print(f"   After chunk 1: '{session.running_transcript}'")

    session.add_transcript("department. There was a charge of RM 1450.")
    print(f"   After chunk 2: '{session.running_transcript}'")
    print(f"   ^ Notice: 'department.' overlap was deduplicated")

    # Simulate risk updates (what happens after each LLM inference)
    print("\n3. Updating risk levels...")

    session.update_verdict("SAFE", "Standard fraud alert", ["No action needed"],
                          caller_claims="Sarah from Maybank, staff ID MBB-4821",
                          info_requested="card ending digits")
    print(f"   After turn 1: risk={session.current_risk_level}, score={session.current_risk_score}")
    print(f"   Notepad caller: {session.caller_identity}")
    print(f"   Notepad info:   {session.info_requested}")

    session.update_verdict("HIGH_RISK", "Transfer to safe account demanded",
                          ["Stop sharing information", "Hang up immediately"],
                          risk_score=85,
                          info_requested="full account transfer, TAC code")
    print(f"   After turn 2: risk={session.current_risk_level}, score={session.current_risk_score}")
    print(f"   Notepad info:   {session.info_requested}")

    # Try to downgrade — it won't work
    session.update_verdict("SAFE", "Caller said goodbye", ["No action needed"])
    print(f"   After turn 3 (tried SAFE): risk={session.current_risk_level}")
    print(f"   ^ HIGH_RISK is sticky — never reverts")

    # Check the history
    print(f"\n4. Risk history:")
    for turn, level, summary in session.risk_history:
        print(f"   Turn {turn}: {level} — {summary}")

    print("""
   KEY TAKEAWAY:
   CallSession is a dataclass. It doesn't call any model.
   It manages: transcript (with dedup), risk level (only escalates),
   notepad (caller identity, info requested), and bank detection.
""")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Context building — what the model sees
# ═══════════════════════════════════════════════════════════════════════════

def section3():
    """
    The model has a 4096 token context window.
    build_context() assembles what goes into it.
    This is the most important function — it determines what the model
    knows when it makes a decision.
    """
    print("\n" + "=" * 60)
    print("SECTION 3: Context building — the model's input")
    print("=" * 60)

    from session import CallSession

    session = CallSession(mode="live")

    # Simulate a few turns of a call
    session.add_transcript("Good afternoon, I'm calling from Maybank.")
    session.add_transcript("There was a charge of RM 1450 on your card ending 7736.")
    session.update_verdict("SAFE", "Standard fraud alert",
                          caller_claims="Maybank fraud dept")

    # Build context for the next inference
    context = session.build_context(pipeline="a")

    print(f"\n1. Context for Pipeline A ({len(context)} chars):\n")
    print("   " + context.replace("\n", "\n   "))

    # Now simulate a longer call to see the notepad
    session.add_transcript("Can you confirm the last 4 digits of your IC?")
    session.update_verdict("SAFE", "Identity verification",
                          info_requested="last 4 IC digits",
                          caller_claims="Maybank fraud dept, staff ID MBB-4821")

    context2 = session.build_context(pipeline="a")
    print(f"\n2. After more turns ({len(context2)} chars):\n")
    print("   " + context2.replace("\n", "\n   "))

    # Show token budget
    print(f"""
3. Token budget breakdown (Pipeline A):
   System prompt:     ~500 tokens
   This context:      ~{len(context2)//4} tokens
   Bank KB (if any):  ~120 tokens
   New ASR text:      ~50-80 tokens
   Output:            ~200 tokens
   ─────────────────────────────
   Total:             ~{500 + len(context2)//4 + 120 + 80 + 200} / 4096 tokens

   Pipeline B uses more tokens for audio blob (~800-1200),
   so the context window for transcript is smaller (~4800 chars vs ~8000).
""")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Bank KB — domain knowledge injection
# ═══════════════════════════════════════════════════════════════════════════

def section4():
    """
    The bank knowledge base is a hardcoded dict of facts about Malaysian banks.
    When a bank name is detected in the transcript, its KB entry gets
    injected into the prompt — giving the model specific knowledge about
    what that bank does and doesn't do.
    """
    print("\n" + "=" * 60)
    print("SECTION 4: Bank Knowledge Base")
    print("=" * 60)

    import server

    # Show available banks
    print(f"\n1. Banks in KB: {list(server.BANK_KB.keys())}")

    # Detect a bank from text
    text = "I'm calling from Maybank about your credit card"
    bank = server._detect_bank(text)
    print(f"\n2. Detecting bank from: \"{text}\"")
    print(f"   Detected: {bank}")

    # Show what gets injected into the prompt
    if bank:
        context = server._bank_context(bank)
        print(f"\n3. Bank context injected into prompt:\n")
        print("   " + context.replace("\n", "\n   "))

    # Try aliases
    print(f"\n4. Alias detection:")
    for text in ["MBB called me", "I got a call from PBB", "CIMB Clicks"]:
        bank = server._detect_bank(text)
        print(f"   \"{text}\" → {bank}")

    print("""
   KEY TAKEAWAY:
   When the model sees "Maybank" in the transcript, the prompt gets
   enriched with Maybank-specific facts: hotline numbers, what they
   NEVER do (ask for OTP), what they MAY do (verify last 4 digits).
   This helps the model distinguish real Maybank calls from impersonators.
""")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Tool calling — how the model responds
# ═══════════════════════════════════════════════════════════════════════════

def section5():
    """
    The model doesn't return free text. It calls a tool:
    analyze_speech(transcription, risk_level, risk_score, summary, ...)

    This is how we get structured output from the LLM.
    When tool calling fails (quantized models often produce malformed JSON),
    we fall back to regex parsing.
    """
    print("\n" + "=" * 60)
    print("SECTION 5: Tool calling and fallback parsing")
    print("=" * 60)

    # Show the tool signature
    print("""
1. The model is given this tool definition:

   analyze_speech(
       transcription: str,     # what the caller said
       risk_level: str,        # SAFE, LOW_RISK, MEDIUM_RISK, HIGH_RISK
       risk_score: int,        # 0-100
       summary: str,           # what's happening
       recommendations: str,   # what to say/ask
       info_requested: str,    # what they asked for
       caller_claims: str,     # who they say they are
   )

   When the model calls this tool, litert_lm invokes our Python function
   with the arguments. We capture them in a dict.
""")

    # Show what happens when tool calling fails
    print("2. When tool calling fails (common with quantized models):")
    print("   The engine throws RuntimeError with the raw output embedded.")
    print("   We extract fields with regex:")

    from session import _parse_tool_from_error

    # Simulate an error string (this is what litert_lm actually produces)
    fake_error = (
        'RuntimeError: Failed to parse tool calls from response: '
        '<|tool_call>call:analyze_speech{risk_level:HIGH_RISK,'
        'summary:<|"|>Caller demanding fund transfer<|"|>,'
        'recommendations:<|"|>Stop sharing information\nHang up<|"|>,'
        'transcription:<|"|>Transfer your funds to account 5628<|"|>,'
        'risk_score:85}<tool_call|>'
    )

    parsed = _parse_tool_from_error(fake_error)
    print(f"\n   Parsed from error:")
    for k, v in parsed.items():
        print(f"     {k}: {v}")

    # Show lenient parsing (when model doesn't use tool at all)
    print("\n3. When model returns raw text instead of tool call:")

    import server
    raw_text = '{"risk_level": "MEDIUM_RISK", "summary": "Caller asking for IC"}'
    parsed2 = server._lenient_parse(raw_text)
    print(f"   Input: {raw_text}")
    print(f"   Parsed: {parsed2}")

    print("""
   KEY TAKEAWAY:
   Three parsing paths, in order of preference:
   1. Tool callback fires → structured data (best)
   2. Engine throws parse error → regex extract from error string
   3. Raw text fallback → lenient JSON/regex parsing
   All three produce the same output format: verdict, summary, recs, etc.
""")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Audio chunking — how streaming works
# ═══════════════════════════════════════════════════════════════════════════

def section6():
    """
    In streaming mode, a long audio file is split into overlapping chunks.
    Each chunk is processed independently, but the CallSession maintains
    context across chunks.
    """
    print("\n" + "=" * 60)
    print("SECTION 6: Audio chunking for streaming")
    print("=" * 60)

    from audio_utils import split_wav

    wav_path = Path("test_audio/en/maybank_gradual_scam.wav")
    if not wav_path.exists():
        print(f"   ! File not found: {wav_path}")
        return
    wav_bytes = wav_path.read_bytes()

    import wave
    with wave.open(str(wav_path), "rb") as w:
        duration = w.getnframes() / w.getframerate()
    print(f"\n1. File: {wav_path.name} ({duration:.1f}s)")

    # Split with different parameters
    for chunk_s, overlap_s in [(4, 0.5), (15, 1), (30, 2)]:
        chunks = split_wav(wav_bytes, chunk_s, overlap_s)
        step = chunk_s - overlap_s
        print(f"\n2. chunk={chunk_s}s, overlap={overlap_s}s → {len(chunks)} chunks")
        print(f"   Step size: {step}s (new audio per chunk)")
        print(f"   First 3 chunks cover: ", end="")
        for i in range(min(3, len(chunks))):
            start = i * step
            end = start + chunk_s
            print(f"[{start:.0f}-{end:.0f}s] ", end="")
        print()

    print("""
   The overlap ensures words at chunk boundaries don't get cut.
   The session's add_transcript() deduplicates the overlapping text.

   On a Pixel 8 at 9 tok/s:
   - 4s chunks → too many inferences, can't keep up
   - 15s chunks → ~10s per inference, fits within 15s window
   - 15s + skip-2 → infer every 30s of audio, easily fits

   KEY TAKEAWAY:
   split_wav(bytes, chunk_s, overlap_s) → list of WAV byte chunks
   Each chunk is a valid WAV file with headers.
""")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: run_turn() — the full inference cycle
# ═══════════════════════════════════════════════════════════════════════════

def section7():
    """
    run_turn() is the core function. It ties everything together:
    ASR → context building → bank KB → LLM inference → parse → update session.

    This requires the LLM engine to be loaded, so it's the heaviest section.
    """
    print("\n" + "=" * 60)
    print("SECTION 7: run_turn() — full inference cycle")
    print("=" * 60)

    print("""
   run_turn() is called once per audio chunk. Here's what it does:

   1. ASR (Pipeline A only)
      wav_bytes → asr.transcribe_wav() → text
      session.add_transcript(text)

   2. Build context
      session.build_context(pipeline) → history + notepad + prev assessment

   3. Bank detection
      session.detect_bank(text, detect_fn) → inject bank KB if found

   4. Assemble prompt
      [context] + [bank KB] + [new transcript] + [instruction]

   5. LLM inference
      engine.create_conversation(system_prompt, tools)
      conv.send_message(prompt) → tool call or raw text

   6. Parse result
      Tool callback → structured data
      OR error fallback → regex extract
      OR raw text → lenient parse

   7. Update session
      session.update_verdict(risk_level, summary, recs, ...)
      session.mark_inference_done()

   8. Compress if needed
      If transcript > 5000 chars → summarize old portion

   9. Return result dict
      {risk_level, risk_score, summary, recommendations, notepad, ...}
""")

    print("   To actually run this, you need the LLM engine loaded.")
    print("   Use the stream monitor instead:")
    print("   uv run tools/eval/stream_monitor.py test_audio/en/maybank_gradual_scam.wav --chunk-seconds 15")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: Risk escalation — how verdicts evolve
# ═══════════════════════════════════════════════════════════════════════════

def section8():
    """
    The risk level can only go up, never down.
    This simulates a full call where risk escalates over time.
    """
    print("\n" + "=" * 60)
    print("SECTION 8: Risk escalation over a call")
    print("=" * 60)

    from session import CallSession

    session = CallSession(mode="live")

    # Simulate a gradual scam call
    turns = [
        ("SAFE", 5, "Caller introduces themselves as Maybank",
         "Sarah from Maybank, staff ID MBB-4821", ""),
        ("SAFE", 10, "Reports a suspicious transaction on your card",
         "", "card ending digits"),
        ("SAFE", 10, "Offers to block the card",
         "", ""),
        ("SAFE", 15, "Asks to confirm last 4 IC digits",
         "", "last 4 IC digits"),
        ("LOW_RISK", 25, "Mentions a 'larger syndicate' affecting accounts",
         "", ""),
        ("HIGH_RISK", 85, "Demands transfer to 'secure holding account'",
         "", "full account transfer"),
        ("HIGH_RISK", 90, "Asks for TAC code",
         "", "TAC code"),
        ("SAFE", 5, "Says thank you and goodbye",
         "", ""),
    ]

    print(f"\n{'Turn':<5} {'Segment':<13} {'Call Level':<13} {'Score':>5}  Summary")
    print("-" * 75)

    for i, (level, score, summary, caller, info) in enumerate(turns):
        session.update_verdict(
            level, summary, [],
            risk_score=score,
            caller_claims=caller,
            info_requested=info,
        )
        print(f"  {i+1:<3}  {level:<13} {session.current_risk_level:<13} {session.current_risk_score:>5}  {summary}")

    print(f"""
   Final: {session.current_risk_level} (score {session.current_risk_score})

   Notice:
   - Turns 1-4: SAFE (routine verification)
   - Turn 5: LOW_RISK escalated the call level
   - Turn 6: HIGH_RISK triggered (transfer demand)
   - Turn 7: HIGH_RISK confirmed (TAC request)
   - Turn 8: Tried to go back to SAFE → BLOCKED (sticky)

   The notepad accumulated:
   - Caller: {session.caller_identity}
   - Info requested: {', '.join(session.info_requested)}
   - Risk factors: {session.risk_factors}
""")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: Post-call analysis — the detailed report
# ═══════════════════════════════════════════════════════════════════════════

def section9():
    """
    After the call ends, run_post_call() sends the full transcript
    and session state to the LLM for a detailed analysis report.
    This is Pass 2 — a slower, more thorough analysis.
    """
    print("\n" + "=" * 60)
    print("SECTION 9: Post-call analysis")
    print("=" * 60)

    from session import CallSession

    session = CallSession(mode="live")
    session.add_transcript("Good afternoon from Maybank. Your card was used for RM 1450.")
    session.add_transcript("Transfer your funds to secure account 5628-1190-3347.")
    session.update_verdict("SAFE", "Fraud alert", [],
                          caller_claims="Sarah from Maybank")
    session.update_verdict("HIGH_RISK", "Transfer demand", [],
                          risk_score=85, info_requested="fund transfer, TAC")

    # Show the summary that would be sent to the LLM
    summary = session.post_call_summary()
    print("\n1. Post-call summary (sent to LLM):\n")
    print(json.dumps(summary, indent=2, default=str))

    print("""
2. The LLM receives:
   - Full transcript
   - Risk timeline (which turns escalated)
   - Notepad (caller identity, info requested)
   - Risk factors and safe indicators

   It returns a JSON report with:
   - final_risk_level, confidence, risk_score
   - call_summary, caller_identity, info_requested
   - risk_factors, safe_indicators
   - timeline of key events
   - recommended_actions
   - verification_questions (what user could have asked)
   - report_for_authorities (only if HIGH_RISK)

   This requires the LLM engine. To see it in action:
   uv run tools/eval/stream_monitor.py test_audio/en/maybank_gradual_scam.wav
""")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: The full picture
# ═══════════════════════════════════════════════════════════════════════════

def section10():
    """
    How everything connects end-to-end.
    """
    print("\n" + "=" * 60)
    print("SECTION 10: The full picture")
    print("=" * 60)

    print("""
   ┌─────────────────────────────────────────────────────────┐
   │                    AUDIO SOURCE                         │
   │          (mic via VAD / test file chunks)                │
   └────────────────────┬────────────────────────────────────┘
                        │ WAV bytes (4-15s chunks)
                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │              ASR (Whisper)                               │
   │         wav_bytes → text ("I'm from Maybank...")        │
   │         Pipeline B skips this, sends raw audio          │
   └────────────────────┬────────────────────────────────────┘
                        │ text
                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │           CallSession.add_transcript()                  │
   │    Dedup overlap → append to running_transcript         │
   │    "...from Maybank" + "Maybank fraud dept" → deduped   │
   └────────────────────┬────────────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │           CallSession.build_context()                   │
   │                                                         │
   │    Layer 1: [Compressed summary of old text]            │
   │    Layer 2: Sliding window of recent transcript         │
   │    Layer 3: Previous assessment: SAFE                   │
   │    Layer 4: Call Notes (caller, reason, info, risks)    │
   │    Layer 5: Bank KB (Maybank hotline, never/may do)     │
   │                                                         │
   │    Total: ~2800 tokens / 4096 budget                    │
   └────────────────────┬────────────────────────────────────┘
                        │ assembled prompt
                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │           LLM (Gemma E2B via litert_lm)                │
   │                                                         │
   │    System: risk rules + verification guidance           │
   │    Tools: [analyze_speech, analyze_document]            │
   │    User: [context + new chunk + instruction]            │
   │                                                         │
   │    Model calls: analyze_speech(                         │
   │      risk_level="HIGH_RISK",                            │
   │      risk_score=85,                                     │
   │      summary="Transfer to safe account demanded",       │
   │      recommendations="Stop sharing info\\nHang up",     │
   │      info_requested="fund transfer, TAC",               │
   │      caller_claims="Sarah from Maybank"                 │
   │    )                                                    │
   └────────────────────┬────────────────────────────────────┘
                        │ structured result
                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │       CallSession.update_verdict()                      │
   │                                                         │
   │    Risk: SAFE → HIGH_RISK (escalated, now sticky)      │
   │    Score: max(0, 85) = 85                               │
   │    Notepad: caller, info_requested, risk_factors        │
   │    History: [(1, SAFE, ...), (2, HIGH_RISK, ...)]      │
   └────────────────────┬────────────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │              WebSocket → Browser                        │
   │                                                         │
   │    {risk_level, risk_score, summary, recommendations,   │
   │     call_risk_level, call_risk_score, notepad,          │
   │     running_transcript, ...}                            │
   │                                                         │
   │    UI updates: big verdict, risk bar, "what to say",    │
   │    history sidebar, notepad panel                       │
   └─────────────────────────────────────────────────────────┘
                        │
                        │ (after all chunks)
                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │         Post-Call Analysis (Pass 2)                     │
   │                                                         │
   │    Full transcript + notepad + risk timeline            │
   │    → LLM generates detailed JSON report                 │
   │    → Modal popup in browser                             │
   └─────────────────────────────────────────────────────────┘

   Files involved:
   - src/asr.py          → Whisper transcription
   - src/session.py      → CallSession + run_turn() + run_post_call()
   - src/server.py       → WebSocket handlers, system prompt, bank KB
   - src/audio_utils.py  → WAV splitting for streaming simulation
   - src/index.html      → Browser UI
""")


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

SECTIONS = {
    "section1": (section1, "ASR — turning audio into text"),
    "section2": (section2, "CallSession — state management"),
    "section3": (section3, "Context building — the model's input"),
    "section4": (section4, "Bank KB — domain knowledge"),
    "section5": (section5, "Tool calling and fallback parsing"),
    "section6": (section6, "Audio chunking for streaming"),
    "section7": (section7, "run_turn() — full inference cycle"),
    "section8": (section8, "Risk escalation over a call"),
    "section9": (section9, "Post-call analysis"),
    "section10": (section10, "The full picture"),
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "help":
        print("\nUsage: uv run learning.py <section>\n")
        print("Available sections:")
        for name, (_, desc) in SECTIONS.items():
            print(f"  {name:<12} — {desc}")
        print(f"  {'all':<12} — run all sections")
        sys.exit(0)

    if sys.argv[1] == "all":
        for name, (fn, desc) in SECTIONS.items():
            fn()
    elif sys.argv[1] in SECTIONS:
        SECTIONS[sys.argv[1]][0]()
    else:
        print(f"Unknown section: {sys.argv[1]}")
        print(f"Available: {', '.join(SECTIONS.keys())}, all")
        sys.exit(1)

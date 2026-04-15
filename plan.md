# Realtime Scam Detection App Plan

## 1. Project Goal

Build an Android app that helps users detect bank impersonation scams during and after a phone call.

The app should:

- monitor a call in progress using speakerphone plus microphone capture
- provide realtime warnings when suspicious behavior is detected
- run a more detailed analysis after the call ends
- benchmark two model pipelines on-device to compare quality, latency, and battery impact

The two pipelines to compare are:

- Pipeline A: `audio -> Whisper ASR -> text -> Gemma 3n E2B -> scam analysis`
- Pipeline B: `audio -> Gemma 3n E2B (direct audio input) -> scam analysis`

## 2. Product Definition

The app is not just a transcription tool. It is a decision-support tool for scam prevention.

During a suspicious call, the app should surface short, actionable advice such as:

- "Do not share OTP or PIN"
- "Do not transfer money during this call"
- "Hang up and call the official bank number"
- "This caller may be using urgency or intimidation"

After the call, the app should provide a detailed report with:

- final scam risk score
- confidence level
- evidence summary
- notable timestamps
- optional diarization and role inference results
- recommended next steps

## 3. Platform Reality and Constraints

This project should be planned around real Android limitations, not idealized access.

Known constraints:

- a normal Android app usually cannot access clean separate audio streams for a cellular call
- a normal app also cannot reliably capture call audio directly from the telephony stack
- the workable approach for a prototype is speakerphone plus microphone capture with user opt-in
- this produces mixed audio, not perfect caller-versus-user separation
- diarization and role inference will be approximate unless the call stack is controlled by the app

Implications:

- realtime detection should not depend on perfect speaker separation
- the first version should analyze the mixed conversation for scam patterns
- diarization should be added as a best-effort post-call enhancement
- if strict speaker separation becomes a hard requirement, a future version should move to an app-controlled VoIP calling flow

## 4. High-Level Architecture

The app should be designed as a two-pass system.

### Pass 1: Realtime Analysis

Purpose:

- produce low-latency risk estimates during the call
- show short warnings without overwhelming the user

Input:

- rolling mixed-audio chunks captured from speakerphone plus mic

Output:

- risk score
- suspicious/not suspicious classification
- short advice message
- optional evidence tags

Example evidence tags:

- asked_for_otp
- asked_for_pin
- asked_for_transfer
- created_urgency
- claimed_bank_security_issue
- asked_to_install_app
- discouraged_hanging_up

### Pass 2: Post-Call Analysis

Purpose:

- produce a more accurate and more detailed report after the call ends

Input:

- full recorded mixed audio from the call
- chunk-level realtime results
- call metadata, if available

Output:

- final risk score
- confidence
- detailed evidence
- suspicious moments timeline
- optional diarization results
- optional speaker role inference
- user-facing summary

## 5. Module Breakdown

The codebase should be split into clear modules with simple boundaries.

### 5.1 Android UI Layer

Responsibilities:

- onboarding and permission flow
- call monitoring state
- live warning cards
- post-call report view
- benchmark screen
- settings for model selection and privacy controls

Recommended stack:

- Kotlin
- Jetpack Compose
- ViewModel + unidirectional state flow

### 5.2 Audio Capture Layer

Responsibilities:

- capture mixed speakerphone plus microphone audio
- normalize sample rate and channel layout
- chunk audio into fixed windows
- buffer rolling context for realtime analysis
- store call recordings for post-call analysis

Suggested behavior:

- mono audio where possible
- `16 kHz` target sample rate if required by the downstream model pipeline
- chunk size of `2-5 seconds`
- overlap of `0.5-1 second`

### Voice Activity Detection

The audio capture layer should include a lightweight Voice Activity Detection (VAD) gate before sending chunks to the inference pipeline.

Purpose:

- avoid wasting compute and battery analyzing silence or background noise
- reduce unnecessary model calls during pauses in conversation
- improve realtime responsiveness by skipping non-speech chunks

Suggested approach:

- use a small on-device VAD model (such as Silero VAD or WebRTC VAD)
- only forward chunks that contain detected speech above a confidence threshold
- keep buffering audio even during silence so context is not lost

### Conversation History Across Chunks

A single 2-5 second chunk rarely contains a full scam signal. Scammers build pressure over minutes through gradual escalation, repeated requests, and context switching. The realtime inference layer must maintain a rolling conversation history.

Strategy:

- maintain a sliding window of the last N chunk transcripts or summaries
- include the rolling history in the prompt so the model can detect patterns across chunks
- for Pipeline A, this means concatenating recent transcript segments
- set a maximum history length to stay within the model's context window
- when the history exceeds the limit, summarize older chunks into a compressed context block

Important:

- isolated chunk analysis will produce high false negative rates for gradual-escalation scams
- the conversation history is what enables detection of patterns like "caller asked for OTP after spending three minutes creating urgency"
- history size must be balanced against inference latency — more context means slower per-chunk inference

### 5.3 Realtime Inference Layer

Responsibilities:

- run Pipeline A and Pipeline B behind a common interface
- produce chunk-level results in a standard schema
- support quick swapping between inference strategies

Suggested interface:

```kotlin
interface ScamAnalysisPipeline {
    suspend fun analyzeChunk(input: AnalysisChunk): ChunkAnalysisResult
}
```

### 5.4 Post-Call Analysis Layer

Responsibilities:

- aggregate all chunk-level results
- run deeper analysis on the full call
- run diarization and role inference
- produce the final report

### 5.5 Risk Decision Layer

Responsibilities:

- convert model outputs into stable user-facing warnings
- apply deterministic business rules on top of model outputs
- reduce false positives through thresholds and evidence requirements

Important rule:

- user safety advice should not depend only on free-form model text
- use structured outputs and deterministic thresholds wherever possible

### 5.6 Model Output Parsing and Validation

Small on-device quantized models frequently produce malformed JSON — missing closing braces, truncated output, hallucinated field names, or wrong value types. The app must not crash or silently fail when this happens.

Required behavior:

- validate all model output against the expected JSON schema before using it
- use a lenient JSON parser that can recover from minor formatting errors (trailing commas, unquoted keys)
- if the output is completely unparseable, fall back to a safe default result with `is_suspicious: false` and `confidence: low` plus a flag indicating the chunk was not successfully analyzed
- reject evidence tags that are not in the defined taxonomy — do not pass hallucinated tags to the UI or risk engine
- log parse failures for debugging and prompt improvement

For structured output reliability:

- keep the JSON schema as simple as possible
- use few-shot examples in the prompt that show the exact expected format
- consider asking the model to output one field at a time if full JSON is too unreliable
- apply regex-based extraction as a fallback for critical fields like risk_score

### 5.7 Notification and Overlay Strategy

During an active phone call, the app needs to display warnings while the phone app is in the foreground. This is a non-trivial Android UX problem.

Options:

- **Bubble notifications** (Android 11+): floating chat-head style alerts that can appear over other apps without `SYSTEM_ALERT_WINDOW`
- **Heads-up notifications**: high-priority notifications that appear as banners at the top of the screen
- **`SYSTEM_ALERT_WINDOW` overlay**: full overlay permission, provides most control but requires user opt-in and triggers stricter Play Store review
- **Picture-in-Picture (PiP) mode**: small floating window, appropriate for persistent risk indicator

Recommended approach for v1:

- use heads-up notifications for urgent warnings (high risk detected)
- use a persistent foreground service notification showing current monitoring status
- defer full overlay UI to a later version, after validating user behavior with simpler notifications
- if the user has granted overlay permission, optionally show a floating risk indicator

### 5.8 Benchmark Layer

Responsibilities:

- run identical test sets through both pipelines
- collect latency, memory, battery, and accuracy metrics
- compare live and post-call performance

## 6. Recommended Development Phases

Build this in phases so that each stage creates a working checkpoint.

### Phase 0: Project Setup

Deliverables:

- create Android app module
- choose package structure
- define shared data models
- define benchmark result schema
- define JSON output contract for both pipelines

Success criteria:

- app launches
- settings screen exists
- test harness can load a local audio file

### Phase 1: Offline Replay Benchmark Harness

Goal:

- make benchmarking reproducible before dealing with live call complexity

Work:

- build a screen to load pre-recorded scam and non-scam calls
- run both pipelines against the same audio files
- save structured results locally
- display latency and output comparison side by side

Success criteria:

- same clip can be replayed through both pipelines
- outputs are saved in a comparable format
- benchmark report can be exported or viewed on device

Why this phase matters:

- it removes call-capture uncertainty early
- it lets model experimentation start immediately

### Phase 2: Realtime Mixed-Audio Prototype

Goal:

- detect suspicious behavior during an active call using speakerphone plus mic capture

Work:

- implement rolling chunk capture
- add live risk scoring
- show short warning banners
- store chunk timestamps and scores

Success criteria:

- app can process live chunks continuously
- warnings appear within an acceptable latency budget
- chunk results are stored for post-call reuse

### Phase 3: Post-Call Deep Analysis

Goal:

- generate a richer report after the call finishes

Work:

- run full-call reanalysis
- add diarization module
- add speaker role inference
- merge evidence into a final report

Success criteria:

- app produces a final structured report within a reasonable wait time
- report includes timestamps and evidence summaries

### Phase 4: Bank-Specific RAG Knowledge Base

Goal:

- enable bank-specific scam detection using curated procedure data

Work:

- define the bank knowledge schema
- build an initial knowledge base from public bank security pages and scam advisories
- implement bank name detection from transcript chunks
- implement retrieval and prompt injection of bank context
- add bank-specific fields to realtime and post-call outputs
- test with known scam scenarios per bank

Success criteria:

- when a caller claims to be from a known bank, the analysis uses that bank's real procedures
- user advice references specific bank policies
- fallback to generic detection works when bank is unknown
- knowledge base can be updated without rebuilding the app

### Phase 5: Call Metadata Integration

Goal:

- improve detection with non-audio signals

Work:

- integrate incoming call metadata
- use call screening information if available
- incorporate caller verification or number trust signals
- add allowlist and blocklist support

Success criteria:

- risk engine can combine model evidence and call metadata
- UI can explain why a warning was issued

### Phase 6: UX Hardening and Safety Guardrails

Goal:

- make the app usable during stressful calls

Work:

- simplify warning language
- tune thresholds for fewer false alarms
- add privacy disclosures and consent flow
- add offline-first behavior
- add failure handling when model inference is too slow

Success criteria:

- app remains understandable under pressure
- user can act on warnings immediately

## 7. Suggested Pipeline Design

### Pipeline A: Whisper First

Flow:

- audio chunk
- speech-to-text
- prompt Gemma with transcript and scam-detection instructions
- receive structured result

Pros:

- easier to inspect and debug
- transcript can be shown and reviewed
- more modular

Cons:

- more latency
- more moving parts
- ASR errors may damage downstream detection

### Pipeline B: Direct Audio to Gemma 3n E2B

**Viability: CONFIRMED (server-side, Python LiteRT engine).**

The Python `litert_lm.Engine` in `src/server.py` already accepts raw audio
blobs alongside the system prompt and returns transcription + verdict in a
single call (see `audio_backend=litert_lm.Backend.CPU` and the per-turn
`{"type": "audio", "blob": ...}` content block). This means Pipeline B is
viable end-to-end on the server today using `gemma-4-E2B-it.litertlm`.

**Still unverified:** whether the same audio-input capability is exposed in
the **Android AI Edge SDK** for on-device inference. The Python `litert_lm`
package and the Android AI Edge runtime are different bindings around the
same model files, and audio input has historically been gated per-binding.
Before committing to Pipeline B on-device, build a minimal Kotlin smoke test
that loads `gemma-4-E2B-it.litertlm` and feeds it a WAV blob — if the SDK
rejects it, Pipeline B falls back to server-only and the Android app must
ship Pipeline A (Whisper ASR + text Gemma) for offline use.

Flow:

- audio chunk
- prompt Gemma with audio input plus scam-detection instructions
- receive structured result

Pros:

- simpler pipeline
- no separate ASR dependency
- may capture prosody and urgency cues that transcripts miss

Cons:

- harder to debug
- output behavior may be less interpretable
- speaker separation remains difficult on mixed audio

## 8. Benchmark Plan

Benchmarking should be designed from the beginning, not added later.

### 8.1 Core Questions

- which pipeline is faster end-to-end on a phone?
- which pipeline is more accurate for scam detection?
- which pipeline produces more useful realtime warnings?
- which pipeline drains less battery or overheats less?

### 8.2 Metrics

For each run, record:

- pipeline name
- model variant
- device model
- chunk size
- overlap size
- time to first warning
- average chunk latency
- final report latency
- memory usage
- battery delta
- device temperature if accessible
- final classification
- confidence
- evidence tags

### 8.3 Accuracy Metrics

Use:

- precision
- recall
- F1
- false positive rate
- false negative rate
- warning timeliness

For this use case, false negatives are especially dangerous, but false positives can also erode trust. Threshold tuning will matter.

### 8.4 Datasets

Build three evaluation buckets:

- obvious scam calls
- borderline or ambiguous calls
- legitimate bank or customer support calls

For each clip, maintain labels for:

- scam or not scam
- suspicious timestamps
- requests for OTP, PIN, transfer, app install, or secrecy
- urgency language

## 9. Realtime Output Contract

The realtime system should not return open-ended prose only. Use a structured schema.

Suggested realtime result:

```json
{
  "schema_version": "1.0",
  "timestamp_start_ms": 0,
  "timestamp_end_ms": 5000,
  "risk_score": 0.82,
  "is_suspicious": true,
  "priority": "high",
  "evidence_tags": [
    "asked_for_otp",
    "created_urgency"
  ],
  "user_advice": "Do not share OTP or PIN. Hang up and call the official bank number.",
  "notes": "The speaker is pressuring for immediate action.",
  "bank_context": {
    "detected_bank": "Bank X",
    "policy_match": "Bank X never asks for OTP over the phone.",
    "callback_number": "+1-800-123-4567"
  }
}
```

Suggested post-call result:

```json
{
  "schema_version": "1.0",
  "final_risk_score": 0.91,
  "is_likely_scam": true,
  "confidence": "high",
  "summary": "Likely bank impersonation scam with repeated requests for account verification and urgency-based pressure.",
  "evidence_tags": [
    "claimed_bank_issue",
    "asked_for_otp",
    "discouraged_hanging_up",
    "asked_for_transfer"
  ],
  "key_moments": [
    {
      "timestamp_ms": 124000,
      "reason": "Caller asked for OTP"
    },
    {
      "timestamp_ms": 201000,
      "reason": "Caller requested transfer to a safe account"
    }
  ],
  "role_inference": {
    "speaker_a": "likely_caller",
    "speaker_b": "likely_user",
    "confidence": "medium"
  },
  "recommended_action": "Do not transfer funds. Contact the bank using the official number.",
  "bank_context": {
    "detected_bank": "Bank X",
    "policies_violated": [
      "Bank X never asks for OTP over the phone",
      "Bank X never requests transfers to safe accounts"
    ],
    "policies_consistent": [
      "Bank X does send SMS before calling"
    ],
    "callback_number": "+1-800-123-4567",
    "branch_locator_url": "https://bankx.example.com/branches"
  }
}
```

## 10. Diarization and Role Inference Plan

Diarization should be treated as an enhancement, not a hard dependency for launch.

### Realtime

Do not require diarization for realtime warnings.

Instead:

- analyze the mixed conversation
- look for scam patterns regardless of exact speaker labels
- infer likely direction only when confidence is high

### Post-Call

Use diarization to:

- split the call into `Speaker A` and `Speaker B`
- estimate speaking turns
- support better evidence summaries

Use role inference to estimate:

- which speaker is likely the caller
- which speaker is likely the phone owner

Signals for role inference may include:

- who asks for OTP or PIN
- who gives personal details
- who expresses confusion or compliance
- who uses institution-claiming language like "this is the bank"

## 11. Bank-Specific Knowledge via RAG

### 11.1 Motivation

Different banks have different procedures. A scam caller claiming "we need your OTP to verify your account" may be immediately suspicious for one bank but sound plausible for another. Without bank-specific context, the app can only apply generic scam heuristics.

By partnering with banks and building a knowledge base of their real procedures, the app can:

- compare what the caller claims against what the bank actually does
- produce more specific and actionable advice
- reduce false positives on legitimate bank calls
- give the user verifiable facts like "Bank X never asks for your OTP over the phone"

### 11.2 Knowledge Base Design

Each bank entry should contain structured data covering:

- bank name and aliases
- official customer service numbers
- list of things the bank will never ask for on a call (OTP, PIN, full card number, transfers to safe accounts, remote app installs)
- list of things the bank may legitimately ask for (partial account number, date of birth for verification, security questions)
- standard call procedures (e.g., "Bank X always sends an SMS before calling", "Bank Y will never call you first about fraud")
- escalation instructions (official number to call back, branch locator URL)
- known scam patterns reported to or by this bank

Storage format:

```json
{
  "bank_id": "bank_x",
  "display_name": "Bank X",
  "aliases": ["BankX", "Bank X Corp"],
  "official_numbers": ["+1-800-123-4567"],
  "never_asks": [
    "full_otp",
    "full_pin",
    "full_card_number",
    "transfer_to_safe_account",
    "install_remote_access_app"
  ],
  "may_ask": [
    "partial_account_number",
    "date_of_birth",
    "security_question"
  ],
  "standard_procedures": [
    "Always sends SMS verification code before calling",
    "Will never initiate outbound calls about fraud alerts",
    "Customer service calls are always from the official number"
  ],
  "escalation": {
    "callback_number": "+1-800-123-4567",
    "branch_locator_url": "https://bankx.example.com/branches"
  },
  "known_scam_patterns": [
    "Caller claims account is frozen and requests OTP to unfreeze",
    "Caller claims suspicious transaction and asks for transfer to safe account"
  ]
}
```

### 11.3 Retrieval Strategy

The RAG pipeline should work as follows:

1. During analysis, the model or a lightweight classifier identifies which bank the caller claims to represent (from phrases like "this is Bank X calling")
2. The bank identifier is used to retrieve the matching bank knowledge entry
3. The bank-specific context is injected into the scam analysis prompt
4. The model can now compare caller claims against the bank's real procedures

For the prototype:

- use a simple keyword or entity match to identify the bank name from the transcript chunk
- look up the bank entry from a local JSON or SQLite store
- inject the relevant fields into the prompt

For production on Android:

- store the knowledge base locally on device (small enough for JSON or a lightweight embedded DB)
- update it periodically via a sync mechanism (API pull from a partner backend, or bundled with app updates)
- no network call required at inference time, keeping the app fully offline-capable

### 11.4 Prompt Integration

When bank context is available, the scam analysis prompt should include it. Example:

```
The caller claims to be from Bank X.

Bank X's known policies:
- Bank X will NEVER ask for your full OTP or PIN over the phone.
- Bank X will NEVER ask you to transfer money to a "safe account."
- Bank X always sends an SMS before making outbound calls.
- If in doubt, hang up and call +1-800-123-4567.

Given these policies, analyze the following conversation chunk for scam indicators...
```

When bank context is not available (bank not identified or not in the knowledge base), fall back to generic scam detection prompts.

### 11.5 Bank Partnership Model

For this feature to scale:

- banks provide their official call procedures and known scam patterns
- the app maintainer curates and validates entries before adding them to the knowledge base
- banks can push updates when their procedures change or new scam patterns emerge
- a simple partner API or data-sharing agreement would support this

Even without formal partnerships, an initial knowledge base can be built from:

- publicly available bank security pages
- published scam advisories from banking regulators
- user-reported patterns (with verification)

### 11.6 Impact on Detection Quality

With bank-specific RAG:

| Scenario | Without RAG | With RAG |
|---|---|---|
| Caller asks for OTP | Generic warning: "Do not share OTP" | Specific: "Bank X never asks for OTP by phone. This is likely a scam." |
| Caller says "we sent you an SMS" | No additional context | Can verify: "Bank X does send SMS before calling" — lowers false positive |
| Caller asks to transfer to safe account | Generic warning | Specific: "Bank X has confirmed they never request safe account transfers" |
| Unknown bank | Generic detection | Falls back to generic detection (no regression) |

## 12. Safety, Privacy, and Policy Requirements

This app deals with highly sensitive audio and potentially financial harm.

Must-have safeguards:

- explicit user consent before monitoring
- clear disclosure that mixed audio may be processed
- on-device processing by default where possible
- short retention windows for stored recordings unless user chooses otherwise
- clear explanation that the app provides assistance, not certainty
- ability to disable monitoring at any time

Policy note:

- assume Google Play review will be stricter for call-related behavior and sensitive permissions
- avoid architectures that depend on unsupported telephony audio capture methods

### Call Recording Legality

Call recording and monitoring laws vary by jurisdiction:

- some jurisdictions allow one-party consent (the user's consent is enough)
- some jurisdictions require two-party or all-party consent (the caller must also be informed)
- some countries ban call recording outright for consumers

The app must:

- clearly document which legal model it operates under
- show a disclaimer that the user is responsible for compliance with local laws
- optionally support an announcement mode where the app plays an audible notice that the call is being monitored
- for v1, target one-party consent jurisdictions and add multi-jurisdiction support later

## 13. Technical Risks

### Risk 1: Realtime latency is too high

Mitigation:

- reduce chunk size carefully
- lower overlap
- simplify prompts
- use a lighter realtime prompt than post-call prompt

### Risk 2: Mixed audio quality harms accuracy

Mitigation:

- test with speakerphone volume calibration
- use denoising and normalization
- benchmark under realistic room conditions

### Risk 3: Too many false alarms

Mitigation:

- add deterministic evidence thresholds
- require multiple suspicious indicators before raising a high-risk warning
- tune with legitimate bank calls

### Risk 4: Bank identification is wrong or missing

Mitigation:

- treat bank detection as best-effort, always keep generic detection as the fallback
- require high confidence before injecting bank-specific context
- if detected bank does not match user's actual bank, the generic advice is still safe
- let the user configure their bank in settings for higher accuracy

### Risk 5: Bank knowledge base becomes stale

Mitigation:

- include a last-updated timestamp per bank entry
- support periodic sync from a backend or bundled updates
- flag entries older than a configurable threshold as potentially outdated
- generic scam rules remain active regardless of knowledge base freshness

### Risk 6: Role inference is unreliable

Mitigation:

- keep it post-call only at first
- present it as "likely" rather than certain
- do not make the user-facing alert depend entirely on role labels

### Risk 7: Pipeline B audio input not supported on-device

Mitigation:

- verify Gemma 3n E2B audio input support in the AI Edge SDK before building Pipeline B
- if not supported, proceed with Pipeline A only and revisit when audio support is confirmed
- do not build the benchmark harness around Pipeline B until viability is confirmed

### Risk 8: Device thermal throttling during long calls

Mitigation:

- monitor device temperature during inference if the API is available
- define a thermal budget with two thresholds: warning and critical
- at warning threshold, reduce inference frequency (analyze every other chunk, or increase chunk interval)
- at critical threshold, switch to a minimal mode (stop realtime inference, continue recording for post-call analysis only)
- log thermal events for benchmark analysis
- consider using the NPU delegate preferentially over GPU or CPU for lower thermal output

### Risk 9: Model produces invalid or unparseable output

Mitigation:

- see Section 5.6 for parsing and validation strategy
- use lenient parsing with fallback defaults
- monitor parse failure rates and use them to drive prompt improvements
- keep prompts simple and use few-shot examples for format consistency

## 14. Language and Locale Strategy

Scams happen in every language. The app needs a language strategy even if v1 targets a single language.

### v1: Single language

- pick one target language for launch (likely English or the primary market language)
- ensure the Whisper model variant supports that language (Whisper large-v3 supports 99 languages; smaller variants may have weaker coverage for non-English)
- write all scam detection prompts in the target language
- bank knowledge base entries in the target language

### Future: Multi-language support

- Whisper handles language detection automatically and can transcribe in the detected language
- Gemma 3n E2B supports multiple languages for text input — verify which languages perform well at the int4 quantization level
- prompts may need per-language variants for best quality (scam patterns and phrasing differ by language and culture)
- bank knowledge base entries should be tagged with a locale
- UI strings need standard Android localization (strings.xml)

### Language-specific scam patterns

Some scam tactics are culture and language-specific:

- specific honorifics or authority language that varies by culture
- references to local payment systems or banking procedures
- impersonation of government agencies specific to a country
- urgency language that differs by culture

The evaluation dataset should include language-specific scam examples for each supported language.

## 15. Suggested Repo Roadmap

The current repository is minimal. A good near-term structure would be:

```text
android-app/
  app/
    src/main/java/.../ui
    src/main/java/.../audio
    src/main/java/.../inference
    src/main/java/.../benchmark
    src/main/java/.../reporting
  build.gradle.kts
benchmarks/
  sample_calls/
  labels/
  reports/
docs/
  prompts/
  schemas/
plan.md
```

If the existing Python project remains useful, it can support:

- offline dataset prep
- label tooling
- benchmark result analysis
- prompt iteration scripts

## 16. Immediate Next Steps

The next implementation steps should be:

1. Build the scam detection prompt in Python using the LiteRT-LM CLI wrapper — get reliable structured JSON output from transcript chunks.
2. Create a small evaluation dataset of scam, borderline, and legitimate call transcripts with ground-truth labels.
3. Build a Python eval harness that runs each sample through the chain and measures accuracy.
4. Add few-shot examples and tune the prompt until false negatives are low and JSON output is reliable.
5. Add Whisper ASR integration in Python (Pipeline A: audio file → Whisper → text → Gemma scam analysis).
6. Verify whether Gemma 3n E2B supports audio input in the LiteRT / AI Edge runtime (Pipeline B viability check).
7. Build the initial bank knowledge base from public sources and integrate RAG into the prompt.
8. Create the Android app skeleton and port the validated prompt and pipeline logic to Kotlin with the AI Edge SDK.

## 17. First Milestone Definition

The first milestone should be considered complete when:

- the Python prototype can analyze a transcript chunk and return valid structured JSON with risk score, evidence tags, and user advice
- a small eval dataset exists with labeled scam and non-scam samples
- precision and recall on the eval set are measured and acceptable
- Pipeline A (Whisper → text → Gemma) works end-to-end on audio files in Python
- Pipeline B viability has been confirmed or ruled out
- bank-specific RAG produces noticeably better advice than generic detection on at least two bank examples

The second milestone (Android) should be considered complete when:

- an Android app can load a recorded call
- both viable pipelines can analyze the same audio on-device
- the app shows a side-by-side result comparison
- latency and output are logged
- a simple post-call report is generated

This two-milestone approach lets model and prompt quality be validated cheaply in Python before committing to Android integration work.

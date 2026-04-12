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

### 5.6 Benchmark Layer

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

### Phase 4: Call Metadata Integration

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

### Phase 5: UX Hardening and Safety Guardrails

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
  "notes": "The speaker is pressuring for immediate action."
}
```

Suggested post-call result:

```json
{
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
  "recommended_action": "Do not transfer funds. Contact the bank using the official number."
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

## 11. Safety, Privacy, and Policy Requirements

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

## 12. Technical Risks

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

### Risk 4: Role inference is unreliable

Mitigation:

- keep it post-call only at first
- present it as "likely" rather than certain
- do not make the user-facing alert depend entirely on role labels

## 13. Suggested Repo Roadmap

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

## 14. Immediate Next Steps

The next implementation steps should be:

1. Create the Android app skeleton in this repo.
2. Define the shared JSON schema for realtime and post-call results.
3. Build the offline replay benchmark harness first.
4. Integrate Pipeline A and Pipeline B behind one interface.
5. Add a benchmark report screen before live call capture.
6. Add live speakerphone plus mic chunk capture.
7. Add post-call diarization and role inference.

## 15. First Milestone Definition

The first milestone should be considered complete when:

- an Android app can load a recorded call
- both pipelines can analyze the same audio
- the app shows a side-by-side result comparison
- latency and output are logged
- a simple post-call report is generated

That milestone gives a solid base for model and UX decisions before the more complex live-call integration work begins.

# Roadmap

## A. Clean production foundation — current
- Generic actions / intents / world state
- Planet adapter + input/capture contracts
- Perception contract + state builder
- Rule-based high-level policy + skill manager
- Stable runtime boundary

## B. Reuse-first low-level baseline — next
1. Benchmark existing game imitation implementations against Planet's dataset.
2. Prefer held/button-state action representation.
3. Integrate the selected implementation behind `policies.low_level.LowLevelPolicy`.
4. Do not modify `AgentRuntime` or the app API when the model changes.

Candidates to evaluate first:
- Microsoft imitation_learning_in_modern_video_games / PixelBC-style recurrent BC
- robomimic BC-RNN where its dataset/trainer infrastructure is useful
- VPT components where action mapping or sequence ideas transfer cleanly

## C. Runtime control
- Temporal executor / keyboard safety
- Live inference latency profiling
- Dry-run -> controlled live rollout

## D. Interactive improvement
- Reuse DAgger design / implementation where possible
- Planet-specific expert override bridge only

## E. Human-like refinement
Only after failure modes are measured. Avoid inventing algorithms without evidence.

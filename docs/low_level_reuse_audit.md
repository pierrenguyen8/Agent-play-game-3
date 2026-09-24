# Low-level policy reuse audit

## Decision

Use **Microsoft PixelBC** as the primary reference / source-reuse candidate for the first Planet low-level behavioural-cloning policy. Keep **robomimic BC-RNN** as a secondary training-framework reference and **OpenAI VPT** as an architecture / large-scale imitation reference, not as the first integration target.

The production contract remains ours:

`WorldState + Intent + Skill + frame -> LowLevelPolicy -> held controls -> PlanetControlCodec -> ControlStateAction -> PlanetAdapter`

This keeps Agent Core game-agnostic while the game-specific action ordering stays under `games/planet/`.

## Why PixelBC first

- MIT licensed.
- Built specifically for behavioural cloning in modern video games.
- Supports temporal LSTM policies trained on random trajectory subsequences.
- Separates visual encoder from policy network.
- Supports binary button-style actions, which maps naturally to Planet held key state.
- Includes Windows online rollout work for CS:GO.
- Uses PyTorch Lightning for checkpointing / training orchestration.

Upstream: https://github.com/microsoft/imitation_learning_in_modern_video_games

## Why not install PixelBC wholesale

Its package metadata pins a large environment (Torch, torchvision, NumPy, Lightning, video and vision libraries). Installing that dependency set directly into the existing Planet runtime environment creates unnecessary dependency-collision risk.

Instead:

1. Keep the upstream checkout under `external/pixelbc/` and record the exact commit.
2. Reuse only the policy / training components we actually need, preserving MIT attribution.
3. Put Planet dataset conversion, intent / skill conditioning, control ordering, and live inference behind our adapters.
4. Do not import MineRL or CS:GO environment code into Agent Core.

## Candidate comparison

### Microsoft PixelBC — primary

Use: LSTM policy structure, sequence training pattern, checkpoint / training orchestration, encoder experiments.

Adapt: Planet dataset reader, binary control layout, intent / skill conditioning, runtime wrapper.

### robomimic BC-RNN — secondary

Use as a reference for mature recurrent BC configuration, dataset sequence loading, recurrent rollout handling, and experiment hygiene. It supports LSTM / GRU BC-RNN but is fundamentally organized around robotics observation / action datasets, so adapting the full framework would introduce more robotics-specific structure than Planet needs.

Upstream: https://github.com/ARISE-Initiative/robomimic

### OpenAI VPT — reference only for now

Useful reference for large-scale gameplay imitation and action modeling. The public code is tightly coupled to MineRL / Minecraft and historically pinned an old PyTorch stack, so it is not the first implementation target for this Python 3.12 Windows project.

Upstream: https://github.com/openai/Video-Pre-Training

### video-game-behavioural-cloning — reference

Useful precedent for screen + keyboard behavioural cloning in multiple games and for game keyboard mappings. The repository is small and older; use it for ideas, not as the main maintained training stack.

Upstream: https://github.com/joonaspu/video-game-behavioural-cloning

## Action contract

The previous atomic `GenericAction` types remain for rule-based / manual logic. Learned policies need simultaneous controls, so `ControlStateAction` is the BC-friendly path. Example: movement and attack can be held in the same frame without inventing game-specific key codes in Agent Core.

Planet control order for the first model:

`left, right, up, down, jump, attack, interact, ability_0..N`

Contradictory directional predictions (`left + right`, `up + down`) decode to neutral for safety.

## Next implementation step

1. Fetch and record PixelBC upstream source.
2. Audit the exact LSTM policy and button-action loss implementation in the pinned checkout.
3. Build a Planet dataset exporter that produces sequences of frames plus held-state vectors.
4. Add `intent_id` and `skill_id` conditioning in a thin Planet / Agent adapter rather than changing Agent Core.
5. Train offline only.
6. Evaluate held-state precision / recall and sequence behaviour.
7. Plug the trained model into `LowLevelActionBridge` and test with `DryRunInputController` before any live keyboard output.

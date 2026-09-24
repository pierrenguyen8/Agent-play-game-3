# Migration from Agent-play-game-2

The old repository is a research archive. Do **not** copy its Python tree into this repo.

## Keep

- Raw Planet recordings (`data/raw`) — irreplaceable demonstrations.
- Final temporal dataset artifacts if needed for reproducible V1 benchmarking.
- Best perception weights (player/enemy/platform).
- `final_temporal_v1/best.pt` only as a legacy benchmark.
- Evaluation JSON / metrics that document old baselines.

## Do not migrate into production code

- `control_policy_v*`, `final`, `structured`, `fixed` variants.
- old `scripts/*` entrypoints.
- smoke scripts.
- duplicated dataset builders.
- V1/V2 training runners and losses.
- old generated `runs/` except selected best weights.

Git/history or the old repo remains the archive.

## Why the old low-level model is not in `src`

Its validation showed strong held-state prediction but weak direct press/release event F1.
It remains useful as a benchmark, not as the architecture contract of the new app.

## Safe migration order

1. Keep old repository untouched.
2. Run `tools/migration/migrate_from_old_repo.ps1`.
3. Run `pytest` in this repository.
4. Benchmark the legacy checkpoint from `artifacts/legacy/`.
5. Integrate the selected open-source BC implementation behind the stable policy interface.
6. Delete/archive old code only after the new runtime reproduces required behavior.

# Step 3 — Tracking, Safety Gate, Dry-Run Motor Seam

## Added
- Ultralytics persistent tracking (`persist=True`) with ByteTrack for Planet player/enemy YOLO adapters.
- Canonical `Detection.track_id`.
- `StabilizedPlanetPerceptionBackend` with a 3-frame default player grace window and confidence decay.
- `Intent.IDLE` / `SkillType.IDLE` safety path when the player is lost.
- `DiagnosticSkillActionPolicy` for motor-seam validation only.
- `tools/runtime/smoke_planet_dry_run.py` using `DryRunInputController`; it never sends real keyboard input.
- Player status reporting: `fresh`, `held`, or `lost`.

## Important behavior
- Short player detector dropout: reuse last player state temporarily (`held`).
- Player missing beyond grace window: `player=None -> IDLE -> StopAction`.
- EXPLORE/MOVE deliberately map to StopAction in the diagnostic policy; navigation behavior is not invented here.
- COMBAT/LOOT/UNSTUCK emit representative dry-run actions only. This is not the final low-level policy.

## Validation
- 13 tests passed.
- `compileall` passed for src/tools/tests.

## Run
```powershell
.\.venv\Scripts\python.exe .\tools\runtime\smoke_planet_observe.py --frames 60
.\.venv\Scripts\python.exe .\tools\runtime\smoke_planet_dry_run.py --frames 30
```

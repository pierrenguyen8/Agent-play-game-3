# Runtime milestone: Planet observation path

This milestone validates the production observation path before enabling motor control.

```text
Planet client window
    -> MSS capture (one frame)
    -> player YOLO
    -> enemy YOLO
    -> platform segmentation
    -> PlanetPerception
    -> PlanetStateBuilder
    -> WorldState
    -> RuleBasedHighLevelPolicy
    -> SkillManager
```

All three perception backends receive the exact same `CapturedFrame`. Third-party model objects stay behind adapters and do not leak into Agent Core.

## Runtime assets

Only selected best checkpoints are migrated from the old repository:

```text
artifacts/planet/perception/player/best.pt
artifacts/planet/perception/enemy/best.pt
artifacts/planet/perception/platform/best.pt
```

No old Python source, training runs, smoke scripts, or versioned policy implementations are copied.

## Local verification on Windows

Install perception/ML dependencies:

```powershell
.\setup_env.cmd -WithML
```

Copy the selected checkpoints from the old repository:

```powershell
.\tools\migration\migrate_runtime_assets.cmd "F:\tailieu\Agent-play-game-2"
```

Open Planet and keep its game window in the foreground, then run the read-only smoke test:

```powershell
python tools\runtime\smoke_planet_observe.py --frames 30
```

This smoke test does not send keyboard input. Motor control should only be enabled after this observation path is stable.

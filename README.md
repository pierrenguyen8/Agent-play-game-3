# Personal Agent

Clean, reuse-first successor to `Agent-play-game-2`.

## Rules of this repository

1. `src/` contains exactly one canonical production implementation.
2. Version numbers belong in configs/checkpoint metadata, not filenames/classes.
3. Existing open-source implementations are evaluated before new ML algorithms are written.
4. Planet-specific integration is ours; generic ML infrastructure should be reused where practical.
5. `training/`, `experiments/`, and `tools/` are never imported by production runtime.

## Current production boundary

```text
UI / future app
      |
      v
AgentRuntime
      |
      +-- Planet Perception Adapter
      +-- State Builder
      +-- Rule High-Level Policy
      +-- Skill Manager
      +-- Low-Level Policy interface   <-- open-source implementation plugs in here
      +-- PlanetAdapter
```

The old CNN/GRU temporal experiments are deliberately **not** copied into `src/`.
Their valuable datasets/checkpoints can be migrated as assets for benchmarking.

## Windows environment setup

Recommended (works even when PowerShell script execution is disabled):

```powershell
.\setup_env.cmd
```

The bootstrap prefers Python 3.12, falls back to 3.11, creates `.venv`, installs
the base Planet runtime + development dependencies, and runs the tests.

For the live Planet perception milestone, install the optional perception/ML stack:

```powershell
.\setup_env.cmd -WithML
```

To rebuild the environment from scratch:

```powershell
.\setup_env.cmd -Recreate
```

Activation is optional. You can always run the environment directly with:

```powershell
.\.venv\Scripts\python.exe <command>
```

## Step 2: migrate Planet runtime assets

The production runtime now has one canonical observation path:

```text
Planet client window
  -> one MSS frame
  -> player + enemy + platform perception
  -> PlanetPerception
  -> WorldState
  -> intent + skill
```

Copy only the three selected perception checkpoints from the old repository:

```powershell
.\tools\migration\migrate_runtime_assets.cmd "F:\tailieu\Agent-play-game-2"
```

Then open Planet, keep it in the foreground, and run the **read-only** smoke test:

```powershell
python tools\runtime\smoke_planet_observe.py --frames 30
```

No keyboard input is sent by that smoke test. See `docs/runtime_milestone.md`.

## Optional legacy benchmark assets

If old raw/processed datasets are needed later for benchmarking, the broader
migration tool remains available:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\migration\migrate_from_old_repo.ps1 `
    -OldRepo "F:\tailieu\Agent-play-game-2" `
    -IncludeProcessedData
```

See `docs/migration.md` before deleting the old repository.

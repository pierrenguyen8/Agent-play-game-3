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

## Setup

Recommended on Windows / VS Code:

```powershell
.\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

The setup script creates `.venv`, upgrades packaging tools, installs the
Planet runtime + development dependencies, and runs the test suite. VS Code
is configured to prefer `.venv\Scripts\python.exe`.

ML dependencies are intentionally optional while the reuse-first low-level
policy is being selected. Install them when needed:

```powershell
.\setup_env.ps1 -WithML
```

To rebuild the environment from scratch:

```powershell
.\setup_env.ps1 -Recreate
```

## Migrate useful assets from the old repository

```powershell
.\tools\migration\migrate_from_old_repo.ps1 `
    -OldRepo "F:\tailieu\Agent-play-game-2" `
    -IncludeProcessedData
```

See `docs/migration.md` before deleting the old repository.


## Windows environment setup

Recommended (works even when PowerShell script execution is disabled):

```powershell
.\setup_env.cmd
```

The bootstrap auto-detects Python 3.11 or 3.12 and creates `.venv`.
To include ML dependencies later:

```powershell
.\setup_env.cmd -WithML
```

You do not need to activate the environment to use it; you can call `.\.venv\Scripts\python.exe` directly.

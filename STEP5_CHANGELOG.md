# Step 5 — PixelBC dataset bridge

Changed / added only:

- Added generic BC sequence manifest / index types under `src/data/low_level/`.
- Added Planet raw-session exporter that maps `keys_down` snapshots to the canonical held-control vector.
- Added session-level train / validation splitting and capture-gap segmentation.
- Added lazy PyTorch sequence loader (`frames`, `held`, optional `intent_id`, `skill_id`).
- Added Windows-friendly `tools/training/export_planet_bc.cmd`.
- Added `tools/vendor/audit_pixelbc_source.*` to inspect the exact fetched PixelBC checkout without importing or installing it.
- Added Step 5 documentation and regression tests.

No model training and no real keyboard output are enabled by this step.

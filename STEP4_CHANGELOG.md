# Step 4 — reuse-first low-level policy integration contract

Changed / added only:

- Added `ControlStateAction` so learned policies can represent simultaneous controls.
- Updated `PlanetAdapter` to hold/release semantic controls safely.
- Added `PlanetControlCodec` for Planet binary held-state vectors.
- Added generic `LowLevelActionBridge` between reused BC policies and Agent Core actions.
- Added PixelBC source-fetch helper that records the exact upstream commit without installing its dependency stack.
- Added `docs/low_level_reuse_audit.md` documenting the reuse decision and boundaries.
- Added regression tests for simultaneous held controls and contradictory direction neutralization.

No real keyboard control is enabled by this step.

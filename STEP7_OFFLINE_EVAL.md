# Step 7 Offline Evaluation + Threshold Calibration

Added an offline evaluator for Planet low-level BC checkpoints.

- Evaluates the full validation split with the exact checkpoint model config.
- Reports precision / recall / F1 per held control at the default 0.50 threshold.
- Calibrates F1-maximizing thresholds only for the seven primary controls; sparse A/Z/X and interact remain at 0.50.
- Uses the seven primary controls (left/right/up/down/jump/attack/ability_0) for the primary mean F1.
- Keeps sparse A/Z/X and unused interact heads in the 11-control contract.
- Saves `thresholds.json` beside the checkpoint by default for later dry-run inference.
- Calibration metrics are explicitly labeled validation-calibrated; they are not an independent test estimate.
- No keyboard or live-control path is enabled.

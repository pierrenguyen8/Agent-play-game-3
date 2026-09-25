# Step 6 validation split + primary metric hotfix

## Changed

- Keep train/validation isolation at whole-session granularity.
- Select validation sessions by sequence-window count, targeting `val_fraction` of all windows instead of `val_fraction` of the session count.
- Export CLI now prints train and validation sequence counts immediately.
- Define the seven primary Planet controls for current evaluation:
  - left
  - right
  - up
  - down
  - jump (Alt)
  - attack (Ctrl)
  - ability_0 (Shift / teleport)
- Training logs and `metrics.jsonl` now report `val_primary_f1` separately from `val_mean_f1_all`.
- Run config and checkpoints record `primary_control_names` for later evaluation.
- A/Z/X and interact remain in the 11-output contract and training target; they simply do not dilute the primary F1 KPI.

## Safety

No runtime input behavior changes. Real keyboard control remains disabled by this patch.

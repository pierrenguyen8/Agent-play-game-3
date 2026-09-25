# Step 6 Planet bindings hotfix

Planet input mapping was aligned with the recorded dataset:

- jump: Alt (`alt`, accepting `alt_l` / `alt_r` through dataset aliases)
- attack: Ctrl (`ctrl`, accepting `ctrl_l` / `ctrl_r`)
- ability_0: Shift / teleport
- ability_1: A
- ability_2: Z
- ability_3: X
- interact: unbound for now; Z is no longer mislabeled as interact

After applying this patch, re-export `data/processed/planet/bc_pixelbc` before training. Existing smoke checkpoints trained with the old mapping should not be reused.

# Step 6 — PixelBC-aligned offline BC training

## Added
- `src/policies/low_level/pixelbc_lstm.py`
  - TorchVision ResNet-18 visual encoder.
  - PixelBC-described temporal layout: embedding projection -> LSTM -> action projection.
  - Uses PyTorch `nn.LSTM`; no custom recurrent algorithm.
- `src/policies/low_level/training_utils.py`
  - held-button class-balance statistics computed once per unique frame.
  - per-control precision/recall/F1/accuracy helpers.
- `tools/training/train_planet_bc.py` + `.cmd`
  - session-split train/validation loading.
  - weighted BCE for held controls.
  - AdamW, gradient clipping, optional CUDA AMP.
  - `best.pt`, `last.pt`, `metrics.jsonl`, `config.json`.
  - records PixelBC audit/commit provenance when available.
- `tests/test_planet_bc_training.py`

## Dependency
- `ml` extra now includes `torchvision>=0.17`.

## Scope
This step trains offline only. It does not execute keyboard input and does not yet wire a checkpoint into live inference.

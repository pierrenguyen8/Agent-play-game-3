# Step 5 — Planet sequence dataset for PixelBC reuse

## What is reused from PixelBC

PixelBC remains the primary upstream implementation target. Its public documentation defines the first integration shape we need:

- a visual encoder followed by a policy network;
- an LSTM policy for temporal context;
- training on trajectory subsequences;
- button-style action outputs;
- separate model / data pipeline concerns.

We do **not** install PixelBC into the Planet runtime environment. The upstream checkout stays under `external/pixelbc/`, and the exact fetched commit is recorded separately.

Step 5 therefore builds a narrow data adapter that can feed a pinned PixelBC training integration without making Planet raw recordings depend on PixelBC's MineRL / CS:GO formats.

## Canonical exported action target

Planet held controls are exported in this fixed order:

`left, right, up, down, jump, attack, interact, ability_0..N`

This is the same order used by `PlanetControlCodec` at runtime. The exporter maps each recorder snapshot's `input_state.keys_down` directly to a binary held vector. It intentionally does not recreate the old press/release/timing targets.

Modifier aliases are normalized (`ctrl`, `ctrl_l`, `ctrl_r`, etc.).

## Conditioning

Each exported frame also contains:

- `intent_id`
- `skill_id`
- `condition_mask`

Old raw recordings commonly do not contain intent / skill labels. Those rows use `-1` and `condition_mask=false`; the exporter does not fabricate labels. A later enrichment pass can add high-level labels before intent-conditioned training.

## Sequence safety

Sequences are generated only inside one recording session. They also break at:

- non-consecutive frame indices;
- non-monotonic timestamps;
- timestamp gaps larger than `gap_factor × nominal frame period`.

This prevents an LSTM training sample from silently spanning a pause, focus loss, or recording discontinuity.

Train / validation splits are made by **session**, never by adjacent frames, to reduce leakage.

## Export command

Example using old Planet recordings:

```powershell
.\tools\training\export_planet_bc.cmd `
  "D:\Agent\Agent-play-game-2\data\raw" `
  "data\processed\planet\bc_pixelbc" `
  --sequence-length 32 `
  --stride 8
```

Output:

```text
data/processed/planet/bc_pixelbc/
├─ manifest.json
├─ frames.jsonl
└─ windows.jsonl
```

The exported JSON metadata is small. Image files remain in the source recording tree and are loaded lazily by `TorchBCSequenceDataset`.

## Exact upstream source audit

After fetching PixelBC:

```powershell
.\tools\vendor\fetch_pixelbc.cmd
.\tools\vendor\audit_pixelbc_source.cmd
```

The audit script does not import PixelBC or install its dependencies. It records the checkout commit plus hashes / Python symbols for the two integration-critical source files:

- `pixelbc/models/bc_model.py`
- `pixelbc/models/utils/model_utils.py`

The resulting `external/pixelbc.audit.json` is the handoff artifact for the training adapter. This keeps the next integration based on the exact pinned upstream implementation rather than memory or a rewritten approximation.

## Next step

Step 6 consumes the exact audit + checkout and adds the smallest PixelBC training adapter needed for Planet button BC. It will keep training offline and will not enable real keyboard output.

## Video-backed Planet sessions

The exporter accepts both legacy PNG sessions and current `video_chunks`
sessions. For video-backed samples it preserves the recorder's canonical
`visual_ref` (`path` + frame index) instead of materializing PNG files.
`TorchBCSequenceDataset` decodes the referenced video frame lazily with OpenCV
and keeps a small per-worker video-handle cache. Causal timing still comes from
`samples.jsonl`; nominal video FPS is storage metadata only.

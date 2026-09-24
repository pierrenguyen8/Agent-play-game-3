# Step 5 video-session hotfix

- Accept Planet recorder `visual_storage=video_chunks` sessions.
- Preserve `visual_ref.path` and `visual_ref.frame_index`; no PNG materialization.
- Extend BC frame metadata with `visual_storage` / `visual_frame_index`.
- Lazily decode video frames in `TorchBCSequenceDataset` with a small handle cache.
- Keep backward compatibility with schema-v1 PNG exports; current schema is v2.
- Regression: 23 tests passed; compileall passed.

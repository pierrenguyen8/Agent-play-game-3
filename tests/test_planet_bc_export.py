from __future__ import annotations

import json
from pathlib import Path

from data.low_level import BCSequenceIndex
from games.planet.config import PlanetBindings
from games.planet.low_level.dataset_export import (
    PlanetBCExportConfig,
    _split_sessions,
    export_planet_bc_dataset,
    held_vector_from_keys,
)


def _write_session(
    root: Path,
    name: str,
    *,
    rows: list[dict],
    fps: float = 10.0,
) -> Path:
    session = root / name
    frames = session / "frames"
    frames.mkdir(parents=True)
    (session / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "game_id": "planet",
                "session_name": name,
                "target_fps": fps,
                "visual_storage": "png",
            }
        ),
        encoding="utf-8",
    )
    lines = []
    for row in rows:
        frame_index = int(row["frame_index"])
        frame_rel = f"frames/{frame_index:08d}.png"
        (session / frame_rel).write_bytes(b"png")
        value = {
            "schema_version": 1,
            "game_id": "planet",
            "frame_index": frame_index,
            "t_ns": int(row["t_ns"]),
            "capture_duration_ns": 1,
            "frame_path": frame_rel,
            "input_state": {"keys_down": row.get("keys_down", [])},
            "intent": row.get("intent"),
            "skill": row.get("skill"),
        }
        lines.append(json.dumps(value))
    (session / "samples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return session


def test_held_vector_uses_planet_order_and_modifier_aliases() -> None:
    held = held_vector_from_keys(
        ("right", "ctrl_l", "alt_l", "shift_l", "a", "z", "x"),
        bindings=PlanetBindings(),
        num_ability_slots=4,
    )
    assert held == (
        False,
        True,
        False,
        False,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
    )


def test_export_builds_session_split_and_conditions(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    step = 100_000_000
    for session_number in range(2):
        rows = []
        for index in range(6):
            rows.append(
                {
                    "frame_index": index,
                    "t_ns": index * step,
                    "keys_down": ["right"] if index < 3 else ["ctrl"],
                    "intent": "explore" if index < 3 else "combat",
                    "skill": "explore" if index < 3 else "combat",
                }
            )
        _write_session(raw, f"session_{session_number}", rows=rows)

    output = tmp_path / "exported"
    result = export_planet_bc_dataset(
        raw,
        output,
        config=PlanetBCExportConfig(
            sequence_length=3,
            stride=2,
            val_fraction=0.5,
            split_seed=1,
        ),
    )

    assert result.manifest.frame_count == 12
    assert result.manifest.sequence_count == 6
    assert len(result.manifest.train_sessions) == 1
    assert len(result.manifest.val_sessions) == 1
    assert result.manifest.control_names[:7] == (
        "left",
        "right",
        "up",
        "down",
        "jump",
        "attack",
        "interact",
    )

    train = BCSequenceIndex(output, split="train")
    val = BCSequenceIndex(output, split="val")
    assert len(train) == 3
    assert len(val) == 3
    first = train.sequence(0)
    assert len(first) == 3
    assert all(row.session_id == first[0].session_id for row in first)
    assert all(row.intent_id >= 0 and row.skill_id >= 0 for row in first)


def test_export_does_not_cross_large_capture_gap(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    # 10 FPS => nominal 100 ms; gap_factor=3 => 300 ms maximum.
    rows = [
        {"frame_index": 0, "t_ns": 0},
        {"frame_index": 1, "t_ns": 100_000_000},
        {"frame_index": 2, "t_ns": 200_000_000},
        {"frame_index": 3, "t_ns": 1_000_000_000},
        {"frame_index": 4, "t_ns": 1_100_000_000},
        {"frame_index": 5, "t_ns": 1_200_000_000},
    ]
    _write_session(raw, "gap_session", rows=rows, fps=10.0)

    output = tmp_path / "exported"
    result = export_planet_bc_dataset(
        raw,
        output,
        config=PlanetBCExportConfig(
            sequence_length=3,
            stride=1,
            val_fraction=0.0,
            gap_factor=3.0,
        ),
    )
    assert result.manifest.sequence_count == 2
    index = BCSequenceIndex(output)
    assert [tuple(row.frame_index for row in index.sequence(i)) for i in range(len(index))] == [
        (0, 1, 2),
        (3, 4, 5),
    ]


def test_missing_conditioning_is_explicit_unknown(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        {"frame_index": i, "t_ns": i * 100_000_000, "keys_down": ["left"]}
        for i in range(3)
    ]
    _write_session(raw, "unlabelled", rows=rows)
    output = tmp_path / "exported"
    export_planet_bc_dataset(
        raw,
        output,
        config=PlanetBCExportConfig(sequence_length=3, stride=1, val_fraction=0.0),
    )
    sequence = BCSequenceIndex(output).sequence(0)
    assert all(row.intent_id == -1 for row in sequence)
    assert all(row.skill_id == -1 for row in sequence)
    assert all(row.has_conditioning is False for row in sequence)


def _write_video_session(
    root: Path,
    name: str,
    *,
    frame_count: int = 4,
    fps: float = 10.0,
) -> Path:
    import cv2
    import numpy as np

    session = root / name
    video_dir = session / "video"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "chunk_00000.avi"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (16, 12),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[:, :, 2] = index * 50  # red after BGR->RGB
        writer.write(frame)
    writer.release()

    (session / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "game_id": "planet",
                "session_name": name,
                "target_fps": fps,
                "visual_storage": "video_chunks",
            }
        ),
        encoding="utf-8",
    )

    rows = []
    step = int(1_000_000_000 / fps)
    for index in range(frame_count):
        rows.append(
            json.dumps(
                {
                    "schema_version": 3,
                    "game_id": "planet",
                    "frame_index": index,
                    "t_ns": index * step,
                    "capture_duration_ns": 1,
                    "frame_path": None,
                    "visual_ref": {
                        "storage": "video",
                        "path": "video/chunk_00000.avi",
                        "frame_index": index,
                    },
                    "input_state": {
                        "keys_down": ["right"] if index < 2 else ["ctrl"]
                    },
                }
            )
        )
    (session / "samples.jsonl").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return session


def test_export_accepts_video_chunk_sessions(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_video_session(raw, "video_session", frame_count=4)

    output = tmp_path / "exported"
    result = export_planet_bc_dataset(
        raw,
        output,
        config=PlanetBCExportConfig(
            sequence_length=3,
            stride=1,
            val_fraction=0.0,
        ),
    )

    assert result.manifest.frame_count == 4
    index = BCSequenceIndex(output)
    first = index.sequence(0)
    assert all(row.visual_storage == "video" for row in first)
    assert [row.visual_frame_index for row in first] == [0, 1, 2]
    assert Path(first[0].frame_path).name == "chunk_00000.avi"


def test_torch_dataset_decodes_video_visual_refs(tmp_path: Path) -> None:
    from data.low_level import TorchBCSequenceDataset

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_video_session(raw, "video_session", frame_count=4)

    output = tmp_path / "exported"
    export_planet_bc_dataset(
        raw,
        output,
        config=PlanetBCExportConfig(
            sequence_length=3,
            stride=1,
            val_fraction=0.0,
        ),
    )

    dataset = TorchBCSequenceDataset(
        output,
        split="train",
        image_width=16,
        image_height=12,
    )
    try:
        sample = dataset[0]
    finally:
        dataset.close()

    assert tuple(sample["frames"].shape) == (3, 3, 12, 16)
    # Synthetic frames increase red intensity over time. MJPG is lossy, so only
    # check monotonic separation rather than exact pixel values.
    red_means = sample["frames"][:, 0].mean(dim=(1, 2))
    assert float(red_means[2]) > float(red_means[1]) > float(red_means[0])


def test_session_split_targets_sequence_fraction_not_session_count() -> None:
    sessions = ["long", "medium", "short_a", "short_b"]
    counts = {
        "long": 700,
        "medium": 200,
        "short_a": 50,
        "short_b": 50,
    }
    train, val = _split_sessions(
        sessions,
        sequence_counts=counts,
        val_fraction=0.2,
        seed=1337,
    )
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(sessions)
    val_sequences = sum(counts[name] for name in val)
    assert val_sequences == 200

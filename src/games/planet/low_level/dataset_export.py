from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

from agent_core.intent import Intent
from agent_core.skills import SkillType
from data.low_level import (
    BCDatasetManifest,
    BCFrameRecord,
    BCSequenceWindow,
    UNKNOWN_CONDITION_ID,
    write_jsonl,
)
from games.planet.config import PlanetBindings
from games.planet.low_level.control_codec import PlanetControlCodec


@dataclass(frozen=True, slots=True)
class PlanetBCExportConfig:
    sequence_length: int = 32
    stride: int = 8
    val_fraction: float = 0.2
    split_seed: int = 1337
    gap_factor: float = 3.0
    num_ability_slots: int = 4
    require_frame_files: bool = True

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.gap_factor <= 1.0:
            raise ValueError("gap_factor must be > 1")
        if self.num_ability_slots < 0:
            raise ValueError("num_ability_slots must be >= 0")


@dataclass(frozen=True, slots=True)
class PlanetBCExportResult:
    output_dir: Path
    manifest: BCDatasetManifest
    skipped_sessions: tuple[str, ...] = ()


_KEY_ALIASES: dict[str, frozenset[str]] = {
    "shift": frozenset(("shift", "shift_l", "shift_r")),
    "ctrl": frozenset(("ctrl", "ctrl_l", "ctrl_r")),
    "alt": frozenset(("alt", "alt_l", "alt_r")),
}


def _normalize_key(value: object) -> str:
    return str(value).strip().lower()


def _binding_is_down(keys_down: Iterable[object], binding: str) -> bool:
    keys = frozenset(_normalize_key(value) for value in keys_down)
    normalized = _normalize_key(binding)
    accepted = _KEY_ALIASES.get(normalized, frozenset((normalized,)))
    return bool(keys.intersection(accepted))


def held_vector_from_keys(
    keys_down: Iterable[object],
    *,
    bindings: PlanetBindings,
    num_ability_slots: int,
) -> tuple[bool, ...]:
    ability_keys = tuple(bindings.ability_keys[:num_ability_slots])
    if len(ability_keys) != num_ability_slots:
        raise ValueError(
            "PlanetBindings does not provide enough ability keys: "
            f"{len(bindings.ability_keys)} < {num_ability_slots}"
        )

    return (
        _binding_is_down(keys_down, bindings.left),
        _binding_is_down(keys_down, bindings.right),
        _binding_is_down(keys_down, bindings.up),
        _binding_is_down(keys_down, bindings.down),
        _binding_is_down(keys_down, bindings.jump),
        _binding_is_down(keys_down, bindings.attack),
        _binding_is_down(keys_down, bindings.interact),
        *(_binding_is_down(keys_down, key) for key in ability_keys),
    )


def _condition_id(value: object, names: tuple[str, ...]) -> int:
    if value is None:
        return UNKNOWN_CONDITION_ID
    normalized = str(value).strip().lower()
    try:
        return names.index(normalized)
    except ValueError:
        return UNKNOWN_CONDITION_ID


def _read_metadata(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "metadata.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_samples(session_dir: Path) -> list[dict[str, Any]]:
    path = session_dir / "samples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing samples.jsonl: {session_dir}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _resolve_visual_source(
    row: dict[str, Any],
    *,
    session_dir: Path,
    declared_storage: str,
    require_files: bool,
) -> tuple[str, Path, int | None]:
    """Resolve one recorder visual reference without materializing frames.

    PNG sessions use ``frame_path``. Video-chunk sessions use the canonical
    ``visual_ref`` object written by GameplayRecorder: storage/path/frame_index.
    ``frame_index`` is interpreted inside the referenced video file.
    """
    frame_path_value = row.get("frame_path")
    if frame_path_value:
        path = Path(str(frame_path_value))
        if not path.is_absolute():
            path = session_dir / path
        if require_files and not path.exists():
            raise FileNotFoundError(
                f"Frame referenced by {session_dir.name} does not exist: {path}"
            )
        return "image", path, None

    visual_ref = row.get("visual_ref")
    if not isinstance(visual_ref, dict):
        raise ValueError(
            f"Session {session_dir.name} sample has neither frame_path nor visual_ref"
        )

    storage = str(visual_ref.get("storage", "")).strip().lower()
    if storage not in ("image", "video"):
        raise ValueError(
            f"Session {session_dir.name} has unsupported visual_ref.storage={storage!r}"
        )

    path_value = visual_ref.get("path")
    if not path_value:
        raise ValueError(
            f"Session {session_dir.name} visual_ref.path must be non-empty"
        )
    path = Path(str(path_value))
    if not path.is_absolute():
        path = session_dir / path
    if require_files and not path.exists():
        raise FileNotFoundError(
            f"Visual source referenced by {session_dir.name} does not exist: {path}"
        )

    if storage == "image":
        return storage, path, None

    frame_index_value = visual_ref.get("frame_index")
    if frame_index_value is None:
        raise ValueError(
            f"Session {session_dir.name} video visual_ref is missing frame_index"
        )
    frame_index = int(frame_index_value)
    if frame_index < 0:
        raise ValueError(
            f"Session {session_dir.name} video frame_index must be >= 0"
        )

    if declared_storage not in ("video_chunks", "video", ""):
        raise ValueError(
            f"Session {session_dir.name} metadata declares {declared_storage!r} "
            "but sample uses video visual_ref"
        )
    return storage, path, frame_index


def _session_segments(
    rows: list[BCFrameRecord],
    *,
    target_fps: float | None,
    gap_factor: float,
) -> list[tuple[int, int]]:
    if not rows:
        return []
    if not target_fps or target_fps <= 0:
        max_gap_ns = None
    else:
        max_gap_ns = int((1_000_000_000 / target_fps) * gap_factor)

    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(rows)):
        previous = rows[index - 1]
        current = rows[index]
        frame_gap = current.frame_index - previous.frame_index
        time_gap = current.t_ns - previous.t_ns
        split = frame_gap != 1 or time_gap <= 0
        if max_gap_ns is not None and time_gap > max_gap_ns:
            split = True
        if split:
            segments.append((start, index))
            start = index
    segments.append((start, len(rows)))
    return segments


def _windows_for_segment(
    *,
    session_id: str,
    global_start: int,
    segment_length: int,
    sequence_length: int,
    stride: int,
) -> list[BCSequenceWindow]:
    if segment_length < sequence_length:
        return []
    last_start = segment_length - sequence_length
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [
        BCSequenceWindow(
            session_id=session_id,
            start_row=global_start + start,
            length=sequence_length,
        )
        for start in starts
    ]


def _split_sessions(
    session_ids: list[str],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered = sorted(session_ids)
    if len(ordered) <= 1 or val_fraction <= 0:
        return tuple(ordered), ()

    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction))
    val_count = min(val_count, len(shuffled) - 1)
    val_set = set(shuffled[:val_count])
    train = tuple(session for session in ordered if session not in val_set)
    val = tuple(session for session in ordered if session in val_set)
    return train, val


def export_planet_bc_dataset(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    config: PlanetBCExportConfig | None = None,
    bindings: PlanetBindings | None = None,
) -> PlanetBCExportResult:
    cfg = config or PlanetBCExportConfig()
    bindings = bindings or PlanetBindings()
    source_root = Path(source_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    codec = PlanetControlCodec(num_ability_slots=cfg.num_ability_slots)
    intent_names = tuple(intent.value for intent in Intent)
    skill_names = tuple(skill.value for skill in SkillType)

    session_dirs = sorted(
        path for path in source_root.iterdir()
        if path.is_dir() and (path / "samples.jsonl").exists()
    )
    if not session_dirs:
        raise ValueError(f"No Planet sessions with samples.jsonl found under {source_root}")

    all_frames: list[BCFrameRecord] = []
    all_windows: list[BCSequenceWindow] = []
    accepted_sessions: list[str] = []
    skipped_sessions: list[str] = []

    for session_dir in session_dirs:
        metadata = _read_metadata(session_dir)
        game_id = str(metadata.get("game_id", "planet")).lower()
        if game_id != "planet":
            skipped_sessions.append(session_dir.name)
            continue
        visual_storage = str(metadata.get("visual_storage", "png")).strip().lower()
        if visual_storage not in ("png", "", "video_chunks", "video"):
            raise ValueError(
                f"Session {session_dir.name} uses unsupported visual storage "
                f"{visual_storage!r}"
            )

        raw_rows = _read_samples(session_dir)
        session_frames: list[BCFrameRecord] = []
        for row in raw_rows:
            row_game_id = str(row.get("game_id", "planet")).lower()
            if row_game_id != "planet":
                continue
            (
                resolved_storage,
                frame_path,
                visual_frame_index,
            ) = _resolve_visual_source(
                row,
                session_dir=session_dir,
                declared_storage=visual_storage,
                require_files=cfg.require_frame_files,
            )

            input_state = row.get("input_state") or {}
            if not isinstance(input_state, dict):
                input_state = {}
            keys_down = input_state.get("keys_down") or ()

            session_frames.append(
                BCFrameRecord(
                    session_id=session_dir.name,
                    frame_index=int(row["frame_index"]),
                    t_ns=int(row["t_ns"]),
                    frame_path=str(frame_path),
                    held=held_vector_from_keys(
                        keys_down,
                        bindings=bindings,
                        num_ability_slots=cfg.num_ability_slots,
                    ),
                    intent_id=_condition_id(row.get("intent"), intent_names),
                    skill_id=_condition_id(row.get("skill"), skill_names),
                    visual_storage=resolved_storage,
                    visual_frame_index=visual_frame_index,
                )
            )

        session_frames.sort(key=lambda value: (value.frame_index, value.t_ns))
        if not session_frames:
            skipped_sessions.append(session_dir.name)
            continue

        global_start = len(all_frames)
        all_frames.extend(session_frames)
        target_fps_value = metadata.get("target_fps")
        target_fps = float(target_fps_value) if target_fps_value is not None else None
        for segment_start, segment_end in _session_segments(
            session_frames,
            target_fps=target_fps,
            gap_factor=cfg.gap_factor,
        ):
            all_windows.extend(
                _windows_for_segment(
                    session_id=session_dir.name,
                    global_start=global_start + segment_start,
                    segment_length=segment_end - segment_start,
                    sequence_length=cfg.sequence_length,
                    stride=cfg.stride,
                )
            )
        accepted_sessions.append(session_dir.name)

    if not all_frames:
        raise ValueError("No usable Planet frames were exported")
    if not all_windows:
        raise ValueError(
            "No sequence windows were produced. Record longer sessions or lower --sequence-length."
        )

    train_sessions, val_sessions = _split_sessions(
        accepted_sessions,
        val_fraction=cfg.val_fraction,
        seed=cfg.split_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "frames.jsonl", [row.to_dict() for row in all_frames])
    write_jsonl(output_dir / "windows.jsonl", [row.to_dict() for row in all_windows])

    manifest = BCDatasetManifest(
        game_id="planet",
        source_root=str(source_root),
        control_names=codec.control_names,
        intent_names=intent_names,
        skill_names=skill_names,
        sequence_length=cfg.sequence_length,
        stride=cfg.stride,
        gap_factor=cfg.gap_factor,
        frame_count=len(all_frames),
        sequence_count=len(all_windows),
        sessions=tuple(sorted(accepted_sessions)),
        train_sessions=train_sessions,
        val_sessions=val_sessions,
    )
    manifest.write(output_dir / "manifest.json")

    return PlanetBCExportResult(
        output_dir=output_dir,
        manifest=manifest,
        skipped_sessions=tuple(sorted(set(skipped_sessions))),
    )

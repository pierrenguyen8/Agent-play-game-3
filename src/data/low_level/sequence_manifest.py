from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


BC_DATASET_SCHEMA_VERSION = 2
SUPPORTED_BC_DATASET_SCHEMA_VERSIONS = frozenset((1, 2))
UNKNOWN_CONDITION_ID = -1


@dataclass(frozen=True, slots=True)
class BCFrameRecord:
    session_id: str
    frame_index: int
    t_ns: int
    frame_path: str
    held: tuple[bool, ...]
    intent_id: int = UNKNOWN_CONDITION_ID
    skill_id: int = UNKNOWN_CONDITION_ID
    visual_storage: str = "image"
    visual_frame_index: int | None = None

    def __post_init__(self) -> None:
        if self.visual_storage not in ("image", "video"):
            raise ValueError(
                "visual_storage must be 'image' or 'video': "
                f"{self.visual_storage!r}"
            )
        if not self.frame_path:
            raise ValueError("frame_path must be non-empty")
        if self.visual_storage == "video":
            if self.visual_frame_index is None or self.visual_frame_index < 0:
                raise ValueError(
                    "video frame records require non-negative visual_frame_index"
                )
        elif self.visual_frame_index is not None:
            raise ValueError(
                "image frame records must not define visual_frame_index"
            )

    @property
    def has_conditioning(self) -> bool:
        return self.intent_id >= 0 and self.skill_id >= 0

    @property
    def is_video(self) -> bool:
        return self.visual_storage == "video"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["held"] = [bool(value) for value in self.held]
        data["condition_mask"] = self.has_conditioning
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BCFrameRecord":
        # Schema v1 only had frame_path and therefore implicitly represented an
        # image. Defaults preserve compatibility with already-exported PNG data.
        visual_storage = str(data.get("visual_storage", "image"))
        visual_frame_index_value = data.get("visual_frame_index")
        visual_frame_index = (
            None
            if visual_frame_index_value is None
            else int(visual_frame_index_value)
        )
        return cls(
            session_id=str(data["session_id"]),
            frame_index=int(data["frame_index"]),
            t_ns=int(data["t_ns"]),
            frame_path=str(data["frame_path"]),
            held=tuple(bool(value) for value in data["held"]),
            intent_id=int(data.get("intent_id", UNKNOWN_CONDITION_ID)),
            skill_id=int(data.get("skill_id", UNKNOWN_CONDITION_ID)),
            visual_storage=visual_storage,
            visual_frame_index=visual_frame_index,
        )


@dataclass(frozen=True, slots=True)
class BCSequenceWindow:
    session_id: str
    start_row: int
    length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BCSequenceWindow":
        return cls(
            session_id=str(data["session_id"]),
            start_row=int(data["start_row"]),
            length=int(data["length"]),
        )


@dataclass(frozen=True, slots=True)
class BCDatasetManifest:
    game_id: str
    source_root: str
    control_names: tuple[str, ...]
    intent_names: tuple[str, ...]
    skill_names: tuple[str, ...]
    sequence_length: int
    stride: int
    gap_factor: float
    frame_count: int
    sequence_count: int
    sessions: tuple[str, ...]
    train_sessions: tuple[str, ...]
    val_sessions: tuple[str, ...]
    schema_version: int = BC_DATASET_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "control_names",
            "intent_names",
            "skill_names",
            "sessions",
            "train_sessions",
            "val_sessions",
        ):
            data[key] = list(data[key])
        return data

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> "BCDatasetManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = int(data.get("schema_version", 0))
        if version not in SUPPORTED_BC_DATASET_SCHEMA_VERSIONS:
            supported = sorted(SUPPORTED_BC_DATASET_SCHEMA_VERSIONS)
            raise ValueError(
                f"Unsupported BC dataset schema {version}; "
                f"supported={supported}"
            )
        return cls(
            schema_version=version,
            game_id=str(data["game_id"]),
            source_root=str(data["source_root"]),
            control_names=tuple(str(v) for v in data["control_names"]),
            intent_names=tuple(str(v) for v in data["intent_names"]),
            skill_names=tuple(str(v) for v in data["skill_names"]),
            sequence_length=int(data["sequence_length"]),
            stride=int(data["stride"]),
            gap_factor=float(data["gap_factor"]),
            frame_count=int(data["frame_count"]),
            sequence_count=int(data["sequence_count"]),
            sessions=tuple(str(v) for v in data["sessions"]),
            train_sessions=tuple(str(v) for v in data["train_sessions"]),
            val_sessions=tuple(str(v) for v in data["val_sessions"]),
        )


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class InputStateSample:
    keys_down: tuple[str, ...] = ()
    mouse_buttons_down: tuple[str, ...] = ()
    mouse_x: int | None = None
    mouse_y: int | None = None


@dataclass(frozen=True, slots=True)
class InputEventSample:
    timestamp_ns: int
    device: str
    event_type: str
    code: str | None = None
    x: int | None = None
    y: int | None = None


@dataclass(frozen=True, slots=True)
class GameplaySample:
    schema_version: int
    game_id: str
    frame_index: int
    t_ns: int
    capture_duration_ns: int
    frame_path: str | None
    input_state: InputStateSample
    intent: str | None = None
    skill: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

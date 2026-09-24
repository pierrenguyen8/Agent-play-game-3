from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True, slots=True)
class InputSnapshot:
    timestamp_ns: int
    keys_down: tuple[str, ...]
    mouse_buttons_down: tuple[str, ...] = ()
    mouse_x: int | None = None
    mouse_y: int | None = None


@dataclass(frozen=True, slots=True)
class InputEvent:
    timestamp_ns: int
    device: str
    event_type: str
    code: str | None
    x: int | None = None
    y: int | None = None


class InputStateTracker:
    def __init__(self) -> None:
        self._keys_down: set[str] = set()
        self._events: list[InputEvent] = []

    def key_down(self, key: str, *, timestamp_ns: int | None = None) -> None:
        if key in self._keys_down:
            return
        timestamp_ns = timestamp_ns or time.perf_counter_ns()
        self._keys_down.add(key)
        self._events.append(InputEvent(timestamp_ns, "keyboard", "key_down", key))

    def key_up(self, key: str, *, timestamp_ns: int | None = None) -> None:
        if key not in self._keys_down:
            return
        timestamp_ns = timestamp_ns or time.perf_counter_ns()
        self._keys_down.remove(key)
        self._events.append(InputEvent(timestamp_ns, "keyboard", "key_up", key))

    def snapshot(self, *, timestamp_ns: int | None = None) -> InputSnapshot:
        return InputSnapshot(
            timestamp_ns=timestamp_ns or time.perf_counter_ns(),
            keys_down=tuple(sorted(self._keys_down)),
        )

    def drain_events(self) -> list[InputEvent]:
        events = self._events
        self._events = []
        return events

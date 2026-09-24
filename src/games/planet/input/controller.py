from __future__ import annotations

from typing import Protocol


class InputController(Protocol):
    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...
    def tap(self, key: str) -> None: ...
    def release_all(self) -> None: ...


class DryRunInputController:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.keys_down: set[str] = set()

    def key_down(self, key: str) -> None:
        if key in self.keys_down:
            return
        self.keys_down.add(key)
        self.events.append(("down", key))

    def key_up(self, key: str) -> None:
        if key not in self.keys_down:
            return
        self.keys_down.remove(key)
        self.events.append(("up", key))

    def tap(self, key: str) -> None:
        self.events.append(("tap", key))

    def release_all(self) -> None:
        for key in sorted(self.keys_down):
            self.events.append(("up", key))
        self.keys_down.clear()


class PynputInputController:
    """Thin runtime adapter; dependency is loaded only when instantiated."""

    def __init__(self) -> None:
        try:
            from pynput.keyboard import Controller, Key
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'planet-runtime' extra to use live keyboard input"
            ) from exc
        self._controller = Controller()
        self._Key = Key
        self._held: set[str] = set()

    def _resolve(self, key: str):
        named = {
            "left": self._Key.left,
            "right": self._Key.right,
            "up": self._Key.up,
            "down": self._Key.down,
            "space": self._Key.space,
            "ctrl": self._Key.ctrl,
            "shift": self._Key.shift,
            "alt": self._Key.alt,
        }
        return named.get(key.lower(), key)

    def key_down(self, key: str) -> None:
        if key in self._held:
            return
        self._controller.press(self._resolve(key))
        self._held.add(key)

    def key_up(self, key: str) -> None:
        if key not in self._held:
            return
        self._controller.release(self._resolve(key))
        self._held.remove(key)

    def tap(self, key: str) -> None:
        resolved = self._resolve(key)
        self._controller.press(resolved)
        self._controller.release(resolved)

    def release_all(self) -> None:
        for key in tuple(self._held):
            self.key_up(key)

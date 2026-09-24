from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import sys


DEFAULT_WINDOW_TITLE = "MapleStory Worlds-MAPLE PLANET"


@dataclass(frozen=True, slots=True)
class PlanetWindow:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


def _require_windows() -> None:
    if sys.platform != "win32":
        raise RuntimeError("Planet window discovery is available only on Windows")


def _client_region(user32, hwnd: int) -> tuple[int, int, int, int] | None:
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None

    point = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
        return None

    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width <= 0 or height <= 0:
        return None
    return int(point.x), int(point.y), width, height


def find_planet_window(
    title_substring: str = DEFAULT_WINDOW_TITLE,
) -> PlanetWindow:
    """Find the largest visible, non-minimized matching client area."""

    _require_windows()
    user32 = ctypes.windll.user32
    matches: list[PlanetWindow] = []
    needle = title_substring.casefold()

    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @enum_proc
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, len(buffer))
        title = buffer.value
        if needle not in title.casefold():
            return True

        region = _client_region(user32, hwnd)
        if region is None:
            return True
        left, top, width, height = region
        matches.append(
            PlanetWindow(
                hwnd=int(hwnd),
                title=title,
                left=left,
                top=top,
                width=width,
                height=height,
            )
        )
        return True

    user32.EnumWindows(callback, 0)
    if not matches:
        raise RuntimeError(
            f"Planet window not found. Expected title containing: {title_substring!r}"
        )
    return max(matches, key=lambda item: item.width * item.height)


def refresh_planet_window(window: PlanetWindow) -> PlanetWindow | None:
    _require_windows()
    user32 = ctypes.windll.user32
    hwnd = window.hwnd
    if not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
        return None
    region = _client_region(user32, hwnd)
    if region is None:
        return None
    left, top, width, height = region
    return PlanetWindow(
        hwnd=hwnd,
        title=window.title,
        left=left,
        top=top,
        width=width,
        height=height,
    )


def is_window_recordable(hwnd: int) -> bool:
    _require_windows()
    user32 = ctypes.windll.user32
    return bool(
        user32.IsWindow(hwnd)
        and user32.IsWindowVisible(hwnd)
        and not user32.IsIconic(hwnd)
        and int(user32.GetForegroundWindow()) == int(hwnd)
    )

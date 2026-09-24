from __future__ import annotations

from dataclasses import dataclass

from games.planet.capture import CaptureRegion, MSSScreenCapture
from games.planet.perception import PlanetPerceptionBackend
from games.planet.window_finder import (
    DEFAULT_WINDOW_TITLE,
    find_planet_window,
    is_window_recordable,
    refresh_planet_window,
)
from runtime import AgentDecision, AgentRuntime


class PlanetNotReady(RuntimeError):
    """Raised when Planet should not be observed or controlled right now."""


@dataclass(frozen=True, slots=True)
class PlanetObservation:
    decision: AgentDecision
    timestamp_ns: int
    width: int
    height: int


class PlanetLiveObserver:
    """Read-only live bridge: Planet window -> one frame -> AgentRuntime.

    This milestone intentionally does not send keyboard input. It validates the
    full observation/state/decision path before motor control is enabled.
    """

    def __init__(
        self,
        *,
        perception: PlanetPerceptionBackend,
        window_title: str = DEFAULT_WINDOW_TITLE,
    ) -> None:
        self.window_title = window_title
        self.window = find_planet_window(window_title)
        self.capture = MSSScreenCapture(region=self._region(self.window))
        self.runtime = AgentRuntime(perception=perception)

    @staticmethod
    def _region(window) -> CaptureRegion:
        return CaptureRegion(
            left=window.left,
            top=window.top,
            width=window.width,
            height=window.height,
        )

    def observe_once(self) -> PlanetObservation:
        refreshed = refresh_planet_window(self.window)
        if refreshed is None:
            self.window = find_planet_window(self.window_title)
        else:
            self.window = refreshed

        if not is_window_recordable(self.window.hwnd):
            raise PlanetNotReady(
                "Planet must be visible, restored, and the foreground window."
            )

        self.capture.set_region(self._region(self.window))
        frame = self.capture.capture()
        decision = self.runtime.observe(frame)
        return PlanetObservation(
            decision=decision,
            timestamp_ns=frame.timestamp_ns,
            width=frame.width,
            height=frame.height,
        )

    def close(self) -> None:
        self.capture.close()

    def __enter__(self) -> "PlanetLiveObserver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True, slots=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    timestamp_ns: int
    capture_start_ns: int
    capture_end_ns: int
    width: int
    height: int
    bgra: bytes

    @property
    def capture_duration_ns(self) -> int:
        return self.capture_end_ns - self.capture_start_ns


class MSSScreenCapture:
    def __init__(self, region: CaptureRegion) -> None:
        try:
            import mss
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'planet-runtime' extra to use MSS capture"
            ) from exc
        self._mss = mss.mss()
        self.region = region

    def capture(self) -> CapturedFrame:
        start = time.perf_counter_ns()
        shot = self._mss.grab(
            {
                "left": self.region.left,
                "top": self.region.top,
                "width": self.region.width,
                "height": self.region.height,
            }
        )
        end = time.perf_counter_ns()
        return CapturedFrame(
            timestamp_ns=end,
            capture_start_ns=start,
            capture_end_ns=end,
            width=shot.width,
            height=shot.height,
            bgra=bytes(shot.bgra),
        )

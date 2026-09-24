from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Capture region width/height must be > 0")

    def as_mss_monitor(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True, init=False)
class CapturedFrame:
    """One screen acquisition on the monotonic perf-counter clock.

    MSS provides BGRA natively. Keep that buffer on the hot capture path and
    convert to packed RGB only when perception actually requests ``rgb``.
    """

    timestamp_ns: int
    capture_duration_ns: int
    capture_start_ns: int
    capture_end_ns: int
    width: int
    height: int
    _bgra: bytes | bytearray | memoryview | None = field(
        default=None, repr=False, compare=False
    )
    _rgb_cache: bytes | None = field(default=None, repr=False, compare=False)

    def __init__(
        self,
        *,
        timestamp_ns: int,
        capture_start_ns: int,
        capture_end_ns: int,
        width: int,
        height: int,
        capture_duration_ns: int | None = None,
        rgb: bytes | None = None,
        bgra: bytes | bytearray | memoryview | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Captured frame width/height must be > 0")
        if (rgb is None) == (bgra is None):
            raise ValueError("Provide exactly one of rgb or bgra")
        if capture_end_ns < capture_start_ns:
            raise ValueError("capture_end_ns must be >= capture_start_ns")

        duration = capture_end_ns - capture_start_ns
        if capture_duration_ns is not None and capture_duration_ns != duration:
            raise ValueError(
                "capture_duration_ns must equal capture_end_ns - capture_start_ns"
            )
        if not capture_start_ns <= timestamp_ns <= capture_end_ns:
            raise ValueError("timestamp_ns must lie inside the capture interval")

        if rgb is not None and len(rgb) != width * height * 3:
            raise ValueError("RGB byte count does not match width/height")
        if bgra is not None and len(bgra) != width * height * 4:
            raise ValueError("BGRA byte count does not match width/height")

        object.__setattr__(self, "timestamp_ns", timestamp_ns)
        object.__setattr__(self, "capture_duration_ns", duration)
        object.__setattr__(self, "capture_start_ns", capture_start_ns)
        object.__setattr__(self, "capture_end_ns", capture_end_ns)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "_bgra", bgra)
        object.__setattr__(self, "_rgb_cache", rgb)

    @property
    def rgb(self) -> bytes:
        cached = self._rgb_cache
        if cached is not None:
            return cached

        bgra = self._bgra
        if bgra is None:
            raise RuntimeError("Captured frame has no pixel buffer")

        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "Lazy BGRA -> RGB conversion requires numpy"
            ) from exc

        image = np.frombuffer(bgra, dtype=np.uint8).reshape(
            self.height, self.width, 4
        )
        rgb = image[:, :, [2, 1, 0]].tobytes()
        object.__setattr__(self, "_rgb_cache", rgb)
        return rgb

    @property
    def has_native_bgra(self) -> bool:
        return self._bgra is not None

    def save_png(self, path: str | Path) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "Saving frames requires opencv-python and numpy"
            ) from exc

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if self._bgra is not None:
            image = np.frombuffer(self._bgra, dtype=np.uint8).reshape(
                self.height, self.width, 4
            )
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            image = np.frombuffer(self.rgb, dtype=np.uint8).reshape(
                self.height, self.width, 3
            )
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not cv2.imwrite(
            str(output), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]
        ):
            raise RuntimeError(f"Failed to save frame: {output}")


class ScreenCaptureBackend(Protocol):
    def capture(self) -> CapturedFrame: ...
    def close(self) -> None: ...


class MSSScreenCapture:
    """MSS capture with lazy startup and lazy BGRA->RGB conversion."""

    def __init__(
        self,
        *,
        region: CaptureRegion | None = None,
        monitor_index: int = 1,
    ) -> None:
        self.region = region
        self.monitor_index = monitor_index
        self._sct = None

    def set_region(self, region: CaptureRegion) -> None:
        self.region = region

    def _ensure_started(self) -> None:
        if self._sct is not None:
            return
        try:
            import mss
        except ImportError as exc:
            raise RuntimeError(
                "Real screen capture requires 'mss'. "
                "Install with: python -m pip install -e '.[planet-runtime]'"
            ) from exc

        self._sct = mss.mss()
        if self.region is None and not (
            0 < self.monitor_index < len(self._sct.monitors)
        ):
            raise ValueError(
                f"Invalid monitor_index={self.monitor_index}; available: "
                f"1..{len(self._sct.monitors) - 1}"
            )

    def capture(self) -> CapturedFrame:
        self._ensure_started()
        assert self._sct is not None
        monitor = (
            self.region.as_mss_monitor()
            if self.region is not None
            else self._sct.monitors[self.monitor_index]
        )

        start_ns = time.perf_counter_ns()
        shot = self._sct.grab(monitor)
        end_ns = time.perf_counter_ns()

        return CapturedFrame(
            timestamp_ns=(start_ns + end_ns) // 2,
            capture_start_ns=start_ns,
            capture_end_ns=end_ns,
            width=shot.width,
            height=shot.height,
            bgra=shot.raw,
        )

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def __enter__(self) -> "MSSScreenCapture":
        self._ensure_started()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

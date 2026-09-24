from __future__ import annotations

from pathlib import Path

from games.planet.capture import CapturedFrame
from .detection import YOLODetector
from .types import Detection


class YOLOEnemyDetector:
    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.25,
        device: int | str | None = None,
        tracking: bool = True,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self._detector = YOLODetector(
            model_path,
            confidence=confidence,
            device=device,
            label="enemy",
            tracking=tracking,
            tracker=tracker,
        )

    def predict(self, frame: CapturedFrame) -> tuple[Detection, ...]:
        return self._detector.predict(frame)

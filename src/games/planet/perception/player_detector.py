from __future__ import annotations

from pathlib import Path

from games.planet.capture import CapturedFrame
from .detection import YOLODetector
from .types import Detection


class YOLOPlayerDetector:
    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.25,
        device: int | str | None = None,
    ) -> None:
        self._detector = YOLODetector(
            model_path,
            confidence=confidence,
            device=device,
            label="player",
        )

    def predict(self, frame: CapturedFrame) -> Detection | None:
        detections = self._detector.predict(frame)
        return detections[0] if detections else None

from __future__ import annotations

from pathlib import Path
from typing import Any

from games.planet.capture import CapturedFrame
from games.planet.perception.types import Box, Detection


class YOLODetector:
    """Thin adapter around an Ultralytics detection checkpoint.

    The rest of the agent only sees canonical ``Detection`` values. This keeps
    Ultralytics-specific result objects from leaking into Agent Core.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.25,
        device: int | str | None = None,
        label: str | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.device = device
        self.label = label
        self._model: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {self.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO perception requires the 'planet-perception' extra"
            ) from exc
        self._model = YOLO(str(self.model_path))

    def predict(self, frame: CapturedFrame) -> tuple[Detection, ...]:
        self._ensure_loaded()
        assert self._model is not None

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "YOLO perception requires Pillow via the 'planet-perception' extra"
            ) from exc

        image = Image.frombytes("RGB", (frame.width, frame.height), frame.rgb)
        results = self._model.predict(
            source=image,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        if not results:
            return ()

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return ()

        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        class_ids = (
            boxes.cls.detach().cpu().tolist()
            if getattr(boxes, "cls", None) is not None
            else [None] * len(xyxy)
        )
        names = getattr(results[0], "names", {}) or {}

        detections: list[Detection] = []
        for coords, score, class_id in zip(xyxy, confidences, class_ids):
            x1, y1, x2, y2 = (float(value) for value in coords[:4])
            x1 = max(0.0, min(x1, float(frame.width - 1)))
            y1 = max(0.0, min(y1, float(frame.height - 1)))
            x2 = max(x1 + 1.0, min(x2, float(frame.width)))
            y2 = max(y1 + 1.0, min(y2, float(frame.height)))

            label = self.label
            if label is None and class_id is not None:
                label = str(names.get(int(class_id), int(class_id)))

            detections.append(
                Detection(
                    box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=float(score),
                    label=label,
                )
            )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return tuple(detections)

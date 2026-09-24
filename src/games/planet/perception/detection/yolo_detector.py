from __future__ import annotations

from pathlib import Path
from typing import Any

from games.planet.capture import CapturedFrame
from games.planet.perception.types import Box, Detection


class YOLODetector:
    """Thin adapter around an Ultralytics detection checkpoint.

    Ultralytics-specific results stay inside this module. When ``tracking`` is
    enabled, consecutive frames use the selected tracker with ``persist=True``
    so stable track IDs can be exposed through canonical ``Detection`` values.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.25,
        device: int | str | None = None,
        label: str | None = None,
        tracking: bool = False,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.device = device
        self.label = label
        self.tracking = tracking
        self.tracker = tracker
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

        if self.tracking:
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError(
                    "YOLO tracking requires NumPy via the "
                    "'planet-perception' extra"
                ) from exc

            # Ultralytics interprets HWC uint8 NumPy inputs as BGR.
            # CapturedFrame.rgb is packed RGB, so reverse channels before
            # entering the tracking path. Copy to avoid a negative-stride view.
            rgb = np.frombuffer(frame.rgb, dtype=np.uint8).reshape(
                frame.height,
                frame.width,
                3,
            )
            image = rgb[:, :, ::-1].copy()
            results = self._model.track(
                source=image,
                conf=self.confidence,
                device=self.device,
                tracker=self.tracker,
                persist=True,
                verbose=False,
            )
        else:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError(
                    "YOLO perception requires Pillow via the "
                    "'planet-perception' extra"
                ) from exc

            # Preserve the proven Step-2 inference contract: PIL inputs are RGB.
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.rgb,
            )
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
        raw_track_ids = getattr(boxes, "id", None)
        track_ids = (
            raw_track_ids.detach().cpu().tolist()
            if raw_track_ids is not None
            else [None] * len(xyxy)
        )
        names = getattr(results[0], "names", {}) or {}

        detections: list[Detection] = []
        for coords, score, class_id, track_id in zip(
            xyxy,
            confidences,
            class_ids,
            track_ids,
        ):
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
                    track_id=(None if track_id is None else int(track_id)),
                )
            )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return tuple(detections)

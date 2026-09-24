from __future__ import annotations

from pathlib import Path

from games.planet.capture import CapturedFrame
from models.vision.segmentation import (
    build_model,
    image_to_tensor,
    letterbox,
    restore_letterbox_array,
)

from .platform_extractor import PlatformExtractorConfig, PlatformSurfaceExtractor
from .types import PlatformSurface


class PlatformDetector:
    """Adapter around the existing Planet platform segmentation checkpoint."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        self.requested_device = device
        self._model = None
        self._device = None
        self._input_width: int | None = None
        self._input_height: int | None = None
        self._extractor = PlatformSurfaceExtractor(
            PlatformExtractorConfig(threshold=self.threshold)
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Platform checkpoint not found: {self.model_path}")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Platform perception requires the 'ml' dependency group."
            ) from exc

        device = self.requested_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint = torch.load(self.model_path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise ValueError("Platform checkpoint must be a dictionary.")

        try:
            width = int(checkpoint["width"])
            height = int(checkpoint["height"])
            model_name = str(checkpoint["model_name"])
            base_channels = int(checkpoint["base_channels"])
            state_dict = checkpoint["model_state_dict"]
        except KeyError as exc:
            raise ValueError(
                f"Platform checkpoint missing required key: {exc.args[0]}"
            ) from exc

        model = build_model(model_name, base_channels=base_channels)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self._model = model
        self._device = device
        self._input_width = width
        self._input_height = height

    def predict(self, frame: CapturedFrame) -> tuple[PlatformSurface, ...]:
        if not isinstance(frame, CapturedFrame):
            raise TypeError("PlatformDetector expects a CapturedFrame.")
        self._load()
        assert self._model is not None
        assert self._device is not None
        assert self._input_width is not None
        assert self._input_height is not None

        try:
            import numpy as np
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Platform perception requires numpy, Pillow and torch."
            ) from exc

        image = Image.frombytes("RGB", (frame.width, frame.height), frame.rgb)
        prepared = letterbox(
            image,
            width=self._input_width,
            height=self._input_height,
        )
        tensor = image_to_tensor(prepared).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            logits = self._model(tensor)
            probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        restored = restore_letterbox_array(
            np.asarray(probability, dtype=np.float32),
            source_width=frame.width,
            source_height=frame.height,
        )
        return self._extractor.extract(restored)

from __future__ import annotations

from torch import nn

from .light_unet import LightUNet


def build_model(model_name: str, *, base_channels: int = 16) -> nn.Module:
    normalized = model_name.strip().lower()
    if normalized == "light_unet":
        return LightUNet(base_channels=base_channels)
    raise ValueError(f"Unsupported segmentation model: {model_name!r}")

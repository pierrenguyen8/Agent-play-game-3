from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .composed_backend import ComposedPlanetPerceptionBackend
from .enemy_detector import YOLOEnemyDetector
from .platform_detector import PlatformDetector
from .player_detector import YOLOPlayerDetector


@dataclass(frozen=True, slots=True)
class PlanetPerceptionModels:
    player: Path = Path("artifacts/planet/perception/player/best.pt")
    enemy: Path = Path("artifacts/planet/perception/enemy/best.pt")
    platform: Path = Path("artifacts/planet/perception/platform/best.pt")
    player_confidence: float = 0.25
    enemy_confidence: float = 0.25
    platform_threshold: float = 0.5


def build_planet_perception_backend(
    config: PlanetPerceptionModels | None = None,
    *,
    device: str | None = None,
) -> ComposedPlanetPerceptionBackend:
    config = config or PlanetPerceptionModels()

    if device is None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Live Planet perception requires the 'ml' dependency group."
            ) from exc
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        torch_device = device

    # Ultralytics accepts CUDA index 0 while the segmentation adapter expects
    # the regular torch device string.
    yolo_device: int | str = 0 if torch_device.startswith("cuda") else "cpu"

    return ComposedPlanetPerceptionBackend(
        player_backend=YOLOPlayerDetector(
            config.player,
            confidence=config.player_confidence,
            device=yolo_device,
        ),
        enemy_backend=YOLOEnemyDetector(
            config.enemy,
            confidence=config.enemy_confidence,
            device=yolo_device,
        ),
        platform_backend=PlatformDetector(
            config.platform,
            threshold=config.platform_threshold,
            device=torch_device,
        ),
    )

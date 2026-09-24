from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .composed_backend import ComposedPlanetPerceptionBackend
from .enemy_detector import YOLOEnemyDetector
from .platform_detector import PlatformDetector
from .player_detector import YOLOPlayerDetector
from .stabilizer import PerceptionStabilityConfig, StabilizedPlanetPerceptionBackend


@dataclass(frozen=True, slots=True)
class PlanetPerceptionModels:
    player: Path = Path("artifacts/planet/perception/player/best.pt")
    enemy: Path = Path("artifacts/planet/perception/enemy/best.pt")
    platform: Path = Path("artifacts/planet/perception/platform/best.pt")
    player_confidence: float = 0.25
    enemy_confidence: float = 0.25
    platform_threshold: float = 0.5
    tracker: str = "bytetrack.yaml"
    enemy_tracking: bool = False
    player_grace_frames: int = 3
    player_confidence_decay: float = 0.85


def build_planet_perception_backend(
    config: PlanetPerceptionModels | None = None,
    *,
    device: str | None = None,
) -> StabilizedPlanetPerceptionBackend:
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

    yolo_device: int | str = 0 if torch_device.startswith("cuda") else "cpu"

    raw_backend = ComposedPlanetPerceptionBackend(
        player_backend=YOLOPlayerDetector(
            config.player,
            confidence=config.player_confidence,
            device=yolo_device,
            tracking=True,
            tracker=config.tracker,
        ),
        enemy_backend=YOLOEnemyDetector(
            config.enemy,
            confidence=config.enemy_confidence,
            device=yolo_device,
            tracking=config.enemy_tracking,
            tracker=config.tracker,
        ),
        platform_backend=PlatformDetector(
            config.platform,
            threshold=config.platform_threshold,
            device=torch_device,
        ),
    )
    return StabilizedPlanetPerceptionBackend(
        raw_backend,
        PerceptionStabilityConfig(
            player_grace_frames=config.player_grace_frames,
            player_confidence_decay=config.player_confidence_decay,
        ),
    )

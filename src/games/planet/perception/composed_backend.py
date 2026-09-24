from __future__ import annotations

from typing import Protocol

from games.planet.capture import CapturedFrame
from .types import Detection, PlanetPerception, PlatformSurface


class PlayerBackend(Protocol):
    def predict(self, frame: CapturedFrame) -> Detection | None: ...


class EnemyBackend(Protocol):
    def predict(self, frame: CapturedFrame) -> tuple[Detection, ...]: ...


class PlatformBackend(Protocol):
    def predict(self, frame: CapturedFrame) -> tuple[PlatformSurface, ...]: ...


class ComposedPlanetPerceptionBackend:
    """Run all Planet perception backends against exactly one frame."""

    def __init__(
        self,
        *,
        player_backend: PlayerBackend,
        enemy_backend: EnemyBackend,
        platform_backend: PlatformBackend,
    ) -> None:
        self.player_backend = player_backend
        self.enemy_backend = enemy_backend
        self.platform_backend = platform_backend

    def predict(self, frame: object) -> PlanetPerception:
        if not isinstance(frame, CapturedFrame):
            raise TypeError("Planet perception expects CapturedFrame")

        player = self.player_backend.predict(frame)
        enemies = self.enemy_backend.predict(frame)
        platforms = self.platform_backend.predict(frame)

        return PlanetPerception(
            player=player,
            player_observed=player is not None,
            enemies=tuple(enemies),
            platforms=tuple(platforms),
            timestamp_ns=frame.timestamp_ns,
            frame_width=frame.width,
            frame_height=frame.height,
        )

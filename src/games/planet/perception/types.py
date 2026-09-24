from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    confidence: float


@dataclass(frozen=True, slots=True)
class PlatformSurface:
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class PlanetPerception:
    player: Detection | None = None
    enemies: tuple[Detection, ...] = ()
    platforms: tuple[PlatformSurface, ...] = ()
    loot: tuple[Detection, ...] = ()

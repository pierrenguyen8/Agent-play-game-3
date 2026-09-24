from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Box requires positive width and height")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    confidence: float
    label: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PlatformSurface:
    points: tuple[tuple[float, float], ...]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("PlatformSurface requires at least two points")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PlanetPerception:
    player: Detection | None = None
    enemies: tuple[Detection, ...] = ()
    platforms: tuple[PlatformSurface, ...] = ()
    loot: tuple[Detection, ...] = ()
    timestamp_ns: int | None = None
    frame_width: int | None = None
    frame_height: int | None = None

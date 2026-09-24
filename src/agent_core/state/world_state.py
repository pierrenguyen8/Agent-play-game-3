from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Position2D:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class EntityState:
    position: Position2D
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class PlatformState:
    points: tuple[Position2D, ...]


@dataclass(frozen=True, slots=True)
class WorldState:
    player: EntityState | None = None
    player_observed: bool = False
    enemies: tuple[EntityState, ...] = ()
    platforms: tuple[PlatformState, ...] = ()
    loot: tuple[EntityState, ...] = ()
    stuck: bool = False

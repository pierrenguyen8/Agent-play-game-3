from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlanetBindings:
    left: str = "left"
    right: str = "right"
    up: str = "up"
    down: str = "down"
    jump: str = "space"
    attack: str = "ctrl"
    interact: str = "z"
    ability_keys: tuple[str, ...] = ("1", "2", "3", "4")

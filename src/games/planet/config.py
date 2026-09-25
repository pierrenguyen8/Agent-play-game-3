from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlanetBindings:
    left: str = "left"
    right: str = "right"
    up: str = "up"
    down: str = "down"
    jump: str = "alt"
    attack: str = "ctrl"
    interact: str | None = None
    # Stable low-level slot order for Planet:
    # 0=teleport (Shift), 1=A skill, 2=Z skill, 3=X skill.
    ability_keys: tuple[str, ...] = ("shift", "a", "z", "x")

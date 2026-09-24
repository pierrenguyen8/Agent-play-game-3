from __future__ import annotations

from dataclasses import dataclass

from agent_core.actions import ControlStateAction


BASE_CONTROL_NAMES = (
    "left",
    "right",
    "up",
    "down",
    "jump",
    "attack",
    "interact",
)


@dataclass(frozen=True, slots=True)
class PlanetControlCodec:
    """Planet-specific glue between a binary BC action vector and Agent Core.

    The learned policy stays game-agnostic: it emits an ordered held-state
    vector. This codec owns Planet's semantic control ordering and converts the
    vector to a generic frame-level ControlStateAction.
    """

    num_ability_slots: int = 4

    def __post_init__(self) -> None:
        if self.num_ability_slots < 0:
            raise ValueError("num_ability_slots must be >= 0")

    @property
    def control_names(self) -> tuple[str, ...]:
        abilities = tuple(f"ability_{index}" for index in range(self.num_ability_slots))
        return BASE_CONTROL_NAMES + abilities

    @property
    def size(self) -> int:
        return len(self.control_names)

    def decode_held(self, held: tuple[bool, ...]) -> ControlStateAction:
        if len(held) != self.size:
            raise ValueError(
                f"Expected {self.size} held controls, got {len(held)}"
            )

        left, right, up, down, jump, attack, interact, *abilities = held

        # Contradictory directions are resolved to neutral instead of guessing.
        move_x = 0.0 if left == right else (-1.0 if left else 1.0)
        move_y = 0.0 if up == down else (-1.0 if up else 1.0)

        return ControlStateAction(
            move_x=move_x,
            move_y=move_y,
            jump=jump,
            attack=attack,
            interact=interact,
            abilities=tuple(abilities),
        )

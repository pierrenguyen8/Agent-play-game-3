from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class MoveAction:
    x: float


@dataclass(frozen=True, slots=True)
class JumpAction:
    pass


@dataclass(frozen=True, slots=True)
class AttackAction:
    pass


@dataclass(frozen=True, slots=True)
class InteractAction:
    pass


@dataclass(frozen=True, slots=True)
class AbilityAction:
    slot: int


@dataclass(frozen=True, slots=True)
class ControlStateAction:
    """Frame-level semantic control state for BC-style policies.

    Unlike the atomic actions above, this action can represent simultaneous
    controls such as moving right while attacking. The game adapter owns the
    mapping from these semantic controls to physical keys or buttons.
    """

    move_x: float = 0.0
    move_y: float = 0.0
    jump: bool = False
    attack: bool = False
    interact: bool = False
    abilities: tuple[bool, ...] = ()


@dataclass(frozen=True, slots=True)
class StopAction:
    pass


GenericAction: TypeAlias = (
    MoveAction
    | JumpAction
    | AttackAction
    | InteractAction
    | AbilityAction
    | ControlStateAction
    | StopAction
)

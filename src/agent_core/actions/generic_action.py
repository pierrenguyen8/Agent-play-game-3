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
class StopAction:
    pass


GenericAction: TypeAlias = (
    MoveAction
    | JumpAction
    | AttackAction
    | InteractAction
    | AbilityAction
    | StopAction
)

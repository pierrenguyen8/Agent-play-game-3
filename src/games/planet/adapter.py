from __future__ import annotations

from agent_core.actions import (
    AbilityAction,
    AttackAction,
    GenericAction,
    InteractAction,
    JumpAction,
    MoveAction,
    StopAction,
)
from games.planet.config import PlanetBindings
from games.planet.input import InputController


class PlanetAdapter:
    game_id = "planet"

    def __init__(self, controller: InputController, bindings: PlanetBindings | None = None) -> None:
        self.controller = controller
        self.bindings = bindings or PlanetBindings()

    def execute(self, action: GenericAction) -> None:
        if isinstance(action, MoveAction):
            self._move(action.x)
        elif isinstance(action, JumpAction):
            self.controller.tap(self.bindings.jump)
        elif isinstance(action, AttackAction):
            self.controller.tap(self.bindings.attack)
        elif isinstance(action, InteractAction):
            self.controller.tap(self.bindings.interact)
        elif isinstance(action, AbilityAction):
            self.controller.tap(self.bindings.ability_keys[action.slot])
        elif isinstance(action, StopAction):
            self.stop_all()
        else:
            raise TypeError(f"Unsupported action: {type(action)!r}")

    def _move(self, x: float) -> None:
        if x > 0:
            self.controller.key_up(self.bindings.left)
            self.controller.key_down(self.bindings.right)
        elif x < 0:
            self.controller.key_up(self.bindings.right)
            self.controller.key_down(self.bindings.left)
        else:
            self.controller.key_up(self.bindings.left)
            self.controller.key_up(self.bindings.right)

    def stop_all(self) -> None:
        self.controller.release_all()

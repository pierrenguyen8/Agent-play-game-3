from __future__ import annotations

from agent_core.actions import (
    AbilityAction,
    AttackAction,
    ControlStateAction,
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
        if isinstance(action, ControlStateAction):
            self._apply_control_state(action)
        elif isinstance(action, MoveAction):
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

    def _set_held(self, key: str, held: bool) -> None:
        if held:
            self.controller.key_down(key)
        else:
            self.controller.key_up(key)

    def _move(self, x: float) -> None:
        self._set_held(self.bindings.left, x < 0)
        self._set_held(self.bindings.right, x > 0)

    def _apply_control_state(self, action: ControlStateAction) -> None:
        self._set_held(self.bindings.left, action.move_x < 0)
        self._set_held(self.bindings.right, action.move_x > 0)
        self._set_held(self.bindings.up, action.move_y < 0)
        self._set_held(self.bindings.down, action.move_y > 0)
        self._set_held(self.bindings.jump, action.jump)
        self._set_held(self.bindings.attack, action.attack)
        self._set_held(self.bindings.interact, action.interact)

        if len(action.abilities) > len(self.bindings.ability_keys):
            raise ValueError("Control state has more ability slots than Planet bindings")
        for index, key in enumerate(self.bindings.ability_keys):
            held = action.abilities[index] if index < len(action.abilities) else False
            self._set_held(key, held)

    def stop_all(self) -> None:
        self.controller.release_all()

from __future__ import annotations

from typing import Protocol

from agent_core.actions import GenericAction


class GameAdapter(Protocol):
    game_id: str

    def execute(self, action: GenericAction) -> None:
        ...

    def stop_all(self) -> None:
        ...

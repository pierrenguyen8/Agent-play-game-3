from __future__ import annotations

from agent_core.intent import Intent

from .types import SkillType


class SkillManager:
    _MAP = {
        Intent.EXPLORE: SkillType.EXPLORE,
        Intent.MOVE: SkillType.MOVE,
        Intent.COMBAT: SkillType.COMBAT,
        Intent.LOOT: SkillType.LOOT,
        Intent.UNSTUCK: SkillType.UNSTUCK,
    }

    def select(self, intent: Intent) -> SkillType:
        return self._MAP[intent]

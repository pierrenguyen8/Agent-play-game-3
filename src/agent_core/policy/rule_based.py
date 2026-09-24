from __future__ import annotations

from agent_core.intent import Intent
from agent_core.state import WorldState


class RuleBasedHighLevelPolicy:
    """Small, inspectable high-level baseline.

    Keep this deterministic until measurements show that a learned high-level
    policy is actually needed.
    """

    def decide(self, state: WorldState) -> Intent:
        if state.stuck:
            return Intent.UNSTUCK
        if state.enemies:
            return Intent.COMBAT
        if state.loot:
            return Intent.LOOT
        return Intent.EXPLORE

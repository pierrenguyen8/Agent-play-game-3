from __future__ import annotations

from dataclasses import dataclass

from agent_core.intent import Intent
from agent_core.policy import RuleBasedHighLevelPolicy
from agent_core.skills import SkillManager, SkillType
from agent_core.state import WorldState
from games.planet.perception import PlanetPerceptionBackend
from games.planet.state_builder import PlanetStateBuilder


@dataclass(frozen=True, slots=True)
class AgentDecision:
    state: WorldState
    intent: Intent
    skill: SkillType


class AgentRuntime:
    """Production orchestration boundary used later by CLI/desktop app.

    Low-level control is intentionally a separate plugin seam so an existing
    BC implementation can be integrated without changing the app surface.
    """

    def __init__(
        self,
        *,
        perception: PlanetPerceptionBackend,
        state_builder: PlanetStateBuilder | None = None,
        high_level_policy: RuleBasedHighLevelPolicy | None = None,
        skill_manager: SkillManager | None = None,
    ) -> None:
        self.perception = perception
        self.state_builder = state_builder or PlanetStateBuilder()
        self.high_level_policy = high_level_policy or RuleBasedHighLevelPolicy()
        self.skill_manager = skill_manager or SkillManager()

    def observe(self, frame: object, *, stuck: bool = False) -> AgentDecision:
        perception = self.perception.predict(frame)
        state = self.state_builder.build(perception, stuck=stuck)
        intent = self.high_level_policy.decide(state)
        skill = self.skill_manager.select(intent)
        return AgentDecision(state=state, intent=intent, skill=skill)

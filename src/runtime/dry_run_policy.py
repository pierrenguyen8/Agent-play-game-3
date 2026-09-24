from __future__ import annotations

from agent_core.actions import (
    AttackAction,
    GenericAction,
    InteractAction,
    JumpAction,
    StopAction,
)
from agent_core.skills import SkillType

from .agent_runtime import AgentDecision


class DiagnosticSkillActionPolicy:
    """Temporary action stub used only to validate the motor-control seam.

    It intentionally avoids inventing navigation behavior. EXPLORE/MOVE remain
    STOP until a real low-level policy is integrated. Other skills emit a
    representative action so the dry-run controller path can be inspected.
    """

    def decide(self, decision: AgentDecision) -> GenericAction:
        if decision.skill is SkillType.COMBAT:
            return AttackAction()
        if decision.skill is SkillType.LOOT:
            return InteractAction()
        if decision.skill is SkillType.UNSTUCK:
            return JumpAction()
        return StopAction()

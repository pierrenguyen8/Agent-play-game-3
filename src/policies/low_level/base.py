from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent_core.intent import Intent
from agent_core.skills import SkillType
from agent_core.state import WorldState


@dataclass(frozen=True, slots=True)
class LowLevelObservation:
    frame: object
    world_state: WorldState
    intent: Intent
    skill: SkillType
    previous_held: tuple[bool, ...] = ()


@dataclass(frozen=True, slots=True)
class LowLevelPrediction:
    held: tuple[bool, ...]
    confidence: tuple[float, ...] = ()


class LowLevelPolicy(Protocol):
    """Stable production seam for reused BC/RNN/VLM implementations."""

    def predict(self, observation: LowLevelObservation) -> LowLevelPrediction:
        ...

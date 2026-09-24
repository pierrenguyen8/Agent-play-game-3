from __future__ import annotations

from typing import Protocol

from agent_core.actions import GenericAction

from .base import LowLevelObservation, LowLevelPolicy, LowLevelPrediction


class HeldActionCodec(Protocol):
    def decode_held(self, held: tuple[bool, ...]) -> GenericAction:
        ...


class LowLevelActionBridge:
    """Thin adapter from a reused BC policy to the Agent Core action contract."""

    def __init__(self, policy: LowLevelPolicy, codec: HeldActionCodec) -> None:
        self.policy = policy
        self.codec = codec

    def predict(self, observation: LowLevelObservation) -> tuple[LowLevelPrediction, GenericAction]:
        prediction = self.policy.predict(observation)
        action = self.codec.decode_held(prediction.held)
        return prediction, action

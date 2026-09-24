from __future__ import annotations

from dataclasses import dataclass, replace

from .backend import PlanetPerceptionBackend
from .types import Detection, PlanetPerception


@dataclass(frozen=True, slots=True)
class PerceptionStabilityConfig:
    player_grace_frames: int = 3
    player_confidence_decay: float = 0.85

    def __post_init__(self) -> None:
        if self.player_grace_frames < 0:
            raise ValueError("player_grace_frames must be >= 0")
        if not 0.0 < self.player_confidence_decay <= 1.0:
            raise ValueError("player_confidence_decay must be in (0, 1]")


class StabilizedPlanetPerceptionBackend:
    """Temporal wrapper for short detector dropouts.

    Tracking remains delegated to Ultralytics. This wrapper only keeps the last
    reliable player box for a small number of missed frames, with confidence
    decay, then deliberately drops the player so the safety gate can stop the
    agent.
    """

    def __init__(
        self,
        backend: PlanetPerceptionBackend,
        config: PerceptionStabilityConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or PerceptionStabilityConfig()
        self._last_player: Detection | None = None
        self._player_missed_frames = 0

    def predict(self, frame: object) -> PlanetPerception:
        raw = self.backend.predict(frame)
        if raw.player is not None:
            self._last_player = raw.player
            self._player_missed_frames = 0
            return replace(raw, player_observed=True)

        self._player_missed_frames += 1
        if (
            self._last_player is not None
            and self._player_missed_frames <= self.config.player_grace_frames
        ):
            confidence = self._last_player.confidence * (
                self.config.player_confidence_decay ** self._player_missed_frames
            )
            held_player = replace(
                self._last_player,
                confidence=max(0.0, min(1.0, confidence)),
            )
            return replace(raw, player=held_player, player_observed=False)

        self._last_player = None
        return replace(raw, player=None, player_observed=False)

    def reset(self) -> None:
        self._last_player = None
        self._player_missed_frames = 0

from __future__ import annotations

from typing import Protocol

from .types import PlanetPerception


class PlanetPerceptionBackend(Protocol):
    def predict(self, frame: object) -> PlanetPerception:
        ...

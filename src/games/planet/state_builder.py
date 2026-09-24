from __future__ import annotations

from agent_core.state import EntityState, PlatformState, Position2D, WorldState
from games.planet.perception import PlanetPerception


class PlanetStateBuilder:
    def build(self, perception: PlanetPerception, *, stuck: bool = False) -> WorldState:
        def entity(detection):
            x, y = detection.box.center
            return EntityState(Position2D(x, y), detection.confidence)

        player = None if perception.player is None else entity(perception.player)
        enemies = tuple(entity(item) for item in perception.enemies)
        loot = tuple(entity(item) for item in perception.loot)
        platforms = tuple(
            PlatformState(tuple(Position2D(x, y) for x, y in surface.points))
            for surface in perception.platforms
        )
        return WorldState(
            player=player,
            enemies=enemies,
            platforms=platforms,
            loot=loot,
            stuck=stuck,
        )

from __future__ import annotations

from games.planet.capture import CapturedFrame
from games.planet.perception.composed_backend import ComposedPlanetPerceptionBackend
from games.planet.perception.types import Box, Detection, PlatformSurface
from runtime import AgentRuntime


def make_frame() -> CapturedFrame:
    # Native BGRA pixel: B=10, G=20, R=30, A=255 -> RGB=(30,20,10)
    return CapturedFrame(
        timestamp_ns=15,
        capture_start_ns=10,
        capture_end_ns=20,
        width=1,
        height=1,
        bgra=bytes((10, 20, 30, 255)),
    )


def test_captured_frame_converts_bgra_lazily() -> None:
    frame = make_frame()
    assert frame.has_native_bgra
    assert frame.rgb == bytes((30, 20, 10))
    assert frame.capture_duration_ns == 10


def test_composed_perception_uses_one_canonical_frame() -> None:
    frame = make_frame()
    seen: list[object] = []

    class Player:
        def predict(self, value):
            seen.append(value)
            return Detection(Box(0, 0, 1, 1), 0.9, "player")

    class Enemy:
        def predict(self, value):
            seen.append(value)
            return (Detection(Box(0, 0, 1, 1), 0.8, "enemy"),)

    class Platform:
        def predict(self, value):
            seen.append(value)
            return (PlatformSurface(((0.0, 0.0), (1.0, 0.0)), 0.7),)

    backend = ComposedPlanetPerceptionBackend(
        player_backend=Player(),
        enemy_backend=Enemy(),
        platform_backend=Platform(),
    )
    perception = backend.predict(frame)

    assert seen == [frame, frame, frame]
    assert perception.timestamp_ns == frame.timestamp_ns
    assert perception.frame_width == 1
    assert perception.frame_height == 1
    assert len(perception.enemies) == 1
    assert len(perception.platforms) == 1


def test_agent_runtime_accepts_composed_planet_perception() -> None:
    frame = make_frame()

    class Player:
        def predict(self, _frame):
            return Detection(Box(0, 0, 1, 1), 0.9, "player")

    class Enemy:
        def predict(self, _frame):
            return ()

    class Platform:
        def predict(self, _frame):
            return (PlatformSurface(((0.0, 0.0), (1.0, 0.0)), 0.8),)

    runtime = AgentRuntime(
        perception=ComposedPlanetPerceptionBackend(
            player_backend=Player(),
            enemy_backend=Enemy(),
            platform_backend=Platform(),
        )
    )
    decision = runtime.observe(frame)

    assert decision.state.player is not None
    assert len(decision.state.platforms) == 1
    assert decision.intent.value == "explore"
    assert decision.skill.value == "explore"

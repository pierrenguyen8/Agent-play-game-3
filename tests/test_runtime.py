from agent_core.intent import Intent
from games.planet.perception import Box, Detection, PlanetPerception
from runtime import AgentRuntime


class FakePerception:
    def predict(self, frame):
        return PlanetPerception(
            player=Detection(Box(0, 0, 10, 10), 0.99),
            enemies=(Detection(Box(20, 0, 30, 10), 0.9),),
        )


def test_runtime_production_boundary():
    runtime = AgentRuntime(perception=FakePerception())
    result = runtime.observe(object())
    assert result.intent == Intent.COMBAT
    assert result.state.player is not None

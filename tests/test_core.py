from agent_core.actions import JumpAction, MoveAction, StopAction
from agent_core.intent import Intent
from agent_core.policy import RuleBasedHighLevelPolicy
from agent_core.state import EntityState, Position2D, WorldState
from games.planet import PlanetAdapter
from games.planet.input import DryRunInputController, InputStateTracker
from policies.low_level import decode_held_transition


def test_input_tracker_ignores_repeat_down():
    tracker = InputStateTracker()
    tracker.key_down("right", timestamp_ns=10)
    tracker.key_down("right", timestamp_ns=11)
    tracker.key_up("right", timestamp_ns=12)
    assert [(e.event_type, e.code) for e in tracker.drain_events()] == [
        ("key_down", "right"),
        ("key_up", "right"),
    ]


def test_planet_adapter_dry_run():
    controller = DryRunInputController()
    adapter = PlanetAdapter(controller)
    adapter.execute(MoveAction(1.0))
    adapter.execute(JumpAction())
    adapter.execute(StopAction())
    assert ("down", "right") in controller.events
    assert ("tap", "space") in controller.events
    assert controller.keys_down == set()


def test_held_transition_decoder():
    result = decode_held_transition(
        (False, True, False),
        (True, False, False),
    )
    assert result.press == (True, False, False)
    assert result.release == (False, True, False)


def test_rule_policy_is_debuggable_baseline():
    policy = RuleBasedHighLevelPolicy()
    assert policy.decide(WorldState()) == Intent.EXPLORE
    enemy = EntityState(Position2D(10, 20))
    assert policy.decide(WorldState(enemies=(enemy,))) == Intent.COMBAT
    assert policy.decide(WorldState(stuck=True)) == Intent.UNSTUCK

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
    assert policy.decide(WorldState()) == Intent.IDLE
    enemy = EntityState(Position2D(10, 20))
    assert policy.decide(WorldState(enemies=(enemy,))) == Intent.IDLE
    player = EntityState(Position2D(0, 0))
    assert policy.decide(WorldState(player=player, enemies=(enemy,))) == Intent.COMBAT
    assert policy.decide(WorldState(player=player, stuck=True)) == Intent.UNSTUCK


def test_planet_control_codec_supports_simultaneous_controls():
    from games.planet.low_level import PlanetControlCodec

    codec = PlanetControlCodec(num_ability_slots=2)
    action = codec.decode_held(
        (False, True, False, False, True, True, False, True, False)
    )
    assert action.move_x == 1.0
    assert action.jump is True
    assert action.attack is True
    assert action.abilities == (True, False)


def test_planet_control_codec_neutralizes_direction_conflicts():
    from games.planet.low_level import PlanetControlCodec

    codec = PlanetControlCodec(num_ability_slots=0)
    action = codec.decode_held((True, True, True, True, False, False, False))
    assert action.move_x == 0.0
    assert action.move_y == 0.0


def test_planet_adapter_control_state_holds_combined_inputs():
    from agent_core.actions import ControlStateAction

    controller = DryRunInputController()
    adapter = PlanetAdapter(controller)
    adapter.execute(ControlStateAction(move_x=1.0, jump=True, attack=True))
    assert controller.keys_down == {"right", "space", "ctrl"}

    # Repeating the same frame-level state must not generate repeat key-downs.
    event_count = len(controller.events)
    adapter.execute(ControlStateAction(move_x=1.0, jump=True, attack=True))
    assert len(controller.events) == event_count

    adapter.execute(ControlStateAction())
    assert controller.keys_down == set()

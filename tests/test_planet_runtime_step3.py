from __future__ import annotations

from agent_core.intent import Intent
from agent_core.skills import SkillType
from games.planet.input import DryRunInputController
from games.planet.perception import Box, Detection, PlanetPerception
from games.planet.perception.stabilizer import (
    PerceptionStabilityConfig,
    StabilizedPlanetPerceptionBackend,
)
from runtime import AgentDecision, DiagnosticSkillActionPolicy
from agent_core.actions import AttackAction, StopAction
from agent_core.state import EntityState, Position2D, WorldState


class SequenceBackend:
    def __init__(self, values):
        self.values = iter(values)

    def predict(self, _frame):
        return next(self.values)


def test_player_short_dropout_is_held_then_released() -> None:
    player = Detection(Box(10, 10, 20, 20), 0.9, "player", track_id=7)
    backend = StabilizedPlanetPerceptionBackend(
        SequenceBackend(
            [
                PlanetPerception(player=player, player_observed=True),
                PlanetPerception(),
                PlanetPerception(),
                PlanetPerception(),
            ]
        ),
        PerceptionStabilityConfig(
            player_grace_frames=2,
            player_confidence_decay=0.5,
        ),
    )

    fresh = backend.predict(object())
    held_1 = backend.predict(object())
    held_2 = backend.predict(object())
    lost = backend.predict(object())

    assert fresh.player_observed is True
    assert fresh.player is not None
    assert held_1.player_observed is False
    assert held_1.player is not None
    assert held_1.player.track_id == 7
    assert held_1.player.confidence == 0.45
    assert held_2.player is not None
    assert held_2.player.confidence == 0.225
    assert lost.player is None


def test_safety_gate_idles_when_player_is_lost() -> None:
    from agent_core.policy import RuleBasedHighLevelPolicy

    policy = RuleBasedHighLevelPolicy()
    enemy = EntityState(Position2D(10, 10))

    assert policy.decide(WorldState(enemies=(enemy,))) is Intent.IDLE
    assert policy.decide(
        WorldState(
            player=EntityState(Position2D(0, 0)),
            enemies=(enemy,),
        )
    ) is Intent.COMBAT


def test_diagnostic_motor_policy_stops_on_idle_and_attacks_on_combat() -> None:
    policy = DiagnosticSkillActionPolicy()

    idle = AgentDecision(WorldState(), Intent.IDLE, SkillType.IDLE)
    combat = AgentDecision(
        WorldState(player=EntityState(Position2D(0, 0))),
        Intent.COMBAT,
        SkillType.COMBAT,
    )

    assert isinstance(policy.decide(idle), StopAction)
    assert isinstance(policy.decide(combat), AttackAction)


def test_dry_run_controller_never_requires_real_keyboard() -> None:
    controller = DryRunInputController()
    controller.key_down("right")
    controller.key_down("right")
    controller.tap("ctrl")
    controller.release_all()

    assert controller.events == [
        ("down", "right"),
        ("tap", "ctrl"),
        ("up", "right"),
    ]


def test_yolo_detector_tracking_branch_preserves_track_id() -> None:
    from games.planet.capture import CapturedFrame
    from games.planet.perception.detection.yolo_detector import YOLODetector

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    class FakeBoxes:
        xyxy = FakeTensor([[0.0, 0.0, 1.0, 1.0]])
        conf = FakeTensor([0.9])
        cls = FakeTensor([0.0])
        id = FakeTensor([42.0])

        def __len__(self):
            return 1

    class FakeResult:
        boxes = FakeBoxes()
        names = {0: "player"}

    class FakeModel:
        def __init__(self):
            self.kwargs = None

        def track(self, **kwargs):
            self.kwargs = kwargs
            return [FakeResult()]

    model = FakeModel()
    detector = YOLODetector(
        "unused.pt",
        tracking=True,
        tracker="bytetrack.yaml",
        label="player",
    )
    detector._model = model

    frame = CapturedFrame(
        timestamp_ns=2,
        capture_start_ns=1,
        capture_end_ns=2,
        width=1,
        height=1,
        bgra=bytes((0, 0, 0, 255)),
    )
    detections = detector.predict(frame)

    assert model.kwargs["persist"] is True
    assert model.kwargs["tracker"] == "bytetrack.yaml"
    assert detections[0].track_id == 42


def test_yolo_detector_preserves_rgb_for_predict_and_bgr_for_track() -> None:
    from games.planet.capture import CapturedFrame
    from games.planet.perception.detection.yolo_detector import YOLODetector

    class EmptyBoxes:
        def __len__(self):
            return 0

    class FakeResult:
        boxes = EmptyBoxes()
        names = {}

    class FakeModel:
        def __init__(self):
            self.predict_source = None
            self.track_source = None

        def predict(self, **kwargs):
            self.predict_source = kwargs["source"]
            return [FakeResult()]

        def track(self, **kwargs):
            self.track_source = kwargs["source"]
            return [FakeResult()]

    # Native BGRA=(10,20,30,255) -> RGB=(30,20,10).
    frame = CapturedFrame(
        timestamp_ns=2,
        capture_start_ns=1,
        capture_end_ns=2,
        width=1,
        height=1,
        bgra=bytes((10, 20, 30, 255)),
    )

    predict_model = FakeModel()
    predict_detector = YOLODetector("unused.pt", tracking=False)
    predict_detector._model = predict_model
    predict_detector.predict(frame)
    assert predict_model.predict_source.getpixel((0, 0)) == (30, 20, 10)

    track_model = FakeModel()
    track_detector = YOLODetector("unused.pt", tracking=True)
    track_detector._model = track_model
    track_detector.predict(frame)
    assert track_model.track_source[0, 0].tolist() == [10, 20, 30]

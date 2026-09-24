from __future__ import annotations

import argparse
import time

from games.planet import PlanetAdapter
from games.planet.input import DryRunInputController
from games.planet.live_runtime import PlanetLiveObserver, PlanetNotReady
from games.planet.perception.factory import build_planet_perception_backend
from runtime import DiagnosticSkillActionPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run Planet control smoke test. Computes representative actions "
            "but never sends real keyboard input."
        )
    )
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--interval", type=float, default=0.05)
    return parser.parse_args()


def _player_status(state) -> str:
    if state.player is None:
        return "lost"
    return "fresh" if state.player_observed else "held"


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be > 0")

    backend = build_planet_perception_backend()
    controller = DryRunInputController()
    adapter = PlanetAdapter(controller)
    action_policy = DiagnosticSkillActionPolicy()

    with PlanetLiveObserver(perception=backend) as observer:
        completed = 0
        print("Planet DRY RUN ready. No real keyboard input will be sent.")
        while completed < args.frames:
            try:
                observation = observer.observe_once()
            except PlanetNotReady as exc:
                print(f"PAUSED: {exc}")
                time.sleep(0.25)
                continue

            decision = observation.decision
            action = action_policy.decide(decision)
            event_start = len(controller.events)
            adapter.execute(action)
            new_events = controller.events[event_start:]

            print(
                f"[{completed + 1:03d}/{args.frames:03d}] "
                f"player={_player_status(decision.state)} "
                f"intent={decision.intent.value} skill={decision.skill.value} "
                f"action={type(action).__name__} events={new_events}"
            )
            completed += 1
            if args.interval > 0:
                time.sleep(args.interval)

    adapter.stop_all()


if __name__ == "__main__":
    main()

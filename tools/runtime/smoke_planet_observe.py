from __future__ import annotations

import argparse
import time

from games.planet.live_runtime import PlanetLiveObserver, PlanetNotReady
from games.planet.perception.factory import build_planet_perception_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only smoke test for Planet capture -> perception -> WorldState."
    )
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--interval", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be > 0")

    backend = build_planet_perception_backend()
    with PlanetLiveObserver(perception=backend) as observer:
        completed = 0
        print("Planet live observation ready. Keep Planet in the foreground.")
        while completed < args.frames:
            try:
                observation = observer.observe_once()
            except PlanetNotReady as exc:
                print(f"PAUSED: {exc}")
                time.sleep(0.25)
                continue

            decision = observation.decision
            state = decision.state
            player = "yes" if state.player is not None else "no"
            print(
                f"[{completed + 1:03d}/{args.frames:03d}] "
                f"{observation.width}x{observation.height} "
                f"player={player} enemies={len(state.enemies)} "
                f"platforms={len(state.platforms)} "
                f"intent={decision.intent.value} skill={decision.skill.value}"
            )
            completed += 1
            if args.interval > 0:
                time.sleep(args.interval)


if __name__ == "__main__":
    main()

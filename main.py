from games.planet import PlanetAdapter
from games.planet.input import DryRunInputController
from agent_core.actions import JumpAction, MoveAction, StopAction


def main() -> None:
    controller = DryRunInputController()
    adapter = PlanetAdapter(controller)
    adapter.execute(MoveAction(1.0))
    adapter.execute(JumpAction())
    adapter.execute(StopAction())
    print("Personal Agent clean bootstrap: OK")
    print("Dry-run events:", controller.events)


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeldTransition:
    press: tuple[bool, ...]
    release: tuple[bool, ...]
    held: tuple[bool, ...]


def decode_held_transition(
    previous: tuple[bool, ...],
    current: tuple[bool, ...],
) -> HeldTransition:
    if len(previous) != len(current):
        raise ValueError("Held-state dimensions must match")
    press = tuple((not before) and after for before, after in zip(previous, current))
    release = tuple(before and (not after) for before, after in zip(previous, current))
    return HeldTransition(press=press, release=release, held=current)

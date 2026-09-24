from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    IDLE = "idle"
    EXPLORE = "explore"
    MOVE = "move"
    COMBAT = "combat"
    LOOT = "loot"
    UNSTUCK = "unstuck"

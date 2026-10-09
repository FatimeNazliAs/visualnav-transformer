"""What every closed-loop policy offers the episode runner: `reset(meta)`, then `act(obs) -> (v, w)` per tick.

Phase 1 plugs in NoMaD (nomad_policy.py); Phase 5 adds MapMaD and the map planner, which read extra inputs
(local map, the robot's own pose) from Observation.extras. Never put the geodesic distance or the target's
position into a policy's inputs: those belong to the runner and the metrics only.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Observation:
    """What the robot knows at one control step."""

    frames: List[np.ndarray]  # camera pictures (H, W, 3) uint8, oldest first; at most Policy.context_frames
    goal_image: Optional[np.ndarray]  # goal photo (H, W, 3) uint8, or None = goal masked (explore)
    step: int
    extras: Dict[str, Any] = field(default_factory=dict)  # e.g. local map (later phases); never target / geodesic


@dataclass
class Command:
    """Velocity for the next control period, plus anything worth logging (JSON-friendly values)."""

    v: float  # m/s, forward
    w: float  # rad/s, positive = turn left
    info: Dict[str, Any] = field(default_factory=dict)


class Policy(ABC):
    """A driver. `reset` starts an episode, `act` is called once per control period."""

    name = "policy"
    context_frames = 1  # how many past pictures the runner keeps for it

    @abstractmethod
    def reset(self, meta: Dict[str, Any]) -> None:
        """Start an episode. meta: {"seed": int (seeds all randomness), "episode_id": str, "goal": "photo"|"masked"}."""

    @abstractmethod
    def act(self, obs: Observation) -> Command:
        ...

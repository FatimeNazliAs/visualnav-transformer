"""Phase 5 arms (configs/p5_exam.yaml `arms`): what each arm's policy is given and which runs it has. Pure Python, so
both containers use the same rules (runner, gate, videos, S-freeze estimate).

    arm = ExamArm.from_config("C", cfg["arms"]["C"])
    arm.outputs            # what the simulator server sends with every frame
    arm.seed_indices([0, 1, 2])   # the planner (deterministic) runs index 0 only
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ExamArm:
    """One arm: which policy, which goal inputs, which server outputs."""

    name: str
    policy: str  # nomad | mapmad | planner
    photo: bool = False
    map: bool = False
    waypoint_index: Optional[int] = None  # MapMaD arms: overrides `mapmad.waypoint_index` (dev waypoint test)
    hfov_deg: float = 66.5

    @classmethod
    def from_config(cls, name: str, spec: Dict[str, Any]) -> "ExamArm":
        return cls(name=name, **spec)

    @property
    def outputs(self) -> Dict[str, Any]:
        """What the server sends with every frame (all arms get the depth camera, so a home is never reloaded
        between arms)."""
        return {"depth": True, "map": self.map_given, "grid": self.policy == "planner"}

    @property
    def map_given(self) -> bool:
        """The policy receives the map tensor (MapMaD with the map goal)."""
        return self.policy == "mapmad" and self.map

    @property
    def deterministic(self) -> bool:
        """The planner has no noise: its seed-index-0 run counts for every seed index."""
        return self.policy == "planner"

    def seed_indices(self, seeds: Sequence[int]) -> List[int]:
        """The seed indices this arm actually runs."""
        return [k for k in seeds if not self.deterministic or k == 0]

    def run_seed_index(self, seed_index: int) -> int:
        """The seed index of the run that stands for `seed_index` (the planner: always 0)."""
        return 0 if self.deterministic else seed_index

    @property
    def reset_goal(self) -> str:
        """The goal named to the policy at episode start: the photo for photo arms, otherwise masked."""
        return "photo" if self.photo else "masked"

    @property
    def mapmad_goal(self) -> str:
        """MapMaD's goal input: the map, the photo (map masked), or both masked."""
        return "map" if self.map else ("photo" if self.photo else "masked")

    @property
    def weights_key(self) -> Optional[str]:
        """Which frozen weights hash this arm's logs carry (`nomad` / `mapmad`; the planner has none)."""
        return self.policy if self.policy in ("nomad", "mapmad") else None


def arms_of(cfg: Dict[str, Any], names: Optional[Sequence[str]] = None) -> List[ExamArm]:
    """The config's arms (all, or these names in this order)."""
    return [ExamArm.from_config(n, cfg["arms"][n]) for n in (names or list(cfg["arms"]))]

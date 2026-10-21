"""MapMaD training/eval modes (Phase 3 confirmation item 8; v2: Phase 10 block item 2): which goal inputs a sample
shows.

Each mode fixes three things: is the photo goal shown, is the map shown, and is the heat layer on (v2: the "target
seen here" layer). Masks follow NoMaD's convention: 1 = hidden (masked), 0 = shown. In v2 both map windows (near,
far) are shown or hidden together in training (no near-/far-only training mode).

| id | name        | dataset     | photo  | map    | heat / target | v1 share | v2 share (drives)      |
|----|-------------|-------------|--------|--------|---------------|----------|------------------------|
| 0  | photo       | Habitat     | shown  | hidden | -             | 20%      | 15% (all)              |
| 1  | map_goal    | Habitat     | hidden | shown  | on            | 30%      | -                      |
| 2  | photo_map   | Habitat     | shown  | shown  | on            | 15%      | -                      |
| 3  | explore_map | Habitat     | hidden | shown  | off           | 15%      | 20% (exploration only) |
| 4  | explore     | Habitat     | hidden | hidden | -             | 20%      | 20% (all)              |
| 5  | gs_photo    | GoStanford  | shown  | hidden | -             | 50/50    | 50/50                  |
| 6  | gs_explore  | GoStanford  | hidden | hidden | -             |          |                        |
| 7  | search_map  | Habitat     | hidden | shown  | on (noisy)    | -        | 45% (search only)      |
"""

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class Mode:
    id: int
    name: str
    goal_mask: int  # 1 = photo hidden
    map_mask: int  # 1 = map hidden
    heat_on: bool  # v1: heat layer drawn; v2: target layer kept (else all zero)


MODES = (
    Mode(0, "photo", 0, 1, False),
    Mode(1, "map_goal", 1, 0, True),
    Mode(2, "photo_map", 0, 0, True),
    Mode(3, "explore_map", 1, 0, False),
    Mode(4, "explore", 1, 1, False),
    Mode(5, "gs_photo", 0, 1, False),
    Mode(6, "gs_explore", 1, 1, False),
    Mode(7, "search_map", 1, 0, True),
)
BY_NAME: Dict[str, Mode] = {m.name: m for m in MODES}
HABITAT_MODES = tuple(m.name for m in MODES[:5])  # v1
HABITAT_MODES_V2 = ("search_map", "explore_map", "explore", "photo")


def habitat_mode_table(shares: Dict[str, float], names: Sequence[str] = HABITAT_MODES) -> np.ndarray:
    """Probabilities in `names` order from the config's `habitat_modes` (must cover all names and sum to 1)."""
    if set(shares) != set(names):
        raise ValueError(f"habitat_modes must name exactly {tuple(names)}, got {sorted(shares)}")
    p = np.array([float(shares[n]) for n in names])
    if abs(p.sum() - 1.0) > 1e-6:
        raise ValueError(f"habitat_modes shares sum to {p.sum()}, not 1")
    return p


def draw_habitat_mode(p: Sequence[float], rng: np.random.Generator, names: Sequence[str] = HABITAT_MODES) -> Mode:
    """One Habitat mode drawn with probabilities p (`names` order)."""
    return BY_NAME[names[int(rng.choice(len(names), p=p))]]


def draw_gostanford_mode(goal_mask_prob: float, rng: np.random.Generator) -> Mode:
    """GoStanford: map always hidden; photo hidden with goal_mask_prob (NoMaD's 50/50)."""
    return BY_NAME["gs_explore"] if rng.random() < goal_mask_prob else BY_NAME["gs_photo"]

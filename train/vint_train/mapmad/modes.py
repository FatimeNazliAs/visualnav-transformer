"""MapMaD training/eval modes (Phase 3 confirmation item 8): which goal inputs a sample shows.

Each mode fixes three things: is the photo goal shown, is the map shown, and is the heat layer on.
Masks follow NoMaD's convention: 1 = hidden (masked), 0 = shown.

| id | name        | dataset     | photo  | map    | heat |
|----|-------------|-------------|--------|--------|------|
| 0  | photo       | Habitat     | shown  | hidden | -    |
| 1  | map_goal    | Habitat     | hidden | shown  | on   |
| 2  | photo_map   | Habitat     | shown  | shown  | on   |
| 3  | explore_map | Habitat     | hidden | shown  | off  |
| 4  | explore     | Habitat     | hidden | hidden | -    |
| 5  | gs_photo    | GoStanford  | shown  | hidden | -    |
| 6  | gs_explore  | GoStanford  | hidden | hidden | -    |
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
    heat_on: bool


MODES = (
    Mode(0, "photo", 0, 1, False),
    Mode(1, "map_goal", 1, 0, True),
    Mode(2, "photo_map", 0, 0, True),
    Mode(3, "explore_map", 1, 0, False),
    Mode(4, "explore", 1, 1, False),
    Mode(5, "gs_photo", 0, 1, False),
    Mode(6, "gs_explore", 1, 1, False),
)
BY_NAME: Dict[str, Mode] = {m.name: m for m in MODES}
HABITAT_MODES = tuple(m.name for m in MODES[:5])


def habitat_mode_table(shares: Dict[str, float]) -> np.ndarray:
    """Probabilities in HABITAT_MODES order from the config's `habitat_modes` (must cover all 5 and sum to 1)."""
    if set(shares) != set(HABITAT_MODES):
        raise ValueError(f"habitat_modes must name exactly {HABITAT_MODES}, got {sorted(shares)}")
    p = np.array([float(shares[n]) for n in HABITAT_MODES])
    if abs(p.sum() - 1.0) > 1e-6:
        raise ValueError(f"habitat_modes shares sum to {p.sum()}, not 1")
    return p


def draw_habitat_mode(p: Sequence[float], rng: np.random.Generator) -> Mode:
    """One Habitat mode drawn with probabilities p (HABITAT_MODES order)."""
    return BY_NAME[HABITAT_MODES[int(rng.choice(len(HABITAT_MODES), p=p))]]


def draw_gostanford_mode(goal_mask_prob: float, rng: np.random.Generator) -> Mode:
    """GoStanford: map always hidden; photo hidden with goal_mask_prob (NoMaD's 50/50)."""
    return BY_NAME["gs_explore"] if rng.random() < goal_mask_prob else BY_NAME["gs_photo"]

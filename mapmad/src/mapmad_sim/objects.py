"""World boxes of HM3D semantic objects.

habitat-sim 0.3.3 fills `obj.aabb` wrongly for HM3D: it is the union of the object's true box with the world
origin (00800: 558 of 661 boxes are > 4 m; a tv at z = -4.85 gets a box from z = 0 to -4.95). `obj.obb` is
right: its centre equals the object positions in ObjectNav v2 (`goals_by_category[...].position`), and in
HM3D v0.2 every OBB is axis-aligned. We still turn its 8 corners into a world box, in case one is rotated.
"""

from typing import Any, Tuple

import numpy as np

UNIT_CORNERS = np.array([[sx, sy, sz, 1.0] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def object_box(obj: Any) -> Tuple[np.ndarray, np.ndarray]:
    """(centre, size) in metres of the axis-aligned world box around a semantic object's OBB."""
    to_world = np.array(obj.obb.local_to_world, dtype=np.float64)  # maps the [-1, 1]^3 cube onto the OBB
    corners = (to_world @ UNIT_CORNERS.T).T[:, :3]
    low, high = corners.min(axis=0), corners.max(axis=0)
    return (low + high) / 2.0, high - low


def name_matches(name: str, words: Any) -> bool:
    """True if one of the words of a free-text name ('bath mat', 'floor') is in `words` (whole words only:
    'information' is not a 'mat')."""
    return any(w in words for w in name.lower().replace("-", " ").replace("_", " ").split())

"""Object and surface labels of one labelled HM3D (or MP3D, Phase 13) home, looked up at robot poses (needs LimoSim(semantic=True)).

One place for the label questions Phases 1, 2 and 5 ask (before Phase 5 the drive sampler built a whole Phase 1
EpisodeBuilder just to reach these):
- which objects exist (semantic id -> object, lower-case category name) and their world boxes (objects.py);
- is a pose on the real floor: the floor probe finds the floor under the robot and the surface it stands on is
  named floor / carpet / rug / mat ... (`floor_names`, whole words), not a bed, a table or stairs;
- how much of the LIMO picture an object fills at a pose (semantic pixels), and how much of the picture is
  near-black (HM3D mesh holes).
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.config import is_mp3d
from mapmad_sim.objects import name_matches, object_box


def category_name(obj: Any, mpcat40: bool) -> str:
    """Lower-case category name of a semantic object: HM3D's free text, or MP3D's mpcat40 category (Phase 13)."""
    if obj.category is None:
        return ""
    return (obj.category.name("mpcat40") if mpcat40 else obj.category.name()).lower()


class SceneLabels:
    """Label lookups of the home loaded in `robot` (a LimoSim with semantic=True)."""

    def __init__(self, robot: Any, floor_names: Sequence[str]) -> None:
        self.robot = robot
        objects = [o for o in robot.sim.semantic_scene.objects if o is not None]
        self.objects: Dict[int, Any] = {o.semantic_id: o for o in objects}
        mp3d = is_mp3d(getattr(robot, "split", ""))
        self.names: Dict[int, str] = {o.semantic_id: category_name(o, mp3d) for o in objects}
        raw = {o.semantic_id: category_name(o, False) for o in objects}
        # MP3D (Phase 13): names = mpcat40 categories; the floor = mpcat40 "floor" or a raw name in floor_names
        self.floor_ids = {i for i, n in self.names.items() if name_matches(n, floor_names) or name_matches(raw[i], floor_names)}

    def box(self, object_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(centre, size) of the object's world box, or None if the home has no such object."""
        obj = self.objects.get(object_id)
        return None if obj is None else object_box(obj)

    def on_real_floor(self, position: np.ndarray) -> bool:
        """Robot placed here stands on floor / carpet / rug / mat (not a bed, stairs or a hole)."""
        try:
            self.robot.place(position, 0.0)
        except ValueError:
            return False
        return self.robot.surface_id() in self.floor_ids

    def target_share(self, position: np.ndarray, yaw: float, object_id: int) -> float:
        """Share of the LIMO picture at this pose that shows the object."""
        self.robot.place(position, yaw)
        return float((self.robot.observe()["semantic"] == object_id).mean())

    def black_share(self) -> float:
        """Share of near-black pixels in the current picture: HM3D meshes have holes (rendered black), e.g.
        where a toilet should be, while the labels around them still say 'toilet'."""
        return float((self.robot.observe()["rgb"].max(axis=2) < 16).mean())

"""The explorer teacher of MapMaD v2 (plan v2 §8.2, Phase 8 block item 1; rules frozen in configs/p8_rules.md).

ONE pure function, `teacher_step(own_map, pose, state, seen, spec)` -> (Decision, new state): the robot's own online
map (whole storey, 10 cm), its pose, the teacher's memory (committed frontier, its recent path lengths, blacklist,
chosen approach goal) and the target instances seen so far -> the path to follow now and its next path point. Nothing
else: never the house plan, never the simulator. Used by
- the drive generator v2 (datagen_v2.py), every control step;
- Phase 11 (DAgger): relabel a visited state = call teacher_step on it (state saved per frame by the generator);
- Phase 12 arm F (map + frontier): the same function with the recogniser's detections as `seen`.

Rules (block item 1):
1. Target not yet seen -> nearest frontier by path length on the own map: 8-connected Dijkstra (steps 0.1 m and
   0.1 sqrt 2 m) through EXPLORED FREE cells not within the inflation (0.195 m + half a cell, centre to centre, as the
   v1 planner arm) of a known obstacle; cells within start_clear_m (0.3 m) of the robot are free (its own position is
   never blocked), and so are UNKNOWN cells within blind_m (0.6 m): at pin025 the floor is first seen ~0.55 m ahead of
   the turning centre, so the cells between the robot and its first view are never explored (implementation note,
   p8_rules.md §8). Frontier = map_v2.frontier_clusters (clusters >= 4 cells); a cluster's goal = its nearest
   reachable cell.
   - Commit to that goal until the robot is <= commit_reach_m (0.5 m) from it or it is no longer a frontier cell.
   - Tie: clusters within tie_margin_m (0.5 m) of the shortest path length -> the goal with the smallest bearing
     from the robot's heading ("more straight ahead").
   - Blacklist: path length to the committed goal shrank by < 0.5 m over the last 40 steps -> drop it for the rest of
     the drive (no goal within blacklist_radius_m of a dropped one is chosen again).
   - Implementation notes (p8_rules.md §8): a goal that was reached (<= 0.5 m) but is still a frontier (its unknown
     neighbour cannot be seen, e.g. a mesh hole) is dropped with radius reached_radius_m (0.2 m), so the rest of its
     cluster stays; goals already within 0.5 m are never chosen (they would count as reached at once).
2. Target seen (sticky) -> approach: shortest path on the own map with UNKNOWN = FREE (as the v1 planner arm),
   replanned every step, to the chosen instance's nearest view point. Several seen -> the instance + view point with
   the shortest path, re-evaluated only when a new instance is seen.
3. Arrived (<= arrive_m from the view point) -> face: yaw to the instance centre; the drive loop turns on the spot
   until within face_tolerance_deg (15 deg), then stops.
Path SHAPE (not a rule): the goal is chosen by plain path length (above); the path handed to the drive loop is then
planned to that goal with Phase 2's clearance preference (cost x (1 + 4 x shortfall below 0.445 m from the nearest
known obstacle cell centre = the robot's 0.195 m + Phase 2's preferred 0.25 m)), so it keeps to the middle of
corridors and doorways where there is room. path_length_m always reports the plain shortest path length.
Bumps (implementation note, p8_rules.md §8): cells where the robot was stopped by something its camera map does not
show (OwnMap.bumped, from the drive loop's collisions) count as obstacles for planning.
No frontier left and target never seen -> mode NO_FRONTIER (search failure); frontiers left but every one dropped ->
BLACKLIST_EXHAUSTED. Poses and points are (X, Y[, yaw]) in the 2D frame of traj_data.pkl (frames.py).
Trajectory ring (p8_rules_v3.md, `frontier_ignore_trajectory_m` > 0, OwnMap.visited): frontier cells within 0.7 m of the
robot's past positions are ignored -- unless no other frontier goal is available (no_frontier / blacklist_exhausted
with the ring): then the plain frontier is used (Decision.info["ring_fallback"]), and a goal chosen that way keeps the
plain frontier until its commitment ends (TeacherState.ring_fallback); then the ring is tried first again.
"""

import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "train") not in sys.path:
    sys.path.insert(0, str(REPO / "train"))  # numpy-only NoMaD-side map v2 (frontier), as online_map.py

from vint_train.mapmad.map_v2 import frontier_clusters  # noqa: E402

XY = Tuple[float, float]
Pose = Tuple[float, float, float]

SEARCH, APPROACH, ARRIVED, EXPLORE = "search", "approach", "arrived", "explore"
NO_FRONTIER, BLACKLIST_EXHAUSTED, NO_PATH = "no_frontier", "blacklist_exhausted", "no_path"
STEPS = ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)))  # each undirected edge once


@dataclass(frozen=True)
class TeacherSpec:
    """Teacher numbers (block item 1; p8_datagen.yaml `teacher`). Only the four marked ones may change in a pilot
    fix attempt (block item 6)."""

    inflation_m: float = 0.195
    start_clear_m: float = 0.3
    blind_m: float = 0.6  # unknown cells this close to the robot are passable (the camera never sees them, see below)
    min_cluster: int = 4
    commit_reach_m: float = 0.5  # fix-attempt parameter
    tie_margin_m: float = 0.5  # fix-attempt parameter
    blacklist_window_steps: int = 40  # fix-attempt parameter
    blacklist_min_progress_m: float = 0.5
    blacklist_radius_m: float = 0.5
    reached_radius_m: float = 0.2  # a reached goal that stays a frontier: cells this close to it are dropped
    arrive_m: float = 0.1
    face_tolerance_deg: float = 15.0  # fix-attempt parameter
    lookahead_m: float = 0.5
    preferred_clearance_m: float = 0.445  # path shape only: obstacle-cell centre distance kept where there is room
    clearance_weight: float = 4.0
    bbox_margin_m: float = 1.0  # planning window = known cells + robot + goals, plus this
    frontier_uninflated_fallback: bool = False  # proposal P4 (S-pilot): no frontier reachable -> once without inflation
    frontier_ignore_trajectory_m: float = 0.0  # p8_rules_v2.md 1b: frontier cells this close to the past path ignored

    @classmethod
    def from_config(cls, cfg: Dict) -> "TeacherSpec":
        return cls(**{k: type(getattr(cls, k))(v) for k, v in cfg.items()})


@dataclass(frozen=True)
class SeenInstance:
    """A target instance the robot has seen (training: simulator truth; Phase 12: the recogniser)."""

    instance_id: int
    centre: XY  # object centre, 2D
    view_points: Tuple[XY, ...]  # usable ObjectNav view points (snapped onto the floor map), 2D


@dataclass(frozen=True)
class TeacherState:
    """The teacher's memory between steps (saved per frame by the generator, so Phase 11 can relabel)."""

    committed: Optional[XY] = None  # committed frontier goal (cell centre)
    lengths: Tuple[float, ...] = ()  # path length to it at each step since it was chosen (last window + 1)
    blacklist: Tuple[Tuple[float, float, float], ...] = ()  # dropped goals: (x, y, radius)
    known_instances: Tuple[int, ...] = ()  # instances seen when the approach goal was last chosen
    approach: Optional[Tuple[int, XY]] = None  # (instance id, view point)
    uninflated: bool = False  # proposal P4: the committed goal was chosen by the uninflated fallback
    ring_fallback: bool = False  # rules v3: the committed goal was chosen on the plain frontier (no ring goal left)


@dataclass
class Decision:
    """What the teacher wants now."""

    mode: str  # search / explore / approach / arrived / no_frontier / blacklist_exhausted / no_path
    path: np.ndarray  # (N, 2) cell-centre path from the robot's cell to the goal (first point = robot position)
    goal: Optional[XY] = None
    next_point: Optional[XY] = None  # path point lookahead_m along the path
    new_choice: bool = False  # a new frontier / approach goal was chosen this step
    path_length_m: float = float("nan")
    face_yaw: Optional[float] = None  # arrived: heading to the instance centre
    n_clusters: int = 0
    dropped: Optional[str] = None  # why the previous commitment ended: reached / gone / blacklisted / unreachable
    info: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class OwnMap:
    """The robot's own map at this step: obstacle / explored as known now (bool, whole storey), plus the cells where
    it bumped into something the camera map does not show (`bumped`, planning only: they count as obstacles here and
    never enter the network's map layers)."""

    obstacle: np.ndarray
    explored: np.ndarray
    origin: XY
    res: float
    bumped: Optional[np.ndarray] = None
    visited: Optional[np.ndarray] = None  # rules v2 1b: cells within frontier_ignore_trajectory_m of the past path


# --- grid helpers -----------------------------------------------------------------------------------------------

def disk(radius_cells: float) -> List[Tuple[int, int]]:
    r = int(math.floor(radius_cells))
    return [(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1) if math.hypot(a, b) <= radius_cells]


def dilate(mask: np.ndarray, radius_cells: float) -> np.ndarray:
    """Cells within radius_cells (centre to centre) of a True cell (v1 planner.dilate)."""
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for di, dj in disk(radius_cells):
        out[max(0, di):h - max(0, -di), max(0, dj):w - max(0, -dj)] |= \
            mask[max(0, -di):h - max(0, di), max(0, -dj):w - max(0, dj)]
    return out


@dataclass(frozen=True)
class Window:
    """The part of the storey map the planner works on: rows i0:i1, cols j0:j1."""

    i0: int
    i1: int
    j0: int
    j1: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.i1 - self.i0, self.j1 - self.j0


def plan_window(m: OwnMap, points: Sequence[XY], margin_m: float) -> Window:
    """Bounding box of the known cells and the given points, plus margin_m, clipped to the map."""
    rows, cols = np.nonzero(m.explored | m.obstacle)
    cells = [cell_of(p, m) for p in points]
    rr = np.concatenate([rows, [c[0] for c in cells]])
    cc = np.concatenate([cols, [c[1] for c in cells]])
    k = int(math.ceil(margin_m / m.res))
    h, w = m.explored.shape
    return Window(max(0, int(rr.min()) - k), min(h, int(rr.max()) + k + 1), max(0, int(cc.min()) - k), min(w, int(cc.max()) + k + 1))


def cell_of(p: XY, m: OwnMap) -> Tuple[int, int]:
    return int(math.floor((p[0] - m.origin[0]) / m.res)), int(math.floor((p[1] - m.origin[1]) / m.res))


def centre_of(i: np.ndarray, j: np.ndarray, m: OwnMap) -> np.ndarray:
    return np.stack([m.origin[0] + (np.asarray(i) + 0.5) * m.res, m.origin[1] + (np.asarray(j) + 0.5) * m.res], axis=-1)


def distance_field(free: np.ndarray, source: Tuple[int, int], res: float,
                   cost: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Path length (m) from `source` to every cell through `free` cells (8-connected, a diagonal step needs both side
    cells free), and the predecessor of each cell (-9999 = none). Cells off `free` are unreachable (inf). With `cost`
    (per-cell factor >= 1) a step costs its length x the mean factor of its two cells."""
    h, w = free.shape
    rows, cols, vals = [], [], []
    r, c = np.nonzero(free)
    for dr, dc, step in STEPS:
        r2, c2 = r + dr, c + dc
        ok = (r2 < h) & (c2 >= 0) & (c2 < w)
        ok[ok] = free[r2[ok], c2[ok]]
        if dr and dc:
            ok[ok] = free[r[ok], c2[ok]] & free[r2[ok], c[ok]]
        rows.append(r[ok] * w + c[ok])
        cols.append(r2[ok] * w + c2[ok])
        factor = 1.0 if cost is None else 0.5 * (cost[r[ok], c[ok]] + cost[r2[ok], c2[ok]])
        vals.append(np.full(int(ok.sum()), step * res) * factor)
    graph = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w)).tocsr()
    dist, pred = dijkstra(graph, directed=False, indices=source[0] * w + source[1], return_predecessors=True)
    return dist.reshape(h, w), pred.reshape(h, w)


def trace(pred: np.ndarray, goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Cells from the source to `goal` (inclusive) along the predecessors."""
    w = pred.shape[1]
    out = [goal]
    while pred[out[-1]] >= 0:
        k = int(pred[out[-1]])
        out.append((k // w, k % w))
    return out[::-1]


def path_point(path: np.ndarray, along_m: float) -> XY:
    """The first path point at least along_m along the path (the last one if shorter)."""
    if len(path) == 1:
        return tuple(path[0])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    return tuple(path[min(int(np.searchsorted(s, along_m)), len(path) - 1)])


def bearing(pose: Pose, p: XY) -> float:
    """|angle| (rad) between the robot's heading and the direction to p."""
    a = math.atan2(p[1] - pose[1], p[0] - pose[0]) - pose[2]
    return abs(math.atan2(math.sin(a), math.cos(a)))


# --- the teacher ------------------------------------------------------------------------------------------------

class _Planner:
    """One step's planning grids on the window (built once per teacher_step)."""

    def __init__(self, m: OwnMap, pose: Pose, extra: Sequence[XY], spec: TeacherSpec) -> None:
        self.m, self.spec = m, spec
        self.win = plan_window(m, [pose[:2], *extra], spec.bbox_margin_m)
        sl = (slice(self.win.i0, self.win.i1), slice(self.win.j0, self.win.j1))
        self.obstacle, self.explored = m.obstacle[sl], m.explored[sl]
        if m.bumped is not None:
            self.obstacle = self.obstacle | m.bumped[sl]
        self.visited = m.visited[sl] if m.visited is not None else None
        self.robot = self.local(cell_of(pose[:2], m))
        blocked = dilate(self.obstacle, spec.inflation_m / m.res + 0.5)
        h, w = blocked.shape
        for di, dj in disk(spec.start_clear_m / m.res):  # the robot's own surroundings are never blocked
            i, j = self.robot[0] + di, self.robot[1] + dj
            if 0 <= i < h and 0 <= j < w:
                blocked[i, j] = False
        self.blocked = blocked
        self.near_robot = self._disk_mask(spec.start_clear_m)
        self.blind = self._disk_mask(spec.blind_m) & ~self.explored & ~blocked
        clearance = ndimage.distance_transform_edt(~self.obstacle) * m.res if self.obstacle.any() else np.full(blocked.shape, np.inf)
        short = np.clip((spec.preferred_clearance_m - clearance) / spec.preferred_clearance_m, 0.0, 1.0)
        self.cost = 1.0 + spec.clearance_weight * short  # path shape only (module docstring)

    def shaped_path(self, free: np.ndarray, goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Cells from the robot to `goal` through `free`, keeping to the middle where there is room."""
        _, pred = distance_field(free, self.robot, self.m.res, self.cost)
        return trace(pred, goal)

    def _disk_mask(self, radius_m: float) -> np.ndarray:
        """Cells within radius_m of the robot's cell (centre to centre)."""
        out = np.zeros_like(self.blocked)
        h, w = out.shape
        for di, dj in disk(radius_m / self.m.res):
            i, j = self.robot[0] + di, self.robot[1] + dj
            if 0 <= i < h and 0 <= j < w:
                out[i, j] = True
        return out

    def local(self, ij: Tuple[int, int]) -> Tuple[int, int]:
        return ij[0] - self.win.i0, ij[1] - self.win.j0

    def inside(self, ij: Tuple[int, int]) -> bool:
        return 0 <= ij[0] < self.win.shape[0] and 0 <= ij[1] < self.win.shape[1]

    def xy(self, ij: Tuple[int, int]) -> XY:
        p = centre_of(ij[0] + self.win.i0, ij[1] + self.win.j0, self.m)
        return float(p[0]), float(p[1])

    def path_xy(self, cells: List[Tuple[int, int]], pose: Pose, end: Optional[XY] = None) -> np.ndarray:
        pts = centre_of(np.array([c[0] for c in cells]) + self.win.i0, np.array([c[1] for c in cells]) + self.win.j0, self.m)
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        pts[0] = pose[:2]
        if end is not None:
            pts = np.vstack([pts, end]) if len(pts) > 1 or not np.allclose(pts[0], end) else pts
        return pts


def _frontier_free(pl: _Planner) -> np.ndarray:
    return (pl.explored & ~pl.blocked) | pl.near_robot | pl.blind  # explored free cells (+ the robot's blind zone)


def _uninflated(pl: _Planner) -> None:
    """Proposal P4: plan the frontier search without inflation (known obstacle cells stay blocked)."""
    pl.blocked = pl.obstacle.copy()
    pl.blocked[pl.near_robot] = False
    pl.blind = pl.blind | (pl._disk_mask(pl.spec.blind_m) & ~pl.explored & ~pl.blocked)


def _frontier_field(pl: _Planner) -> Tuple[np.ndarray, np.ndarray]:
    free = _frontier_free(pl)
    if not pl.inside(pl.robot):
        return np.full(free.shape, np.inf), np.full(free.shape, -9999)
    return distance_field(free, pl.robot, pl.m.res)


def _dropped(p: XY, blacklist: Sequence[Tuple[float, float, float]]) -> bool:
    """p lies within the radius of a dropped goal."""
    return any(math.hypot(p[0] - x, p[1] - y) <= r for x, y, r in blacklist)


def _explore(m: OwnMap, pose: Pose, state: TeacherState, spec: TeacherSpec, mode: str) -> Tuple[Decision, TeacherState]:
    """Rule 1 with the rules v3 trajectory ring (module docstring): the ring frontier first; the plain frontier only
    when the ring leaves no goal, or while a goal chosen on it is committed."""
    stuck = (NO_FRONTIER, BLACKLIST_EXHAUSTED)
    if m.visited is None:
        return _explore_inflation(m, pose, state, spec, mode)
    plain = replace(m, visited=None)
    if state.ring_fallback and state.committed is not None:
        dec, st = _explore_inflation(plain, pose, state, spec, mode)
        if not dec.new_choice and dec.mode not in stuck:  # the fallback commitment goes on
            dec.info["ring_fallback"] = True
            return dec, st
        state = replace(st, committed=None, lengths=(), ring_fallback=False)  # it ended: the ring first again
        dropped = dec.dropped
    else:
        dropped = None
    dec, st = _explore_inflation(m, pose, state, spec, mode)
    dec.dropped = dec.dropped or dropped
    if dec.mode not in stuck:
        return dec, replace(st, ring_fallback=False)
    dec2, st2 = _explore_inflation(plain, pose, st, spec, mode)
    dec2.dropped = dec2.dropped or dec.dropped
    dec2.info["ring_fallback"] = dec2.mode not in stuck  # the plain frontier gave a goal
    return dec2, replace(st2, ring_fallback=st2.committed is not None)


def _explore_inflation(m: OwnMap, pose: Pose, state: TeacherState, spec: TeacherSpec,
                       mode: str) -> Tuple[Decision, TeacherState]:
    """Rule 1: nearest frontier, commitment, tie, blacklist (P4: if nothing is reachable, once more uninflated)."""
    extra = [state.committed] if state.committed else []
    if not (spec.frontier_uninflated_fallback and state.uninflated and state.committed):
        dec, st = _explore_on(_Planner(m, pose, extra, spec), m, pose, state, spec, mode)
        if not spec.frontier_uninflated_fallback or dec.mode not in (NO_FRONTIER, BLACKLIST_EXHAUSTED):
            return dec, replace(st, uninflated=False)
    pl = _Planner(m, pose, extra, spec)  # P4: nothing reachable with inflation (or the commitment came from here)
    _uninflated(pl)
    dec2, st2 = _explore_on(pl, m, pose, state, spec, mode)
    dec2.info["uninflated"] = True
    return dec2, replace(st2, uninflated=st2.committed is not None)


def _explore_on(pl: "_Planner", m: OwnMap, pose: Pose, state: TeacherState, spec: TeacherSpec,
                mode: str) -> Tuple[Decision, TeacherState]:
    """Rule 1 on one planning grid."""
    labels = frontier_clusters(pl.obstacle, pl.explored, spec.min_cluster, pl.visited)
    n_clusters = int(labels.max())
    dist, pred = _frontier_field(pl)
    blacklist = list(state.blacklist)  # (x, y, radius)
    dropped = None
    # --- keep the committed goal? ---
    if state.committed is not None:
        c = pl.local(cell_of(state.committed, m))
        d = float(dist[c]) if pl.inside(c) else math.inf
        lengths = (state.lengths + (d,))[-(spec.blacklist_window_steps + 1):]
        reach = math.hypot(state.committed[0] - pose[0], state.committed[1] - pose[1]) <= spec.commit_reach_m
        if not (pl.inside(c) and labels[c] > 0):
            dropped = "gone"
        elif reach:
            dropped = "reached"
            blacklist.append((*state.committed, spec.reached_radius_m))  # still a frontier: its unknown side is unseeable
        elif not math.isfinite(d):
            dropped = "unreachable"
        elif len(lengths) > spec.blacklist_window_steps and lengths[0] - lengths[-1] < spec.blacklist_min_progress_m:
            dropped = "blacklisted"
            blacklist.append((*state.committed, spec.blacklist_radius_m))
        else:
            path = pl.path_xy(pl.shaped_path(_frontier_free(pl), c), pose)
            st = replace(state, lengths=lengths, blacklist=tuple(blacklist))
            return Decision(mode, path, state.committed, path_point(path, spec.lookahead_m), False, d,
                            n_clusters=n_clusters), st
    # --- choose a new goal ---
    candidates = []  # (path length, bearing, goal cell)
    excluded_by_blacklist = 0
    for k in range(1, n_clusters + 1):
        ci, cj = np.nonzero(labels == k)
        dk = dist[ci, cj]
        for idx in np.argsort(dk, kind="stable"):
            if not math.isfinite(dk[idx]):
                break
            cell = (int(ci[idx]), int(cj[idx]))
            p = pl.xy(cell)
            if math.hypot(p[0] - pose[0], p[1] - pose[1]) <= spec.commit_reach_m:
                continue
            if _dropped(p, blacklist):
                excluded_by_blacklist += 1
                continue
            candidates.append((float(dk[idx]), bearing(pose, p), cell))
            break
    clear = replace(state, committed=None, lengths=(), blacklist=tuple(blacklist))
    if not candidates:
        why = BLACKLIST_EXHAUSTED if excluded_by_blacklist else NO_FRONTIER
        return Decision(why, np.asarray([pose[:2]], dtype=np.float64), n_clusters=n_clusters, dropped=dropped), clear
    best = min(c[0] for c in candidates)
    tied = [c for c in candidates if c[0] <= best + spec.tie_margin_m]
    length, _, cell = min(tied, key=lambda c: (c[1], c[0], c[2]))
    goal = pl.xy(cell)
    path = pl.path_xy(pl.shaped_path(_frontier_free(pl), cell), pose)
    st = replace(clear, committed=goal, lengths=(length,))
    return Decision(mode, path, goal, path_point(path, spec.lookahead_m), True, length, n_clusters=n_clusters,
                    dropped=dropped, info={"candidates": len(candidates), "tied": len(tied)}), st


def _approach(m: OwnMap, pose: Pose, state: TeacherState, seen: Sequence[SeenInstance],
              spec: TeacherSpec) -> Tuple[Decision, TeacherState]:
    """Rules 2-3: shortest path with unknown = free to the chosen view point; arrived -> face the instance."""
    ids = tuple(sorted(s.instance_id for s in seen))
    vps = [(s.instance_id, vp) for s in seen for vp in s.view_points]
    pl = _Planner(m, pose, [vp for _, vp in vps], spec)
    for inflated, vp_free in ((True, ~pl.blocked), (False, ~pl.obstacle | pl.near_robot)):  # uninflated if unreachable
        dist, pred = (distance_field(vp_free, pl.robot, m.res) if pl.inside(pl.robot)
                      else (np.full(vp_free.shape, np.inf), np.full(vp_free.shape, -9999)))
        new_choice = state.approach is None or ids != state.known_instances
        choice = state.approach
        if new_choice:
            scored = []
            for inst, vp in vps:
                c = pl.local(cell_of(vp, m))
                if pl.inside(c) and math.isfinite(dist[c]):
                    scored.append((float(dist[c]), inst, vp))
            if not scored:
                continue
            _, inst, vp = min(scored, key=lambda s: (s[0], s[1], s[2]))
            choice = (inst, vp)
        inst, vp = choice
        c = pl.local(cell_of(vp, m))
        if not (pl.inside(c) and math.isfinite(dist[c])):
            continue
        centre = next(s.centre for s in seen if s.instance_id == inst)
        st = replace(state, approach=choice, known_instances=ids, committed=None, lengths=())
        if math.hypot(vp[0] - pose[0], vp[1] - pose[1]) <= spec.arrive_m:
            return Decision(ARRIVED, np.asarray([pose[:2]], dtype=np.float64), vp, vp, new_choice, 0.0,
                            face_yaw=math.atan2(centre[1] - pose[1], centre[0] - pose[0])), st
        path = pl.path_xy(pl.shaped_path(vp_free, c), pose, end=vp)
        return Decision(APPROACH, path, vp, path_point(path, spec.lookahead_m), new_choice, float(dist[c]),
                        info={"instance": inst, "inflated": inflated}), st
    return Decision(NO_PATH, np.asarray([pose[:2]], dtype=np.float64)), state


def teacher_step(own_map: OwnMap, pose: Pose, state: TeacherState, seen: Sequence[SeenInstance] = (),
                 spec: TeacherSpec = TeacherSpec(), searching: bool = True) -> Tuple[Decision, TeacherState]:
    """The teacher's decision at this step and its new memory (module docstring). `searching` = a search drive
    (mode label "search" while exploring); False = an exploration drive (mode "explore", `seen` ignored)."""
    if searching and seen:
        return _approach(own_map, pose, state, seen, spec)
    return _explore(own_map, pose, state, spec, SEARCH if searching else EXPLORE)

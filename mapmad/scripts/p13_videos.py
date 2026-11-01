"""Phase 13 checkable videos (step 13.1): one video per (arm, episode) of a finished closed-loop run, made from its
step log and its recogniser "dets" file, with automatic checks that the video shows exactly what was logged.
Inside naz_mapmad_habitat, from /app/visualnav-transformer (a YOLOE sidecar on --recog-port, started by the caller):

    python mapmad/scripts/p13_videos.py render --config <phase config> --subdir <run subdir> --arms A B ... \
        --episodes E1 E2 ... --seed-index 0 --gpu G --recog-port P --out <folder>     # resumable
    python mapmad/scripts/p13_videos.py render-plan --config mapmad/configs/p13_claim.yaml --subdir videos_rerun \
        --gpu G --recog-port P --out <folder> [--shard k/n]   # videos.list picks x videos.arms, set video
        [--list <video list> --arms A B --set S]               # another list (pick-rules) / arms / set
    python mapmad/scripts/p13_videos.py pick-rules --config <c> --set hm3d --subdir runs/hm3d --maps M21b \
        --hidden M21bh --f-scores <F scores.jsonl> --out <list.json>  # Phase 12's 4 rule picks for another pair
    python mapmad/scripts/p13_videos.py check --out <folder> [--report <json>] [--config <c> [--list ..] [--arms ..]]
        # boxes + poses; with --config also: every planned video exists
    python mapmad/scripts/p13_videos.py same-run --new <logs root> --old <logs root> (--arms ... --episodes ... |
        --config <config with a videos block>) --out <json> [--seed-index 0]

What Naz can answer by watching: where is the target, did YOLOE see it, when was the run scored, was the stop on the
right object. Replay: the run is replayed pose by pose through the exam's own server code exactly as p12_videos.py
does (`replay_states`: sidecar_server.SimService in-process, the arm's settings from search_wire.arm_wire, the logged
noise seed; robot.move_to on every logged pose, so a mesh hole keeps the last floor height), which rebuilds the online
map; the semantic camera of the replay gives the true target boxes.

Frame (1920 x 1080; real time: one 0.25 s step = 6 frames at 24 fps):
- top band (100 px): episode id, home, find: '<category>', the arm's full name; step, time and the events so far;
- left: the camera (320 x 240 x 3 = 960 x 720, nearest neighbour) with this frame's recogniser boxes from the dets file
  (orange, label "<category> <raw score>" with the title's category name, the YOLOE prompt word in the legend; thick
  when the box's map value >= tau) and the true target boxes (green,
  "TRUE <category> #<id>": bounding box of the instance's pixels in the replay's semantic picture); below it the legend;
- right: the whole storey north up (fixed, 760 px): unknown, floor not seen yet (faint), explored, obstacles, the target
  layer (value > 0: colour scale up to tau; >= tau yellow; "rule-ready" = >= tau and seen from >= 2 views, red), the
  true objects (green boxes) with their view points (green discs of 0.2 m = the success zone), start, path, robot,
  the rule's best cell of the step (magenta ring) and, from the rule stop on, the triggering cell (magenta X);
- under the map: the timeline of the whole run (max_steps): rule value with the tau line, a strip where simulator truth
  sees a target, a strip where a YOLOE box is on a true target (IoU >= 0.3), markers (first true sight, first YOLOE box
  (any object), first YOLOE box on a true target, at a target view spot = oracle success, rule stop: right object /
  wrong object / near miss / outside), the cursor.
- scored moments (oracle success step, rule stop step): a 2 s freeze, everything but the camera and the map dimmed, a
  large caption ("at a target view spot (sim truth)"; "rule stop: right object / wrong object / ..."); after the
  oracle success (the scored moment) the rest of the run is dimmed below the top band, with a note.
Event rules: mapmad_sim/video_events.py (first YOLOE box (any object); first YOLOE box on a true target: IoU >= 0.3
between a logged box and a true box of the same frame) and closed_loop/p12_score.score_log (sight, oracle, rule stop and how it ended).

Outputs in --out: <name>.mp4 (render: <episode>__s<k>__<arm>_<file name>; render-plan: <pick name>__<arm>; H.264
CRF 23, faststart), small/<name>.mp4 (<= 1280 px wide, CRF 31, faststart), <name>.manifest.json (inputs + sha256s,
events, per step the boxes drawn and the replayed pose, the frame table), run_info.render.json (config, seed, git
commit); a run whose log or dets file is missing -> <name>.missing.json instead (no video; `check` lists it as "not
reproducible"). check -> --report (default <out>/video_check.json), with a "summary" line:
1. boxes: 20 seeded random frames per video (seed 13101 + video index, videos in name order): drawn boxes = the dets
   file's boxes of that step (exact list, order, values, label text, thickness);
2. poses: every replayed pose = the logged pose (steps >= 1 bit for bit, yaw within 1e-12 after wrapping; the start
   within 1e-6, the log rounds to 6 digits); mesh-hole frames (last floor height kept) counted;
   the log and dets sha256 = the ones rendered. Info only: replay's own recogniser boxes and rule values = the logged
   ones (the map rebuild is exact). Exit 3 on a failure.
same-run: re-run log vs the Phase 12 exam log of each (arm, episode): pose sequence (start + position and yaw per step)
over the common prefix -> "reproducible" / "not reproducible" (first differing step), and whether the two files are
byte-identical; JSON report with "pass" and a "summary" line ("94/96 reproducible; not: ..."); exit 3 if any is not
reproducible.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mapmad_bridge import steplog  # noqa: E402
from mapmad_sim import video_events as ve  # noqa: E402
from mapmad_sim.arm_table import ArmTable  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.robot import wrap_angle  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import RunLayout, run_id, write_text_atomic  # noqa: E402
from mapmad_sim.search_wire import RECOG_HOST, arm_wire  # noqa: E402
from mapmad_sim.sidecar_server import SimService, limo_factory  # noqa: E402
from p12_videos import ARMS_PLAIN, Writer, replay_states  # noqa: E402
from p12_videos import font as _font  # noqa: E402
from vint_train.mapmad.closed_loop import video  # noqa: E402

SIZE = (1920, 1080)
FPS, REPEAT = 24, 6  # one 0.25 s control step = 6 frames at 24 fps: real time
FREEZE = 2 * FPS  # a scored moment is held 2 s
DIM = 0.45  # brightness of the frames after the oracle success (below the top band)
CRF, SMALL_CRF, SMALL_W = 23, 31, 1280
STEP_S = 0.25
HEAD = 100
CAM_SCALE = 3
CAM = (320 * CAM_SCALE, 240 * CAM_SCALE)
MAP = 760
MAP_XY = (960 + (960 - MAP) // 2, HEAD + 4)
TL = (980, HEAD + 4 + MAP + 8, 920, SIZE[1] - (HEAD + 4 + MAP + 8))  # timeline x, y, w, h
CHECK_SEED, CHECK_FRAMES = 13101, 20
MANIFEST, MISSING = ".manifest.json", ".missing.json"
# colours (RGB)
YOLO, TRUE_C, SIGHT_C, ORACLE_C = (255, 150, 0), (60, 235, 60), (90, 200, 255), (60, 235, 60)
RIGHT_C, WRONG_C, NEAR_C = (60, 235, 60), (255, 60, 60), (255, 170, 60)
OUTSIDE_PX, FLOOR_PX, EXPL_PX, OBST_PX = (20, 20, 20), (52, 52, 64), (105, 105, 105), (235, 235, 235)
HIGH_C, READY_C, VP_C, PATH_C, RULE_C = (255, 230, 0), (255, 40, 40), (0, 120, 50), (0, 210, 255), (255, 0, 255)
HOW_C = {"right object": RIGHT_C, "wrong object": WRONG_C, "near miss": NEAR_C, "outside": WRONG_C}
font = lru_cache(maxsize=None)(_font)  # the TrueType font at px pixels, loaded once per size
BOX_C = (255, 205, 130)  # first YOLOE box (any object): lighter than YOLO
KIND_C = {"sight": SIGHT_C, "box": BOX_C, "hit": YOLO, "oracle": ORACLE_C}


# --- inputs -------------------------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_names(table: ArmTable, arm: str) -> Tuple[str, str]:
    """(file name part, full name) of an arm: p12_videos.ARMS_PLAIN, else the arm table's label."""
    if arm in ARMS_PLAIN:
        return f"{arm}_{ARMS_PLAIN[arm][0]}", ARMS_PLAIN[arm][1]
    label = table.arm(arm).label or arm
    return f"{arm}_" + "".join(c if c.isalnum() else "_" for c in label).strip("_"), label


def video_name(episode_id: str, k: int, file_part: str) -> str:
    return f"{run_id(episode_id, k)}__{file_part}"


def prompt_of(episode: Dict[str, Any]) -> str:
    """The word the recogniser was asked (as sidecar_server: target.prompt, else photo_labels.PROMPTS)."""
    from mapmad_sim.photo_labels import PROMPTS

    t = episode["target"]
    return t.get("prompt") or PROMPTS[t["category"]]


def score_run(log: List[Dict[str, Any]], episode: Dict[str, Any], run: Dict[str, Any], tau: float) -> Dict[str, Any]:
    """p12_score.score_log of the run at tau (with the episode's instances, as `p12_score score` does)."""
    from vint_train.mapmad.closed_loop import p12_score

    log = [dict(log[0], instances=episode["target"]["instances"], stuck_steps=run["stuck_steps"])] + log[1:]
    return p12_score.score_log(log, taus=[tau], success_m=float(run["success_m"]), revisit_m=float(run["revisit_cell_m"]),
                               at=int(run["success_at"][0]))


# --- the replay (pass 1) --------------------------------------------------------------------------------------------
class TrueBoxes:
    """A robot factory around limo_factory whose robots also compute the true target boxes of every picture they take
    (`last`), for the instance ids in `ids` (set per episode)."""

    def __init__(self, gpu: int) -> None:
        self.base = limo_factory(gpu)
        self.ids: List[int] = []
        self.last: List[Dict[str, Any]] = []

    def __call__(self, *a: Any, **k: Any) -> Any:
        robot = self.base(*a, **k)
        observe = robot.observe

        def observe_and_box() -> Dict[str, np.ndarray]:
            obs = observe()
            self.last = ve.true_boxes(obs["semantic"], self.ids) if "semantic" in obs else []
            return obs

        robot.observe = observe_and_box
        return robot


def replay(service: SimService, boxes: TrueBoxes, log_path: Path, episode: Dict[str, Any], wire: Any,
           min_views: int) -> Dict[str, Any]:
    """Pass 1: the run replayed (p12_videos.replay_states); per step the camera picture, the true boxes, the replayed
    pose and hole flag, the rule-ready mask (packed bits) and the replay's own recogniser boxes / rule value; at the end
    the storey map's first-seen arrays and the target layer's updates (any step's map can be rebuilt)."""
    boxes.ids = [int(i) for i in episode["target"]["category_ids"]]
    steps: List[Dict[str, Any]] = []
    tmap, fails = None, 0
    for head, tmap, t, reply, path, yaw in replay_states(service, log_path, episode, wire.search_v2, wire.camera):
        robot = service.robot
        hole = robot.floor_probe_failures > fails if t > 0 else False
        fails = robot.floor_probe_failures
        count = service.search.fusions["noisy"].count
        steps.append({"rgb": np.array(reply["rgb"], copy=True), "true": boxes.last,
                      "pose": ([float(v) for v in robot.position], float(robot.yaw)), "hole": bool(hole),
                      "ready_bits": np.packbits(count >= min_views),
                      "replay_dets": reply.get("dets"), "replay_rule": (reply.get("rule") or {}).get("value")})
    s = service.search
    return {"steps": steps, "tmap": tmap, "shape": s.fmap.shape, "origin": np.asarray(s.fmap.origin, np.float64),
            "res": float(s.spec.map_spec.resolution),
            "obstacle_first": s.fmap.first_seen["obstacle"].copy(), "explored_first": s.fmap.first_seen["explored"].copy(),
            "updates": s.fusions["noisy"].updates()}


# --- the storey map (right) -----------------------------------------------------------------------------------------
class FloorView:
    """The whole storey north up at `side` px: p12's TopdownMap crop (480 px) scaled by side / 480, with each pixel's
    online-map cell (X = -z, Y = -x of Habitat; frames.to_2d)."""

    def __init__(self, tmap: video.TopdownMap, side: int, origin: np.ndarray, res: float, shape: Tuple[int, int]) -> None:
        self.t, self.side, self.s = tmap, side, tmap.scale * side / video.MAP_PX
        nav = cv2.resize(tmap.base, (side, side), interpolation=cv2.INTER_NEAREST)
        self.walkable = nav[..., 0] == video.GREY[0]
        rows, cols = np.mgrid[0:side, 0:side]
        x = tmap.origin[0] + ((cols + 0.5) / self.s + tmap.c0) * tmap.m_per_px
        z = tmap.origin[1] + ((rows + 0.5) / self.s + tmap.r0) * tmap.m_per_px
        i = np.floor((-z - origin[0]) / res).astype(np.int64)
        j = np.floor((-x - origin[1]) / res).astype(np.int64)
        self.inside = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
        self.flat = np.where(self.inside, i * shape[1] + j, 0)

    def px(self, p: Sequence[float]) -> Tuple[int, int]:
        """(col, row) of Habitat point p = (x, y, z)."""
        c = ((p[0] - self.t.origin[0]) / self.t.m_per_px - self.t.c0) * self.s
        r = ((p[2] - self.t.origin[1]) / self.t.m_per_px - self.t.r0) * self.s
        return int(math.floor(c)), int(math.floor(r))

    def px2d(self, xy: Sequence[float]) -> Tuple[int, int]:
        """(col, row) of a 2D-frame point (X, Y) (rule cells): Habitat x = -Y, z = -X."""
        return self.px([-xy[1], 0.0, -xy[0]])

    def metres(self, d: float) -> float:
        return d / self.t.m_per_px * self.s


def layer_image(fv: FloorView, obst: np.ndarray, expl: np.ndarray, value: np.ndarray, ready: np.ndarray,
                tau: float) -> np.ndarray:
    """Cell layers (flat arrays) -> storey picture (fv.side px): colours of the module docstring."""
    img = np.empty((fv.side, fv.side, 3), np.float32)
    img[:] = OUTSIDE_PX
    img[fv.walkable] = FLOOR_PX
    g = lambda a: a[fv.flat] & fv.inside  # noqa: E731
    img[g(expl)] = EXPL_PX
    img[g(obst)] = OBST_PX
    v = np.where(fv.inside, value[fv.flat], 0.0)
    low = (v > 0) & (v < tau - 1e-12)
    a = (0.35 + 0.65 * np.clip(v / tau, 0, 1))[..., None]
    img = np.where(low[..., None], img * (1 - a) + np.array(YOLO, np.float32) * a, img)
    high = v >= tau - 1e-12
    img[high] = HIGH_C
    img[high & g(ready)] = READY_C
    return img.astype(np.uint8)


def map_panel(fv: FloorView, base: np.ndarray, episode: Dict[str, Any], path: List[Sequence[float]], yaw: float,
              rule_cell: Optional[Sequence[float]], stop_cell: Optional[Sequence[float]]) -> np.ndarray:
    """The storey picture with view points, true boxes, start, path, robot, the rule's cell now and the stop cell."""
    spots = base.copy()
    r = max(2, int(round(fv.metres(0.2))))
    for inst in episode["target"]["instances"]:
        for vp in inst["view_points"]:
            cv2.circle(spots, fv.px(vp), r, VP_C, -1, cv2.LINE_AA)
    img = cv2.addWeighted(spots, 0.6, base, 0.4, 0)  # the success zone, the map still visible under it
    for inst in episode["target"]["instances"]:
        c, s = inst["centre"], inst["size"]
        cv2.rectangle(img, fv.px([c[0] - s[0] / 2, 0, c[2] - s[2] / 2]), fv.px([c[0] + s[0] / 2, 0, c[2] + s[2] / 2]),
                      TRUE_C, 3)
    cv2.circle(img, fv.px(episode["start_position"]), 8, (255, 255, 255), 2, cv2.LINE_AA)
    if len(path) > 1:
        cv2.polylines(img, [np.array([fv.px(p) for p in path], np.int32)], False, PATH_C, 2, cv2.LINE_AA)
    if rule_cell is not None:
        cv2.circle(img, fv.px2d(rule_cell), 7, RULE_C, 2, cv2.LINE_AA)
    if stop_cell is not None:
        cv2.drawMarker(img, fv.px2d(stop_cell), RULE_C, cv2.MARKER_TILTED_CROSS, 22, 4)
    here = np.array(fv.px(path[-1]), np.float64)
    fwd, side_ = np.array([-math.sin(yaw), -math.cos(yaw)]), np.array([-math.cos(yaw), math.sin(yaw)])
    tri = np.round(np.array([here + 16 * fwd, here - 9 * fwd + 9 * side_, here - 9 * fwd - 9 * side_])).astype(np.int32)
    cv2.fillPoly(img, [tri], (255, 255, 255), cv2.LINE_AA)
    cv2.polylines(img, [tri], True, (0, 0, 0), 2, cv2.LINE_AA)
    bar = int(round(fv.metres(1.0)))  # 1 m scale bar
    cv2.line(img, (16, fv.side - 18), (16 + bar, fv.side - 18), (255, 255, 255), 3)
    return img


# --- the camera (left) ----------------------------------------------------------------------------------------------
def inside_x(x: int, s: str, px: int = 20) -> int:
    """x moved left so that the text s (px font) ends inside the camera picture."""
    return max(0, min(x, CAM[0] - 8 - int(font(px).getlength(s))))


def camera_panel(rgb: np.ndarray, dets: List[Dict[str, Any]], trues: List[Dict[str, Any]], category: str,
                 tau: float, text: Writer, y0: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """The camera x3 with the logged recogniser boxes and the true boxes; returns (picture, the boxes drawn: logged
    box, label, thick -- what `check` compares with the dets file)."""
    img = cv2.resize(rgb, CAM, interpolation=cv2.INTER_NEAREST)
    k = CAM_SCALE
    for t in trues:
        x0, y_0, x1, y1 = (int(round(v * k)) for v in t["box"])
        cv2.rectangle(img, (x0, y_0), (x1 - 1, y1 - 1), TRUE_C, 3)
        s = f"TRUE {category} #{t['id']}"
        text.add((inside_x(x0 + 6, s), y0 + max(y1 - 30, y_0 + 4)), s, 20, TRUE_C)
    drawn = []
    for d in dets:
        if d.get("box") is None:
            continue
        label, thick = ve.box_label(category, d["score"]), d["value"] >= tau - 1e-12
        x0, y_0, x1, y1 = (int(round(v * k)) for v in d["box"])
        cv2.rectangle(img, (x0, y_0), (x1, y1), YOLO, 7 if thick else 2)
        text.add((inside_x(x0 + 6, label), y0 + y_0 + 4), label, 20, YOLO)
        drawn.append({"box": list(d["box"]), "label": label, "thick": bool(thick)})
    return img, drawn


# --- the timeline (under the map) ------------------------------------------------------------------------------------
def step_x(t: int, n: int) -> int:
    return int(round(TL[0] + 10 + (TL[2] - 20) * t / max(n, 1)))


def tl_label(e: ve.Event) -> str:
    """The short timeline label of an event (the top band has the full text, same colour)."""
    return {"sight": f"sight {e.step}", "box": f"YOLOE box {e.step}", "hit": f"YOLOE box on target {e.step}",
            "oracle": f"view spot {e.step}", "rule": f"stop {e.step}: {e.how}"}[e.kind]


def timeline_base(n: int, rule_values: List[float], tau: float, sees: List[bool], hits: List[bool],
                  events: List[ve.Event]) -> np.ndarray:
    """The static timeline picture (TL size, full frame coordinates shifted by TL's corner): rule value + tau line,
    truth-sees strip, YOLOE-hit strip, event markers with labels."""
    x0, y0, w, h = TL
    img = np.zeros((h, w, 3), np.uint8)
    text = Writer()
    text.add((4, 2), f"timeline: whole run, {n} steps ({n * STEP_S / 60:.1f} min)", 20, (200, 200, 200))
    top, bot = 40, 80
    sx = lambda t: step_x(t, n) - x0  # noqa: E731
    ty = int(round(bot - (bot - top) * tau))
    for x in range(sx(0), sx(n), 10):
        cv2.line(img, (x, ty), (x + 5, ty), HIGH_C, 1)
    pts = np.array([[sx(t), int(round(bot - (bot - top) * min(max(v, 0.0), 1.0)))] for t, v in enumerate(rule_values)],
                   np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, YOLO, 2, cv2.LINE_AA)
    text.add((sx(n) - 92, ty - 24), f"tau {tau:.2f}", 18, HIGH_C)
    for flags, (ya, yb), colour in ((sees, (84, 94), SIGHT_C), (hits, (97, 107), YOLO)):
        cv2.rectangle(img, (sx(0), ya), (sx(n), yb), (45, 45, 45), -1)
        for t, f in enumerate(flags):
            if f:
                cv2.rectangle(img, (sx(t), ya), (max(sx(t + 1) - 1, sx(t)), yb), colour, -1)
    rows: List[int] = []  # right end of the last label in each row
    for e in events:
        colour = HOW_C[e.how] if e.kind == "rule" else KIND_C[e.kind]
        x = sx(e.step)
        cv2.line(img, (x, top - 4), (x, 110), colour, 3)
        label = tl_label(e)
        wpx = int(text.fonts.setdefault(18, font(18)).getlength(label)) + 12
        lx = min(max(x - wpx // 2, 0), w - wpx)
        r = next((k for k, end in enumerate(rows) if lx > end), len(rows))
        if r == len(rows) and r < 4:
            rows.append(0)
        r = min(r, len(rows) - 1)  # 4 label rows at most
        rows[r] = lx + wpx
        text.add((lx, 113 + 22 * r), label, 18, colour)
    return text.draw(img)


def draw_cursor(frame: np.ndarray, t: int, n: int) -> None:
    x = step_x(t, n)
    y = TL[1]
    cv2.line(frame, (x, y + 36), (x, y + 110), (255, 255, 255), 2)
    cv2.fillPoly(frame, [np.array([[x - 7, y + 28], [x + 7, y + 28], [x, y + 38]], np.int32)], (255, 255, 255))


# --- the legend (under the camera) ----------------------------------------------------------------------------------
def legend_panel(category: str, prompt: str, tau: float) -> np.ndarray:
    """Colour legend (960 x 260): camera + timeline on the left, the storey map on the right."""
    w, h = CAM[0], SIZE[1] - HEAD - CAM[1]
    img = np.zeros((h, w, 3), np.uint8)
    text = Writer()
    left = [("box2", YOLO, f"YOLOE box + raw score (‘{category} 0.42’)"), ("box7", YOLO, f"YOLOE box, map value ≥ tau {tau:.2f}"),
            ("box3", TRUE_C, f"TRUE {category} #id (simulator truth)"), ("strip", SIGHT_C, "timeline: sim truth sees a target"),
            ("strip", YOLO, "timeline: YOLOE box on a true target"), ("line", YOLO, "timeline: rule value; dashed = tau"),
            ("none", None, "IoU ≥ 0.3 = on a true target"), ("none", None, f"YOLOE prompt: ‘{prompt}’")]
    right = [("fill", FLOOR_PX, "floor, not seen yet"), ("fill", EXPL_PX, "explored"), ("fill", OBST_PX, "obstacle / wall"),
             ("grad", None, f"target layer, 0 … {tau:.2f}"), ("fill", HIGH_C, f"target value ≥ tau {tau:.2f}"),
             ("fill", READY_C, "rule-ready: ≥ tau, seen ≥ 2 views"), ("vp", None, "true object + view spots (0.2 m)"),
             ("robot", None, "path, robot; ○ start"), ("rule", None, "rule's cell now; ✕ = stop cell")]
    for col, items in ((0, left), (1, right)):
        x = 12 + col * 488
        for k, (kind, colour, name) in enumerate(items):
            y = 8 + k * 28
            if kind.startswith("box"):
                cv2.rectangle(img, (x, y + 2), (x + 30, y + 20), colour, int(kind[3:]) // 2 + 1)
            elif kind == "strip":
                cv2.rectangle(img, (x, y + 6), (x + 30, y + 16), colour, -1)
            elif kind == "line":
                cv2.line(img, (x, y + 15), (x + 30, y + 6), colour, 2)
            elif kind == "fill":
                cv2.rectangle(img, (x, y + 2), (x + 30, y + 20), colour, -1)
            elif kind == "grad":
                for i in range(31):
                    a = 0.35 + 0.65 * i / 30
                    c = tuple(int(EXPL_PX[q] * (1 - a) + YOLO[q] * a) for q in range(3))
                    cv2.line(img, (x + i, y + 2), (x + i, y + 20), c, 1)
            elif kind == "vp":
                cv2.circle(img, (x + 8, y + 11), 7, VP_C, -1)
                cv2.rectangle(img, (x + 16, y + 2), (x + 30, y + 20), TRUE_C, 2)
            elif kind == "robot":
                cv2.line(img, (x, y + 11), (x + 14, y + 11), PATH_C, 2)
                cv2.fillPoly(img, [np.array([[x + 30, y + 11], [x + 18, y + 4], [x + 18, y + 18]], np.int32)], (255, 255, 255))
            elif kind == "rule":
                cv2.circle(img, (x + 8, y + 11), 7, RULE_C, 2)
                cv2.drawMarker(img, (x + 24, y + 11), RULE_C, cv2.MARKER_TILTED_CROSS, 14, 3)
            text.add((x + 40 if kind != "none" else x, y), name, 20)
    return text.draw(img)


# --- one frame --------------------------------------------------------------------------------------------------------
def caption(text: Writer, episode: Dict[str, Any], name: str, t: int, n: int, events: List[ve.Event]) -> None:
    text.add((12, 8), f"{episode['episode_id']}  |  home {episode['home']}  |  find: ‘{episode['target']['category']}’"
                      f"  |  {name}", 30)
    x = text.add((12, 56), f"step {t} / {n}   time {t * STEP_S:6.2f} s", 26, (200, 200, 200)) + 30
    shown = [e for e in events if t >= e.step]
    one_row = x + sum(font(26).getlength(e.text) + 26 for e in shown) <= SIZE[0] - 12
    px, y, x0 = (26, 56, x) if one_row else (20, 50, x)  # too many events: two rows of smaller text
    for e in shown:
        if x + font(px).getlength(e.text) > SIZE[0] - 12:
            x, y = x0, y + 24
        x = text.add((x, y), e.text, px, HOW_C[e.how] if e.kind == "rule" else KIND_C[e.kind]) + px


def freeze_frame(frame: np.ndarray, kind: str, lines: List[str], colour: Tuple[int, int, int]) -> np.ndarray:
    """A scored moment: everything but the camera and the map dimmed, coloured frames around both, a large caption
    over the (dimmed) legend."""
    out = (frame.astype(np.float32) * 0.35).astype(np.uint8)
    cy = HEAD
    out[cy:cy + CAM[1], :CAM[0]] = frame[cy:cy + CAM[1], :CAM[0]]
    mx, my = MAP_XY
    out[my:my + MAP, mx:mx + MAP] = frame[my:my + MAP, mx:mx + MAP]
    cv2.rectangle(out, (0, cy), (CAM[0] - 1, cy + CAM[1] - 1), colour, 8)
    cv2.rectangle(out, (mx - 4, my - 4), (mx + MAP + 3, my + MAP + 3), colour, 6)
    y = cy + CAM[1]
    cv2.rectangle(out, (0, y), (CAM[0] - 1, SIZE[1] - 1), (15, 15, 15), -1)
    cv2.rectangle(out, (0, y), (CAM[0] - 1, SIZE[1] - 1), colour, 6)
    text = Writer()
    text.add((24, y + 18), lines[0], 44, colour)
    for k, line in enumerate(lines[1:]):
        text.add((24, y + 84 + 40 * k), line, 26)
    return text.draw(out)


def dim_after(frame: np.ndarray, step: int) -> np.ndarray:
    """A frame after the oracle success: everything below the top band dimmed, a note at the top of the camera."""
    out = frame.copy()
    out[HEAD:] = (out[HEAD:].astype(np.float32) * DIM).astype(np.uint8)
    cv2.rectangle(out, (0, HEAD), (CAM[0] - 1, HEAD + 40), (15, 15, 15), -1)
    text = Writer()
    text.add((12, HEAD + 8), f"after the scored moment (at a target view spot, step {step}): dimmed", 22, ORACLE_C)
    return text.draw(out)


def freeze_lines(e: ve.Event, rule_value: float, tau: float, category: str) -> Tuple[List[str], Tuple[int, int, int]]:
    """Caption and colour of a scored moment."""
    if e.kind == "oracle":
        return ["at a target view spot (sim truth)",
                f"step {e.step}: walking distance ≤ 0.2 m to a view",
                f"point of a true {category}",
                "oracle success (the robot does not stop here by itself)"], ORACLE_C
    return [f"rule stop: {e.how}",
            f"step {e.step}: map value {rule_value:.3f} ≥ tau {tau:.2f}",
            "(best cell ≤ 1.0 m ahead, ±30°, seen from ≥ 2 views)",
            {"right object": "the robot is at a target view spot: success",
             "wrong object": "the triggering cell is > 1.0 m from every true target box",
             "near miss": "not at a view spot, but within 1.2 m walking distance",
             "outside": "not at a view spot and no true target nearby"}[e.how]], HOW_C[e.how]


# --- render -----------------------------------------------------------------------------------------------------------
def render_one(service: SimService, boxes: TrueBoxes, layout: RunLayout, table: ArmTable, arm: str,
               episode: Dict[str, Any], k: int, args: argparse.Namespace, tau: float, vname: str) -> Dict[str, Any]:
    """One video <out>/<vname>.mp4 + its small copy + its manifest (module docstring); returns the manifest."""
    from vint_train.mapmad.config import load_config
    from vint_train.mapmad.map_v2 import target_at

    t0 = time.time()
    rid = run_id(episode["episode_id"], k)
    log_path, dets_path = layout.log(arm, rid), layout.dets(arm, rid)
    name = arm_names(table, arm)[1]
    log = steplog.read(log_path)
    dets = ve.read_dets(dets_path)
    steps_log = steplog.steps(log)
    n_max = int(log[0]["max_steps"])
    if sorted(dets) != list(range(len(steps_log) + 1)):
        raise SystemExit(f"{dets_path}: {len(dets)} frames, the log has {len(steps_log) + 1}")
    run = layout.cfg["run"]
    score = score_run(log, episode, run, tau)
    w = arm_wire(table, arm, load_config, (RECOG_HOST, args.recog_port), tau)
    rep = replay(service, boxes, log_path, episode, w, int(run["rule_stop"]["min_views"]))
    steps = rep["steps"]
    hits = [ve.hit(dets[t], steps[t]["true"]) for t in range(len(steps))]
    hit_step = next((t for t, h in enumerate(hits) if h), None)
    box_step = ve.first_box({t: dets[t] for t in range(len(steps))})
    events = ve.timeline_events(score, tau, hit_step, box_step)
    rule_values = [float(log[0].get("start_rule", {}).get("value") or 0.0)] + [float(s["rule_value"] or 0.0) for s in steps_log]
    rule_cells = [None] + [s.get("rule_cell") for s in steps_log]
    sees = [bool(log[0]["start"].get("in_view"))] + [bool(s["in_view"]) for s in steps_log]
    prompt, cat = prompt_of(episode), episode["target"]["category"]
    fv = FloorView(rep["tmap"], MAP, rep["origin"], rep["res"], rep["shape"])
    tl = timeline_base(n_max, rule_values, tau, sees, hits, events)
    legend = legend_panel(cat, prompt, tau)
    rule_ev = next((e for e in events if e.kind == "rule"), None)
    oracle_ev = next((e for e in events if e.kind == "oracle"), None)
    stop_cell = rule_cells[rule_ev.step] if rule_ev is not None else None
    freezes = {e.step: [] for e in events if e.kind in ("oracle", "rule")}
    for e in events:
        if e.kind in ("oracle", "rule"):
            freezes[e.step].append(e)
    out = Path(args.out) / f"{vname}.mp4"
    vw = video.VideoWriter(out, SIZE, FPS, crf=CRF, faststart=True)
    table_rows, drawn_all, posed = [], {}, []
    n_cells = rep["shape"][0] * rep["shape"][1]
    obst_f, expl_f, upd = rep["obstacle_first"], rep["explored_first"], rep["updates"]
    path = [log[0]["start"]["position"]]
    frame = shown = None
    for t, st in enumerate(steps):
        if t > 0:
            path.append(steps_log[t - 1]["position"])
        yaw = float(log[0]["start"]["yaw"] if t == 0 else steps_log[t - 1]["yaw"])
        value = target_at(upd["cell"], upd["frame"], upd["value"], t, rep["shape"]).ravel()
        ready = np.unpackbits(st["ready_bits"], count=n_cells).astype(bool)
        obst, expl = (obst_f >= 0) & (obst_f <= t), (expl_f >= 0) & (expl_f <= t)
        base = layer_image(fv, obst, expl, value, ready, tau)
        rc = rule_cells[t]
        mp = map_panel(fv, base, episode, path, yaw, rc, stop_cell if (rule_ev and t >= rule_ev.step) else None)
        frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
        text = Writer()
        cam, drawn = camera_panel(st["rgb"], dets[t], st["true"], cat, tau, text, HEAD)
        frame[HEAD:HEAD + CAM[1], :CAM[0]] = cam
        frame[HEAD + CAM[1]:, :CAM[0]] = legend
        mx, my = MAP_XY
        frame[my:my + MAP, mx:mx + MAP] = mp
        frame[TL[1]:TL[1] + TL[3], TL[0]:TL[0] + TL[2]] = tl
        draw_cursor(frame, t, n_max)
        text.add((mx + 8, my + 6), "whole storey, north up (fixed)", 20)
        text.add((mx + 16, my + MAP - 46), "1 m", 18)
        caption(text, episode, name, t, n_max, events)
        frame = text.draw(frame)
        dimmed = oracle_ev is not None and t > oracle_ev.step
        shown = dim_after(frame, oracle_ev.step) if dimmed else frame
        for _ in range(REPEAT):
            vw.write(shown)
        table_rows.append([t, REPEAT, "step_dimmed" if dimmed else "step"])
        drawn_all[str(t)] = drawn
        posed.append({"position": st["pose"][0], "yaw": st["pose"][1], "hole": st["hole"]})
        for e in freezes.get(t, []):
            lines, colour = freeze_lines(e, rule_values[t], tau, cat)
            fz = freeze_frame(frame, e.kind, lines, colour)
            for _ in range(FREEZE):
                vw.write(fz)
            table_rows.append([t, FREEZE, f"freeze_{e.kind}"])
    for _ in range(FREEZE):  # hold the last frame 2 s
        vw.write(shown)
    table_rows.append([len(steps) - 1, FREEZE, "hold"])
    vw.close()
    small = Path(args.out) / "small" / out.name
    small.parent.mkdir(parents=True, exist_ok=True)
    import imageio_ffmpeg

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(out), "-vf",
                    f"scale='min({SMALL_W},iw)':-2", "-c:v", "libx264", "-crf", str(SMALL_CRF), "-movflags", "+faststart",
                    "-an", str(small)], check=True)
    replay_dets_same = sum(st["replay_dets"] == dets[t] for t, st in enumerate(steps) if st["replay_dets"] is not None)
    replay_rule_same = sum(st["replay_rule"] is not None and round(float(st["replay_rule"]), 6) == rule_values[t]
                           for t, st in enumerate(steps))
    man = {"video": out.name, "small": f"small/{out.name}", "arm": arm, "arm_name": name, "episode_id": episode["episode_id"],
           "seed_index": k, "category": cat, "prompt": prompt, "tau": tau, "config": str(args.config), "subdir": args.subdir,
           "log": str(log_path), "log_sha256": sha256(log_path), "dets": str(dets_path), "dets_sha256": sha256(dets_path),
           "max_steps": n_max, "steps": len(steps) - 1, "fps": FPS, "repeat": REPEAT, "freeze_frames": FREEZE,
           "size": list(SIZE), "crf": CRF, "small_crf": SMALL_CRF, "iou_hit": ve.IOU_HIT,
           "events": [e.__dict__ for e in events], "first_yoloe_box_any": box_step, "first_yoloe_box_on_target": hit_step,
           "dimmed_after_step": oracle_ev.step if oracle_ev is not None else None,
           "score": {k_: score.get(k_) for k_ in ("oracle_success", "oracle_step", "first_sight")},
           "rule": (score.get("rule") or {}).get(f"{tau:.2f}"),
           "replay_check": {"frames": len(steps), "dets_equal_frames": int(replay_dets_same),
                            "rule_value_equal_frames": int(replay_rule_same),
                            "rule_value_max_abs_diff": round(max((abs(float(st["replay_rule"]) - rule_values[t])
                                                                  for t, st in enumerate(steps)
                                                                  if st["replay_rule"] is not None), default=0.0), 6),
                            "dets_count_equal_frames": int(sum(st["replay_dets"] is not None and
                                                               len(st["replay_dets"]) == len(dets[t])
                                                               for t, st in enumerate(steps)))},
           "frame_table": table_rows, "frames_total": int(sum(r[1] for r in table_rows)), "drawn": drawn_all,
           "true_boxes": {str(t): st["true"] for t, st in enumerate(steps)}, "poses": posed,
           "render_s": round(time.time() - t0, 1), "bytes": out.stat().st_size, "small_bytes": small.stat().st_size}
    write_text_atomic(Path(args.out) / f"{vname}{MANIFEST}", json.dumps(man) + "\n")
    return man


def video_plan(layout: RunLayout, list_path: Optional[Path] = None,
               arms: Optional[List[str]] = None) -> Tuple[List[Tuple[str, str]], List[str]]:
    """The config's `videos` block: ([(pick name, episode id)] of videos.list (below out_dir), the arms rendered =
    videos.render_arms if given (Phase 13: the 4 review arms; budget), else videos.arms). list_path / arms override
    both (a pick-rules list)."""
    v = layout.cfg["videos"]
    lst = json.loads((Path(list_path) if list_path else layout.base / v["list"]).read_text())
    return [(p["name"], p["episode"]) for p in lst["picks"]], list(arms or v.get("render_arms") or v["arms"])


def render_all(args: argparse.Namespace, jobs: List[Tuple[str, str, str]]) -> None:
    """Render the jobs (video name, arm, episode id) of seed index args.seed_index from <out_dir>/<subdir> (resumable:
    a video whose manifest names the same log sha256 is skipped; a missing log or dets file -> <name>.missing.json, no
    video, listed by `check`)."""
    layout = RunLayout.load(args.config, subdir=args.subdir)
    table = ArmTable.from_layout(layout)
    tau = float(table.choices()["tau"])
    eps = {e["episode_id"]: e for e in load_episodes(layout.base / layout.cfg["sets"][args.set]["folder"] / "episodes.json")[0]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_run_info(out, {"args": {k: str(v) for k, v in vars(args).items()}, "phase_config": layout.cfg}, 0,
                  name="run_info.render.json")
    boxes = TrueBoxes(args.gpu)
    service = SimService(boxes)
    try:
        for vname, arm, e_id in jobs:
            rid = run_id(e_id, args.seed_index)
            man_f, miss_f = out / f"{vname}{MANIFEST}", out / f"{vname}{MISSING}"
            log_path, dets_path = layout.log(arm, rid), layout.dets(arm, rid)
            if not log_path.exists() or not dets_path.exists():
                write_text_atomic(miss_f, json.dumps({"video": f"{vname}.mp4", "arm": arm, "episode_id": e_id,
                                                      "seed_index": args.seed_index, "log": str(log_path),
                                                      "log_exists": log_path.exists(), "dets": str(dets_path),
                                                      "dets_exists": dets_path.exists()}) + "\n")
                print(json.dumps({"video": vname, "skipped": "missing run log or dets file"}), flush=True)
                continue
            if miss_f.exists():
                miss_f.rename(out / f".{vname}{MISSING}.old")  # the run exists now (never deleted: kept hidden)
            if man_f.exists() and (out / f"{vname}.mp4").exists() and \
                    json.loads(man_f.read_text())["log_sha256"] == sha256(log_path):
                print(json.dumps({"video": vname, "skipped": "finished"}), flush=True)
                continue
            man = render_one(service, boxes, layout, table, arm, eps[e_id], args.seed_index, args, tau, vname)
            print(json.dumps({k: man[k] for k in ("video", "steps", "frames_total", "render_s", "bytes", "small_bytes",
                                                  "events", "replay_check")}), flush=True)
    finally:
        service.shutdown()


def cmd_render(args: argparse.Namespace) -> None:
    """Videos <episode>__s<k>__<arm>_<file name>.mp4 for --episodes x --arms."""
    table = ArmTable.from_layout(RunLayout.load(args.config))
    args.set = args.set or "exam"
    render_all(args, [(video_name(e, args.seed_index, arm_names(table, a)[0]), a, e)
                      for e in args.episodes for a in args.arms])


def cmd_render_plan(args: argparse.Namespace) -> None:
    """Videos <pick name>__<arm>.mp4 for every pick of videos.list x every arm of videos.arms (set `video` unless
    --set is given)."""
    render_all(args, plan_jobs(args, *(int(x) for x in args.shard.split("/"))))


def plan_jobs(args: argparse.Namespace, k: int = 0, n: int = 1) -> List[Tuple[str, str, str]]:
    """render-plan's jobs (video name <pick>__<arm>, arm, episode id); --shard k/n = every n-th job from k."""
    picks, arms = video_plan(RunLayout.load(args.config), args.list, args.arms)
    args.set = args.set or "video"
    jobs = [(f"{name}__{a}", a, e) for name, e in picks for a in arms]
    return [j for i, j in enumerate(jobs) if i % n == k]


def cmd_pick_rules(args: argparse.Namespace) -> None:
    """Phase 12's four rule picks (p12_videos.cmd_pick) for another maps / maps-hidden pair on one set: the first
    episode of a seeded order (--seed) where (a) maps succeeds and hidden fails (oracle), (b) maps has a stuck streak,
    (c) maps' rule stop (tau) is on a wrong object, (d) F succeeds and maps fails. Scores of seed index 0 from
    <out_dir>/<subdir>/scores; F's from --f-scores (a scores file). Writes --out (a video list for render-plan --list)."""
    layout = RunLayout.load(args.config)
    tau = f"{float(ArmTable.from_layout(layout).choices()['tau']):.2f}"
    eps, fp = load_episodes(layout.base / layout.cfg["sets"][args.set]["folder"] / "episodes.json")
    ids = [e["episode_id"] for e in eps]
    order = [ids[int(i)] for i in np.random.default_rng(args.seed).permutation(len(ids))]

    def scores(f: Path) -> Dict[str, Dict[str, Any]]:
        rows = [json.loads(x) for x in f.read_text().splitlines()]
        return {r["episode_id"]: r for r in rows if r["seed_index"] == 0}

    sdir = layout.base / args.subdir / "scores"
    mp, hd, fs = scores(sdir / f"{args.maps}.jsonl"), scores(sdir / f"{args.hidden}.jsonl"), scores(args.f_scores)

    def first(pred) -> Optional[str]:
        return next((e for e in order if e in mp and pred(e)), None)

    rules = {"maps_ok_hidden_fails": first(lambda e: mp[e]["oracle_success"] and e in hd and not hd[e]["oracle_success"]),
             "maps_stuck": first(lambda e: mp[e]["stuck_streaks"] > 0),
             "maps_wrong_object": first(lambda e: mp[e]["rule"][tau]["wrong_object"]),
             "F_ok_maps_fails": first(lambda e: e in fs and fs[e]["oracle_success"] and not mp[e]["oracle_success"])}
    out = {"seed": args.seed, "tau": tau, "set": args.set, "fingerprint": fp, "subdir": args.subdir, "seed_index": 0,
           "maps": args.maps, "hidden": args.hidden, "f_scores": str(args.f_scores),
           "picks": [{"name": f"rule_{n}", "episode": e, "why": n} for n, e in rules.items() if e is not None],
           "rules_without_episode": [n for n, e in rules.items() if e is None]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(args.out, json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


# --- check ------------------------------------------------------------------------------------------------------------
def step_of_frame(table_rows: List[List[Any]], f: int) -> Tuple[int, str]:
    """The step (and the kind of frame) shown on video frame f (0-based) by the manifest's frame table."""
    for step, count, kind in table_rows:
        if f < count:
            return int(step), kind
        f -= count
    raise IndexError("frame beyond the video")


def check_boxes(man: Dict[str, Any], dets: Dict[int, List[Dict[str, Any]]], index: int) -> Dict[str, Any]:
    """Check 1 on CHECK_FRAMES seeded random frames of one video."""
    rng = np.random.default_rng(CHECK_SEED + index)
    frames = sorted(int(f) for f in rng.choice(man["frames_total"], size=min(CHECK_FRAMES, man["frames_total"]), replace=False))
    bad, rows = [], []
    for f in frames:
        t, kind = step_of_frame(man["frame_table"], f)
        why = ve.compare_boxes(man["drawn"][str(t)], dets[t], man["category"], man["tau"])
        rows.append({"frame": f, "step": t, "kind": kind, "boxes": len(man["drawn"][str(t)]), "ok": why is None})
        if why is not None:
            bad.append({"frame": f, "step": t, "why": why})
    return {"seed": CHECK_SEED + index, "frames": rows, "mismatches": bad, "pass": not bad}


def check_poses(man: Dict[str, Any], log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check 2: every replayed pose = the logged pose (module docstring)."""
    logged = ve.poses(log)
    rep = man["poses"]
    bad = []
    if len(rep) != len(logged):
        bad.append(f"{len(rep)} replayed poses, {len(logged)} logged")
    for t, (r, (p, yaw)) in enumerate(zip(rep, logged)):
        tol = 1e-6 if t == 0 else 0.0
        if np.max(np.abs(np.asarray(r["position"]) - np.asarray(p))) > tol or abs(wrap_angle(r["yaw"] - yaw)) > max(tol, 1e-12):
            bad.append(t)
    holes = [t for t, r in enumerate(rep) if r["hole"]]
    return {"frames": len(rep), "pose_mismatches": bad, "hole_frames": holes, "kept_floor_height": len(holes),
            "pass": not bad}


def cmd_check(args: argparse.Namespace) -> None:
    out = Path(args.out)
    mans = sorted(out.glob(f"*{MANIFEST}"))
    missing = [json.loads(p.read_text()) for p in sorted(out.glob(f"*{MISSING}"))]
    rows, ok = [], bool(mans)
    for index, mf in enumerate(mans):
        man = json.loads(mf.read_text())
        log_p, dets_p = Path(man["log"]), Path(man["dets"])
        r = {"video": man["video"], "index": index, "log_sha256_matches": sha256(log_p) == man["log_sha256"],
             "dets_sha256_matches": sha256(dets_p) == man["dets_sha256"], "replay_check": man["replay_check"]}
        r["boxes"] = check_boxes(man, ve.read_dets(dets_p), index)
        r["poses"] = check_poses(man, steplog.read(log_p))
        r["pass"] = r["boxes"]["pass"] and r["poses"]["pass"] and r["log_sha256_matches"] and r["dets_sha256_matches"]
        ok &= r["pass"]
        rows.append(r)
        print(json.dumps({"video": man["video"], "pass": r["pass"], "boxes": r["boxes"]["pass"],
                          "poses": r["poses"]["pass"], "kept_floor_height": r["poses"]["kept_floor_height"],
                          "replay_check": man["replay_check"]}), flush=True)
    # with --config: every planned video (render-plan's jobs) must exist (a crashed render shard is a failure)
    have = {r["video"] for r in rows} | {m["video"] for m in missing}
    not_rendered = [f"{v}.mp4" for v, _, _ in plan_jobs(args) if f"{v}.mp4" not in have] if args.config else []
    ok &= not not_rendered
    res = {"pass": ok, "videos": len(rows), "check_seed": CHECK_SEED, "frames_per_video": CHECK_FRAMES,
           "planned": len(have) + len(not_rendered) if args.config else None, "not_rendered": not_rendered,
           "kept_floor_height_total": sum(r["poses"]["kept_floor_height"] for r in rows),
           "not_reproducible": [dict(m, verdict="not reproducible", why="run log or dets file missing: no video")
                                for m in missing],
           "summary": f"{sum(r['pass'] for r in rows)}/{len(rows)} videos pass (boxes + poses); "
                      f"{len(missing)} not rendered (missing run: not reproducible)"
                      + (f"; {len(not_rendered)} planned videos not rendered (render failed)" if not_rendered else ""),
           "rows": rows}
    report = Path(args.report) if args.report else out / "video_check.json"
    write_text_atomic(report, json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("pass", "videos", "kept_floor_height_total", "summary")}), flush=True)
    raise SystemExit(0 if ok else 3)


# --- same-run ---------------------------------------------------------------------------------------------------------
def cmd_same_run(args: argparse.Namespace) -> None:
    """Re-run logs vs Phase 12 logs (module docstring); arms / episodes from --arms / --episodes, else from the
    config's videos block."""
    arms, episodes = args.arms, args.episodes
    if not arms or not episodes:
        if args.config is None:
            raise SystemExit("same-run: give --arms and --episodes, or --config with a videos block")
        picks, plan_arms = video_plan(RunLayout.load(args.config))
        arms = arms or plan_arms
        episodes = episodes or list(dict.fromkeys(e for _, e in picks))
    rows, ok = [], True
    for arm in arms:
        for e_id in episodes:
            name = f"{run_id(e_id, args.seed_index)}.jsonl"
            fn, fo = Path(args.new) / arm / name, Path(args.old) / arm / name
            if not fn.exists() or not fo.exists():
                r = {"verdict": "not reproducible", "why": f"missing log: {fn if not fn.exists() else fo}"}
            else:
                r = ve.compare_runs(steplog.read(fn), steplog.read(fo))
                r["byte_identical"] = fn.read_bytes() == fo.read_bytes()  # as `cmp`
            r.update(arm=arm, episode_id=e_id, seed_index=args.seed_index, new=str(fn), old=str(fo))
            ok &= r["verdict"] == "reproducible"
            rows.append(r)
            print(json.dumps(r), flush=True)
    bad = [f"{r['arm']} {r['episode_id']} (step {r.get('first_differing_step')})" if "why" not in r
           else f"{r['arm']} {r['episode_id']} (missing)" for r in rows if r["verdict"] != "reproducible"]
    n_ok = len(rows) - len(bad)
    summary = f"{n_ok}/{len(rows)} reproducible" + (f"; not: {', '.join(bad)}" if bad else "") + \
        f"; byte-identical logs: {sum(bool(r.get('byte_identical')) for r in rows)}/{len(rows)}"
    res = {"pass": ok, "summary": summary, "all_reproducible": ok,
           "rule": "start pose + position and yaw of every step equal (exact) over the common prefix of steps; a "
                   "shorter run alone is no difference (prefix_steps reported); byte_identical = the files are equal",
           "rows": rows}
    print(summary, flush=True)
    write_text_atomic(Path(args.out), json.dumps(res, indent=1) + "\n")
    raise SystemExit(0 if ok else 3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["render", "render-plan", "check", "same-run", "pick-rules"])
    p.add_argument("--config", type=Path, help="check: also require every render-plan video (with --list / --arms)")
    p.add_argument("--list", type=Path, help="render-plan / check: a video list in place of videos.list")
    p.add_argument("--maps", help="pick-rules: the maps-shown arm")
    p.add_argument("--hidden", help="pick-rules: the maps-hidden arm")
    p.add_argument("--f-scores", type=Path, help="pick-rules: F's scores file on the set")
    p.add_argument("--seed", type=int, default=12501, help="pick-rules: the seeded episode order (Phase 12's)")
    p.add_argument("--subdir", help="render: the runs' subdir below out_dir (logs/ and dets/)")
    p.add_argument("--set", help="render / render-plan: the config's set of the episodes (default exam / video)")
    p.add_argument("--arms", nargs="+")
    p.add_argument("--episodes", nargs="+")
    p.add_argument("--seed-index", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--recog-port", type=int)
    p.add_argument("--out", type=Path, required=True, help="render/check: the video folder; same-run: the JSON report")
    p.add_argument("--shard", default="0/1", help="render-plan: k/n = only videos k, k + n, ... (one call per GPU, "
                   "each with its own sidecar and --out the same folder)")
    p.add_argument("--report", type=Path, help="check: the JSON report (default <out>/video_check.json)")
    p.add_argument("--new", type=Path, help="same-run: logs root of the re-runs (<root>/<arm>/<episode>__s<k>.jsonl)")
    p.add_argument("--old", type=Path, help="same-run: logs root of the Phase 12 runs (read only)")
    args = p.parse_args()
    {"render": cmd_render, "render-plan": cmd_render_plan, "check": cmd_check,
     "same-run": cmd_same_run, "pick-rules": cmd_pick_rules}[args.cmd](args)


if __name__ == "__main__":
    main()

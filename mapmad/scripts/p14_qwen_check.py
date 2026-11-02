"""Phase 14a step 14.2: the Qwen3-VL check on top of the chosen Rule 2 stop (kickoff 14.2, p14a_rules.md §5), re-scored
on the logged practice runs (driving continues as logged after a veto: every run drove 1,000 steps). Inside
naz_mapmad_habitat, from /app/visualnav-transformer, with one Qwen3-VL server (mapmad_recog.server --model qwen3vl)
per worker:

    python mapmad/scripts/p14_qwen_check.py --config mapmad/configs/p14a_finish.yaml --subdir runs/baseline \
        --arms M21b M21bh N F VF --qwen-port 5990 --gpu 0 --worker 0 --workers 2

Per run, replaying its stop evidence (mapmad_sim/stop_rules.py) with the Rule 2 pick (stopbench/STOP_PICK.json):
- when Rule 2 fires on a cell (not within 0.5 m of a rejected spot), the robot asks about it at the first frame from
  that step on whose view holds the cell (within `wait_steps` = 20 steps; none -> it stops at the firing step);
- the question (kickoff): "Is there a {word} inside the marked box? Answer yes, no, or unsure." with two pictures,
  the crop around the marked box (box x 1.5, at least 96 px) and the full frame with the box drawn; the box = the
  frame's recogniser box covering the cell with the largest map value, else the cell's projection (0.05-1.0 m above
  the floor, padded 30 %); pictures rendered at the logged pose with the arm's camera; {word} = the recogniser's
  prompt word (photo_labels.PROMPTS: e.g. sofa -> couch);
- answer (first word): yes -> stop at that step; no -> veto: the spot (the cell) is rejected, driving goes on;
  anything else = unsure -> asked again at the next frame holding the cell (within 20 steps of the last ask), up to
  3 asks; no "no" within 3 asks (or no frame within 20 steps) -> stop at the last ask's step.
Writes <out>/answers/<arm>/<run>.jsonl (every ask: step, cell, box source, answer text, ms) and
<out>/runs/<arm>/<run>.json ({"stop": [step, [x, y]] | null, "asks", "vetoes"}); resumable (a finished run is
skipped). Latency per call = the server's model time + the round trip.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mapmad_bridge import protocol, steplog  # noqa: E402
from mapmad_sim.arm_table import ArmTable  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.photo_labels import PROMPTS  # noqa: E402
from mapmad_sim.recogniser import connect  # noqa: E402
from mapmad_sim.run_layout import RunLayout, parse_run_id, write_text_atomic  # noqa: E402
from mapmad_sim.search_wire import camera_wire  # noqa: E402
from mapmad_sim.sidecar_server import limo_factory  # noqa: E402
from mapmad_sim.stop_evidence import load as load_evidence  # noqa: E402
from mapmad_sim.stop_rules import EvidenceReplay, StopSpec, rule_function  # noqa: E402

QUESTION = "Is there a {word} inside the marked box? Answer yes, no, or unsure."
WAIT_STEPS = 20
MAX_ASKS = 3


def answer_of(text: str) -> str:
    """yes | no | unsure from the model's text (first word, lower case, punctuation stripped)."""
    words = text.strip().lower().replace(".", " ").replace(",", " ").replace("!", " ").split()
    w = words[0] if words else ""
    return w if w in ("yes", "no") else "unsure"


def cell_box(cell_xy: Tuple[float, float], res: float, floor_y: float, cam_tf: np.ndarray, hfov_deg: float,
             hw: Tuple[int, int]) -> Optional[List[float]]:
    """The cell's projection (corners x heights 0.05 / 1.0 m above the floor) as a padded pixel box, or None when
    behind the camera / outside the picture."""
    h, w = hw
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    inv = np.linalg.inv(cam_tf)
    X, Y = cell_xy
    pts = []
    for dx in (-res / 2, res / 2):
        for dy in (-res / 2, res / 2):
            for hgt in (0.05, 1.0):
                p = inv @ np.array([-(Y + dy), floor_y + hgt, -(X + dx), 1.0])
                if p[2] >= -1e-3:
                    return None
                pts.append((w / 2.0 + f * p[0] / -p[2], h / 2.0 - f * p[1] / -p[2]))
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(us), max(us), min(vs), max(vs)
    px, py = 0.3 * (x1 - x0), 0.3 * (y1 - y0)
    box = [max(0.0, x0 - px), max(0.0, y0 - py), min(float(w), x1 + px), min(float(h), y1 + py)]
    return box if box[2] - box[0] >= 2 and box[3] - box[1] >= 2 else None


def crop(rgb: np.ndarray, box: Sequence[float], scale: float = 1.5, min_px: int = 96) -> np.ndarray:
    h, w = rgb.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side_x, side_y = max((box[2] - box[0]) * scale, min_px), max((box[3] - box[1]) * scale, min_px)
    x0, x1 = int(max(0, cx - side_x / 2)), int(min(w, cx + side_x / 2))
    y0, y1 = int(max(0, cy - side_y / 2)), int(min(h, cy + side_y / 2))
    return np.ascontiguousarray(rgb[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)])


def marked(rgb: np.ndarray, box: Sequence[float]) -> np.ndarray:
    out = np.ascontiguousarray(rgb.copy())
    cv2.rectangle(out, (int(box[0]), int(box[1])), (int(box[2]) - 1, int(box[3]) - 1), (255, 0, 0), 2)
    return out


class Asker:
    """Renders the logged pose and asks Qwen3-VL about one cell (module docstring)."""

    def __init__(self, robot: Any, qwen: Any, camera: Dict[str, Any], log: List[Dict[str, Any]], ev: Dict[str, Any],
                 word: str) -> None:
        self.robot, self.qwen, self.hfov = robot, qwen, float(camera["hfov_deg"])
        self.poses = [(log[0]["start"]["position"], log[0]["start"]["yaw"])] + \
            [(s["position"], s["yaw"]) for s in log if s["kind"] == "step"]
        self.ev, self.word = ev, word
        self.calls: List[Dict[str, Any]] = []

    def box(self, t: int, cell: int, xy: Tuple[float, float], res: float, rgb: np.ndarray) -> Tuple[Optional[List[float]], str]:
        ev = self.ev
        best = None
        for k in np.flatnonzero(ev["det_t"] == t).tolist():
            cells = ev["det_cells"][ev["det_cells_ptr"][k]:ev["det_cells_ptr"][k + 1]]
            if cell in cells and np.isfinite(ev["det_box"][k]).all():
                cand = (float(ev["det_value"][k]), int(ev["det_px"][k]), k)
                best = cand if best is None or cand[:2] > best[:2] else best
        if best is not None:
            return [float(v) for v in ev["det_box"][best[2]]], "recogniser"
        return cell_box(xy, res, self.robot.floor_y, self.robot.camera_transform(), self.hfov, rgb.shape[:2]), "projection"

    def ask(self, t: int, cell: int, xy: Tuple[float, float], res: float) -> str:
        pos, yaw = self.poses[t]
        self.robot.move_to(np.asarray(pos, np.float64), float(yaw))
        rgb = self.robot.observe()["rgb"]
        box, src = self.box(t, cell, xy, res, rgb)
        if box is None:
            self.calls.append({"step": t, "cell": list(xy), "box": None, "source": src, "answer": "unsure",
                               "text": "", "ms": 0.0})
            return "unsure"
        t0 = time.perf_counter()
        r = self.qwen.request({"cmd": "ask", "images": [crop(rgb, box), marked(rgb, box)],
                               "question": QUESTION.format(word=self.word)})
        ms = (time.perf_counter() - t0) * 1e3
        a = answer_of(r["text"])
        self.calls.append({"step": t, "cell": [round(xy[0], 3), round(xy[1], 3)], "box": [round(v, 1) for v in box],
                           "source": src, "answer": a, "text": r["text"], "ms": round(ms, 1),
                           "model_ms": round(float(r["model_ms"]), 1)})
        return a


def veto_stop(ev: Dict[str, Any], key: str, asker: Asker, spec: StopSpec = StopSpec()) -> Dict[str, Any]:
    """The Rule 2 stop with the Qwen check (module docstring) -> {"stop": [step, [x, y]] | None, "asks", "vetoes"}."""
    rep = EvidenceReplay(ev, spec)
    rule = rule_function(key)
    rep.advance()
    rejected: List[Tuple[float, float]] = []
    pending: Optional[Dict[str, Any]] = None  # {"cell", "xy", "fired", "asks", "last"}
    vetoes = 0
    for t in range(1, rep.frames):
        rep.advance()
        if pending is not None:
            view = ev["view"][ev["view_ptr"][t]:ev["view_ptr"][t + 1]]
            if pending["cell"] in view:
                a = asker.ask(t, pending["cell"], pending["xy"], rep.res)
                pending["asks"] += 1
                pending["last"] = t
                if a == "yes" or (a == "unsure" and pending["asks"] >= MAX_ASKS):
                    return {"stop": [t, list(pending["xy"])], "asks": len(asker.calls), "vetoes": vetoes}
                if a == "no":
                    rejected.append(pending["xy"])
                    vetoes += 1
                    pending = None
                continue
            if t - pending["last"] > WAIT_STEPS:
                return {"stop": [pending["last"], list(pending["xy"])], "asks": len(asker.calls), "vetoes": vetoes}
            continue
        if not rep.ever.any():
            continue
        f = rule(rep)
        cells = f.cells[~rep.excluded(f.cells, rejected)] if rejected else f.cells
        if cells.size == 0:
            continue
        c = int(cells[0])
        pending = {"cell": c, "xy": rep.cell_xy(c), "fired": t, "asks": 0, "last": t}
        view = ev["view"][ev["view_ptr"][t]:ev["view_ptr"][t + 1]]
        if c in view:  # ask now; otherwise wait (<= 20 steps) for a frame that sees it
            a = asker.ask(t, c, pending["xy"], rep.res)
            pending["asks"] = 1
            if a == "yes":
                return {"stop": [t, list(pending["xy"])], "asks": len(asker.calls), "vetoes": vetoes}
            if a == "no":
                rejected.append(pending["xy"])
                vetoes += 1
                pending = None
    if pending is not None:
        return {"stop": [pending["last"], list(pending["xy"])], "asks": len(asker.calls), "vetoes": vetoes}
    return {"stop": None, "asks": len(asker.calls), "vetoes": vetoes}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--out", default="qwen")
    p.add_argument("--qwen-host", default="naz_mapmad_recog")
    p.add_argument("--qwen-port", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--episodes", nargs="*", default=[])
    p.add_argument("--rule", help="a stop-rule key in place of the Rule 2 pick (DRY tests only)")
    args = p.parse_args()
    layout = RunLayout.load(args.config, subdir=args.subdir)
    key = args.rule or json.loads((layout.base / "stopbench" / "STOP_PICK.json").read_text())["rule2_pick"]
    table = ArmTable.from_layout(layout)
    eps: Dict[str, Dict[str, Any]] = {}
    for name in layout.cfg["sets"]["practice"]["parts"]:
        e, _ = load_episodes(layout.base / layout.cfg["sets"][name]["folder"] / "episodes.json")
        eps.update({x["episode_id"]: x for x in e})
    jobs = []
    for arm in args.arms:
        for f in sorted((layout.root / "logs" / arm).glob("*.jsonl")):
            rid = f.name[:-len(".jsonl")]
            eid = parse_run_id(rid)[0]
            if args.episodes and eid not in args.episodes:
                continue
            jobs.append((eps[eid]["home"], arm, rid, eid))
    jobs = sorted(jobs)[args.worker::args.workers]
    qwen = connect("qwen3vl", args.qwen_host, args.qwen_port, 0.0, 1.0, True, protocol.auth_key(), wait_s=1800)
    factory = limo_factory(args.gpu)
    robot, robot_key = None, None
    out = layout.base / args.out
    print(f"worker {args.worker}: {len(jobs)} runs, rule {key}", flush=True)
    try:
        for home, arm, rid, eid in jobs:
            res_file = out / "runs" / arm / f"{rid}.json"
            if res_file.exists():
                continue
            cam = table.camera_name(table.arm(arm))
            camera = camera_wire(cam) if cam.endswith(".yaml") else {"name": cam, **layout.cfg["cameras"][cam]}
            e = eps[eid]
            rk = (e["split"], home, json.dumps(camera, sort_keys=True))
            if rk != robot_key:
                if robot is not None:
                    robot.close()
                robot = factory(e["split"], home, float(camera["hfov_deg"]), depth=True, camera=camera)
                robot_key = rk
            log = steplog.read(layout.root / "logs" / arm / f"{rid}.jsonl")
            ev = load_evidence(layout.root / "evidence" / arm / f"{rid}.npz")
            asker = Asker(robot, qwen, camera, log, ev, e["target"].get("prompt") or PROMPTS[e["target"]["category"]])
            t0 = time.time()
            res = veto_stop(ev, key, asker)
            ans = out / "answers" / arm / f"{rid}.jsonl"
            write_text_atomic(ans, "".join(json.dumps(c) + "\n" for c in asker.calls))
            write_text_atomic(res_file, json.dumps(dict(res, rule=key, seconds=round(time.time() - t0, 1))) + "\n")
            print(json.dumps({"arm": arm, "run": rid, **res}), flush=True)
    finally:
        if robot is not None:
            robot.close()
        qwen.close()


if __name__ == "__main__":
    main()

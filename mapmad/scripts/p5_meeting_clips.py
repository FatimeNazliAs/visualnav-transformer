"""Phase 5 presentation clips from existing step logs (no new runs), built on p5_focused_video.py. Inside
naz_mapmad_habitat, from /app/visualnav-transformer:

    python mapmad/scripts/p5_meeting_clips.py --config mapmad/configs/p5_exam.yaml --episode p5e-00810-bed-0 \
        --arms C B --title "Same start: ..." --out videos/meeting/1_success_C_vs_B.mp4
    python mapmad/scripts/p5_meeting_clips.py --config mapmad/configs/p5_exam.yaml --pick-bin 1 --arms C \
        --title "... {distance:.1f} m ..." --out videos/meeting/2_success_C_long.mp4

One driver: the focused layout (p5_focused_video.full_frame). Two drivers: two 640 px columns side by side (camera;
map input with the logged plan | whole-house overview; step, walking distance to the nearest view point,
collisions, STUCK), the same start, the same clock; a finished driver holds its last frame with its SUCCESS /
FAILED stamp. Every clip: 1280 px wide, a 3 s title card (the one-line title), 12 fps = 3x real time (4 Hz
control), the last frame held 2 s, H.264 CRF 28, +faststart, no audio.
--cut-stuck-s S: a stuck streak longer than S s of clip time is shown for S s, then a 1 s card says how many steps
were skipped and the clip continues where the streak ends.
--pick-bin B: the episode = the arm's (first --arms) fastest success at the seed index in distance bin B (fewest
steps; ties by episode id); the title may use {distance} (start walking distance, m), {category}, {seconds}.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import p5_focused_video as F  # same folder (puts the repository's train/ on sys.path via p5_videos)
from mapmad_sim.run_layout import RunLayout, parse_run_id
from vint_train.mapmad.closed_loop import video

FPS = 12  # 3x real time
TITLE_S, HOLD_S, CUT_CARD_S = 3, 2, 1
COL_W = F.WIDTH // 2
HEAD, MAP_COL, TEXT_H, TOP = 40, COL_W // 2, 90, 44
SHORT_LABELS = ("map input (6.4 m)", "map (NOT given)")


def pick_fastest(layout: RunLayout, arm: str, distance_bin: int, seed_index: int) -> Tuple[str, int]:
    """(episode id, success step) of the arm's fastest successful run at seed_index in distance_bin."""
    best = []
    for p in sorted((layout.root / "summaries" / arm).glob("*.json")):
        if p.name.endswith(".crash.json") or parse_run_id(p.name)[1] != seed_index:
            continue
        r = json.loads(p.read_text())
        if r["success"] and int(r["distance_bin"]) == distance_bin:
            best.append((int(r["success_step"]), r["episode_id"]))
    if not best:
        raise SystemExit(f"no {arm} success in bin {distance_bin} at seed index {seed_index}")
    step, episode_id = min(best)
    return episode_id, step


def column(d: F.Driver, shot: F.Shot, name: str) -> np.ndarray:
    """One driver's 640 px column for a side-by-side clip."""
    head = np.zeros((HEAD, COL_W, 3), np.uint8)
    video.text(head, f"{name}: {F.P.ARM_NAMES.get(name, F.P.ARM_NAMES.get(d.arm.name, d.arm.policy))}", (10, 27), 0.65)
    if shot.step == d.last:
        s, c = F.status(d)
        video.text(head, s, (COL_W - 175, 28), 0.65, c)
    local = F.map_picture(shot.tensor, d.arm.map_given, shot.points, shot.chosen, MAP_COL, SHORT_LABELS)
    plan = F.letterbox(shot.plan, MAP_COL, MAP_COL)
    text = np.full((TEXT_H, COL_W, 3), 25, np.uint8)
    vp = "-" if shot.vp is None else f"{shot.vp:.2f} m"
    video.text(text, f"step {shot.step}   to target {vp}", (12, 36), 0.7)
    video.text(text, f"collisions {shot.collisions}", (12, 74), 0.7)
    if shot.stuck:
        cv2.rectangle(text, (COL_W - 200, 14), (COL_W - 14, 76), video.RED, -1)
        video.text(text, "STUCK", (COL_W - 178, 62), 1.3)
    col = np.vstack([head, shot.cam, np.hstack([local, plan]), text])
    cv2.line(col, (0, 0), (0, col.shape[0] - 1), (90, 90, 90), 2)
    return col


def top_bar(e: Dict, width: int) -> np.ndarray:
    bar = np.zeros((TOP, width, 3), np.uint8)
    s = (f"{e['episode_id']} | find: {e['target']['category']} | start {e['start_geodesic_m']:.1f} m walking "
         f"distance to the nearest view point | 3x speed")
    video.text(bar, s, (10, 29), F.fit_scale(s, width - 20, 0.6))
    return bar


def card(size: Tuple[int, int], s: str, largest: float = 1.0) -> np.ndarray:
    """A dark card with s centred, wrapped onto at most two rows."""
    w, h = size
    img = np.full((h, w, 3), 15, np.uint8)
    words, rows = s.split(), [s]
    if cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, largest, 2)[0][0] > w - 80 and len(words) > 1:
        cut = min(range(1, len(words)), key=lambda k: abs(len(" ".join(words[:k])) - len(s) / 2))
        rows = [" ".join(words[:cut]), " ".join(words[cut:])]
    scale = min(F.fit_scale(r, w - 80, largest) for r in rows)
    for k, r in enumerate(rows):
        (tw, th), _ = cv2.getTextSize(r, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        y = h // 2 + int((k - (len(rows) - 1) / 2) * th * 2.2) + th // 2
        cv2.putText(img, r, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, video.WHITE, 2, cv2.LINE_AA)
    return img


def skipped_ranges(drivers: Sequence[F.Driver], cut_frames: int) -> List[Tuple[int, int]]:
    """Pose ranges [a, b) cut from the clip: each streak longer than cut_frames keeps its first cut_frames poses;
    the final pose is never cut (it carries the end stamp)."""
    if cut_frames <= 0:
        return []
    end = max(d.last for d in drivers)
    out = sorted((first + cut_frames, min(first + n, end)) for d in drivers for first, n in d.streaks
                 if n > cut_frames and first + cut_frames < end)
    merged: List[Tuple[int, int]] = []
    for a, b in out:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", default="runs")
    p.add_argument("--episode", help="episode id (or --pick-bin)")
    p.add_argument("--pick-bin", type=int, help="distance bin: pick the first arm's fastest success there")
    p.add_argument("--arms", nargs="+", required=True, help="one or two config arms")
    p.add_argument("--names", nargs="*", default=[], help="display names, e.g. C_wp1=C")
    p.add_argument("--seed-index", type=int, default=0)
    p.add_argument("--title", required=True)
    p.add_argument("--cut-stuck-s", type=float, default=0.0)
    p.add_argument("--out", required=True, help="below <outputs>/<out_dir>")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    if len(args.arms) not in (1, 2) or (args.episode is None) == (args.pick_bin is None):
        raise SystemExit("one or two --arms, and exactly one of --episode / --pick-bin")
    layout = RunLayout.load(args.config, subdir=args.subdir)
    episode_id = args.episode
    if args.pick_bin is not None:
        episode_id, step = pick_fastest(layout, args.arms[0], args.pick_bin, args.seed_index)
        print(f"picked {episode_id}: {args.arms[0]} success at step {step} (bin {args.pick_bin})")
    names = dict(n.split("=", 1) for n in args.names)
    drivers = [F.Driver.load(layout, episode_id, a, args.seed_index) for a in args.arms]
    e = drivers[0].episode
    title = args.title.format(distance=e["start_geodesic_m"], category=e["target"]["category"],
                              seconds=drivers[0].last / 4.0)
    shots = [F.replay(d, args.gpu) for d in drivers]
    current: List[Optional[F.Shot]] = [None] * len(drivers)
    n = max(d.last for d in drivers) + 1
    cuts = skipped_ranges(drivers, int(args.cut_stuck_s * FPS))
    out = layout.base / args.out
    writer, img = None, None
    for i in range(n):
        for k, d in enumerate(drivers):
            if i <= d.last:
                current[k] = next(shots[k])
        cut = next(((a, b) for a, b in cuts if a <= i < b), None)
        if cut is not None:
            if i == cut[1] - 1:  # end of a skipped stretch: say so for 1 s
                note = f"... stuck streak cut: steps {cut[0]}-{cut[1] - 1} skipped ({(cut[1] - cut[0]) / 4:.0f} s real time) ..."
                for _ in range(FPS * CUT_CARD_S):
                    writer.write(card((img.shape[1], img.shape[0]), note, 0.8))
            continue
        if len(drivers) == 1:
            img = F.full_frame(drivers[0], current[0], F.focused_title(drivers[0]))
        else:
            cols = [column(d, s, names.get(d.arm.name, d.arm.name)) for d, s in zip(drivers, current)]
            img = np.vstack([top_bar(e, F.WIDTH), np.hstack(cols)])
        if writer is None:
            writer = video.VideoWriter(out, (img.shape[1], img.shape[0]), FPS, crf=F.CRF, faststart=True)
            for _ in range(FPS * TITLE_S):
                writer.write(card((img.shape[1], img.shape[0]), title))
        writer.write(img)
    for _ in range(FPS * HOLD_S - 1):
        writer.write(img)
    writer.close()
    print(f"wrote {out} ({episode_id}, arms {args.arms}, {n - 1} steps, cut {cuts or 'none'}): {title}")


if __name__ == "__main__":
    main()

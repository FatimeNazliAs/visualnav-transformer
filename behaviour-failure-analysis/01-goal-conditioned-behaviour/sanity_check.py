#!/usr/bin/env python
# 01-goal-conditioned-behaviour/sanity_check.py
"""
The small first test. Run this before anything else, and after any change to
common/.

It answers one question — "is the thing I think I loaded the thing I loaded?" —
in nine checks, and runs in seconds. It writes nothing and measures nothing
about behaviour; every experiment that follows assumes these nine facts, so they
are asserted once, here, rather than trusted everywhere.

    1  config resolves
    2  the checkpoint on disk is the one the config names   (path + md5)
    3  the weights load with no key mismatch                (0 missing / 0 unexpected)
    4  the parameter count matches the architecture         (19,049,675)
    5  the case's trajectory really is in the test split
    6  the frames load, with the documented index layout
    7  the action decode round-trips                        (pure numpy, no GPU)
    8  one denoising run produces a finite, plausibly-scaled path
    9  goal masking actually changes the context vector

Check 9 is the one that is easy to skip and expensive to get wrong: the whole
`exploration` control assumes masking does something, and NoMaD masks the goal
token out of ATTENTION rather than zeroing it at the input — so the goal token
itself comes back identical either way and only c_t moves. If that assumption
ever breaks, every control comparison silently becomes a comparison of a thing
with itself.

Usage (from the host, via run.sh, or inside the container):

    python 01-goal-conditioned-behaviour/sanity_check.py
    python 01-goal-conditioned-behaviour/sanity_check.py --traj no31vc_12_0 --frame 20
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# behaviour-failure-analysis/ onto sys.path, so `common` imports. See
# common/__init__.py for why the hyphenated folders are never imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common import actions as action_space
from common import config as config_mod
from common import data, model, settings

TASK_DIR = Path(__file__).resolve().parent

# The architecture in settings.py, fully instantiated. A mismatch here means a
# hyperparameter drifted away from the checkpoint's nomad.yaml.
EXPECTED_PARAMS = 19_049_675


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--traj", default=None,
                        help="trajectory to probe (default: first usable test-split one)")
    parser.add_argument("--frame", type=int, default=None,
                        help="current frame index (default: a safe frame mid-trajectory)")
    return parser.parse_args()


def rule(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 62 - len(title)))


def md5(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def first_usable_test_case(cfg: config_mod.ExperimentConfig) -> config_mod.Case:
    """
    The first test-split trajectory long enough to probe, at a mid frame.

    Deterministic — it walks the split file in order — so the sanity check
    always exercises the same data unless told otherwise.
    """
    reach = max(cfg.goal_distance, settings.NUM_ACTIONS)
    need = settings.CONTEXT_SIZE + reach + 2
    for traj in config_mod.load_split("test"):
        traj_dir = settings.RAW_DATA_DIR / traj
        if not traj_dir.is_dir():
            continue
        n_frames = len(list(traj_dir.glob("*.jpg")))
        if n_frames >= need:
            return config_mod.Case(
                case_id="sanity",
                traj=traj,
                frame=(settings.CONTEXT_SIZE + n_frames - reach - 1) // 2,
                goal_distance=cfg.goal_distance,
                expectation="first usable test-split trajectory; not a curated case",
            )
    raise RuntimeError("no test-split trajectory long enough to probe")


def main() -> int:
    args = parse_args()

    # ── 1. Config ─────────────────────────────────────────────────────────────
    rule("1  config")
    cfg = config_mod.load_experiment(TASK_DIR)
    print(f"  {cfg.summary()}")
    print(f"  out_dir    : {cfg.out_dir}  (nothing is written by this script)")

    # ── 2. Checkpoint identity ────────────────────────────────────────────────
    rule("2  checkpoint on disk")
    ckpt = cfg.checkpoint            # resolves lazily; raises if absent
    print(f"  path       : {ckpt}")
    print(f"  size       : {ckpt.stat().st_size:,} bytes")
    print(f"  md5        : {md5(ckpt)}")

    # ── 5. Split membership (before the slow parts) ───────────────────────────
    rule("5  case & split membership")
    if args.traj is not None:
        case = config_mod.Case(
            case_id="sanity", traj=args.traj,
            frame=args.frame if args.frame is not None else settings.CONTEXT_SIZE,
            goal_distance=cfg.goal_distance, expectation="operator-supplied")
    else:
        case = first_usable_test_case(cfg)
        if args.frame is not None:
            case = config_mod.Case(case_id=case.case_id, traj=case.traj,
                                   frame=args.frame, goal_distance=case.goal_distance,
                                   expectation=case.expectation)
    found = config_mod.split_of(case.traj)
    print(f"  {case.summary()}")
    print(f"  split      : {found}")
    assert found == "test", f"{case.traj} is in {found}, not the test split"

    # ── 3 & 4. Load the model ─────────────────────────────────────────────────
    rule("3/4  weights")
    net, info = model.load_model(cfg)
    assert info["missing_keys"] == 0, f"{info['missing_keys']} missing keys"
    assert info["unexpected_keys"] == 0, f"{info['unexpected_keys']} unexpected keys"
    assert info["n_params"] == EXPECTED_PARAMS, (
        f"parameter count {info['n_params']:,} != expected {EXPECTED_PARAMS:,} — "
        f"a hyperparameter in settings.py has drifted from the checkpoint")
    print(f"  key mismatch: {info['missing_keys']} missing / "
          f"{info['unexpected_keys']} unexpected")
    print(f"  params      : {info['n_params']:,}  (expected {EXPECTED_PARAMS:,})")

    # ── 6. Frames ─────────────────────────────────────────────────────────────
    rule("6  frames")
    scene = data.load_sample(case)
    assert len(scene.obs_raw) == settings.N_OBS_FRAMES
    assert scene.obs_idxs[-1] == case.frame, "obs[-1] must be the current frame"
    assert scene.goal_idx == case.frame + case.goal_distance
    print(f"  obs stack  : {scene.obs_input.shape}  (3 x {settings.N_OBS_FRAMES} channels)")
    print(f"  goal       : {scene.goal_input.shape}")

    truth = data.load_ground_truth(case)
    print(f"  recorded   : {truth['waypoints'].shape}, "
          f"{action_space.travelled(truth['waypoints']):.2f} m travelled")

    # ── 7. The decode round-trips (pure numpy) ────────────────────────────────
    rule("7  action decode round-trip")
    units = truth["waypoints"] / settings.METRIC_WAYPOINT_SPACING
    back = action_space.to_waypoints(action_space.to_normalised_deltas(units))
    err = float(np.abs(back - truth["waypoints"]).max())
    assert err < 1e-9, f"round trip drifted by {err}"
    print(f"  to_waypoints(to_normalised_deltas(u)) == u * {settings.METRIC_WAYPOINT_SPACING}")
    print(f"  max abs error: {err:.2e} m")

    # ── 8. One denoising run ──────────────────────────────────────────────────
    rule("8  one denoising run")
    from common import denoise as denoise_mod

    device = info["device"]
    nav = model.encode_tokens(net, scene.obs_batch, scene.goal_batch, device,
                              goal_mask=model.GOAL_VISIBLE)
    result = denoise_mod.denoise(net, nav.context, device, seed=0)
    path = result.path
    assert np.isfinite(path).all(), "non-finite waypoints"
    assert result.n_steps == settings.K_DENOISING, "wrong number of denoising steps"
    reach = float(np.linalg.norm(path[-1]))
    print(f"  descent    : {result.trajectory.shape}  (K={result.n_steps} + the noise it started from)")
    print(f"  path       : {path.shape}, {action_space.travelled(path):.2f} m travelled, "
          f"endpoint {reach:.2f} m out")
    print(f"  endpoint   : forward {path[-1, 0]:+.2f} m, left {path[-1, 1]:+.2f} m")
    assert 0.0 < reach < 10.0, (
        f"endpoint {reach:.2f} m from the robot is not a plausible {settings.NUM_ACTIONS}-step "
        f"move at {settings.METRIC_WAYPOINT_SPACING} m spacing — suspect the decode")

    dist = denoise_mod.distance_to_goal(net, nav.context, device)
    print(f"  distance   : predicted {dist:.2f} steps  (goal is {case.goal_distance} ahead)")

    # ── 9. Goal masking changes c_t, and only c_t ─────────────────────────────
    rule("9  goal masking")
    exp = model.encode_tokens(net, scene.obs_batch, scene.goal_batch, device,
                              goal_mask=model.GOAL_HIDDEN)
    ctx_shift = float(np.linalg.norm(nav.context - exp.context))
    print(f"  |c_t(nav) - c_t(explore)| = {ctx_shift:.4f}")
    assert ctx_shift > 1e-6, (
        "masking the goal did not change c_t — the exploration control would be "
        "comparing navigation against itself")
    assert np.allclose(nav.goal_token, exp.goal_token), (
        "the goal token changed under masking; NoMaD masks it out of ATTENTION, "
        "it does not zero it at the input")
    assert np.allclose(nav.obs_tokens, exp.obs_tokens), "observation tokens changed"
    print("  goal + observation tokens identical under both masks, as expected")

    rule("result")
    print("  all 9 checks passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

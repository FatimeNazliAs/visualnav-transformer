"""Phase 7 sweep on the NoMaD side: run order with camera / episode-type arms, NoMaD's context spacing option."""

import numpy as np
import torch

from mapmad_sim.exam_arms import ExamArm
from mapmad_sim.run_layout import try_claim, write_text_atomic
from vint_train.mapmad.closed_loop.nomad_policy import NomadPolicy
from vint_train.mapmad.closed_loop.policy import Observation
from vint_train.mapmad.closed_loop.run_exam import work

EPS = [{"episode_id": f"iv{i}", "type": "in_view"} for i in range(3)] + \
      [{"episode_id": f"oov{i}", "type": "out_of_view"} for i in range(2)]
B = ExamArm("B_c", "nomad", photo=True, episode_type="in_view")
A = ExamArm("A_c", "nomad", photo=False, episode_type="out_of_view")


def test_work_filters_episode_types_and_keeps_every_run():
    runs = list(work(EPS, [B, A], [0, 1, 2], 0, 1))
    assert len(runs) == 3 * 3 + 2 * 3
    assert all((arm is B) == (e["type"] == "in_view") for e, _, arm in runs)


def test_claims_hand_each_run_to_exactly_one_runner(tmp_path):
    """Dynamic hand-out: runners walking the same list in turn take disjoint runs that together cover the list; a
    new launch token starts with no claims."""
    todo = [(e["episode_id"], k, arm.name) for e, k, arm in work(EPS, [B, A], [0, 1, 2], 0, 1, arm_major=True)]
    taken = {0: [], 1: [], 2: []}
    for i, run in enumerate(todo * 3):  # every runner tries every run
        r = i % 3
        if try_claim(tmp_path / "t1" / run[2] / f"{run[0]}__s{run[1]}"):
            taken[r].append(run)
    got = [run for runs in taken.values() for run in runs]
    assert sorted(got) == sorted(todo) and len(got) == len(set(got))
    assert try_claim(tmp_path / "t2" / "B_c" / "iv0__s0")


def test_atomic_write_leaves_only_the_whole_file(tmp_path):
    path = tmp_path / "summaries" / "B_c" / "iv0__s0.json"
    write_text_atomic(path, '{"a": 1}\n')
    write_text_atomic(path, '{"a": 2}\n')
    assert path.read_text() == '{"a": 2}\n' and [q.name for q in path.parent.iterdir()] == ["iv0__s0.json"]


def test_arm_major_order_is_the_same_runs_arm_by_arm():
    minor = list(work(EPS, [B, A], [0, 1], 1, 2))
    major = list(work(EPS, [B, A], [0, 1], 1, 2, arm_major=True))
    key = lambda r: (r[0]["episode_id"], r[1], r[2].name)  # noqa: E731
    assert sorted(map(key, minor)) == sorted(map(key, major))
    names = [r[2].name for r in major]
    assert names == sorted(names, key=lambda n: n != "B_c")  # all of B first, then A


class Probe(torch.nn.Module):
    """Stands in for the network: never called by encoder_inputs."""


def policy(spacing: int) -> NomadPolicy:
    cfg = {"context_size": 3, "image_size": [96, 96], "num_diffusion_iters": 10, "len_traj_pred": 8, "normalize": True}
    return NomadPolicy("", cfg, torch.device("cpu"), context_spacing=spacing, model=Probe())


def frames(n: int):
    return [np.full((24, 32, 3), 10 * k, np.uint8) for k in range(n)]


def test_context_spacing_one_is_the_deployment_queue():
    p = policy(1)
    assert p.context_frames == 4 and not p.ready(Observation(frames(3), None, 0)) and p.ready(Observation(frames(4), None, 0))
    a = p.encoder_inputs(Observation(frames(4), None, 0))["obs_img"]
    b = p._pictures(frames(4))
    assert torch.equal(a, b)


def test_context_spacing_three_takes_every_third_frame_and_waits():
    p = policy(3)
    assert p.context_frames == 10
    assert not p.ready(Observation(frames(9), None, 0)) and p.ready(Observation(frames(10), None, 0))
    got = p.encoder_inputs(Observation(frames(10), None, 0))["obs_img"]
    want = p._pictures([frames(10)[k] for k in (0, 3, 6, 9)])
    assert torch.equal(got, want)


# --- choice rule (p7_rules.md §6) ---------------------------------------------------------------------------------

from vint_train.mapmad.closed_loop.p7_choice import choose  # noqa: E402

LENS = {c: ("fisheye" if c.startswith("fish") else "pinhole") for c in
        ["pin018", "fish018", "pin025", "fish025", "pin045", "fish045", "pin055", "fish055"]}
HEIGHT = {c: int(c[-3:]) / 100 for c in LENS}


def rates(**over):
    s = {c: 0.30 for c in LENS}
    s.update(over)
    return s


def test_training_like_wins_within_five_points():
    win, _ = choose(rates(pin045=0.50, fish025=0.46, fish055=0.40), rates(), LENS, HEIGHT)
    assert win == "fish025"
    win, _ = choose(rates(pin045=0.50, fish025=0.40, fish055=0.45), rates(), LENS, HEIGHT)
    assert win == "fish055"  # the better of the two training-like cameras, exactly 0.05 below Best


def test_best_wins_alone_or_by_collisions():
    win, _ = choose(rates(pin045=0.60, fish025=0.40, fish055=0.40), rates(), LENS, HEIGHT)
    assert win == "pin045"  # nobody within 0.03
    coll = rates(pin045=40.0, pin055=20.0)
    win, _ = choose(rates(pin045=0.60, pin055=0.58, fish025=0.40, fish055=0.40), coll, LENS, HEIGHT)
    assert win == "pin055"  # within 0.03 and fewer collisions


def test_collision_tie_prefers_fisheye_then_indoor_height():
    s = rates(pin045=0.60, fish045=0.58, fish018=0.585, fish025=0.40, fish055=0.40)
    coll = rates(pin045=20.0, fish045=20.5, fish018=20.4)
    win, steps = choose(s, coll, LENS, HEIGHT)
    assert win == "fish018" and steps[-1].startswith("4.")  # fisheye; |0.18-0.25| = 0.07 < |0.45-0.55| = 0.10

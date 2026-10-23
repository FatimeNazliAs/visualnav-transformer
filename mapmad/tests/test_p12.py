"""Phase 12 (Habitat side): arm table, rule-stop value, arm F's mark, round-robin draw, unique set ids, plan."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from mapmad_sim import config
from mapmad_sim.arm_table import Arm, ArmTable, file_sha256
from mapmad_sim.exam_set import draw, round_robin
from mapmad_sim.explorer import OwnMap, TeacherSpec
from mapmad_sim.run_layout import read_config
from mapmad_sim.search_exam import FrontierAgent, FrontierSpec, RuleSpec, nearest_free, rule_value, strongest_cell
from mapmad_sim.search_set import search_set_ok, set_episode_id

REPO = Path(__file__).resolve().parents[2]


def table(tmp_path, name="p12_exam.yaml"):
    cfg = read_config(config.CONFIG_DIR / name)
    return ArmTable.from_config(cfg, tmp_path / cfg["out_dir"], tmp_path, REPO), cfg


# --- arm table ----------------------------------------------------------------------------------------------------
def test_arm_table_phase12_arms(tmp_path):
    t, cfg = table(tmp_path)
    assert t.arm("M2").policy == "mapmad_v2" and t.arm("M2h").maps is False
    assert t.arm("F").deterministic and t.seeds("F", [0, 1, 2]) == [0]
    assert t.arm("N").policy == "nomad" and t.arm("V1").map_zero == ("heat",)
    assert t.seeds("M21h", [0, 1, 2]) == [0, 1, 2]
    assert t.seeds("M2h", [0, 1, 2]) == [0, 1, 2]  # before the dev choices: all seeds
    (t.base).mkdir(parents=True)
    (t.base / "P12_DEV_CHOICES.json").write_text(json.dumps({"headline": "v21", "tau": 0.5}))
    assert t.seeds("M2h", [0, 1, 2]) == [0]  # not the headline pair -> 1 seed (block item 3)
    assert t.checkpoint("headline").key == "v21" and t.checkpoint("headline").sha256.startswith("ebaabe28")
    (t.base / "P12_DEV_CHOICES.json").write_text(json.dumps({"headline": "v2", "tau": 0.5}))
    assert t.seeds("M2h", [0, 1, 2]) == [0, 1, 2] and t.checkpoint("headline").key == "v2"


def test_arm_table_phase11_config_unchanged(tmp_path):
    t, cfg = table(tmp_path, "p11_dagger.yaml")
    assert t.arm("run1_hidden").seeds == (0,) and t.seeds("run1_hidden", [0, 1, 2]) == [0]
    ck = t.checkpoint("run1")
    assert ck.path == tmp_path / "p10_v2_train/runs/v2_s1_2026_10_21_08_42_50/ema_19.pth"
    assert ck.config == REPO / "train/config/mapmad_v2.yaml"


def test_checkpoint_sha256_mismatch_raises(tmp_path):
    f = tmp_path / "w.pth"
    f.write_bytes(b"weights")
    cfg = {"out_dir": "x", "checkpoints": {"a": {"path": str(f), "sha256": "0" * 64, "config": str(f)},
                                           "b": {"path": str(f), "sha256": file_sha256(f), "config": str(f)}}}
    t = ArmTable.from_config(cfg, tmp_path / "x", tmp_path, REPO)
    with pytest.raises(RuntimeError, match="sha256"):
        t.verify("a")
    assert t.verify("b").sha256 == file_sha256(f)


def test_arm_rules():
    with pytest.raises(ValueError):
        Arm("bad", target="recogniser")  # needs a recogniser
    with pytest.raises(ValueError):
        Arm("bad__name", checkpoint="v2")
    with pytest.raises(ValueError):
        Arm("F2", policy="frontier", checkpoint="v2")


# --- rule-stop value ---------------------------------------------------------------------------------------------
def test_rule_value_ahead_cone_and_views():
    shape, origin, res = (40, 40), (0.0, 0.0), 0.1
    value = np.zeros(shape[0] * shape[1], np.float32)
    count = np.zeros_like(value, dtype=np.int32)
    pose = (2.0, 2.0, 0.0)  # facing +X

    def put(x, y, v, n):
        k = int(x / res) * shape[1] + int(y / res)
        value[k], count[k] = v, n

    put(2.55, 2.05, 0.6, 2)   # 0.5 m ahead: counts
    put(2.85, 2.05, 0.9, 1)   # ahead but one view only: no
    put(2.05, 2.75, 0.95, 3)  # 0.7 m to the left (90 deg): no
    put(3.25, 2.05, 0.99, 5)  # 1.25 m ahead: too far
    v, cell = rule_value(value, count, shape, origin, res, pose, RuleSpec())
    assert v == pytest.approx(0.6) and cell == pytest.approx((2.55, 2.05))
    put(2.85, 2.35, 0.8, 2)   # 0.88 m, bearing ~21 deg: counts and is larger
    assert rule_value(value, count, shape, origin, res, pose, RuleSpec())[0] == pytest.approx(0.8)
    assert rule_value(value, np.zeros_like(count), shape, origin, res, pose, RuleSpec()) == (0.0, None)


def test_strongest_cell_exclude():
    value = np.array([0.1, 0.9, 0.5], np.float32)
    count = np.array([2, 2, 3])
    assert strongest_cell(value, count, 2) == (pytest.approx(0.9), 1)
    assert strongest_cell(value, count, 2, np.array([False, True, False])) == (pytest.approx(0.5), 2)
    assert strongest_cell(value, np.array([1, 1, 1]), 2) == (0.0, -1)


def test_nearest_free_avoids_inflated_obstacles():
    obstacle = np.zeros((30, 30), bool)
    explored = np.zeros((30, 30), bool)
    explored[5:25, 5:25] = True
    obstacle[15, 15] = True  # the object itself
    own = OwnMap(obstacle, explored, (0.0, 0.0), 0.1)
    goal = nearest_free(own, (1.55, 1.55), 1.0, 0.195)
    d = math.hypot(goal[0] - 1.55, goal[1] - 1.55)
    assert 0.2 < d <= 1.0  # outside the inflation, within the mark radius
    assert nearest_free(OwnMap(obstacle, np.zeros_like(explored), (0.0, 0.0), 0.1), (1.55, 1.55), 1.0, 0.195) is None


class FakeSearch:
    """The bits of search_v2.SearchMap that FrontierAgent reads."""

    def __init__(self, shape=(60, 60)) -> None:
        from types import SimpleNamespace

        n = shape[0] * shape[1]
        self.fmap = SimpleNamespace(shape=shape, origin=np.zeros(2), first_seen={"obstacle": np.full(n, -1),
                                                                                "explored": np.zeros(n, np.int64)})
        self.spec = SimpleNamespace(map_spec=SimpleNamespace(resolution=0.1))
        self.bumped = np.zeros(shape, bool)
        self.fusions = {"noisy": SimpleNamespace(value=np.zeros(n, np.float32), count=np.zeros(n, np.int32))}
        self.seen = {}


def test_frontier_agent_goes_to_a_mark_and_remembers_it():
    s = FakeSearch()
    k = 40 * 60 + 30  # cell (40, 30): x 4.05, y 3.05
    s.fusions["noisy"].value[k], s.fusions["noisy"].count[k] = 0.8, 2
    agent = FrontierAgent(FrontierSpec(TeacherSpec(), tau=0.5))
    d = agent.step(s, (1.05, 3.05, 0.0))
    assert d["mode"] == "approach" and d["mark"] == pytest.approx([4.05, 3.05])
    assert math.hypot(d["goal"][0] - 4.05, d["goal"][1] - 3.05) <= 1.0
    goal = d["goal"]
    d = agent.step(s, (goal[0], goal[1], 0.0))  # standing on the goal, facing the mark (+X)
    assert d["mode"] == "arrived" and d.get("mark_done")
    d = agent.step(s, (goal[0], goal[1], 0.0))  # the mark is visited: no seen instance any more
    assert d["mark"] is None and d["mode"] != "approach"
    agent2 = FrontierAgent(FrontierSpec(TeacherSpec(), tau=0.9))  # below tau: not seen
    assert agent2.step(s, (1.05, 3.05, 0.0))["mark"] is None


# --- sets ---------------------------------------------------------------------------------------------------------
def ep(i, cat, b, home="00800-x"):
    return {"episode_id": f"p12e-{home[:5]}-{cat}-{i}", "home": home, "distance_bin": b,
            "target": {"category": cat, "view_points": [[0, 0, 0]]}}


def test_round_robin_order_and_draw():
    eps = [ep(i, c, 0) for i, c in enumerate(["chair"] * 4 + ["bed"] * 2 + ["toilet"])]
    order = round_robin(eps)
    assert [e["target"]["category"] for e in order] == ["bed", "chair", "toilet", "bed", "chair", "chair", "chair"]
    kept = [ep(i, c, i % 3, home=f"{i % 7:05d}-h") for i, c in enumerate(["chair", "bed", "sofa", "toilet"] * 30)]
    a, ra = draw(kept, 30, 5, 3, 12201, "round_robin_category")
    b, rb = draw(kept, 30, 5, 3, 12201, "round_robin_category")
    assert a == b and len(a) == 30 and rb["order"] == "round_robin_category"
    _, r0 = draw(kept, 30, 5, 3, 12201)
    assert "order" not in r0  # Phases 5 / 11 reports unchanged


def test_set_ids_unique_and_clean():
    assert set_episode_id("p12o-val_seen", "00877-4ok", "washer-dryer", "283") == "p12o-val_seen-00877-washer_dryer-283"
    assert set_episode_id("p12e", "00800-x", "tv_monitor", "7") == "p12e-00800-tv_monitor-7"
    eps = [ep(1, "chair", 0), ep(1, "chair", 1)]
    assert search_set_ok(eps) == "duplicate episode ids"
    assert search_set_ok([ep(1, "chair", 0), ep(2, "chair", 0)]) is None


def test_plan_calls_order_and_trims(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "train/scripts/mapmad_p12"))
    import p12_plan

    out = tmp_path / "out"
    base = out / "p12_search_exam"
    (base / "exam_set").mkdir(parents=True)
    (base / "exam_set/episodes.json").write_text(json.dumps({"episodes": [{"episode_id": f"e{i}"} for i in range(10)]}))
    (base / "P12_DEV_CHOICES.json").write_text(json.dumps({"headline": "v2", "tau": 0.5}))
    plan = p12_plan.Plan(str(config.CONFIG_DIR / "p12_exam.yaml"), str(out))
    labels = [c[0] for c in plan.calls("exam")]
    assert labels == ["exam_pair", "exam_N", "exam_V1", "exam_F", "exam_other", "exam_MT", "exam_Nfish"]
    assert plan.calls("exam")[0][5] == ["M2", "M2h"]
    assert plan.runs(plan.calls("exam")[0]) == 60 and plan.runs(plan.calls("exam")[3]) == 10  # F: 1 run each
    plan.trims = ["drop_nfish", "ovon_n_1seed", "drop_explore_v1", "other_hidden_1seed"]
    labels = [c[0] for c in plan.calls("exam")]
    assert "exam_Nfish" not in labels and ("exam_other_h", "0") in [(c[0], c[4]) for c in plan.calls("exam")]
    assert "X_V1" not in plan.calls("explore")[0][5]

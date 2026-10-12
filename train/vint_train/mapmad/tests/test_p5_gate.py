"""Gate G5 row 1 (fairness) in exam_gate.py."""

from vint_train.mapmad.closed_loop.exam_gate import fairness

CFG = {"run": {"seeds": 2}, "nomad": {"weights_sha256": "n"}, "mapmad": {"checkpoint_sha256": "m"},
       "arms": {"A": {"policy": "nomad"}, "C": {"policy": "mapmad"}, "E": {"policy": "planner"}}}
EPS = [{"episode_id": f"e{i}"} for i in range(100)]


def rows(skip=(), sha=None):
    out = []
    for e in EPS:
        for arm, k, s in (("A", 2, "n"), ("C", 2, "m"), ("E", 1, "")):
            for i in range(k):
                if (arm, e["episode_id"], i) not in skip:
                    out.append({"arm": arm, "episode_id": e["episode_id"], "seed_index": str(i), "episodes_fingerprint": "fp",
                                "weights_sha256": sha if (sha and arm == "C") else s})
    return out


def test_complete_runs_pass():
    assert fairness(CFG, EPS, "fp", rows(), [])["pass"]


def test_missing_run_or_wrong_hash_fails():
    assert not fairness(CFG, EPS, "fp", rows(skip={("C", "e3", 1)}), [])["pass"]
    assert not fairness(CFG, EPS, "fp", rows(sha="x"), [])["pass"]


def test_removed_episodes_limit():
    two = [r for r in rows() if r["episode_id"] not in ("e1", "e2")]
    assert fairness(CFG, EPS, "fp", two, ["e1", "e2"])["pass"]  # 2% removed: allowed
    three = [r for r in two if r["episode_id"] != "e3"]
    assert not fairness(CFG, EPS, "fp", three, ["e1", "e2", "e3"])["pass"]  # 3% -> a bug (block item 13b)

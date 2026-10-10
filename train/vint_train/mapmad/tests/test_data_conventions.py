"""Phase 2.1: NoMaD's action frame on GoStanford first, then on our Habitat drives (same to_local_coords).

straight -> forward x > 0, y ~ 0; left turn (yaw growing) -> y > 0. Needs the NoMaD container (torch);
GoStanford from train/config/nomad.yaml's data_folder, our frames via mapmad_sim.frames (PYTHONPATH mapmad/src),
our drives from paths.yaml's mapmad_data (mapmad_sim.config) if a generated dataset is there.
"""

import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from vint_train.data.data_utils import to_local_coords

TRAIN = Path(__file__).resolve().parents[3]
GO_STANFORD = Path(yaml.safe_load((TRAIN / "config" / "nomad.yaml").read_text())["datasets"]["go_stanford"]["data_folder"])
frames = pytest.importorskip("mapmad_sim.frames")
MAPMAD_DATA = pytest.importorskip("mapmad_sim.config").paths()["mapmad_data"]


def load(path: Path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["position"].astype(np.float64), d["yaw"].astype(np.float64).reshape(-1)


def segments(position, yaw, ahead: int = 4):
    """(local end point, yaw change, path length) of every `ahead`-frame window."""
    for i in range(len(yaw) - ahead):
        loc = to_local_coords(position[i:i + ahead + 1], position[i], yaw[i])[-1]
        dyaw = math.atan2(math.sin(yaw[i + ahead] - yaw[i]), math.cos(yaw[i + ahead] - yaw[i]))
        yield loc, dyaw, float(np.hypot(*np.diff(position[i:i + ahead + 1], axis=0).T).sum())


def check_convention(trajs, min_len: float):
    straight, left = [], []
    for position, yaw in trajs:
        for loc, dyaw, length in segments(position, yaw):
            if length < min_len:
                continue
            if abs(dyaw) < 0.02:
                straight.append(loc)
            elif dyaw > 0.2:
                left.append(loc)
    straight, left = np.array(straight), np.array(left)
    assert len(straight) > 20 and len(left) > 20
    assert (straight[:, 0] > 0).mean() > 0.99 and np.median(np.abs(straight[:, 1]) / straight[:, 0]) < 0.05
    assert (left[:, 1] > 0).mean() > 0.9


@pytest.mark.skipif(not GO_STANFORD.exists(), reason="GoStanford not mounted")
def test_go_stanford_convention():
    names = sorted(p for p in GO_STANFORD.iterdir() if (p / "traj_data.pkl").exists())[:200]
    check_convention([load(p / "traj_data.pkl") for p in names], min_len=0.2)


def test_habitat_frame_matches():
    """A unicycle simulated in Habitat's frame (yaw 0 looks along -z, + = left) and converted with frames.to_2d."""
    pos, yaw, out_p, out_y = np.zeros(3), 0.3, [], []
    for k in range(80):
        w = 0.0 if k < 40 else 0.4  # straight, then a left arc
        out_p.append(frames.to_2d(pos))
        out_y.append(yaw)
        yaw += w * 0.25
        pos = pos + 0.05 * np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
    check_convention([(np.array(out_p), np.array(out_y))], min_len=0.1)


def dataset_dirs():
    found = [d for d in (MAPMAD_DATA / n for n in ("habitat_mapmad", "pilot_habitat_mapmad")) if d.exists()]
    return [p for d in found for p in sorted(d.glob("*_*/traj_data.pkl"))[:100]]


@pytest.mark.skipif(not dataset_dirs(), reason="no generated MapMaD drives")
def test_generated_drives_convention():
    check_convention([load(p) for p in dataset_dirs()], min_len=0.1)

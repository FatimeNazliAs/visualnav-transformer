"""Phase 9 plug-in replay (scripts/p9_plugin_run.py) without the simulator: a fake robot renders synthetic depth and
semantic pictures along a fake Phase 8 drive; the drive's clean layer is made by Phase 8's own code (datagen_v2's
path: frame_cells -> target_layer.detect -> target_layer.frame_scores -> TargetFusion). The replay with the truth
recogniser must give exactly that layer (the parity check of G9 row 4), and a dummy model must write the Phase 8
target-file format."""

import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mapmad_sim import objectnav
from mapmad_sim.depth_map import FirstSeenMap, MapSpec
from mapmad_sim.recogniser import Detection, TruthRecogniser
from mapmad_sim.target_layer import NoiseSpec, detect, frame_cells, frame_scores

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train"))
from vint_train.mapmad.map_v2 import TARGET_FILE, TargetFusion  # noqa: E402

GEN = {"map": {"resolution_m": 0.1, "max_depth_m": 4.0, "obstacle_band_m": [0.05, 0.55], "floor_min_m": -0.10,
               "min_points": 3, "margin_m": 1.0},
       "target_layer": {"min_pixels": 200, "max_depth_m": 4.0, "confidence": [0.4, 1.0], "miss_p": 0.2, "false_p": 0.05,
                        "false_value": [0.3, 0.7], "false_blob_cells": 3}}
H, W, T = 60, 80, 6


def load_script():
    spec = importlib.util.spec_from_file_location("p9_plugin_run", ROOT / "mapmad" / "scripts" / "p9_plugin_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeRobot:
    """Renders a chair (id 7) that slides across the picture with the pose index; depth 2 m, a far band at 6 m."""

    def __init__(self, *a, **k):
        self.spec = SimpleNamespace(hfov_deg=66.5)
        self.pathfinder = SimpleNamespace(get_bounds=lambda: (np.array([-5.0, 0.0, -5.0]), np.array([5.0, 2.0, 5.0])))
        self.floor_y = 0.0
        self.t = 0

    def place(self, position, yaw):
        self.t = int(round(position[0] * 10))
        self.yaw = yaw

    def observe(self):
        sem = np.zeros((H, W), np.uint32)
        x0 = 5 + 8 * self.t
        sem[20:50, x0:x0 + 25] = 7
        sem[0:10, :] = 3  # something else
        depth = np.full((H, W), 2.0, np.float32)
        depth[:, :5] = 6.0  # beyond 4 m: no cell
        return {"rgb": np.zeros((H, W, 3), np.uint8), "depth": depth, "semantic": sem}

    def camera_transform(self):
        tf = np.eye(4)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        tf[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        tf[:3, 3] = [0.0, 0.25, 0.0]
        return tf

    def close(self):
        pass


def make_drive(root: Path) -> str:
    """A fake Phase 8 drive folder with a clean layer drawn by Phase 8's code."""
    drive = "p8d-00746-fake-search-000"
    folder = root / "ds" / drive
    folder.mkdir(parents=True)
    pos3 = np.array([[t / 10.0, 0.0, 0.0] for t in range(T)])
    yaw = np.linspace(0.0, 0.5, T)
    np.savez_compressed(folder / "mapmad_frames.npz", position_habitat=pos3)
    with open(folder / "traj_data.pkl", "wb") as f:
        pickle.dump({"position": np.stack([-pos3[:, 2], -pos3[:, 0]], 1), "yaw": yaw}, f)
    (folder / "mapmad_meta.json").write_text(json.dumps({"home": "00746-fake", "category": "chair", "split": "train"}))
    robot, spec, nspec = FakeRobot(), MapSpec.from_config(GEN["map"]), NoiseSpec.from_config(GEN["target_layer"])
    fmap = FirstSeenMap.around(spec, *robot.pathfinder.get_bounds())
    clean = TargetFusion(fmap.shape)
    rng = np.random.default_rng(0)
    for t in range(T):
        robot.place(pos3[t], yaw[t])
        obs = robot.observe()
        cells = frame_cells(obs["depth"], 66.5, robot.camera_transform(), 0.0, fmap, spec, nspec.max_depth_m)
        sc = frame_scores(t, detect(obs["semantic"], obs["depth"], [7], nspec), cells, fmap.shape, nspec, rng)
        clean.update(t, cells.view, sc.clean)
    np.savez_compressed(folder / TARGET_FILE, **{f"clean_{k}": v for k, v in clean.updates().items()})
    return drive


def run_replay(tmp_path, monkeypatch, rec):
    mod = load_script()
    monkeypatch.setenv("MAPMAD_MAPMAD_DATA", str(tmp_path))
    monkeypatch.setattr(mod, "LimoSim", FakeRobot)
    monkeypatch.setattr(mod, "robot_spec", lambda gen: None)
    monkeypatch.setattr(mod.objectnav, "home_goals", lambda split, home: [objectnav.Goal(7, "chair", [0, 0, 0], [])])
    drive = make_drive(tmp_path)
    out = tmp_path / "out" / drive
    summary = mod.replay(drive, {"drives": {"dataset": "ds"}}, GEN, rec, 0, out)
    return summary, out


def test_truth_replay_equals_phase8_clean(tmp_path, monkeypatch):
    rec = TruthRecogniser({"chair": [7]}, NoiseSpec.from_config(GEN["target_layer"]))
    summary, out = run_replay(tmp_path, monkeypatch, rec)
    assert summary["plugin_truth_equals_phase8_clean"] == {"cell": True, "frame": True, "value": True}
    assert summary["truth_layer_equals_phase8_clean"] == {"cell": True, "frame": True, "value": True}
    z = np.load(out / TARGET_FILE)
    assert z["clean_cell"].size > 0 and summary["steps_with_detections"] == T
    assert (out / "video.mp4").stat().st_size > 0 and json.loads((out / "steps.json").read_text())["frames"] == T


class Dummy:
    """A model that marks the top-left corner (a false alarm) with mapped score 0.7 and raw score 0.33."""

    name = "dummy"

    def detect(self, rgb, depth, category, semantic=None):
        m = np.zeros(depth.shape, bool)
        m[40:60, 60:80] = True
        return [Detection(0.7, m, raw_score=0.33)]


def test_model_replay_writes_phase8_format(tmp_path, monkeypatch):
    summary, out = run_replay(tmp_path, monkeypatch, Dummy())
    z = np.load(out / TARGET_FILE)
    for k in ("shape", "category", "clean_cell", "clean_frame", "clean_value", "noisy_cell", "noisy_frame", "noisy_value",
              "det_frame", "det_instance", "det_pixels", "det_confidence", "det_missed", "det_raw_score",
              "false_frame", "false_cell", "false_value"):
        assert k in z.files, k
    assert z["det_frame"].tolist() == list(range(T)) and np.allclose(z["det_confidence"], 0.7)
    assert np.allclose(z["det_raw_score"], 0.33) and (z["det_instance"] == -1).all()
    assert z["noisy_value"].max() == np.float32(0.7) and not np.array_equal(z["noisy_cell"], z["clean_cell"])
    assert "plugin_truth_equals_phase8_clean" not in summary

"""Our copies of NoMaD's deployment rules give exactly what deployment/src gives (ROS modules stubbed out)."""

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from vint_train.mapmad.closed_loop import deployment

DEPLOY_SRC = Path(__file__).resolve().parents[4] / "deployment" / "src"


def stub_ros(monkeypatch):
    """Fake rospy / *_msgs / ros_data so the deployment modules import outside ROS."""
    for name in ("rospy", "geometry_msgs", "geometry_msgs.msg", "std_msgs", "std_msgs.msg", "sensor_msgs",
                 "sensor_msgs.msg", "ros_data"):
        module = types.ModuleType(name)
        for attr in ("Twist", "Float32MultiArray", "Bool", "Image", "ROSData"):
            setattr(module, attr, type(attr, (), {"__init__": lambda self, *a, **k: None}))
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(DEPLOY_SRC))
    monkeypatch.chdir(DEPLOY_SRC)  # pd_controller.py reads ../config/robot.yaml at import


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"deployment_{name}", DEPLOY_SRC / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pd_controller_matches_deployment(monkeypatch):
    stub_ros(monkeypatch)
    original = load("pd_controller")
    rng = np.random.default_rng(0)
    cases = list(rng.uniform(-0.3, 0.3, size=(200, 2))) + [np.array([0.0, 0.1]), np.array([0.0, -0.1]),
                                                             np.array([-0.05, 0.02]), np.array([1.0, 0.0])]
    for wp in cases:
        assert deployment.pd_controller(wp, original.MAX_V, original.MAX_W, original.DT) == pytest.approx(
            tuple(float(x) for x in original.pd_controller(wp)))
    assert (original.MAX_V, original.MAX_W, original.DT) == (0.2, 0.4, 0.25)  # robot.yaml: what our config copies


def test_transform_images_matches_deployment(monkeypatch):
    stub_ros(monkeypatch)
    original = load("utils")
    rng = np.random.default_rng(1)
    pictures = [PILImage.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)) for _ in range(4)]
    ours = deployment.transform_images(pictures, [96, 96])
    theirs = original.transform_images(pictures, [96, 96], center_crop=False)
    assert ours.shape == (1, 12, 96, 96) and torch.equal(ours, theirs)

"""MapMaD Habitat-side checks. Run inside naz_mapmad_habitat:

    cd /app/visualnav-transformer/mapmad && python -m pytest -q -p no:cacheprovider

Tests that need the HM3D mount or a GPU are skipped where those are missing.
"""

from pathlib import Path

import numpy as np
import pytest

from mapmad_sim import config

HOME = "00800-TEEsavR23oF"  # labelled minival home used by the Phase 0 render
needs_hm3d = pytest.mark.skipif(not config.hm3d_scene("minival", HOME).exists(), reason="HM3D mount (/hm3d) missing")


def test_paths_come_from_config_and_env(monkeypatch):
    assert config.paths()["hm3d_scenes"] == Path("/hm3d")
    monkeypatch.setenv("MAPMAD_HM3D_SCENES", "/elsewhere")
    assert config.paths()["hm3d_scenes"] == Path("/elsewhere")


def test_hm3d_file_names():
    scenes = Path("/hm3d")
    assert config.hm3d_scene("val", "00877-4ok3usBNeis", scenes) == scenes / "val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"
    assert config.hm3d_scene_dataset_config("minival", scenes).name == "hm3d_annotated_minival_basis.scene_dataset_config.json"
    with pytest.raises(ValueError):
        config.hm3d_scene_dataset_config("test", scenes)


def test_robot_config_has_camera_facts():
    camera = config.robot()["camera"]
    assert 0.0 < camera["height_m"] < 1.0 and 30.0 < camera["hfov_deg"] < 180.0
    assert {"height_status", "hfov_status", "resolution_status"} <= set(camera)


@needs_hm3d
def test_labelled_home_counts():
    assert [len(config.labelled_homes(s)) for s in ("minival", "val", "train")] == [4, 36, 145]


@needs_hm3d
def test_hm3d_renders_on_nvidia_with_aligned_labels():
    """One frame of a labelled minival home: rendered by the NVIDIA GPU, labels present, boxes filled."""
    from habitat_starter import make_sim
    from habitat_starter.semantics import box_center_size
    from habitat_starter.sim import renderer_name

    settings = config.hm3d_sim_settings("minival", HOME, width=160, height=120, semantic_sensor=True)
    with make_sim(settings) as sim:
        obs = sim.get_sensor_observations()
        assert "NVIDIA" in renderer_name()
        assert obs["color_sensor"].shape == (120, 160, 4) and obs["depth_sensor"].max() > 0
        assert (obs["semantic_sensor"] > 0).mean() > 0.9
        sizes = [box_center_size(o.aabb)[1] for o in sim.semantic_scene.objects if o is not None]
        assert np.mean([np.all(s > 0) for s in sizes]) > 0.9

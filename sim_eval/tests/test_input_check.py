"""Pin the input check's arithmetic — the numbers a camera decision rests on.

`p5_1_input_check.py` is where P5 decides whether the sim camera shows the
model what training did, and a camera change made on its say-so changes what
every arm sees in P6. So the parts that can be wrong without looking wrong are
pinned here:

  * **training's crop.** The check says "the 4:3 crop is a no-op on both
    sources" on the strength of its own copy of `resize_and_aspect_crop`. That
    copy is deliberate (fidelity to training, not to a shared helper), and it
    was the one function in the file nothing tested.
  * **the field-of-view conversion** the figure and the table label every
    angle with.
  * **the lens measurement** and its rule. The rule is fixed in code before
    anything runs — the lowest `nearest_ratio` wins — and what matters is that
    it picks the setting whose frames sit with training's, and that training
    measured against itself reads as ~1. The same rule chooses a field of view
    and a camera tilt.

No GPU, no simulator, no checkpoint: images are PIL, embeddings are numpy.

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import p5_1_input_check as check  # noqa: E402


# --- training's crop -----------------------------------------------------------

def test_a_4_3_frame_is_left_alone():
    assert check.aspect_crop(Image.new("RGB", (640, 480))).size == (640, 480)
    assert check.aspect_crop(Image.new("RGB", (160, 120))).size == (160, 120)


def test_a_wide_frame_loses_its_sides_not_its_top_and_bottom():
    """16:9 at 480 tall keeps the full height and the central 640 columns."""
    frame = Image.new("RGB", (854, 480))
    frame.paste((255, 0, 0), (0, 0, 107, 480))       # a red strip at the left
    cropped = check.aspect_crop(frame)
    assert cropped.size == (640, 480)
    assert cropped.getpixel((0, 240)) == (0, 0, 0)   # the strip was cropped off


def test_a_tall_frame_loses_its_top_and_bottom():
    assert check.aspect_crop(Image.new("RGB", (480, 640))).size == (480, 360)


# --- the field of view --------------------------------------------------------

def test_the_sims_45_degrees_vertical_is_58_horizontal_at_4_3():
    assert check.horizontal_fov_deg(45, 640, 480) == pytest.approx(57.82, abs=0.01)


def test_a_square_image_has_equal_fields_of_view():
    assert check.horizontal_fov_deg(90, 96, 96) == pytest.approx(90.0)


# --- the lens measurement -----------------------------------------------------

def test_the_nearest_distance_is_zero_for_a_set_inside_the_reference():
    points = np.eye(4)
    assert check.mean_nearest_distance(points[:2], points) == 0.0


def test_the_centroid_gap_is_zero_when_the_means_coincide():
    reference = np.array([[0.0, 0.0], [2.0, 2.0]])
    assert check.centroid_gap(np.array([[1.0, 1.0]]), reference) == 0.0


def a_dataset(rng, count=200, dims=8):
    return rng.normal(size=(count, dims))


def test_training_against_itself_reads_as_one():
    """The yardstick: a held-out half of the training frames is as close to the
    other half as training frames are to each other."""
    rng = np.random.default_rng(0)
    dataset = a_dataset(rng)
    lens = check.camera_measurement({45: dataset[100:]}, dataset)
    assert lens["training_against_itself"]["nearest_ratio"] == pytest.approx(1.0)


def test_the_angle_whose_frames_sit_with_training_is_chosen():
    """Three sweeps of the same 'poses': one shifted far off, one a little off,
    one drawn from training's own distribution. The rule has to find the last."""
    rng = np.random.default_rng(1)
    dataset = a_dataset(rng)
    sim = {
        45: rng.normal(size=(60, 8)) + 3.0,
        90: rng.normal(size=(60, 8)) + 1.0,
        110: rng.normal(size=(60, 8)),
    }
    lens = check.camera_measurement(sim, dataset)
    assert lens["chosen"] == 110
    assert lens["centroid_agrees"]
    ratios = [lens["by_setting"][angle]["nearest_ratio"]
              for angle in (45, 90, 110)]
    assert ratios == sorted(ratios, reverse=True)


def test_the_table_is_ordered_by_angle_whatever_order_it_was_measured_in():
    rng = np.random.default_rng(2)
    dataset = a_dataset(rng)
    sim = {angle: rng.normal(size=(20, 8)) for angle in (90, 45, 120)}
    assert list(check.camera_measurement(sim, dataset)["by_setting"]) == [
        45, 90, 120]

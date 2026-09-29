"""Pin the four numbers that steer, and the label that records them.

The point of `DriverConfig` is evidence: a row of the metrics table must say
what the policy was set to, so a P5 tuning sweep produces comparable tables
instead of indistinguishable ones, and P6's "every arm faced identical
settings" is checkable rather than asserted.

That makes two things worth pinning. The **label** is written into every row,
so its spelling is a data format — change it and old tables stop comparing.
And **`is_deployment_default`** is what turns plan decision E ("mirror the real
LoCoBot exactly; tune nothing") from a sentence in a plan into something a test
can fail on.

No torch here, which is the other half of why this module exists: reading the
configuration must not cost the whole NoMaD stack.

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver import DriverConfig  # noqa: E402


def test_the_defaults_are_navigate_pys_own_flags():
    driver = DriverConfig()
    assert driver.as_dict() == {"num_samples": 8, "waypoint": 2, "radius": 4,
                                "close_threshold": 3}


def test_the_label_is_the_row_this_run_writes():
    """Written into every row, so its spelling is a data format."""
    assert DriverConfig().label() == "n8w2r4t3"
    assert DriverConfig(num_samples=4, waypoint=3).label() == "n4w3r4t3"


def test_deployment_defaults_are_recognised_as_such():
    assert DriverConfig().is_deployment_default()
    assert DriverConfig(num_samples=8).is_deployment_default()
    assert not DriverConfig(num_samples=16).is_deployment_default()


def test_a_tuned_driver_says_so_where_a_person_will_read_it():
    """The run prints this line. A departure from the real robot's settings is
    not something to discover afterwards, in a column."""
    assert "TUNED" not in DriverConfig().summary()
    assert "TUNED" in DriverConfig(waypoint=4).summary()


def test_a_config_block_becomes_a_driver():
    driver = DriverConfig.from_dict({"num_samples": 4, "close_threshold": 5})
    assert (driver.num_samples, driver.close_threshold) == (4, 5)
    # Unnamed knobs keep the deployment value rather than becoming zero.
    assert (driver.waypoint, driver.radius) == (2, 4)


def test_an_empty_or_absent_block_is_the_default():
    assert DriverConfig.from_dict(None).is_deployment_default()
    assert DriverConfig.from_dict({}).is_deployment_default()


def test_nonsense_is_refused_at_load_rather_than_mid_episode():
    """Every one of these would otherwise fail deep inside the diffusion loop,
    hundreds of ticks into a run."""
    with pytest.raises(ValueError):
        DriverConfig(num_samples=0)
    with pytest.raises(ValueError):
        DriverConfig(waypoint=-1)
    with pytest.raises(ValueError):
        DriverConfig(radius=-1)
    with pytest.raises(TypeError):
        DriverConfig(waypoints=2)

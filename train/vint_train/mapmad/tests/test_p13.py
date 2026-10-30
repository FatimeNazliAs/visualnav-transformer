"""Phase 13 NoMaD-side pure parts: Holm adjustment, the pre-set reading of G13 row 1, the equal-count rule.

    docker exec -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad bash -lc "cd /app/visualnav-transformer/train && python -m pytest vint_train/mapmad/tests/test_p13.py -q"
"""

import numpy as np
import pytest

from vint_train.mapmad import modes as M
from vint_train.mapmad.closed_loop.p13_tables import holm, reading
from vint_train.mapmad.dataset_v2 import MapMaDV2Dataset


def test_holm_step_down():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.0001, 0.9]) == pytest.approx([1.0, 0.0003, 1.0])


def test_row1_reading():
    assert "holds with equal practice" in reading({"diff": 0.15}, True)
    assert "narrows to +0.060" in reading({"diff": 0.06}, True)
    assert "map + practice" in reading({"diff": 0.02}, False)


class _Fake:
    """Just the attributes attach_dagger_hidden reads."""

    def __init__(self, n):
        self.pools = {"search": np.arange(n), "explore": np.arange(n)}


def test_equal_count_q():
    ds = MapMaDV2Dataset.__new__(MapMaDV2Dataset)
    ds.mode_p = M.habitat_mode_table({"search_map": 0.45, "explore_map": 0.20, "explore": 0.20, "photo": 0.15},
                                     M.HABITAT_MODES_V2)
    ds.pools = {"search": np.arange(1_775_000)}
    with pytest.raises(ValueError):
        ds.attach_dagger_hidden(_Fake(10))  # map-DAgger must be attached first
    ds.dagger = _Fake(524_409)
    ds.attach_dagger_hidden(_Fake(100))
    s = 524_409 / (1_775_000 + 524_409)
    assert ds.hidden_q == pytest.approx(0.45 * s / 0.20)
    # expected draws per Habitat draw: map-DAgger = hidden-DAgger
    assert 0.45 * s == pytest.approx(0.20 * ds.hidden_q)

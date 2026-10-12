"""Overview drawing (video.TopdownMap): the box, path and heat use one pixel convention."""

import numpy as np

from vint_train.mapmad.closed_loop import video


def test_pixel_centres_map_back_to_their_pixel():
    grid = np.zeros((300, 200), bool)
    grid[20:280, 30:170] = True
    t = video.TopdownMap(grid, [-3.21, 4.07], 0.05, [[0.0, 0.0, 9.0]])
    x, z = t.centres()
    for r, c in [(0, 0), (17, 250), (300, 123), (video.MAP_PX - 1, video.MAP_PX - 1)]:
        assert t.px([x[r, c], 0.0, z[r, c]]) == (c, r)

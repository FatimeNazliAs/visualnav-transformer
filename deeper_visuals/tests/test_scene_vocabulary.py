"""
The frame names, and the fact that they follow settings.

P3 used to carry these five words as a literal list, with a comment saying they
were "deliberately the words P1 and P2 already used". The comment was the only
thing holding the agreement together, and the list baked in CONTEXT_SIZE = 3 —
so raising the context size would have left P3's attention figure mislabelled
with nothing failing anywhere.

These need no GPU, no checkpoint and no dataset: Scene.token_labels is derived
from settings alone, which is exactly why it can be pinned here.
"""

from deeper_visuals.common import figures, settings
from deeper_visuals.common.data import Scene


def _scene(n_obs=None):
    n_obs = n_obs or settings.N_OBS_FRAMES
    return Scene(obs_raw=[None] * n_obs, goal_raw=None,
                 obs_idxs=list(range(n_obs)), goal_idx=n_obs,
                 traj_name="test")


def test_there_is_one_label_per_token():
    assert len(_scene().token_labels) == settings.N_TOKENS


def test_the_last_two_are_now_and_goal():
    labels = _scene().token_labels
    assert labels[-2:] == ["now", "goal"]


def test_the_past_frames_count_backwards_from_the_context_size():
    labels = _scene().token_labels
    assert labels[0] == f"t − {settings.CONTEXT_SIZE}"
    assert labels[settings.CONTEXT_SIZE - 1] == "t − 1"


def test_the_vocabulary_tracks_settings_rather_than_a_literal():
    """
    The regression that motivated this: change the context size and the labels
    must follow. A hardcoded list would fail here.
    """
    original = settings.CONTEXT_SIZE
    try:
        settings.CONTEXT_SIZE = 5
        labels = _scene(n_obs=6).token_labels
        assert labels == ["t − 5", "t − 4", "t − 3", "t − 2", "t − 1",
                          "now", "goal"], labels
    finally:
        settings.CONTEXT_SIZE = original


def test_every_frame_kind_has_a_house_style():
    """A caller names what a frame IS; the style table decides how it looks."""
    for kind in ("now", "past", "goal"):
        colour, width = figures.FRAME_STYLES[kind]
        assert colour.startswith("#") and width > 0


def test_the_current_frame_and_the_goal_share_one_emphasis():
    """
    Four call sites picked the emphasis weight independently and one picked
    2.2 where the others picked 2.6.
    """
    assert (figures.FRAME_STYLES["now"][1]
            == figures.FRAME_STYLES["goal"][1]
            > figures.FRAME_STYLES["past"][1])

#!/usr/bin/env python3
# deeper_visuals/p2_encoders/occlusion.py
"""
P2, second figure — what each encoder's token actually depends on.

The obvious question after "the picture becomes 256 numbers" is "which parts of
the picture?". The 256-vector cannot be turned back into an image: 27,648 → 256
is lossy by 108x, pooling and the swish nonlinearity discard information for
good, and no decoder was ever trained. Anything that produced a picture from
that vector would be a second network's guess, not a view of what the encoder
kept.

So this measures dependence instead, which is the better-posed question:

    cover a patch of the frame → re-encode → how far did the token move?

Cosine distance from the unoccluded token, accumulated over a sliding patch.
Two properties make it the right method here rather than Grad-CAM:

1. Resolution is set by the stride, not by the network. psi's final feature map
   is 1280 x 3 x 3 for a 96 x 96 input, so a class-activation map would have
   nine cells across the whole image — a tic-tac-toe board, not a saliency map.
2. It is causal and needs no gradients. It perturbs the input and reads the
   token the transformer is genuinely handed, rather than a gradient proxy for
   it.

Both maps come out of the *same* real forward passes, captured on the same
hooks run_model.py uses, so nothing here re-implements the encoders.

Known limitation, stated rather than hidden: a covered patch is itself an input
the encoder never saw in training, so a map shows what the token is sensitive
to, not a faithful account of what the network "looks at". It is read for its
shape — where the peaks are, and whether the two encoders differ — not for its
absolute values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deeper_visuals.common import measure, model as model_lib, viz
from deeper_visuals.common.data import to_model_input
from deeper_visuals.common.viz import plt

# A patch big enough to remove a whole object at 96 x 96, stepped finely enough
# that the map reads as a map: 19 x 19 = 361 probes per encoder.
#
# The stride is what buys smoothness, and it buys it with more measurement
# rather than with a blur — at stride 8 the overlap-averaged map still stepped
# visibly every 8 pixels, and smoothing that afterwards would have been the
# figure inventing detail it had not measured.
PATCH = 24
STRIDE = 4
BATCH = 25


# ══════════════════════════════════════════════════════════════════════════════
# The measurement
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DependenceProbe:
    """
    What one occlusion sweep produced: two maps, and the settings behind them.

    A record rather than a dict because the two halves are read by different
    callers for different reasons — the figure wants the pixels, facts.json
    wants the scalars — and a bag mixing (96, 96) arrays with scalars leaves
    every caller to know which keys are which. run_model.py previously carried
    a function whose whole job was knowing that; here the knowledge sits with
    the data.

    The settings are kept alongside the maps because they are not incidental:
    `overlap` moves with the patch size and stride (it read 0.26 at stride 8
    and 0.47 at stride 4), so quoting it without them would be quoting half a
    measurement.
    """

    psi: np.ndarray          # (H, W) — how far psi's token moves, per pixel
    phi: np.ndarray          # (H, W) — the same for phi's token
    probes: int
    patch: int
    stride: int

    @property
    def overlap(self) -> float:
        """
        How much the two dependence maps agree, as a correlation over pixels.

        1.0 would mean the two encoders lean on exactly the same regions.
        Computed because "the maps differ" is the figure's claim, and a claim
        about a picture should come with the number that backs it.
        """
        a = self.psi.ravel() - self.psi.mean()
        b = self.phi.ravel() - self.phi.mean()
        return measure.cosine(a, b)

    def facts(self) -> dict:
        """
        The findings, without the two full-resolution maps.

        facts.json is read by build_page.py and quoted on the page, so it holds
        what the page can say — two 96 x 96 float grids would bloat it by orders
        of magnitude and no page would ever reference a pixel.
        """
        return {
            "probes":      self.probes,
            "patch":       self.patch,
            "stride":      self.stride,
            "psi_peak":    float(self.psi.max()),
            "phi_peak":    float(self.phi.max()),
            "map_overlap": self.overlap,
        }


def _cosine_distance(base: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """1 - cosine similarity, per row. 0 = the token did not move at all."""
    return 1.0 - measure.cosine_to_each(base, moved)


def _occluded(frame: np.ndarray, top: int, left: int, fill: np.ndarray) -> np.ndarray:
    """
    A copy of `frame` with one patch replaced.

    Filled with the frame's own per-channel mean rather than flat grey or black:
    a patch that matches the image statistics removes the *content* of a region
    without also introducing an unusually bright or dark blob, which would move
    the token for a reason that has nothing to do with what was covered.
    """
    out = frame.copy()
    out[top:top + PATCH, left:left + PATCH] = fill
    return out


def probe_dependence(model, sample: dict, device: str) -> DependenceProbe:
    """
    Slide a patch over the current frame and map how far each token moves.

    Both encoders are measured against the *same* occlusions of the *same*
    frame, which is what makes the two maps comparable as shapes.

    phi is also given the goal frame, unoccluded, throughout — the question
    being asked of it is "which parts of the current view does the goal token
    depend on", not "which parts of the goal".
    """
    obs_frames, goal_frame = sample.obs_raw, sample.goal_raw
    current = obs_frames[-1]
    height, width = current.shape[:2]
    fill = current.reshape(-1, 3).mean(axis=0).astype(np.uint8)

    goal_input = to_model_input([goal_frame])
    context = obs_frames[:-1]

    def obs_stack(frame: np.ndarray) -> np.ndarray:
        return to_model_input(list(context) + [frame])

    baseline = model_lib.encode_tokens(
        model, obs_stack(current)[None], goal_input[None], device)
    base_psi, base_phi = baseline.obs_tokens[0, -1], baseline.goal_token[0]

    positions = [(top, left)
                 for top in range(0, height - PATCH + 1, STRIDE)
                 for left in range(0, width - PATCH + 1, STRIDE)]

    # Accumulate each probe's distance over every pixel its patch covered, then
    # divide by the coverage count. Overlapping patches average into a smooth
    # map; assigning each probe to its centre instead would give a blocky
    # 10 x 10 grid and throw away the finer stride.
    totals = {"psi": np.zeros((height, width)), "phi": np.zeros((height, width))}
    counts = np.zeros((height, width))

    for start in range(0, len(positions), BATCH):
        chunk = positions[start:start + BATCH]
        obs_batch = np.stack([obs_stack(_occluded(current, t, l, fill))
                              for t, l in chunk])
        goal_batch = np.repeat(goal_input[None], len(chunk), axis=0)

        probed = model_lib.encode_tokens(model, obs_batch, goal_batch, device)
        distances = {
            # [:, -1] is the current frame — the only one an occlusion of it
            # can have moved. common.model.encode_tokens owns the unpacking.
            "psi": _cosine_distance(base_psi, probed.obs_tokens[:, -1]),
            "phi": _cosine_distance(base_phi, probed.goal_token),
        }

        for i, (top, left) in enumerate(chunk):
            window = (slice(top, top + PATCH), slice(left, left + PATCH))
            counts[window] += 1
            for name, values in distances.items():
                totals[name][window] += values[i]

    return DependenceProbe(
        psi=totals["psi"] / counts,
        phi=totals["phi"] / counts,
        probes=len(positions),
        patch=PATCH,
        stride=STRIDE,
    )


# ══════════════════════════════════════════════════════════════════════════════
# The figure
# ══════════════════════════════════════════════════════════════════════════════

def _draw_map(ax, frame: np.ndarray, dependence: np.ndarray, title: str,
              *, edge: str):
    """
    The frame, drained of colour, with its dependence map laid over it.

    The greyscale is deliberately lifted into the top half of the range before
    the overlay goes on. At full contrast the corridor is mostly dark, the
    overlay sits on top of it, and the result is a dark picture with coloured
    blobs on it that a reader cannot connect to anything they recognise — which
    is exactly the complaint the first version drew.
    """
    grey = frame.mean(axis=2)
    ax.imshow(grey, cmap="gray", vmin=-90, vmax=300)
    # Normalised to its own peak: the two encoders are different networks, so
    # what is comparable between the panels is *where* the map is hot, not how
    # hot. Sharing a scale would make the quieter panel look like nothing.
    heat = ax.imshow(dependence / dependence.max(), cmap="inferno", alpha=0.5,
                     vmin=0, vmax=1)
    viz.plate(ax, edge=edge, width=2.2, title=title, bold=True)
    return heat


def plot_occlusion(sample: dict, probe: "DependenceProbe", save_path):
    """
    Two dependence maps of one frame, side by side, plus the goal for reference.

    The third panel is not decoration: the goal encoder's map cannot be read
    without knowing what goal it was working towards.
    """
    current, goal = sample.now, sample.goal_raw

    fig, axes = plt.subplots(
        1, 4, figsize=(13.0, 4.3),
        gridspec_kw={"width_ratios": [1, 1, 1, 1], "wspace": 0.10},
    )

    # The untouched frame comes first. Without somewhere to look that has no
    # colour on it, a reader has nothing to compare the two maps against and
    # cannot tell which shapes are the room and which are the measurement.
    axes[0].imshow(current)
    viz.plate(axes[0], edge=viz.COLOR_CURRENT, width=2.2, bold=True,
              title="the frame being tested",
              subtitle="no marks — this is the view")
    heat = _draw_map(axes[1], current, probe.psi,
                     "what the camera encoder uses", edge=viz.COLOR_CURRENT)
    _draw_map(axes[2], current, probe.phi,
              "what the goal encoder uses", edge=viz.COLOR_GOAL)
    axes[3].imshow(goal)
    viz.plate(axes[3], edge=viz.COLOR_GOAL, width=2.2, bold=True,
              title="the goal it was aiming at",
              subtitle="a different place, further on")

    fig.text(
        0.5, 0.075,
        "Cover a small square, run it through again, and see whether the "
        "numbers change. They did most where it is brightest.",
        ha="center", fontsize=10, color=viz.COLOR_MUTED,
    )
    fig.text(
        0.5, 0.022,
        f"{probe.probes} squares tested per network · "
        f"{probe.patch}-pixel square · each map scaled to its own brightest spot",
        ha="center", fontsize=8.5, color=viz.COLOR_MUTED,
    )
    fig.subplots_adjust(bottom=0.20, top=0.88)

    # After the layout settles, or the key is placed against positions that
    # subplots_adjust is about to move out from under it.
    viz.settle(fig)
    viz.worded_key(fig, heat, [axes[1], axes[2]],
                   ["barely matters", "matters most"], drop=0.055)

    return viz.save_fig(fig, save_path)

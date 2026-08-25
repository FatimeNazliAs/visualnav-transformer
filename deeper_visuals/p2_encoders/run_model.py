#!/usr/bin/env python3
# deeper_visuals/p2_encoders/run_model.py
"""
P2 — Encoders. Pixels in, one 256-number token per frame out.

The two encoders, since the Greek is easy to mix up (see deeper_visuals/README):

    psi  = vision_encoder.obs_encoder   observation / camera encoder,
                                        3 channels, run once per frame
    phi  = vision_encoder.goal_encoder  goal encoder, 6 channels, run once,
                                        fed the current frame AND the goal

Separate EfficientNet-B0 instances, not one network used twice.

Writes out/p2/<tag>/:
    stage2_token_strips.png        each frame beside the token it encodes to
    stage2_what_it_depends_on.png  which parts of the frame each token needs
    facts.json                     token shapes, encoder sizes, how alike the
                                   tokens are, how much the two maps agree

Adapted from the frozen debug_visuals/visualize_stage2.py. Three things changed.

1. The tokens are *captured off the real forward pass* rather than recomputed.
   The old module reimplemented NoMaD_ViNT.forward's encoder half —
   extract_features -> _avg_pooling -> _dropout -> compress — once per encoder,
   which is a copy of upstream code that can silently drift from it. Here a
   forward hook on each compression layer reads the tokens out while
   model("vision_encoder", …) runs, so what is plotted is by construction what
   the transformer is about to be handed.
2. The goal encoder phi is folded in rather than being its own stage. phi is a
   separate 6-channel EfficientNet-B0 that NoMaD_ViNT feeds the current frame
   concatenated with the goal image, so the figure shows it both frames — one
   row, two thumbnails, and no separate stage-3 figure to reconcile.
3. The scene arrives on a PhaseConfig instead of module globals, and the heavy
   matplotlib suptitle is gone; the advisor page supplies its own heading.

Run (inside the container, from the repo root):
    python3 deeper_visuals/p2_encoders/run_model.py
or, the usual way:
    ./deeper_visuals/p2_encoders/update.sh
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from deeper_visuals.common import measure, settings, viz
from deeper_visuals.common.config import load_config
from deeper_visuals.common.data import load_sample
from deeper_visuals.common.facts import write_facts
from deeper_visuals.common import model as model_lib
from deeper_visuals.common.model import load_model
from deeper_visuals.common.viz import plt
from deeper_visuals.p2_encoders import occlusion

PHASE_DIR = Path(__file__).resolve().parent
FIGURE_NAME = "stage2_token_strips.png"
OCCLUSION_FIGURE_NAME = "stage2_what_it_depends_on.png"

# Column widths, in units of one thumbnail. The figure reads left to right as
# the data does — frames, the network, the numbers it produced.
#
# Column 0 carries a thumbnail only on the goal row: that overhang to the left
# is what makes phi's second input read as an extra thing phi gets, rather than
# as one more frame in the observation sequence.
WIDTH_THUMB, WIDTH_ENCODER, WIDTH_HEATMAP = 1.0, 1.55, 5.8
COL_EXTRA_THUMB, COL_THUMB, COL_ENCODER, COL_HEATMAP = 0, 1, 2, 3

# The goal row, in units of an observation row.
HEIGHT_GOAL_ROW = 1.5


# ══════════════════════════════════════════════════════════════════════════════
# The forward pass
# ══════════════════════════════════════════════════════════════════════════════

def encode_tokens(model, sample: dict, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    This phase's scene, encoded. Returns (N_OBS_FRAMES, 256) and (256,).

    A batch of one over common.model.encode_tokens, which owns the hooks and
    the frame-major unpacking. Nothing about reading tokens out of NoMaD lives
    in this phase.

    Its c_t is ignored here on purpose: this phase stops at the tokens, and c_t
    is only meaningful once the transformer has mixed them — which is P3.
    """
    obs_batch, goal_batch = sample.obs_batch, sample.goal_batch
    encoding = model_lib.encode_tokens(model, obs_batch, goal_batch, device)
    return encoding.obs_tokens[0], encoding.goal_token[0]


# ══════════════════════════════════════════════════════════════════════════════
# The figure
# ══════════════════════════════════════════════════════════════════════════════

def _draw_token(ax, token: np.ndarray, vabs: float, *, xlabel: str = ""):
    """
    One token as a single-row heatmap, on a scale shared with every other row.

    viz.vector_strip owns the 1 x N decision and the reason for it.
    """
    image = viz.vector_strip(ax, token, vabs=vabs)
    viz.plate(ax, edge=viz.COLOR_MUTED, width=0.6, subtitle=xlabel)
    return image


def _draw_encoder(ax, name: str, detail: list[str], *, color: str, fill: str) -> None:
    """
    The network itself, as a box between the frames and the tokens.

    One box per encoder, not one per row: psi spans all four observation rows
    because it *is* one network run four times, and phi gets its own box on the
    goal row. Drawing four separate psi boxes would say there are five networks
    here, which is the misconception this phase exists to correct — and leaving
    the boxes out entirely, as the first draft did, showed pictures turning into
    numbers with the thing that does the turning missing from the picture.
    """
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.04, 0.05), 0.92, 0.90,
        boxstyle="round,pad=0.01,rounding_size=0.05", transform=ax.transAxes,
        facecolor=fill, edgecolor=color, linewidth=1.6, clip_on=False,
    ))
    # Anchored apart rather than both centred: psi's box spans four rows and
    # phi's spans one, and two centred blocks that fit the tall box overlap
    # each other in the short one.
    ax.text(0.5, 0.54, name, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10.5, color=color, fontweight="bold")
    ax.text(0.5, 0.44, "\n".join(detail), transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color=viz.COLOR_MUTED,
            linespacing=1.6)


def _flow_arrow(fig, x_from: float, x_to: float, y: float, *, color: str) -> None:
    """An arrow in figure coordinates, from one laid-out axes edge to another."""
    fig.add_artist(FancyArrowPatch(
        (x_from, y), (x_to, y), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=11, color=color, linewidth=1.4,
        shrinkA=3, shrinkB=3,
    ))


def plot_token_strips(sample: dict, obs_tokens: np.ndarray, goal_token: np.ndarray,
                      encoders: dict, save_path):
    """
    The hero figure, read left to right: frames in, encoder, 256 numbers out.

    `encoders` is measure_encoders' output, so the sizes printed on the boxes
    are the ones measured off the loaded model rather than a second copy of them
    typed into the drawing code — the same rule the page's numbers follow.

    One row per token, every row on the same colour scale within its encoder so
    the rows are comparable. The last row is the goal encoder's and carries two
    thumbnails, because phi is fed the current frame concatenated with the goal
    image. Showing only the goal there would reproduce the misconception this
    phase exists to correct — that there is one shared encoder, or that the goal
    is encoded on its own.
    """
    obs_raw, goal_raw = sample.obs_raw, sample.goal_raw
    obs_idxs, goal_idx = sample.obs_idxs, sample.goal_idx
    n_obs = len(obs_tokens)
    n_rows = n_obs + 1

    # One scale per encoder, not one across both. psi and phi are separate
    # networks, so the raw size of their outputs is not comparable in the first
    # place — and phi's range is the wider of the two, so a single shared scale
    # spends most of it on the goal row and flattens all four camera rows into
    # near-white. That would hide exactly what this figure is for. Zero still
    # means zero on both, which is why the colour key can stay a single bar.
    vabs_obs  = float(np.abs(obs_tokens).max())
    vabs_goal = float(np.abs(goal_token).max())

    fig = plt.figure(figsize=(12.6, 1.32 * (n_obs + HEIGHT_GOAL_ROW) + 1.0))
    grid = gridspec.GridSpec(
        n_rows, 4, figure=fig,
        width_ratios=[WIDTH_THUMB, WIDTH_THUMB, WIDTH_ENCODER, WIDTH_HEATMAP],
        # The goal row carries the taller of the two encoder boxes in a single
        # row, so it gets the slack. Thumbnails are square and width-bound, so
        # this makes the row roomier without making its pictures bigger.
        height_ratios=[1.0] * n_obs + [HEIGHT_GOAL_ROW],
        wspace=0.16, hspace=0.42,
    )

    # ── Observation frames -> encoder psi ─────────────────────────────────────
    obs_rows = []
    for row, (frame, frame_idx, token) in enumerate(zip(obs_raw, obs_idxs, obs_tokens)):
        steps_back = n_obs - 1 - row
        is_current = steps_back == 0
        edge = viz.COLOR_CURRENT if is_current else viz.COLOR_MUTED
        heading = "now  (t)" if is_current else f"t − {steps_back}"

        ax_thumb = fig.add_subplot(grid[row, COL_THUMB])
        ax_thumb.imshow(frame)
        viz.plate(ax_thumb, edge=edge, width=2.6 if is_current else 1.0,
                  title=f"{heading}\nframe {frame_idx}")
        ax_heat = fig.add_subplot(grid[row, COL_HEATMAP])
        obs_image = _draw_token(ax_heat, token, vabs_obs)
        obs_rows.append((ax_thumb, ax_heat))

    ax_psi = fig.add_subplot(grid[:n_obs, COL_ENCODER])
    _draw_encoder(ax_psi, "camera encoder",
                  ["one network,", "run once per frame",
                   f"{encoders['psi_params_m']:.2f}M learned settings"],
                  color=viz.COLOR_MUTED, fill="#f3f5f6")

    # ── Current frame + goal -> encoder phi ───────────────────────────────────
    ax_now  = fig.add_subplot(grid[-1, COL_EXTRA_THUMB])
    ax_goal = fig.add_subplot(grid[-1, COL_THUMB])
    ax_now.imshow(obs_raw[-1])
    viz.plate(ax_now, edge=viz.COLOR_CURRENT, width=2.6,
              title=f"now  (t)\nframe {obs_idxs[-1]}")
    ax_goal.imshow(goal_raw)
    viz.plate(ax_goal, edge=viz.COLOR_GOAL, width=2.6,
              title=f"goal  (t + {settings.NUM_ACTIONS})\nframe {goal_idx}")
    ax_heat_goal = fig.add_subplot(grid[-1, COL_HEATMAP])
    _draw_token(ax_heat_goal, goal_token, vabs_goal,
                xlabel=f"the {goal_token.size} numbers in one token")

    ax_phi = fig.add_subplot(grid[-1, COL_ENCODER])
    _draw_encoder(ax_phi, "goal encoder",
                  ["a different network,", "fed both frames at once",
                   f"{encoders['phi_params_m']:.2f}M learned settings"],
                  color=viz.COLOR_GOAL, fill="#f7f1fa")

    # Room under the rows for the colour key, which is outside every axes.
    fig.subplots_adjust(left=0.035, right=0.985, top=0.94, bottom=0.14)

    viz.settle(fig)
    for ax_thumb, ax_heat in obs_rows:
        _connect(fig, ax_thumb, ax_psi, ax_heat, color=viz.COLOR_MUTED)
    _connect(fig, ax_goal, ax_phi, ax_heat_goal, color=viz.COLOR_GOAL)
    _annotate_goal_row(fig, ax_now, ax_goal)
    viz.worded_key(fig, obs_image, [ax_heat_goal], ["lower", "0", "higher"],
                   caption="each encoder on its own scale", drop=0.085)

    return viz.save_fig(fig, save_path)


def _connect(fig, ax_thumb, ax_encoder, ax_heatmap, *, color: str) -> None:
    """Frame -> encoder -> token, as two arrows on one row's centre line."""
    thumb, encoder, heatmap = (a.get_position()
                               for a in (ax_thumb, ax_encoder, ax_heatmap))
    y = (heatmap.y0 + heatmap.y1) / 2
    _flow_arrow(fig, thumb.x1, encoder.x0, y, color=color)
    _flow_arrow(fig, encoder.x1, heatmap.x0, y, color=color)


def _annotate_goal_row(fig, ax_now, ax_goal) -> None:
    """The '+' that says phi gets these two frames stacked, not one of them."""
    now_box, goal_box = ax_now.get_position(), ax_goal.get_position()
    fig.text(
        (now_box.x1 + goal_box.x0) / 2, (goal_box.y0 + goal_box.y1) / 2, "+",
        ha="center", va="center", fontsize=17, color=viz.COLOR_GOAL, fontweight="bold",
    )


# ══════════════════════════════════════════════════════════════════════════════
# The numbers
# ══════════════════════════════════════════════════════════════════════════════

def measure_encoders(model) -> dict:
    """
    The two encoders, measured off the loaded model rather than asserted.

    psi and phi are separate EfficientNet-B0 instances — not one shared network
    used twice. phi differs in its stem: 6 input channels, because it is fed the
    current frame and the goal image concatenated. Both end in a linear layer
    that compresses EfficientNet's 1280 pooled features to the 256 the
    transformer works in.
    """
    encoder = model.vision_encoder
    psi_params = sum(p.numel() for p in encoder.obs_encoder.parameters())
    phi_params = sum(p.numel() for p in encoder.goal_encoder.parameters())
    compress_params = sum(
        p.numel()
        for layer in (encoder.compress_obs_enc, encoder.compress_goal_enc)
        for p in layer.parameters()
    )

    return {
        "backbone":          "EfficientNet-B0",
        "shared":            encoder.obs_encoder is encoder.goal_encoder,
        "psi_in_channels":   encoder.obs_encoder._conv_stem.in_channels,
        "phi_in_channels":   encoder.goal_encoder._conv_stem.in_channels,
        "psi_params":        psi_params,
        "phi_params":        phi_params,
        "psi_params_m":      measure.millions(psi_params),
        "phi_params_m":      measure.millions(phi_params),
        "encoder_params_m":  measure.millions(
                                 psi_params + phi_params + compress_params),
        "feature_dim":       encoder.num_obs_features,
        "token_dim":         encoder.obs_encoding_size,
        "compression":       f"{encoder.num_obs_features} → {encoder.obs_encoding_size}",
    }


def measure_tokens(sample: dict, obs_tokens: np.ndarray, goal_token: np.ndarray) -> dict:
    """
    What the encoders produced, measured off the captured tokens.

    The two similarity numbers are the phase's payoff and the reason they are
    worth computing at all: four frames of the same corridor land on nearly the
    same token, and the goal — a different place — lands somewhere else. That is
    what "these numbers mean something" looks like without opening the network.
    """
    numbers_per_frame = int(np.prod(sample.obs_raw[0].shape))
    token_dim = obs_tokens.shape[1]
    all_tokens = np.vstack([obs_tokens, goal_token[None, :]])

    return {
        "n_tokens":           len(all_tokens),
        "n_obs_tokens":       len(obs_tokens),
        "token_dim":          token_dim,
        "obs_tokens_shape":   f"{obs_tokens.shape[0]} × {obs_tokens.shape[1]}",
        "goal_token_shape":   f"1 × {goal_token.size}",
        "numbers_per_frame":  numbers_per_frame,
        "shrink_factor":      numbers_per_frame // token_dim,
        "value_min":          float(all_tokens.min()),
        "value_max":          float(all_tokens.max()),
        "obs_similarity":     measure.mean_cosine(obs_tokens, obs_tokens),
        "goal_similarity":    measure.mean_cosine(obs_tokens, goal_token[None, :]),
    }


def main() -> None:
    cfg = load_config(PHASE_DIR)
    print(f"\n  P2 — Encoders\n  {cfg.summary()}\n  {cfg.description}\n")

    sample = load_sample(cfg)
    model, info = load_model(cfg)

    obs_tokens, goal_token = encode_tokens(model, sample, info["device"])
    print(f"  obs tokens : {obs_tokens.shape}")
    print(f"  goal token : {goal_token.shape}")

    encoders = measure_encoders(model)
    strips = plot_token_strips(sample, obs_tokens, goal_token, encoders,
                               cfg.out_dir / FIGURE_NAME)

    print("  Probing what each encoder depends on …")
    probe = occlusion.probe_dependence(model, sample, info["device"])
    occlusion_map = occlusion.plot_occlusion(
        sample, probe, cfg.out_dir / OCCLUSION_FIGURE_NAME)
    print(f"  map overlap: {probe.overlap:.2f}")

    write_facts(
        cfg,
        {
            "model":      info,
            "encoders":   encoders,
            "tokens":     measure_tokens(sample, obs_tokens, goal_token),
            "dependence": probe.facts(),
        },
        figures=[strips, occlusion_map],
    )


if __name__ == "__main__":
    main()

# debug/run_pipeline_report.py
"""
Same pipeline as run_pipeline.py, but instead of printing everything to the
terminal it captures each stage's output and writes a single Markdown report
to debug/reports/. Each section is labelled with what it is (which trajectory/
frame was used, what stage produced it, what the shapes/values mean) so you
can scroll one file instead of a long terminal dump.

Run from the repo root inside the container:
    conda activate vint_train
    cd /app/visualnav-transformer
    python -m debug.run_pipeline_report

Output:
    debug/reports/<traj_name>_f<frame_idx>_<timestamp>.md

To debug a different sample, edit TRAJ_NAME and FRAME_IDX in debug/config.py
(same as run_pipeline.py).
"""
import sys
import io
import contextlib
import inspect
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from debug.config import (
    CHECKPOINT, DEVICE, CONTEXT_SIZE, NUM_ACTIONS,
    ENCODING_SIZE, DOWN_DIMS, K_DENOISING, TRAJ_NAME, FRAME_IDX,
)
from debug import stage1_data, stage2_obs_enc, stage3_goal_enc
from debug import stage4_transformer, stage5_diffusion

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


@contextlib.contextmanager
def _capture():
    """Captures everything printed inside the `with` block and yields a
    list whose single element is filled in with the captured text once the
    block exits."""
    buf = io.StringIO()
    box = [None]
    with contextlib.redirect_stdout(buf):
        yield box
    box[0] = buf.getvalue()


def _doc(fn):
    """First paragraph of a function's docstring, for a one-line summary."""
    doc = inspect.getdoc(fn)
    if not doc:
        return ""
    return doc.strip()


def _section(title, summary, captured_text):
    parts = [f"## {title}\n"]
    if summary:
        parts.append(f"{summary}\n")
    parts.append("```\n" + captured_text.rstrip() + "\n```\n")
    return "\n".join(parts)


def load_model():
    print(f"  Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=ENCODING_SIZE,
        context_size=CONTEXT_SIZE,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=ENCODING_SIZE,
        down_dims=DOWN_DIMS,
        cond_predict_scale=False,
    )
    dist_pred_net = DenseNetwork(embedding_dim=ENCODING_SIZE)

    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_net,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING — missing keys  : {missing[:3]} ...")
    if unexpected:
        print(f"  WARNING — unexpected keys: {unexpected[:3]} ...")

    model = model.to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters : {n_params:.1f}M")
    print(f"  Device     : {DEVICE}")
    return model


def main():
    sections = []
    t0 = datetime.now()

    print("Running NoMaD inference pipeline -> Markdown report ...")

    with _capture() as box:
        model = load_model()
    sections.append(_section("Model Loading", "", box[0]))
    print("  [1/6] model loaded")

    with _capture() as box:
        obs, goal = stage1_data.load_one_sample()
    sections.append(_section(
        "Stage 1 — Data loading", _doc(stage1_data.load_one_sample), box[0]
    ))
    print("  [2/6] stage 1 (data loading) done")

    with _capture() as box:
        obs_tokens = stage2_obs_enc.run_obs_encoder(model, obs)
    sections.append(_section(
        "Stage 2 — Observation encoder (ψ)",
        _doc(stage2_obs_enc.run_obs_encoder), box[0]
    ))
    print("  [3/6] stage 2 (observation encoder) done")

    with _capture() as box:
        goal_token = stage3_goal_enc.run_goal_encoder(model, obs, goal)
    sections.append(_section(
        "Stage 3 — Goal encoder (φ)",
        _doc(stage3_goal_enc.run_goal_encoder), box[0]
    ))
    print("  [4/6] stage 3 (goal encoder) done")

    with _capture() as box:
        ct, dist = stage4_transformer.run_transformer(model, obs, goal, mask=False)
        print()  # blank line between the two sub-runs in the captured text
        stage4_transformer.compare_masked_vs_unmasked(model, obs, goal)
    sections.append(_section(
        "Stage 4 — Transformer + Goal Masking",
        _doc(stage4_transformer.run_transformer), box[0]
    ))
    print("  [5/6] stage 4 (transformer + goal masking) done")

    with _capture() as box:
        a_final = stage5_diffusion.run_diffusion(model, ct)
    sections.append(_section(
        f"Stage 5 — Diffusion denoising (K={K_DENOISING})",
        _doc(stage5_diffusion.run_diffusion), box[0]
    ))
    print("  [6/6] stage 5 (diffusion denoising) done")

    # ── Final output table ──────────────────────────────────────────────
    final_lines = [
        f"  {'step':>5}  {'linear_vel':>12}  {'angular_vel':>12}",
        f"  {'────':>5}  {'──────────':>12}  {'──────────':>12}",
    ]
    for h in range(NUM_ACTIONS):
        v = a_final[0, h, 0].item()
        w = a_final[0, h, 1].item()
        final_lines.append(f"  {h:>5}  {v:>12.4f}  {w:>12.4f}")
    final_text = "\n".join(final_lines)
    final_text += f"\n\n  dist_pred : {dist[0, 0].item():.2f} steps to goal"
    sections.append(_section(
        "Final Output — Predicted Action Sequence",
        "Normalised velocities (normalize=True in yaml). Only step 0 would "
        "go to the robot in deployment. Re-running gives a different "
        "sequence — that's the multimodality of the diffusion policy.",
        final_text,
    ))

    # ── Assemble report ─────────────────────────────────────────────────
    duration = (datetime.now() - t0).total_seconds()
    header = (
        f"# NoMaD Inference Pipeline — Debug Report\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| Generated | {t0.strftime('%Y-%m-%d %H:%M:%S')} |\n"
        f"| Run time | {duration:.1f}s |\n"
        f"| Trajectory | `{TRAJ_NAME}` |\n"
        f"| Frame idx (t) | {FRAME_IDX} |\n"
        f"| Context size (P) | {CONTEXT_SIZE} |\n"
        f"| Action horizon (H) | {NUM_ACTIONS} |\n"
        f"| Denoising steps (K) | {K_DENOISING} |\n"
        f"| Checkpoint | `{CHECKPOINT}` |\n"
        f"| Device | {DEVICE} |\n"
    )

    report = header + "\n" + "\n".join(sections)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{TRAJ_NAME}_f{FRAME_IDX}_{t0.strftime('%Y%m%d_%H%M%S')}.md"
    out_path.write_text(report)

    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()

# debug_visuals/run_pipeline.py
"""
Entry point for the NoMaD debug_visuals pipeline (presentation version).

Generates 8 presentation-ready PNGs per sample, organized by stage:

    stage0_architecture.png       — architecture overview diagram
    stage1_frame_strip.png        — input frames (obs context + goal)
    stage2_encoder_tokens.png     — image → 256-dim token heatmaps
    stage4_attention.png          — transformer attention (nav vs explore)
    stage4_ct_comparison.png      — goal masking effect on context vector
    stage5_1_denoising_strip.png  — noise → clean trajectory
    stage5_2_final_trajectory.png — final path on camera view
    stage5_3_multimodal_runs.png  — multiple valid paths from same input

Run from the repo root inside the container:

    conda activate vint_train
    cd /app/visualnav-transformer
    python -m debug_visuals.run_pipeline

To visualise a different sample, edit TRAJ_NAME and FRAME_IDX in
debug_visuals/config.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from debug_visuals.config import TRAJ_NAME, FRAME_IDX, OUTPUTS_DIR
from debug_visuals import visualize_stage0
from debug_visuals import visualize_stage1
from debug_visuals import visualize_stage2
from debug_visuals import visualize_stage4
from debug_visuals import visualize_stage5

SEP = "─" * 60


def main() -> None:
    run_dir = OUTPUTS_DIR / f"{TRAJ_NAME}_f{FRAME_IDX}"

    print(f"\n{SEP}")
    print("  NoMaD — debug_visuals pipeline (presentation)")
    print(f"  Sample  : {TRAJ_NAME}  frame {FRAME_IDX}")
    print(f"  Outputs : {run_dir}")
    print(SEP)

    print(f"\n── Stage 0: Architecture diagram ───────────────────────")
    visualize_stage0.plot_architecture(save_path=run_dir / "stage0_architecture.png")

    print(f"\n── Stage 1: Model input ────────────────────────────────")
    sample = visualize_stage1.load_sample_frames()
    visualize_stage1.plot_frame_strip(
        sample, save_path=run_dir / "stage1_frame_strip.png"
    )

    print(f"\n── Stage 2: Encoder tokens ψ / φ ───────────────────────")
    model = visualize_stage2.load_model()
    obs_tokens = visualize_stage2.extract_obs_tokens(model, sample["obs_raw"])
    goal_token = visualize_stage2.extract_goal_token(
        model, sample["obs_raw"], sample["goal_raw"]
    )
    visualize_stage2.plot_token_heatmap(
        obs_tokens, goal_token,
        sample["obs_raw"], sample["goal_raw"],
        save_path=run_dir / "stage2_encoder_tokens.png",
    )

    print(f"\n── Stage 4: Transformer + goal masking ─────────────────")
    result4 = visualize_stage4.run_stage4(model, obs_tokens, goal_token, save_dir=run_dir)

    print(f"\n── Stage 5: Diffusion denoising ─────────────────────────")
    visualize_stage5.run_stage5(
        model, result4["ct_nav"], sample["obs_raw"][-1], save_dir=run_dir
    )

    print(f"\n{SEP}")
    print(f"  Done — 8 PNGs saved to {run_dir}")
    print("  To try another sample: edit TRAJ_NAME / FRAME_IDX in")
    print("  debug_visuals/config.py, then re-run.")
    print(SEP)


if __name__ == "__main__":
    main()

# debug_visuals/run_pipeline.py
"""
Entry point for the NoMaD debug_visuals pipeline.

Drives each stage's plotting functions in sequence and saves their PNGs
under a per-sample subfolder, so different trajectories/frames don't
collide or pile up in one flat directory:

    debug_visuals/outputs/<traj>_f<idx>/stage1_frame_strip.png
    debug_visuals/outputs/<traj>_f<idx>/stage1_pixel_distributions.png
    …

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
from debug_visuals import visualize_stage1
from debug_visuals import visualize_stage2

SEP = "─" * 60


def main() -> None:
    run_dir = OUTPUTS_DIR / f"{TRAJ_NAME}_f{FRAME_IDX}"

    print(f"\n{SEP}")
    print("  NoMaD — debug_visuals pipeline")
    print(f"  Sample  : {TRAJ_NAME}  frame {FRAME_IDX}")
    print(f"  Outputs : {run_dir}")
    print(SEP)

    print(f"\n── Stage 1: Model input ────────────────────────────────")
    sample = visualize_stage1.load_sample_frames()
    visualize_stage1.plot_frame_strip(
        sample, save_path=run_dir / "stage1_frame_strip.png"
    )
    visualize_stage1.plot_pixel_distributions(
        sample, save_path=run_dir / "stage1_pixel_distributions.png"
    )
    visualize_stage1.plot_normalised_comparison(
        sample, save_path=run_dir / "stage1_normalised_comparison.png"
    )

    print(f"\n── Stage 2: Encoder tokens ψ / φ ───────────────────────")
    model = visualize_stage2.load_model()
    obs_tokens = visualize_stage2.extract_obs_tokens(model, sample["obs_raw"])
    goal_token = visualize_stage2.extract_goal_token(
        model, sample["obs_raw"], sample["goal_raw"]
    )
    visualize_stage2.plot_token_barchart(
        obs_tokens, goal_token, save_path=run_dir / "stage2_obs_tokens.png"
    )

    print(f"\n{SEP}")
    print(f"  Done.  Open the PNGs in {run_dir} to inspect.")
    print("  To try another sample: edit TRAJ_NAME / FRAME_IDX in")
    print("  debug_visuals/config.py, then re-run.")
    print(SEP)


if __name__ == "__main__":
    main()

# vint_train/mapmad/ — MapMaD, NoMaD side

Runs in the `naz_mapmad` container (conda env `vint_train`, Python 3.8).

`closed_loop/` (Phase 1, reused in Phase 5): a policy drives the virtual LIMO in Habitat through the simulator
bridge (`mapmad/src/mapmad_bridge`, so `mapmad/src` must be on `PYTHONPATH`).
- `policy.py` — the interface: `reset(meta)`, `act(obs) -> Command(v, w)`; never target position or geodesic.
- `deployment.py` — `transform_images` and `pd_controller` copied from `deployment/src` without ROS
  (`tests/test_deployment_rules.py` checks they give exactly what the originals give).
- `nomad_policy.py` — the official NoMaD with the deployment pipeline (4 frames, 8 samples, sample 0, waypoint 2).
- `runner.py` — one episode: JSONL log per step (byte-identical for the same seed), result, video.
- `metrics.py`, `tables.py` — SR, success@N, SPL, collisions, bootstrap CIs, paired differences, SECONDARY metrics.
- `analysis.py` — from the step logs: stuck streaks, failure types (reads logs with `mapmad_bridge.steplog`).
- `diagnostics.py` — panels of what NoMaD was given in a stuck streak (after `mapmad/scripts/p1_replay.py frames`).
- `video.py` — mp4 (H.264 via imageio-ffmpeg): camera + top-down map + caption; overlays only on video frames.
- `run_arms.py` — command line (see `mapmad/README.md`, "Run the Phase 1 baseline").

Phase 2 (practice drives; numpy/OpenCV only, so the Habitat container imports them too — keep `__init__.py` light):
- `local_map.py` — cuts the robot-centred 64 × 64 window (10 cm cells, forward = up, left = left) out of a drive's
  `mapmad_map.npz` at frame t: a cell counts only if its first-seen frame is ≤ t (no future information).
  `local_map(load_drive_map(drive_dir), t, (x, y, yaw))` with the pose from `traj_data.pkl`.
- `map_viz.py` — pictures of a local map (unknown dark grey, explored light grey, obstacle red, robot, FOV, target).
- `format_check.py` — gate G2 row 3: the unchanged `ViNT_Dataset` reads a MapMaD dataset (built as `train.py` does).
- `tests/test_local_map.py` (centre, rotation sign, first-seen cut), `tests/test_data_conventions.py`
  (`to_local_coords` on GoStanford, a simulated Habitat drive and our drives: straight → x > 0, left turn → y > 0).

Planned contents:

- map encoder: 64 × 64 local map (obstacles · explored · goal heat) → one extra token (Phase 3);
- goal heat drawing, incl. the warm spot on the map border for far targets (Phase 3).

Original NoMaD files get only small hooks behind `map_input` (`false` = the old NoMaD, byte-identical).
Extra packages go in `train/requirements-mapmad.txt` (Phase 1: imageio-ffmpeg, pytest).

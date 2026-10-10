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

Phase 3 (the MapMaD model; NoMaD + one map token):

**The switch.** `map_input: true` in a train config builds MapMaD; `false` (or missing) is the old NoMaD, byte for
byte (gate G3 row 1). With the switch on:
- `map_encoder.py` turns the map [3, 64, 64] (obstacle 0/1 · explored 0/1 · heat 0-1) into one 256-number token:
  4 conv layers (3x3, stride 2, GroupNorm, ReLU; 3→32→64→128→128) → flatten 4×4×128 → Linear → 256. The last
  Linear starts at zero, so at step 0 the map token is just its positional row.
- `nomad_vint.py` appends the map token LAST (6 tokens: 4 frames, goal, map); the sine positional table grows 5 → 6
  rows. Two masks per sample, `input_goal_mask` and `input_map_mask` (1 = hidden); a hidden token is excluded from
  attention and from the pooling. Map hidden → exactly NoMaD's old pooling (G3 row 1b); map shown → mean over the
  visible tokens.
- `weights.py` loads the official `nomad.pth` strictly; only `vision_encoder.map_encoder.*` is new.

**The five Habitat modes** (`modes.py`; drawn per sample with a Generator keyed on (seed, epoch, draw)):

| mode | photo goal | map | heat | share |
|---|---|---|---|---|
| photo | shown | hidden | – | 20% |
| map_goal | hidden | shown | on | 30% |
| photo_map | shown (same drive, never a negative) | shown | on | 15% |
| explore_map | hidden | shown | all 0 | 15% |
| explore | hidden | hidden | – | 20% |

GoStanford: map hidden (all-zero map), photo shown/hidden 50/50, NoMaD's own goal sampling, negatives and action
mask. Photo goals for Habitat: k ∈ {0..min(20, (len − 1 − t) // s)} waypoints ahead at spacing s (k = 0 → a frame
of another drive). The distance loss counts only shown photos.

**The heat rules** (`heat.py`, Gaussian σ 0.3 m, peak 1, drawn with the same pose functions as `local_map.py`):
- spot target: exp(−d²/2σ²), d = distance from the cell centre to the spot;
- object target: d = distance to the footprint rectangle (the box around the object's oriented box, from
  `mapmad_meta.json`), so heat = 1 on the footprint; drawn clipped when partly inside;
- target (whole footprint) outside the 6.4 m window: one blob where the line robot → target centre crosses the
  window edge, moved 0.3 m back toward the robot;
- training only, in 30% of heat-on samples, one of: target shifted 0.3–1.0 m · σ × 2 · one false blob ≥ 1 m away.

**Files.** `map_sample.py` (map tensor of one sample, per-worker LRU of floor maps), `dataset.py` (`MapMaDDataset`:
NoMaD's sample + map, goal_mask, map_mask, mode; collision drives dropped; index cached per spacing),
`sampler.py` (epoch = 200,000 weighted draws, 70% Habitat / 30% GoStanford, reseeded per epoch), `train_setup.py`
+ `train_loop.py` (used by `train.py`; per-mode losses, lr stepped once per epoch, unwrapped checkpoints,
`run_info.json` + `metrics.jsonl`), `config.py` (`base_config:` layered configs), `reference_forward.py` +
`checks_g3.py` (G3 rows 1–3), `offline_eval.py` (G3 rows 4–5), `sample_sheet.py` (stop-T picture check).

**Train** (inside `naz_mapmad`, from `/app/visualnav-transformer/train`; `nvidia-smi` first; one GPU per job —
DataParallel on 2 GPUs was 2.3× slower):

    python train.py -c config/mapmad_smoke.yaml       # 20 steps on the tiny splits
    python train.py -c config/mapmad.yaml             # the Phase 3 run (5 epochs)

**Offline check** (the frozen list: `/outputs/mapmad/p3_model/offline/offline_samples.json`, sha256 in
`offline_samples.sha256` next to this README):

    CUDA_VISIBLE_DEVICES=0 python -m vint_train.mapmad.offline_eval score --config config/mapmad.yaml \
        --checkpoint <run>/ema_latest.pth --list /outputs/mapmad/p3_model/offline/offline_samples.json --out <dir>
    python -m vint_train.mapmad.offline_eval compare --scores <dir>/scores.npz --out <dir>
    CUDA_VISIBLE_DEVICES=0 python -m vint_train.mapmad.offline_eval gostanford --config config/mapmad.yaml \
        --checkpoint <run>/ema_latest.pth --official /outputs/mapmad/weights/official/nomad.pth --out <dir>

**G3 rows 1–3:** `python -m vint_train.mapmad.checks_g3 ...` (command in its docstring). Tests:
`PYTHONPATH=/app/visualnav-transformer/mapmad/src python -m pytest -q vint_train/mapmad/tests`.

Original NoMaD files get only small hooks behind `map_input` (`false` = the old NoMaD, byte-identical):
`nomad_vint.py`, `nomad.py`, `vint_dataset.py` (`_index_path`), `train.py` (loaders, weight load, loop,
`logs_root`, `base_config`). Extra packages go in `train/requirements-mapmad.txt` (none added in Phase 3).

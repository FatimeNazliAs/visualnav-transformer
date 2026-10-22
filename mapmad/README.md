# mapmad/ — MapMaD, Habitat side

Code that runs in the `naz_mapmad_habitat` container: the simulator setup (Phase 0), the virtual LIMO and the
simulator server for closed-loop runs (Phase 1), later the practice-drive generator (Phase 2) and the exam
(Phase 5). The NoMaD side lives in `train/vint_train/mapmad/` (closed-loop policy and runner: `closed_loop/`).

- `docker/` — the two containers (`Dockerfile.habitat`, `run_containers.sh`, `paths.env`); see `docker/README.md`.
- `configs/paths.yaml` — every path as seen inside the containers; override one with `MAPMAD_<NAME>`.
- `configs/robot_limo.yaml` — LIMO camera, depth, lidar, drive and compute facts (plan D9), measured 2026-10-09; incl. the sim camera pitch, the sim motion settings (`sim`) and the LIMO-sized floor map (`navmesh`; radius 0.195 m and height 0.55 m from the AgileX LIMO Cobot spec sheet).
- `configs/p1_baseline.yaml` — Phase 1 episodes, run settings, NoMaD settings, arms and paired comparisons.
- `configs/p2_datagen.yaml`, `configs/p2_home_splits.json` — Phase 2 generator settings and frozen home splits.
- `configs/objectnav_categories.yaml` — HM3D object names -> ObjectNav categories, learnt from the train goals (`scripts/build_category_map.py`).
- `src/mapmad_sim/` — the Python package (`config.py`: paths, HM3D file names; `camera.py`: FOV, horizon row, floor-plane fit; `run_info.py`: config + seed + git commit per run; `robot.py`: the virtual LIMO; `objects.py`: object boxes; `objectnav.py`: the only reader of ObjectNav goal files; `episodes.py`: Phase 1 episodes; `categories.py`: name map; `run_layout.py`: run config + where every run input/output lives; `sidecar_server.py`: simulator server; Phase 2: `frames.py` Habitat -> NoMaD 2D frame, `home_splits.py`, `drive_sampler.py` starts/targets, `floor_grid.py` + `expert.py` the expert driver, `route.py` variations + detour routes, `depth_map.py` first-seen maps, `datagen.py` drive loop + writer).
- `src/mapmad_bridge/` — used in both containers: wire format + client of the simulator bridge (numpy), `steplog.py` (reading step logs; stdlib only).
- `scripts/` — `bench_render.py` (G0 row 1 speed), `render_test.py` (G0 row 2 pictures), `objectnav_stats.py` (G0 row 3), `horizon_check.py` (G0 row 4: sim vs real horizon), `build_category_map.py`, `make_p1_episodes.py`, `run_p1.sh`, `p1_replay.py` (Phase 1), `make_p2_splits.py`, `generate_drives.py`, `run_p2_gen.sh`, `p2_report.py` (Phase 2).
- `tests/` — `pytest` checks.

The package depends on habitat-starter (habitat-sim 0.3.3, Python 3.9) as installed in the image at
`/opt/habitat-starter`; extra packages go in `pyproject.toml` and are baked into the image. The source is not
installed: the image puts `/app/visualnav-transformer/mapmad/src` on `PYTHONPATH`, so edits apply at once.

## Setup from scratch (on cukurovaai, normal terminal)

```bash
cd ~/projects && unzip -n habitat-starter.zip              # -> ~/projects/habitat-starter (holds the habitat-sim package)
cd ~/projects/nomad-mapmad
screen -S mapmad_build mapmad/docker/run_containers.sh build   # image naz-mapmad-habitat:0.1 (~5 min)
mapmad/docker/run_containers.sh up                         # network mapmad-net + naz_mapmad_habitat + naz_mapmad
mapmad/docker/run_containers.sh check                      # both containers see 2 GPUs
```

HM3D Semantics v0.2 (one-off, ~1 min; our copy is byte-identical to the labels already inside the shared scene folder):

```bash
D=/mnt/shared_disk/nazli/hm3d_semantics_v0.2
for s in minival val train; do mkdir -p $D/$s; for k in configs annots; do
  tar -xkf /mnt/shared_disk/hm3d/hm3d-$s-semantic-$k-v0.2.tar -C $D/$s; done; done
```

## Phase 0 checks (inside the Habitat container)

```bash
docker exec -it naz_mapmad_habitat bash
cd /opt/habitat-starter && python scripts/check_setup.py && python -m pytest -q -p no:cacheprovider   # renderer + toolbox tests
cd /app/visualnav-transformer
python mapmad/scripts/bench_render.py --gpu 0 && python mapmad/scripts/bench_render.py --gpu 1
python mapmad/scripts/render_test.py --seed 7
python mapmad/scripts/objectnav_stats.py
python mapmad/scripts/horizon_check.py          # needs p0_setup/robot/limo_frames/
cd mapmad && python -m pytest -q -p no:cacheprovider
```

Outputs go to `/outputs/mapmad/p0_setup/` (= `/mnt/shared_disk/nazli/nomad_outputs/mapmad/p0_setup/`).

## Run the Phase 1 baseline (normal NoMaD in Habitat)

Inputs: the official `nomad.pth` + released `nomad.yaml` in `/outputs/mapmad/weights/official/` (downloaded from
the upstream README's Google Drive; see `SOURCE.md`, `SHA256SUMS` there). From the worktree, on the host:

```bash
# 1. episodes (Habitat container, ~1 min): 10 labelled train homes x 3 per type, frozen with a sha256
docker exec naz_mapmad_habitat python mapmad/scripts/make_p1_episodes.py --gpu 0      # --force to rebuild

# 2. tests (both containers; bridge + step-log tests run in both)
docker exec -w /app/visualnav-transformer/mapmad naz_mapmad_habitat python -m pytest -q -p no:cacheprovider
docker exec -w /app/visualnav-transformer/mapmad -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad \
    python -m pytest -q -p no:cacheprovider tests/test_bridge.py tests/test_steplog.py
docker exec -w /app/visualnav-transformer/train -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad \
    python -m pytest -q -p no:cacheprovider vint_train/mapmad/tests

# 3. determinism (gate G1 row 1): the same episode twice on one GPU -> byte-identical logs
for r in a b; do PAIRS=1 mapmad/scripts/run_p1.sh --arms explore_oov --episodes p1-oov-00081-000 --subdir checks/determinism_$r; done
cmp /mnt/shared_disk/nazli/nomad_outputs/mapmad/p1_baseline/checks/determinism_{a,b}/logs/explore_oov/p1-oov-00081-000.jsonl

# 4. all arms (nvidia-smi first; ~1 h on 2 pairs per GPU; in screen)
screen -S mapmad_p1_runs
PAIRS="0 0 1 1" mapmad/scripts/run_p1.sh --arms explore_oov photo_iv photo_oov explore_iv photo_iv_fov120 photo_oov_fov120

# 5. diagnostics from the logs (no policy runs: the camera is put back on logged poses)
docker exec naz_mapmad_habitat python mapmad/scripts/p1_replay.py anyside     # SECONDARY view-point metrics
docker exec naz_mapmad_habitat python mapmad/scripts/p1_replay.py black       # mesh holes in stuck inputs
docker exec naz_mapmad_habitat python mapmad/scripts/p1_replay.py frames --arm photo_iv --episode p1-iv-00081-000 --step 118
docker exec -w /app/visualnav-transformer/train -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad \
    python -m vint_train.mapmad.closed_loop.diagnostics                       # panels for the frames above

# 6. tables
docker exec -w /app/visualnav-transformer/train -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad \
    python -m vint_train.mapmad.closed_loop.tables
```

Outputs in `/outputs/mapmad/p1_baseline/`: `episodes/`, `logs/<arm>/<episode>.jsonl` (one line per step; no
times), `summaries/`, `videos/<arm>/<episode>.mp4`, `timing/`, `run_info.*.json`, `anyside/`, `black/`,
`diagnostics/`, `tables.md`, `tables.csv`, `paired.csv` (layout: `mapmad_sim/run_layout.py`).

Rerun with another config (every script above takes `--config`): `configs/p1_baseline_spec.yaml` is the same run
with LIMO's official size (spec sheet), written to `/outputs/mapmad/p1_baseline_spec/`. Before reusing a frozen
episode file on a changed floor map, check it: `python mapmad/scripts/check_p1_episodes.py` (start and target
point navigable, on the real floor, on one island, geodesic in range; report `floor_map_check.json` next to the
episode file); if any episode fails, rebuild the episodes. The runner refuses an episode whose stored floor-map
hash differs from the live one unless that report passed it on the live floor map.

How a run works: `run_p1.sh` starts one simulator server per pair in `naz_mapmad_habitat`
(`mapmad_sim.sidecar_server`, TCP on `mapmad-net`, never published) and one NoMaD client per pair in
`naz_mapmad` (`vint_train.mapmad.closed_loop.run_arms`), with a fresh shared key in the environment. Every 0.25 s
the client sends (v, w); the server moves the virtual LIMO and returns the next picture.

## Generate the practice drives (Phase 2)

Config: `configs/p2_datagen.yaml` (drive mix, sampling, variations, expert driver, discard rule, map). Frozen home
splits: `configs/p2_home_splits.json` (val-drive = 10 Phase 1 homes + 30 random unlabelled train homes; pilot = 20
homes; each list with its sha256). From the worktree, on the host:

```bash
# 0. (once) freeze the home splits -- already done; --force would redraw them
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/make_p2_splits.py homes

# 1. tests (both containers)
docker exec -w /app/visualnav-transformer/mapmad naz_mapmad_habitat python -m pytest -q -p no:cacheprovider
docker exec -w /app/visualnav-transformer/train -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad \
    python -m pytest -q -p no:cacheprovider vint_train/mapmad/tests

# 2. pilot (20 homes x 10 drives, ~2 min) and its report (stats, 3 videos, 10 alignment overlays)
nvidia-smi && mapmad/scripts/run_p2_gen.sh pilot 2
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p2_report.py --dataset pilot

# 3. full run: 800 train homes, 12,200 drives, 12 workers (6 per GPU, 1 CPU core each), ~0.5 h; restart = resume
nvidia-smi && df -h /mnt/shared_disk && mapmad/scripts/run_p2_gen.sh full 6
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p2_report.py --dataset full --videos 0 --overlays 10

# 4. NoMaD split files (+ a tiny split for the format check); the loader's LMDB cache lands next to them
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/make_p2_splits.py traj-names --dataset full
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/make_p2_splits.py traj-names --dataset full --tiny 4
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/make_p2_splits.py collisions --dataset full  # drives to leave out

# 5. format check (gate G2 row 3): the unchanged loader reads the full dataset, then one training epoch on the tiny split
docker exec -w /app/visualnav-transformer/train -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad \
    python -m vint_train.mapmad.format_check --config config/mapmad_format_check.yaml \
    --split-dir /mapmad_data/splits/habitat_mapmad --out /outputs/mapmad/p2_datagen/format_check/loader.json
docker exec naz_mapmad bash -c 'R=/outputs/mapmad/p2_datagen/format_check/run && mkdir -p $R && cd $R && \
    ln -sfn /app/visualnav-transformer/train/config config && \
    python /app/visualnav-transformer/train/train.py -c config/mapmad_format_check.yaml'
```

The first command builds the full splits' LMDB cache (~= the JPG size, next to the split files). train.py writes
`logs/` into its working directory, hence the run folder.

What a drive folder holds (`<mapmad_data>/habitat_mapmad/<home>_<k>/`, layout in `src/mapmad_sim/datagen.py`):
- `0.jpg ... N-1.jpg` — 320 x 240, JPG q90, one per 0.25 s control step;
- `traj_data.pkl` — `position` (N, 2) m and `yaw` (N,) rad in NoMaD's frame: X = -z, Y = -x of Habitat, yaw unchanged
  (`src/mapmad_sim/frames.py`); `data_config.yaml` `habitat_mapmad.metric_waypoint_spacing` = 0.05;
- `mapmad_map.npz` — whole-floor `obstacle_first_seen` / `explored_first_seen` (int32, -1 = never) in 10 cm cells,
  `origin`, `resolution`, `floor_height`; read with `vint_train/mapmad/local_map.py`;
- `mapmad_frames.npz` — per frame: command after it (v, w), phase, collision, forced spot turn, target pixel share,
  Habitat pose, real floor height; the route followed and the shortest path;
- `mapmad_meta.json` — target (type, category, instance, box, end view point, end_fallback), start (heading,
  wall-recovery distance, visibility), variations, seed, home, floor, floor-map sha256, discarded attempts, stats.
Homes are written to `_tmp/<home>/` first and moved in when finished; `_homes/<home>.json` marks a home done (with
its floor-check counts and discards); `_runs/` keeps config + seed + git commit of every worker.

## Things to know

- **HM3D dataset config:** use the per-split `hm3d_annotated_<split>_basis.scene_dataset_config.json` next to the
  homes (`config.hm3d_scene_dataset_config`). The episodes' own `scene_dataset_config` path and the generic
  `hm3d_annotated_basis` config don't match the shared folder layout.
- **Object boxes need the colour camera:** without a colour sensor habitat-sim 0.3.3 loads no textures and every HM3D
  object box has size 0.
- **habitat-sim 0.3.3 object boxes are wrong for HM3D:** `obj.aabb` is the union of the object's box with the world
  origin (00800: 558 of 661 boxes > 4 m). Use `obj.obb` (`mapmad_sim.objects.object_box`): its centre equals the
  ObjectNav v2 object positions. Phase 0's "goal inside its box" check passed only because the boxes were huge.
- **The navmesh floats above the floor** (0.05-0.18 m with HM3D's baked navmesh, varies): the robot measures the
  real floor under itself at every step (a 4x4 downward depth camera) and sets the camera 0.18 m above it.
  Raycasts would need the Bullet build of habitat-sim.
- **HM3D's baked navmesh is for a 1.5 m agent:** the space under tables is "blocked", so a 25 cm robot hits
  invisible walls where its camera sees open floor. `LimoSim` rebuilds the navmesh per home at load (0.1 s) with
  LIMO's size (`robot_limo.yaml` `navmesh`) and logs its sha256.
- **HM3D minival homes are ObjectNav val (exam) homes:** all of 00800, 00802, 00803, 00808 are among the 36. Never
  tune or test anything there.
- **Sim camera = LIMO camera:** 0.18 m above the real floor, centred pinhole, HFOV 66.5°, pitch −2.0°. The pitch
  stands in for LIMO's off-centre principal point (cy = 213 of 480) plus ~1° of real upward tilt, so both
  horizons sit on row ~223 of 480 (`horizon_check.py`).

## Run the Phase 5 exam

Config: `configs/p5_exam.yaml` (sets, set rules, run settings, map settings, the six arms A-F, paired comparisons);
`configs/p5_dev.yaml` = the same with the dev set and the waypoint-test arms `C_wp1` / `C_wp2`. Rules that judge
results are frozen with their sha256 before the runs: `/outputs/mapmad/p5_exam/p5_dev_rules.md`, `p5_rules.md`.
New code: `src/mapmad_sim/exam_set.py` (set rules + draw), `exam_arms.py` (the arms' rules, used by both containers),
`run_layout.py` (run ids `<episode>__s<k>` and every run file), `online_map.py` (exam-time map = the Phase 2 builder +
the training crop/heat code), `scene_labels.py` (floor / object label lookups), `sidecar_server.py` (protocol v3:
`outputs` map / grid, `vp_geodesic_m`); NoMaD side `train/vint_train/mapmad/closed_loop/` (`mapmad_policy.py`,
`planner.py` + `planner_policy.py`, `exam_runner.py`, `run_exam.py`, `exam_analysis.py`, `exam_stats.py`,
`exam_tables.py`). From the worktree, on the host (`nvidia-smi` first; long runs in `screen`):

```bash
# 1. episode sets (Habitat container): candidates per home (resumable), then the seeded draw + freeze
for k in 0 1 2 3; do docker exec -d -w /app/visualnav-transformer naz_mapmad_habitat \
    python mapmad/scripts/make_p5_sets.py candidates --set exam --gpu $((k % 2)) --worker $k --workers 4; done
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/make_p5_sets.py freeze --set exam
#    (--set dev for the dev set; outputs: <set folder>/episodes.json + .sha256, build_report.md, goal_photos/)

# 2. tests (both containers)
docker exec -w /app/visualnav-transformer/mapmad naz_mapmad_habitat python -m pytest -q -p no:cacheprovider
docker exec -w /app/visualnav-transformer/train -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad \
    python -m pytest -q -p no:cacheprovider vint_train/mapmad/tests

# 3. runs: 2 simulator + runner pairs per GPU; resumable (finished runs are skipped)
screen -dmS mapmad_p5_runs bash -c 'PAIRS="0 0 1 1" mapmad/scripts/run_p5.sh \
    --config /app/visualnav-transformer/mapmad/configs/p5_exam.yaml --arms A B C D E F --subdir runs'

# 4. checks: map parity (arm C with --trace-maps on 2 episodes) and determinism (one C episode twice, cmp the logs)
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p5_map_parity.py \
    --config mapmad/configs/p5_dev.yaml --traces /outputs/mapmad/p5_exam/checks/parity/traces/C_wp2

# 5. tables (from the logs alone) and videos (logs replayed in Habitat)
docker exec -w /app/visualnav-transformer/train -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad \
    python -m vint_train.mapmad.closed_loop.exam_tables --config /app/visualnav-transformer/mapmad/configs/p5_exam.yaml \
    --subdir runs --out tables
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p5_videos.py \
    --config mapmad/configs/p5_exam.yaml --subdir runs --episode <id> --arms A B C D E F --out videos/<name>.mp4
# one driver per file, 1280 px (camera, map input with the logged waypoints, overview, STUCK label)
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p5_focused_video.py \
    --config mapmad/configs/p5_exam.yaml --episode <id> --arm C --out videos/focused/<id>_C.mp4
# presentation clips (1 or 2 drivers side by side, 3x speed, title card; --pick-bin, --cut-stuck-s)
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p5_meeting_clips.py \
    --config mapmad/configs/p5_exam.yaml --episode <id> --arms C E --title "..." --out videos/meeting/<name>.mp4
```

Unattended (everything after S-dev up to the G5 table, one command; Naz 2026-10-11): `scripts/run_p5_rest.sh` runs
the dev B/F runs with the 1a goal photos (`make_p5_sets.py rephoto`, `configs/p5_dev_1a.yaml`), the exam set, the
rules file (`scripts/p5_freeze.py rules`, template `configs/p5_rules_template.md`), the automatic S-freeze check
(`p5_freeze.py check`), the exam runs on 6 pairs, the determinism re-run, tables, figures, videos and the G5 table;
it stops with `STOP_reason.txt` on any failed condition or suspected bug, and a re-launch resumes. Test it first on
3 dev episodes with `configs/p5_dryrun.yaml` (see that file).

```bash
nvidia-smi && screen -dmS mapmad_p5_rest bash mapmad/scripts/run_p5_rest.sh   # log: p5_exam/driver/run_p5_rest.log
```

Outputs in `/outputs/mapmad/p5_exam/`: `dev/`, `exam_set/`, `runs/` (`logs/<arm>/<episode>__s<k>.jsonl`,
`summaries/`, `timing/`, `progress.shard<k>.log`, `run_info.*.json`), `tables/`, `figures/`, `videos/`, `checks/`,
`phase5_log.md`.

## Phase 7: fair camera

Finds the camera where the **original NoMaD** drives best and fixes it for later phases (`configs/camera_fair.yaml`).
Rules: `/outputs/mapmad/p7_camera/p7_rules.md` (+ `.sha256`); config `configs/p7_sweep.yaml`; outputs in
`/outputs/mapmad/p7_camera/` (`audit/`, `lens/`, `height/`, `episodes/`, `sweep/`, `phase7_log.md`).

New code: `src/mapmad_sim/lens.py` (double-sphere fisheye + equidistant / OpenCV pinhole models: project, unproject),
`src/mapmad_sim/camera_setting.py` (a sweep camera: lens x height x pitch, applied to `RobotSpec`), `RobotSpec.fisheye`
(None = the pinhole, unchanged; else the colour and semantic cameras become that fisheye, black outside its image
circle), the optional `camera` at `reset` (server, client, protocol), `ExamArm.camera` / `episode_type` /
`context_spacing`, `run_exam.py --arm-major` and `--claims` (dynamic hand-out: with `HANDOUT=dynamic`, `run_p5.sh`
gives every pair the next unclaimed run instead of a fixed shard; the Phase 7 driver uses it); summaries and crash
records are written atomically (`run_layout.write_text_atomic`); `src/mapmad_sim/floor_height.py` (Method B, floor
motion; tests `tests/test_floor_height.py`).

```bash
# 7.0 audit: the official repo at a pinned commit vs our arms (read-only) -> audit/audit_table.md
git clone https://github.com/robodhruv/visualnav-transformer.git <scratch> && git -C <scratch> checkout dca79815b704
docker exec -w /app/visualnav-transformer/train -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad \
    python -m pytest -q -p no:cacheprovider vint_train/mapmad/tests/test_deployment_rules.py

# 7.2 SACSoN: download (resumable, sizes + SHA256SUMS), convert with NoMaD's own process_bags.py (rosbag shim)
SACSON_RAW=/mnt/shared_disk/nazli/nomad_data/sacson_raw mapmad/scripts/p7_download_sacson.sh
docker exec -w /app/visualnav-transformer/train naz_mapmad bash -c 'mkdir -p /tmp/in && for f in /data/sacson_raw/*/; do
    mkdir -p /tmp/in/$(basename $f) && ln -sf $f*.bag /tmp/in/$(basename $f)/; done &&
    python -m vint_train.mapmad.sacson_convert -d sacson -i /tmp/in -o /data/sacson -s 4.0'

# 7.1 lens: fitted to SACSoN's frames (no calibration file without sign-in) -> lens/sacson_lens.json; tests
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p7_lens.py fit
docker exec -w /app/visualnav-transformer/mapmad -e PYTHONPATH=/app/visualnav-transformer/mapmad/src:/app/visualnav-transformer/mapmad/tests \
    naz_mapmad_habitat python -m pytest -q -p no:cacheprovider tests/test_lens.py tests/test_camera_setting.py
# (tests/p7_render_reference.py recorded the pinhole reference hashes with mapmad @ e97b875)

# 7.4 height: Method B per dataset (+ synthetic check); SACSoN odometry scale from AR tags; Method A = height/method_a.md
for d in recon sacson go_stanford; do docker exec -e OMP_NUM_THREADS=1 -w /app/visualnav-transformer naz_mapmad_habitat \
    python mapmad/scripts/p7_height.py estimate --dataset $d --workers 10; done
docker exec -w /app/visualnav-transformer/train naz_mapmad python -m vint_train.mapmad.sacson_scale \
    --bags /data/sacson_raw --out /outputs/mapmad/p7_camera/height/sacson_odom_scale.json

# 7.3 S-lens sheet (real frames vs renders, also at 96 x 96; layout rows at 0.15-0.55 m)
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p7_lens.py sheet --gpu 1

# 7.5 episodes: 60 new in view, + Phase 1's 90 in exam format, freeze (drops step-0 successes), shares per camera
for s in new assemble freeze shares; do docker exec -w /app/visualnav-transformer naz_mapmad_habitat \
    python mapmad/scripts/make_p7_episodes.py $s --gpu 1; done

# 7.8 sweep, unattended (after S-sweep; code committed first): checks, determinism, 16 arms x 3 seeds on 6 pairs,
# tables + choice, report-only extra, video, camera_fair.yaml. Dry run: P7_CONFIG=p7_dryrun.yaml P7_ALLOW_DIRTY=1
nvidia-smi && screen -dmS mapmad_p7_sweep bash -c 'P7_COMMIT=<hash> bash mapmad/scripts/run_p7_sweep.sh'
# or wait for free GPUs (both -> 6 pairs, one -> 3 pairs), re-run the rendering tests there, launch, watch the session:
screen -dmS mapmad_p7_launch bash -c 'P7_COMMIT=<hash> bash mapmad/scripts/p7_wait_and_launch.sh'

# 7.9 tables / choice / video again from the logs
docker exec -w /app/visualnav-transformer/train -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad \
    python -m vint_train.mapmad.closed_loop.p7_tables --config /app/visualnav-transformer/mapmad/configs/p7_sweep.yaml
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p7_videos.py \
    --config mapmad/configs/p7_sweep.yaml --subdir sweep/runs --pick 7710
```

## Phase 8: search map v2, explorer teacher, drives

Map v2 (`train/vint_train/mapmad/map_v2.py`: frontier layer, sparse target fusion, near/far windows), the goal-blind
explorer teacher (`mapmad_sim/explorer.py`, one pure function `teacher_step`), the drive generator v2
(`mapmad_sim/datagen_v2.py`, `target_layer.py`) and the OVON val converter (`mapmad_sim/ovon.py`). Rules: frozen in
`configs/p8_rules.md` (+ `p8_rules_v2.md`, `p8_rules_v3.md`, each with its sha256 in /outputs/mapmad/p8_search_data/).
The full dataset `habitat_mapmad_v2` was made with option A = `configs/p8_datagen_v3a.yaml`.

```bash
# 8.0 checks: OVON splits (download + convert + loader), camera = camera_fair.yaml, map parity (2 dev drives)
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p8_ovon_check.py
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p8_camera_check.py --gpu 1
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p8_map_parity.py --gpu 1 \
    --config mapmad/configs/p8_datagen_v3a.yaml --out-name map_parity_v3a.json

# GPU free-watcher (20-min rule; the drivers read its gpu_free.json), once per session
screen -dmS mapmad_p8_gpuwatch bash mapmad/scripts/gpu_free_watch.sh /mnt/shared_disk/nazli/nomad_outputs/mapmad/p8_search_data/driver

# pilot (20 homes x 10 drives, ~5 min; CONFIG/TAG for another rule set) and its report (numbers, sheet, 2 videos)
CONFIG=mapmad/configs/p8_datagen_v3a.yaml TAG=rules3a W0=0 bash mapmad/scripts/p8_launch.sh pilot
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p8_pilot_report.py --gpu 1 \
    --config mapmad/configs/p8_datagen_v3a.yaml --out /outputs/mapmad/p8_search_data/pilot_rules3a

# full run (committed code only; 12 workers on GPU 1; W0=0 keeps GPU 0 out; a GPU that stops being free is withdrawn;
# relaunch = resume, unfinished homes are redone from their seeds)
W0=0 CONFIG=mapmad/configs/p8_datagen_v3a.yaml bash mapmad/scripts/p8_launch.sh full

# after it: splits (+ tiny), collision list, stats, determinism (5 drives regenerated), format check (G8 row 4)
C=mapmad/configs/p8_datagen_v3a.yaml
docker exec -w /app/visualnav-transformer naz_mapmad_habitat bash -c "python mapmad/scripts/make_p2_splits.py traj-names --config $C \
    && python mapmad/scripts/make_p2_splits.py traj-names --config $C --tiny 4 && python mapmad/scripts/make_p2_splits.py collisions --config $C"
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p8_full_stats.py --config $C
docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p8_regen_check.py --gpu 1 --config $C
docker exec -w /app/visualnav-transformer/train -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad python -m \
    vint_train.mapmad.format_check --config config/mapmad_v2_format_check.yaml --split-dir /mapmad_data/splits/habitat_mapmad_v2
docker exec naz_mapmad bash -c 'R=/outputs/mapmad/p8_search_data/format_check/run && mkdir -p $R && cd $R && ln -sfn \
    /app/visualnav-transformer/train/config config && python /app/visualnav-transformer/train/train.py -c config/mapmad_v2_format_check.yaml'
```
- The teacher's frontier for a frame (option A) = `map_v2.teacher_frontier_at`: the plain frontier with the cells
  bumped before that frame (`mapmad_frames.npz` `bump_events`) counted as obstacles, as `explorer._Planner` does
  (Phase 10 finding: `map_v2.layers_at`'s frontier leaves the bumped cells out and differs on ~2.5% of frames).

## Phase 10: MapMaD v2 training (NoMaD side; rules /outputs/mapmad/p10_v2_train/p10_rules.md)

Training, checks and drivers are on the NoMaD side (`train/vint_train/mapmad/README.md`, `train/scripts/mapmad_p10/`).
Habitat side: `explorer.teacher_frontier` (the frontier the live teacher clusters at one step) and the parity check
`scripts/p10_frontier_parity.py` (live teacher = `map_v2.teacher_frontier_at` = the per-drive cache on the frames
`frontier_cache check` picked; chain step 1), tests `tests/test_p10_frontier.py`.
- `teacher_ms` (wall-clock ms per teacher step) is the only saved value that differs between two runs of one drive id.

## Phase 9: recognisers (photo test + plug-in)

A recogniser finds the named object in each picture and draws the map's "target seen here" layer; the network never
sees the name. Phase 9 compares 9 local models + simulator truth on still frames of the 10 labelled Phase 1 homes and
picks R1, R2 by the rule in `configs/p9_rules.md`. Code: `mapmad_sim/photo_labels.py` (counted / near-synonym labels),
`photo_frames.py` (frames, items, halves), `photo_scoring.py` (pixel + map metrics, threshold, choice rule),
`recogniser.py` (the plug-in: `Recogniser.detect(rgb, depth, category)`, truth, sidecar client, score mapping);
`mapmad_recog/` (models + sidecar server, its own container; each wrapper = forward pass + a pure output mapping,
tested without weights in `tests/test_p9_model_outputs.py`); `mapmad_bridge/detections.py` (the one detection
format: reply arrays and raw part files, used by the server, the photo test, the scorer and the plug-in).
Configs `p9_frames.yaml`, `p9_models.yaml`.
Outputs `/outputs/mapmad/p9_recognisers/` (`STATUS.md`, `frames/`, `photo_test/`, `ovon_test/`, `plugin/`, `sheets/`).

```bash
# containers: the Phase 9 worktree's own Habitat container + the recogniser sidecar (image from Dockerfile.recog,
# packages from docker/requirements-recog.txt; weights in /mnt/shared_disk/nazli/model_cache = /model_cache)
HABITAT_CONTAINER=naz_mapmad_habitat_p9 bash mapmad/docker/run_containers.sh habitat
bash mapmad/scripts/p9_job.sh recog_build /mnt/shared_disk/nazli/nomad_outputs/mapmad/p9_recognisers/jobs/recog_build \
    bash mapmad/docker/run_containers.sh recog-build
bash mapmad/docker/run_containers.sh recog
# downloads (CPU; SAM 3 is gated: ~/.cache/huggingface/token, mounted read-only, never printed) -> photo_test/weights.json
bash mapmad/scripts/p9_job.sh download /mnt/shared_disk/nazli/nomad_outputs/mapmad/p9_recognisers/jobs/download \
    docker exec -e HOME=/model_cache/home -e YOLO_OFFLINE=False -e CUDA_VISIBLE_DEVICES= naz_mapmad_recog \
    python mapmad/scripts/p9_download.py
# GPU watcher (Phase 10's processes count as busy; Phase 9 priority rule in p9_gpu_watch.usable_gpus)
D=/mnt/shared_disk/nazli/nomad_outputs/mapmad/p9_recognisers/driver
bash mapmad/scripts/p9_job.sh gpuwatch $D python3 mapmad/scripts/p9_gpu_watch.py $D --ours naz_mapmad_habitat_p9 naz_mapmad_recog
# the whole GPU part, unattended and resumable (committed code only): frames -> rules freeze -> 9 models -> scoring
# + choice -> plug-in (truth parity, R1, R2). The key only lives in the environment.
MAPMAD_SIDECAR_KEY=$(openssl rand -hex 16) bash mapmad/scripts/p9_job.sh driver $D python3 mapmad/scripts/p9_driver.py
# single steps (in naz_mapmad_habitat_p9, -w /app/visualnav-transformer):
#   python mapmad/scripts/p9_build_frames.py candidates|select|render --gpu G
#   python mapmad/scripts/p9_photo_test.py raw|timing --model M --port P [--threshold T]   (server: python -m mapmad_recog.server)
#   python mapmad/scripts/p9_score.py tune|score|choose|sheets [--model M] [--fresh]
#   python mapmad/scripts/p9_plugin_run.py --recogniser truth|M --gpu G [--port P]
# tests (CPU): docker exec -w /app/visualnav-transformer/mapmad naz_mapmad_habitat_p9 python -m pytest tests/test_p9_*.py
#   (tests/test_p9_model_outputs.py, test_p9_text_reuse.py, test_p9_detections.py also in naz_mapmad_recog)
```
- Every long job runs in `screen` through `p9_job.sh` (EXIT trap: `STOP_reason.txt` on any abnormal exit, `DONE.txt`
  on a normal one). Claude Code itself may run in VS Code; it can be closed once the driver runs.
- Unique ids: frames `p9f-<home>-<rnd|drv>-<n>`, items `p9q-<frame>-<category>` / `p9o-<frame>-<OVON name>`.

## Phase 11: closed-loop search harness, dev set, DAgger labels (rules /outputs/mapmad/p11_dagger/p11_rules.md)

The v2 closed loop (also Phase 12's): the simulator server builds the training map online (`mapmad_sim/search_v2.py`:
`SearchMap` = Phase 8 builder + simulator-truth target layer with the Phase 8 noise + the teacher's frontier incl. bumped
cells -> near / far windows; `outputs={"search_v2": ...}` at reset, settings from `search_wire.search_settings`; `save`
writes a recorded run in the Phase 8 drive format); the policy runs in `naz_mapmad`
(`vint_train.mapmad.closed_loop.run_search`). Episode sets (`search_set.py`): the dev set (plan 8.7 builder on the 10
Phase 1 dev homes; Phase 12 reuses it) and the rollout set (Phase 8 search starts, seed 1101). DAgger labels
(`dagger_labels.py`): `teacher_step` on the robot's own map at every visited frame (blacklist counter frozen), then a
40-step kinematic look-ahead -> 8 waypoints. Config `configs/p11_dagger.yaml` (DRY: `p11_dryrun.yaml`).

```bash
# sets (GPU, minutes): dev candidates per worker + freeze; rollout starts per worker + freeze
python mapmad/scripts/make_p11_sets.py dev-candidates --gpu 0 --worker 0 --workers 2 && python mapmad/scripts/make_p11_sets.py dev-freeze
python mapmad/scripts/make_p11_sets.py rollout-candidates --gpu 1 --worker 0 --workers 6 && python mapmad/scripts/make_p11_sets.py rollout-freeze
# online map = generator map (2 dev drives; pose replay pass/fail, action replay report only) -> checks/map_parity.json
python mapmad/scripts/p11_map_parity.py --gpu 1
# closed-loop pairs on one GPU (the chain does this), e.g. 3 pairs on GPU 1 for the dev set
GPU=1 TOKEN=x mapmad/scripts/run_p11_pairs.sh --config /app/visualnav-transformer/mapmad/configs/p11_dagger.yaml \
    --set dev --arms run1 runC --subdir step0
# labels (CPU): label check on 50 Phase 8 test drives; labels of every rollout drive; splits; 10-state sheet
python mapmad/scripts/p11_labels.py check|label|splits|sheet [--workers N]
# tests (CPU; the replay test needs MAPMAD_TEST_GPU=<gpu>)
cd mapmad && python -m pytest -q -p no:cacheprovider tests/test_p11_dagger.py
```
The unattended chain and twin: `train/scripts/mapmad_p11/` (README there).

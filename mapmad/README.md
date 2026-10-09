# mapmad/ — MapMaD, Habitat side

Code that runs in the `naz_mapmad_habitat` container: the simulator setup (Phase 0), the virtual LIMO and the
simulator server for closed-loop runs (Phase 1), later the practice-drive generator (Phase 2) and the exam
(Phase 5). The NoMaD side lives in `train/vint_train/mapmad/` (closed-loop policy and runner: `closed_loop/`).

- `docker/` — the two containers (`Dockerfile.habitat`, `run_containers.sh`, `paths.env`); see `docker/README.md`.
- `configs/paths.yaml` — every path as seen inside the containers; override one with `MAPMAD_<NAME>`.
- `configs/robot_limo.yaml` — LIMO camera, depth, lidar, drive and compute facts (plan D9), measured 2026-10-09; incl. the sim camera pitch, the sim motion settings (`sim`) and the LIMO-sized floor map (`navmesh`; radius 0.195 m and height 0.55 m from the AgileX LIMO Cobot spec sheet).
- `configs/p1_baseline.yaml` — Phase 1 episodes, run settings, NoMaD settings, arms and paired comparisons.
- `configs/objectnav_categories.yaml` — HM3D object names -> ObjectNav categories, learnt from the train goals (`scripts/build_category_map.py`).
- `src/mapmad_sim/` — the Python package (`config.py`: paths, HM3D file names; `camera.py`: FOV, horizon row, floor-plane fit; `run_info.py`: config + seed + git commit per run; `robot.py`: the virtual LIMO; `objects.py`: object boxes; `objectnav.py`: the only reader of ObjectNav goal files; `episodes.py`: Phase 1 episodes; `categories.py`: name map; `run_layout.py`: run config + where every run input/output lives; `sidecar_server.py`: simulator server).
- `src/mapmad_bridge/` — used in both containers: wire format + client of the simulator bridge (numpy), `steplog.py` (reading step logs; stdlib only).
- `scripts/` — `bench_render.py` (G0 row 1 speed), `render_test.py` (G0 row 2 pictures), `objectnav_stats.py` (G0 row 3), `horizon_check.py` (G0 row 4: sim vs real horizon), `build_category_map.py`, `make_p1_episodes.py`, `run_p1.sh`, `p1_replay.py` (Phase 1).
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

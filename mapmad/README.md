# mapmad/ — MapMaD, Habitat side

Code that runs in the `naz_mapmad_habitat` container: the simulator setup (Phase 0), later the practice-drive
generator (Phase 2) and the exam robot (Phase 5). The NoMaD side lives in `train/vint_train/mapmad/`.

- `docker/` — the two containers (`Dockerfile.habitat`, `run_containers.sh`, `paths.env`); see `docker/README.md`.
- `configs/paths.yaml` — every path as seen inside the containers; override one with `MAPMAD_<NAME>`.
- `configs/robot_limo.yaml` — LIMO camera, depth, lidar, drive and compute facts (plan D9), measured 2026-10-09; incl. the sim camera pitch.
- `src/mapmad_sim/` — the Python package (`config.py`: paths, HM3D file names; `camera.py`: FOV, horizon row, floor-plane fit; `run_info.py`: config + seed + git commit per run).
- `scripts/` — `bench_render.py` (G0 row 1 speed), `render_test.py` (G0 row 2 pictures), `objectnav_stats.py` (G0 row 3), `horizon_check.py` (G0 row 4: sim vs real horizon).
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

## Things to know

- **HM3D dataset config:** use the per-split `hm3d_annotated_<split>_basis.scene_dataset_config.json` next to the
  homes (`config.hm3d_scene_dataset_config`). The episodes' own `scene_dataset_config` path and the generic
  `hm3d_annotated_basis` config don't match the shared folder layout.
- **Object boxes need the colour camera:** without a colour sensor habitat-sim 0.3.3 loads no textures and every HM3D
  object box has size 0.
- **The navmesh floats above the floor** (~0.16 m in 00800): heights must be measured from the real floor.
- **Sim camera = LIMO camera:** 0.18 m above the real floor, centred pinhole, HFOV 66.5°, pitch −2.0°. The pitch
  stands in for LIMO's off-centre principal point (cy = 213 of 480) plus ~1° of real upward tilt, so both
  horizons sit on row ~223 of 480 (`horizon_check.py`).

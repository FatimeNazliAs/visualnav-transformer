# mapmad/docker/ — MapMaD's two containers (plan §0.7)

| | `naz_mapmad` (NoMaD side) | `naz_mapmad_habitat` (Habitat side) |
|---|---|---|
| Image | `nomad:latest` (conda `vint_train`, Python 3.8) + `train/requirements-mapmad.txt` | `naz-mapmad-habitat:0.1` from `Dockerfile.habitat` |
| GPUs | `--gpus all` | `--gpus all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` |
| User | `--user $(id -u):$(id -g)`, `HOME=/tmp` | same |
| Network | `mapmad-net` | `mapmad-net` |

Mounts (host side from `paths.env`; **ro** = read-only):

| Container path | `naz_mapmad` | `naz_mapmad_habitat` |
|---|---|---|
| `/app/visualnav-transformer` | worktree | worktree |
| main repo `.git` (same path, **ro**) | yes | yes |
| `/data` | `nomad_data` | — |
| `/outputs` | `nomad_outputs` | `nomad_outputs` |
| `/mapmad_data` | `mapmad_data` | `mapmad_data` |
| `/tmp/.config/wandb` | `wandb_config` | — |
| `/hm3d` | — | HM3D v0.2 scenes (**ro**) |
| `/hm3d_semantics` | — | `hm3d_semantics_v0.2` (**ro**) |
| `/objectnav` | — | ObjectNav HM3D v2 episodes (**ro**) |
| `/opt/habitat-starter/data` | — | `~/projects/habitat-starter/data` (**ro**, toolbox tests) |

## Commands

```bash
mapmad/docker/run_containers.sh build    # docker build -f Dockerfile.habitat --build-context mapmad=<repo>/mapmad <habitat-starter>
mapmad/docker/run_containers.sh up       # docker network create mapmad-net (once) + both docker run
mapmad/docker/run_containers.sh check    # nvidia-smi -L / EGL driver / torch.cuda.device_count() + who is on mapmad-net
mapmad/docker/run_containers.sh down     # docker rm -f naz_mapmad_habitat naz_mapmad (nothing else)
```

`run_containers.sh` prints every docker command before running it. Override a host path for one call with an
environment variable, e.g. `HM3D_SCENES=/other/path mapmad/docker/run_containers.sh habitat`.
Don't edit `run_containers.sh` while it runs (bash reads scripts as it goes).

## Why the image looks like this

- **habitat-sim** is installed from the official conda package that `~/projects/habitat-starter.zip` ships in
  `vendor/.cache/`, checked against habitat-starter's pinned sha256 (no 195 MB download). Alternative: run
  `scripts/fetch_habitat_sim.py` without `--conda-file` (downloads the same file from anaconda.org).
- **NVIDIA EGL vendor file:** the container toolkit mounts `libEGL_nvidia.so.0`, but copies the glvnd file that
  registers it (`10_nvidia.json`) only from the host, and cukurovaai has none. Without it EGL only finds Mesa and
  habitat renders on the CPU (llvmpipe, ~9 steps/s instead of ~600). The Dockerfile writes that file.
- **`--user`:** works for both images; files on the shared disk are Naz's, not root's. Mount points under
  `/mnt/shared_disk/nazli/` are created by the script first so docker doesn't create them as root.
- **git inside the containers:** the worktree's `.git` points into `~/projects/nomad/.git`, mounted read-only at
  the same path, so every run can record its commit.
- **Extra NoMaD packages:** listed in `train/requirements-mapmad.txt`; `run_containers.sh nomad` installs them
  (`pip install --user`) when it creates the container. Never install by hand.
- **Sidecar (Phase 1):** the simulator server listens only on `mapmad-net`; never publish its port with `-p`
  (its messages are pickle).

## VLFM port check (Phase 13, report-only)

Runs the **original** VLFM code (bdaiinstitute/vlfm @ `584ed56`, MIT) with its own settings on 100 seeded HM3D v2
val ObjectNav episodes, to compare its success rate with the paper's HM3D number (63.6 % SR).

- Image `naz-mapmad-vlfm:0.1` from `Dockerfile.vlfm` (CUDA 11.3 devel + miniforge, Python 3.9, torch 1.12.1+cu113,
  habitat-sim 0.2.4 conda, habitat-lab/baselines 0.2.420230405, GroundingDINO @ eeba084 with its CUDA op,
  LAVIS 1.0.2). Every deviation from VLFM's README recipe is listed at the top of the Dockerfile.
- Container `naz_mapmad_vlfm`: `run_containers.sh vlfm-build | vlfm | vlfm-down`. Mounts HM3D (`/hm3d`) and the
  ObjectNav v2 episodes (`/objectnav`) read-only, the scene root (`$VLFM_SCENE_ROOT` -> `/scene_root`, the HM3D
  v0.2 layout VLFM's episodes expect), `$MODEL_CACHE` (`/model_cache`; detector weights in `vlfm/`, BLIP-2 and
  BERT downloads land in `vlfm/torch` and `vlfm/hf`) and `$NOMAD_OUTPUTS` (`/outputs`).
- Launcher (unattended, resumable): `train/scripts/mapmad_p13/vlfm_port.sh <gpu> <n|all> <out_dir>`. It starts the
  container if needed, starts VLFM's 4 model servers on per-GPU ports (12181+100*gpu ... +3), runs `vlfm.run` on
  the not-yet-finished episodes, writes `results.jsonl` + `summary.json`, and stops only its own processes.
- Episode list: `$NOMAD_OUTPUTS/mapmad/p13_claim/vlfm_port/episodes_100.json` (+ `.sha256`, checked at start).

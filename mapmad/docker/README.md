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

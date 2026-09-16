# sim_eval — closed-loop simulation evaluation

Open-loop evaluation replays recorded frames: the world never reacts. This
directory is the other thing — a simulator the model actually drives, so an
action changes the next view. Plan: `.claude/plans/nomad-sim-evaluation.md`.

**P0 (this phase) only stands the simulator up.** No model, no metrics.

## Layout

| Path | What it is |
|------|------------|
| `sim.Dockerfile` | `nomad_sim:latest` — `nomad:latest` + iGibson 2.2.2, headless EGL |
| `lib.sh` | shared names/paths and the one way into the container |
| `sim_up.sh` | build image · create screen session · create/start container |
| `download_assets.sh` | fetch LoCoBot + Rs scene onto the shared disk |
| `gpu.py` | picks the GPU, pins it, and verifies the pin took |
| `configs/locobot_rs_static.yaml` | LoCoBot in the static Rs house |
| `p0_1_smoke_test.py` | drive forward, render, assert the frame is not blank |
| `run_p0_smoke_test.sh` | run the above in the container, pinned to one GPU |
| `tests/` | GPU-free unit tests (`python -m pytest sim_eval/tests`) |
| `outputs/` | all generated files (gitignored, numbered `pN_M_*`) |

## Running it

Everything runs **from the host, from the repo root**. The scripts `docker exec`
themselves — you never need to enter the container.

1. Check the GPUs are free. This is a shared machine.
   ```
   nvidia-smi
   ```
2. Bring up the container (first run builds the image, ~10 min).
   ```
   ./sim_eval/sim_up.sh
   ```
3. Download the assets (~1 GB, once; skipped if already present).
   ```
   ./sim_eval/download_assets.sh
   ```
4. Run the smoke test.
   ```
   ./sim_eval/run_p0_smoke_test.sh
   ```
   It prints how far the robot moved and the frame's pixel statistics, then
   writes `sim_eval/outputs/p0_1_camera.png`. It exits non-zero if the frame
   is blank.

`SIM_GPU=0 ./sim_eval/run_p0_smoke_test.sh` picks the other GPU. Default is 1,
because GPU 0 also drives the machine's X server.

Running the Python directly is fine too, but it will **refuse to start** unless
you say which GPU — `SIM_GPU=1 python sim_eval/p0_1_smoke_test.py`, or
`--gpu 1`. See "Picking a GPU" below for why it will not guess.

## Names this workstream owns

Container `naz_nomad_sim` · screen `nomad_sim` · image `nomad_sim:latest` ·
assets `/mnt/shared_disk/nazli/igibson_data`. Nothing else on the box is ours —
in particular **never** touch `nomad_debug_visuals` or another user's container.

## Under the hood

- **EGL, not X.** Rendering goes straight to the GPU with no display server, so
  it works over SSH. Two things make that work and both are easy to lose:
  the container must run with `NVIDIA_DRIVER_CAPABILITIES=all` (the default
  `compute,utility` hides the driver's GL libraries), and GLVND needs
  `10_nvidia.json` or it silently falls back to Mesa software rendering.
- **Picking a GPU is enforced, not assumed.** `CUDA_VISIBLE_DEVICES` restricts
  torch; `GIBSON_DEVICE_ID` picks the GPU EGL renders on. They are not
  interchangeable — iGibson enumerates render devices through the driver, which
  ignores `CUDA_VISIBLE_DEVICES`. Setting only that one is how P0's first run
  asked for GPU 1 and rendered on GPU 0 without saying so. `gpu.py` now sets
  both, verifies afterwards that the renderer actually landed on the right GPU,
  and refuses to guess when told nothing. On a shared machine a wrong pin costs
  somebody else their work, and it is invisible: the frame looks perfect.
- **One conda env.** iGibson is installed into `vint_train` alongside torch, not
  in an env of its own, because from P1 the bridge runs the simulator and the
  policy in a single process.
- **Assets are not in the image.** `GIBSON_ASSETS_PATH` and friends point at the
  `/igibson_data` mount.

## Common errors

| Symptom | Cause |
|---------|-------|
| `container 'naz_nomad_sim' is not running` | run `./sim_eval/sim_up.sh` |
| `FAILED: No GPU was specified` | running the Python directly — set `SIM_GPU` or pass `--gpu` |
| `FAILED: GPU pin failed: asked for GPU N, renderer is on GPU M` | working as intended; do not override, find out why |
| `Failed to initialize EGL` / falls back to CPU | container started without `NVIDIA_DRIVER_CAPABILITIES=all` |
| `assets_path does not exist` | `./sim_eval/download_assets.sh` not run yet |
| `fatal error: GL/gl.h` when building | GL dev headers missing — see LAYER 1 of `sim.Dockerfile` |

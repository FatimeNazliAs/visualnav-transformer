# sim_eval — closed-loop simulation evaluation

Open-loop evaluation replays recorded frames: the world never reacts. This
directory is the other thing — a simulator the model actually drives, so an
action changes the next view. Plan: `.claude/plans/nomad-sim-evaluation.md`.

**P0** stood the simulator up. **P1** wired NoMaD's brain to the sim's body —
one checkpoint driving itself along a hand-made trail. **P2 (this phase) makes
the trails.** The scene's own shortest path replaces the human teleop of
`create_topomap.sh`: plan a route, drive it, keep every Nth frame. Still no
metrics, no episode sampling and no video overlay: those are P3 and P4.

## Layout

| Path | What it is |
|------|------------|
| `sim.Dockerfile` | `nomad_sim:latest` — `nomad:latest` + iGibson 2.2.2, headless EGL |
| `lib.sh` | shared names/paths and the one way into the container |
| `sim_up.sh` | build image · create screen session · create/start container |
| `download_assets.sh` | fetch LoCoBot + Rs scene onto the shared disk |
| `gpu.py` | picks the GPU, pins it, and verifies the pin took |
| `configs/locobot_rs_static.yaml` | LoCoBot in the static Rs house (P0) |
| `configs/locobot_rs_bridge.yaml` | the same world, driven in SI units at 4 Hz (P1) |
| `configs/checkpoints.yaml` | which checkpoints are evaluated, and where they are |
| `p0_1_smoke_test.py` | drive forward, render, assert the frame is not blank |
| `run_p0_smoke_test.sh` | run the above in the container, pinned to one GPU |
| `checkpoints.py` | resolve a checkpoint name to weights + its training config |
| `nomad_policy.py` | the brain: `navigate.py`'s policy step, ported |
| `pd_control.py` | the steering: `pd_controller.py`, ported |
| `bridge.py` | the body ends, and the control tick that wires them together |
| `p1_0_make_topomap.py` | drive a fixed route open-loop, keep frames as a trail |
| `p1_1_drive_test.py` | one checkpoint follows that trail; frames + GIF out |
| `run_p1_drive_test.sh` | run the above two in the container, pinned to one GPU |
| `configs/topomap.yaml` | every knob the trail builder has (P2) |
| `path_follow.py` | the hand that drives the planned path — pure pursuit |
| `topomap_builder.py` | plan a route, drive it, keep every Nth frame (P2) |
| `p2_1_build_test.py` | build one trail; path plot + thumbnails + format check |
| `run_p2_build_test.sh` | run the above in the container, pinned to one GPU |
| `run_tests.sh` | the GPU-free unit tests, in the container |
| `tests/` | GPU-free unit tests (`./sim_eval/run_tests.sh`) |
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

5. Run the P1 drive test. The first run needs `--make-topomap` to build the
   trail; after that the trail is reused.
   ```
   ./sim_eval/run_p1_drive_test.sh --make-topomap
   ```
   It prints one line per control tick (which topomap node the model thinks it
   is at, the waypoint it chose, the `(v, w)` it commanded), then a summary,
   then writes every frame plus `drive.gif` to
   `sim_eval/outputs/p1_1_drive_test/`.

   `--checkpoint clean_stock` runs the other headline arm instead — at its own
   96x96, automatically.

6. Build a reference-path topomap.
   ```
   ./sim_eval/run_p2_build_test.sh
   ```
   It prints one line per node as the trail is captured, then a summary, then
   writes the trail to `sim_eval/outputs/p2_1_topomap/` (`0.png`, `1.png`, ...
   plus `metadata.json` and the `world.yaml` it was driven in), a top-down
   `p2_1_path.png` and a `p2_1_thumbnails.png` strip. It ends by loading the
   result back through P1's `bridge.load_topomap` — a trail the bridge cannot
   read is not a topomap.

   Which house, where the trail runs and how far apart its nodes are all live
   in `sim_eval/configs/topomap.yaml`. Nothing about the trail is hardcoded, so
   more houses and more trails (plan §6, §7) are edits to that file.

`SIM_GPU=0 ./sim_eval/run_p0_smoke_test.sh` picks the other GPU. Default is 1,
because GPU 0 also drives the machine's X server.

Running the Python directly is fine too, but it will **refuse to start** unless
you say which GPU — `SIM_GPU=1 python sim_eval/p0_1_smoke_test.py`, or
`--gpu 1`. See "Picking a GPU" below for why it will not guess.

## Names this workstream owns

Container `naz_nomad_sim` · screen `nomad_sim` · image `nomad_sim:latest` ·
assets `/mnt/shared_disk/nazli/igibson_data`. Nothing else on the box is ours —
in particular **never** touch `nomad_debug_visuals` or another user's container.

## How one control tick works (P1)

`bridge.py` is `deployment/src/navigate.py` + `pd_controller.py` with only the
two ends swapped: the ROS camera topic becomes iGibson's rendered frame, and
the `/cmd_vel` publisher becomes iGibson's differential-drive controller
(plan §4, decision E). In order:

1. grab RGB from the sim
2. resize to **the checkpoint's own** `image_size` — 160x120 for
   `best_combined`, 96x96 for `clean_stock`, read from each run's training log
3. push onto the rolling `context_size + 1` frame queue
4. goal mask `0` — goal-directed navigation, not exploration
5. the distance head localizes to the closest node in a +-4-node window
6. the next node becomes the subgoal
7. the action head draws 8 diffusion samples; take sample 0, waypoint #2
8. un-normalize into metres (`* max_v / frame_rate`)
9. PD controller: `v = dx/DT`, `w = atan(dy/dx)/DT`, clipped to 0.2 / 0.4
10. step the sim one 4 Hz period, and repeat

Nothing in that list is tuned for the sim. The numbers come from
`deployment/config/robot.yaml` and `navigate.py`'s own argparse defaults, and
`tests/test_pd_control.py` pins them so they cannot drift.

Steps 5 and 6 — picking the current node and the subgoal out of the distance
head's scores — live in `nomad_policy.localization_window` and
`nomad_policy.localize` rather than inline in `act()`, because upstream shipped
an off-by-one there once (commit `7b5b24c`: a window-relative index used as an
absolute one). That bug does not crash; it steers confidently at the wrong node
and looks fine in a replay. Keeping the arithmetic in pure functions lets
`tests/test_localization.py` pin it without torch, a checkpoint or a GPU. The
numbers are unchanged from `navigate.py` — verified by re-running the drive test
and diffing all 39 ticks.

## How a trail gets made (P2)

`create_topomap.sh` builds a real topomap by having a person joystick the robot
down the route while frames are saved. There is nobody to joystick a simulator,
so the scene's own traversability graph says where the perfect route goes and
`path_follow.py` drives it (plan §5). In order:

1. sample a start/goal pair, or take the one in the config
2. reject it unless both ends sit on the nav mesh and the geodesic path between
   them is 3-8 m — neither trivial nor longer than the house affords
3. plan it: iGibson's A* over the traversability graph, `entire_path=True`
4. face the robot down the path and drive it, pure pursuit at the robot's own
   `max_v` / `max_w` / 4 Hz — the same envelope the policy gets
5. keep the camera frame every `spacing_ticks` ticks — the "edge length"
6. **accept the trail only if the drive arrived, cleanly and without wedging**;
   otherwise throw the pair back and sample another (see below)
7. write `0.png, 1.png, ...` plus `metadata.json` and the `world.yaml` it ran in

The frames are saved at the render resolution, not at any checkpoint's
`image_size`, so one trail serves every arm — each resizes it to its own
training resolution, which the fairness protocol (plan §7) requires.

`metadata.json` is what P3 scores against: the goal pose success is measured
from, the geodesic length SPL divides by, the spacing, the camera intrinsics,
every node's pose and both paths (planned and driven).

## Under the hood

- **iGibson's LoCoBot turns the wrong way, and the bridge corrects it.**
  `Locobot.base_control_idx` is `[1, 0]` for `[left, right]`, but the URDF
  lists `wheel_left_joint` first — so the differential drive controller's
  left-wheel velocity is applied to the physical right wheel. Magnitude is
  right, sign is not: commanding `w = +0.4 rad/s` measures as `-0.39 rad/s` of
  yaw. NoMaD inherits the ROS convention where `+w` turns left, and so does
  every turn in its training data, so passing `w` straight through would mirror
  every turn and make a trail impossible to follow.
  `SimBody.ANGULAR_VELOCITY_SIGN` flips it. This is a wiring correction on the
  body side, not a tuned gain — the policy's `(v, w)` is untouched.
- **The context queue is seeded, not filled.** navigate.py waits for the real
  camera's stream to accumulate `context_size + 1` frames. There is no stream
  to wait for at t=0, so the episode starts with frame 0 repeated.
- **`image_size` comes out of the training log, not a config file.**
  `train.py` writes weights into the run folder but no config beside them; it
  merges `config/defaults.yaml` with the run's yaml, `print`s the result and
  moves on, and the capstone run yamls are gone. The fairness protocol
  (plan §7) needs each arm at its *own* resolution, so `checkpoints.py` parses
  that printed dict. If it cannot find it, it raises rather than defaulting.
- **`initial_pos_z_offset` is tied to the control rate.** iGibson asserts the
  spawn offset exceeds how far gravity pulls the robot in one action timestep —
  0.31 m at 4 Hz against 0.05 m at P0's 10 Hz. Hence 0.4 in the bridge config
  where P0 had 0.1.
- **SI commands, not normalized ones.** `action_normalize: false` *and*
  `command_input_limits: null` are both needed: the first only overrides the
  controller's input limits when true, and the controller's own default input
  range is `[-1, 1]`. With both set, iGibson takes `(v, w)` in m/s and rad/s
  exactly as the PD controller emits them.
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
- **Rs's traversability map is optimistic, so a trail is accepted on the drive,
  not on the plan.** Gibson's `floor_trav_0.png` is derived from a floor plan
  and does not know about all the furniture in the mesh: A* happily routes
  through a patch of Rs living room that the robot physically climbs onto and
  wedges in. There is no map to fix — so `topomap_builder` drives every
  candidate route and keeps only the pairs that arrive, collision-free and
  without getting stuck. About one planned path in three survives that in Rs,
  which is why `drive_attempts` exists and why it is not 1. If P3 wants a
  higher yield, the lever is the interactive iGibson houses, whose trav maps
  come from actual object placement (plan §12, still open).
- **Turning counts as progress, or the stuck detector eats the house.** A
  differential drive spins on the spot to line up with its path, and a start
  pose facing the wrong way needs up to `pi / max_w` = 7.9 s of pure rotation —
  longer than the 5 s stuck window. Measured on translation alone, that healthy
  turn is indistinguishable from a wedged robot, and the first version of this
  rejected most of Rs for exactly that reason. `stuck_progress_rad` is the fix.
- **`trav_map_type` does nothing for static Gibson scenes.** `env_base.load`
  passes it to `InteractiveIndoorScene` but not to `StaticIndoorScene`, which
  keeps its own `with_obj` default. The `no_obj` in the world configs is
  therefore inert — worth knowing before anyone "fixes" a planning problem by
  changing it. Rs ships only `floor_trav_0.png` anyway.
- **The driven path is usually shorter than the geodesic, and that is correct.**
  A* runs on an 8-connected grid, so its path zigzags between cell centres;
  pure pursuit smooths that out. SPL divides by the geodesic (the planner's
  number), never by what the reference drive happened to travel.

## Common errors

| Symptom | Cause |
|---------|-------|
| `container 'naz_nomad_sim' is not running` | run `./sim_eval/sim_up.sh` |
| `FAILED: No GPU was specified` | running the Python directly — set `SIM_GPU` or pass `--gpu` |
| `FAILED: GPU pin failed: asked for GPU N, renderer is on GPU M` | working as intended; do not override, find out why |
| `Failed to initialize EGL` / falls back to CPU | container started without `NVIDIA_DRIVER_CAPABILITIES=all` |
| `assets_path does not exist` | `./sim_eval/download_assets.sh` not run yet |
| `fatal error: GL/gl.h` when building | GL dev headers missing — see LAYER 1 of `sim.Dockerfile` |
| `FAILED: weights for ... not found` | running outside the container, where `/outputs` is not mounted |
| `FAILED: no line starting {'project_name'` | that run's training log is missing or truncated, so its `image_size` is unknown |
| `FAILED: ... has no route.json` | run `./sim_eval/run_p1_drive_test.sh --make-topomap` first |
| `initial_pos_z_offset is too small` | a config running at 4 Hz still has P0's 0.1 |
| the robot turns away from every subgoal | `ANGULAR_VELOCITY_SIGN` was "cleaned up" — see Under the hood |
| `FAILED: no drivable reference path in ... after N attempts` | the nav mesh keeps planning through furniture — raise `acceptance.drive_attempts`, or widen the geodesic bounds |
| `FAILED: no start/goal pair between X and Y m` | the geodesic bounds are wider than the house; Rs's longest path is ~7.7 m |
| `FAILED: configured start ... is not on the nav mesh` | a hand-picked start/goal in `topomap.yaml` is inside a wall |
| `FAILED: ... has no metadata.json` | that topomap predates P2 (the P1 trail has `route.json` instead) |

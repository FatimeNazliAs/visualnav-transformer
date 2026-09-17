# sim_eval — closed-loop simulation evaluation

Open-loop evaluation replays recorded frames: the world never reacts. This
directory is the other thing — a simulator the model actually drives, so an
action changes the next view. Plan: `.claude/plans/nomad-sim-evaluation.md`.

**P0** stood the simulator up. **P1** wired NoMaD's brain to the sim's body —
one checkpoint driving itself along a hand-made trail. **P2** made the trails:
the scene's own shortest path replaces the human teleop of `create_topomap.sh`.
**P3 (this phase) is the ruler.** A fixed, seeded set of those trails becomes a
task set every checkpoint faces; each task is run as an episode that ends on
success or timeout, and every episode lands as one row of a metrics table. No
video overlay yet — that is P4.

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
| `configs/eval.yaml` | what gets scored, on what, and when an episode ends (P3) |
| `metrics.py` | the five metrics of plan §6, and the table they go in |
| `episode_runner.py` | run one episode to success or timeout, and score it |
| `task_set.py` | the fixed, seeded task set every checkpoint shares |
| `run_eval.py` | score one checkpoint over the whole task set, unattended |
| `run_eval.sh` | run the above in the container, pinned to one GPU |
| `p3_1_score_test.py` | score a small task set, then re-derive every metric |
| `run_p3_score_test.sh` | run the above in the container, pinned to one GPU |
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

7. Score a checkpoint. The small version first — three tasks in one house,
   end to end, unattended:
   ```
   ./sim_eval/run_p3_score_test.sh
   ```
   It builds the task set (printing one line per trail), runs each task as an
   episode, prints the per-episode metrics, and then **re-derives every metric
   in the table from the table** — SPL from the path lengths beside it, success
   against the distance it stopped at, the tick budget against the timeout
   formula. It writes `sim_eval/outputs/p3_1_score_test.csv` plus a `.jsonl`
   of the same episodes with their pose traces.

   The full version scores every task in `configs/eval.yaml` for one arm:
   ```
   ./sim_eval/run_eval.sh --checkpoint best_combined
   ```
   It is long and unattended — run it inside `screen -r nomad_sim` so it
   survives the SSH connection dropping. `--build-only` builds the shared task
   set and stops. `--resume` continues a table that was interrupted.

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

## How an episode is scored (P3)

A **task** is one of P2's trails plus its metadata. A **task set** is a fixed,
seeded list of them, built once and shared by every checkpoint — that sharing
is the whole of the fairness protocol (plan §7), so the set is fingerprinted
and cannot be quietly rebuilt under a half-finished comparison. An **episode**
is one checkpoint attempting one task. In order:

1. seed numpy and torch with the *task's* seed, so every arm meets the same
   reset and the same diffusion noise
2. place the robot at the pose the reference drive started from
3. hand the trail to the bridge and tick it (P1's loop, unchanged)
4. after each tick, measure the distance to the goal pose
5. end on **success** — inside the goal radius — or on **timeout**, and on
   nothing else
6. count collisions every tick, and never stop for one
7. append one row to the CSV, and one JSON line holding the pose trace

**The rules, in numbers:**

| Rule | Value | Where it comes from |
|------|-------|---------------------|
| success radius | **1.0 m** from the goal pose, **geodesic** | plan §5 |
| timeout | `ceil(4 x geodesic_m / (max_v x dt) + (pi / max_w) / dt)` ticks | plan §6, "tied to reference-path length" |
| collisions | counted, never terminal | plan §6 |

The timeout's first term is how many ticks the shortest path would take at the
robot's top speed — the floor no agent can beat — and the 4x is the room it
gets to steer, overshoot and correct. The second is a flat allowance for
turning on the spot, which buys no distance at all: a differential drive that
starts off-heading spends up to `pi / max_w` = 7.9 s lining up before it moves.
Both scale with the robot's own limits, so changing the control rate cannot
silently change what a timeout means. For a 3.3 m trail that is 295 ticks, or
74 s of robot time.

**The five metrics** (plan §6), per episode:

| Metric | Formula |
|--------|---------|
| success | final distance to the goal pose <= 1.0 m |
| collision rate | distinct collisions per metre, and the fraction of ticks in contact |
| SPL | `success x geodesic / max(driven, geodesic)` |
| final distance-to-goal | geodesic, from where it stopped to the goal pose |
| steps / time | ticks, and ticks x 0.25 s |

`geodesic` is the task's own shortest-path length, measured by A* on the nav
mesh when the trail was built — never what the reference drive happened to
travel.

**Why the success radius is geodesic.** It is the same distance the table
reports as `final_distance_to_goal`, so success and the number beside it cannot
mean different things, and it is the standard reading in the point-goal
literature. It is also not a hypothetical difference: the first episode ever
scored here stopped **0.97 m** from its goal in a straight line and **1.08 m**
around the furniture — a success under one rule and a timeout under the other.
`episode.success_metric` in `configs/eval.yaml` switches it, and every row of
the table records which rule scored it.

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

- **The straight-line distance gates the geodesic one, and that is a proof, not
  a shortcut.** A path around the furniture is never shorter than the line
  through it, so an agent outside the radius in a straight line is outside it
  geodesically too — no A* needed. Only the handful of ticks already inside the
  radius pay for one. That matters because `scene.get_shortest_path` *mutates*
  the scene: an endpoint that is not on the traversability graph is grafted on
  as a new node. Asking every tick would add hundreds of nodes per episode. The
  grafted node is always a leaf, and no shortest path routes through a leaf, so
  the queries that do happen cannot shorten a later answer.
- **An agent with no geodesic to the goal has not arrived.** It has left the
  traversable component — in Rs, that means it climbed onto the furniture. The
  distance column is left blank rather than filled with a 0 that would read as
  "arrived".
- **SPL is clamped at 1, and the clamp earns its place here.** A* runs on an
  8-connected grid and zigzags between cell centres, so an agent that drives the
  same route smoothly covers *less* than the geodesic it is divided by. Without
  `max(P, L)` those episodes score above 1 and quietly inflate the mean.
- **The model claiming it has arrived is recorded, not obeyed.** The bridge
  stops when the distance head localizes onto the last node — `navigate.py`'s
  rule on the real robot. That is a claim about where the agent thinks it is,
  and the second episode ever scored here made it at tick 57 and then ended
  1.5 m from the goal. Ending on it would let a lost agent score, and would
  stop the odometer early and inflate its SPL, so it is logged as
  `declared_arrival_tick` and the episode runs on.
- **A collision never ends an episode, and one of them can fill it.** Plan §6
  is count-and-continue, so an agent that wedges against furniture spends its
  whole budget there: an early run logged 298 colliding ticks out of 341 as a
  *single* collision event. That is why the table carries both counts, and why
  the per-metre rate divides **events** by distance rather than contact ticks —
  the tick version reported "60 collisions per metre" for that episode, which
  is a number about being stuck, not about hitting things. How stuck it was is
  `contact_tick_fraction`.
- **The task set is immutable once anything has been scored against it.** It is
  fingerprinted from the config that built it, and a build against a changed
  config refuses instead of rebuilding. Half a table scored on one set of
  problems and half on another is not a comparison, and nothing in the CSV
  would say so. The episode rules are deliberately *not* in the fingerprint:
  they change how a task is scored, not which tasks exist.
- **Rows are appended as they finish, not written at the end.** A run is tens
  of slow episodes; a crash on the last one must not cost the rest, and a run
  in progress should be readable with `tail -f`. Hence also `python -u` in the
  wrapper scripts — `docker exec` hands Python a pipe, not a tty, so stdout is
  block-buffered otherwise and an unattended run looks hung for minutes at a
  time.

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
| `FAILED: the task set in ... was built from a different config` | working as intended — restore the config, or build the new set in a new directory and re-run *every* arm against it |
| `FAILED: ... has no manifest.json` | no task set there yet; `run_eval.py` builds one (`--build-only` to stop after) |
| `FAILED: ... pins a start/goal pair` | the topomap config has `start`/`goal` set, so every task would be the same trail |
| `FAILED: the metrics table does not hold up` | a metric disagrees with the numbers beside it in its own row — the message names which |
| `success_metric must be one of ('geodesic', 'euclidean')` | a typo in `episode.success_metric` |
| `RuntimeWarning: divide by zero` from `point_nav_fixed_task.py` | harmless — iGibson's own built-in task computes its own SPL at reset, with a zero path length. The bridge ignores that task entirely; the goal is the topomap. |

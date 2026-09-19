# sim_eval — closed-loop simulation evaluation

Open-loop evaluation replays recorded frames: the world never reacts. This
directory is the other thing — a simulator the model actually drives, so an
action changes the next view. Plan: `.claude/plans/nomad-sim-evaluation.md`.

**P0** stood the simulator up. **P1** wired NoMaD's brain to the sim's body —
one checkpoint driving itself along a hand-made trail. **P2** made the trails:
the scene's own shortest path replaces the human teleop of `create_topomap.sh`.
**P3 is the ruler.** A fixed, seeded set of those trails becomes a task set
every checkpoint faces; each task is run as an episode that ends on success or
timeout, and every episode lands as one row of a metrics table. **P4 makes a
row watchable**: the same episode, filmed as it is scored, as a
three-panel MP4 — camera, top-down map, subgoal and waypoints. It is a layer
over the scorer and changes nothing about it; switched off, an episode runs
exactly the code P3 ran. **P5 (this phase) uses all of it to ask why.** It is
diagnostic, not a feature: a small set of episodes is run, filmed and taken
apart, the frames the sim feeds the encoder are held up against the frames the
model was trained on, and each failure gets a name. Nothing about how the
policy steers is tuned — plan decision E, and plan §7's fairness protocol,
both depend on P6 facing the real robot's settings.

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
| `bridge.py` | the robot end of the simulator, and the control tick that wires the two together |
| `sim_scene.py` | the world end: floor height, geodesics, the traversability map |
| `driver.py` | the four numbers that steer, as a config block |
| `p1_0_make_topomap.py` | drive a fixed route open-loop, keep frames as a trail |
| `p1_1_drive_test.py` | one checkpoint follows that trail; frames + GIF out |
| `run_p1_drive_test.sh` | run the above two in the container, pinned to one GPU |
| `configs/topomap.yaml` | every knob the trail builder has (P2) |
| `path_follow.py` | the hand that drives the planned path — pure pursuit |
| `topomap_builder.py` | plan a route, drive it, keep every Nth frame (P2) |
| `p2_1_build_test.py` | build one trail; path plot + thumbnails + format check |
| `run_p2_build_test.sh` | run the above in the container, pinned to one GPU |
| `configs/eval.yaml` | what gets scored, when an episode ends (P3), what is filmed (P4) |
| `metrics.py` | the five metrics of plan §6, and the table they go in |
| `episode_runner.py` | run one episode to success or timeout, and score it |
| `task_set.py` | the fixed, seeded task set every checkpoint shares |
| `run_eval.py` | score one checkpoint over the whole task set, unattended |
| `run_eval.sh` | run the above in the container, pinned to one GPU |
| `p3_1_score_test.py` | score a small task set, then re-derive every metric |
| `run_p3_score_test.sh` | run the above in the container, pinned to one GPU |
| `recorder.py` | the three-panel video layer over the scorer (P4) |
| `p4_1_record_test.py` | film one episode, then prove filming changed no metric |
| `run_p4_record_test.sh` | run the above in the container, pinned to one GPU |
| `p5_1_input_check.py` | what the sim feeds the encoder, against what training did (P5) |
| `run_p5_1_input_check.sh` | run the above in the container, pinned to one GPU |
| `diagnosis.py` | why an episode failed — a name, from the evidence in its trace |
| `p5_2_diagnose.py` | run a few episodes, filmed, and name each failure |
| `run_p5_2_diagnose.sh` | run the above in the container, pinned to one GPU |
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

8. Watch an episode. One task, filmed, plus the same task unfilmed to prove
   the recorder moved no metric:
   ```
   ./sim_eval/run_p4_record_test.sh
   ```
   It writes `sim_eval/outputs/videos/best_combined/Rs_00.mp4` — one frame per
   control tick, three panels — and fails if the frame count and the tick count
   disagree or if any column of the metrics row differs between the two runs.

   To film a real scoring run, turn it on in `configs/eval.yaml` or from the
   command line:
   ```
   ./sim_eval/run_eval.sh --checkpoint best_combined --record --record-tasks Rs_00,Rs_07
   ```

9. Check the model is being fed what it was trained on. It should be the
   first thing run whenever the behaviour looks wrong:
   ```
   ./sim_eval/run_p5_1_input_check.sh
   ```
   It prints channel order, value range, aspect and crop, field of view and
   camera height for a sim frame and a GoStanford frame side by side, and
   writes `sim_eval/outputs/p5_1_input_check.png` — both sources at every stage
   of the transform, the same sim pose rendered through six fields of view,
   and a strip of real training frames to hold them against. It also embeds 60
   sim poses at every angle and 200 training frames with the checkpoint's own
   observation encoder, and prints which field of view the model thinks looks
   most like training (see "The FOV experiment").

10. Find out why the episodes that fail, fail:
   ```
   ./sim_eval/run_p5_2_diagnose.sh
   ```
   It runs three tasks in one scene, filmed, then reads the traces back and
   names each failure. Everything lands in
   `sim_eval/outputs/p5_2_diagnostics/`: the metrics table, the traces, one
   flat per-tick CSV per episode under `ticks/`, the videos under `videos/`,
   and `p5_2_diagnosis.csv`. `--tasks 5` runs more; `--checkpoint clean_stock`
   diagnoses the other arm.

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

An episode is a **stream**: you iterate it to run it, and each tick arrives as
it is decided, carrying the frame the model saw.

```python
episode = runner.episode(task, "best_combined")
for record in episode:
    print(record.summary())      # or encode a video frame, or both
result = episode.result()
```

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

## How an episode is filmed (P4)

Recording is a **layer over** the scorer, not a change to it. The episode is a
stream, so the consumer already holds the tick loop; filming is one line of it,
and an unrecorded episode gets a `NullRecording` whose `capture` does nothing:

```python
with film.episode(task, checkpoint, scene) as video:
    for record in episode:
        video.capture(record)
```

Each frame is one `TickRecord`, drawn as three panels, so the panels cannot
disagree with each other:

| Panel | What it shows |
|-------|---------------|
| camera | the frame the encoder was fed this tick, at render resolution |
| top-down | the house's traversability map, the reference trail, the path driven so far, the start, the goal and its success radius, every collision, and the node being steered at |
| overlay | the subgoal image, the distance head's reading for it, and all eight diffusion samples in the robot's frame with waypoint #2 — the one the PD controller acts on — marked |

The tick, `v` and `w` sit across the top, in red on a colliding tick.

Videos land in `outputs/videos/<checkpoint>/<task_id>.mp4` — one directory per
arm, so the same task under two checkpoints is two files to watch side by side.
The knobs are the `recording:` block of `configs/eval.yaml`:

| Knob | Values |
|------|--------|
| `enabled` | `false` (default) / `true`; `--record` and `--no-record` override it |
| `fps` | playback speed. The loop is 4 Hz, so **4 is real time** and the default 10 is 2.5x |
| `tasks` | `all`, a count (the first N of each scene), or a list of task ids |
| `directory` | where the per-checkpoint folders go |

**Filming costs about 0.09 s per tick** — one matplotlib frame drawn and
encoded — measured as 13.4 s against 8.0 s for the same 58-tick episode, video
at ~26 kB per frame. That is why it is off by default: worth paying to watch an
episode, not worth paying for twenty nobody will open.

## Why an episode failed (P5)

P0-P4 built a ruler and it works. What it measured first was one success in
three, and a success rate cannot say *why*: the same row — timed out, 1.5 m
short — is written by an agent wedged under a sofa, an agent spinning on the
spot, an agent that drove confidently past the goal, and an agent whose
distance head decided at tick 57 that it had already arrived. Four problems,
four different answers, one number.

**P5 is diagnostic. It tunes nothing.** Plan decision E mirrors the real
LoCoBot's settings and plan §7 requires every arm in P6 to face identical ones,
so the driver block, the control rate, the waypoint index and the localization
parameters are untouched here. What P5 adds is the ability to see.

### Is the model even being fed the right thing?

The first question, and the cheapest — `./sim_eval/run_p5_1_input_check.sh`.
A policy fed BGR, or fed a differently cropped frame than every frame in its
training set, fails in a way that is indistinguishable from "the sim looks
different from a real corridor". So one sim frame and one GoStanford frame go
through **the transform the policy itself uses** and come out side by side:

| Checked | Sim | GoStanford |
|---------|-----|------------|
| channel order | R 166 · G 151 · B 140 | R 127 · G 124 · B 118 |
| encoder range | [-2.12, +2.24] | [-2.12, +2.64] |
| native size | 640x480, aspect 1.333 | 160x120, aspect 1.333 |
| training's 4:3 centre crop | no-op | no-op |
| resize | to the checkpoint's own `image_size` | the same |
| lens | 45 deg vertical / 58 deg horizontal, rectilinear | no intrinsics in the dataset |
| camera height | 0.88 m above the floor | not recoverable |

The first five rows agree, which rules out the cheap explanations: the channel
order matches, the value range matches (training normalized with the same
ImageNet statistics — `train/train.py` line 68, over `[0, 1]` tensors from
`resize_and_aspect_crop`), and the 4:3 crop is a no-op on both because both
sources are already 4:3.

The lens row is the one that does not agree, and it cannot be settled with a
number because GoStanford ships frames and odometry and nothing about the rig.
So the figure settles it with pictures: the same sim pose rendered at 45, 70,
90 and 110 degrees vertical, printed directly above a strip of real training
frames. Read them against each other.

### Why did *this* episode end here?

`./sim_eval/run_p5_2_diagnose.sh` runs a small set — three tasks, one scene,
one checkpoint — with the recorder on, and writes a bundle to
`outputs/p5_2_diagnostics/`:

| File | What it is |
|------|------------|
| `<checkpoint>.csv` | the metrics table P3 would have written, unchanged |
| `<checkpoint>.jsonl` | the same episodes with their pose traces |
| `ticks/<task_id>.csv` | **one row per control tick** — pose, node, subgoal, both distance-head readings, the waypoint, `(v, w)`, contact |
| `videos/<checkpoint>/*.mp4` | P4's three-panel replay of each |
| `p5_2_diagnosis.csv` | one named failure mode per episode, with its evidence |

`diagnosis.py` names the mode from the trace. The modes are **ordered, and the
order is the argument**: they are not mutually exclusive (a wedged agent has
also failed to advance its trail, and has also driven a strange path), so the
first match wins and they are tried most-specific first, which makes the name
point at the earliest thing that went wrong rather than at its consequence.

| Mode | What it means |
|------|---------------|
| reached the goal | inside the success radius before the budget ran out |
| wedged | in contact for most of the episode and did not move in its last quarter |
| turning on the spot | most ticks below 0.02 m/s — the waypoint never lands in front |
| false arrival | the distance head localized onto the last node, and the agent finished somewhere else |
| trail not advanced | localized under a quarter of the way along the trail |
| wandered | drove at least twice the shortest path and finished short |
| ran out of ticks | made progress, nothing singular went wrong, the budget ran out |

Those thresholds are for **reading, not for scoring**. Nothing in the metrics
table depends on them, and moving one renames an episode without moving a
single number P6 will report.

### What the first diagnostic run found

`best_combined`, three tasks in Rs, `n8w2r4t3` (the deployment defaults):

| Task | Outcome | Mode |
|------|---------|------|
| `Rs_00` | success, 58 ticks, SPL 1.0 | reached the goal |
| `Rs_01` | timeout, 341 ticks, 1.51 m short | wedged — 87% of ticks in contact |
| `Rs_02` | timeout, 439 ticks, 1.73 m short | wedged — 87% of ticks in contact |

**NoMaD navigates.** That is the first thing the per-tick logs say and it is
easy to lose behind two timeouts. `Rs_00` drives 4.4 m to its goal cleanly, and
both failures track their reference trail to within **0.15-0.18 m** — about the
robot's own body radius — for their first 43 and 55 ticks, with the distance
head reading under 1.5 the whole time. Neither episode is lost when it fails.

**Both failures are one contact, never recovered from**, and the tick log makes
the chain exact (`ticks/Rs_01.csv`):

| Tick | What happened |
|------|---------------|
| 43 | first contact, 0.15 m off a reference path that was driven collision-free, head reading 0.21, subgoal advancing normally |
| 44 | still advancing — node 11, subgoal 12 |
| 45 | head reading jumps to **3.48**, past `close_threshold` 3, so the subgoal stops advancing |
| 46-47 | 9.58, then 12.54 — the head no longer recognizes anything |
| 48-340 | `v` pinned at 0.2 m/s into the obstacle for 296 of the next 297 ticks; the head reads past the threshold for 296 of them |

The view explains the collapse: once the robot is against the furniture, the
camera fills with a close-up of a table underside, which matches no node of the
trail, so every temporal distance is large and the subgoal freezes on the node
the agent is already at. It then drives at full speed into what it is stuck on
for 87% of its budget. `Rs_02` is the same sequence at tick 55.

So the interesting question is not "why did it get lost" — it did not — but
**why it clipped something 0.16 m off a clean path, and why one scrape is
terminal in this harness**. Two facts bear on the first, and they are the
reason `p5_1_input_check.py` exists:

- the sim camera sees **58 degrees horizontally**, rectilinear. GoStanford was
  shot through a lens wide enough to put both walls, the floor and the ceiling
  of a corridor in one frame. An obstacle the robot is about to clip with its
  shoulder is in that training frame and is off the edge of this one.
- the sim's eye sits **0.88 m** above the floor.

Neither is tuned here. Changing the camera an episode runs under is an edit to
`vertical_fov` in the world config and it changes what every arm sees, so it
belongs to a decision, not to a diagnostic. On the second — plan §6 is
count-and-continue by design, and static Gibson furniture cannot be pushed out
of the way the way a real chair can.

### The FOV experiment: bug or gap?

The first run pointed at the camera: the sim sees 58 degrees horizontally and
GoStanford was shot through a fisheye, so perhaps the robot clipped things at
its shoulder because it was never *shown* them — a model fed wrong, and a
correctness fix. The rule for reading the result was set before it ran:
**clips stop -> it was the feed; clips persist -> it is the domain gap.**

**Choosing the angle, on the input only.** `p5_1_input_check.py` renders the
same 60 sim poses at 45, 60, 75, 90, 105 and 120 degrees vertical, embeds them
with the checkpoint's own observation encoder (`NomadPolicy.embed_frames`), and
picks the angle whose frames sit nearest 200 GoStanford frames — the rule
fixed in code beforehand, and no episode run under any candidate. 120 won in
4 of 4 measurements (best_combined at seeds 0, 1, 2, and clean_stock); 45 was
worst or next to it in every one. 120 is also the edge of the sweep, which is
capped there because a rectilinear camera stretches its edges past use beyond
it — so it is the nearest a pinhole gets to a fisheye, not a match.

**Same tasks, same driver, one change.** The 120-degree task set was rebuilt
from the same seeds: start, goal, planned path and the reference drive are
byte-identical to the 45-degree set, the reference drives have 0 collisions in
both, and every row reads `n8w2r4t3`. Only the camera differs — which is also
the first live proof of the fingerprint fix: the 45-degree set refused to load
under the new camera, as it should.

| Task | 45 deg | 120 deg |
|------|--------|---------|
| `Rs_00` | success, 58 ticks, SPL 1.0 — no contact | **timeout**, 383 ticks, SPL 0 — first contact tick 40, **+62.7 deg** (left), 0.30 m off trail |
| `Rs_01` | timeout, 341 ticks — contact tick 43, **+82.0 deg**, 0.15 m off trail | timeout, 341 ticks — contact tick 44, **+86.5 deg**, 0.12 m off trail |
| `Rs_02` | timeout, 439 ticks — contact tick 55, **+64.5 deg**, 0.18 m off trail | timeout, 439 ticks — contact tick 44, **+60.5 deg**, 0.26 m off trail |

The bearing is where on the robot it was touched: 0 ahead, +90 its left
shoulder (`contact_bearing_deg` in the trace). The 45-degree camera sees
+-28.9 degrees either side; the 120-degree camera, +-66.5.

**Verdict: the domain gap, not the feed.**

- The side clips persisted, 3 of 3, every one at the left shoulder.
- Two of the three 120-degree contacts — +62.7 and +60.5 — were **inside** the
  wider view. The obstacle was in frame and the robot clipped it anyway, so
  "it could not see it" is not the explanation.
- The mechanism after contact is unchanged: the distance head reads past
  `close_threshold` on 100% of the ticks after it, and the robot is at full
  speed on 85-100% of them.
- It got worse, not better: the only success became a wedge, and `Rs_02` hit
  eleven ticks earlier. With three tasks that is weak evidence of harm; the
  persistence of the clips is the robust part.

So the bridge camera stays at 45 degrees, and plan §10's standing framing
applies: the result is read as a ranking of arms, not an absolute. Two more
things the experiment turned up:

- **The drift before contact is steady, not noisy, and has no fixed side.**
  In every failed episode the robot sat on one side of its trail for 32-46
  consecutive ticks and never crossed back; `Rs_00` drifted right at 45
  degrees (and arrived) and left at 120 (and clipped). No fixed side rules out
  a sign error in the steering; a held offset is what a policy does when it
  recognizes the place but not precisely where in it it is.
- **The camera is pitched 21 degrees down** (and centred: -0.001 m lateral,
  0.0 degrees yaw). GoStanford's horizon sits near mid-frame. That is a third
  optical difference beside the lens and the 0.88 m height, and like the
  height it has no training reference to set it from. Untouched.

### The pitch experiment: bug or gap?

The FOV experiment left one lead: before contact, the robot holds a steady
5-30 cm offset to one side of its trail. The LoCoBot's camera is pitched 20
degrees down by a *fixed* joint in its URDF (`head_tilt_joint`, ~21 at rest),
and GoStanford's horizon sits near mid-frame — so perhaps the model misjudges
where it is because it sees mostly floor. Same rule as before: **drift shrinks
-> a wrongly fed camera; drift unchanged -> the domain gap.**

**How the pitch is changed.** The joint is fixed, so it cannot be driven, and
the URDF is a shared iGibson asset that also carries the head's collision mesh.
`camera_tilt_deg` in the world config re-aims the camera on the *render* side
instead (`SimBody.observe` -> `_render_rgb`): iGibson's own robot-camera render,
line for line, with the view rotated about the camera's own optical centre. So
the camera height (0.88 m), the physics and the collision geometry are
untouched. At the URDF's own 20 degrees the new path matches iGibson's sensor
frame **pixel for pixel** (0 pixels differ); with the key absent, the sensor
path runs exactly as it always has. The key lives in the world config, so it
is saved beside every trail and the task-set fingerprint covers it.

**Choosing the pitch, on the input only.** `p5_1_input_check.py` now also
renders the same 60 poses at 0, 7, 14 and 20 degrees down (at the 45-degree
field of view) and applies the same pre-registered rule as the lens. Unlike the
lens, it found nothing: the rule picked **14, 14, 20, 20** across best_combined
at seeds 0-2 and clean_stock, the centroid gap disagreed every time, and the
four pitches sat within ~1% of each other against ~5% of seed noise. Level
(0) was never chosen. With no single answer, two pitches were committed to
*before either ran*: **0** (level — the stated target, and the strongest test)
and **14** (the rule's pick averaged over the four measurements). The drift
criterion was fixed at the same time: mean |off-trail| before first contact
lower in at least 2 of 3 tasks, *and* the longest one-sided run shorter in at
least 2 of 3.

**Same tasks, same driver, one change.** Both pitches' task sets were rebuilt
from the same seeds; start, goal, planned path and reference drive are
byte-identical to the baseline's, with 0 collisions; every row reads
`n8w2r4t3`; the field of view stays 45 and the height 0.88 m.

One seed per task, as every earlier run:

| Pitch | Task | Outcome | Ticks | SPL | mean \|off\| (m) | longest one-sided run | first contact |
|------:|------|---------|------:|----:|------:|-----:|------|
| 20 (baseline) | Rs_00 | success | 58 | 1.0 | 0.089 | 39 | — |
| 20 | Rs_01 | timeout | 341 | 0 | 0.089 | 38 | left, +82.0 |
| 20 | Rs_02 | timeout | 439 | 0 | 0.089 | 46 | left, +64.5 |
| 0 | Rs_00 | timeout | 383 | 0 | 0.084 | 33 | left, +66.2 |
| 0 | Rs_01 | success | 56 | 1.0 | 0.052 | 44 | — |
| 0 | Rs_02 | timeout | 439 | 0 | 0.070 | 25 | left, +43.7 |
| 14 | Rs_00 | timeout | 383 | 0 | 0.042 | 14 | left, +50.0 |
| 14 | Rs_01 | success | 56 | 1.0 | 0.040 | 44 | — |
| 14 | Rs_02 | success | 84 | 1.0 | 0.192 | 63 | — |

Read literally, 0 passes the criterion (|off| lower 3 of 3, run shorter 2 of
3) and 14 fails it. But the outcomes shuffle between configurations in a way no
camera explains — each task succeeds under a different pitch — so before
calling it, the same comparison was **replicated**: 20 and 0 again, over two
more diffusion seeds per task (`--seed-offset`; the row's `seed` column records
the one used). The pitch was not chosen from these; they measure whether its
effect beats run-to-run noise.

| Pitch | Episodes | Successes | Side clips | mean \|off\| (m), mean / range | longest one-sided run, mean / range |
|------:|---------:|----------:|-----------:|------|------|
| 20 | 9 | 2 | 7 | 0.106 / 0.033-0.344 | 29 / 4-46 |
| 0 | 9 | 2 | 7 | 0.078 / 0.051-0.121 | 36 / 25-52 |

**Verdict: the domain gap, not the feed.** Over three seeds per task the
criterion fails — |off| lower in 2 of 3 tasks, but the one-sided run shorter in
only 1 of 3, and *longer* overall — and successes and clips are identical. The
single-seed pass was inside the noise: at a fixed 20 degrees, the same task's
mean |off| ranges from 0.057 to 0.344 m with nothing but the diffusion seed
changed. So the camera stays at the URDF's 20 degrees, and plan §10 applies.
NoMaD does not recentre on its trail and has no obstacle avoidance beyond what
its images taught it: a held offset toward something it cannot recognize is a
clip, and a clip is terminal.

**What this means for P6: one seed per task is not enough to rank anything.**
With the camera, the task and the checkpoint all fixed, `Rs_00` went success,
timeout, success over three diffusion seeds. Every earlier comparison in this
phase — including "120 degrees made it worse" — rested on one seed per task and
is weaker than it looked. P6's 20 tasks x 1 seed per arm will carry the same
variance into its headline table; replicating each task over several seeds
(same seeds for every arm, so fairness holds) is the obvious remedy, and a
decision for the plan rather than for this phase.

## Talking to the simulator

Two adapters, split by what they are asked about. Nothing else in the package
touches iGibson:

| Ask | Where |
|-----|-------|
| observe · command · pose · place · reset | `SimBody` (`bridge.py`) — the robot |
| camera intrinsics · "is EGL on the GPU I pinned" | `SimBody` — the renderer is the body's |
| floor height · geodesic distance · shortest path · random point · traversability map | `body.scene`, a `SimScene` (`sim_scene.py`) — the world |

The floor is bound into `SimScene` when the simulator is opened, so it is not a
parameter anybody forwards. `SimBody._env` is private: reaching through it is
the friction both halves exist to remove (see Under the hood).

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
- **The consumer holds the tick loop, and nothing keeps a record.** A
  `TickRecord` pins a decoded 640x480 frame — 0.92 MB — so an episode's worth
  is up to 405 MB and a 20-task scene's worth is gigabytes. The episode yields
  them one at a time and keeps only what the metrics need (poses, collision
  flags, the last step), so a consumer that wants numbers holds no frames and
  one that wants a video encodes each frame and drops it. That is also the seam
  P4 attaches to: the recorder adds a line to the loop in `run_scene` rather
  than widening a callback the runner has to know about. A callback could only
  ever be handed what the runner thought to pass, which is why the old
  `on_tick` — a `TickRecord` and nothing else, no task, no output path — was
  never going to carry P4.
- **The simulator is reached through two adapters, and never around them.**
  `SimBody` had only a robot interface, so twelve call sites across seven files
  reached through `body.env` for world answers — `body.env.scene.floor_heights`,
  `body.env.simulator.renderer` at five sites for the GPU check. The P1
  architecture review predicted exactly this ("P3 needs geodesic distance ...;
  P4 needs the traversability map. All would leak the same way") and both
  phases proved it. `SimScene` is the widened seam; `_env` is now private. The
  payoff is not tidiness — it is that the geodesic rule which decides success,
  the floor lookup and the axis flip are now pinned by tests that need no GPU,
  and `verify_gpu` exists once instead of five times.
- **Every row records how the policy was steered.** `num_samples`, the waypoint
  index, the localization radius and the close threshold decide the action as
  much as the checkpoint does, and they used to be reachable only by editing
  `nomad_policy.py` — the only dials here with neither a config section nor a
  column. They are `driver:` in the config and a `driver` column (`n8w2r4t3`)
  in every row now. The defaults are unchanged, and are meant to stay that way
  (plan decision E); the column is what lets a table *prove* it, which matters
  because P5 is allowed to tune them and P6 requires that every arm faced the
  same ones. A run whose settings depart from the robot's prints **TUNED** in
  its header.
- **Recording off is off by construction, not by inspection.** The alternative
  — `if recording:` inside the tick loop — is one edit away from filming an
  unattended 20-task run by accident, and the loop that scores would no longer
  be the loop that was tested. A null object cannot drift: the loop is the same
  loop, and `p4_1_record_test.py` runs the same episode twice, filmed and not,
  and fails if any column of the metrics row moves.
- **The overlay recovers the localization window; it is not handed one.**
  `PolicyStep` names its nodes by absolute trail index but carries the distance
  head's scores by window offset, so the number under the subgoal image is
  `distances[subgoal_node - (closest_node - argmin(distances))]`. Off by one
  there puts a plausible number under a plausible picture, for the wrong node,
  and nothing looks wrong — `tests/test_recorder.py` pins it, including the
  clamped window at the start of a trail.
- **The panel figure is built once per episode and only its data changes.**
  matplotlib is fast at `set_data` and slow at `subplots`; at hundreds of ticks
  an episode and tens of episodes a run, rebuilding the figure per frame would
  cost minutes per arm. The map, the trail and the start/goal markers cannot
  change during an episode, so they are drawn at open time and never touched.
  The waypoint axes are fixed on the first tick for the same reason a video
  needs it: an autoscaled axis reads as the world lurching rather than the
  prediction changing.
- **The top-down panel plots world metres, not map pixels.** iGibson's
  traversability map is a square image centred on the world origin, so handing
  matplotlib its bounds as an `extent` removes every per-point conversion from
  the tick loop. It is drawn `origin="lower"` so `+y` is up: a left turn on the
  map is then a left turn in the camera beside it. (`p2_1_build_test.py` draws
  its still in pixel coordinates, which mirrors y — fine for one picture,
  confusing in something you watch.)
- **A `TickRecord` holds the policy's output rather than copying it.**
  `record.step` is the `PolicyStep` itself, so `distances` (the temporal
  distance to every node in the localization window) and `samples` (all eight
  diffusion trajectories) reach an overlay. The nine-field copy that preceded
  it silently dropped both, although `nomad_policy` annotates them as being
  "for inspection and overlays".
- **There is one answer to "when is an episode over".** The bridge makes ticks;
  `episode_runner` ends them. `NomadBridge.run()` used to be a second answer
  that stopped when the distance head localized onto the last node — the real
  robot's rule, and a claim about where the agent *thinks* it is. It is gone,
  because the next person wanting "just run an episode" would have found it
  first, on the object they already had. `p1_1_drive_test.py` keeps that rule
  in its own three-line loop, where it belongs, and keeps every frame on
  purpose because it writes a GIF.
- **The body reports contact; it does not count it.** `SimBody.collision_ticks`
  was a tally on the robot that two later tallies superseded while still
  looking authoritative. A body has no business knowing how it is being scored.
- **Rows are appended as they finish, not written at the end.** A run is tens
  of slow episodes; a crash on the last one must not cost the rest, and a run
  in progress should be readable with `tail -f`. Hence also `python -u` in the
  wrapper scripts — `docker exec` hands Python a pipe, not a tty, so stdout is
  block-buffered otherwise and an unattended run looks hung for minutes at a
  time.
- **The distance head's own reading is a column now, and it had to be.** The
  trace recorded which node the agent localized onto and which it steered at,
  and those two cannot distinguish "the trail is not advancing" from "the head
  is confident": the subgoal only moves past the closest node when the head's
  reading for that node falls under `close_threshold` (3), so an episode whose
  readings sit above it steers at the node it believes it is *already at*,
  tick after tick. That number existed nowhere but a pixel on a video frame.
  It is `dist_closest` and `dist_subgoal` in the per-tick log, and the
  arithmetic that recovers it moved onto `PolicyStep` — the step names its two
  nodes by absolute trail index but carries the head's scores by window offset,
  and a second copy of that recovery is exactly where an off-by-one hides.
  `tests/test_localization.py` pins it beside the index convention it undoes.
- **The field of view is rendered, not asserted.** `SimBody.render_at_vertical_fov`
  changes the camera, takes one frame and puts it back in a `finally`, so a
  diagnostic cannot leave the rollout camera altered behind it. Changing the
  camera an episode actually runs under is an edit to `vertical_fov` in the
  world config, reviewed as such — which is the point: P5 is allowed to *show*
  that 45 degrees is not what GoStanford was shot through without quietly
  becoming the phase that changed it.
- **A pixel statistic cannot tell the two lenses apart, and one was tried.**
  The obvious candidate is the dark border a fisheye leaves in the corners of a
  rectangular frame. Measured over 120 training frames and 32 sim frames it
  came out *higher for the sim* — 1.6% against 0.3% at a threshold of 10 —
  because GoStanford's frames are crops from inside the lens circle rather than
  the whole circle, and Rs has dark furniture. The comparison is two bands of
  pictures in `p5_1_input_check.png` for that reason, not for lack of trying to
  make it a number.
- **A task set is fingerprinted on its world's contents, not its path.** The
  camera, the robot and the physics are half of what a task is: every trail
  image is rendered through that camera, and every episode reopens the copy of
  the world saved beside its trail. The fingerprint used to hash
  `scene_config: configs/locobot_rs_bridge.yaml` as a *string*, so an edit to
  `vertical_fov` in that file matched the old fingerprint: an existing set was
  reused at the old camera without a word, and a new set could not be told
  apart from it. It now hashes the resolved world as well
  (`TaskSetConfig.world()`), and `tests/test_task_set.py` pins that a camera
  edit moves it. Found by the P5 architecture review, which is why every task
  set built before it has to be rebuilt once — with the same seeds, the same
  trails come back.
- **A diagnosis is a reading, not a measurement.** `diagnosis.py` computes
  nothing the metrics table does not already contain; it only names what is
  there. Its thresholds are therefore allowed to be judgement calls, and moving
  one renames an episode without moving a number P6 reports — which is the
  whole reason it is a separate module from `metrics.py` rather than another
  column in it.

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
| `FAILED: the task set in ... was built from a different config` | working as intended — restore the config, or build the new set in a new directory and re-run *every* arm against it. Editing the world config (the camera, the robot) counts; a set built before the P5 review needs one `--rebuild-tasks` |
| `FAILED: ... has no manifest.json` | no task set there yet; `run_eval.py` builds one (`--build-only` to stop after) |
| `FAILED: ... pins a start/goal pair` | the topomap config has `start`/`goal` set, so every task would be the same trail |
| `FAILED: the metrics table does not hold up` | a metric disagrees with the numbers beside it in its own row — the message names which |
| `success_metric must be one of ('geodesic', 'euclidean')` | a typo in `episode.success_metric` |
| `No module named 'imageio'` when recording | the container predates LAYER 3b of `sim.Dockerfile` — `pip install "imageio<3" imageio-ffmpeg` inside it, or rebuild the image |
| `FAILED: ... holds N frames but the episode ran M ticks` | the video and the run disagree; the encoder dropped or doubled a frame |
| `FAILED: filming changed the episode` | the recorder touched the run it was meant to watch — the message names every column that moved |
| `recording tasks must be 'all', a count, or a list of task ids` | a typo in `recording.tasks` |
| `TypeError: __init__() got an unexpected keyword argument` on startup | a typo in a config knob — the section names it; nothing is silently ignored |
| the run header says `driver: ... (TUNED — not navigate.py's defaults)` | working as intended: `driver:` in the config departs from the real robot's settings (plan decision E) |
| `RuntimeWarning: divide by zero` from `point_nav_fixed_task.py` | harmless — iGibson's own built-in task computes its own SPL at reset, with a zero path length. The bridge ignores that task entirely; the goal is the topomap. |
| `FAILED: no GoStanford at /data/...` | running the input check outside the container, where `/data` is not mounted |
| the diagnosis says `false arrival` | working as intended: the distance head localized onto the last node while the agent was somewhere else. The episode ran on (plan §6) and the tick it claimed is in the table. |
| a diagnosis disagrees with the video | the video wins. A mode is a label on something you can watch, and the thresholds in `diagnosis.py` are for reading, not scoring. |

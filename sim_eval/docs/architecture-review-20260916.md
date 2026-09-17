# sim_eval — architecture review after P1

**Date:** 2026-09-16
**Branch:** `feature/sim-eval-p1-bridge`
**Scope reviewed:** all of `sim_eval/` (1,920 lines, 12 modules, 37 tests at review time)
**Plan:** `.claude/plans/nomad-sim-evaluation.md`
**HTML version:** `architecture-review-20260916-183050.html` (same directory — richer diagrams, same findings)

This is the Markdown companion to the HTML report, kept so the reasoning is
diff-able in git rather than locked inside a styled page.

---

## Why this review exists, and what it asked

P1 closed the evaluation loop: NoMaD's brain now drives the simulator's body, so
an action changes the next view. The review did **not** ask "is this code good".
It asked a narrower, more useful question:

> Where will P2, P3 and P4 have to **change** this code rather than add to it?

A phase that must rewrite existing code is the signal of a missing or misplaced
seam. That framing is what ordered the candidates below.

### Vocabulary

There is no `CONTEXT.md` and no `docs/adr/` in this repo, so the review took its
domain terms from the plan and `CLAUDE.md`, and its architecture terms from the
`/improve-codebase-architecture` skill.

- **Domain:** trail (topomap) · node · subgoal · control tick · episode ·
  rollout · arm (a checkpoint under test) · scorer · recorder.
- **Architecture:** depth (implementation much larger than interface) · shallow ·
  seam · adapter · locality · leverage · **deletion test** (inline a module into
  its caller — does complexity *concentrate*, meaning it was shallow, or
  *spread*, meaning it was deep?).

---

## The pattern behind every finding

P1's three **brain-side** seams are clean and deep:

| Module | Why it is deep |
|---|---|
| `nomad_policy.py` | hides the whole network behind `act()` |
| `pd_control.py` | hides the steering behind `pd_controller()`, pinned by 10 tests |
| `checkpoints.py` | hides "which arm, at what resolution" behind `load()` |

All the friction is on the **harness** side. Two of the five components plan §4
names — the **topomap builder** and the **scorer/recorder** — have no module at
all, so their responsibilities are spread thin across `bridge.py` and the two
entry-point scripts. That was fine for P1, which only had to prove one rollout
works. It stops being fine at P2 (which rewrites the trail producer) and P3
(which owns episode rules).

---

## Candidates

Eight, ordered by how soon the next phase collides with them. G is first because
it is the only one that can make the final **numbers wrong** rather than merely
make the next phase awkward.

### G — The localization window is a proven bug site with zero tests

**Strength: Strong. Chosen and implemented.**

**Files:** `nomad_policy.py:173–188` (the window math) · `nomad_policy.py:139`
(`load_model` inside `__init__`) · 232 lines, 0 tests — the largest untested
block in the package.

**Problem.** Five lines of index arithmetic decide which topomap node the robot
steers toward:

```python
start = max(closest_node - radius, 0)
end   = min(closest_node + radius + 1, goal_node)
min_idx      = argmin(dists)
closest_node = min_idx + start
sg_idx = min(min_idx + (dists[min_idx] < thresh), len - 1)
```

Pure index arithmetic over a numpy array — and **upstream has already shipped
this bug once.** Commit `7b5b24c` ("Fix closest node update for topomap
localization in navigate.py", Ajay Sridhar, 2024-07-13) replaced:

```python
closest_node = np.argmin(distances)          # window-relative  (WRONG)
```

with:

```python
closest_node = start + min_dist_idx          # absolute         (RIGHT)
```

That is a window-relative index being used as an absolute one. The port in
`sim_eval/` has it right. Nothing proved it would stay right.

**Why this failure mode is the dangerous one.** An off-by-one here does not
raise. It steers the robot confidently toward the wrong node. That looks
*plausible* in a replay GIF, and it would silently corrupt every metric computed
from the run. No rollout can catch it; only assertions can. The whole
justification for building this harness is that the open-loop "best" rested on a
proxy (`gc_action_loss`) flagged as not-real-navigation — a silent off-by-one
here would replace one untrustworthy ruler with another.

**Why it was not tested.** `NomadPolicy.__init__:139` calls `load_model` →
`torch.load` → `.to(device)`. The object cannot exist without real weights and a
GPU. To reach five lines of arithmetic you would have to stub the entire model
protocol: a string-dispatching callable answering three different call shapes
(`vision_encoder`, `dist_pred_net`, `noise_pred_net`), plus `DDPMScheduler` and
`get_action`.

**Solution chosen — and the one rejected.**

The exploration agent proposed adding a **model factory/protocol parameter** to
`NomadPolicy.__init__` so a fake model could be injected. **This was rejected**
for three reasons:

1. **It buys testability at the wrong price.** It changes the constructor of a
   module whose entire value is being a verbatim port of `navigate.py`. Plan
   decision E and `CLAUDE.md`'s "reuse by copy, not import" rule both exist so a
   future reader can diff this file against the deployment stack. A new
   injection parameter is a visible divergence from the source.
2. **It tests the wrong thing.** Injecting a fake model lets you exercise `act()`
   end to end, but the *risk* is not in the plumbing — it is in five lines of
   arithmetic. A test that stubs five collaborators to reach those five lines is
   dominated by its own scaffolding, and scaffolding is where test bugs live.
3. **It is more expensive than the alternative.** The injection seam is a
   structural change to a ported module. Naming the arithmetic is a move of five
   lines.

**Solution implemented instead:** lift the arithmetic into two pure functions —
`localization_window(closest_node, goal_node, radius)` and
`localize(distances, start, num_window_nodes, close_threshold)`. The numbers are
unchanged, so fidelity to `navigate.py` is preserved and still diff-able, but
the logic becomes testable with a hand-written list of distances — no torch, no
checkpoint, no GPU.

The signature now *names* the distinction the upstream bug got wrong:

- `closest_node` is an **absolute** trail index — what the next tick centres its
  window on, and what "reached the goal" is tested against.
- `subgoal_offset` is **relative to the window** — it indexes the encoder output
  for this window, which has `num_window_nodes` rows.

**Benefits.**

1. *The asymmetry is the argument.* The other verbatim port, `pd_control.py`, has
   10 tests and is the best-pinned module in the package. The policy port —
   longer, and the one with a real bug in its history — had none.
2. *Leverage:* the deletion test inverts here. Extracting **concentrates** the
   risky logic into a named, tested unit and leaves `act()` as plumbing.
3. *Tests:* window clamping at both ends, the `start` offset, the close-threshold
   step, and tie-breaking become ~15 assertions costing nothing to run.

**Verification.** After extraction, the drive test was re-run and the tick-by-tick
trace diffed against the pre-refactor run: **all 39 tick lines byte-for-byte
identical**, same final pose, same 0.33 m, same `reached goal: True`.

---

### A — The trail is a concept with no module

**Strength: Strong. Open — do before P2 starts.**

**Files:** `p1_0_make_topomap.py:101,144–148` (writes) · `bridge.py:57`
`load_topomap` (reads the images) · `p1_1_drive_test.py:63–75,137–139` (reads the
poses).

**Problem.** A trail is one thing: numbered node images, the pose each was taken
from, and the start pose. Its format is invented in one file, its images loaded
in a second, and its poses dug out of a raw dict in a third —
`route["nodes"][-1]["pose"]` at `p1_1_drive_test.py:139`. Nothing owns the
contract, so nothing can validate it and nothing can be tested against it. The
`route.json` schema is unversioned and unvalidated, read at three sites, and it
is precisely the file P2 is going to rewrite.

**Solution.** A `trail.py` that reads and writes its own format:
`Trail.save(dir)` / `Trail.load(dir)`, with `.nodes` (image + pose),
`.start_pose`, `.goal_pose`, `.reference_path_length`. `load_topomap` moves out
of `bridge.py`; `read_start_pose` disappears from the script.

**Benefits.** One file answers "what is a trail" (today: three). P2 replaces only
the *producer* — scripted route → the sim's own planner — and inherits the reader
unchanged. P3 gets `reference_path_length`, which it needs for the timeout *and*
for SPL's denominator, from the module that already knows the node poses.
Round-tripping a fabricated trail needs no GPU, so the format becomes the test
surface.

---

### B — Episode rules are scattered across four places

**Strength: Strong. Open — do at the start of P3, together with C.**

**Files:** `bridge.py:234` `reached_goal` · `bridge.py:273–292` `run` ·
`bridge.py:137,146,165` `collision_ticks` on `SimBody` ·
`p1_1_drive_test.py:35` `DEFAULT_MAX_TICKS` · `p1_1_drive_test.py:96–120`
`report`.

**Problem.** "When does an episode end, and how did it go" is four fragments: the
stopping rule on `NomadBridge`, the tick budget as a script constant, the
collision count as a mutable counter on `SimBody` (the *body* has no business
scoring), and the outcome computed inside the script's `report()` as print
statements rather than a value. Plan §6 defines five metrics and two termination
conditions; **P3 cannot add any of them without editing `run()`**.

Three sharper points found on a second pass:

1. `reached_goal` is *topomap localization*, not geometry — plan §5 defines
   success as within ≈1 m of the goal **pose**, and the bridge holds no goal pose
   at all (`start_episode` takes only a start pose). This property must change,
   not extend.
2. `collision_ticks` counts *ticks with any contact*, not collision events, and
   has no per-metre normalisation — plan §6 asks for "collisions per episode /
   per meter". It is also reset by `SimBody.reset`, so the *body* controls when
   the *score* resets.
3. `report()` at `p1_1_drive_test.py:100–107` already computes two of §6's five
   metrics (final distance-to-goal, path length) — inside a print function, in a
   phase-numbered script. P3 must lift or duplicate them.

**Solution.** An `episode.py` owning termination and outcome. `NomadBridge`
narrows to one job — produce the next `TickRecord` — and the episode decides
whether to ask for another. Success radius, timeout-from-reference-length,
collision tally and distance-to-goal become fields on an `EpisodeOutcome` that P3
writes to CSV and P1's script prints.

**Benefits.** P3 becomes additive instead of a rewrite of the loop P1 just
validated — the riskiest possible change. Termination is pure logic over poses and
counters, so fed a fake tick stream it tests with no GPU, no iGibson and no model
— which is exactly the logic that decides every number in the final table, and is
0% covered today.

---

### C — `run()` holds every frame of the episode in memory

**Strength: Strong. Open — do with B.**

**Files:** `bridge.py:273–292` `run() -> list[TickRecord]` · `bridge.py:206`
`TickRecord.frame` · `p1_1_drive_test.py:84–94` `write_frames`, after the run
finishes.

**Problem.** `run()` returns a list and every element pins a decoded 640×480
frame. Measured: **0.92 MB per frame in RAM**, so P1's 39-tick run held 36 MB and
a 200-tick episode would hold **184 MB** before a byte is written. Worse, frames
are written only *after* the loop — so a crash or timeout at tick 199 loses the
whole recording, which is the one artefact you wanted from a failed episode. On
disk P1 measured 0.51 MB/frame; P6's 40 rollouts at 200 ticks project to **~4 GB
of PNGs**.

Two real bugs in the writer (both **fixed** alongside G, since they were in code
about to be committed):

1. `write_frames` raised `IndexError` at `p1_1_drive_test.py:92` (`frames[0]`) on
   an empty record list — reachable with `--max-ticks 0`.
2. It deletes every `*.png` in the output directory before writing. Harmless for
   one flat run; actively destructive once P4 puts per-episode subdirectories
   there.

**Solution.** Invert it. The `on_tick` callback already exists at
`bridge.py:280` — promote it from a progress hook to *the* interface, make the
tick loop a generator, and stop returning the list. A consumer that wants frames
(P4's recorder) encodes each as it arrives; a consumer that only wants numbers
(P3's scorer) never holds a frame at all.

---

### D — `SimBody`'s seam already leaks; callers reach through `.env`

**Strength: Worth exploring. Open.**

**Files:** `bridge.py:112–190` `SimBody` · `p1_0_make_topomap.py:121`
`body.env.simulator.renderer` · `p1_0_make_topomap.py:126–127`
`body.env.task.initial_pos/orn` · `p1_1_drive_test.py:149`
`body.env.simulator.renderer` · `p0_1_smoke_test.py:97,102` builds `iGibsonEnv`
directly.

**Problem.** Not a prediction — already happening. `SimBody` is meant to be *the*
adapter over iGibson, but call sites reach past it into `.env.simulator.renderer`
and `.env.task`. The interface does not cover what its callers need, so they
bypass it. The cause is a conflation: `SimBody` mixes **robot** questions
(observe, command, pose) with **world** questions (which GPU is rendering, where
may the robot start). P3 needs geodesic distance and start/goal sampling for SPL;
P4 needs the traversability map. All would leak the same way.

Also: `gpu.verify_renderer(body.env.simulator.renderer, …)` is repeated verbatim
at three call sites, each reaching three levels in. And
`ANGULAR_VELOCITY_SIGN` — the line the README itself names as the most breakable
in the package — **has no test**, because `SimBody` is untestable by
construction.

**Solution.** Split along the conflation: `SimBody` keeps
`observe / command / pose / place`; a `SimScene` answers world questions —
`sample_pose`, `geodesic_distance`, `trav_map`, `verify_gpu` — and `.env` stops
being public. Lighter alternative: keep one class and widen it until nothing
reaches through.

---

### E — Two scene configs share 38 of 51 substantive lines

**Strength: Worth exploring. Open — pays off when P2 adds a second house.**

**Files:** `configs/locobot_rs_static.yaml` (92 lines, 51 substantive) ·
`configs/locobot_rs_bridge.yaml` (111 lines, 51 substantive) — measured: 13
substantive lines differ.

**Problem.** The bridge config is the smoke-test config with six settings
changed. The other 38 lines — camera, traversability, rewards, sensors — are
duplicated, and they are exactly the settings defining *what world is rendered*.
If `vertical_fov` or `image_width` is tuned in one (and P5 is explicitly expected
to revisit FoV), P0's smoke test quietly starts describing a different world than
the bridge drives. Plan §B scales to 2 then 3 houses, so `scene_id` must vary —
today that means more whole-file copies.

**Solution.** One base config plus small named overlays, merged in Python.
**Verified feasible:** iGibson's `parse_config` accepts a Mapping as well as a
path (`igibson/utils/utils.py:29`), so `iGibsonEnv(config_file=<dict>)` works —
no temp files needed. Each overlay then documents itself: the bridge overlay is
six lines that are exactly the six deliberate differences.

---

### F — The config validator and its consumer disagree

**Strength: Worth exploring. FIXED alongside G.**

**Files:** `checkpoints.py:27–37` `REQUIRED_PARAMS` · `checkpoints.py:104–117`
`validate_params` · `nomad_policy.py:66–98` `build_model` (the actual consumer).

**Problem.** `checkpoints.py` promises to fail loudly rather than guess — that is
its whole reason to exist, because a wrong `image_size` would silently invalidate
the fairness protocol. But `REQUIRED_PARAMS` omitted five keys `build_model`
actually reads: `mha_num_attention_heads`, `mha_num_attention_layers`,
`mha_ff_dim_factor`, `down_dims`, `cond_predict_scale`. A log missing any of them
passed validation and then died with a bare `KeyError` deep inside model
construction — the exact failure mode the module was written to prevent.

**Fix applied.** All five keys added, grouped by which consumer reads them, and
pinned by value in `tests/test_checkpoints.py` so the list cannot silently shrink
again. Both real arms still resolve (`best_combined` 160×120, `clean_stock`
96×96).

---

### H — The package has no packaging

**Strength: Worth exploring. Partly fixed alongside G.**

**Problem.** Individually trivial, collectively a pattern. `sim_eval/` has no
`__init__.py`, no `conftest.py` and (at review time) no test runner, so every
module and test re-derives its own environment. All verified:

| Duplication | Count | Belongs in |
|---|---|---|
| `sys.path.insert(0, parents[1])` | ×4 (all test files) | `conftest.py` |
| `SIM_EVAL_DIR = Path(__file__)…` | ×5 modules | `paths.py` |
| shell arg re-quoting + `chown` | ×2, **byte-identical** | `lib.sh` |
| rgb→PIL conversion (`save_rgb` / `frame_to_pil`) | ×2 | `frame_to_pil` |
| stale-PNG clearing | ×2 | — |
| opening `iGibsonEnv` | ×2 (P0 never migrated to `SimBody`) | `SimBody` |

**And the docstrings overclaimed.** Three test files said *"Runs anywhere: no
GPU, no iGibson, no container."* Verified false: the host has no `python` on
`PATH` (only `python3`) and no `pytest`. Worse, `bridge.py:33` imported
`pybullet` at *module* scope while importing iGibson lazily — so
`test_context_queue.py` needed the container, directly contradicting its own
docstring. A comment claiming a property nothing enforces is worse than no
comment: a future reader trusts it.

**Partly fixed alongside G:** `pybullet` made lazy, all four test docstrings
rewritten to say what is actually required, and `run_tests.sh` added (every other
entry point had a runner; the thing run most often did not). The remaining
duplications are still open.

---

## Applied the deletion test, left them alone

Recorded so a later review does not re-litigate them.

| Thing | Verdict |
|---|---|
| `ContextQueue` (`bridge.py:74–110`) | 37 lines behind a 4-method interface — suspiciously thin. But inlining it into `NomadBridge` would put an off-by-one that silently feeds the encoder the wrong frame count back into the tick loop, where it is untestable. Complexity would **spread**. Keep. |
| `pd_control.py` (92 lines) | Deleting it concentrates nothing — it would just move the verbatim port into `bridge.py` and cost the 10 tests that pin decision E. Best-tested module in the package. Keep, untouched. |
| the training-log parse (`checkpoints.py:78–101`) | Parsing a repr'd dict out of a 590 KB log looks like a hack worth removing. It is not: it is the only surviving record of what each arm was trained with, correctly isolated behind one small interface. Keep the seam; F only tightens what it checks. |
| `TickRecord` (`bridge.py:196–220`) | **Verdict revised.** First passed as a fine value object. On a second look it is the *clearest* shallow class here: 9 positional params → 9 identical attributes, and it is an exhaustive copy that **already drops the two fields P4 needs** — `PolicyStep.distances` and `.samples` are set at `nomad_policy.py:195–196` and never read anywhere (grep-verified). Every future phase pays a 3-site edit. Fold into candidate B. |

---

## Deliberately not proposed

Nothing in this review touches the ported control logic. The diffusion loop
(`nomad_policy.py:198–232`), the PD mapping (`pd_control.py:70–92`) and the
retained no-op `split`/`cat` (`nomad_policy.py:152–155`) are verbatim from
`deployment/src/` by requirement (plan decision E, `CLAUDE.md` "reuse by copy").
Fidelity beats elegance there, and the fact that it *looks* improvable is the
point — a future reader must be able to diff it against the deployment stack.

Likewise `SimBody.ANGULAR_VELOCITY_SIGN` is not a magic constant to clean up: it
is a measured correction for an iGibson bug (`Locobot.base_control_idx = [1, 0]`
for `[left, right]` while the URDF lists `wheel_left_joint` first, so the
controller's left-wheel velocity reaches the physical right wheel; commanding
`w=+0.4 rad/s` measures as `−0.39`).

---

## Outside the code: this workstream has no conventions file of its own

The auto-loaded `CLAUDE.md` documents a *different* workstream —
`deeper_visuals/`, container `naz_nomad_deeper_viz`, the visualization plan. It
never mentions `sim_eval`. So sim_eval's real conventions live only in plan
§3b/§3c, and the two sources already conflict where they overlap: `CLAUDE.md`
makes the architecture pass mandatory at the end of *every* phase, while plan §3b
makes it explicitly optional. The "reuse by copy, not import" rule is scoped to
`debug_visuals/` there, yet `sim_eval` is the code actually relying on it.

There is also no `CONTEXT.md` and no `docs/adr/`, which is why this review had to
take its domain vocabulary from a plan file. Worth fixing before P3 multiplies
the number of terms (*episode*, *rollout*, *arm*, *trail*, *scorer*) that
currently have no written definition.

---

## Method note: where the strongest finding came from

Six candidates (A–F) came from a direct evidence pass over the code. A
read-only exploration agent was then run in parallel as a second pair of eyes,
deliberately not anchored on the author's own choices.

**It found candidate G — stronger than any of the original six.** Its claims were
verified independently before being accepted, rather than taken at face value:

| Claim | Verification |
|---|---|
| upstream shipped this bug | `git show 7b5b24c` — confirmed, and the diff shows exactly the window-relative/absolute confusion |
| `nomad_policy.py` has 0 tests | confirmed by inspection |
| `PolicyStep.distances` / `.samples` never read | grep across `sim_eval/**.py` — confirmed |
| host cannot run the tests | `which python` → none; `python3 -m pytest` → no module — confirmed |
| `write_frames` crashes on empty records | confirmed at `p1_1_drive_test.py:92` |
| arg-quoting blocks identical | `diff` of the two blocks — no output |

Its *proposed fix* for G was rejected (see G above), but its *diagnosis* was
correct and better than the author's own. Its `TickRecord` observation also
overturned an earlier verdict in this review.

---

## Recommended order

1. **G** — name and pin the localization math. *Before committing P1.* ✅ **Done.**
2. **A** — give the trail a module. *Before P2 starts*, since P2 is the trail builder.
3. **B + C together** — at the start of P3. Both reshape `run()`; doing them
   separately means touching it twice. B also absorbs the `TickRecord` revision.
4. **F** and the `write_frames` crash — ~20-minute fixes. ✅ **Done** alongside G.
5. **E** — when P2 adds a second house.
6. **H** — one cleanup sitting, mostly so the test docstrings stop claiming
   something untrue. ⚠️ **Partly done** (pybullet, docstrings, `run_tests.sh`).
7. **D** — revisit once P3 exists and its second consumer makes the seam fully real.

---

## What was implemented on 2026-09-16

| Change | Files |
|---|---|
| `localization_window()` + `localize()` extracted as pure functions | `nomad_policy.py` |
| 15 tests pinning the window math, incl. a direct `7b5b24c` regression test | `tests/test_localization.py` |
| `write_frames` returns `None` on an empty run; `report()` and the final print handle it | `p1_1_drive_test.py` |
| `REQUIRED_PARAMS` completed (5 missing keys) and pinned by value | `checkpoints.py`, `tests/test_checkpoints.py` |
| `pybullet` made a lazy import; four test docstrings corrected | `bridge.py`, `tests/*.py` |
| `run_tests.sh` added | `sim_eval/run_tests.sh` |

**Result:** 53 tests passing, pyflakes clean, and the drive test's 39-tick trace
reproduced byte-for-byte after the refactor. Committed as `29ae8ab`.

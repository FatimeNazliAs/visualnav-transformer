# sim_eval — architecture review after P4

**Date:** 2026-09-17
**Branch:** `feature/sim-eval-p4-recorder`
**Scope reviewed:** all of `sim_eval/` (4,800 lines Python across 17 modules; 176 tests at review time, 209 after)
**Plan:** `.claude/plans/nomad-sim-evaluation.md`
**HTML version:** `architecture-review-20260917-145341.html` (same directory — richer diagrams, same findings)
**Predecessor:** `architecture-review-20260916.md` (after P1)

Markdown companion to the HTML report, kept so the reasoning is diff-able in
git rather than locked inside a styled page.

---

## What this review asked

The same narrow question the P1 review asked, moved on two phases:

> Where will **P5** and **P6** have to *change* this code rather than add to it?

A phase that must rewrite existing code is the signal of a missing or misplaced
seam. That framing is what ordered the candidates.

### Vocabulary

Still no `CONTEXT.md` and no `docs/adr/` in this repo, so the domain terms come
from the plan and the README (*trail · node · subgoal · control tick · episode ·
arm · scorer · recorder*) and the architecture terms from the
`/improve-codebase-architecture` skill (*depth · shallow · seam · adapter ·
locality · leverage · the deletion test*).

---

## The pattern behind this round

P4 went in cleanly. The recorder is a deep module — 597 lines behind
`capture(record)` — and it did not touch `episode_runner.py` at all, because
P3.5 had already opened the right seam (an episode is a stream; the consumer
holds the tick loop). That is the system working.

But to draw its top-down panel, P4 had to reach `runner.body.env.scene` —
through the adapter, into the simulator. The P1 review predicted exactly that,
in writing:

> "P3 needs geodesic distance and start/goal sampling for SPL; P4 needs the
> traversability map. **All would leak the same way.**"
> — *architecture-review-20260916.md, candidate D, marked Open*

Both predictions came true. The leak count went from ~5 to **12**.

---

## Candidates

| | Candidate | Strength | Outcome |
|---|---|---|---|
| **A** | The simulator adapter leaks — 12 reach-throughs past `SimBody` into `.env` | Strong | **Implemented** |
| **B** | The 4 policy knobs P5 will tune have no config and no provenance | Strong | **Implemented** |
| **C** | `run_eval.py` has no run object; signatures widen every phase | Worth exploring | Open |
| **D** | Six shell runners are the same 13 lines | Worth exploring | Open |
| **E** | `EvalConfig` — 307 lines, 0 tests — is what P6 runs on | Worth exploring | **Implemented** |
| **F** | Two scene configs still share 38 substantive lines (carried forward) | Speculative | Open |

### A — The simulator adapter leaks, and P4 widened the hole

**Strength: Strong. Implemented.**

**Problem.** `SimBody` is meant to be *the* adapter over iGibson, but its
interface covered only the **robot** — `observe · command · pose · place`.
Callers needed **world** answers too, so twelve call sites across seven files
reached through `.env`:

| Site | Reaching for |
|---|---|
| `episode_runner.py:284` | `body.env.scene.floor_heights[floor]` |
| `episode_runner.py:385` | `body.env.scene` → a geodesic helper |
| `run_eval.py:181` | `runner.body.env.scene` → the recorder's map panel (**new in P4**) |
| `run_eval.py:207`, `p1_0:121`, `p1_1:184`, `p2_1:210` | `body.env.simulator.renderer` → the GPU check |
| `topomap_builder.py:249` | `body.env.simulator.renderer` → camera intrinsics |
| `topomap_builder.py:496`, `p2_1:219` | `body.env.scene` |
| `p1_0:126,127` | `body.env.task.initial_pos/orn` |

Two aggravations: `gpu.verify_renderer(body.env.simulator.renderer, gpu)`
appeared **verbatim at 5 sites**, each reaching three levels in — and getting it
wrong means silently rendering on a teammate's GPU, the one mistake plan §3c
says must never happen. And `ANGULAR_VELOCITY_SIGN`, the line the README names
as the most breakable in the package, had **no test**, because `SimBody` could
not be constructed without a GPU.

**Deletion test.** Inlining `SimBody` would scatter iGibson trivia across seven
files — complexity *spreads*, so it is deep and stays. It was not a shallow
module to delete; it was a **deep module with too narrow an interface**.

**What was done.** Split along the conflation the P1 review named:

- New `sim_scene.py` — `SimScene` wraps iGibson's `IndoorScene` and **binds the
  floor once**: `floor_height`, `geodesic_distance`, `shortest_path`,
  `random_point`, `has_node`, `trav_map`, `trav_map_resolution`, `world_to_map`,
  `scene_id`. No iGibson import, so all of it is testable against a fake.
- `SimBody` keeps the robot and gains what is genuinely the *renderer's*:
  `intrinsics`, `verify_gpu(expected)`, plus `initial_pose` for P1's script.
  `env` became `_env`.
- `episode_runner.geodesic_distance` and its `NO_PATH_ERRORS` moved onto
  `SimScene`; `topomap_builder.plan_path` and `camera_intrinsics` dissolved into
  `scene.shortest_path` and `body.intrinsics`; `p2_1`'s `to_plot_xy` became
  `scene.world_to_map`.
- **`floor` stopped being a parameter.** `EpisodeRunner(policy, body, rules)`
  lost it, `PanelFigure` and `Recorder.episode` lost it, `sample_start_goal` and
  `resolve_start_goal` lost it. It is bound where the simulator is opened.

**Result:** 12 reach-throughs → **0**. (`p0_1_smoke_test.py` still builds a raw
`iGibsonEnv` and checks its renderer directly — that is P0 proving the raw stack
works, not a reach-through past an adapter.)

**Tests:** new `tests/test_sim_scene.py` (12 tests) pins the geodesic rule
including *both* networkx failures, the re-raise of everything else, the axis
flip and the bound floor. The existing fakes in `test_episode_runner.py` and
`test_topomap_builder.py` were rewired to wrap a real `SimScene`, so 55 more
tests exercise the adapter for free.

### B — The four knobs P5 is about to turn had no config and no provenance

**Strength: Strong. Implemented.**

**Problem.** `num_samples=8`, `waypoint=2`, `radius=4`, `close_threshold=3` are
`navigate.py`'s argparse defaults, and they were reachable only by editing
`nomad_policy.py`. Every other dial in the harness has a config section **and** a
column in the metrics table; these had neither. So a table written under a
changed value was byte-indistinguishable from one written under the default —
while P5's entire job is *"sanity-tune only what fidelity allows"* and P6's
fairness protocol (plan §7) requires every arm to face identical settings.

**What was done.**

- New `driver.py` — `DriverConfig` with `label()` (`n8w2r4t3`),
  `is_deployment_default()` and a `summary()` that prints **TUNED** when the
  settings depart from the real robot's. Its own module, not
  `nomad_policy.py`'s, so reading configuration does not cost the torch stack.
- `NomadPolicy(spec, device, driver=None)` takes it; the four keyword arguments
  are gone.
- `driver:` section in `configs/eval.yaml`, and a **`driver` column in every row**
  of the metrics table.

**This does not violate plan decision E** ("mirror the real LoCoBot exactly;
tune nothing"). No default changed — `test_eval_config.py` asserts the shipped
config is still the deployment default, so decision E is now machine-checked
rather than asserted in prose.

### C — The driver threads pass-through state through widening signatures

**Strength: Worth exploring. Open.**

`run_scene` went 6 → 7 parameters at P4, `score_checkpoint` 7 → 8, `evaluate` is
at 9; five of `run_scene`'s seven are threaded through untouched. A `ScoringRun`
holding config, policy, checkpoint name, table, recorder and GPU would take it to
two. Left open deliberately: A already removed one of those parameters (`floor`),
and the right moment is when P5 or P6 forces the next signature change anyway.

### D — Six shell runners are the same thirteen lines

**Strength: Worth exploring. Open.**

`diff run_p3_score_test.sh run_p4_record_test.sh` returns only the header comment
and the script name. The same duplication exists in Python: the `select_gpu`
prologue at 7 sites (in two different spellings of the same print), the `--gpu`
argument at 7, the `__main__` try/except at 4. Left open: each runner's header is
real documentation, and the copy has not yet cost a bug. Revisit at P5's runner.

### E — The config that decides everything P6 runs had zero tests

**Strength: Worth exploring. Implemented.**

`EvalConfig` composes the other four config objects and every command goes
through it; it had 0 tests while `TaskSetConfig` (17), `TopomapConfig` (23),
`EpisodeRules` (32) and `RecordingConfig` (14) had 86 between them.

**What was done.** `tests/test_eval_config.py` (14 tests) pins the three
behaviours that are silent when they break: a typo in any section fails at load
rather than being ignored; relative paths resolve against `sim_eval/` and not the
cwd; a missing section yields a usable default, and for the recorder a default
that is **off**. It also pins the checked-in `configs/eval.yaml` itself — a
driver knob edited and left behind in it would otherwise silently become P6's
headline settings.

### F — Two scene configs still share 38 substantive lines

**Strength: Speculative. Open, carried forward unchanged from the P1 review.**

Downgraded from *Worth exploring* because P2 made each task write its own
`world.yaml` beside its trail, so the world a run uses is pinned per task and
travels with it. Re-rank the moment P6 adds the second house, or P5 tunes the
camera FoV in one file and not the other.

---

## Applied the deletion test, left them alone

Recorded so a later review does not re-litigate them. All four are P4's code.

| Thing | Verdict |
|---|---|
| `NullRecording` (18 lines) | **Keep.** Maximally shallow-looking, but deleting it puts `if recording:` inside the tick loop — complexity *concentrates* in the most correctness-critical loop in the package, and "recording off changes nothing" stops being true by construction. |
| `parse_subset` (7 lines) | **Keep.** The single place the CLI and the YAML agree on what `all` and `3` mean. Inlining lets the two spellings drift silently. |
| `TrailImages` (25 lines) | **Keep, marginally.** Inlining moves a lazy decode-and-cache into the drawing code, where it reads as a drawing concern. Complexity mostly *moves* rather than concentrating — the weakest keep here. |
| `recorder.py` as one module (597 lines, 3 concerns) | **Do not split.** A `panels.py` would always change in the same commit as its only caller. Locality says keep one module; depth is fine. |

---

## Deliberately not proposed

- **The ported control logic.** The diffusion loop, the PD mapping and the
  retained no-op `split`/`cat` are verbatim from `deployment/src/` by
  requirement. Fidelity beats elegance; that it *looks* improvable is the point,
  because a reader must be able to diff it against the deployment stack.
- **A cross-arm comparison module.** P6 needs one and it does not exist — but
  that is an *add*, not a change, so it is not architectural friction.
  `metrics.read_table` and `aggregate` already expose the seam for it.
- **The two-GPU launch.** Already solved and documented in `run_eval.sh`'s
  header: the same script twice, `SIM_GPU=0` and `SIM_GPU=1`, both reading one
  immutable task set.

---

## What was implemented on 2026-09-17

Candidates **A**, **B** and **E**, in that order, on `feature/sim-eval-p4-recorder`.

**New modules:** `sim_scene.py` · `driver.py`
**New tests:** `tests/test_sim_scene.py` (12) · `tests/test_driver.py` (7) ·
`tests/test_eval_config.py` (14) — **176 → 209 tests**, still GPU-free and still
2.4 s.
**Touched:** `bridge.py` · `episode_runner.py` · `topomap_builder.py` ·
`nomad_policy.py` · `metrics.py` · `run_eval.py` · `recorder.py` ·
`p1_0_make_topomap.py` · `p1_1_drive_test.py` · `p2_1_build_test.py` ·
`configs/eval.yaml` · `README.md` · 4 test files.

### How it was verified that nothing moved

The risk in an architecture pass on a measuring instrument is that a number
changes. So the refactor was checked against the simulator, not only against the
unit tests:

1. `p3_1_score_test.csv` was saved before the refactor, the 3-episode scorer was
   re-run after it, and the two tables were **diffed: identical**, across 838
   ticks including a success, two timeouts and 682 colliding ticks.
2. `p4_1_record_test.py` re-ran the same episode filmed and unfilmed and
   confirmed every column still agrees between the two.
3. The full GPU-free suite: 209 passed.

The only intended difference in the table is B's new `driver` column.

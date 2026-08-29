# STOPPED — do not build on this branch

**Branch:** `experiment/behaviour-failure-analysis`
**Started:** 2026-08-25 · **Stopped:** 2026-08-29
**Status:** abandoned by decision, not by failure. **Do not continue it.**

## What this was

The first task of a *NoMaD Behaviour and Failure Analysis*: a preliminary study
asking in which recorded situations goal-conditioned NoMaD produces coherent
predictions, and in which its behaviour becomes uncertain or suspicious. It was
never the thesis contribution — it was meant to surface a reproducible weakness
that a later, focused enhancement could justify.

## Why it stopped

A different idea came up that we preferred. Work stopped here by choice while
the scaffolding was healthy and before any experiment ran. Nothing here failed,
and nothing here is broken — it is simply unfinished, and it is **not** going to
be finished.

## How far it got

Done, committed, and verified:

- `behaviour-failure-analysis/common/` — a NoMaD inference layer (load the
  checkpoint, load a scene, decode actions, run the diffusion head, mask the
  goal), copied and adapted from `deeper_visuals/common/` on
  `feature/deeper-visualization`.
- `behaviour-failure-analysis/01-goal-conditioned-behaviour/sanity_check.py` —
  nine provenance and correctness checks. All nine passed on the real
  checkpoint.

Never written: case selection, the calibration sweep, the experiment runner,
the analysis, the figures. **No experiment was ever run, and no result was ever
produced.** There are no findings here about NoMaD's behaviour, because the part
that would have produced them does not exist.

## Read this before reusing anything

The sanity check passing means the *plumbing* is right, not that any conclusion
was reached. Anyone tempted to lift code from here should take `common/` — which
is a straight adaptation of the well-tested `deeper_visuals/common/` — and
ignore everything else. Do not resume the experiment design; it was never
finished enough to be trustworthy, and it should be re-derived from scratch if
the question is ever asked again.

## Three findings worth keeping (verified, independent of this experiment)

These came out of the inspection phase and matter for **any** future work on
this checkpoint:

1. **`deeper_visuals/common/samples.yaml` is nine-tenths training data.** Nine
   of its ten curated scenes — including the hero scene `no11vc_9_1`, threaded
   through P1–P5 — are in the go_stanford **train** split. Harmless for
   explaining an architecture. Invalid for measuring behaviour, where a case
   drawn from `train` measures memorisation and looks identical to one that does
   not. Splits are at `/data/splits/go_stanford/{train,test}/traj_names.txt`,
   are disjoint, and cover all 3696 trajectories (2956 train / 740 test).

2. **`latest.pth` and `99.pth` are the same weights.** All 649 entries inside
   the two torch archives share CRCs. Their `md5sum`s and file sizes differ only
   because torch names the archive's internal root directory after the file
   (`latest/…` vs `99/…`). `ema_99.pth` is genuinely distinct — 639 of 649
   entries differ — and is the right default, matching NoMaD's own
   `evaluate_nomad`. All load with zero key mismatch at 19,049,675 parameters.
   There is no `ema_latest.pth`: `train_eval_loop.py` builds the path and prints
   "Saved EMA model to …" but never calls `torch.save` (upstream bug).

3. **The config of record is `training-setup` (32ef15e).** `train/config/nomad.yaml`
   reached its current form in 7531ac9 on 2026-06-13 09:48; the run
   `nomad_2026_06_13_18_04_23` started eight hours later. The only later change
   touched `wandb_entity` alone. The run directory itself archives **no**
   config.yaml, so this date check is the provenance. `main` is pristine upstream
   and lacks the Dockerfile, the dataset config, and the GoStanford2 data-format
   fixes (`yaw_rotmat` squeeze, position/yaw float64 coercion) without which
   `to_local_coords` builds wrong rotation matrices.

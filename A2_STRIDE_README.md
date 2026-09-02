# Ablation A2 — context frame **stride** (subsampled history)

This folder is a **git worktree**, not a clone. It exists only to run Ablation A2.

| | |
|---|---|
| worktree folder | `~/projects/nomad-a2` |
| branch | `experiment/stride-ablation` |
| parent branch | `experiment/context-size-ablation` (Ablation A1) |
| container | `naz_nomad_stride_ablation` |
| host screen | `nomad_stride_ablation` |
| `project_name` | `nomad_stride_ablation` → `/outputs/nomad_stride_ablation/` |
| run names | `stride2`, `stride3` |
| GPU | `CUDA_VISIBLE_DEVICES=0` |

> **Do not touch `~/projects/nomad`.** That is the A1 working copy; A1 may still be
> training there on **GPU1** in container `naz_nomad_ctx_ablation`. A2 lives entirely in
> this folder and entirely on **GPU0**.

## What A2 tests

A1 asked *"do more context tokens help?"* by growing `context_size` (3 → 10 → 20).
A2 asks the cheaper question: **"does reaching further back in time help, for free?"**

`context_size` stays at **3** — the same token budget, the same VRAM, the same effective
batch as A1's `ctx03` arm — but the **stride between context frames** grows, so those 3
frames span a longer stretch of history:

| arm | context_size | stride | effective temporal window (`context_size × stride`) |
|---|---|---|---|
| `ctx03` (reused from A1) | 3 | 1 | 3 |
| `stride2` | 3 | 2 | 6 |
| `stride3` | 3 | 3 | 9 |

**stride 1 is not re-run.** A1's `ctx03` *is* the stride-1 arm.

### Why the reuse is valid

- `index_context_size: 20` is held in every A1 and A2 arm. The sample index deletes the
  first `20 × waypoint_spacing` frames of each trajectory, so all arms train and evaluate
  on the **identical** set of `(trajectory, curr_time)` pairs.
- The deepest look-back here is `context_size × max_stride = 3 × 3 = 9 ≤ 20`, so no arm
  runs off the start of a trajectory and no padding is needed.
- Effective batch is **256** everywhere, via microbatch 32 × grad-accum 8 — same as A1.

### The one thing that must not move

Context-frame spacing and **action/goal target spacing** are the same variable in the
stock dataloader (`waypoint_spacing`). A2 therefore adds an **independent `context_stride`
key** that changes *only* which context frames are read. Prediction targets, goal
sampling, distance labels and normalisation stay exactly as they are — otherwise the
ablation would be confounded.

## Proof that the stride knob is decoupled from the targets

The whole ablation rests on one claim: stride changes *which frames the model looks at*
and nothing else. `train/ablation/probe_context_stride.py` demonstrates it on real
samples rather than asserting it — run it inside the container from `train/`:

```
python ablation/probe_context_stride.py
```

Its output is kept at
`/outputs/nomad_stride_ablation/sweep_logs/context_stride_probe.log`. Excerpt:

```
sample index: 26648 samples
  stride 1:  same (trajectory, curr_time) set as stride 1: True
  stride 2:  same (trajectory, curr_time) set as stride 1: True
  stride 3:  same (trajectory, curr_time) set as stride 1: True
stride 1 vs pre-A2 formula: 26648 samples checked, 0 mismatches

sample 5000  traj=no1vc_17_0  curr_time=249
  stride 1:  context frames [246, 247, 248, 249]   window 3 waypoints  (4 frames)
  stride 2:  context frames [243, 245, 247, 249]   window 6 waypoints  (4 frames)
  stride 3:  context frames [240, 243, 246, 249]   window 9 waypoints  (4 frames)
  action target frames, every stride: [249, 250, 251, 252, 253, 254, 255, 256, 257]
  stride 1:  max |action - stride1 action| = 0.0e+00
  stride 2:  max |action - stride1 action| = 0.0e+00
  stride 3:  max |action - stride1 action| = 0.0e+00
```

Four claims, all checked over the whole test split or bit-exactly:

1. **Stride 1 is the baseline.** The new code selects exactly the frames the pre-A2
   dataloader selected, on all 26,648 samples — which is what makes A1's `ctx03` a
   legitimate stride-1 arm rather than a near-enough one.
2. **The context window widens.** 4 frames either way, spanning 3 / 6 / 9 waypoints.
3. **The targets do not move.** The action target frames are the same list at every
   stride, and the returned action tensors are bit-identical (max |Δ| = 0).
4. **The sample set is shared.** Same 26,648 `(trajectory, curr_time)` pairs at every
   stride — the same cached index pickle A1 built, since its filename keys on
   `index_context_size`, not on context size or stride.

The earliest sample in the split sits at `curr_time = 20`, so even stride 3 reaches back
only to frame 11: no arm runs off the start of a trajectory, and no padding is needed.

## Smoke run (2026-09-01)

`stride2` on GPU0 for ~600 optimizer steps, then stopped:

| | |
|---|---|
| batches / epoch | **3329** — identical to A1's `ctx03` |
| optimizer steps / epoch | **417** (3329 ÷ accum 8) |
| throughput | ~9.8 microbatch/s |
| peak VRAM | **3708 MiB** — as predicted, the `ctx03` footprint |
| `gc_action_loss` (train) | 41.1 → 36.1 → 28.2 → 22.4, descending |
| non-finite losses | 0 |
| GPU1 (A1) | unaffected throughout |

Its partial output directory was deleted afterwards, so `/outputs/nomad_stride_ablation/`
holds only real runs.

## Checking status

```bash
# from this folder, on the host
./train/ablation/run_stride_sweep.sh --status   # container process table + GPU + newest log

screen -ls                                      # the host screen: nomad_stride_ablation
screen -r nomad_stride_ablation                 # attach (Ctrl-A D to detach)

nvidia-smi                                      # GPU0 = A2, GPU1 = A1 — must stay separate
docker top naz_nomad_stride_ablation            # is the trainer alive?

tail -f /mnt/shared_disk/nazli/nomad_outputs/nomad_stride_ablation/sweep_logs/*.log
```

Checkpoints and logs land under `/outputs/nomad_stride_ablation/` (host:
`/mnt/shared_disk/nazli/nomad_outputs/nomad_stride_ablation/`) — **outside the repo**, so
nothing generated is ever committed.

## Aggregation (Phase 5)

`train/ablation/evaluate_sweep.py` scores both ablations in one table. It is A1's script,
extended rather than replaced: the arms are now ordered by **effective temporal window**
(`context_size × context_stride`) instead of by `context_size`, which is the axis A1 and
A2 share. A1 buys window with tokens; A2 buys it with stride.

Run it inside the container from `train/`, once both sweeps have finished:

```bash
CUDA_VISIBLE_DEVICES=0 python ablation/evaluate_sweep.py \
    --run ctx03=/outputs/nomad_ctx_ablation/ctx03_2026_08_31_17_47_18 \
    --run ctx10=/outputs/nomad_ctx_ablation/ctx10_<stamp> \
    --run ctx20=/outputs/nomad_ctx_ablation/ctx20_<stamp> \
    --run stride2=/outputs/nomad_stride_ablation/stride2_<stamp> \
    --run stride3=/outputs/nomad_stride_ablation/stride3_<stamp>
```

All five configs live in this worktree's `train/config/`, since this branch descends from
A1's, so one container scores every arm. `ctx03` is scored once and serves as both A1's
baseline and A2's stride-1 arm.

The arms sort into this order, which is the point of the whole exercise:

```
arm           context_size  context_stride   frames   window
------------------------------------------------------------
ctx03                    3               1        4        3
stride2                  3               2        4        6
stride3                  3               3        4        9
ctx10                   10               1       11       10
ctx20                   20               1       21       20
```

**`stride3` and `ctx10` land adjacent** — window 9 vs 10, reached with 4 frames vs 11.
That pair answers the question the two ablations exist to ask: *do you need the tokens,
or just the reach?* If `stride3` matches `ctx10`, the reach was doing the work and the
extra tokens were waste; if it does not, the density mattered.

### Score caching

Each arm's per-sample scores are cached to `--cache-dir`
(`/outputs/nomad_ctx_ablation/eval_cache/<arm>_ema29.npz`, shared with A1). An arm's
scores are fully determined by its checkpoint, its config and the seed, so a cached arm
is reused rather than re-scored: a failure late in a sweep costs only the arm that
failed, and re-rendering the table against a different `--baseline` is free. Pass
`--no-cache` to force re-evaluation.

The A1 arms cached by A1's own run are valid here without re-scoring, because `ctx03`,
`ctx10` and `ctx20` are stride-1 arms under either version of the script. Verified: the
cached `ctx03` reproduces a fresh stride-aware run to five decimals on every metric.

### Two correctness fixes this required

`build_test_loader` did not pass `context_stride` to the dataset — it predates the knob.
Left alone it would have scored the stride-trained checkpoints on stride-1 inputs, i.e.
on a history they were never trained to read, and the arms would have looked falsely bad.
It now follows each arm's own config, defaulting to 1 so A1's stride-less configs are
unaffected.

Second, scoring several arms in one process crashed with `MDB_BAD_RSLOT`: each arm needs
its own dataset (the observation tensor is `3 * (context_size + 1)` channels, so the arms
cannot share one), and LMDB reader slots are per-process, so the previous arm's image
cache has to be released first. `ViNT_Dataset.close()` does that, called from a `finally`
block so a failure mid-arm still releases the env.

### Reading the table

- **Marginal block**: each arm's mean ± SE over the full test split.
- **Paired block**: the per-sample difference against `--baseline` (default `ctx03`).
  Inputs are bit-identical across arms — same samples, same sampled goals and negatives,
  same denoising noise — so pairing cancels sample-to-sample variance and is far more
  sensitive than comparing two independent means.
- `~` means within 2 SE of the baseline; `better`/`worse` means beyond it.
- **n = 1 seed per arm.** The SE is test-set estimation noise, *not* seed variance. A gap
  beyond 2 SE exceeds estimation noise; it is not a seed-level significance claim.

## Aggregation (Phase 5)

Both ablations are scored by the same post-hoc script, `train/ablation/evaluate_sweep.py`,
on the full test split with a pinned seed. Because every arm in both studies shares one
sample index, `shuffle=False` and `num_workers=0` give every arm bit-identical inputs —
same samples, same sampled goals, same negatives, same denoising noise — so the
comparison is *paired*: the per-sample difference cancels sample-to-sample variance.

Run it from this worktree's container once both sweeps have finished. All five configs
live in this branch's `train/config/`, so one invocation covers both ablations:

```bash
docker exec -w /app/visualnav-transformer/train naz_nomad_stride_ablation bash -c '
CUDA_VISIBLE_DEVICES=0 python ablation/evaluate_sweep.py \
  --run ctx03=/outputs/nomad_ctx_ablation/ctx03_2026_08_31_17_47_18 \
  --run ctx10=/outputs/nomad_ctx_ablation/ctx10_2026_08_31_20_50_56 \
  --run ctx20=/outputs/nomad_ctx_ablation/ctx20_2026_09_01_04_35_29 \
  --run stride2=/outputs/nomad_stride_ablation/stride2_<stamp> \
  --run stride3=/outputs/nomad_stride_ablation/stride3_<stamp> \
  --baseline ctx03'
```

Note `ctx03_2026_08_31_17_47_18` — A1's *completed* ctx03. The `13_12_14` directory is an
aborted first attempt with only 7 checkpoints; scoring it would silently compare a
2-epoch model against 30-epoch ones.

The table is ordered by **effective temporal window** (`context_size × context_stride`),
the axis the two ablations share:

```
arm           context_size  context_stride   frames   window
------------------------------------------------------------
ctx03                    3               1        4        3
stride2                  3               2        4        6
stride3                  3               3        4        9
ctx10                   10               1       11       10
ctx20                   20               1       21       20
```

That ordering puts **stride3 next to ctx10** on purpose. They reach back almost equally
far (9 vs 10 waypoints) but stride3 does it with 4 frames where ctx10 needs 11, so the
gap between those two rows is the headline of the pair of studies: *do you need the
tokens, or just the reach?* `ctx03` is the shared baseline both are measured against.

Caveats that belong with any reading of the table:

- **n = 1 seed per arm.** The standard errors describe test-set estimation noise, not
  training-seed variance. A gap beyond 2 SE exceeds estimation noise; it is not a
  seed-level significance claim.
- Stride arms are scored *at their own stride*. `evaluate_sweep.py` reads `context_stride`
  from each arm's config, so a stride-trained checkpoint is never scored on the
  adjacent-frame history it was not trained to read.

## Removing the worktree when A2 is done

```bash
docker rm -f naz_nomad_stride_ablation      # optional: drop the container
screen -S nomad_stride_ablation -X quit     # optional: drop the screen
cd ~/projects/nomad
git worktree remove ~/projects/nomad-a2     # add --force if the tree is dirty
```

The branch `experiment/stride-ablation` survives the worktree removal; only the checkout
goes away. Results under `/outputs/nomad_stride_ablation/` are untouched by this.

# Capstone results — best-combined recipe vs stock NoMaD

## Setup

- Every number is scored by `eval_paired.py` on the same **26,648** test samples (`index_context_size = 20`), each model at its own native input recipe. Scoring is deterministic: the same checkpoint scores bit-identically twice.
- Primary metric **`gc_action_loss`** (↓). Final EMA weights throughout.
- **Improved recipe** = `context_size 3` + `context_stride 3` + `image_size [160, 120]`. **Stock recipe** = `context_size 3`, stride 1, `[96, 96]`. Both use the pinned sample index and the 32×8 effective batch of 256 — except vanilla.

## Answer

**1. Does the improved recipe beat stock NoMaD? Yes.** At a matched 100-epoch budget, on
the same pinned data and the same 32×8 batch, best-combined@100 scores **3.2871** against
clean-stock@100's **3.4723**: **Δ = −0.185 ± 0.025**, 7.3× its combined uncertainty (5.3%
lower loss). It holds against the seed-matched baseline alone (6.3×), and on the
deterministic denoising loss in both branches (14.4× goal-conditioned, 7.8× goal-masked).

**2. Do the ablation gains stack? Approximately, yes.** At 30 epochs, stride3 (−0.123)
plus img160×120 (−0.282) predict −0.406; best-combined@30 delivers −0.359 against the same
ctx03 seed. The shortfall, **+0.047 ± 0.031**, is 1.5× its uncertainty, which can't be told
apart from exact additivity. The point estimate realises 88% of the sum, so a small overlap
between the two knobs is possible but not shown.

## Findings the grid adds

**The advantage halves with training but does not vanish.** The recipe effect is −0.370
at 30 epochs and −0.185 at 100. The narrowing, +0.185 ± 0.030, is itself real (6.2×).
The stock recipe gains −0.341 from the extra 70 epochs; the improved recipe gains only
−0.157. Part of what the recipe buys is therefore *faster* learning, which the stock recipe
partly catches up on given time. A real, roughly halved gap remains at 100 epochs. Whether
it keeps narrowing past 100 is not tested here.

**The improved recipe reaches the stock result on about half the compute.**
best-combined@30 (5.7 GPU-hours) vs clean-stock@100 (10.5 GPU-hours):
**Δ = −0.029 ± 0.025**, not distinguishable. This contrast compares compute, not epochs:
it deliberately sets 30 improved-recipe epochs against 100 stock epochs, because the recipe
costs about 1.8× per epoch (11.5 vs 6.4 min at 160×120). Only the headline is epoch-matched.
So at about half the GPU-hours it is indistinguishable from clean-stock@100; at equal epochs
it is better.

**The vanilla confound, measured directly: 0.157, not 0.44.** Clean-stock@100 vs
vanilla@100 share recipe and budget. They differ only in vanilla's +26% indexed data and
its true 256 batch, and vanilla wins by **0.157 ± 0.025** (6.4×). The ~0.44 estimate from
epoch 29 overstated it for two reasons. Indexing 26% more samples per epoch meant vanilla
had also taken 26% more optimizer steps by then. And the two runs sat at different points of
their cosine schedules (T_max 100 vs 30). At matched final budget the confound is about 0.16
— still almost as large as the 0.185 recipe effect.

**That is why promoting clean-stock@100 was the right call.** Against vanilla@100 the
improved recipe scores −0.029 ± 0.029, a tie. That tie is the real −0.185 recipe gain
minus the 0.157 confound. Compared against vanilla, a clear effect would have read as no
effect.

## Path B

The plan's trigger was: *if best-combined@100 does not beat vanilla `ema_99` by a margin
clearly exceeding the ctx03 across-seed SD, run the clean stock@100.* It **fires**: the
margin is −0.029 ± 0.029. The prescribed remedy has **already been carried out**. After
Phase 2, clean-stock@100 was promoted from conditional fallback to primary baseline and run
with two seeds. Nothing is left to trigger.

## How robust the headline is to the ruler

- The ruler σ = 0.011 was measured at 30 epochs on the stock recipe. At the headline's
  own budget, clean-stock@100's two seeds land **0.0031** apart in SD — about 3.5× tighter,
  with one degree of freedom. That fits the expectation that seed spread shrinks once
  cosine has fully annealed, so the 30-epoch ruler used above is **conservative**.
- Pessimistic case: with three seeds, the 95% interval for σ reaches 0.069. At that σ the
  headline's seed noise grows to 0.085, and −0.185 still clears its combined uncertainty
  by 2.1×. The headline survives even the worst plausible seed noise, if only just.
- best-combined@100 is **one seed**, as decided (a second 160×120 seed was on the critical
  path). Its seed noise is borrowed from the stock-recipe ruler, which assumes the improved
  recipe is no more seed-sensitive than the stock one. That is an assumption, not a
  measurement.

## What the secondary metrics support

Judged against their own seed rulers (Phase 4), the headline is carried by
`gc_action_loss` and both denoising losses. On waypoint cosine similarity, goal-masked action
loss and distance loss it is **not distinguishable**. Those three metrics' seed noise is too
large for single-seed runs to resolve a gap of this size. Do not claim an improvement on
them.

## Honest-framing checklist

- [x] **Vanilla config reconstructed** — from git provenance (last edit 8 h before the run,
  data paths only) corroborated by the checkpoint's tensor shapes; `docs/capstone_audit.md` §1.
- [x] **All three vanilla-vs-baseline axes named** (epochs; +26% data; batch mechanics via
  the nonlinear `action_reduce`, *not* BatchNorm). The two extras favour vanilla, and are now
  measured together at 0.157.
- [x] **Across-seed SD reported next to paired SE** in every contrast, combined in
  quadrature. The paired SE is never presented alone as the is-it-better uncertainty.
- [x] **Ablation deltas within seed noise flagged** — every `gc_action_loss` delta clears
  the ruler. Most `uc_action_loss`, cosine and distance deltas do not;
  `docs/capstone_variance_ruler.md`.
- [x] **best-combined@30 vs vanilla@100 labelled secondary** — 3.444 vs 3.316 is
  confounded by both budget and data, and appears only in the grid, never as a contrast.
- [x] **The loss is a proxy.** `gc_action_loss` measures open-loop action prediction
  against logged trajectories. Closed-loop rollout (success rate, distance-to-goal,
  collisions) is **deferred** and not claimed.

## Limitations

- Open-loop proxy only; no closed-loop navigation evaluated.
- One dataset (`go_stanford`), one test split. Improved-recipe runs are single-seed.
- The headline is epoch-matched; the compute-matched reading rests on one contrast.
- The additivity check uses seed-0 runs only, so its seed noise is 2σ. It can rule out
  large interactions, not small ones.
- Behaviour beyond 100 epochs is unknown, and the recipe advantage was still narrowing.

## Two uncertainties, and how they are combined

| error bar | what it answers | value on `gc_action_loss` |
|---|---|---|
| **paired SE** (within-run) | could this gap be test-set sampling noise? | per contrast, ≈ 0.012–0.024 |
| **across-seed SD** (between-run), ctx03 @30, n = 3 | would a reseed land this far away? | **σ = 0.01100** — the ruler used below |
| across-seed SD, clean-stock @100, n = 2 | same, at the headline's budget | 0.00311 (1 degree of freedom — a spot check, not a ruler) |

Each contrast's **seed noise** is σ scaled by how many single-seed runs it is built from (averaging k seeds of a recipe divides that recipe's seed variance by k). Its **combined** uncertainty adds seed noise and paired SE in quadrature. A contrast is called only when it clears the combined uncertainty by 2×.

Absolute cells are shown without a marginal SE on purpose: that SE (≈ 0.03) is dominated by sample-to-sample variance every run shares, so it says nothing about whether two cells differ. Compare cells through the contrasts, not by eye.

## The grid — `gc_action_loss` ↓

| | stock recipe (ctx 3 · stride 1 · 96×96) | improved recipe (ctx 3 · stride 3 · 160×120) |
|---|---|---|
| **30 epochs** | ctx03<br>**3.8134**<br>n = 3 seeds (3.8022, 3.8242, 3.8139)<br>across-seed SD 0.0110 | best-combined@30<br>**3.4436**<br>n = 1 seed |
| **100 epochs** — primary | clean-stock@100<br>**3.4723**<br>n = 2 seeds (3.4701, 3.4745)<br>across-seed SD 0.0031 | best-combined@100<br>**3.2871**<br>n = 1 seed |
| 100 epochs — secondary, confounded | vanilla@100 (`ema_99`)<br>**3.3158**<br>n = 1 seed<br>*+26% train data, true 256 batch* | *not run — no improved-recipe counterpart on vanilla's data* |

## Contrasts — `gc_action_loss`

| contrast | Δ | paired SE | seed noise | combined | Δ / combined | verdict |
|---|---|---|---|---|---|---|
| **HEADLINE — recipe effect at 100 epochs** | -0.18524 | ±0.02151 | ±0.01347 | ±0.02538 | 7.3 | better |
| **recipe effect at 100 epochs, vs stock seed 0 only** | -0.18304 | ±0.02444 | ±0.01556 | ±0.02897 | 6.3 | better |
| **recipe effect at 30 epochs** | -0.36979 | ±0.01619 | ±0.01270 | ±0.02058 | 18.0 | better |
| **does the recipe effect shrink with training?** | +0.18455 | ±0.02321 | ±0.01852 | ±0.02969 | 6.2 | advantage narrows |
| **additivity shortfall at 30 epochs** | +0.04714 | ±0.02192 | ±0.02200 | ±0.03106 | 1.5 | not distinguishable |
| **budget effect, stock recipe** | -0.34112 | ±0.01567 | ±0.01004 | ±0.01861 | 18.3 | better |
| **budget effect, improved recipe** | -0.15657 | ±0.02192 | ±0.01556 | ±0.02688 | 5.8 | better |
| **compute check — best-combined@30 vs clean-stock@100** | -0.02867 | ±0.02115 | ±0.01347 | ±0.02508 | 1.1 | not distinguishable |
| **SECONDARY — best-combined@100 vs vanilla@100** | -0.02870 | ±0.02407 | ±0.01556 | ±0.02867 | 1.0 | not distinguishable |
| **confound measured — clean-stock@100 vs vanilla@100** | +0.15654 | ±0.02062 | ±0.01347 | ±0.02463 | 6.4 | confound favours vanilla |

- **HEADLINE — recipe effect at 100 epochs** — best-combined@100 − clean-stock@100 (2-seed mean). Same pinned data, same 32×8 batch, same budget: only the recipe differs.
- **recipe effect at 100 epochs, vs stock seed 0 only** — The same contrast against the seed-matched baseline alone, as a robustness check.
- **recipe effect at 30 epochs** — best-combined@30 − ctx03 (3-seed mean).
- **does the recipe effect shrink with training?** — (recipe effect @100) − (recipe effect @30). Positive = the advantage narrows given more epochs.
- **additivity shortfall at 30 epochs** — (bc30 − ctx03) − [(stride3 − ctx03) + (img160x120 − ctx03)] = bc30 − stride3 − img160x120 + ctx03, all seed 0. Zero = the two knobs add exactly; positive = they overlap. Four single-seed runs, so its seed noise is 2σ.
- **budget effect, stock recipe** — clean-stock@100 − ctx03@30.
- **budget effect, improved recipe** — best-combined@100 − best-combined@30.
- **compute check — best-combined@30 vs clean-stock@100** — Epoch-MISmatched on purpose: best-combined@30 took ~5.7 GPU-hours, clean-stock@100 ~10.5. Asks whether the improved recipe reaches the stock result on about half the compute.
- **SECONDARY — best-combined@100 vs vanilla@100** — Real-world reference only: vanilla trained on 26% more data with a true 256 batch. Confounded in vanilla's favour; see the confound row below.
- **confound measured — clean-stock@100 vs vanilla@100** — Same recipe and budget; differs only in index_context_size (vanilla +26% data) and batch mechanics. This measures the confound directly at matched budget.

## The headline on every metric, each against its own ruler

The Phase 4 ruler showed the secondary metrics are far more seed-sensitive than `gc_action_loss`, so each is judged against its own across-seed SD.

| metric | best-combined@100 | clean-stock@100 | Δ | own σ (ctx03) | combined | Δ / combined | verdict |
|---|---|---|---|---|---|---|---|
| `gc_diffusion_loss` ↓ | 0.03236 | 0.04076 | -0.00840 | 0.00046 | ±0.00058 | 14.4 | better |
| `gc_action_loss` ↓ | 3.28706 | 3.47230 | -0.18524 | 0.01100 | ±0.02538 | 7.3 | better |
| `gc_action_waypts_cos_sim` ↑ | 0.93572 | 0.93360 | +0.00213 | 0.00160 | ±0.00205 | 1.0 | not distinguishable |
| `uc_diffusion_loss` ↓ | 0.03522 | 0.04436 | -0.00914 | 0.00094 | ±0.00117 | 7.8 | better |
| `uc_action_loss` ↓ | 3.64767 | 3.74337 | -0.09570 | 0.10468 | ±0.13020 | 0.7 | not distinguishable |
| `uc_action_waypts_cos_sim` ↑ | 0.92862 | 0.92879 | -0.00017 | 0.00088 | ±0.00124 | 0.1 | not distinguishable |
| `gc_dist_loss` ↓ | 14.471 | 14.566 | -0.09530 | 0.67248 | ±0.82877 | 0.1 | not distinguishable |

## Runs

| run | checkpoint | sha256 | recipe |
|---|---|---|---|
| `ctx03_s0` | `ctx03_2026_08_31_17_47_18/ema_29.pth` | `c2cb5cc0b24e` | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=0 |
| `ctx03_s1` | `ctx03_s1_2026_09_09_09_45_41/ema_29.pth` | `cd4a1f74bf85` | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=1 |
| `ctx03_s2` | `ctx03_s2_2026_09_09_09_45_41/ema_29.pth` | `03a7611f82cf` | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=2 |
| `stock100_s0` | `stock100_s0_2026_09_09_12_54_12/ema_99.pth` | `e0b4fc4b125d` | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (32x8), epochs=100, lr=1e-4, seed=0 |
| `stock100_s1` | `stock100_s1_2026_09_09_23_27_46/ema_99.pth` | `027b1b463a1c` | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (32x8), epochs=100, lr=1e-4, seed=1 |
| `bc30_s0` | `bc30_s0_2026_09_09_12_54_42/ema_29.pth` | `8b7ede7521c6` | context_size=3, context_stride=3, frames=4, window=9, image_size=160x120, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=0 |
| `bc100_s0` | `bc100_s0_2026_09_09_18_39_12/ema_99.pth` | `14d5bb1c7ca7` | context_size=3, context_stride=3, frames=4, window=9, image_size=160x120, eff_batch=256 (32x8), epochs=100, lr=1e-4, seed=0 |
| `stride3_s0` | `stride3_2026_09_01_15_21_50/ema_29.pth` | `a9af7c00a5be` | context_size=3, context_stride=3, frames=4, window=9, image_size=96x96, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=0 |
| `img160x120_s0` | `img160x120_2026_09_02_16_29_26/ema_29.pth` | `c454fc15dfd0` | context_size=3, context_stride=1, frames=4, window=3, image_size=160x120, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=0 |
| `vanilla_ema99` | `nomad_2026_06_13_18_04_23/ema_99.pth` | `1cdb5cfb5af4` | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (256x1), epochs=100, lr=1e-4, seed=0 |

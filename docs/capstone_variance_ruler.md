# Phase 4 — across-seed variance ruler (ctx03, seeds 0/1/2, ema_29)

- Test samples per run: **26,648**, the aligned set (`index_context_size = 20`); identical across runs, which is what makes the paired deltas below valid.
- Reference run for the paired deltas: `ctx03_s0`.
- Produced by `ablation/report_capstone.py` from per-sample scores written by `ablation/eval_paired.py`; re-rendering does not re-score.

## Runs

| run | checkpoint | recipe |
|---|---|---|
| `ctx03_s0` | `ema_29.pth` (`c2cb5cc0b24e`) | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=0 |
| `ctx03_s1` | `ema_29.pth` (`cd4a1f74bf85`) | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=1 |
| `ctx03_s2` | `ema_29.pth` (`03a7611f82cf`) | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=2 |

## Marginal means ±SE

| metric | `ctx03_s0` | `ctx03_s1` | `ctx03_s2` |
|---|---|---|---|
| `gc_diffusion_loss` ↓ | 0.02899 ±0.00017 | 0.02945 ±0.00017 | 0.02854 ±0.00017 |
| `gc_action_loss` ↓ | 3.80219 ±0.03112 | 3.82418 ±0.03085 | 3.81391 ±0.03150 |
| `gc_action_waypts_cos_sim` ↑ | 0.92881 ±0.00086 | 0.92702 ±0.00089 | 0.92562 ±0.00092 |
| `uc_diffusion_loss` ↓ | 0.03091 ±0.00018 | 0.03139 ±0.00018 | 0.02957 ±0.00018 |
| `uc_action_loss` ↓ | 4.17445 ±0.03231 | 4.18019 ±0.03205 | 3.99607 ±0.03220 |
| `uc_action_waypts_cos_sim` ↑ | 0.92256 ±0.00088 | 0.92134 ±0.00089 | 0.92306 ±0.00092 |
| `gc_dist_loss` ↓ | 19.02107 ±0.15673 | 19.82705 ±0.15676 | 20.35652 ±0.15930 |

## Paired per-sample difference vs `ctx03_s0`

| metric | `ctx03_s1` | `ctx03_s2` |
|---|---|---|
| `gc_diffusion_loss` | +0.00047 ±0.00011 **worse** | -0.00045 ±0.00011 **better** |
| `gc_action_loss` | +0.02199 ±0.02306 ~ | +0.01172 ±0.02365 ~ |
| `gc_action_waypts_cos_sim` | -0.00179 ±0.00062 **worse** | -0.00319 ±0.00064 **worse** |
| `uc_diffusion_loss` | +0.00048 ±0.00012 **worse** | -0.00134 ±0.00012 **better** |
| `uc_action_loss` | +0.00574 ±0.02433 ~ | -0.17838 ±0.02472 **better** |
| `uc_action_waypts_cos_sim` | -0.00121 ±0.00062 ~ | +0.00050 ±0.00066 ~ |
| `gc_dist_loss` | +0.80598 ±0.11826 **worse** | +1.33545 ±0.10192 **worse** |

`~` = within 2 SE of `ctx03_s0`; **better**/**worse** = beyond it.

## Across-seed ruler — `ctx03`, 3 seeds

These runs share one recipe and differ only in `seed`. How far apart they land
is how much of any effect could be a reseed rather than the knob under test.
It is a different, and usually larger, uncertainty than the paired SE above.

| metric | per-seed means | mean | across-seed SD | range |
|---|---|---|---|---|
| `gc_diffusion_loss` | 0.02899, 0.02945, 0.02854 | 0.02899 | **0.00046** | 0.00091 |
| `gc_action_loss` | 3.80219, 3.82418, 3.81391 | 3.81342 | **0.01100** | 0.02199 |
| `gc_action_waypts_cos_sim` | 0.92881, 0.92702, 0.92562 | 0.92715 | **0.00160** | 0.00319 |
| `uc_diffusion_loss` | 0.03091, 0.03139, 0.02957 | 0.03062 | **0.00094** | 0.00182 |
| `uc_action_loss` | 4.17445, 4.18019, 3.99607 | 4.11690 | **0.10468** | 0.18412 |
| `uc_action_waypts_cos_sim` | 0.92256, 0.92134, 0.92306 | 0.92232 | **0.00088** | 0.00171 |
| `gc_dist_loss` | 19.02107, 19.82705, 20.35652 | 19.73488 | **0.67248** | 1.33545 |

With 3 seeds the SD has only 2 degrees of freedom, so it is itself a rough estimate: the 95% interval for the true sigma spans roughly 0.00572 to 0.06920 on `gc_action_loss`. Read it as an order-of-magnitude bar, and treat an effect clearing it by only a small factor as unproven.

## Effects judged against the ruler — `gc_action_loss` (SD = 0.01100)

| effect | delta | / seed SD | verdict |
|---|---|---|---|
| A1 ctx10 vs ctx03 | +0.06184 | 5.6x | exceeds the ruler |
| A1 ctx20 vs ctx03 | +0.06201 | 5.6x | exceeds the ruler |
| A2 stride2 vs ctx03 | -0.10393 | 9.4x | exceeds the ruler |
| A2 stride3 vs ctx03 | -0.12332 | 11.2x | exceeds the ruler |
| B img112 vs img096 | -0.05117 | 4.7x | exceeds the ruler |
| B img128x96 vs img096 | -0.14995 | 13.6x | exceeds the ruler |
| B img160x120 vs img096 | -0.28236 | 25.7x | exceeds the ruler |
| predicted bc30 if additive | -0.40568 | 36.9x | exceeds the ruler |

## Reading

**The ruler is tight on the primary metric, and the primary metric alone.**

`gc_action_loss` across-seed SD is **0.01100** — smaller than the *within-run paired SE* of about 0.023. Two seeds of this recipe are not even distinguishable from one another by the paired test on 26,648 samples. Training is, on this metric, remarkably reproducible, and every ablation delta measured so far clears the ruler by 4.7x to 25.7x. The plan predicted `img112`'s -0.051 would be borderline against seed noise; it is not, at 4.7x.

**The secondary metrics do not survive it.** Judged against their own across-seed SDs:

- `uc_action_loss` (SD 0.10468, 9.5x the gc SD): both stride arms and `img112` fall **within** seed noise, and `ctx10` / `img128x96` are only comparable to it. The A2 table called stride2 and stride3 "better" here on 3.3 paired SE. Against a reseed they are not distinguishable. `ctx20` and `img160x120` still clear it.
- `gc_action_waypts_cos_sim` (SD 0.00160): **nothing** clears the ruler. The largest effect, `img160x120` at +0.00312, is 2.0x — right on the line.
- `gc_dist_loss` (SD 0.67248): **nothing** clears the ruler; every A1/A2/B distance claim dissolves into seed noise.

So the write-up can carry claims on `gc_action_loss` and `gc_diffusion_loss`. Claims on the goal-masked action loss, the waypoint cosine similarity and the distance head are not supported by n = 1 seed, and the published "better"/"worse" verdicts on those rows must be restated as within-noise. This is a vindication of the plan's choice of `gc_action_loss` as primary, not a contradiction of it.

**The unequal seed sensitivity is itself a result.** The goal-masked branch is ~9.5x more seed-sensitive than the goal-conditioned one on the same runs, and `ctx03_s2`'s `uc_action_loss` of 3.996 against 4.174 / 4.180 is a genuine outlier rather than a graded spread. `goal_mask_prob: 0.5` means the exploration branch sees a random half of each batch, so it is trained on a noisier signal — a plausible mechanism, not a tested one.

**Caveat that limits all of the above.** Three seeds give the SD two degrees of freedom. At the pessimistic end of its 95% interval the true sigma could be ~0.069, at which point `stride3`'s -0.123 is 1.8 sigma and only `img160x120`'s -0.282 (4.1 sigma) stays clearly outside noise. The resolution knob is robust; the stride knob is real at the point estimate but not bulletproof.

These standard errors describe estimation noise on this test set. They are **not**
training-seed variance, and a gap beyond 2 SE here is not a claim that a rerun with
another seed would reproduce it. The across-seed ruler is built in Phase 4.

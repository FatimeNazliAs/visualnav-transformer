# Ablation A1 — NoMaD temporal context (context_size sweep)

- Test samples per arm: **26,648** (identical inputs across arms, so the paired deltas below are valid)
- Checkpoint: `ema_29.pth` · baseline arm: `ctx03` · **n = 1 seed per arm**

## Arms

| arm | config |
|---|---|
| `ctx03` | context_size=3, index_context_size=20, image_size=96x96, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=0 |
| `ctx10` | context_size=10, index_context_size=20, image_size=96x96, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=0 |
| `ctx20` | context_size=20, index_context_size=20, image_size=96x96, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=0 |

## Marginal means ±SE

| metric | `ctx03` | `ctx10` | `ctx20` |
|---|---|---|---|
| `gc_action_loss` ↓ | 3.80219 ±0.03112 | 3.86403 ±0.03116 | 3.86419 ±0.03148 |
| `gc_action_waypts_cos_sim` ↑ | 0.92881 ±0.00086 | 0.93100 ±0.00080 | 0.93083 ±0.00081 |
| `uc_action_loss` ↓ | 4.17444 ±0.03231 | 4.32786 ±0.03243 | 4.53490 ±0.03323 |
| `uc_action_waypts_cos_sim` ↑ | 0.92256 ±0.00088 | 0.92264 ±0.00082 | 0.91937 ±0.00082 |
| `gc_dist_loss` ↓ | 19.02107 ±0.15673 | 19.52963 ±0.16517 | 19.94427 ±0.16814 |

## Paired per-sample difference vs `ctx03`

| metric | `ctx10` | `ctx20` |
|---|---|---|
| `gc_action_loss` | +0.06184 ±0.02319 **worse** | +0.06201 ±0.02351 **worse** |
| `gc_action_waypts_cos_sim` | +0.00219 ±0.00058 **better** | +0.00202 ±0.00059 **better** |
| `uc_action_loss` | +0.15341 ±0.02534 **worse** | +0.36046 ±0.02598 **worse** |
| `uc_action_waypts_cos_sim` | +0.00008 ±0.00062 ~ | -0.00318 ±0.00063 **worse** |
| `gc_dist_loss` | +0.50856 ±0.07538 **worse** | +0.92320 ±0.08768 **worse** |

`~` = within 2 SE of the baseline; **better**/**worse** = beyond 2 SE.

## Finding

On gc_action_loss vs ctx03, more context HURTS (ctx10 +0.06184±0.02319, ctx20 +0.06201±0.02351); n = 1 seed, so these SEs are test-set estimation noise, not seed variance.

Standard errors describe estimation noise on this test set, not training-seed
variance. With n = 1 seed per arm, a gap larger than 2 SE means it exceeds
test-set estimation noise; it is not a seed-level significance claim.

Regenerate with `ablation/evaluate_sweep.py --write-results <dir>` (reads the cached per-sample scores; does not re-score).

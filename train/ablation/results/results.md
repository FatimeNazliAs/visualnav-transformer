# Ablation B — NoMaD input resolution (image_size sweep)

- Test samples per arm: **26,648** (identical inputs across arms, so the paired deltas below are valid)
- Checkpoint: `ema_29.pth` · baseline arm: `img096` · **n = 1 seed per arm**

## Arms

| arm | config |
|---|---|
| `img096` | context_size=3, index_context_size=20, image_size=96x96, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=0 |
| `img112` | context_size=3, index_context_size=20, image_size=112x112, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=0 |
| `img128x96` | context_size=3, index_context_size=20, image_size=128x96, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=0 |
| `img160x120` | context_size=3, index_context_size=20, image_size=160x120, eff_batch=256 (32x8), epochs=30, lr=1e-4, seed=0 |

## Marginal means ±SE

| metric | `img096` | `img112` | `img128x96` | `img160x120` |
|---|---|---|---|---|
| `gc_action_loss` ↓ | 3.80219 ±0.03112 | 3.75101 ±0.03119 | 3.65224 ±0.03048 | 3.51983 ±0.03065 |
| `gc_action_waypts_cos_sim` ↑ | 0.92881 ±0.00086 | 0.92852 ±0.00088 | 0.92941 ±0.00089 | 0.93193 ±0.00088 |
| `uc_action_loss` ↓ | 4.17444 ±0.03231 | 4.11393 ±0.03258 | 4.01944 ±0.03221 | 3.86824 ±0.03216 |
| `uc_action_waypts_cos_sim` ↑ | 0.92256 ±0.00088 | 0.92289 ±0.00089 | 0.92329 ±0.00091 | 0.92627 ±0.00089 |
| `gc_dist_loss` ↓ | 19.02107 ±0.15673 | 18.49414 ±0.15447 | 18.73432 ±0.15448 | 18.55347 ±0.15670 |

## Paired per-sample difference vs `img096`

| metric | `img112` | `img128x96` | `img160x120` |
|---|---|---|---|
| `gc_action_loss` | -0.05117 ±0.02127 **better** | -0.14995 ±0.01992 **better** | -0.28236 ±0.02067 **better** |
| `gc_action_waypts_cos_sim` | -0.00029 ±0.00058 ~ | +0.00060 ±0.00056 ~ | +0.00312 ±0.00058 **better** |
| `uc_action_loss` | -0.06052 ±0.02244 **better** | -0.15501 ±0.02164 **better** | -0.30621 ±0.02231 **better** |
| `uc_action_waypts_cos_sim` | +0.00034 ±0.00058 ~ | +0.00073 ±0.00057 ~ | +0.00371 ±0.00059 **better** |
| `gc_dist_loss` | -0.52693 ±0.06813 **better** | -0.28675 ±0.06628 **better** | -0.46760 ±0.07145 **better** |

`~` = within 2 SE of the baseline; **better**/**worse** = beyond 2 SE.

## Contrasts

**`img112` vs `img096`** — resolution only, aspect held 1:1 (9,216 -> 12,544 px; both pool to the same 3x3 encoder grid)

| metric | paired delta ±SE | |
|---|---|---|
| `gc_action_loss` | -0.05117 ±0.02127 | **better** (2.4 SE) |
| `gc_action_waypts_cos_sim` | -0.00029 ±0.00058 | ~ (0.5 SE) |
| `uc_action_loss` | -0.06052 ±0.02244 | **better** (2.7 SE) |
| `uc_action_waypts_cos_sim` | +0.00034 ±0.00058 | ~ (0.6 SE) |
| `gc_dist_loss` | -0.52693 ±0.06813 | **better** (7.7 SE) |

**`img128x96` vs `img112`** — ASPECT only, pixels matched within 2% (12,544 px at 1:1 vs 12,288 px at 4:3)

| metric | paired delta ±SE | |
|---|---|---|
| `gc_action_loss` | -0.09878 ±0.02013 | **better** (4.9 SE) |
| `gc_action_waypts_cos_sim` | +0.00090 ±0.00055 | ~ (1.6 SE) |
| `uc_action_loss` | -0.09449 ±0.02171 | **better** (4.4 SE) |
| `uc_action_waypts_cos_sim` | +0.00040 ±0.00056 | ~ (0.7 SE) |
| `gc_dist_loss` | +0.24018 ±0.06843 | **worse** (3.5 SE) |

**`img160x120` vs `img128x96`** — RESOLUTION only, aspect held 4:3 (12,288 -> 19,200 px)

| metric | paired delta ±SE | |
|---|---|---|
| `gc_action_loss` | -0.13241 ±0.01780 | **better** (7.4 SE) |
| `gc_action_waypts_cos_sim` | +0.00252 ±0.00051 | **better** (5.0 SE) |
| `uc_action_loss` | -0.15120 ±0.01963 | **better** (7.7 SE) |
| `uc_action_waypts_cos_sim` | +0.00298 ±0.00051 | **better** (5.8 SE) |
| `gc_dist_loss` | -0.18084 ±0.06307 | **better** (2.9 SE) |

## Finding

On gc_action_loss vs img096, higher input resolution HELPS (img112 -0.05117±0.02127, img128x96 -0.14995±0.01992, img160x120 -0.28236±0.02067); n = 1 seed, so these SEs are test-set estimation noise, not seed variance.

Standard errors describe estimation noise on this test set, not training-seed
variance. With n = 1 seed per arm, a gap larger than 2 SE means it exceeds
test-set estimation noise; it is not a seed-level significance claim.

Regenerate with `ablation/evaluate_sweep.py --write-results <dir>` (reads the cached per-sample scores; does not re-score).

# Phase 2 — vanilla NoMaD training trajectory (eval only, never retrained)

- Test samples per run: **26,648**, the aligned set (`index_context_size = 20`); identical across runs, which is what makes the paired deltas below valid.
- Reference run for the paired deltas: `vanilla_ema99`.
- Produced by `ablation/report_capstone.py` from per-sample scores written by `ablation/eval_paired.py`; re-rendering does not re-score.

## Runs

| run | checkpoint | recipe |
|---|---|---|
| `vanilla_ema9` | `ema_9.pth` (`eb35b524f6a7`) | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (256x1), epochs=100, lr=1e-4, seed=0 |
| `vanilla_ema19` | `ema_19.pth` (`9540eb4d9a67`) | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (256x1), epochs=100, lr=1e-4, seed=0 |
| `vanilla_ema29` | `ema_29.pth` (`c7dcf38c01dd`) | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (256x1), epochs=100, lr=1e-4, seed=0 |
| `vanilla_ema39` | `ema_39.pth` (`88be1206ca80`) | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (256x1), epochs=100, lr=1e-4, seed=0 |
| `vanilla_ema49` | `ema_49.pth` (`f1f21c72d91e`) | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (256x1), epochs=100, lr=1e-4, seed=0 |
| `vanilla_ema59` | `ema_59.pth` (`9022121b2695`) | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (256x1), epochs=100, lr=1e-4, seed=0 |
| `vanilla_ema69` | `ema_69.pth` (`ef98b68add94`) | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (256x1), epochs=100, lr=1e-4, seed=0 |
| `vanilla_ema79` | `ema_79.pth` (`1aac3605b596`) | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (256x1), epochs=100, lr=1e-4, seed=0 |
| `vanilla_ema89` | `ema_89.pth` (`47e0b0fad2cf`) | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (256x1), epochs=100, lr=1e-4, seed=0 |
| `vanilla_ema99` | `ema_99.pth` (`1cdb5cfb5af4`) | context_size=3, context_stride=1, frames=4, window=3, image_size=96x96, eff_batch=256 (256x1), epochs=100, lr=1e-4, seed=0 |

## Marginal means ±SE

| metric | `vanilla_ema9` | `vanilla_ema19` | `vanilla_ema29` | `vanilla_ema39` | `vanilla_ema49` | `vanilla_ema59` | `vanilla_ema69` | `vanilla_ema79` | `vanilla_ema89` | `vanilla_ema99` |
|---|---|---|---|---|---|---|---|---|---|---|
| `gc_diffusion_loss` ↓ | 0.02541 ±0.00014 | 0.02530 ±0.00015 | 0.03045 ±0.00019 | 0.03580 ±0.00023 | 0.04024 ±0.00025 | 0.04133 ±0.00026 | 0.04151 ±0.00026 | 0.04020 ±0.00025 | 0.03832 ±0.00025 | 0.03740 ±0.00024 |
| `gc_action_loss` ↓ | 3.79295 ±0.02997 | 3.35679 ±0.02935 | 3.36381 ±0.02944 | 3.35883 ±0.02888 | 3.36953 ±0.02862 | 3.37950 ±0.02853 | 3.38540 ±0.02852 | 3.37070 ±0.02876 | 3.33735 ±0.02899 | 3.31576 ±0.02899 |
| `gc_action_waypts_cos_sim` ↑ | 0.92824 ±0.00087 | 0.93429 ±0.00088 | 0.93624 ±0.00084 | 0.93567 ±0.00084 | 0.93574 ±0.00083 | 0.93546 ±0.00083 | 0.93536 ±0.00083 | 0.93569 ±0.00083 | 0.93645 ±0.00083 | 0.93725 ±0.00082 |
| `uc_diffusion_loss` ↓ | 0.02612 ±0.00014 | 0.02718 ±0.00016 | 0.03288 ±0.00020 | 0.03884 ±0.00024 | 0.04346 ±0.00027 | 0.04460 ±0.00028 | 0.04480 ±0.00028 | 0.04352 ±0.00027 | 0.04157 ±0.00026 | 0.04064 ±0.00026 |
| `uc_action_loss` ↓ | 4.07336 ±0.03082 | 3.79609 ±0.03127 | 3.74924 ±0.03130 | 3.66562 ±0.03039 | 3.64216 ±0.02994 | 3.64195 ±0.02983 | 3.63826 ±0.02980 | 3.64107 ±0.03000 | 3.63339 ±0.03041 | 3.63833 ±0.03067 |
| `uc_action_waypts_cos_sim` ↑ | 0.92324 ±0.00089 | 0.92626 ±0.00092 | 0.92858 ±0.00087 | 0.92946 ±0.00086 | 0.92995 ±0.00085 | 0.92996 ±0.00085 | 0.93009 ±0.00085 | 0.93008 ±0.00085 | 0.93047 ±0.00085 | 0.93102 ±0.00084 |
| `gc_dist_loss` ↓ | 21.35482 ±0.15711 | 16.56909 ±0.14817 | 15.25767 ±0.14759 | 14.86993 ±0.14935 | 14.80899 ±0.15115 | 14.79432 ±0.15128 | 14.76573 ±0.15170 | 14.42709 ±0.15063 | 14.06002 ±0.14973 | 13.73164 ±0.14974 |

## Paired per-sample difference vs `vanilla_ema99`

| metric | `vanilla_ema9` | `vanilla_ema19` | `vanilla_ema29` | `vanilla_ema39` | `vanilla_ema49` | `vanilla_ema59` | `vanilla_ema69` | `vanilla_ema79` | `vanilla_ema89` |
|---|---|---|---|---|---|---|---|---|---|
| `gc_diffusion_loss` | -0.01199 ±0.00019 **better** | -0.01210 ±0.00016 **better** | -0.00696 ±0.00011 **better** | -0.00160 ±0.00009 **better** | +0.00283 ±0.00010 **worse** | +0.00392 ±0.00010 **worse** | +0.00410 ±0.00010 **worse** | +0.00279 ±0.00009 **worse** | +0.00091 ±0.00006 **worse** |
| `gc_action_loss` | +0.47719 ±0.02452 **worse** | +0.04103 ±0.01903 **worse** | +0.04805 ±0.01449 **worse** | +0.04307 ±0.01281 **worse** | +0.05377 ±0.01249 **worse** | +0.06374 ±0.01250 **worse** | +0.06964 ±0.01213 **worse** | +0.05494 ±0.01112 **worse** | +0.02160 ±0.00855 **worse** |
| `gc_action_waypts_cos_sim` | -0.00901 ±0.00067 **worse** | -0.00296 ±0.00058 **worse** | -0.00101 ±0.00045 **worse** | -0.00158 ±0.00042 **worse** | -0.00151 ±0.00041 **worse** | -0.00179 ±0.00040 **worse** | -0.00189 ±0.00040 **worse** | -0.00156 ±0.00038 **worse** | -0.00080 ±0.00030 **worse** |
| `uc_diffusion_loss` | -0.01452 ±0.00020 **better** | -0.01346 ±0.00016 **better** | -0.00776 ±0.00012 **better** | -0.00180 ±0.00010 **better** | +0.00282 ±0.00011 **worse** | +0.00396 ±0.00011 **worse** | +0.00416 ±0.00011 **worse** | +0.00288 ±0.00009 **worse** | +0.00093 ±0.00006 **worse** |
| `uc_action_loss` | +0.43504 ±0.02477 **worse** | +0.15777 ±0.01971 **worse** | +0.11091 ±0.01542 **worse** | +0.02729 ±0.01317 **worse** | +0.00384 ±0.01282 ~ | +0.00362 ±0.01281 ~ | -0.00007 ±0.01244 ~ | +0.00274 ±0.01152 ~ | -0.00494 ±0.00893 ~ |
| `uc_action_waypts_cos_sim` | -0.00778 ±0.00067 **worse** | -0.00476 ±0.00060 **worse** | -0.00244 ±0.00046 **worse** | -0.00156 ±0.00041 **worse** | -0.00107 ±0.00041 **worse** | -0.00107 ±0.00041 **worse** | -0.00093 ±0.00040 **worse** | -0.00095 ±0.00038 **worse** | -0.00056 ±0.00031 ~ |
| `gc_dist_loss` | +7.62318 ±0.15510 **worse** | +2.83745 ±0.10375 **worse** | +1.52603 ±0.07563 **worse** | +1.13829 ±0.06297 **worse** | +1.07735 ±0.05919 **worse** | +1.06268 ±0.05854 **worse** | +1.03408 ±0.05582 **worse** | +0.69545 ±0.04718 **worse** | +0.32838 ±0.03369 **worse** |

`~` = within 2 SE of `vanilla_ema99`; **better**/**worse** = beyond it.

## Reading

On the primary metric `gc_action_loss`, **`ema_99` is the best of all ten checkpoints**, and every other one is worse than it beyond 2 SE. It is therefore the fair, strongest-possible vanilla reference for the capstone — using it makes beating vanilla harder, not easier.

The curve behind that number is **not** a clean descent, and the two loss families disagree about what happened after epoch 19:

- `gc_action_loss` drops sharply to epoch 19 (3.793 -> 3.357), then sits on a plateau with a small hump (up to 3.385 at epoch 69) before descending to its minimum at epoch 99 (3.316). The whole plateau spans 0.07 loss units — less than half the gap the resolution ablation alone opened at 30 epochs.
- `gc_diffusion_loss`, the training objective itself measured on held-out data, is **lowest at epoch 19** (0.02530) and rises steadily to 0.04151 at epoch 69 before annealing back to 0.03740 at epoch 99 — still 48% above its epoch-19 value. That is a textbook overfitting signature, partially masked by the cosine schedule late in training.

The two are not in conflict: the denoising loss measures per-timestep noise prediction, while `gc_action_loss` measures the trajectory that falls out of all ten reverse steps. A model can predict noise slightly worse and still sample slightly better. But it does mean **the last 80 epochs of vanilla training bought very little**: epoch 19 is already within 0.041 +-0.019 of epoch 99 on the primary metric, and is strictly better on the training objective.

`gc_dist_loss` is the one metric that improves monotonically throughout (21.35 -> 13.73). The distance head carries weight `alpha = 1e-4` in the total loss, so it is essentially free to keep improving while the diffusion head overfits.

These standard errors describe estimation noise on this test set. They are **not**
training-seed variance, and a gap beyond 2 SE here is not a claim that a rerun with
another seed would reproduce it. The across-seed ruler is built in Phase 4.

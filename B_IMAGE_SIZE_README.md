# Ablation B — Input Resolution (`image_size`)

This file documents **this worktree only**. The upstream fork's own docs stay in `README.md`.

## What this is

An isolated git worktree for **Ablation B**: does feeding NoMaD *sharper input frames*
improve it? A1 (`context_size`) and A2 (context stride) changed how much **time** the
model sees; B changes how much **detail** it sees **within each frame**. Nothing else moves.

| | |
|---|---|
| worktree | `~/projects/nomad-b` |
| branch | `experiment/image-size-ablation` |
| based on | `experiment/context-size-ablation` (A1, committed) |
| container | `naz_nomad_img_ablation` |
| host screen | `nomad_img_ablation` |
| `project_name` | `nomad_img_ablation` → `/outputs/nomad_img_ablation/` |
| configs | `train/config/nomad_<run>.yaml` (e.g. run `img120` → `nomad_img120.yaml`) |

B branches off A1 rather than `main` so it inherits A1's gradient accumulation and
`index_context_size` pinning. `image_size` is orthogonal to both.

**Do not touch `~/projects/nomad` (A1) or `~/projects/nomad-a2` (A2) from here.**
This worktree is the only folder B writes to.

## The 160×120 ceiling — why there is no true 160×160 arm

Every one of the **198,126 GoStanford JPGs is stored 160×120** (verified, not assumed).
That is the hard ceiling on real detail:

- **120 pixels on the short side, 160 on the wide side.** Above that the pipeline is
  inventing pixels by interpolation, not revealing detail.
- A true **`160×160` arm would upsample the vertical 120 → 160**. Any "improvement" it
  showed would be an artifact of the resampling filter, not of resolution. **Excluded.**
- `128×128` mildly upsamples the vertical (120 → 128, ~7%) and is therefore also not a
  clean "real detail" point.

**Aspect-ratio confound.** The baseline `image_size: [96, 96]` is *square*, so the
dataloader squishes the native 4:3 frame to 1:1 (`resize_and_aspect_crop` center-crops to
4:3 — a no-op on already-4:3 GoStanford — then resizes anisotropically). So a naïve
"native `[160, 120]` vs `[96, 96]`" comparison moves **resolution and aspect ratio at the
same time** and cannot attribute the result to either. Any arm set must either stay square
or carry a control that separates the two.

## The arms

| arm | `image_size` | aspect | encoder grid | pixels | peak VRAM @ micro 32 | est. 30 ep |
|---|---|---|---|---|---|---|
| `img096` | `[96, 96]` | 1:1 | 3×3 | 9,216 (1.00×) | 2.98 GiB | **reused from A1 `ctx03`** |
| `img112` | `[112, 112]` | 1:1 | 3×3 | 12,544 (1.36×) | 3.81 GiB | ~4h05m |
| `img128x96` | `[128, 96]` | 4:3 | 3×4 | 12,288 (1.33×) | 3.83 GiB | ~3h58m |
| `img160x120` | `[160, 120]` | 4:3 | 3×5 | 19,200 (2.08×) | 5.55 GiB | ~5h52m |

Each arm answers a specific question, and the set is deliberately tight:

- **`img112` is not a redundant square point.** It is the *same-grid control* (96 and 112
  both pool to a 3×3 feature map, so a flat 096→112 result localises the bottleneck at the
  encoder rather than at the input), and it is the *pixel-match anchor* for `img128x96`.
- **`img128x96` is the aspect-only control.** It is pixel-matched to `img112` to within 2%
  (12,288 vs 12,544) but carries the native 4:3 aspect. If the two land together, aspect is
  not the story and any `img160x120` gain is real detail; if `img128x96` alone beats
  `img112`, the win is un-squishing, not resolution.
- **`img160x120` is the headline.** Native aspect at the source resolution, and the only
  arm that applies **no resampling at all** — the crop is a no-op on already-4:3 frames and
  the resize is identity.

A square `120×120` arm was considered and dropped: it shares `img112`'s 3×3 grid, so it
adds a third point to the resolution axis without adding a control, and `img160x120`
already carries that axis to the ceiling.

## Held fixed across every arm

`context_size: 3` · `index_context_size: 20` · `goal_mask_prob` · diffusion steps and
scheduler · optimizer, LR and schedule, weight decay · **30 epochs, cosine `T_max = 30`** ·
dataset and splits · seed · **effective batch 256** (microbatch × grad-accum chosen per size
to fit 24 GB; the *effective* batch never floats).

Holding `context_size: 3` and `index_context_size: 20` means the `img096` arm is
config-identical to A1's `ctx03`. **`img096` reuses `ctx03_2026_08_31_17_47_18` — it is
not re-run.**

## Scoring the sweep

`evaluate_sweep.py` recovers each arm's config from the **run-directory basename**
(`basename.split("_")[0]` → `nomad_<arm>.yaml`), and builds a fresh test loader per arm, so
per-arm `image_size` is honoured with no edit to the eval script.

The baseline arm is labelled `img096` but points at A1's `ctx03` run directory. That is
deliberate: the basename resolves to `nomad_ctx03.yaml`, which already carries
`image_size: [96, 96]`, `context_size: 3` and `index_context_size: 20` — the exact baseline
config. **There is no `nomad_img096.yaml`**, precisely so there is no second file to drift
out of sync with `nomad_ctx03.yaml`.

```bash
CUDA_VISIBLE_DEVICES=1 python ablation/evaluate_sweep.py --baseline img096 \
    --cache-dir /outputs/nomad_img_ablation/eval_cache \
    --run img096=/outputs/nomad_ctx_ablation/ctx03_2026_08_31_17_47_18 \
    --run img112=/outputs/nomad_img_ablation/img112_<stamp> \
    --run img128x96=/outputs/nomad_img_ablation/img128x96_<stamp> \
    --run img160x120=/outputs/nomad_img_ablation/img160x120_<stamp>
```

Note `--cache-dir`: it defaults to `/outputs/nomad_ctx_ablation/eval_cache` (inherited
from A1), so **pass B's own path** or B's per-sample scores land in A1's cache directory.

Scoring four arms in one process is what surfaced the LMDB reader-slot crash
(`lmdb.BadRslotError: MDB_BAD_RSLOT`). Two guards are in place: `close_dataset` in
`evaluate_sweep.py` (cherry-picked from A1's `b90f1ee`, the primary fix) and `lock=False`
in `_build_caches` (a second, independent guard). They do not conflict; drop the
`lock=False` guard once one clean end-to-end aggregate proves the explicit close suffices.

Verify the baseline has not moved with:

```bash
CUDA_VISIBLE_DEVICES=1 python ablation/check_image_size_equivalence.py
```

## Status

```bash
./train/ablation/run_sweep.sh --status
```

Before launching anything, check that no A1/A2 job is still using a GPU:

```bash
nvidia-smi
```

## Cleanup

```bash
git worktree remove ~/projects/nomad-b
```

Add `--force` only if the worktree has uncommitted changes you intend to discard.
Removing the worktree does not delete the branch (`git branch -d experiment/image-size-ablation`)
or anything under `/outputs/nomad_img_ablation/`.

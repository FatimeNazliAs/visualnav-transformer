# BASELINE.md

`baseline` is the shared foundation every experiment in this fork branches from:
NoMaD training on go_stanford, plus the evaluation tooling to compare runs fairly.
`main` stays the untouched upstream repo.

## What baseline contains

| Piece | What it gives you | Commit(s) |
|---|---|---|
| Training setup | GoStanford2 data-format fix, `nomad.Dockerfile`, `huggingface_hub==0.23.0` pin, personal wandb entity, `train/logs` ignored | 7531ac9, 6d6d443, 7368e05, bd3d0db, 32ef15e (= tag `baseline-v0`) |
| Config-driven context | `context_size` and gradient-accumulation batch splitting read from config; A1 configs `nomad_ctx03/10/20.yaml`; sweep scripts | 630e360, f012ab8 |
| Context stride | `context_stride` option (spacing between context frames); defaults to 1 when a config omits it (`train/train.py`), so `nomad.yaml` trains exactly as upstream; A2 configs `nomad_stride2/3.yaml` | e77738e |
| Per-arm evaluation | `evaluate_sweep.py` scores each run on its own stride; `ViNT_Dataset.close()` releases the LMDB cache between arms; per-arm `.npz` score cache | 1a6d313 |
| A1 + A2 results | `train/ablation/results/results.{csv,md}` — ctx03, stride2, stride3, ctx10, ctx20 | 470286c |
| Paired evaluator | `train/ablation/eval_paired.py`: deterministic per-sample scoring (fixed seeds and noise), identical samples across runs | 4422349 |
| Context-size history | `experiment/context-size-ablation` merged for ancestry only — its content (b90f1ee LMDB fix, f275375 A1 results) was already in 1a6d313 / 470286c, so the merge changes no files | 3bc7b5d |

## Rules

1. `main` never changes (only pulls from upstream if ever needed).
2. Every new experiment is one branch from `baseline`, one commit per phase (phase tags optional).
3. Anything reusable a task produces (tool, fix, config) is promoted into `baseline` as its own deliberate commit or merge, and listed below.
4. A finished task gets a tag `archive/<task>` and its branch is deleted.
5. Parallel runs: one worktree folder per active branch (e.g. `nomad-clip`).

## Promoted into baseline

_Nothing since the 2026-09-23 restructure._ Add one line per promotion: what, from which task, commit hash.

# deeper_visuals

A phase-by-phase visual explanation of the NoMaD architecture. Each phase ships
two surfaces: an **advisor webpage** (published to a fixed Claude Code artifact
URL) and a **technical Notion page group**. No retraining — every figure comes
from a forward pass on an already-trained checkpoint.

Phases: **P0** overview (static) · **P1** inputs · **P2** encoders ·
**P3** transformer & goal masking · **P4** diffusion policy ·
**P5** output & multimodality.

> This folder is a copy-and-adapt of `debug_visuals/`, not an extension of it.
> It never imports from that folder. `debug_visuals/` backs an already-delivered
> presentation and stays frozen.

## Layout

```
deeper_visuals/
  common/                  shared library — phases are thin slices over this
    settings.py            container paths + training hyperparameters
    samples.yaml           the 10 curated go_stanford scenes
    config.py              config.yaml -> PhaseConfig (scene + weights resolved)
    data.py                frame loading + ImageNet normalisation
    model.py               build NoMaD and load a checkpoint onto it
    facts.py               facts.json read/write — the run_model <-> build_page seam
    viz.py                 save_fig + the shared figure palette
    build_page.py          facts.json + PNGs -> latest.html (never loads torch)
    smoke_test.py          prove the checkpoint + forward pass still work
    update.sh              the driver every phase's update.sh execs
    template/              page.html + style.css, and the per-phase stubs
  pN_name/                 one folder per phase (created by that phase)
    config.yaml            sample + checkpoint
    page.yaml              the advisor-facing words
    run_model.py           forward pass -> facts.json + PNGs
    update.sh              4-line stub -> common/update.sh
  out/                     generated, gitignored
    pN/<tag>/              facts.json, *.png, latest.html per checkpoint
    pN/latest.html         the promoted page the preview server serves
```

## Run environment

```bash
# On the host
screen -S nomad_deeper_viz

docker run --gpus all --shm-size=8g -it --name naz_nomad_deeper_viz \
  -v /mnt/shared_disk/nazli/nomad_data:/data \
  -v /mnt/shared_disk/nazli/nomad_outputs:/outputs \
  -v /home/nazli/projects/nomad:/app/visualnav-transformer \
  -v /mnt/shared_disk/nazli/wandb_config:/root/.config/wandb \
  -p 8001:8001 \
  nomad:latest bash

# Inside the container, every fresh shell
source /opt/conda/etc/profile.d/conda.sh && conda activate vint_train
cd /app/visualnav-transformer
```

Data lives at `/data/raw/go_stanford/go_stanford`; training runs at
`/outputs/nomad/`.

**Preview port is 8001, not 8000.** On this host `:8000` is held by the NWM
walkthrough's `http.server`. Override with `--port` or `PORT=` if that changes.

## The per-phase loop

1. Edit `deeper_visuals/pN_name/config.yaml` — `sample:` (a key from
   `common/samples.yaml`) and `checkpoint:` (`ema` by default, or `latest`).
2. Run `./deeper_visuals/pN_name/update.sh`.
   Add `--page-only` to skip the forward pass when only wording or layout
   changed — no GPU needed, sub-second.
3. Preview at `http://localhost:8001/pN/latest.html` (VS Code forwards the port).
4. Republish: in that phase's Claude Code chat, say
   `republish pN/latest.html -> <the phase's fixed artifact URL>`.
   The URL never changes, so the advisor's link keeps working.

Starting a new phase: copy `common/template/update.sh.stub` and
`common/template/page.yaml.stub` into the phase folder as `update.sh` and
`page.yaml`.

## Checkpoint

Settled — not something any phase needs to re-decide.

| `checkpoint:` | File | Use |
| --- | --- | --- |
| `ema` *(default)* | `ema_99.pth` | EMA weights at the final epoch — cleaner figures |
| `latest` | `latest.pth` | raw epoch-99 weights — fallback |

Both live in `/outputs/nomad/nomad_2026_06_13_18_04_23/`, a verified full
**100/100-epoch** run (`train/config/nomad.yaml` sets `epochs: 100`; the folder
holds `0.pth`–`99.pth`, `ema_0.pth`–`ema_99.pth` and `latest.pth` from a clean
~14 h run). The 15 aborted same-day runs have been deleted — this is the only
run on disk.

EMA is the default because NoMaD's own `evaluate_nomad` runs on the EMA weights
and they give cleaner trajectories in advisor figures. Switching a phase to raw
weights is a one-word edit in its `config.yaml`.

Two implementation details, so they are not rediscovered the hard way:

1. `latest.pth` is **not** a distinct checkpoint — it is the same tensors as
   `99.pth`. `train_eval_loop.py` saves `model.state_dict()` to `{epoch}.pth`
   and `latest.pth` back-to-back.
2. There is **no `ema_latest.pth`** to point at. `train_eval_loop.py` builds
   that path and prints `Saved EMA model to …` but never calls `torch.save` on
   it (upstream bug). That is why `common/settings.py` names the numbered file.

## Smoke test

```bash
python -m deeper_visuals.common.smoke_test
python -m deeper_visuals.common.smoke_test --sample left_turn --checkpoint ema
```

Loads the checkpoint and pushes one scene through encoders → transformer →
context vector → diffusion, printing shapes at each boundary. It fails loudly on
a partial state-dict load, on goal masking that does not change c_t, and on
non-finite actions.

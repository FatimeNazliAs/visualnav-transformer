# mapmad_recog — Phase 9 recognisers (runs in `naz_mapmad_recog`)

- `models.py`: one class per model (YOLOE, YOLO-World, LLMDet, Grounding DINO, OWLv2, RF-DETR, SAM 3, Qwen3-VL,
  Molmo2), one call `detect(rgb, prompt) -> [RawDet]`; `MaskMaker` = MobileSAM for box models; the VLM prompts.
- `server.py`: one model behind a TCP socket on mapmad-net (wire format `mapmad_bridge.protocol`, shared key).
- Checkpoints: `mapmad/configs/p9_models.yaml` (pinned revisions); weights in `/model_cache` (HF_HOME there).
- Image: `mapmad/docker/Dockerfile.recog` + `requirements-recog.txt` (torch 2.6.0 for CUDA 12.4).
- The Habitat side talks to it through `mapmad_sim.recogniser.SidecarRecogniser`.

## Phase 14b-2: fine-tuned YOLOE
- `finetune.py` (in `naz_mapmad_recog`): `prep` builds the YOLO segmentation dataset from the R1 frames
  (`p14b2_recog/trainset/selection.json`), `train --variant lp|ft` fine-tunes YOLOE-11-L-seg with the 6 Phase 9 prompts
  as a fixed vocabulary (LP: class projections only; FT: every layer), `models` writes `<out>/models.yaml` = the
  Phase 9 models plus `yoloe_lp` / `yoloe_ft` entries (`weights` below the outputs root, `vocab`, `sha256`).
- `models.py` serves such an entry: weights loaded from the outputs root (sha256 checked at load), the asked category
  = its class index (`classes=[k]`), no text encoding at run time. Start a server with `--config <out>/models.yaml`.

# mapmad_recog — Phase 9 recognisers (runs in `naz_mapmad_recog`)

- `models.py`: one class per model (YOLOE, YOLO-World, LLMDet, Grounding DINO, OWLv2, RF-DETR, SAM 3, Qwen3-VL,
  Molmo2), one call `detect(rgb, prompt) -> [RawDet]`; `MaskMaker` = MobileSAM for box models; the VLM prompts.
- `server.py`: one model behind a TCP socket on mapmad-net (wire format `mapmad_bridge.protocol`, shared key).
- Checkpoints: `mapmad/configs/p9_models.yaml` (pinned revisions); weights in `/model_cache` (HF_HOME there).
- Image: `mapmad/docker/Dockerfile.recog` + `requirements-recog.txt` (torch 2.6.0 for CUDA 12.4).
- The Habitat side talks to it through `mapmad_sim.recogniser.SidecarRecogniser`.

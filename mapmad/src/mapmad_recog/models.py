"""Phase 9 recognisers (block item 5), run in naz_mapmad_recog. One class per model, one shared call:

    model = build("grounding_dino", spec, device="cuda:0"); dets = model.detect(rgb, "chair")

`rgb` = (H, W, 3) uint8 RGB; `prompt` = one category name (one category per call, as in driving). Returns RawDet
(box in pixels x0, y0, x1, y1 and/or mask (H, W) bool and/or point (x, y) in pixels, score), best score first, at
most `max_det`, score >= `min_score`. Box-only models get their masks from MobileSAM afterwards (MaskMaker), never
here. VLMs return score 1.0 (one operating point). Prompts and the VLM questions are fixed in VLM_PROMPTS
(copied into p9_rules.md).
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

VLM_PROMPTS = {
    "qwen3vl": ('Locate every {name} in the image. Answer only with a JSON list in which each element is '
                '{{"bbox_2d": [x1, y1, x2, y2], "label": "{name}"}}. If there is no {name} in the image, answer [].'),
    "molmo2": "Point to every {name} in the image. If there is no {name} in the image, say that there is none.",
}
# RF-DETR has a fixed COCO list: the asked word -> its COCO class name
COCO_NAME = {"chair": "chair", "couch": "couch", "bed": "bed", "toilet": "toilet", "tv": "tv",
             "potted plant": "potted plant"}


@dataclass
class RawDet:
    score: float
    box: Optional[np.ndarray] = None  # (4,) float x0, y0, x1, y1
    mask: Optional[np.ndarray] = None  # (H, W) bool
    point: Optional[np.ndarray] = None  # (2,) float x, y


def keep_top(dets: List[RawDet], min_score: float, max_det: int) -> List[RawDet]:
    dets = [d for d in dets if d.score >= min_score]
    return sorted(dets, key=lambda d: -d.score)[:max_det]


def clip_box(box: Any, w: int, h: int) -> np.ndarray:
    b = np.asarray(box, dtype=np.float64).reshape(4)
    return np.array([np.clip(b[0], 0, w), np.clip(b[1], 0, h), np.clip(b[2], 0, w), np.clip(b[3], 0, h)])


class Recogniser:
    """Base: holds the spec, the device and the raw-pass limits."""

    def __init__(self, spec: Dict[str, Any], cache: Path, device: str, min_score: float, max_det: int) -> None:
        self.spec, self.cache, self.device = spec, cache, device
        self.min_score, self.max_det = min_score, max_det

    def detect(self, rgb: np.ndarray, prompt: str) -> List[RawDet]:
        raise NotImplementedError


class _Ultralytics(Recogniser):
    """YOLOE (masks) and YOLO-World (boxes): one class per call, text embeddings cached per prompt."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        from ultralytics import YOLO, YOLOE

        folder = self.cache / "ultralytics"
        os.chdir(folder)  # text encoders were saved here by p9_download.py
        self.yoloe = self.spec["ultralytics"].startswith("yoloe")
        self.model = (YOLOE if self.yoloe else YOLO)(str(folder / self.spec["ultralytics"]))
        self.current: Optional[str] = None
        self.pe: Dict[str, Any] = {}

    def _set(self, prompt: str) -> None:
        if prompt == self.current:
            return
        if self.yoloe:
            if prompt not in self.pe:
                self.pe[prompt] = self.model.get_text_pe([prompt])
            self.model.set_classes([prompt], self.pe[prompt])
        else:
            self.model.set_classes([prompt])
        self.current = prompt

    def detect(self, rgb: np.ndarray, prompt: str) -> List[RawDet]:
        self._set(prompt)
        r = self.model.predict(rgb[..., ::-1], conf=self.min_score, max_det=self.max_det, device=self.device,
                               verbose=False, retina_masks=True)[0]
        h, w = rgb.shape[:2]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        masks = r.masks.data.cpu().numpy() > 0.5 if (self.yoloe and r.masks is not None) else None
        out = []
        for k in range(len(scores)):
            m = masks[k] if masks is not None and masks[k].shape == (h, w) else None
            out.append(RawDet(float(scores[k]), clip_box(boxes[k], w, h), m))
        return keep_top(out, self.min_score, self.max_det)


class _GroundingDino(Recogniser):
    """Grounding DINO-T and LLMDet-T (MM-Grounding-DINO weights) through transformers; text "<prompt>."."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(self.spec["hf"], revision=self.spec["revision"])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.spec["hf"], revision=self.spec["revision"]).to(self.device).eval()

    def detect(self, rgb: np.ndarray, prompt: str) -> List[RawDet]:
        h, w = rgb.shape[:2]
        inputs = self.proc(images=rgb, text=f"{prompt}.", return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=self.min_score,
                                                               text_threshold=0.0, target_sizes=[(h, w)])[0]
        return keep_top([RawDet(float(s), clip_box(b.cpu().numpy(), w, h))
                         for s, b in zip(res["scores"], res["boxes"])], self.min_score, self.max_det)


class _Owlv2(Recogniser):
    """OWLv2-B/16 (self-trained checkpoint); the image is padded to a square, so boxes are mapped to that square."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.proc = Owlv2Processor.from_pretrained(self.spec["hf"], revision=self.spec["revision"])
        self.model = Owlv2ForObjectDetection.from_pretrained(self.spec["hf"], revision=self.spec["revision"]).to(self.device).eval()

    def detect(self, rgb: np.ndarray, prompt: str) -> List[RawDet]:
        h, w = rgb.shape[:2]
        inputs = self.proc(text=[[prompt]], images=rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        side = max(h, w)
        res = self.proc.post_process_object_detection(outputs, threshold=self.min_score, target_sizes=[(side, side)])[0]
        dets = [RawDet(float(s), clip_box(b.cpu().numpy(), w, h)) for s, b in zip(res["scores"], res["boxes"])]
        return keep_top([d for d in dets if d.box[2] > d.box[0] and d.box[3] > d.box[1]], self.min_score, self.max_det)


class _RfDetr(Recogniser):
    """RF-DETR-M, fixed COCO list: keeps only the detections of the asked category's COCO class."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        import rfdetr

        folder = self.cache / "rfdetr"
        os.chdir(folder)
        self.model = {"medium": rfdetr.RFDETRMedium}[self.spec["rfdetr"]](device=self.device.split(":")[0])
        names = self.model.class_names
        self.class_of = {v: int(k) for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}

    def detect(self, rgb: np.ndarray, prompt: str) -> List[RawDet]:
        from PIL import Image

        h, w = rgb.shape[:2]
        det = self.model.predict(Image.fromarray(rgb), threshold=self.min_score)
        cid = self.class_of[COCO_NAME[prompt]]
        keep = det.class_id == cid
        return keep_top([RawDet(float(s), clip_box(b, w, h)) for s, b in zip(det.confidence[keep], det.xyxy[keep])],
                        self.min_score, self.max_det)


class _Sam3(Recogniser):
    """SAM 3 (segmentation with a text prompt) through transformers."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.proc = Sam3Processor.from_pretrained(self.spec["hf"], revision=self.spec["revision"])
        self.model = Sam3Model.from_pretrained(self.spec["hf"], revision=self.spec["revision"]).to(self.device).eval()

    def detect(self, rgb: np.ndarray, prompt: str) -> List[RawDet]:
        h, w = rgb.shape[:2]
        inputs = self.proc(images=rgb, text=prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.proc.post_process_instance_segmentation(outputs, threshold=self.min_score, mask_threshold=0.5,
                                                           target_sizes=inputs.get("original_sizes").tolist())[0]
        out = []
        for k in range(len(res["scores"])):
            m = res["masks"][k].cpu().numpy().astype(bool)
            out.append(RawDet(float(res["scores"][k]), clip_box(res["boxes"][k].cpu().numpy(), w, h), m))
        return keep_top(out, self.min_score, self.max_det)


def parse_json_list(text: str) -> List[Any]:
    """The first JSON list in a VLM answer ([] if none can be read)."""
    t = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = t.find("["), t.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        val = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return []
    return val if isinstance(val, list) else []


def qwen_boxes(text: str, w: int, h: int) -> List[np.ndarray]:
    """Qwen3-VL boxes ("bbox_2d", coordinates 0-1000 relative to the image) -> pixel boxes."""
    out = []
    for el in parse_json_list(text):
        b = el.get("bbox_2d") if isinstance(el, dict) else None
        if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            x0, y0, x1, y1 = (float(v) for v in b)
            box = clip_box([x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h], w, h)
            if box[2] > box[0] and box[3] > box[1]:
                out.append(box)
    return out


def molmo_points(text: str, w: int, h: int) -> List[np.ndarray]:
    """Molmo points -> pixels. Molmo2: <points coords="1 x y;2 x y ...">, coordinates 0-1000 (also "t i x y" with
    a time index); Molmo (v1): <point x="12.3" y="45.6"> / <points x1=.. y1=..>, coordinates in percent."""
    out = []
    for coords in re.findall(r'coords="([^"]*)"', text):
        for part in coords.split(";"):
            nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", part)]
            if len(nums) >= 2:
                x, y = nums[-2], nums[-1]
                out.append(np.array([x / 1000 * w, y / 1000 * h]))
    if not out:
        for xs, ys in re.findall(r'x\d*="([\d.]+)"\s+y\d*="([\d.]+)"', text):
            out.append(np.array([float(xs) / 100 * w, float(ys) / 100 * h]))
    return [np.clip(p, [0, 0], [w - 1, h - 1]) for p in out]


class _Vlm(Recogniser):
    """Qwen3-VL-8B-Instruct (boxes as text) and Molmo2-8B (points as text); greedy decoding, score 1.0."""

    def __init__(self, key: str, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.key, self.torch = key, torch
        remote = key == "molmo2"
        self.proc = AutoProcessor.from_pretrained(self.spec["hf"], revision=self.spec["revision"], trust_remote_code=remote)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.spec["hf"], revision=self.spec["revision"], trust_remote_code=remote, dtype=torch.bfloat16,
            device_map=self.device).eval()
        self.last_text = ""

    def detect(self, rgb: np.ndarray, prompt: str) -> List[RawDet]:
        from PIL import Image

        h, w = rgb.shape[:2]
        question = VLM_PROMPTS[self.key].format(name=prompt)
        messages = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(rgb)},
                                                 {"type": "text", "text": question}]}]
        inputs = self.proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        text = self.proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        self.last_text = text
        if self.key == "qwen3vl":
            return [RawDet(1.0, box=b) for b in qwen_boxes(text, w, h)][: self.max_det]
        return [RawDet(1.0, point=p) for p in molmo_points(text, w, h)][: self.max_det]


class MaskMaker:
    """MobileSAM through ultralytics (block item 5: box-only models -> MobileSAM masks): one call per item for all
    its boxes."""

    def __init__(self, weights: Path, device: str) -> None:
        from ultralytics import SAM

        self.model, self.device = SAM(str(weights)), device

    def masks(self, rgb: np.ndarray, boxes: List[np.ndarray]) -> List[np.ndarray]:
        if not boxes:
            return []
        h, w = rgb.shape[:2]
        r = self.model.predict(rgb[..., ::-1], bboxes=[b.tolist() for b in boxes], device=self.device, verbose=False)[0]
        data = r.masks.data.cpu().numpy() > 0.5
        return [m if m.shape == (h, w) else np.zeros((h, w), bool) for m in data]


def build(key: str, spec: Dict[str, Any], cache: Path, device: str, min_score: float, max_det: int) -> Recogniser:
    """The recogniser of p9_models.yaml entry `key`."""
    args = (spec, cache, device, min_score, max_det)
    if "ultralytics" in spec:
        return _Ultralytics(*args)
    if key in ("grounding_dino", "llmdet"):
        return _GroundingDino(*args)
    if key == "owlv2":
        return _Owlv2(*args)
    if key == "rfdetr":
        return _RfDetr(*args)
    if key == "sam3":
        return _Sam3(*args)
    if key in ("qwen3vl", "molmo2"):
        return _Vlm(key, *args)
    raise KeyError(key)

"""Phase 14b-2 R2: fine-tune YOLOE-11-L-seg on the R1 training frames with a fixed 6-class vocabulary, in
naz_mapmad_recog (rules p14b2_rules.md §R2; config p14b2_recog.yaml `finetune`):

    python -m mapmad_recog.finetune prep  --config mapmad/configs/p14b2_recog.yaml          # YOLO dataset (CPU)
    python -m mapmad_recog.finetune train --config ... --variant lp                         # one variant, one GPU

prep: selection.json -> <out>/yolo/{images,labels}/{train,val} (+ data.yaml). Ignore pixels (Phase 9 ignore list,
small target instances) are painted grey (114, 114, 114: ultralytics' padding colour), so they teach neither class nor
background. Each labelled instance -> one polygon (parts joined by ultralytics' merge_multi_segment).
train: ultralytics' YOLOEPESegTrainer (text embeddings of the 6 Phase 9 prompts fused into the head). `lp` freezes
everything except the class projections cv3.*.2 (ultralytics' linear-probing recipe); `ft` trains every layer. Fixed
recipe from the config; the LAST epoch's weights are used (no picking on val). -> <out>/train/<variant>/weights/last.pt,
<out>/train/<variant>/result.json (sha256, batch actually used, seconds).
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import yaml

from mapmad_sim import config, photo_labels as pl
from mapmad_sim.run_layout import read_config

GREY = 114
CLASS_NAMES = [pl.PROMPTS[c] for c in pl.CATEGORIES]  # class k (0-based) = cls png value k + 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def instance_polygons(inst: np.ndarray, cls: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """(class 0..5, polygon (N, 2) in pixels) per labelled instance of a frame; parts of one instance are joined."""
    from ultralytics.data.converter import merge_multi_segment

    out = []
    for i in np.unique(inst):
        if i == 0:
            continue
        m = (inst == i).astype(np.uint8)
        k = int(np.bincount(cls[m > 0]).argmax()) - 1
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        parts = [c.reshape(-1, 2) for c in contours if len(c) >= 3]
        if not parts:
            continue
        poly = parts[0] if len(parts) == 1 else np.concatenate(merge_multi_segment([p.reshape(-1).tolist() for p in parts]))
        out.append((k, np.asarray(poly, np.float64).reshape(-1, 2)))
    return out


def label_lines(polys: Sequence[Tuple[int, np.ndarray]], hw: Tuple[int, int]) -> List[str]:
    """YOLO segmentation label lines: class, then x y pairs normalised to the picture."""
    h, w = hw
    return [f"{k} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in p) for k, p in polys]


def grey_ignore(rgb_bgr: np.ndarray, cls: np.ndarray) -> np.ndarray:
    out = rgb_bgr.copy()
    out[cls == 255] = GREY
    return out


def setup(cfg_path: Path) -> Tuple[Dict[str, Any], Path]:
    cfg = read_config(cfg_path)
    return cfg, config.paths()["outputs"] / cfg["out_dir"]


def cmd_prep(cfg: Dict[str, Any], out: Path) -> None:
    sel = json.loads((out / "trainset" / "selection.json").read_text())
    root = out / "yolo"
    counts = {"train": 0, "val": 0, "instances": 0, "empty": 0}
    for f in sel["frames"]:
        sp = f["split"]
        src = out / "trainset" / "cand" / f["home"] / f["frame_id"]
        img_out = root / "images" / sp / f"{f['frame_id']}.jpg"
        lab_out = root / "labels" / sp / f"{f['frame_id']}.txt"
        if lab_out.exists():
            counts[sp] += 1
            continue
        img = cv2.imread(f"{src}.jpg")
        inst = cv2.imread(f"{src}_inst.png", cv2.IMREAD_UNCHANGED)
        cls = cv2.imread(f"{src}_cls.png", cv2.IMREAD_UNCHANGED)
        lines = label_lines(instance_polygons(inst, cls), img.shape[:2])
        img_out.parent.mkdir(parents=True, exist_ok=True)
        lab_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_out), grey_ignore(img, cls), [cv2.IMWRITE_JPEG_QUALITY, 95])
        tmp = lab_out.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
        os.replace(tmp, lab_out)
        counts[sp] += 1
        counts["instances"] += len(lines)
        counts["empty"] += not lines
    data = {"path": str(root), "train": "images/train", "val": "images/val",
            "names": {k: n for k, n in enumerate(CLASS_NAMES)}}
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    (root / "prep.json").write_text(json.dumps({**counts, "selection_sha256": sha256_file(out / "trainset" / "selection.json")},
                                               indent=1) + "\n")
    print(json.dumps(counts))


def lp_freeze(model: Any) -> List[str]:
    """Ultralytics' linear-probing freeze list: everything except the head's class projections cv3.*.2."""
    head = len(model.model.model) - 1
    freeze = [str(f) for f in range(head)]
    for name, _ in model.model.model[-1].named_children():
        if "cv3" not in name:
            freeze.append(f"{head}.{name}")
    freeze += [f"{head}.cv3.{a}.{b}" for a in range(3) for b in range(2)]
    return freeze


def cmd_train(cfg: Dict[str, Any], out: Path, variant: str, models_cfg: Path) -> None:
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

    ft = cfg["finetune"]
    cache = Path(yaml.safe_load(models_cfg.read_text())["cache"]) / "ultralytics"
    os.chdir(cache)  # ultralytics finds its text encoder (mobileclip) here, as in models.py
    base = cache / ft["base_weights"]
    proj = out / "train"
    dest = proj / variant
    if (dest / "result.json").exists():
        print(f"{variant}: done already")
        return
    if dest.exists():  # an unfinished earlier attempt: kept aside, never deleted
        shutil.move(str(dest), str(proj / f"_partial_{variant}_{time.strftime('%m%d_%H%M%S')}"))
    model = YOLOE(str(base))
    extra = {"freeze": lp_freeze(model)} if variant == "lp" else {}
    t0 = time.time()
    model.train(data=str(out / "yolo" / "data.yaml"), epochs=int(ft["epochs"]), imgsz=int(ft["imgsz"]),
                batch=ft["batch"], seed=int(ft["seed"]), deterministic=bool(ft["deterministic"]),
                workers=int(ft["workers"]), trainer=YOLOEPESegTrainer, project=str(proj), name=variant,
                exist_ok=False, device=0, plots=False, verbose=False, **extra)
    last = dest / "weights" / "last.pt"
    args = yaml.safe_load((dest / "args.yaml").read_text())
    res = {"variant": variant, "weights": str(last.relative_to(out)), "sha256": sha256_file(last),
           "base_sha256": sha256_file(base), "seconds": round(time.time() - t0, 1), "batch_used": getattr(model.trainer, "batch_size", args.get("batch")),
           "names": CLASS_NAMES, "recipe": ft, "frozen_layers": len(extra.get("freeze", []))}
    (dest / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


def cmd_models(cfg: Dict[str, Any], out: Path, models_cfg: Path) -> Path:
    """<out>/models.yaml: the Phase 9 models file plus one `yoloe_<variant>` entry per finished variant (weights below
    the outputs root, fixed vocabulary, sha256 checked by the server at load). kind `fixed`: no OVON items."""
    doc = yaml.safe_load(models_cfg.read_text())
    root = config.paths()["outputs"]
    for v in cfg["finetune"]["variants"]:
        res_f = out / "train" / v / "result.json"
        if not res_f.exists():
            continue
        res = json.loads(res_f.read_text())
        doc["models"][f"yoloe_{v}"] = {**doc["models"]["yoloe"], "name": f"YOLOE-11-L-seg fine-tuned ({v}, Phase 14b-2)",
                                      "kind": "fixed", "weights": str((out / res["weights"]).relative_to(root)),
                                      "sha256": res["sha256"], "vocab": CLASS_NAMES}
    path = out / "models.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"{path}: {sorted(k for k in doc['models'] if k.startswith('yoloe'))}")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["prep", "train", "models"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--variant", choices=["lp", "ft"])
    p.add_argument("--models", type=Path, default=Path(__file__).resolve().parents[2] / "configs" / "p9_models.yaml")
    a = p.parse_args()
    cfg, out = setup(a.config)
    if a.step == "prep":
        cmd_prep(cfg, out)
    elif a.step == "models":
        cmd_models(cfg, out, a.models)
    else:
        if not a.variant:
            sys.exit("--variant lp|ft")
        cmd_train(cfg, out, a.variant, a.models)


if __name__ == "__main__":
    main()

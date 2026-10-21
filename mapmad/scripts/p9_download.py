"""Download the Phase 9 recogniser checkpoints into the model cache and record every weight file's sha256
(block item 5, task 9.0.3). Runs in naz_mapmad_recog, on the CPU only (text encoders are triggered on the CPU):

    docker exec -e HOME=/model_cache/home -e YOLO_OFFLINE=False naz_mapmad_recog \
        python mapmad/scripts/p9_download.py [--only sam3]

Order = p9_models.yaml `download_order` (SAM 3, gated, last). A model that cannot be downloaded (gate, licence,
network after 2 tries) is recorded as skipped with the reason; nothing stops. The HF token is read by huggingface_hub
from HF_TOKEN_PATH (/hf_token/token, mounted read-only) and is never printed or written anywhere.
Writes <outputs>/p9_recognisers/photo_test/weights.json (merged with an earlier run: one entry per model).
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p9_models.yaml"
OUT = Path(os.environ.get("MAPMAD_OUTPUTS", "/outputs/mapmad")) / "p9_recognisers" / "photo_test" / "weights.json"
HF_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.py", "*.model", "*.jinja", "*.tiktoken", "*.md", "LICENSE*"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def weight_files(folder: Path) -> Dict[str, str]:
    """sha256 of every weight file (.safetensors / .pt / .pth / .ts / .bin) below a folder (symlinks resolved)."""
    out = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix in (".safetensors", ".pt", ".pth", ".ts", ".bin"):
            out[str(p.relative_to(folder))] = sha256(p.resolve())
    return out


def get_hf(repo: str, revision: str, patterns: List[str]) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))


def get_ultralytics(cache: Path, name: str) -> Path:
    """The .pt file plus its text encoder (downloaded on first use; triggered on the CPU here)."""
    from ultralytics import YOLO, YOLOE

    folder = cache / "ultralytics"
    folder.mkdir(parents=True, exist_ok=True)
    os.chdir(folder)  # ultralytics saves assets it fetches (weights, MobileCLIP) into the working folder
    if name.startswith("yoloe"):
        m = YOLOE(str(folder / name))
        m.set_classes(["chair"], m.get_text_pe(["chair"]))
    else:
        m = YOLO(str(folder / name))
        m.set_classes(["chair"])
    return folder


def get_rfdetr(cache: Path, size: str) -> Path:
    import rfdetr

    folder = cache / "rfdetr"
    folder.mkdir(parents=True, exist_ok=True)
    os.chdir(folder)
    {"medium": rfdetr.RFDETRMedium}[size](device="cpu")
    return Path.home() / ".roboflow" / "models"  # rfdetr keeps its weights there (HOME=/model_cache/home)


def download(key: str, spec: Dict[str, Any], cache: Path) -> Dict[str, Any]:
    if "hf" in spec:
        folder = get_hf(spec["hf"], spec["revision"], [spec["file"]] if "file" in spec else HF_PATTERNS)
        entry = {"source": f"huggingface.co/{spec['hf']}", "revision": spec["revision"]}
    elif "ultralytics" in spec:
        folder = get_ultralytics(cache, spec["ultralytics"])
        entry = {"source": f"ultralytics assets ({spec['ultralytics']})", "revision": "ultralytics==8.4.183"}
    else:
        folder = get_rfdetr(cache, spec["rfdetr"])
        entry = {"source": f"rfdetr {spec['rfdetr']}", "revision": "rfdetr==1.11.2"}
    return {**entry, "status": "ok", "folder": str(folder), "files": weight_files(folder)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--tries", type=int, default=2)
    a = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    cache = Path(cfg["cache"])
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    keys: List[str] = a.only or ["mobile_sam"] + cfg["download_order"]
    for key in keys:
        spec = cfg["mobile_sam"] if key == "mobile_sam" else cfg["models"][key]
        if done.get(key, {}).get("status") == "ok" and not a.only:
            continue
        for attempt in range(1, a.tries + 1):
            t0 = time.time()
            try:
                entry = download(key, spec, cache)
                break
            except Exception as exc:  # gate / licence / network: recorded, the next model goes on
                reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
                traceback.print_exc(limit=2)
                entry = {"status": "failed", "reason": reason, "attempts": attempt}
                time.sleep(5)
        entry["seconds"] = round(time.time() - t0, 1)
        entry["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        done[key] = entry
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(done, indent=1) + "\n")
        tmp.replace(OUT)
        print(f"{key}: {entry['status']} {entry.get('reason', '')} ({entry['seconds']} s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())

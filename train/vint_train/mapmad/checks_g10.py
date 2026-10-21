"""Gate G10 rows 1-2 (Phase 10 block item 10) on the fixed batch of `reference_forward.py`, saved before any Phase 10
code change (/outputs/mapmad/p10_v2_train/checks/reference_forward_cpu.pt, CPU), checked on the step-0 model:

- row 1  switch off: NoMaD built with map_input false + nomad.pth -> every output torch.equal to the reference;
- row 2  maps hidden: MapMaD v2 (mapmad_v2.yaml: two map tokens, 4 layers) + nomad.pth exactly as training starts
         (weights.build_with_official), both map windows hidden, with non-zero maps AND non-zero encoder last layers
         (so the masks are what hide them) -> allclose(atol 1e-6, rtol 0) to the reference, photo shown and hidden;
         token count 7 (5 + near + far); every nomad.pth key loads; the only new keys are vision_encoder.map_encoder.*
         (near.* and far.*); positional rows 0-4 equal to the checkpoint's;
- report: a mixed batch (per-sample photo/near/far masks) equals the per-mask results row by row (up to kernel
         rounding, <= 1e-4: hidden and shown rows run as sub-batches); a shown map changes
         the output; v1 (one token, mapmad_p4_C.yaml) with the map hidden still equals the reference (G3 row 1b).

    CUDA_VISIBLE_DEVICES= python -m vint_train.mapmad.checks_g10 --config config/mapmad_v2.yaml \
        --v1-config config/mapmad_p4_C.yaml --reference /outputs/mapmad/p10_v2_train/checks/reference_forward_cpu.pt \
        --out /outputs/mapmad/p10_v2_train/checks/g10_rows_1_2.json
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import torch  # noqa: E402

from vint_train.mapmad.checks_g3 import ATOL, compare  # noqa: E402
from vint_train.mapmad.closed_loop.nomad_policy import load_nomad  # noqa: E402
from vint_train.mapmad.config import git_commit, load_config  # noqa: E402
from vint_train.mapmad.reference_forward import enforce_determinism, run_forward  # noqa: E402
from vint_train.mapmad.weights import build_with_official  # noqa: E402


def test_maps_v2(batch: int, seed: int = 1) -> torch.Tensor:
    """A non-zero [B, 2, 4, 64, 64] map batch: random 0/1 obstacle, explored, frontier; random target values."""
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand((batch, 2, 4, 64, 64), generator=g) > 0.7).float()
    m[:, :, 3] = torch.rand((batch, 2, 64, 64), generator=g)
    return m


def live_encoders(model: torch.nn.Module) -> None:
    """Non-zero last layers of both map encoders (zero-initialised at step 0)."""
    with torch.no_grad():
        for enc in model.vision_encoder.map_encoder.values():
            torch.nn.init.normal_(enc.proj.weight, std=0.05)


def mask_report(model: torch.nn.Module, inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Token count, per-sample masks vs per-mask results, a shown map changes the output (report)."""
    enc = model.vision_encoder
    obs, goal = inputs["obs_img"].to(device), inputs["goal_img"].to(device)
    b = obs.shape[0]
    maps = test_maps_v2(b).to(device)
    rows = torch.arange(b, device=device)
    with torch.no_grad():
        image_tokens = enc._encode_images(obs, goal)
        zeros = torch.zeros(b, dtype=torch.long, device=device)
        tokens, _ = enc.map_tokens(image_tokens, zeros, maps, torch.zeros((b, 2), dtype=torch.long, device=device))
        gm, mm = rows % 2, torch.stack([(rows // 2) % 2, (rows // 4) % 2], dim=1)
        run = (lambda g, m: model("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=g, map_img=maps,
                                  input_map_mask=m))
        mixed = run(gm, mm)
        worst = 0.0  # rows of a mixed batch run in sub-batches (maps hidden / shown): kernel rounding only
        for i in range(b):
            per = run(torch.full((b,), int(gm[i]), device=device), mm[i].repeat(b, 1))
            worst = max(worst, float((mixed[i] - per[i]).abs().max()))
        ok = worst <= 1e-4
        shown = run(zeros, torch.zeros((b, 2), dtype=torch.long, device=device))
        flipped = model("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=zeros, map_img=maps.flip(-1),
                        input_map_mask=torch.zeros((b, 2), dtype=torch.long, device=device))
    return {"old_tokens": int(image_tokens.shape[1]), "new_tokens": int(tokens.shape[1]),
            "mixed_batch_matches_per_mask": bool(ok), "mixed_batch_max_abs_diff": worst,
            "shown_map_changes_output": bool((shown - flipped).abs().max() > 1e-4)}


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="the v2 train config (mapmad_v2.yaml)")
    parser.add_argument("--v1-config", required=True, help="the v1 train config (mapmad_p4_C.yaml)")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    enforce_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config, "config/defaults.yaml")
    ref = torch.load(args.reference, weights_only=False)
    if str(device) != ref["device"]:
        raise SystemExit(f"reference made on {ref['device']}, this run on {device}: byte identity needs the same device")
    inputs, reference = ref["inputs"], ref["outputs"]
    b = inputs["obs_img"].shape[0]
    weights = cfg["init_weights"]
    result: Dict[str, Any] = {"git_commit": git_commit(), "reference": args.reference,
                              "reference_commit": ref["git_commit"], "device": str(device)}

    torch.manual_seed(ref["seed"])
    off = load_nomad(weights, dict(cfg, map_input=False), device)
    r1 = compare(run_forward(off, inputs, cfg, device), reference)
    result["row1"] = {"outputs": r1, "pass": all(v["equal"] for v in r1.values())}
    del off

    torch.manual_seed(int(cfg["seed"]))
    model, report = build_with_official(cfg, weights, device)
    live_encoders(model)
    hidden = {"map_img": test_maps_v2(b), "input_map_mask": torch.ones((b, 2), dtype=torch.long)}
    r2 = compare(run_forward(model, inputs, cfg, device, extra_encoder_kwargs=hidden), reference)
    masks = mask_report(model, inputs, device)
    keys_ok = (all(k.startswith("vision_encoder.map_encoder.") for k in report["new_keys"])
               and {k.split(".")[2] for k in report["new_keys"]} == {"near", "far"})
    result["row2"] = {"outputs": r2, "atol": ATOL, "load_report": report, "masks": masks,
                      "max_abs_diff": max(v["max_abs_diff"] for v in r2.values()),
                      "pass": (all(v["max_abs_diff"] <= ATOL for v in r2.values()) and masks["new_tokens"] == 7
                               and keys_ok and report["positional_encoding"].get("first_rows_equal", False))}

    v1 = load_config(args.v1_config, "config/defaults.yaml")
    torch.manual_seed(int(v1["seed"]))
    v1_model, _ = build_with_official(v1, weights, device)
    with torch.no_grad():
        torch.nn.init.normal_(v1_model.vision_encoder.map_encoder.proj.weight, std=0.05)
    v1_hidden = {"map_img": test_maps_v2(b)[:, 0, :3], "input_map_mask": torch.ones(b, dtype=torch.long)}
    rv1 = compare(run_forward(v1_model, inputs, v1, device, extra_encoder_kwargs=v1_hidden), reference)
    result["v1_map_hidden_report"] = {"max_abs_diff": max(v["max_abs_diff"] for v in rv1.values()),
                                      "pass": all(v["max_abs_diff"] <= ATOL for v in rv1.values())}
    result["pass"] = {"row1": result["row1"]["pass"], "row2": result["row2"]["pass"]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"pass": result["pass"], "row2_max_diff": result["row2"]["max_abs_diff"], "masks": masks,
                      "new_keys": len(report["new_keys"]), "new_parameters": report["new_parameters"],
                      "positional_encoding": report["positional_encoding"],
                      "v1_map_hidden": result["v1_map_hidden_report"]}, indent=1))


if __name__ == "__main__":
    main()

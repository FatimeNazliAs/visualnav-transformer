"""Gate G3 rows 1, 1b, 2, 3 on the fixed batch saved by `reference_forward.py`.

- row 1  switch off: NoMaD built with map_input false + nomad.pth -> every output torch.equal to the reference;
- row 1b map hidden: MapMaD (map_input true) + nomad.pth, map mask = hidden (with a non-zero map and a non-zero
         map encoder, so the mask is what hides it) -> allclose(atol 1e-6, rtol 0) to the reference, photo shown
         and hidden;
- row 2  weights: every nomad.pth key loads; new keys are map_encoder.* only; positional rows 0-4 equal;
- row 3  shapes: token count = old + 1; every goal/map mask combination, heat on and off, forward-passes with
         finite outputs (and a shown map changes the output, a hidden one does not).

Run inside naz_mapmad from /app/visualnav-transformer/train:
    CUDA_VISIBLE_DEVICES=0 python -m vint_train.mapmad.checks_g3 \
        --model-config /outputs/mapmad/weights/official/nomad.yaml \
        --weights /outputs/mapmad/weights/official/nomad.pth \
        --reference /outputs/mapmad/p3_model/reference_forward.pt --out /outputs/mapmad/p3_model/g3_rows_1_3.json
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from typing import Any, Dict, List

import torch

from vint_train.mapmad.closed_loop.nomad_policy import load_nomad
from vint_train.mapmad.config import git_commit
from vint_train.mapmad.reference_forward import enforce_determinism, load_yaml, run_forward
from vint_train.mapmad.weights import build_with_official

ATOL = 1e-6


def compare(outputs: Dict[str, Dict[str, torch.Tensor]], reference: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """Per output: equal? max |difference|."""
    res = {}
    for m, outs in reference.items():
        for k, ref in outs.items():
            got = outputs[m][k]
            res[f"{m}.{k}"] = {"equal": bool(torch.equal(got, ref)), "max_abs_diff": float((got - ref).abs().max())}
    return res


def test_maps(batch: int, seed: int = 1) -> torch.Tensor:
    """A non-zero [B, 3, 64, 64] map batch (random 0/1 obstacle + explored, random heat)."""
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand((batch, 3, 64, 64), generator=g) > 0.7).float()
    m[:, 2] = torch.rand((batch, 64, 64), generator=g)
    return m


def row3_shapes(model: torch.nn.Module, inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Token count and every mask combination forward-passing."""
    enc = model.vision_encoder
    obs, goal = inputs["obs_img"].to(device), inputs["goal_img"].to(device)
    b = obs.shape[0]
    maps = test_maps(b).to(device)
    out: Dict[str, Any] = {}
    with torch.no_grad():
        image_tokens = enc._encode_images(obs, goal)
        zeros = torch.zeros(b, dtype=torch.long, device=device)
        tokens, padding = enc.map_tokens(image_tokens, zeros, maps, zeros)
        out["old_tokens"], out["new_tokens"] = int(image_tokens.shape[1]), int(tokens.shape[1])
        combos: List[Dict[str, Any]] = []
        for goal_mask in (0, 1):
            for map_mask in (0, 1):
                for heat_on in (True, False):
                    m = maps.clone()
                    if not heat_on:
                        m[:, 2] = 0
                    gm = torch.full((b,), goal_mask, dtype=torch.long, device=device)
                    mm = torch.full((b,), map_mask, dtype=torch.long, device=device)
                    cond = model("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm, map_img=m,
                                 input_map_mask=mm)
                    noise = model("noise_pred_net", sample=inputs["noisy_action"].to(device),
                                  timestep=inputs["timesteps"].to(device), global_cond=cond)
                    combos.append({"goal_mask": goal_mask, "map_mask": map_mask, "heat_on": heat_on,
                                   "cond_shape": list(cond.shape), "finite": bool(torch.isfinite(cond).all()
                                                                               and torch.isfinite(noise).all())})
        # a mixed batch (per-sample masks) must equal the per-mask results row by row
        gm = torch.arange(b, device=device) % 2
        mm = (torch.arange(b, device=device) // 2) % 2
        mixed = model("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm, map_img=maps, input_map_mask=mm)
        per = {(g, mk): model("vision_encoder", obs_img=obs, goal_img=goal,
                              input_goal_mask=torch.full((b,), g, dtype=torch.long, device=device), map_img=maps,
                              input_map_mask=torch.full((b,), mk, dtype=torch.long, device=device))
               for g in (0, 1) for mk in (0, 1)}
        mixed_ok = all(torch.allclose(mixed[i], per[(int(gm[i]), int(mm[i]))][i], atol=ATOL, rtol=0) for i in range(b))
        shown = model("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=zeros, map_img=maps,
                      input_map_mask=zeros)
        shown2 = model("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=zeros, map_img=maps.flip(-1),
                       input_map_mask=zeros)
    out["combinations"] = combos
    out["mixed_batch_matches_per_mask"] = bool(mixed_ok)
    out["shown_map_changes_output"] = bool((shown - shown2).abs().max() > 1e-4)
    out["pass"] = (out["new_tokens"] == out["old_tokens"] + 1 and all(c["finite"] for c in combos)
                   and mixed_ok and out["shown_map_changes_output"])
    return out


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    enforce_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(args.model_config)
    ref = torch.load(args.reference, weights_only=False)
    inputs, reference = ref["inputs"], ref["outputs"]
    b = inputs["obs_img"].shape[0]
    result: Dict[str, Any] = {"git_commit": git_commit(), "reference": args.reference, "device": str(device)}

    # row 1: switch off
    torch.manual_seed(ref["seed"])
    off = load_nomad(args.weights, dict(cfg, map_input=False), device)
    r1 = compare(run_forward(off, inputs, cfg, device), reference)
    result["row1"] = {"outputs": r1, "pass": all(v["equal"] for v in r1.values())}
    del off

    # row 2 + row 1b: MapMaD with nomad.pth, map hidden; a random (non-zero) map encoder Linear and map
    model, report = build_with_official(cfg, args.weights, device)
    result["row2"] = dict(report, **{"pass": all(k.startswith("vision_encoder.map_encoder.") for k in report["new_keys"])
                                     and report["positional_encoding"].get("first_rows_equal", False)})
    with torch.no_grad():
        torch.nn.init.normal_(model.vision_encoder.map_encoder.proj.weight, std=0.05)
    hidden = {"map_img": test_maps(b), "input_map_mask": torch.ones(b, dtype=torch.long)}
    r1b = compare(run_forward(model, inputs, cfg, device, extra_encoder_kwargs=hidden), reference)
    result["row1b"] = {"outputs": r1b, "atol": ATOL,
                       "pass": all(v["max_abs_diff"] <= ATOL for v in r1b.values())}

    # row 3: shapes and combinations (with the non-zero map encoder)
    result["row3"] = row3_shapes(model, inputs, device)
    result["pass"] = {k: bool(result[k]["pass"]) for k in ("row1", "row1b", "row2", "row3")}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"pass": result["pass"], "row1b_max_diff": max(v["max_abs_diff"] for v in r1b.values()),
                      "new_keys": report["new_keys"], "new_parameters": report["new_parameters"],
                      "positional_encoding": report["positional_encoding"],
                      "tokens": [result["row3"]["old_tokens"], result["row3"]["new_tokens"]]}, indent=1))


if __name__ == "__main__":
    main()

"""GPU placement module (train/scripts/mapmad_p10/gpu_place.py, standard library only): the watcher's free list
(stale file = nothing free), the live check, preferred GPUs first. No nvidia-smi: the live check is a fake."""

import importlib.util
import json
import os
import time

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "mapmad_p10")


def place():
    spec = importlib.util.spec_from_file_location("gpu_place", os.path.join(SCRIPTS, "gpu_place.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def free_file(tmp_path, free, check_s=300):
    f = tmp_path / "gpu_free.json"
    f.write_text(json.dumps({"check_s": check_s, "free": free}))
    return str(f)


def test_free_list_reads_owner_and_fixed(tmp_path):
    p = place()
    f = free_file(tmp_path, {"chain": ["0", "1"], "twin": ["1"]})
    assert p.free_list(f, "chain") == ["0", "1"]
    assert p.free_list(f, "twin") == ["1"]
    assert p.free_list(f, "part3") == []
    assert p.free_list(f, "chain", fixed="1") == ["1"]  # DRY runs
    assert p.free_list(str(tmp_path / "missing.json"), "chain") == []


def test_stale_file_means_nothing_free(tmp_path):
    p = place()
    f = free_file(tmp_path, {"chain": ["0", "1"]}, check_s=300)
    assert p.free_list(f, "chain", now=time.time() + 899) == ["0", "1"]
    assert p.free_list(f, "chain", now=time.time() + 901 + 1) == []  # > 3 watcher intervals


def test_usable_and_pick_apply_the_live_check_and_preference():
    p = place()
    busy = {"0"}
    check = lambda g, owner: g not in busy  # noqa: E731
    assert p.usable(["0", "1"], "chain", check) == ["1"]
    assert p.pick(["0", "1"], "chain", ["1"], check) == "1"
    busy.clear()
    assert p.pick(["0", "1"], "chain", ["1"], check) == "1"  # preferred first
    assert p.pick(["0", "1"], "chain", [], check) == "0"
    assert p.pick(["0"], "chain", ["1"], check) == "0"  # a preferred GPU not in the free list is never taken
    busy.update({"0", "1"})
    assert p.pick(["0", "1"], "chain", ["1"], check) is None


def test_cli_free_and_fixed(tmp_path, capsys):
    p = place()
    f = free_file(tmp_path, {"chain": ["1"]})
    assert p.main(["free", "--owner", "chain", "--free-file", f]) == 0
    assert capsys.readouterr().out.strip() == "1"
    assert p.main(["free", "--owner", "chain", "--free-file", f, "--fixed", "0 1"]) == 0
    assert capsys.readouterr().out.strip() == "0 1"

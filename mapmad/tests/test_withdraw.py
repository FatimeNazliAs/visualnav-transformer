"""GPU withdrawal (Phase 8 driver fix): a worker whose GPU stops being free gives its home back before the next drive
(incomplete drives kept in _partial/, claim removed), and a worker on a free GPU picks the home up again. No simulator:
HomeGeneratorV2 is replaced by a stand-in that writes one folder per drive."""

import importlib.util
from pathlib import Path

import pytest

from mapmad_sim import datagen_v2
from mapmad_sim.datagen_v2 import HomeWithdrawn, generate_home, withdraw_home
from mapmad_sim.run_layout import try_claim

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_drives_v2.py"


def load_worker():
    spec = importlib.util.spec_from_file_location("generate_drives_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeGen:
    """Stands in for HomeGeneratorV2: every drive = one folder with one file."""

    def __init__(self, cfg, home, labelled, gpu):
        self.calls = []

    def drive(self, did, kind, out, failed_dir=None, on_step=None):
        (out / did).mkdir(parents=True)
        (out / did / "traj_data.pkl").write_bytes(b"x")
        return {"drive_id": did, "kind": kind, "kept": True, "outcome": "complete"}

    def close(self):
        pass


def test_should_stop_withdraws_before_the_next_drive(tmp_path, monkeypatch):
    monkeypatch.setattr(datagen_v2, "HomeGeneratorV2", FakeGen)
    flag = tmp_path / "STOP_gpu0_tok"
    calls = []

    def should_stop():
        calls.append(1)
        if len(calls) == 3:  # the GPU stops being free after 2 drives
            flag.touch()
        return flag.exists()

    with pytest.raises(HomeWithdrawn):
        generate_home({}, tmp_path, "00001-aaa", False, 0, "p8d", {"explore": 4}, False, should_stop=should_stop)
    assert sorted(p.name for p in (tmp_path / "_tmp" / "00001-aaa").iterdir()) == [
        "p8d-00001-aaa-explore-000", "p8d-00001-aaa-explore-001"]
    assert not (tmp_path / "_homes" / "00001-aaa.json").exists()


def test_withdraw_keeps_partial_drives_and_releases_the_claim(tmp_path):
    (tmp_path / "_tmp" / "00001-aaa" / "p8d-00001-aaa-explore-000").mkdir(parents=True)
    claim = tmp_path / "_claims" / "tok" / "00001-aaa"
    assert try_claim(claim)
    dest = withdraw_home(tmp_path, "00001-aaa", claim, "tok")
    assert dest == tmp_path / "_partial" / "00001-aaa__tok"
    assert (dest / "p8d-00001-aaa-explore-000").is_dir()
    assert not (tmp_path / "_tmp" / "00001-aaa").exists() and not claim.exists()
    assert withdraw_home(tmp_path, "00002-bbb", None, "tok") is None  # nothing written yet: nothing to keep


def test_a_released_home_is_picked_up_in_a_later_pass(tmp_path):
    worker = load_worker()
    claims = tmp_path / "_claims" / "tok"
    order = ["00001-aaa", "00002-bbb"]
    assert try_claim(claims / "00001-aaa")  # another worker holds home 1
    got = []
    for home in worker.queue_passes(order, tmp_path, claims):
        got.append(home)
        if home == "00002-bbb":
            (tmp_path / "_homes").mkdir(exist_ok=True)
            (tmp_path / "_homes" / "00002-bbb.json").write_text("{}")
            (claims / "00001-aaa").unlink()  # ... which is then withdrawn: its claim is released
    assert got == ["00002-bbb", "00001-aaa"]


def test_a_withdrawn_worker_claims_no_further_home(tmp_path):
    worker = load_worker()
    claims = tmp_path / "_claims" / "tok"
    flag = tmp_path / "STOP_gpu0_tok"
    got = []
    for home in worker.queue_passes(["00001-aaa", "00002-bbb"], tmp_path, claims, flag.exists):
        got.append(home)
        flag.touch()  # the GPU stops being free while home 1 runs
    assert got == ["00001-aaa"] and not (claims / "00002-bbb").exists()


def test_stop_file_name_is_per_gpu_and_launch(tmp_path):
    worker = load_worker()
    assert worker.gpu_stop_file(tmp_path, 0, "tok").name == "STOP_gpu0_tok"

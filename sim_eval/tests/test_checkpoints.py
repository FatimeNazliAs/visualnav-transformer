"""Pin how a checkpoint's own training config is recovered.

The fairness protocol (plan §7) evaluates each arm at its own image_size, and
the only surviving record of that is the config dict `train.py` prints into the
training log. This parse is therefore load-bearing, and the failure mode if it
quietly stops working — falling back to some default resolution — would make
every later comparison meaningless. So it fails loudly instead, and that is
what these tests pin.

Needs no GPU, no weights and no simulator — but run it in the container,
which is where pytest lives:

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import checkpoints  # noqa: E402

# The real line, trimmed to the keys under test, exactly as train.py prints it.
CONFIG_LINE = (
    "{'project_name': 'nomad_capstone', 'run_name': 'bc100_s0', "
    "'model_type': 'nomad', 'image_size': [160, 120], 'context_size': 3, "
    "'len_traj_pred': 8, 'num_diffusion_iters': 10, 'normalize': True, "
    "'vision_encoder': 'nomad_vint', 'encoding_size': 256}"
)


def test_parses_the_config_dict_out_of_a_noisy_log():
    log = "wandb: WARNING something\n" + CONFIG_LINE + "\nEpoch 0 loss 1.23\n"
    params = checkpoints.parse_train_config(log)
    assert params["image_size"] == [160, 120]
    assert params["context_size"] == 3


def test_a_log_without_the_config_line_is_an_error_not_a_guess():
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.parse_train_config("Epoch 0 loss 1.23\n")


def test_a_truncated_config_line_is_an_error():
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.parse_train_config("{'project_name': 'nomad_capstone', 'im")


def test_missing_required_params_are_named():
    params = {"model_type": "nomad", "image_size": [96, 96]}
    with pytest.raises(checkpoints.CheckpointError) as error:
        checkpoints.validate_params(params, "a log")
    assert "context_size" in str(error.value)


def test_a_non_nomad_config_is_rejected():
    params = dict.fromkeys(checkpoints.REQUIRED_PARAMS, 1)
    params["model_type"] = "vint"
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.validate_params(params, "a log")


def test_an_unknown_checkpoint_name_lists_the_known_ones():
    with pytest.raises(checkpoints.CheckpointError) as error:
        checkpoints.load("does_not_exist")
    assert "best_combined" in str(error.value)


def test_required_params_covers_everything_its_consumers_read():
    """Pinned by value, because this list had a hole.

    The five mha_/down_dims/cond_predict_scale keys are read by
    nomad_policy.build_model and were validated nowhere, so a log missing any of
    them passed validation and then died with a bare KeyError inside model
    construction — the exact failure this module exists to replace with a
    readable error. Asserting the full set by value (rather than importing
    nomad_policy, which would drag torch into a torch-free test) means the list
    cannot silently shrink again.
    """
    assert set(checkpoints.REQUIRED_PARAMS) == {
        "model_type", "image_size", "context_size", "len_traj_pred",
        "num_diffusion_iters", "normalize",
        "vision_encoder", "encoding_size", "mha_num_attention_heads",
        "mha_num_attention_layers", "mha_ff_dim_factor", "down_dims",
        "cond_predict_scale",
    }


def test_the_registry_entries_are_complete():
    """Every arm needs both halves: the weights, and how it was trained."""
    registry = checkpoints.read_registry()
    assert registry
    for name, entry in registry.items():
        assert "weights" in entry, name
        assert "train_log" in entry, name


def test_the_spec_exposes_image_size_as_transform_images_wants_it():
    spec = checkpoints.CheckpointSpec(
        name="fake", weights_path=Path("/nowhere"),
        model_params=yaml.safe_load("image_size: [160, 120]\ncontext_size: 3\n"))
    assert spec.image_size == [160, 120]
    assert spec.context_size == 3


def test_the_spec_reads_the_context_stride_the_run_was_trained_with():
    """best_combined and stride3 were trained on context three ticks apart."""
    params = checkpoints.parse_train_config(
        CONFIG_LINE[:-1] + ", 'context_stride': 3}")
    spec = checkpoints.CheckpointSpec(
        name="fake", weights_path=Path("/nowhere"), model_params=params)
    assert spec.context_stride == 3


def test_a_log_without_context_stride_means_stride_one():
    """Stock runs predate the knob; train.py and vint_dataset both default it to 1."""
    spec = checkpoints.CheckpointSpec(
        name="fake", weights_path=Path("/nowhere"),
        model_params=checkpoints.parse_train_config(CONFIG_LINE))
    assert spec.context_stride == 1

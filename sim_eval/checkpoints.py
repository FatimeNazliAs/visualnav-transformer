"""Resolve a checkpoint name to its weights and the config it was trained with.

The fairness protocol (plan §7) says each arm is evaluated at *its own*
training resolution — best-combined is 160x120, clean-stock is 96x96, and that
difference is part of what each arm *is*. So nothing downstream may hardcode an
image size; it asks here.

Getting that from the checkpoint is more awkward than it should be. `train.py`
writes weights into the run folder but no config beside them: it merges
`config/defaults.yaml` with the run's own yaml, prints the result, and moves
on. The run yamls for the capstone arms are gone (they lived in wandb's
offline run dirs). What survives is that printed dict, in the training log —
so that is what this module parses. It is the run's real config, recorded at
training time, rather than a number re-typed from memory into a new file.

    spec = load("best_combined")
    spec.model_params["image_size"]  # -> [160, 120]
"""

import ast
from pathlib import Path

import yaml

SIM_EVAL_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = SIM_EVAL_DIR / "configs" / "checkpoints.yaml"

# The line train.py prints: `print(config)` on a dict whose first key is
# project_name, because that is the first key of config/defaults.yaml.
CONFIG_LINE_PREFIX = "{'project_name'"

# Fields the bridge cannot run without. Anything missing means the log is from
# a different trainer, was truncated, or is not a NoMaD run — all of which are
# better as an error here than as a shape mismatch deep inside the model.
#
# This list must cover every key its consumers read, or the loud failure this
# module promises turns into a bare KeyError halfway through building the model.
# It did not, at first: the five mha_/down_dims/cond_predict_scale keys below
# were read by nomad_policy.build_model and validated nowhere.
# tests/test_checkpoints.py pins the whole set by value so it cannot silently
# shrink again.
REQUIRED_PARAMS = (
    # read by nomad_policy.NomadPolicy and bridge.NomadBridge
    "model_type",
    "image_size",
    "context_size",
    "len_traj_pred",
    "num_diffusion_iters",
    "normalize",
    # read by nomad_policy.build_model
    "vision_encoder",
    "encoding_size",
    "mha_num_attention_heads",
    "mha_num_attention_layers",
    "mha_ff_dim_factor",
    "down_dims",
    "cond_predict_scale",
)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be resolved, or its config cannot be."""


class CheckpointSpec:
    """One evaluatable checkpoint: its weights, and how it was trained."""

    def __init__(self, name, weights_path, model_params, description=""):
        self.name = name
        self.weights_path = weights_path
        self.model_params = model_params
        self.description = description

    @property
    def image_size(self):
        """[width, height] the model was trained at, as transform_images wants."""
        return list(self.model_params["image_size"])

    @property
    def context_size(self):
        """Number of *past* frames; the observation is this many plus the current one."""
        return int(self.model_params["context_size"])

    def summary(self):
        return "{} — {}x{}, context {}, {} diffusion steps\n  weights: {}".format(
            self.name, self.image_size[0], self.image_size[1],
            self.context_size, self.model_params["num_diffusion_iters"],
            self.weights_path)


def read_registry(registry_path=REGISTRY_PATH):
    """Load configs/checkpoints.yaml."""
    with open(registry_path, "r") as handle:
        return yaml.safe_load(handle)


def parse_train_config(log_text):
    """Pull the merged training config out of a training log's text.

    The dict is printed with Python's repr, so `ast.literal_eval` reads it back
    exactly — and, unlike `eval`, cannot execute anything a log happens to
    contain.
    """
    for line in log_text.splitlines():
        if line.startswith(CONFIG_LINE_PREFIX):
            try:
                return ast.literal_eval(line)
            except (ValueError, SyntaxError) as error:
                raise CheckpointError(
                    "found the config line in the training log but could not "
                    "parse it: {}".format(error))
    raise CheckpointError(
        "no line starting {!r} in the training log, so the config this "
        "checkpoint was trained with is unknown. Refusing to guess an "
        "image_size: the fairness protocol depends on it being the run's own."
        .format(CONFIG_LINE_PREFIX))


def validate_params(params, source):
    """Fail on a config that is not a usable NoMaD run."""
    missing = [key for key in REQUIRED_PARAMS if key not in params]
    if missing:
        raise CheckpointError(
            "config from {} is missing {}".format(source, ", ".join(missing)))
    if params["model_type"] != "nomad":
        raise CheckpointError(
            "config from {} is a {!r} model; only nomad is supported."
            .format(source, params["model_type"]))
    return params


def load(name, registry_path=REGISTRY_PATH):
    """Resolve a registry name to a CheckpointSpec, or raise CheckpointError."""
    registry = read_registry(registry_path)
    if name not in registry:
        raise CheckpointError(
            "unknown checkpoint {!r}. Known: {}. Add new ones to {}."
            .format(name, ", ".join(sorted(registry)), registry_path))

    entry = registry[name]
    weights_path = Path(entry["weights"])
    train_log = Path(entry["train_log"])

    if not weights_path.exists():
        raise CheckpointError(
            "weights for {!r} not found at {}. These are container paths — run "
            "this inside naz_nomad_sim, where the shared disk is mounted."
            .format(name, weights_path))
    if not train_log.exists():
        raise CheckpointError(
            "training log for {!r} not found at {}, so its image_size cannot "
            "be established.".format(name, train_log))

    params = validate_params(parse_train_config(train_log.read_text()), train_log)
    return CheckpointSpec(
        name=name,
        weights_path=weights_path,
        model_params=params,
        description=entry.get("description", ""),
    )

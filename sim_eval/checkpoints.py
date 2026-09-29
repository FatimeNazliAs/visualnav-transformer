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

The same holds for `context_stride`: best-combined was trained on context
frames three ticks apart, and feeding it consecutive ticks shows it a window it
never saw. So the stride comes from the log too (`CheckpointSpec.context_stride`).
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
        # Filled by `load_goal_arm`: where each goal-feeding key came from.
        self.goal_provenance = None

    @property
    def image_size(self):
        """[width, height] the model was trained at, as transform_images wants."""
        return list(self.model_params["image_size"])

    @property
    def context_size(self):
        """Number of *past* frames; the observation is this many plus the current one."""
        return int(self.model_params["context_size"])

    @property
    def context_stride(self):
        """How many ticks apart the context frames are — the run's own, like image_size.

        Training samples the context at `curr - k * waypoint_spacing *
        context_stride` (vint_dataset.py `_context_times`). go_stanford's
        waypoint_spacing is train.py's default of 1 frame, and its frames are
        the same 4 Hz as a control tick — which is why stock NoMaD is fed
        consecutive ticks — so one stride unit is one tick.

        Absent means 1, and that is not a guess: runs trained before the knob
        existed were stock (stride 1), and both train.py and vint_dataset
        default it to 1. The stride-1 arms' logs do not carry the key; the
        stride-3 arms' logs all do.
        """
        return int(self.model_params.get("context_stride", 1))

    @property
    def goal_type(self):
        """"image" (stock NoMaD) or "clip"; absent means image (pre-CLIP runs)."""
        return self.model_params.get("goal_type", "image")

    @property
    def clip_fusion(self):
        return self.model_params.get("clip_fusion", "none")

    def summary(self):
        text = ("{} — {}x{}, context {} (stride {}), {} diffusion steps\n"
                "  weights: {}").format(
            self.name, self.image_size[0], self.image_size[1],
            self.context_size, self.context_stride,
            self.model_params["num_diffusion_iters"], self.weights_path)
        if self.goal_provenance is not None:
            text += "".join("\n  {:<15} {!s:<14} ({})".format(
                key, self.model_params[key], self.goal_provenance[key])
                for key in GOAL_CONFIG_KEYS)
        return text


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


# --- the language-goal sim (Phase 7): what kind of goal an arm takes ---------
#
# A single-goal arm is fed a word, a photo or nothing, and feeding it the wrong
# kind does not crash: a CLIP model handed a photo through the image path, or
# a model trained on spaced context fed consecutive ticks, still drives. So an
# arm's goal feeding is read from its own run, never assumed, and a key that
# cannot be established refuses the arm (plan, Phase 7: "refuses if missing").
#
# Two of the five are younger than some runs, and those logs lack them. A
# missing key is then accepted only on evidence, and the evidence is named in
# `CheckpointSpec.goal_provenance`:
#
#   goal_type, clip_fusion   the weights say it: an image-goal model carries
#                            `vision_encoder.goal_encoder.*`, a Phase 2b CLIP
#                            model `clip_goal_proj.*`, a fused one
#                            `clip_fusion_proj.*`. A logged value is checked
#                            against the weights too.
#   context_stride           train.py sets 1 when the config has none, after it
#                            prints the config (train.py: `if "context_stride"
#                            not in config: config["context_stride"] = 1`), so
#                            an absent key is what that run trained with.
#
# context_size and image_size are REQUIRED_PARAMS already: no log, no arm.

GOAL_CONFIG_KEYS = ("goal_type", "clip_fusion", "context_size", "context_stride",
                    "image_size")

# What a CLIP-goal arm needs to build its goal the way training did.
CLIP_PARAMS = ("clip_model", "clip_text_template", "clip_mu_img", "clip_mu_txt",
               "clip_center")

STRIDE_DEFAULT_EVIDENCE = "absent from log; train.py sets 1 when absent"


def goal_structure(weight_keys):
    """(goal_type, clip_fusion) as the weights' own modules say."""
    keys = list(weight_keys)
    has = lambda prefix: any(key.startswith(prefix) for key in keys)  # noqa: E731
    image = has("vision_encoder.goal_encoder.")
    projection = has("vision_encoder.clip_goal_proj.")
    fused_mlp = has("vision_encoder.clip_fusion_proj.0.")
    fused_linear = has("vision_encoder.clip_fusion_proj.weight")
    found = [name for name, present in (
        ("image", image), ("clip/none", projection),
        ("clip/interaction", fused_mlp), ("clip/concat_linear", fused_linear))
        if present]
    if len(found) != 1:
        raise CheckpointError(
            "the weights do not name one goal encoder (found: {})".format(
                ", ".join(found) or "none"))
    return {"image": ("image", "none"), "clip/none": ("clip", "none"),
            "clip/interaction": ("clip", "interaction"),
            "clip/concat_linear": ("clip", "concat_linear")}[found[0]]


def resolve_goal_config(params, weight_keys, source):
    """`params` with every GOAL_CONFIG_KEYS entry set, and where each came from.

    Raises CheckpointError for a key that is missing and cannot be established,
    for a logged value the weights contradict, and for a CLIP arm without the
    CLIP settings it was trained with.
    """
    params = dict(params)
    provenance = {}
    goal_type, clip_fusion = goal_structure(weight_keys)
    for key, from_weights in (("goal_type", goal_type), ("clip_fusion", clip_fusion)):
        if key in params:
            if params[key] != from_weights:
                raise CheckpointError(
                    "{}: the log says {} {!r} but the weights are {!r}".format(
                        source, key, params[key], from_weights))
            provenance[key] = "train log (weights agree)"
        else:
            params[key] = from_weights
            provenance[key] = "absent from log; from the weights' modules"
    if "context_stride" in params:
        provenance["context_stride"] = "train log"
    else:
        params["context_stride"] = 1
        provenance["context_stride"] = STRIDE_DEFAULT_EVIDENCE
    for key in ("context_size", "image_size"):
        if key not in params:
            raise CheckpointError("{}: config has no {}".format(source, key))
        provenance[key] = "train log"
    if params["goal_type"] == "clip":
        missing = [key for key in CLIP_PARAMS if key not in params]
        if missing:
            raise CheckpointError(
                "{}: a CLIP-goal run without {} cannot have its goals built as "
                "training built them".format(source, ", ".join(missing)))
    return params, provenance


def load_goal_arm(name, registry_path=REGISTRY_PATH):
    """`load`, plus the goal feeding checked against the weights (Phase 7)."""
    import torch

    spec = load(name, registry_path)
    weights = torch.load(str(spec.weights_path), map_location="cpu")
    spec.model_params, spec.goal_provenance = resolve_goal_config(
        spec.model_params, weights.keys(), spec.weights_path)
    return spec

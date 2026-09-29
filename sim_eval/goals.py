"""What an arm is told to go to: a word, a photo, or nothing.

The language-goal sim (Phase 7) runs every arm in single-goal mode — one goal
for the whole episode, no topomap — and the arms differ in the goal they are
handed:

    word     the target word, through CLIP's text encoder         (V3)
    photo    a photo of the target instance, 1 m in front of it   (V1, V2)
    masked   no goal: the goal token is masked out, as training
             masks it with probability goal_mask_prob             (control)

A goal is built the way the arm's own training built it, and nowhere else:

  * **word** — `prep(encode_words([word], template), mu_txt)`, the Phase 4 V3
    recipe, with the arm's own `clip_text_template` and `clip_mu_txt`.
  * **photo, CLIP arm** — `prep(encode_images(preprocess(photo)), mu_img)`,
    Phase 1's cache recipe: the full-resolution render through CLIP's own
    224 px preprocessing, never the 96 px frame NoMaD sees.
  * **photo, image arm** — the photo itself; the policy resizes it to the
    arm's `image_size` like every other frame.
  * **masked** — `input_goal_mask = 1`, which drops the goal token from the
    transformer's attention and its average pool. What the token holds then
    cannot matter; it is a zero vector (CLIP arm) or the current frame (image
    arm), and `tests/test_goals.py` checks the "cannot matter" claim.

All the CLIP arithmetic is `vint_train.data.clip_goal_utils`, the module
training, precompute and the offline harness share, so the recipe cannot
drift between them.
"""

import numpy as np
import torch

GOAL_KINDS = ("word", "photo", "masked")


class GoalError(ValueError):
    """Raised for a goal an arm cannot be given."""


class Goal:
    """One episode's goal, ready for the policy.

    `vec` is the centred CLIP embedding a CLIP arm reads ([512], or None);
    `image` is the photo an image arm reads and the recorder shows (or None);
    `masked` says the goal token is masked out.
    """

    def __init__(self, kind, label, vec=None, image=None, masked=False):
        if kind not in GOAL_KINDS:
            raise GoalError("goal kind must be one of {}, not {!r}".format(GOAL_KINDS, kind))
        self.kind = kind
        self.label = label
        self.vec = vec
        self.image = image
        self.masked = bool(masked)


class ClipEncoder:
    """Frozen CLIP with the arm's own centring means, loaded once per arm."""

    def __init__(self, model_params, device):
        from vint_train.data.clip_goal_utils import load_clip, load_mu

        self.params = model_params
        self.device = device
        self.model, self.preprocess, self.tokenizer = load_clip(
            model_params["clip_model"], device)
        self.template = model_params["clip_text_template"]
        self.center = bool(model_params["clip_center"])
        self.mu_img = load_mu(model_params["clip_mu_img"]).to(device)
        self.mu_txt = load_mu(model_params["clip_mu_txt"]).to(device)

    def _centred(self, raw, mu):
        from vint_train.data.clip_goal_utils import l2_normalize, prep

        return prep(raw, mu) if self.center else l2_normalize(raw)

    def raw_word(self, word):
        from vint_train.data.clip_goal_utils import encode_words

        return encode_words(self.model, self.tokenizer, [word], self.template, self.device)[0]

    def raw_image(self, pil_image):
        """CLIP's embedding of a full-resolution frame, through its own preprocess."""
        from vint_train.data.clip_goal_utils import encode_images

        batch = self.preprocess(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        return encode_images(self.model, batch)[0]

    def word(self, word):
        return self._centred(self.raw_word(word), self.mu_txt)

    def image(self, pil_image):
        return self._centred(self.raw_image(pil_image), self.mu_img)


def render_goal_photo(body, instance):
    """The goal photo: the robot camera at the instance's `view_from`, facing it.

    Rendered through the same body and camera the episode runs on, so a photo
    goal is a frame the arm could itself have seen. Leaves the robot there; the
    episode's own start places it afterwards.
    """
    x, y = instance.view_from
    body.reset()
    body.place([float(x), float(y), body.scene.floor_height],
               [0.0, 0.0, instance.view_yaw()])
    return body.observe()


def build_goal(kind, spec, clip, word=None, photo=None):
    """The Goal an arm is handed for one task.

    `spec` is the arm's CheckpointSpec (its goal_type decides how a photo is
    fed); `clip` is its ClipEncoder, or None for an image arm.
    """
    goal_type = spec.goal_type
    if kind == "word":
        if goal_type != "clip":
            raise GoalError("{} is an image-goal arm; it cannot be given a word"
                            .format(spec.name))
        return Goal("word", word, vec=clip.word(word))
    if kind == "photo":
        if photo is None:
            raise GoalError("a photo goal needs the task's goal photo")
        vec = clip.image(photo) if goal_type == "clip" else None
        return Goal("photo", "photo of the {}".format(word), vec=vec, image=photo)
    if kind == "masked":
        vec = (torch.zeros(spec.model_params.get("clip_embed_dim", 512), device=clip.device)
               if goal_type == "clip" else None)
        return Goal("masked", "no goal (masked)", vec=vec, masked=True)
    raise GoalError("unknown goal kind {!r}".format(kind))


def cosine(a, b):
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

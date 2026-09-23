"""CLIP goal embeddings: loading CLIP, encoding goals, and closing the modality gap.

Training feeds the goal *image's* CLIP embedding; testing feeds a *word's* CLIP text
embedding. CLIP places image and text embeddings in separate regions of its space (the
modality gap), so each modality is mean-centred before the goal adapter sees it:

    prep(v, mu) = l2_normalize(l2_normalize(v) - mu)

where mu is the mean L2-normalised embedding of that modality (mu_img over the training
frames, mu_txt over a broad prompt set). Precompute, training, sanity checks and eval all
go through this module, so the recipe cannot drift between them.

open_clip is imported lazily, so the dataset can use `prep` in an environment without it.
"""
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Config `clip_model` name -> (open_clip architecture, pretrained tag). The OpenAI weights
# were trained with QuickGELU, so the "-quickgelu" architecture is the exact match.
CLIP_MODELS = {"ViT-B/32": ("ViT-B-32-quickgelu", "openai")}

# Object and scene words a language goal is drawn from. mu_txt is averaged over these
# (wrapped in the text template) plus SCENE_CAPTIONS; Phase 3 maps VLM answers onto them.
GOAL_WORDS = [
    "door", "doorway", "hallway", "corridor", "stairs", "elevator", "wall", "window",
    "chair", "table", "desk", "bench", "sofa", "sign", "poster", "plant", "tree",
    "bicycle", "car", "trash can", "whiteboard", "bookshelf", "cabinet", "counter",
    "column", "railing", "lamp", "screen", "building", "sidewalk", "lobby", "office",
]

SCENE_CAPTIONS = [
    "an empty hallway in an office building",
    "a long corridor with doors on both sides",
    "a lobby with glass doors",
    "a staircase inside a building",
    "an open office with desks and chairs",
    "a classroom with tables",
    "a kitchen area with a counter",
    "a lounge with sofas",
    "a corner where two hallways meet",
    "a wall with posters",
    "an elevator hallway",
    "a building entrance",
    "a sidewalk next to a building",
    "a courtyard with trees",
    "a parking lot with cars",
    "a bike rack outside a building",
    "a view through a glass door onto daylight",
    "a dim hallway at night",
    "a room seen from a low robot camera",
    "the floor of an indoor room",
]


def load_clip(clip_model: str, device: torch.device):
    """Frozen CLIP in eval mode, plus its image preprocessing and text tokenizer."""
    import open_clip

    architecture, pretrained = CLIP_MODELS[clip_model]
    model, _, preprocess = open_clip.create_model_and_transforms(
        architecture, pretrained=pretrained
    )
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, preprocess, open_clip.get_tokenizer(architecture)


def l2_normalize(v: torch.Tensor) -> torch.Tensor:
    return F.normalize(v, dim=-1)


def prep(v: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Centre an embedding on its modality mean: l2_normalize(l2_normalize(v) - mu)."""
    return l2_normalize(l2_normalize(v) - mu)


@torch.no_grad()
def encode_images(model, images: torch.Tensor) -> torch.Tensor:
    """Raw CLIP image embeddings for a batch already run through CLIP's preprocess."""
    return model.encode_image(images).float()


@torch.no_grad()
def encode_captions(model, tokenizer, captions: Sequence[str], device: torch.device) -> torch.Tensor:
    """Raw CLIP text embeddings for full sentences, encoded as written."""
    return model.encode_text(tokenizer(list(captions)).to(device)).float()


def encode_words(model, tokenizer, words: Sequence[str], template: str, device: torch.device) -> torch.Tensor:
    """Raw CLIP text embeddings for goal words, always wrapped in `template`.

    This is the only path a goal word takes into CLIP; a bare word is never encoded.
    """
    return encode_captions(model, tokenizer, apply_template(words, template), device)


def apply_template(words: Sequence[str], template: str) -> List[str]:
    assert "{}" in template, f"text template {template!r} has no '{{}}' slot"
    return [template.format(word) for word in words]


def modality_mean(embeddings: torch.Tensor) -> torch.Tensor:
    """mu: the mean of the L2-normalised embeddings (itself not re-normalised)."""
    return l2_normalize(embeddings).mean(dim=0)


def cache_key(traj_name: str, time: int) -> bytes:
    """LMDB key of one frame's CLIP image embedding."""
    return f"{traj_name}/{time}".encode()


def load_mu(path: str) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).float()

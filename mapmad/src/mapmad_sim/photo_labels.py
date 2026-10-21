"""Phase 9 photo test: which HM3D raw labels count as each asked category, and which are ignored (block item 2).

The 6 categories (ObjectNav HM3D v2) and the words each recogniser is asked:
    chair -> "chair", sofa -> "couch", bed -> "bed", toilet -> "toilet", tv_monitor -> "tv", plant -> "potted plant".
- counted names: the raw labels that Phase 8's truth maps to the category (configs/objectnav_categories.yaml, the
  same mapping; its sha256 goes into p9_rules.md);
- near-synonyms (ignore): raw labels in the 10 photo-test homes that a reasonable recogniser may call the category
  but the mapping does not (e.g. tv <- "led tv", "wall tv", "laptop"; chair <- "stool", "bench"). Built by a fixed
  word rule (NEAR_WORDS, minus NOT_NEAR) over the homes' label lists, then frozen in p9_rules.md before scoring.
OVON (block item 6): a category name counts the raw labels equal to it (lower case, "-" / "_" -> " "); a raw label
whose words contain all of the name's words, or are all contained in them, is a near-synonym (e.g. "chair" vs
"desk chair"). Names map by name only (the converter's integer ids are file-local).
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

CATEGORIES = ("chair", "sofa", "bed", "toilet", "tv_monitor", "plant")
PROMPTS = {"chair": "chair", "sofa": "couch", "bed": "bed", "toilet": "toilet", "tv_monitor": "tv",
           "plant": "potted plant"}

# A raw label is a near-synonym of a category if one of its words is listed here (and it is neither counted nor in
# NOT_NEAR). Words, not substrings: "bedside table" has the word "bedside", not "bed".
NEAR_WORDS: Dict[str, Tuple[str, ...]] = {
    "chair": ("chair", "chairs", "stool", "bench", "seat", "ottoman", "pouf", "pouffe", "footstool", "armchair"),
    "sofa": ("sofa", "couch", "loveseat", "settee", "futon", "armchair", "pouffe", "ottoman", "chaise", "daybed"),
    "bed": ("bed", "beds", "crib", "cot", "mattress", "headboard", "bedding", "comforter", "daybed", "futon"),
    "toilet": ("toilet", "bidet", "urinal", "chamber"),
    "tv_monitor": ("tv", "television", "monitor", "screen", "display", "laptop", "computer", "projector", "tablet"),
    "plant": ("plant", "plants", "flower", "flowers", "flowerpot", "vase", "pot", "planter", "tree", "cactus",
              "bonsai", "orchid", "succulent", "fern", "bush", "leaves"),
}
# Raw labels that share a word above but are clearly other things (furniture, accessories): background, not ignore.
NOT_NEAR: Dict[str, Tuple[str, ...]] = {
    "chair": ("chair cushion", "chair rail", "seat cushion"),
    "sofa": ("sofa table", "sofa cushion", "couch cushion", "couch table"),
    "bed": ("bed table", "bed curtain"),
    "toilet": ("toilet paper", "toilet brush", "toilet paper dispenser", "wall toilet paper", "toilet paper holder",
               "toilet paper roll", "toilet roll holder", "toilet brush holder", "toilet cleaner"),
    "tv_monitor": ("tv stand", "tv cabinet", "tv unit", "tv table", "computer desk", "computer mouse",
                   "computer chair", "display cabinet", "screen door", "fire screen", "monitor stand",
                   "computer table", "computer keyboard", "shower screen"),
    "plant": ("cat tree", "pot lid", "cooking pot", "plant stand", "tree trunk"),
}


def normalise(name: str) -> str:
    """Lower case, '-' / '_' -> ' ', single spaces."""
    return " ".join(name.lower().replace("-", " ").replace("_", " ").split())


def words(name: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalise(name))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counted_names(mapping: Dict[str, str]) -> Dict[str, Set[str]]:
    """Category -> raw labels counted as it (the Phase 8 name mapping)."""
    out: Dict[str, Set[str]] = {c: set() for c in CATEGORIES}
    for name, cat in mapping.items():
        out[cat].add(normalise(name))
    return out


def near_synonyms(home_names: Iterable[str], mapping: Dict[str, str]) -> Dict[str, List[str]]:
    """Category -> sorted raw labels (of the homes) that are near-synonyms (module docstring)."""
    counted = counted_names(mapping)
    names = sorted({normalise(n) for n in home_names if n.strip()})
    out: Dict[str, List[str]] = {}
    for cat in CATEGORIES:
        near = set(NEAR_WORDS[cat])
        out[cat] = [n for n in names if n not in counted[cat] and n not in NOT_NEAR[cat] and near & set(words(n))]
    return out


class LabelTable:
    """For one asked category (6-category test) or one OVON name: which raw labels count, which are ignored.
    `role(name)` -> "counted" | "ignore" | "background"."""

    def __init__(self, key: str, prompt: str, counted: Sequence[str], ignore: Sequence[str]) -> None:
        self.key, self.prompt = key, prompt
        self.counted = frozenset(normalise(n) for n in counted)
        self.ignore = frozenset(normalise(n) for n in ignore) - self.counted

    def role(self, name: str) -> str:
        n = normalise(name)
        if n in self.counted:
            return "counted"
        if n in self.ignore:
            return "ignore"
        return "background"

    def as_dict(self) -> Dict[str, object]:
        return {"key": self.key, "prompt": self.prompt, "counted": sorted(self.counted), "ignore": sorted(self.ignore)}


def six_tables(home_names: Iterable[str], mapping: Dict[str, str]) -> Dict[str, LabelTable]:
    """The 6 categories' tables (block item 2)."""
    counted = counted_names(mapping)
    near = near_synonyms(list(home_names), mapping)
    return {c: LabelTable(c, PROMPTS[c], sorted(counted[c]), near[c]) for c in CATEGORIES}


def ovon_table(name: str, home_names: Iterable[str]) -> LabelTable:
    """One OVON category: counted = raw labels equal to the name; ignore = word-subset relatives (module docstring)."""
    n = normalise(name)
    nw = set(words(n))
    counted, ignore = [], []
    for raw in sorted({normalise(h) for h in home_names if h.strip()}):
        if raw == n:
            counted.append(raw)
        elif nw and (nw <= set(words(raw)) or set(words(raw)) <= nw) and words(raw):
            ignore.append(raw)
    return LabelTable(n, n, counted, ignore)


def slug(name: str) -> str:
    """'potted plant' -> 'potted_plant' (for item ids)."""
    return "_".join(words(name))

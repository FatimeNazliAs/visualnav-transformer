# deeper_visuals/common/build_page.py
"""
Turn a phase's facts.json + PNGs into its advisor page (latest.html).

This never imports torch and never loads the model — that is the whole point of
the two-script split. It reads three things:

    out/<phase>/<tag>/facts.json   the numbers, written by run_model.py
    out/<phase>/<tag>/*.png        the figures, written by run_model.py
    <phase_dir>/page.yaml          the advisor-facing words

so `update.sh --page-only` can re-render wording and layout in well under a
second, on a machine with no GPU, while every number on the page still traces
back to a real forward pass.

The PNGs are base64-embedded rather than linked. A published artifact runs under
a CSP that blocks external hosts, so latest.html has to be one self-contained
file — which also means republishing is a single-file operation.

The page follows the two-surface contract from CLAUDE.md: one plain paragraph,
one hero visual, a small numbers callout, one line on where it sits in the
paper. Depth belongs on the Notion side, not here.

Run:
    python -m deeper_visuals.common.build_page deeper_visuals/p1_inputs
"""

from __future__ import annotations

import argparse
import base64
import html
import re
import sys
from pathlib import Path
from string import Template

import yaml

from deeper_visuals.common.config import load_config
from deeper_visuals.common.facts import read_facts

TEMPLATE_DIR = Path(__file__).resolve().parent / "template"


# ══════════════════════════════════════════════════════════════════════════════
# facts.json access
# ══════════════════════════════════════════════════════════════════════════════

def flatten(d: dict, prefix: str = "") -> dict:
    """
    Flatten nested facts to dotted keys, so page.yaml can reference any of them
    as {scene.traj} or {shapes.c_t} without knowing the nesting.
    """
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


# Deliberately not str.format: it reads "{scene.sample}" as attribute access on
# an object named `scene`, not as a key in the flattened dict. Substituting by
# regex keeps the dotted path meaning exactly what facts.json shows.
_REF = re.compile(r"\{([A-Za-z0-9_.]+)\}")


def fill(text: str, facts_flat: dict) -> str:
    """Substitute {dotted.key} references in a page.yaml string from facts."""

    def sub(m: "re.Match") -> str:
        key = m.group(1)
        if key not in facts_flat:
            raise KeyError(
                f"page.yaml references {{{key}}}, which is not in facts.json. "
                f"Available keys: {', '.join(sorted(facts_flat))}"
            )
        return str(facts_flat[key])

    return _REF.sub(sub, str(text))


# ══════════════════════════════════════════════════════════════════════════════
# Fragment builders
# ══════════════════════════════════════════════════════════════════════════════

def embed_png(path: Path) -> str:
    """Read a PNG and return it as a data: URI."""
    if not path.is_file():
        raise FileNotFoundError(
            f"figure not found: {path} — run update.sh without --page-only first."
        )
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_figures(page: dict, out_dir: Path, facts_flat: dict) -> str:
    """
    Build the <figure> blocks.

    The contract says one hero visual; a phase may add at most one supporting
    figure (P5 does: multimodal hero, three-view detail). More than that and the
    page stops being one screen, so it is refused here rather than in review.
    """
    figures = page.get("figures", [])
    if not figures:
        raise ValueError("page.yaml has no 'figures:' entries")
    if len(figures) > 2:
        raise ValueError(
            f"{len(figures)} figures — the advisor page allows a hero plus at "
            f"most one supporting figure. Move the rest to the Notion group."
        )

    blocks = []
    for fig in figures:
        src = embed_png(out_dir / fig["file"])
        alt = html.escape(fill(fig.get("alt", fig["file"]), facts_flat))
        caption = fig.get("caption")
        cap_html = ""
        if caption:
            # Caption allows <b> for emphasis, so escape then re-admit that tag.
            safe = html.escape(fill(caption, facts_flat))
            safe = safe.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
            cap_html = f"\n    <figcaption>{safe}</figcaption>"
        blocks.append(
            f'  <figure class="figure">\n'
            f'    <div class="plate" tabindex="0">\n'
            f'      <img src="{src}" alt="{alt}">\n'
            f'    </div>{cap_html}\n'
            f'  </figure>'
        )
    return "\n\n".join(blocks)


def render_stats(page: dict, facts_flat: dict) -> str:
    """Build the numbers callout — small by design, 2 to 5 cells."""
    stats = page.get("stats", [])
    if not 2 <= len(stats) <= 5:
        raise ValueError(
            f"{len(stats)} stats — the callout is meant to stay tiny; use 2 to 5."
        )
    rows = []
    for s in stats:
        value = html.escape(fill(s["value"], facts_flat))
        label = html.escape(str(s["label"]))
        rows.append(
            f'    <li><span class="value">{value}</span>'
            f'<span class="label">{label}</span></li>'
        )
    return "\n".join(rows)


def render_provenance(facts: dict) -> str:
    """
    The footer line that makes the page auditable: which scene, which weights,
    when. Advisors ignore it; it is what lets the author trust the figure.
    """
    scene = facts["scene"]
    model = facts.get("model", {})
    items = [
        ("Scene", f"{scene['sample']} ({scene['traj']} @ frame {scene['frame']})"),
        ("Weights", f"{model.get('checkpoint_file', 'n/a')} · {model.get('run', 'n/a')}"),
        ("Generated", facts.get("generated", "n/a")),
    ]
    return "\n".join(
        f"    <span><b>{html.escape(k)}</b> {html.escape(str(v))}</span>"
        for k, v in items
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def build(phase_dir: Path) -> Path:
    phase_dir = Path(phase_dir).resolve()
    cfg = load_config(phase_dir)

    page_path = phase_dir / "page.yaml"
    if not page_path.is_file():
        raise FileNotFoundError(f"no page.yaml in {phase_dir}")
    with open(page_path) as fh:
        page = yaml.safe_load(fh) or {}

    facts = read_facts(cfg.out_dir)
    facts_flat = flatten(facts)

    template = Template((TEMPLATE_DIR / "page.html").read_text())
    styles = (TEMPLATE_DIR / "style.css").read_text()

    rendered = template.substitute(
        title=html.escape(page["title"]),
        phase_label=html.escape(page.get("phase_label", cfg.phase.upper())),
        eyebrow=html.escape(fill(page.get("eyebrow", cfg.description), facts_flat)),
        heading=html.escape(fill(page["heading"], facts_flat)),
        lead=html.escape(fill(page["lead"], facts_flat)),
        paper_note=html.escape(fill(page["paper_note"], facts_flat)),
        figures=render_figures(page, cfg.out_dir, facts_flat),
        stats=render_stats(page, facts_flat),
        provenance=render_provenance(facts),
        styles=styles,
    )

    dest = cfg.out_dir / "latest.html"
    dest.write_text(rendered)
    size_kb = dest.stat().st_size / 1024
    print(f"  Built : {dest}  ({size_kb:.0f} KB)")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="build a phase's advisor page")
    ap.add_argument("phase_dir", help="e.g. deeper_visuals/p1_inputs")
    args = ap.parse_args()
    try:
        build(Path(args.phase_dir))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        sys.exit(f"build_page: {exc}")


if __name__ == "__main__":
    main()

# deeper_visuals/common/build_page.py
"""
Build a phase's advisor page.

Every page in the series is rendered here, through one shell:

    common/template/shell.html   the document — title, fonts, <style>, body slot
    common/template/style.css    the shared look, the single copy of it

What fills the shell's body slot is what differs between phases:

  * A model-backed phase (P1-P5) fills common/template/page.html from its own
    page.yaml (the words) and out/<phase>/<tag>/facts.json (the numbers).
  * A static phase (P0) has no forward pass and so no facts to fill anything
    from. It supplies its body markup directly, as body.html, and may add a
    style.css of its own for markup the shared template does not have.

Routing both through the shell is what keeps the stylesheet in one place. P0
previously carried its own pasted copy with a "re-paste this when style.css
changes" comment, which is the kind of duplication that is correct exactly once.

This module never imports torch and never loads the model. It reads facts.json,
the PNGs and the phase's own files, nothing else — which is what makes
`update.sh --page-only` a sub-second, GPU-free operation while every number on
the page still traces back to a real forward pass.

The PNGs are base64-embedded rather than linked. A published artifact runs under
a CSP that blocks external hosts, so latest.html has to be one self-contained
file — which also makes republishing a single-file operation.

The page follows the two-surface contract from CLAUDE.md: a short heading, one
line of framing, 3-6 bullets, one hero visual, a small numbers callout, one line
on where it sits in the paper — plus an optional collapsed disclosure for the
mechanism a reader asks about only after the page has landed. Everything past
that belongs on the Notion side.

Run:
    python3 -m deeper_visuals.common.build_page deeper_visuals/p1_inputs
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

from deeper_visuals.common.config import OUT_ROOT, load_config, phase_for
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
# The advisor-page contract
# ══════════════════════════════════════════════════════════════════════════════
# The limits ARE the contract, so they are stated together, as data, rather than
# spelled out inside each renderer. An advisor page is scanned in one pass, not
# read; prose is what it drifts back into the moment nothing stops it. Enforcing
# that here means a page that drifts fails the build instead of the review — and
# retuning the contract is one edit to this block, not a hunt through five
# functions.

COPY_LIMITS = {          # field -> maximum characters
    "heading": 60,
    "lead":    150,
    "point":   130,
    "summary": 70,
    # A detail is the one place the page is allowed to say more than a bullet,
    # because it is collapsed by default and the reader chose to open it. Still
    # capped: two lines of explanation, not a paragraph.
    "detail":  200,
}

LIST_LIMITS = {          # field -> (minimum, maximum) entries
    "points":  (3, 6),
    "stats":   (2, 5),
    "figures": (1, 2),
    "details": (3, 8),
}

CONTRACT_HINT = (
    "The advisor page is scanned in one pass: short heading, one line of "
    "framing, then points. A little more depth can go in the collapsed "
    "`details:` block; anything past that belongs in the phase's Notion group."
)


def one_line(value: object) -> str:
    """
    Collapse a YAML scalar to a single line.

    Copy is written with `key: >` for readability in page.yaml, which keeps the
    fold's internal line breaks and a trailing newline. Left alone those land in
    the HTML — inside an alt attribute, or as stray whitespace before a closing
    tag — so every piece of copy passes through here first.
    """
    return " ".join(str(value).split())


_BOLD = re.compile(r"</?b>")


def within_length(field: str, text: str) -> str:
    """
    Check a piece of copy against its cap.

    Measured on the text a reader actually sees: after {dotted.key} substitution
    (so a cap is not spent on the length of a reference) and ignoring <b> markup
    (so bolding a term cannot trip it).
    """
    visible = _BOLD.sub("", text)
    limit = COPY_LIMITS[field]
    if len(visible) > limit:
        raise ValueError(
            f"{field} is {len(visible)} characters; keep it under {limit}. "
            f"{CONTRACT_HINT}\n  {visible[:70]}…"
        )
    return text


def within_count(field: str, items: list) -> list:
    low, high = LIST_LIMITS[field]
    if not low <= len(items) <= high:
        raise ValueError(
            f"{len(items)} {field} — use {low} to {high}. {CONTRACT_HINT}"
        )
    return items


# ══════════════════════════════════════════════════════════════════════════════
# Fragment builders
# ══════════════════════════════════════════════════════════════════════════════

def resolve(value: object, facts_flat: dict) -> str:
    """
    Fold a YAML scalar to one line and substitute {dotted.key} from facts.

    This is the text a reader sees, before escaping — which is what the contract
    caps are measured against.
    """
    return fill(one_line(value), facts_flat)


def as_text(text: str) -> str:
    """Escape resolved copy for HTML. Any <b> in it becomes literal."""
    return html.escape(text)


def as_rich(text: str) -> str:
    """
    Escape resolved copy, then re-admit <b>.

    Copy is escaped by default because it is YAML text going into HTML; <b> is
    the one tag worth having for the key term in a sentence.
    """
    return (html.escape(text)
            .replace("&lt;b&gt;", "<b>")
            .replace("&lt;/b&gt;", "</b>"))


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
    Build the <figure> blocks: a hero, plus at most one supporting figure.

    P5 is the case for two — multimodal hero, three-view detail. More than that
    and the page stops being one screen.
    """
    blocks = []
    for fig in within_count("figures", page.get("figures", [])):
        src = embed_png(out_dir / fig["file"])
        # alt is an attribute, so it never gets <b> — plain, not rich.
        alt = as_text(resolve(fig.get("alt", fig["file"]), facts_flat))
        caption = fig.get("caption")
        cap_html = (
            f'\n    <figcaption>{as_rich(resolve(caption, facts_flat))}</figcaption>'
            if caption else ""
        )
        blocks.append(
            f'  <figure class="figure">\n'
            f'    <div class="plate" tabindex="0">\n'
            f'      <img src="{src}" alt="{alt}">\n'
            f'    </div>{cap_html}'
            f'{render_details(fig, facts_flat)}\n'
            f'  </figure>'
        )
    return "\n\n".join(blocks)


def render_heading(page: dict, facts_flat: dict) -> str:
    """The page's claim, as a headline rather than a sentence."""
    return as_text(within_length("heading", resolve(page["heading"], facts_flat)))


def render_lead(page: dict, facts_flat: dict) -> str:
    """The single line of framing under the heading."""
    return as_rich(within_length("lead", resolve(page["lead"], facts_flat)))


def render_points(page: dict, facts_flat: dict) -> str:
    """The bullets that carry the page's content."""
    points = within_count("points", page.get("points", []))
    return "\n".join(
        f'      <li>'
        f'{as_rich(within_length("point", resolve(p, facts_flat)))}'
        f'</li>'
        for p in points
    )


def render_stats(page: dict, facts_flat: dict) -> str:
    """The numbers callout — small by design."""
    return "\n".join(
        f'    <li><span class="value">{as_text(resolve(s["value"], facts_flat))}</span>'
        f'<span class="label">{as_text(resolve(s["label"], facts_flat))}</span></li>'
        for s in within_count("stats", page.get("stats", []))
    )


def render_details(figure: dict, facts_flat: dict) -> str:
    """
    A figure's optional collapsed disclosure, rendered directly beneath it.

    The advisor page is scanned in one pass, which is why the visible part is
    capped at 3-6 short bullets. That cap kept pushing out one specific kind of
    content: what a reader wants *after* a picture has caught them — how to read
    it, what the parts of it mean, why it was made this way. Sending them to
    Notion for that means losing them. Collapsed, it costs the scan nothing.

    It hangs off the figure rather than the page because that is the question it
    answers: not "tell me more about this phase" but "what am I looking at". A
    page with two figures needs two different answers, and a disclosure parked
    at the bottom cannot say which picture it belongs to.

    A figure with nothing to disclose omits the key and the section vanishes;
    it is not a required part of the contract.
    """
    block = figure.get("details")
    if not block:
        return ""

    items = "\n".join(
        f'        <li>'
        f'{as_rich(within_length("detail", resolve(d, facts_flat)))}'
        f'</li>'
        for d in within_count("details", block.get("points", []))
    )
    summary = as_rich(within_length("summary", resolve(block["summary"], facts_flat)))
    return (f'\n    <details class="more">\n'
            f'      <summary>{summary}</summary>\n'
            f'      <ul>\n{items}\n      </ul>\n'
            f'    </details>')


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
# Body sources
# ══════════════════════════════════════════════════════════════════════════════

def render_phase_body(page: dict, out_dir: Path, phase: str,
                      eyebrow_default: str) -> str:
    """Fill the shared page template from page.yaml + facts.json."""
    facts = read_facts(out_dir)
    facts_flat = flatten(facts)
    template = Template((TEMPLATE_DIR / "page.html").read_text())
    return template.substitute(
        phase_label=as_text(one_line(page.get("phase_label", phase.upper()))),
        eyebrow=as_text(resolve(page.get("eyebrow", eyebrow_default), facts_flat)),
        heading=render_heading(page, facts_flat),
        lead=render_lead(page, facts_flat),
        points=render_points(page, facts_flat),
        paper_note=as_text(resolve(page["paper_note"], facts_flat)),
        figures=render_figures(page, out_dir, facts_flat),
        stats=render_stats(page, facts_flat),
        provenance=render_provenance(facts),
    )


def read_body_file(phase_dir: Path, name: str) -> str:
    """
    A static phase's hand-written markup, taken as-is.

    Nothing is substituted into it. A phase writes body.html precisely because
    it has no facts.json to substitute from, and a {brace} in hand-written HTML
    should stay a brace.
    """
    path = phase_dir / name
    if not path.is_file():
        raise FileNotFoundError(
            f"page.yaml names body: {name}, but {path} does not exist"
        )
    return path.read_text().strip()


def collect_styles(phase_dir: Path) -> str:
    """
    The shared sheet, then the phase's own additions if it has any.

    Order matters: the phase file extends the shared tokens, so it has to come
    second. Only a phase with markup the shared template does not produce needs
    one at all — P0's diagram and mode cards are the only case so far.
    """
    sheets = [(TEMPLATE_DIR / "style.css").read_text()]
    phase_sheet = phase_dir / "style.css"
    if phase_sheet.is_file():
        sheets.append(phase_sheet.read_text())
    return "\n\n".join(sheets)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def read_page_yaml(phase_dir: Path) -> dict:
    path = phase_dir / "page.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no page.yaml in {phase_dir}")
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def build(phase_dir: Path) -> Path:
    """
    Render <phase_dir> to its page, promote it, and return the served path.

    Two directories are written on purpose:

        out/<phase>/<tag>/latest.html   the build, kept per checkpoint so
                                        switching `checkpoint:` never clobbers
                                        the other variant
        out/<phase>/latest.html         the promoted copy — the one stable path
                                        the preview server and every republish
                                        point at

    A static phase has no checkpoint to tag with, so its build directory is
    out/<phase>/ and the promotion is a no-op.

    Deciding this here rather than in update.sh means the out/ layout is
    described in exactly one language. When bash also knew the rule, the two
    could disagree, and a disagreement sends the forward pass and the page
    builder to different directories — which fails confusingly.
    """
    phase_dir = Path(phase_dir).resolve()
    phase = phase_for(phase_dir)
    page = read_page_yaml(phase_dir)

    if body_file := page.get("body"):
        out_dir = OUT_ROOT / phase
        body = read_body_file(phase_dir, body_file)
    else:
        cfg = load_config(phase_dir)
        out_dir = cfg.out_dir
        body = render_phase_body(page, out_dir, phase, cfg.description)

    shell = Template((TEMPLATE_DIR / "shell.html").read_text())
    rendered = shell.substitute(
        title=html.escape(page["title"]),
        styles=collect_styles(phase_dir),
        body=body,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    built = out_dir / "latest.html"
    built.write_text(rendered)
    print(f"  Built : {built}  ({built.stat().st_size / 1024:.0f} KB)")

    served = OUT_ROOT / phase / "latest.html"
    if served != built:
        served.write_text(rendered)
    print(f"  Served: {served}")
    return served


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

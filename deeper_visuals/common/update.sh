#!/usr/bin/env bash
# deeper_visuals/common/update.sh
#
# The driver behind every phase's update.sh. One command, four steps:
#
#   1. run_model.py   forward pass on the real checkpoint -> facts.json + PNGs
#   2. build_page.py  facts.json + PNGs -> out/<phase>/<tag>/latest.html
#   3. promote        copy that page to out/<phase>/latest.html (the served path)
#   4. serve          http.server on $PORT so the page previews in the browser
#
# Step 1 is the only one that needs the GPU, and it is the only one --page-only
# skips — which is what makes wording and layout iteration fast.
#
# Usage (from the repo root, inside the container):
#   ./deeper_visuals/common/update.sh deeper_visuals/p1_inputs
#   ./deeper_visuals/common/update.sh deeper_visuals/p1_inputs --page-only
#   ./deeper_visuals/common/update.sh deeper_visuals/p0_overview --static
#
# Flags:
#   --page-only   skip run_model.py (no GPU); rebuild the page from existing facts
#   --static      no model and no facts at all — the phase ships a hand-written
#                 latest.html (P0). Promotes and serves it.
#   --no-serve    do everything except start the preview server
#   --port N      preview port (default 8001; 8000 is taken by NWM on this host)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="$REPO_ROOT/deeper_visuals/out"
PORT="${PORT:-8001}"

PAGE_ONLY=0
STATIC=0
SERVE=1
PHASE_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --page-only) PAGE_ONLY=1; shift ;;
    --static)    STATIC=1; shift ;;
    --no-serve)  SERVE=0; shift ;;
    --port)      PORT="$2"; shift 2 ;;
    -h|--help)   sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)          echo "unknown flag: $1" >&2; exit 2 ;;
    *)           PHASE_DIR="$1"; shift ;;
  esac
done

if [[ -z "$PHASE_DIR" ]]; then
  echo "usage: $(basename "$0") <phase_dir> [--page-only|--static] [--no-serve] [--port N]" >&2
  exit 2
fi

# Accept either an absolute path or one relative to the repo root.
[[ "$PHASE_DIR" = /* ]] || PHASE_DIR="$REPO_ROOT/$PHASE_DIR"
PHASE_DIR="${PHASE_DIR%/}"
[[ -d "$PHASE_DIR" ]] || { echo "no such phase folder: $PHASE_DIR" >&2; exit 1; }


cd "$REPO_ROOT"

# run_model.py is executed by path, so Python puts the phase folder on sys.path
# rather than the repo root — `import deeper_visuals` would miss without this.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

rule() { printf '%s\n' "──────────────────────────────────────────────────────────────"; }

# ── Steps 1–2: generate ───────────────────────────────────────────────────────
# Phase id and checkpoint tag both come from common/config.py, so bash never
# keeps its own copy of a rule that Python also implements.
PHASE="$(python -c "
from deeper_visuals.common.config import phase_for
print(phase_for('$PHASE_DIR'))
")"

if [[ $STATIC -eq 1 ]]; then
  echo "  [static] $PHASE — hand-written page, no model, no facts"
  SRC="$PHASE_DIR/latest.html"
  [[ -f "$SRC" ]] || { echo "  missing $SRC" >&2; exit 1; }
else
  # Which checkpoint the phase is on decides which out/ subdir holds the page.
  TAG="$(python -c "
from deeper_visuals.common.config import load_config
print(load_config('$PHASE_DIR').checkpoint_tag)
")"

  if [[ $PAGE_ONLY -eq 0 ]]; then
    rule; echo "  1/3  run_model.py — forward pass ($PHASE)"; rule
    python "$PHASE_DIR/run_model.py"
  else
    echo "  [--page-only] skipping the forward pass; reusing existing facts.json"
  fi

  rule; echo "  2/3  build_page.py — facts + PNGs -> latest.html"; rule
  python -m deeper_visuals.common.build_page "$PHASE_DIR"
  SRC="$OUT_ROOT/$PHASE/$TAG/latest.html"
fi

# ── Step 3: promote to the served path ────────────────────────────────────────
rule; echo "  3/3  promote -> out/$PHASE/latest.html"; rule
mkdir -p "$OUT_ROOT/$PHASE"
cp "$SRC" "$OUT_ROOT/$PHASE/latest.html"
echo "  Served page : deeper_visuals/out/$PHASE/latest.html"

# The container runs as root, so everything under out/ lands root-owned and the
# host user cannot delete it. Generated output should never need sudo.
chmod -R a+rwX "$OUT_ROOT" 2>/dev/null || true

# ── Step 4: serve ─────────────────────────────────────────────────────────────
if [[ $SERVE -eq 1 ]]; then
  if curl -s -o /dev/null --max-time 1 "http://localhost:$PORT/" 2>/dev/null; then
    echo "  Preview     : already serving on :$PORT — just refresh"
  else
    echo "  Preview     : starting http.server on :$PORT"
    ( cd "$OUT_ROOT" && exec python -m http.server "$PORT" ) &
    sleep 1
  fi
  echo "  Open        : http://localhost:$PORT/$PHASE/latest.html"
  echo
  echo "  To publish  : in this phase's Claude Code chat, say"
  echo "                republish $PHASE/latest.html -> <phase's fixed artifact URL>"
fi

#!/usr/bin/env bash
# deeper_visuals/common/update.sh
#
# The driver behind every phase's update.sh. Run it from the HOST:
#
#   ./deeper_visuals/p1_inputs/update.sh
#   ./deeper_visuals/p1_inputs/update.sh --page-only
#
# You do not enter the container yourself. This script does the docker/conda
# dance for you — the same arrangement as the NWM walkthrough — because the one
# thing you actually do per iteration is edit a config.yaml, and anything else
# between that edit and seeing the page is friction.
#
# It also works unchanged from *inside* the container: if /.dockerenv is there,
# the steps run in place instead of shelling into anything.
#
# Three steps:
#
#   1. run_model.py   forward pass on the real checkpoint -> facts.json + PNGs
#   2. build_page.py  facts.json + PNGs -> out/<phase>/latest.html
#   3. serve          http.server on $PORT so the page previews in the browser
#
# Step 1 is the only one that needs the checkpoint, and the only one --page-only
# skips — which is what makes wording and layout iteration fast. A phase with no
# run_model.py at all (P0, the hand-written primer) skips it every time; there
# is no flag for that, the absence of the file is the fact.
#
# Where the page lands is build_page.py's decision, not this script's — see the
# docstring on build_page.build. Bash keeping its own copy of the out/ layout
# meant two rules that could disagree.
#
# Flags:
#   --page-only   skip run_model.py; rebuild the page from the existing facts
#   --no-serve    do everything except start the preview server
#   --port N      preview port (default 8001; 8000 is taken by NWM on this host)
#
# Env overrides: PORT, CONTAINER, PY

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="$REPO_ROOT/deeper_visuals/out"
PORT="${PORT:-8001}"
PY="${PY:-python3}"

# The container the forward pass runs in, and where this repo is mounted inside
# it (see deeper_visuals/README.md for the docker run that creates it).
CONTAINER="${CONTAINER:-naz_nomad_deeper_viz}"
CONTAINER_REPO="/app/visualnav-transformer"
CONDA_ENV="vint_train"
ARTIFACTS="$REPO_ROOT/deeper_visuals/common/artifacts.yaml"

PAGE_ONLY=0
SERVE=1
PHASE_DIR=""

# Prints the header block above, whatever length it grows to: every comment
# line after the shebang, stopping at the first line that is not one. A fixed
# line range drifted the moment the header changed and started printing code.
usage() { awk 'NR == 1 {next} /^#/ {print; next} {exit}' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --page-only) PAGE_ONLY=1; shift ;;
    --no-serve)  SERVE=0; shift ;;
    --port)      PORT="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    -*)          echo "unknown flag: $1" >&2; exit 2 ;;
    *)           PHASE_DIR="$1"; shift ;;
  esac
done

if [[ -z "$PHASE_DIR" ]]; then
  echo "usage: $(basename "$0") <phase_dir> [--page-only] [--no-serve] [--port N]" >&2
  exit 2
fi

# Resolve the phase folder against the current directory first, then the repo
# root. Both spellings have to work: a phase's update.sh passes its own
# `dirname $BASH_SOURCE`, which is relative to wherever the user invoked it,
# while the documented form is a repo-root-relative path.
resolve_phase_dir() {
  local candidate
  for candidate in "$1" "$REPO_ROOT/$1"; do
    if [[ -d "$candidate" ]]; then
      (cd "$candidate" && pwd)
      return 0
    fi
  done
  return 1
}

PHASE_DIR="$(resolve_phase_dir "${PHASE_DIR%/}")" || {
  echo "no such phase folder: $PHASE_DIR" >&2; exit 1
}
# Everything handed across the container boundary is repo-relative, because the
# repo sits at a different absolute path on each side of it.
PHASE_REL="${PHASE_DIR#"$REPO_ROOT"/}"

rule() { printf '%s\n' "──────────────────────────────────────────────────────────────"; }


# ── Running things against the repo, here or in the container ────────────────

inside_container() { [[ -f /.dockerenv ]]; }

require_container() {
  docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER" && return 0
  {
    echo
    echo "  The '$CONTAINER' container is not running, and the forward pass"
    echo "  needs it — numpy, torch and matplotlib live in its conda env, not"
    echo "  on the host."
    echo
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
      echo "  It exists but is stopped. Start it with:"
      echo "      docker start $CONTAINER"
    else
      echo "  It does not exist yet. Create it with:"
      echo "      screen -S nomad_deeper_viz"
      echo "      docker run --gpus all --shm-size=8g -it --name $CONTAINER \\"
      echo "        -v /mnt/shared_disk/nazli/nomad_data:/data \\"
      echo "        -v /mnt/shared_disk/nazli/nomad_outputs:/outputs \\"
      echo "        -v $REPO_ROOT:$CONTAINER_REPO \\"
      echo "        -v /mnt/shared_disk/nazli/wandb_config:/root/.config/wandb \\"
      echo "        -p $PORT:$PORT \\"
      echo "        nomad:latest bash"
    fi
    echo
  } >&2
  exit 1
}

# Run a shell snippet with the repo as the working directory and importable.
# On the host that means shelling into the container; inside it, in place.
run_in_repo() {
  local snippet="$1"
  if inside_container; then
    ( cd "$REPO_ROOT" && export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
                                PHASE_REL="$PHASE_REL" && eval "$snippet" )
  else
    require_container
    docker exec -e PHASE_REL="$PHASE_REL" "$CONTAINER" bash -c "
      set -e
      source /opt/conda/etc/profile.d/conda.sh
      conda activate $CONDA_ENV
      cd $CONTAINER_REPO
      export PYTHONPATH=$CONTAINER_REPO
      $snippet
    "
  fi
}

# The phase id comes from common/config.py so bash never keeps its own copy of a
# rule that Python also implements. Run here rather than through run_in_repo:
# phase_for is a pure string operation on a path, it needs nothing but the
# standard library, and routing it through the container cost a docker round
# trip on every run — most of the wall time of the --page-only loop.
PHASE="$(cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT" PHASE_REL="$PHASE_REL" "$PY" -c 'import os
from deeper_visuals.common.config import phase_for
print(phase_for(os.environ["PHASE_REL"]))')"


# ── The scene, echoed back before anything slow happens ──────────────────────
# So a typo in config.yaml is visible immediately, not after the model has run.
if [[ -f "$PHASE_DIR/config.yaml" ]]; then
  echo
  echo "  scene from $PHASE_REL/config.yaml"
  # Every key any phase reads. It listed num_seeds — a key no phase has — while
  # omitting seed, which P4 does read, so the one phase-specific knob in the
  # repo was the one this echo could not show you.
  grep -E '^(sample|checkpoint|seed|num_seeds):' "$PHASE_DIR/config.yaml" | sed 's/^/      /' || true
fi

# ── Step 1: the forward pass ──────────────────────────────────────────────────
echo
if [[ ! -f "$PHASE_DIR/run_model.py" ]]; then
  echo "  [$PHASE] no run_model.py — hand-written page, nothing to run"
elif [[ $PAGE_ONLY -eq 1 ]]; then
  echo "  [--page-only] skipping the forward pass; reusing the existing facts.json"
else
  rule; echo "  1/2  run_model.py — forward pass ($PHASE)"; rule
  run_in_repo "$PY \$PHASE_REL/run_model.py"
fi

# ── Step 2: the page ──────────────────────────────────────────────────────────
rule; echo "  2/2  build_page.py — page -> out/$PHASE/latest.html"; rule
run_in_repo "$PY -m deeper_visuals.common.build_page \$PHASE_REL"

# The container runs as root, so anything it wrote under out/ lands root-owned
# and the host user cannot delete it. Generated output should never need sudo.
chmod -R a+rwX "$OUT_ROOT" 2>/dev/null || true

# ── Step 3: serve ─────────────────────────────────────────────────────────────
# Served from wherever this script is running. out/ is bind-mounted, so the host
# and the container are looking at the same files either way.
if [[ $SERVE -eq 1 ]]; then
  if curl -s -o /dev/null --max-time 1 "http://localhost:$PORT/" 2>/dev/null; then
    echo "  Preview     : already serving on :$PORT — just refresh the tab"
  else
    echo "  Preview     : starting http.server on :$PORT"
    # setsid with all three streams detached, or the server inherits this
    # script's stdout and anything piping us (| tee, | tail) hangs on EOF.
    ( cd "$OUT_ROOT" && setsid nohup "$PY" -m http.server "$PORT" \
        </dev/null >/dev/null 2>&1 & )
    sleep 1
  fi
  echo "  Open        : http://localhost:$PORT/$PHASE/latest.html"
  echo

  # The registry is a flat "phase: url" map on purpose — it keeps this lookup a
  # one-line sed, so printing the URL costs no python and no container.
  ARTIFACT_URL="$(sed -n "s|^$PHASE: *||p" "$ARTIFACTS" 2>/dev/null || true)"
  if [[ -n "$ARTIFACT_URL" ]]; then
    echo "  To publish  : say \"republish $PHASE webpage\" in this phase's Claude Code chat"
    echo "                -> $ARTIFACT_URL"
  else
    echo "  To publish  : say \"publish $PHASE webpage\" in this phase's Claude Code chat."
    echo "                $PHASE has no artifact yet; the URL it returns gets recorded"
    echo "                in common/artifacts.yaml and never changes after that."
  fi
fi

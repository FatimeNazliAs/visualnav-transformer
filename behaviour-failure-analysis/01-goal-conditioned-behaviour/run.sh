#!/usr/bin/env bash
# 01-goal-conditioned-behaviour/run.sh
#
# Run a script from this task inside the container, from the HOST:
#
#   ./behaviour-failure-analysis/01-goal-conditioned-behaviour/run.sh sanity_check.py
#   ./behaviour-failure-analysis/01-goal-conditioned-behaviour/run.sh sanity_check.py --traj no31vc_12_0
#
# You do not enter the container yourself. This does the docker exec + conda
# activate, mirroring deeper_visuals/common/update.sh, because the thing you
# actually do per iteration is edit a config.yaml and anything between that and
# a result is friction.
#
# It also works unchanged from INSIDE the container: if /.dockerenv is there,
# the script runs in place instead of shelling into anything.
#
# Env overrides: CONTAINER, PY

set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TASK_DIR/../.." && pwd)"
PROJECT_REL="behaviour-failure-analysis"
TASK_REL="${TASK_DIR#"$REPO_ROOT"/}"

CONTAINER="${CONTAINER:-naz_nomad_failure_analysis}"
CONTAINER_REPO="/app/visualnav-transformer"
CONDA_ENV="vint_train"
PY="${PY:-python}"

if [[ $# -lt 1 ]]; then
  echo "usage: $(basename "$0") <script.py> [args…]" >&2
  echo "  e.g. $(basename "$0") sanity_check.py" >&2
  exit 2
fi

SCRIPT="$1"; shift

# Quote the remaining args for the shell inside the container. Built here rather
# than inline, because `printf '%q ' "$@"` with zero arguments still applies the
# format once and emits an empty quoted string — which the script then receives
# as a stray argument and rejects.
SCRIPT_ARGS=""
if [[ $# -gt 0 ]]; then SCRIPT_ARGS="$(printf '%q ' "$@")"; fi
if [[ ! -f "$TASK_DIR/$SCRIPT" ]]; then
  echo "no such script in $TASK_REL: $SCRIPT" >&2; exit 1
fi

inside_container() { [[ -f /.dockerenv ]]; }

require_container() {
  docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER" && return 0
  {
    echo
    echo "  The '$CONTAINER' container is not running, and the forward pass"
    echo "  needs it — torch, diffusers and matplotlib live in its conda env,"
    echo "  not on the host."
    echo
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
      echo "  It exists but is stopped. Start it with:"
      echo "      docker start $CONTAINER"
    else
      echo "  It does not exist yet. Create it with:"
      echo "      docker run -dit --name $CONTAINER \\"
      echo "        --gpus all --shm-size=8g \\"
      echo "        -v /mnt/shared_disk/nazli/nomad_data:/data \\"
      echo "        -v /mnt/shared_disk/nazli/nomad_outputs:/outputs \\"
      echo "        -v $REPO_ROOT:$CONTAINER_REPO \\"
      echo "        -v /mnt/shared_disk/nazli/wandb_config:/root/.config/wandb \\"
      echo "        nomad:latest bash"
    fi
    echo
  } >&2
  exit 1
}

if inside_container; then
  cd "$REPO_ROOT"
  PYTHONPATH="$REPO_ROOT/$PROJECT_REL${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" "$TASK_REL/$SCRIPT" "$@"
else
  require_container
  docker exec "$CONTAINER" bash -c "
    set -e
    source /opt/conda/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    cd $CONTAINER_REPO
    export PYTHONPATH=$CONTAINER_REPO/$PROJECT_REL
    $PY $TASK_REL/$SCRIPT $SCRIPT_ARGS
  "
  # The container runs as root, so anything it writes under the results tree
  # lands root-owned. Generated output should never need sudo to delete.
  RESULTS="/mnt/shared_disk/nazli/nomad_outputs/behaviour_analysis"
  [[ -d "$RESULTS" ]] && chmod -R a+rwX "$RESULTS" 2>/dev/null || true
fi

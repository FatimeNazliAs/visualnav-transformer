#!/usr/bin/env bash
# ==============================================================================
# sim_eval/sim_up.sh — bring up this workstream's simulator container.
#
# Builds nomad_sim:latest if it is missing, makes sure the `nomad_sim` screen
# session exists, and creates (or restarts) the `naz_nomad_sim` container.
# Idempotent: safe to run when everything is already up.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/sim_up.sh [--rebuild]
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source sim_eval/lib.sh

REBUILD=0
[ "${1:-}" = "--rebuild" ] && REBUILD=1

# --- image --------------------------------------------------------------------
if [ "$REBUILD" = 1 ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "==> building $IMAGE (this compiles iGibson's renderer; several minutes)"
    docker build -f sim_eval/sim.Dockerfile -t "$IMAGE" .
else
    echo "==> image $IMAGE already built"
fi

# --- screen session -----------------------------------------------------------
# Long rollouts must survive an SSH drop, so this workstream gets its own
# session. Never reuse another one.
if screen -ls | grep -q "[.]${SCREEN_SESSION}[[:space:]]"; then
    echo "==> screen session $SCREEN_SESSION already exists"
else
    screen -dmS "$SCREEN_SESSION"
    echo "==> created screen session $SCREEN_SESSION"
fi

# --- container ----------------------------------------------------------------
mkdir -p "$HOST_IGIBSON_DATA"

if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "==> container $CONTAINER already running"
elif docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "==> starting existing container $CONTAINER"
    docker start "$CONTAINER" >/dev/null
else
    echo "==> creating container $CONTAINER"
    # No --rm, by policy: this container is long-lived and must not evaporate
    # when a command exits. `sleep infinity` keeps it up for docker exec.
    docker run -d \
        --name "$CONTAINER" \
        --gpus all \
        --shm-size=8g \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -v "$HOST_REPO:$CONTAINER_REPO" \
        -v "$HOST_DATA:/data" \
        -v "$HOST_OUTPUTS:/outputs" \
        -v "$HOST_IGIBSON_DATA:/igibson_data" \
        "$IMAGE" sleep infinity >/dev/null
fi

echo
docker ps --filter "name=$CONTAINER" --format 'container: {{.Names}} ({{.Image}}) — {{.Status}}'
echo "screen:    $SCREEN_SESSION"
echo "GPU:       $SIM_GPU (override with SIM_GPU=N)"
echo
echo "Next: ./sim_eval/download_assets.sh"

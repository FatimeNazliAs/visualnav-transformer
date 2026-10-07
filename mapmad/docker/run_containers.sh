#!/bin/bash
# Build and run MapMaD's two containers (plan §0.7). Every docker command MapMaD needs lives here.
#
#   mapmad/docker/run_containers.sh build     # image naz-mapmad-habitat:<version> (~10 min, run it in screen)
#   mapmad/docker/run_containers.sh up        # network + both containers (skips what already exists)
#   mapmad/docker/run_containers.sh check     # GPUs seen by both containers
#   mapmad/docker/run_containers.sh down      # stop + remove naz_mapmad and naz_mapmad_habitat only
#
# Single steps: network | habitat | nomad. Host paths come from paths.env (override with env vars).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck source=paths.env
source "$HERE/paths.env"

HABITAT_CONTAINER=naz_mapmad_habitat
NOMAD_CONTAINER=naz_mapmad
# Run as Naz, not root, so files written to the shared disk are hers. HOME must be writable.
USER_ARGS=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
# The worktree's .git is a pointer into the main repo's git folder; mount that folder read-only at the
# same path so `git rev-parse HEAD` works inside the containers (every run records its commit, §0.8).
GIT_COMMON_DIR="$(cd "$REPO" && cd "$(git rev-parse --git-common-dir)" && pwd)"
GIT_ARGS=(-v "$GIT_COMMON_DIR":"$GIT_COMMON_DIR":ro)

run() { echo "+ $*" >&2; "$@"; }
exists() { docker container inspect "$1" >/dev/null 2>&1; }

make_host_dirs() {
    # Created here, owned by Naz: otherwise docker creates missing mount points as root.
    mkdir -p "$MAPMAD_DATA" "$NOMAD_OUTPUTS/mapmad"
}

cmd_network() {
    docker network inspect "$MAPMAD_NETWORK" >/dev/null 2>&1 \
        && echo "network $MAPMAD_NETWORK exists" \
        || run docker network create "$MAPMAD_NETWORK"
}

cmd_build() {
    run docker build \
        -f "$HERE/Dockerfile.habitat" \
        --build-context mapmad="$REPO/mapmad" \
        -t "$HABITAT_IMAGE" \
        "$HABITAT_STARTER"
}

cmd_habitat() {
    if exists "$HABITAT_CONTAINER"; then echo "$HABITAT_CONTAINER exists"; return; fi
    make_host_dirs
    run docker run -d --name "$HABITAT_CONTAINER" \
        --network "$MAPMAD_NETWORK" \
        --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        "${USER_ARGS[@]}" --shm-size 8g \
        -v "$REPO":/app/visualnav-transformer "${GIT_ARGS[@]}" \
        -v "$HM3D_SCENES":/hm3d:ro \
        -v "$HM3D_SEMANTICS":/hm3d_semantics:ro \
        -v "$OBJECTNAV_V2":/objectnav:ro \
        -v "$MAPMAD_DATA":/mapmad_data \
        -v "$NOMAD_OUTPUTS":/outputs \
        -v "$HABITAT_STARTER/data":/opt/habitat-starter/data:ro \
        "$HABITAT_IMAGE" sleep infinity
}

cmd_nomad() {
    if exists "$NOMAD_CONTAINER"; then echo "$NOMAD_CONTAINER exists"; return; fi
    make_host_dirs
    run docker run -d --name "$NOMAD_CONTAINER" \
        --network "$MAPMAD_NETWORK" \
        --gpus all \
        "${USER_ARGS[@]}" --shm-size 8g \
        -v "$REPO":/app/visualnav-transformer "${GIT_ARGS[@]}" \
        -v "$NOMAD_DATA":/data \
        -v "$NOMAD_OUTPUTS":/outputs \
        -v "$MAPMAD_DATA":/mapmad_data \
        -v "$WANDB_CONFIG":/tmp/.config/wandb \
        "$NOMAD_IMAGE" sleep infinity
    # extra NoMaD-side packages (train/requirements-mapmad.txt), installed into the user's site dir
    if grep -qvE '^\s*(#|$)' "$REPO/train/requirements-mapmad.txt"; then
        run docker exec "$NOMAD_CONTAINER" pip install --user -r /app/visualnav-transformer/train/requirements-mapmad.txt
    fi
}

cmd_up() { cmd_network; cmd_habitat; cmd_nomad; }

cmd_down() {
    for c in "$HABITAT_CONTAINER" "$NOMAD_CONTAINER"; do
        if exists "$c"; then run docker rm -f "$c"; else echo "$c not present"; fi
    done
}

cmd_check() {
    run docker exec "$HABITAT_CONTAINER" nvidia-smi -L
    run docker exec "$HABITAT_CONTAINER" python -c \
        "import habitat_sim, habitat_starter.sim as s; print('habitat-sim', habitat_sim.__version__, '| NVIDIA EGL driver:', s.nvidia_egl_available())"
    run docker exec "$NOMAD_CONTAINER" python -c \
        "import torch; n = torch.cuda.device_count(); print('torch', torch.__version__, '| cuda devices:', n, [torch.cuda.get_device_name(i) for i in range(n)])"
    docker network inspect "$MAPMAD_NETWORK" --format '{{range .Containers}}{{.Name}} {{end}}' | xargs echo "on $MAPMAD_NETWORK:"
}

case "${1:-}" in
    network | build | habitat | nomad | up | down | check) "cmd_$1" ;;
    *) sed -n '2,10p' "$0"; exit 1 ;;
esac

# ==============================================================================
# sim_eval/lib.sh — shared settings and the one way into the container.
#
# Sourced by the sim_eval scripts; not executable on its own. Everything runs
# from the host, from the repo root: the scripts shell into the container
# themselves, so nobody has to `docker exec` by hand.
#
# `cukurovaai` is a shared machine. The names below are owner-prefixed and
# dedicated to this workstream — these scripts must never touch another
# container, screen or process.
# ==============================================================================

IMAGE=nomad_sim:latest
CONTAINER=naz_nomad_sim
SCREEN_SESSION=nomad_sim

HOST_REPO=/home/nazli/projects/nomad
CONTAINER_REPO=/app/visualnav-transformer

# Big, shared-disk-resident state. None of it belongs in the image or the repo.
HOST_DATA=/mnt/shared_disk/nazli/nomad_data
HOST_OUTPUTS=/mnt/shared_disk/nazli/nomad_outputs
HOST_IGIBSON_DATA=/mnt/shared_disk/nazli/igibson_data

# Which GPU this workstream may use. Override per invocation:  SIM_GPU=0 ./...
# Default 1: GPU 0 also drives the machine's X server, GPU 1 is bare.
# Always check `nvidia-smi` before claiming one — teammates share this box.
SIM_GPU=${SIM_GPU:-1}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_container_running() {
    docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" \
        || die "container '$CONTAINER' is not running. Start it with: ./sim_eval/sim_up.sh"
}

# Run a command inside the sim container, in the vint_train env, pinned to one
# GPU. Works whether or not the caller is already inside the container.
#
# Both variables are needed and they are not interchangeable:
#   CUDA_VISIBLE_DEVICES  restricts torch (the policy, from P1 on).
#   GIBSON_DEVICE_ID      picks the GPU that EGL renders on.
# iGibson enumerates render devices through the driver, which does NOT respect
# CUDA_VISIBLE_DEVICES — set only that, and the renderer happily lands on GPU 0
# no matter what you asked for. Verified: it did exactly that. GIBSON_DEVICE_ID
# is a physical device minor, so it is deliberately *not* remapped.
sim_exec() {
    if [ -f /.dockerenv ]; then
        CUDA_VISIBLE_DEVICES="$SIM_GPU" GIBSON_DEVICE_ID="$SIM_GPU" bash -lc "$*"
    else
        require_container_running
        docker exec \
            -e CUDA_VISIBLE_DEVICES="$SIM_GPU" \
            -e GIBSON_DEVICE_ID="$SIM_GPU" \
            -w "$CONTAINER_REPO" \
            "$CONTAINER" bash -lc "$*"
    fi
}

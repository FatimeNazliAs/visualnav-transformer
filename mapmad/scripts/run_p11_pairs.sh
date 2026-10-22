#!/bin/bash
# Phase 11 closed-loop runs on ONE GPU (same pattern as run_p5.sh): PAIRS_PER_GPU pairs of one simulator server
# (naz_mapmad_habitat) + one runner (naz_mapmad, vint_train.mapmad.closed_loop.run_search), talking over mapmad-net.
# All runners of one launch share a run queue (--claims TOKEN): the chain calls this once per free GPU with the SAME
# token, so a GPU added later joins the queue, and a GPU withdrawn (STOP_FILE created) only finishes its current runs.
# Run on the host (the chain driver does it; nvidia-smi first):
#
#   GPU=1 TOKEN=step0_x STOP_FILE=/outputs/... PORT0=5560 mapmad/scripts/run_p11_pairs.sh \
#       --config /app/visualnav-transformer/mapmad/configs/p11_dagger.yaml --set dev --arms run1 runC --subdir step0
#
# Every argument goes to run_search (see its --help). A fresh bridge key per call goes through the environment (never
# a command line). Server output: <outputs>/<out_dir>/<subdir>/server_gpu<g>_<k>.log. A server that dies (non-zero
# exit) is restarted after 3 s; the runner reconnects and retries the run. STOP_FILE is a path INSIDE the containers.
set -euo pipefail

HABITAT=naz_mapmad_habitat
NOMAD=naz_mapmad
GPU=${GPU:?set GPU}
TOKEN=${TOKEN:?set TOKEN}
PAIRS_PER_GPU=${PAIRS_PER_GPU:-3}
OWNER=${OWNER:-chain}  # marks servers and runners for the GPU watcher (MAPMAD_P11_OWNER)
PORT0=${PORT0:-5560}
STOP_ARGS=()
[[ -n "${STOP_FILE:-}" ]] && STOP_ARGS=(--stop-file "$STOP_FILE")
SUBDIR=""
CONFIG=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    [[ "${args[$i]}" == "--subdir" ]] && SUBDIR="${args[$((i + 1))]}"
    [[ "${args[$i]}" == "--config" ]] && CONFIG="${args[$((i + 1))]}"
done
[[ -n "$CONFIG" && -n "$SUBDIR" ]] || { echo "--config and --subdir are required" >&2; exit 2; }
LOG_DIR="$(docker exec "$HABITAT" python -c "from mapmad_sim.run_layout import RunLayout
r = RunLayout.load('${CONFIG}', subdir='${SUBDIR}').root; r.mkdir(parents=True, exist_ok=True); print(r)")"

export MAPMAD_SIDECAR_KEY
MAPMAD_SIDECAR_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"

run() { echo "+ $*" >&2; "$@"; }

pids=()
for ((k = 0; k < PAIRS_PER_GPU; k++)); do
    port=$((PORT0 + 10 * GPU + k))
    docker exec "$HABITAT" pkill -f "sidecar_server --port $port" 2>/dev/null || true  # our own leftover server
    run docker exec -d -e MAPMAD_SIDECAR_KEY -e OMP_NUM_THREADS=2 -e MAPMAD_P11_OWNER="$OWNER" "$HABITAT" bash -c \
        "until python -m mapmad_sim.sidecar_server --port $port --gpu $GPU >> $LOG_DIR/server_gpu${GPU}_$k.log 2>&1; do
             echo \"server exited with \$?, restarting \$(date -Is)\" >> $LOG_DIR/server_gpu${GPU}_$k.log; sleep 3; done"
    run docker exec -e MAPMAD_SIDECAR_KEY -e CUDA_VISIBLE_DEVICES="$GPU" -e MPLCONFIGDIR=/tmp/mpl \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e OMP_NUM_THREADS=2 \
        -e MAPMAD_P11_OWNER="$OWNER" -e PYTHONPATH=/app/visualnav-transformer/mapmad/src -w /app/visualnav-transformer/train \
        "$NOMAD" python -m vint_train.mapmad.closed_loop.run_search --port "$port" --shard "$((10 * GPU + k))" \
        --claims "$TOKEN" --shutdown-server "${STOP_ARGS[@]}" "$@" &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status

#!/bin/bash
# Phase 5 closed-loop runs (same pattern as run_p1.sh): pairs of one simulator server (naz_mapmad_habitat) + one
# runner (naz_mapmad, vint_train.mapmad.closed_loop.run_exam), talking over mapmad-net. PAIRS lists the GPU of each
# pair; the episodes are split between the pairs (shard k = every n-th episode), pair k uses port BASE_PORT + k.
# Run on the host, long runs inside `screen` (nvidia-smi first):
#
#   PAIRS="0 0 1 1" mapmad/scripts/run_p5.sh --config /app/visualnav-transformer/mapmad/configs/p5_dev.yaml \
#       --arms C_wp1 C_wp2 --seeds 0 --subdir runs/wp_test
#
# Every argument goes to run_exam (see its --help); --config is required. A fresh bridge key per call goes through
# the environment (never a command line). Server output: <outputs>/<out_dir>/<subdir>/server_pair<k>.log.
# A server that dies (non-zero exit, e.g. a segfault) is restarted after 3 s; the runner reconnects and retries the
# episode (block item 13b). Runners get the §0.7 memory setting (expandable_segments; memory handling only).
set -euo pipefail

HABITAT=naz_mapmad_habitat
NOMAD=naz_mapmad
PAIRS=(${PAIRS:-0 1})
BASE_PORT=${BASE_PORT:-5560}
SUBDIR="runs"
CONFIG=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    [[ "${args[$i]}" == "--subdir" ]] && SUBDIR="${args[$((i + 1))]}"
    [[ "${args[$i]}" == "--config" ]] && CONFIG="${args[$((i + 1))]}"
done
[[ -n "$CONFIG" ]] || { echo "--config is required" >&2; exit 2; }
LOG_DIR="$(docker exec "$HABITAT" python -c "from mapmad_sim.run_layout import RunLayout
r = RunLayout.load('${CONFIG}', subdir='${SUBDIR}').root; r.mkdir(parents=True, exist_ok=True); print(r)")"

export MAPMAD_SIDECAR_KEY
MAPMAD_SIDECAR_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"

run() { echo "+ $*" >&2; "$@"; }

pids=()
for i in "${!PAIRS[@]}"; do
    g=${PAIRS[$i]}
    port=$((BASE_PORT + i))
    docker exec "$HABITAT" pkill -f "sidecar_server --port $port" 2>/dev/null || true  # our own leftover server
    run docker exec -d -e MAPMAD_SIDECAR_KEY -e OMP_NUM_THREADS=2 "$HABITAT" bash -c \
        "until python -m mapmad_sim.sidecar_server --port $port --gpu $g >> $LOG_DIR/server_pair$i.log 2>&1; do
             echo \"server exited with \$?, restarting \$(date -Is)\" >> $LOG_DIR/server_pair$i.log; sleep 3; done"
    run docker exec -e MAPMAD_SIDECAR_KEY -e CUDA_VISIBLE_DEVICES="$g" -e MPLCONFIGDIR=/tmp/mpl \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        -e PYTHONPATH=/app/visualnav-transformer/mapmad/src -w /app/visualnav-transformer/train "$NOMAD" \
        python -m vint_train.mapmad.closed_loop.run_exam --port "$port" --shard "$i" --num-shards "${#PAIRS[@]}" \
        --shutdown-server "$@" &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status

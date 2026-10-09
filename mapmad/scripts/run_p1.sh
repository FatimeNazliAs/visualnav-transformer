#!/bin/bash
# Phase 1 closed-loop runs: pairs of one simulator server (naz_mapmad_habitat) + one NoMaD client (naz_mapmad),
# talking over mapmad-net. PAIRS lists the GPU of each pair; episodes are split between the pairs (shard k =
# every n-th episode), pair k uses port BASE_PORT + k. Run on the host:
#
#   PAIRS="0 0 1 1" mapmad/scripts/run_p1.sh --arms photo_iv explore_iv   # 2 pairs per GPU
#   PAIRS=1 mapmad/scripts/run_p1.sh --arms photo_iv --limit 1 --subdir checks/smoke
#   PAIRS="0 0 1 1" mapmad/scripts/run_p1.sh --config /app/visualnav-transformer/mapmad/configs/p1_baseline_spec.yaml --arms ...
#
# Every argument goes to vint_train.mapmad.closed_loop.run_arms (see its --help). The shared key for the bridge is
# made fresh per call and passed through the environment (never on a command line). Server output goes to
# <outputs>/<out_dir>/<subdir>/server_pair<k>.log (paths.yaml + the run config, p1_baseline.yaml by default). Long runs: start this inside `screen`.
set -euo pipefail

HABITAT=naz_mapmad_habitat
NOMAD=naz_mapmad
PAIRS=(${PAIRS:-0 1})
BASE_PORT=${BASE_PORT:-5550}
SUBDIR=""
CONFIG=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    [[ "${args[$i]}" == "--subdir" ]] && SUBDIR="${args[$((i + 1))]}"
    [[ "${args[$i]}" == "--config" ]] && CONFIG="${args[$((i + 1))]}"
done
# run output folder as the containers see it (paths.yaml + the run config, via mapmad_sim.run_layout), created there
LOG_DIR="$(docker exec "$HABITAT" python -c "from mapmad_sim.run_layout import RunLayout
r = RunLayout.load('${CONFIG}' or None, subdir='${SUBDIR}').root; r.mkdir(parents=True, exist_ok=True); print(r)")"

export MAPMAD_SIDECAR_KEY
MAPMAD_SIDECAR_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"

run() { echo "+ $*" >&2; "$@"; }

pids=()
for i in "${!PAIRS[@]}"; do
    g=${PAIRS[$i]}
    port=$((BASE_PORT + i))
    docker exec "$HABITAT" pkill -f "sidecar_server --port $port" 2>/dev/null || true  # our own leftover server
    run docker exec -d -e MAPMAD_SIDECAR_KEY "$HABITAT" bash -c \
        "python -m mapmad_sim.sidecar_server --port $port --gpu $g > $LOG_DIR/server_pair$i.log 2>&1"
    run docker exec -e MAPMAD_SIDECAR_KEY -e CUDA_VISIBLE_DEVICES="$g" -e MPLCONFIGDIR=/tmp/mpl \
        -e PYTHONPATH=/app/visualnav-transformer/mapmad/src -w /app/visualnav-transformer/train "$NOMAD" \
        python -m vint_train.mapmad.closed_loop.run_arms --port "$port" --shard "$i" --num-shards "${#PAIRS[@]}" \
        --shutdown-server "$@" &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status

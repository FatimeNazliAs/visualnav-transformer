#!/bin/bash
# Phase 12 closed-loop runs on ONE GPU (run_p11_pairs.sh + the recogniser sidecar): PAIRS_PER_GPU pairs of one
# simulator server (naz_mapmad_habitat) + one recogniser server (naz_mapmad_recog, model RECOG, default yoloe; none =
# no sidecar) + one runner (naz_mapmad, vint_train.mapmad.closed_loop.run_search), talking over mapmad-net. All runners
# of one launch share a run queue (--claims TOKEN): the chain calls this once per free GPU with the SAME token.
# Run on the host (the chain driver does it; nvidia-smi first):
#
#   GPU=1 TOKEN=exam_x STOP_FILE=/outputs/... PORT0=5560 RPORT0=5700 RECOG=yoloe mapmad/scripts/run_p12_pairs.sh \
#       --config /app/visualnav-transformer/mapmad/configs/p12_exam.yaml --set exam --arms M2 M2h --subdir runs/exam
#
# Every argument goes to run_search (see its --help). A fresh bridge key per call goes through the environment (never
# a command line); both servers of a pair use it. Server output: <outputs>/<out_dir>/<subdir>/server_gpu<g>_<k>.log and
# recog_gpu<g>_<k>.log. A server that dies (non-zero exit) is restarted after 3 s (the runner reconnects and retries the
# run); every recogniser restart is counted in <subdir>/recog_restarts/gpu<g>_<k> (one line each) -- the chain stops
# when a recogniser had to be restarted more than 3 times (block item 9). STOP_FILE is a path INSIDE the containers.
set -euo pipefail

HABITAT=naz_mapmad_habitat
NOMAD=naz_mapmad
RECOGC=naz_mapmad_recog
GPU=${GPU:?set GPU}
TOKEN=${TOKEN:?set TOKEN}
PAIRS_PER_GPU=${PAIRS_PER_GPU:-3}
OWNER=${OWNER:-chain}  # marks servers and runners for the GPU watcher (MAPMAD_P12_OWNER)
PORT0=${PORT0:-5560}
RPORT0=${RPORT0:-5700}
RECOG=${RECOG:-yoloe}
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
    rport=$((RPORT0 + 10 * GPU + k))
    docker exec "$HABITAT" pkill -f "sidecar_server --port $port " 2>/dev/null || true  # our own leftover server
    run docker exec -d -e MAPMAD_SIDECAR_KEY -e OMP_NUM_THREADS=2 -e MAPMAD_P12_OWNER="$OWNER" "$HABITAT" bash -c \
        "until python -m mapmad_sim.sidecar_server --port $port --gpu $GPU >> $LOG_DIR/server_gpu${GPU}_$k.log 2>&1; do
             echo \"server exited with \$?, restarting \$(date -Is)\" >> $LOG_DIR/server_gpu${GPU}_$k.log; sleep 3; done"
    recog_args=()
    if [[ "$RECOG" != none ]]; then
        docker exec "$RECOGC" pkill -f "mapmad_recog.server --model [a-z0-9_]* --port $rport " 2>/dev/null || true
        mkdir_cmd="mkdir -p $LOG_DIR/recog_restarts $LOG_DIR/text_reuse"
        run docker exec -d -e MAPMAD_SIDECAR_KEY -e CUDA_VISIBLE_DEVICES="$GPU" -e OMP_NUM_THREADS=4 -e HOME=/model_cache/home \
            -e MAPMAD_P12_OWNER="$OWNER" "$RECOGC" bash -c \
            "$mkdir_cmd; until python -m mapmad_recog.server --model $RECOG --port $rport --text-reuse-out $LOG_DIR/text_reuse/${RECOG}_gpu${GPU}_$k.json >> $LOG_DIR/recog_gpu${GPU}_$k.log 2>&1; do
                 c=\$?; echo \"recogniser exited with \$c, restarting \$(date -Is)\" >> $LOG_DIR/recog_gpu${GPU}_$k.log
                 echo \"\$(date -Is) exit \$c\" >> $LOG_DIR/recog_restarts/gpu${GPU}_$k; sleep 3; done"
        recog_args=(--recog-port "$rport")
    fi
    run docker exec -e MAPMAD_SIDECAR_KEY -e CUDA_VISIBLE_DEVICES="$GPU" -e MPLCONFIGDIR=/tmp/mpl \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e OMP_NUM_THREADS=2 \
        -e MAPMAD_P12_OWNER="$OWNER" -e PYTHONPATH=/app/visualnav-transformer/mapmad/src -w /app/visualnav-transformer/train \
        "$NOMAD" python -m vint_train.mapmad.closed_loop.run_search --port "$port" --shard "$((10 * GPU + k))" \
        --claims "$TOKEN" --shutdown-server "${recog_args[@]}" "${STOP_ARGS[@]}" "$@" &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$RECOG" != none ]]; then  # the runners are done: stop this launch's recogniser servers (and their restart loops)
    for ((k = 0; k < PAIRS_PER_GPU; k++)); do
        rport=$((RPORT0 + 10 * GPU + k))
        docker exec "$RECOGC" pkill -f "until python -m mapmad_recog.server --model $RECOG --port $rport " 2>/dev/null || true
        docker exec "$RECOGC" pkill -f "mapmad_recog.server --model $RECOG --port $rport " 2>/dev/null || true
    done
fi
exit $status

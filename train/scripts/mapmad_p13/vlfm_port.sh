#!/bin/bash
# MapMaD Phase 13 "VLFM port check" launcher (report-only): runs the ORIGINAL VLFM code (image naz-mapmad-vlfm,
# commit 584ed56) with its own settings on the first <n> of the 100 seeded HM3D v2 val ObjectNav episodes
# (vlfm_port/episodes_100.json), unattended. Resumable: finished episodes (VLFM's own per-episode logs in
# <out_dir>/episode_logs) are skipped; a crashed run is restarted on the remaining episodes.
#
#   train/scripts/mapmad_p13/vlfm_port.sh <gpu> <n_episodes|all> <out_dir>
#
# <out_dir> must lie under $NOMAD_OUTPUTS (mounted at /outputs in naz_mapmad_vlfm).
# Writes: <out_dir>/results.jsonl (one row per finished episode), <out_dir>/summary.json, logs/, episode_logs/.
# Exit: 0 = all requested episodes finished; 1 = run failed / episodes missing; 2 = usage; 3 = GPU busy.
# Starts naz_mapmad_vlfm via mapmad/docker/run_containers.sh if needed; at the end kills only its own processes
# (model servers on its ports, its vlfm.run) and removes the container only if it created it and nothing else runs.
set -euo pipefail

usage() { sed -n '2,13p' "$0"; exit 2; }
[ $# -eq 3 ] || usage
GPU="$1"; N_ARG="$2"; OUT="$3"
[[ "$GPU" =~ ^[0-9]+$ ]] || usage
[[ "$N_ARG" =~ ^([0-9]+|all)$ ]] || usage

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
DOCKER_DIR="$REPO/mapmad/docker"
# shellcheck source=../../../mapmad/docker/paths.env
source "$DOCKER_DIR/paths.env"
: "${VLFM_PORT_DIR:=$NOMAD_OUTPUTS/mapmad/p13_claim/vlfm_port}"   # holds episodes_100.json (+ .sha256)
: "${VLFM_SERVER_TIMEOUT:=1800}"                                  # s to wait for the 4 model servers (1st run downloads BLIP-2)
: "${VLFM_MAX_STALLS:=2}"                                         # restarts allowed without new finished episodes
EPISODES="$VLFM_PORT_DIR/episodes_100.json"
export MAPMAD_P13_OWNER=chain

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
case "$OUT/" in "$NOMAD_OUTPUTS"/*) ;; *) echo "out_dir must be under $NOMAD_OUTPUTS" >&2; exit 2 ;; esac
OUT_C="/outputs${OUT#"$NOMAD_OUTPUTS"}"            # same folder as seen inside the container
EPISODES_C="/outputs${EPISODES#"$NOMAD_OUTPUTS"}"
mkdir -p "$OUT/logs" "$OUT/episode_logs"
LOG="$OUT/logs/launcher.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG" >&2; }

# ---- checks -------------------------------------------------------------------------------------------------
[ -s "$EPISODES" ] || { log "missing $EPISODES"; exit 1; }
(cd "$VLFM_PORT_DIR" && sha256sum -c --quiet episodes_100.json.sha256) || { log "episodes_100.json sha256 mismatch"; exit 1; }
BUSY="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU" || true)"
if [ -n "$BUSY" ]; then log "GPU $GPU has compute processes, not starting: $BUSY"; exit 3; fi
nvidia-smi -i "$GPU" --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"

# ---- scene root: HM3D v0.2 in the layout VLFM's (v2) episodes expect -----------------------------------------
mkdir -p "$VLFM_SCENE_ROOT/hm3d_v0.2"
ln -sfn /hm3d/val "$VLFM_SCENE_ROOT/hm3d_v0.2/val"
SDC="hm3d_annotated_basis.scene_dataset_config.json"
[ -s "$VLFM_SCENE_ROOT/hm3d_v0.2/$SDC" ] || cp "$(dirname "$HM3D_SCENES")/$SDC" "$VLFM_SCENE_ROOT/hm3d_v0.2/$SDC"

# ---- container ------------------------------------------------------------------------------------------------
STARTED_CONTAINER=0
if ! docker container inspect "$VLFM_CONTAINER" >/dev/null 2>&1; then
    "$DOCKER_DIR/run_containers.sh" vlfm 2>&1 | tee -a "$LOG"
    STARTED_CONTAINER=1
fi
[ "$(docker inspect -f '{{.State.Running}}' "$VLFM_CONTAINER")" = true ] || { log "$VLFM_CONTAINER not running"; exit 1; }

BASE_PORT=$((12181 + 100 * GPU))   # per-GPU ports so two launchers can share the container
GDINO_PORT=$BASE_PORT; ITM_PORT=$((BASE_PORT + 1)); SAM_PORT=$((BASE_PORT + 2)); YOLO_PORT=$((BASE_PORT + 3))
EXEC_OPTS=(-e MAPMAD_P13_OWNER=chain -e CUDA_VISIBLE_DEVICES="$GPU"
           -e GROUNDING_DINO_PORT="$GDINO_PORT" -e BLIP2ITM_PORT="$ITM_PORT" -e SAM_PORT="$SAM_PORT"
           -e YOLOV7_PORT="$YOLO_PORT" -w /opt/vlfm)
DEXEC=(docker exec "${EXEC_OPTS[@]}" "$VLFM_CONTAINER")

cleanup() {
    local rc=$?
    set +e
    [ -n "${GPU_MON_PID:-}" ] && kill "$GPU_MON_PID" 2>/dev/null
    log "cleanup: stopping own processes (ports $GDINO_PORT-$YOLO_PORT, $OUT_C/dataset)"
    for p in "$GDINO_PORT" "$ITM_PORT" "$SAM_PORT" "$YOLO_PORT"; do
        docker exec "$VLFM_CONTAINER" pkill -f -- "--port $p\$" 2>/dev/null
    done
    docker exec "$VLFM_CONTAINER" pkill -f -- "$OUT_C/dataset" 2>/dev/null
    sleep 3
    if [ "$STARTED_CONTAINER" = 1 ] && ! docker exec "$VLFM_CONTAINER" pgrep -f python >/dev/null 2>&1; then
        "$DOCKER_DIR/run_containers.sh" vlfm-down 2>&1 | tee -a "$LOG"
    fi
    log "exit $rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# ---- python helper (runs inside the container) ----------------------------------------------------------------
# usage: helper <cmd> <args...>; extra `docker exec` env options in the HELPER_ENV array
HELPER_ENV=()
helper() { docker exec -i "${EXEC_OPTS[@]}" "${HELPER_ENV[@]}" "$VLFM_CONTAINER" python - "$@" <<'PY'
import glob, gzip, json, os, subprocess, sys

cmd, episodes_path, n_arg, out = sys.argv[1:5]
doc = json.load(open(episodes_path))
sel = doc["episodes"] if n_arg == "all" else doc["episodes"][: int(n_arg)]
logdir = os.path.join(out, "episode_logs")

def done_logs():
    """Finished episodes: {(scene, episode_id in episodes_100.json): (VLFM's log dict, log mtime)}.

    habitat-lab renumbers episode_id to the position in the (subset) content file, so each attempt logs into
    its own folder a<k>/ whose map.json maps {scene: [our episode_id per subset position]}.
    """
    res = {}
    for adir in sorted(glob.glob(os.path.join(logdir, "a*"))):
        mp = os.path.join(adir, "map.json")
        if not os.path.exists(mp):
            continue
        idmap = json.load(open(mp))
        for fn in glob.glob(os.path.join(adir, "*_*.json")):
            if os.path.getsize(fn) == 0:
                continue
            d = json.load(open(fn))
            res[(d["scene_id"], idmap[d["scene_id"]][int(d["episode_id"])])] = (d, os.path.getmtime(fn))
    return res

def todo_list():
    done = done_logs()
    return [e for e in sel if (e["scene"], e["episode_id"]) not in done]

if cmd == "count":
    print(len(todo_list()))

elif cmd == "prepare":
    # subset dataset with the not-yet-finished episodes (file order kept) + a new attempt log folder
    todo = todo_list()
    if not todo:
        print("none 0")
        sys.exit(0)
    ds = os.path.join(out, "dataset", "objectnav_hm3d_v2", "val")  # VLFM needs "hm3d" in data_path
    os.makedirs(os.path.join(ds, "content"), exist_ok=True)
    for old in glob.glob(os.path.join(ds, "content", "*.json.gz")):
        os.replace(old, old + ".old")  # never rm; the next prepare overwrites
    top = json.load(gzip.open("/objectnav/val/val.json.gz"))
    json.dump(top, gzip.open(os.path.join(ds, "val.json.gz"), "wt"))
    by_file = {}
    for e in todo:
        by_file.setdefault(e["content_file"], set()).add(e["episode_id"])
    idmap = {}
    for fn, ids in by_file.items():
        d = json.load(gzip.open(os.path.join("/objectnav/val/content", fn)))
        keep, pos = [], []
        for i, e in enumerate(d["episodes"]):  # our episode_id = position in the full content file
            if str(i) in ids:
                keep.append(e)
                pos.append(str(i))
        assert len(keep) == len(ids), fn
        d["episodes"] = keep
        idmap[os.path.basename(keep[0]["scene_id"]).split(".")[0]] = pos
        json.dump(d, gzip.open(os.path.join(ds, "content", fn), "wt"))
    k = len(glob.glob(os.path.join(logdir, "a*"))) + 1
    adir = os.path.join(logdir, f"a{k}")
    os.makedirs(adir)
    json.dump(idmap, open(os.path.join(adir, "map.json"), "w"))
    print(f"a{k} {len(todo)}")

elif cmd == "summarize":
    deviations = json.loads(os.environ["VLFM_DEVIATIONS"])
    done = done_logs()
    starts = sorted(float(x) for x in open(os.path.join(out, "run_starts.txt")).read().split())
    # wall time per episode = its log's mtime minus the later of (its run's start, previous log's mtime)
    rows, wall, prev = [], {}, 0.0
    for m, k in sorted((m, k) for k, (d, m) in done.items()):
        run_start = max([x for x in starts if x <= m], default=m)
        wall[k] = round(m - max(run_start, prev), 1)
        prev = m
    for e in sel:
        k = (e["scene"], e["episode_id"])
        if k not in done:
            continue
        d = done[k][0]
        rows.append({"scene": e["scene"], "episode_id": e["episode_id"], "object_category": e["object_category"],
                     "geodesic_distance": e["geodesic_distance"], "success": float(d.get("success", 0.0)),
                     "spl": float(d.get("spl", 0.0)), "soft_spl": d.get("soft_spl"),
                     "distance_to_goal": d.get("distance_to_goal"), "num_steps": d.get("num_steps"),
                     "failure_cause": d.get("failure_cause"), "wall_time_s": wall[k]})
    with open(os.path.join(out, "results.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    gpu_peak = None
    gm = os.path.join(out, "logs", "gpu_mem.csv")
    if os.path.exists(gm):
        vals = [int(x) for x in open(gm).read().split() if x.strip().isdigit()]
        gpu_peak = max(vals) if vals else None
    summary = {
        "episodes": n, "requested": len(sel),
        "success_rate": (sum(r["success"] for r in rows) / n) if n else None,
        "spl": (sum(r["spl"] for r in rows) / n) if n else None,
        "mean_wall_time_s": (sum(r["wall_time_s"] for r in rows) / n) if n else None,
        "gpu_mem_peak_mib_whole_gpu": gpu_peak,
        "paper_hm3d_sr": 0.636, "paper_hm3d_spl": 0.304,
        "episodes_file": episodes_path, "episodes_sha256": doc["sha256_of_episodes"],
        "vlfm_commit": open("/opt/vlfm/.git/HEAD").read().strip(),  # detached checkout = the SHA (git 2.25: no safe.directory),
        "image": os.environ.get("VLFM_IMAGE_ID", ""),
        "settings": json.load(open(os.path.join(out, "settings.json"))),
        "deviations": deviations,
    }
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("episodes", "requested", "success_rate", "spl")}))

elif cmd == "settings":
    # the effective habitat config VLFM will use (same composition as vlfm.run + our overrides)
    import vlfm.run  # noqa: registers plugins / structured configs
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir="/opt/vlfm/config", version_base=None):
        cfg = compose(config_name="experiments/vlfm_objectnav_hm3d", overrides=sys.argv[5:])
    from omegaconf import OmegaConf
    open(os.path.join(out, "logs", "effective_config.yaml"), "w").write(OmegaConf.to_yaml(cfg))
    h = cfg.habitat
    s = {"max_episode_steps": h.environment.max_episode_steps,
         "success_distance": h.task.measurements.success.success_distance,
         "turn_angle": h.simulator.turn_angle, "forward_step_size": h.simulator.forward_step_size,
         "policy": cfg.habitat_baselines.rl.policy.name, "split": cfg.habitat_baselines.eval.split,
         "data_path": h.dataset.data_path, "scene_dataset": h.simulator.scene_dataset,
         "overrides": sys.argv[5:]}
    json.dump(s, open(os.path.join(out, "settings.json"), "w"), indent=1)
    print(json.dumps(s))
PY
}

# Every difference from the original VLFM code/settings (also written into summary.json).
DEVIATIONS='[
 "episodes: HM3D ObjectNav v2 val (HM3D v0.2 scenes), 100 seeded episodes (episodes_100.json); paper used ObjectNav HM3D v1 val (2000 episodes)",
 "episodes identified by their position in the scene content file (v2 val ids repeat within a scene; habitat-lab renumbers episode_id by file position anyway); episode content unchanged",
 "habitat.dataset.data_path -> per-run subset of the v2 val files (only not-yet-finished episodes), habitat.simulator.scene_dataset -> v0.2 annotated scene dataset config",
 "habitat-sim 0.2.4 from the official aihabitat conda package (headless+bullet) instead of pip install from git v0.2.4",
 "VLFM git deps pinned to last commit <= VLFM commit date: frontier_exploration 8b4cb93, MobileSAM 34bbbfd, depth_camera_filtering 39d6e2f, yolov7 a207844",
 "model servers started as background processes (docker exec -d) instead of tmux; ports 12181+100*gpu .. +3 instead of 12181..12184",
 "flask==2.3.3 / werkzeug==2.3.8 / seaborn==0.12.2 / open3d==0.17.0 pinned (VLFM lists >= versions)",
 "GroundingDINO CUDA op compiled for sm_86 (TORCH_CUDA_ARCH_LIST=8.6, runs on sm_89 Ada)",
 "habitat_baselines.video_dir / tensorboard_dir moved into out_dir; no videos (VLFM default video_option=[])"
]'
OVERRIDES=(
    "habitat.dataset.data_path=$OUT_C/dataset/objectnav_hm3d_v2/val/val.json.gz"   # literal: hydra overrides reject braces
    "habitat.simulator.scene_dataset=data/scene_datasets/hm3d_v0.2/$SDC"
    "habitat_baselines.video_dir=$OUT_C/video_dir"
    "habitat_baselines.tensorboard_dir=$OUT_C/tb"
)

# ---- model servers (VLFM's scripts/launch_vlm_servers.sh, without tmux) ---------------------------------------
port_open() { "${DEXEC[@]}" python -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(('localhost', $1)))" 2>/dev/null; }
start_server() {  # module port name
    if port_open "$2"; then log "server $3 already up on port $2"; return; fi
    log "starting server $3 on port $2"
    "${DEXEC[@]}" bash -c "nohup python -m $1 --port $2 > $OUT_C/logs/server_$3.log 2>&1 &"
}
start_server vlfm.vlm.grounding_dino "$GDINO_PORT" gdino
start_server vlfm.vlm.blip2itm "$ITM_PORT" blip2itm
start_server vlfm.vlm.sam "$SAM_PORT" sam
start_server vlfm.vlm.yolov7 "$YOLO_PORT" yolov7
t0=$(date +%s)
for p in "$GDINO_PORT" "$ITM_PORT" "$SAM_PORT" "$YOLO_PORT"; do
    until port_open "$p"; do
        if (( $(date +%s) - t0 > VLFM_SERVER_TIMEOUT )); then log "servers not up after ${VLFM_SERVER_TIMEOUT}s (see $OUT/logs/server_*.log)"; exit 1; fi
        if ! docker exec "$VLFM_CONTAINER" pgrep -f -- "--port $p\$" >/dev/null; then log "server on port $p died (see $OUT/logs/server_*.log)"; exit 1; fi
        sleep 10
    done
done
log "all 4 model servers up after $(( $(date +%s) - t0 )) s"

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" -l 5 > "$OUT/logs/gpu_mem.csv" 2>/dev/null &
GPU_MON_PID=$!

# ---- evaluation loop (restart on crash while episodes remain) -------------------------------------------------
helper settings "$EPISODES_C" "$N_ARG" "$OUT_C" "${OVERRIDES[@]}" | tee -a "$LOG"
stalls=0; attempt=0
while :; do
    read -r adir remaining < <(helper prepare "$EPISODES_C" "$N_ARG" "$OUT_C" | tail -1)
    log "remaining episodes: $remaining"
    [ "$remaining" -eq 0 ] && break
    if [ "$stalls" -gt "$VLFM_MAX_STALLS" ]; then log "no progress after $stalls restarts, giving up"; break; fi
    attempt=$((attempt + 1))
    date +%s.%N >> "$OUT/run_starts.txt"
    log "vlfm.run attempt $attempt on $remaining episodes (logs: episode_logs/$adir, logs/run_$attempt.log)"
    set +e
    docker exec "${EXEC_OPTS[@]}" -e ZSOS_LOG_DIR="$OUT_C/episode_logs/$adir" "$VLFM_CONTAINER" \
        python -um vlfm.run "${OVERRIDES[@]}" > "$OUT/logs/run_$attempt.log" 2>&1
    rc=$?
    set -e
    after=$(helper count "$EPISODES_C" "$N_ARG" "$OUT_C" | tail -1)
    log "vlfm.run attempt $attempt exit $rc, remaining $after"
    if [ "$after" -ge "$remaining" ]; then stalls=$((stalls + 1)); else stalls=0; fi
done

HELPER_ENV=(-e VLFM_DEVIATIONS="$DEVIATIONS" -e VLFM_IMAGE_ID="$VLFM_IMAGE $(docker inspect -f '{{.Image}}' "$VLFM_CONTAINER")")
SUMMARY_LINE=$(helper summarize "$EPISODES_C" "$N_ARG" "$OUT_C" | tail -1) || { log "summarize failed"; exit 1; }
log "summary: $SUMMARY_LINE"
[ "$remaining" -eq 0 ] && exit 0 || exit 1

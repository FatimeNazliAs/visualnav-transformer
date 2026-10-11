# scripts/mapmad_p4 — Phase 4 run scripts (host side)

Run on the host (they call `docker exec naz_mapmad ...`), each in its own `screen`. The copies that actually ran in
Phase 4 stay in `/outputs/mapmad/p4_train/` with their sha256 in `p4_rules_v2.sha256`; these repo versions differ
from them only in reading the paths from environment variables.

| Script | What it does |
|---|---|
| `run_supervised.sh <label> <config>` | trains one run (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`); if it does not finish, resumes ONCE from the run's `latest.pth`; a second failure writes `STOP_<label>.txt` |
| `p4_post_v2.sh` | waits for runs C and D; C's final eval (full offline check + GoStanford harness + selection-bias report) and the G4' rows (`final_eval_v2/g4_prime.json`) as soon as C ends; D's eval only if D finishes; curves; stops the memory logger |
| `gpu_mem_log.sh <run on GPU0> <run on GPU1> [csv]` | GPU memory (nvidia-smi) every 10 s with each run's current epoch |

Environment (values used on cukurovaai):

    export MAPMAD_P4_HOST_DIR=/mnt/shared_disk/nazli/nomad_outputs/mapmad/p4_train
    export MAPMAD_P4_CONTAINER_DIR=/outputs/mapmad/p4_train
    export MAPMAD_OFFLINE_LIST=/outputs/mapmad/p3_model/offline/offline_samples.json
    export MAPMAD_OFFICIAL_WEIGHTS=/outputs/mapmad/weights/official/nomad.pth

Phase 4 confirmation launch (from `train/scripts/mapmad_p4`, after `nvidia-smi`):

    screen -dmS mapmad_p4_C ./run_supervised.sh C config/mapmad_p4_C.yaml
    screen -dmS mapmad_p4_D ./run_supervised.sh D config/mapmad_p4_D.yaml
    screen -dmS mapmad_p4_gpumem ./gpu_mem_log.sh C D gpu_mem_v2.csv
    screen -dmS mapmad_p4_post_v2 ./p4_post_v2.sh

`screen` passes the current environment on, so export the variables first.

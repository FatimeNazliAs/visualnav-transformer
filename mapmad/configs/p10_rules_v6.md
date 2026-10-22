# p10_rules_v6.md — Phase 10 run C rerun from nomad.pth after a container restart (pre-set before launch)

Version 6, 2026-10-22 07:4x. Naz's instruction (option 2). **Deviation from v5** (v5 §3 said: a crash in resume 2 ->
v2 = run 1, no further run), made before any run C result was judged: no run C checkpoint was ever scored. Rules v4
(sha256 `a526a467…bd6f`) §2-4 judge run C unchanged; v5 (sha256 `45644afa…d124`) is replaced where this file differs.

## 1. Why (record)
After v5's resume 2 crashed (07:05, epoch 13, DataLoader worker segfault), the run-1-recipe twin (no oversampling)
segfaulted at its 3rd step on GPU0, with no Phase 9 job running and ~85 GB RAM available; a CPU-only run of the exact
training loader (600 batches of epoch 13, 12 workers) was clean. So the crashes are an environment problem, not run C's
change, and run C was never fairly tested. Suspected: the long-running `naz_mapmad` container (up 5 days; 735 stale
/dev/shm files, 18 zombie worker processes before the restart).

## 2. Restart (done before launch, logged in driver/shm_restart.log)
`docker restart naz_mapmad` (no Phase 9 job uses it). Before: ShmSize 8 GiB, /dev/shm 2.9M / 735 files, ipcs -m empty,
18 zombies. After: /dev/shm 0 / 0 files, ipcs -m empty, 0 zombies; torch 2.4.1+cu121 sees CUDA.

## 3. Run C rerun
- From the official `nomad.pth`, the same recipe and launch setting as run C (`launch/c.yaml` = `mapmad_v2_seen.yaml`,
  seed 1, 20 epochs, seen-before-out-of-view oversampling 0.25, batch 256 x 1, 12 workers) in a NEW run folder
  (`runs/v2c_s1_<time>`; the crashed folder `runs/v2c_s1_2026_10_21_21_44_11` stays as a record). **ema_19 of the
  rerun is fixed in advance.** First GPU that passes the free check. RAM / swap logged at start and after every epoch.
- **Health check:** a crash (any unfinished run) before 2 epochs are complete -> **stop for Naz** (server problem);
  v2 = run 1 ema_19 stays.
- After 2 complete epochs: **one automatic resume** from latest.pth; a second crash (or a NaN loss) -> **stop for Naz**.
- When 2 epochs are complete the driver writes `driver/RUN_C_PAST_EPOCH2.txt` (Phase 9 may then run its held checks
  3-4 on the other GPU; Phase 10 does not start or stop Phase 9 jobs).

## 4. Judgment (unchanged, rules v4 §2-4)
Seen-list rule (CI lower bound of target_zeroed − maps > 0) + G10 rows 1-4 + guard (CI upper bound of maps C − maps
run 1 ≤ +0.05) -> C_IS_V2 / RUN1_IS_V2 / STOP (`p10_c_decide.py`, `offline/c_ema19/decision.md`).

## 5. Twin
After the verdict (not after a stop): seed 2 of the winning recipe, report only, on a free GPU, one training at a time.
A twin crash stops only the twin (report only; the v2 choice stands).

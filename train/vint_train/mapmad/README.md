# vint_train/mapmad/ — MapMaD, NoMaD side

Runs in the `naz_mapmad` container (conda env `vint_train`, Python 3.8). Planned contents:

- map encoder: 64 × 64 local map (obstacles · explored · goal heat) → one extra token (Phase 3);
- local-map crop: cut the robot-centred, rotating 64 × 64 window out of a whole-house map (Phase 2–3);
- goal heat drawing, incl. the warm spot on the map border for far targets (Phase 2–3).

Original NoMaD files get only small hooks behind `map_input` (`false` = the old NoMaD, byte-identical).
Extra packages go in `train/requirements-mapmad.txt`. Empty in Phase 0.

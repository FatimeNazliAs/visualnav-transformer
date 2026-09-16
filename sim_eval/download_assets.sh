#!/usr/bin/env bash
# ==============================================================================
# sim_eval/download_assets.sh — fetch the iGibson assets this phase needs.
#
#   --download_assets     robot models (incl. the LoCoBot URDF) + base textures
#   --download_demo_data  the Rs scene: a static Gibson mesh of a real house
#
# Both are public, no licence click-through. Everything lands in
# /igibson_data on the shared disk (never in the image, never in the repo),
# via the GIBSON_*_PATH env vars baked into the image.
#
# iGibson skips anything already present, so re-running is cheap.
#
# NOT downloaded here: the 15-house interactive iGibson dataset. It sits behind
# a terms-of-use agreement, and whether the evaluation uses interactive houses
# or static meshes is still an open decision (plan §12).
#
# Usage (from the repo root, on the host):
#     ./sim_eval/download_assets.sh
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source sim_eval/lib.sh

echo "==> downloading robot + object assets"
sim_exec "python -m igibson.utils.assets_utils --download_assets"

echo "==> downloading the Rs scene"
sim_exec "python -m igibson.utils.assets_utils --download_demo_data"

echo
echo "==> installed under $HOST_IGIBSON_DATA:"
du -sh "$HOST_IGIBSON_DATA"/* 2>/dev/null || true

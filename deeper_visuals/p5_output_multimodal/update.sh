#!/usr/bin/env bash
# deeper_visuals/p5_output_multimodal/update.sh
#
# Artifact URL (fixed — every republish targets this same link):
#   https://claude.ai/code/artifact/69585ff5-375c-43d0-bb17-a34e1cd61ecc
#
# Also listed on the P0 Notion "Models & datasets used" page, which is the URL
# registry for all six phases. Republishing from this repo path keeps the link;
# publishing a different path would mint a new one and strand the advisor.
#
# Usage — from the host, from the repo root. No need to enter the container;
# common/update.sh does that itself.
#   ./deeper_visuals/p5_output_multimodal/update.sh                full run
#   ./deeper_visuals/p5_output_multimodal/update.sh --page-only    wording only
#
# The full run encodes the scene once and then runs the diffusion descent
# num_seeds times over, so it needs /data, /outputs and the GPU. It is the
# slowest phase in the series for that reason — N descents, not one.
#
# Everything real lives in common/update.sh; this stub only pins the phase.
exec "$(dirname "${BASH_SOURCE[0]}")/../common/update.sh" \
  "$(dirname "${BASH_SOURCE[0]}")" "$@"

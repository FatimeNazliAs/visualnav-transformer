#!/usr/bin/env bash
# deeper_visuals/p3_transformer_masking/update.sh
#
# Artifact URL (fixed — every republish targets this same link):
#   https://claude.ai/code/artifact/52beac61-bada-4600-89a1-b52892f1992a
#
# Also listed on the P0 Notion "Models & datasets used" page, which is the URL
# registry for all six phases. Republishing from this repo path keeps the link;
# publishing a different path would mint a new one and strand the advisor.
#
# Usage — from the host, from the repo root. No need to enter the container;
# common/update.sh does that itself.
#   ./deeper_visuals/p3_transformer_masking/update.sh                full run
#   ./deeper_visuals/p3_transformer_masking/update.sh --page-only    wording only
#
# The full run pushes the scene through the transformer twice — goal token
# visible, then hidden from attention — so it needs /data, /outputs and the GPU.
#
# Everything real lives in common/update.sh; this stub only pins the phase.
exec "$(dirname "${BASH_SOURCE[0]}")/../common/update.sh" \
  "$(dirname "${BASH_SOURCE[0]}")" "$@"

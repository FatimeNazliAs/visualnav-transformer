#!/usr/bin/env bash
# deeper_visuals/p2_encoders/update.sh
#
# Artifact URL (fixed — every republish targets this same link):
#   https://claude.ai/code/artifact/aed15c20-7968-4d4c-aff3-3cec670fb980
#
# Also listed on the P0 Notion "Models & datasets used" page, which is the URL
# registry for all six phases. Republishing from this repo path keeps the link;
# publishing a different path would mint a new one and strand the advisor.
#
# Usage — from the host, from the repo root. No need to enter the container;
# common/update.sh does that itself.
#   ./deeper_visuals/p2_encoders/update.sh                full run
#   ./deeper_visuals/p2_encoders/update.sh --page-only    wording/layout only
#
# P2 is the first phase that really loads the checkpoint: the full run needs
# /data and /outputs mounted, and wants the GPU (it will fall back to CPU).
#
# Everything real lives in common/update.sh; this stub only pins the phase.
exec "$(dirname "${BASH_SOURCE[0]}")/../common/update.sh" \
  "$(dirname "${BASH_SOURCE[0]}")" "$@"

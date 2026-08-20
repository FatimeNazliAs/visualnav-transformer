#!/usr/bin/env bash
# deeper_visuals/p1_inputs/update.sh
#
# Artifact URL (fixed — every republish targets this same link):
#   https://claude.ai/code/artifact/2fa526f2-d3ac-40bc-b7b5-3dce08190389
#
# Also listed on the P0 Notion "Models & datasets used" page, which is the URL
# registry for all six phases. Republishing from this repo path keeps the link;
# publishing a different path would mint a new one and strand the advisor.
#
# Usage — from the host, from the repo root. No need to enter the container;
# common/update.sh does that itself.
#   ./deeper_visuals/p1_inputs/update.sh                full run
#   ./deeper_visuals/p1_inputs/update.sh --page-only    wording/layout only
#
# P1 reads frames off disk rather than running a forward pass, so the full run
# needs /data and /outputs mounted but no GPU.
#
# Everything real lives in common/update.sh; this stub only pins the phase.
exec "$(dirname "${BASH_SOURCE[0]}")/../common/update.sh" \
  "$(dirname "${BASH_SOURCE[0]}")" "$@"

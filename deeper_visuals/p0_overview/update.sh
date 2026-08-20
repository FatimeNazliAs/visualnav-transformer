#!/usr/bin/env bash
# deeper_visuals/p0_overview/update.sh
#
# Artifact URL (fixed — every republish targets this same link):
#   https://claude.ai/code/artifact/1681b5a2-1e17-499a-af10-29bc876ba130
#
# Also listed on the P0 Notion "Models & datasets used" page, which is the URL
# registry for all six phases. Republishing from this repo path keeps the link;
# publishing a different path would mint a new one and strand the advisor.
#
# P0 is the odd one out: a hand-written primer, not a forward pass. It has no
# config.yaml, no run_model.py and no facts.json — so common/update.sh skips
# straight to the page build. Nothing here has to say so; the absent
# run_model.py is what it reads.
#
# Usage:
#   ./deeper_visuals/p0_overview/update.sh              build + serve on :8001
#   ./deeper_visuals/p0_overview/update.sh --no-serve   build only
#
# Edit the primer by editing body.html (the markup) or style.css (P0-only
# rules) in this folder, then rerun. --page-only is accepted but redundant:
# every run of P0 is already the page-only path.
exec "$(dirname "${BASH_SOURCE[0]}")/../common/update.sh" \
  "$(dirname "${BASH_SOURCE[0]}")" "$@"

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
# P0 is the odd one out: it is a hand-written primer, not a forward pass. There
# is no config.yaml, no run_model.py and no facts.json, so --static is baked in
# here rather than passed each time — it tells common/update.sh to promote
# p0_overview/latest.html as-is and serve it.
#
# Usage:
#   ./deeper_visuals/p0_overview/update.sh              promote + serve on :8001
#   ./deeper_visuals/p0_overview/update.sh --no-serve   promote only
#
# Edit the page by editing latest.html in this folder, then rerun. There is no
# --page-only here; every run of P0 is already the page-only path.
exec "$(dirname "${BASH_SOURCE[0]}")/../common/update.sh" \
  "$(dirname "${BASH_SOURCE[0]}")" --static "$@"

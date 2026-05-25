#!/usr/bin/env bash
set -euo pipefail

LI="/Users/lion/projects/.venv/bin/li"
WT="/Users/lion/projects/khive/lattice"

"$LI" agent -a reviewer --bypass --effort high --timeout 900 --cwd "$WT" "
Re-review PR #76 round 3 after round-2 fixes (commit ba9fc47).

## Round 2 findings and fixes:

1. **Major: Trusted i8 path double-scans** → FIXED: Extracted private
   dot_product_i8_dispatch() helper. Trusted callers now call dispatch
   directly after debug_assert, skipping the release assert in raw.

2. **Medium: SDOT safety docs/test assume NEON only** → FIXED: Updated
   safety docs to require FEAT_DotProd, gated test on dotprod_enabled,
   updated kernel wrapper comment.

## Focus for round 3
- Verify the dispatch split is correct (trusted path skips assert, public path asserts)
- Verify SDOT test gate works
- Any remaining issues from round 1 that weren't fully resolved
- No need to re-review layer_norm, decode softmax, or exp boundary (those were confirmed fixed in round 2)

Write findings to codex_review_pr76_round3.md.
"

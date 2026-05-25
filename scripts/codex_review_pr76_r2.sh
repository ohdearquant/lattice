#!/usr/bin/env bash
set -euo pipefail

LI="/Users/lion/projects/.venv/bin/li"
WT="/Users/lion/projects/khive/lattice"

"$LI" agent -a reviewer --bypass --effort high --timeout 900 --cwd "$WT" "
Re-review PR #76 after round-1 fixes (commit 02228f0).

## Round 1 findings and fixes applied:

1. **SDOT without dotprod check** → FIXED: Added dotprod_enabled to SimdConfig,
   gated SDOT dispatch on is_aarch64_feature_detected!(\"dotprod\").

2. **Public i8 APIs allow -128** → FIXED: Restored release assert! at public
   entry points (dot_product_i8, dot_product_i8_raw). Inner hot paths keep debug_assert.

3. **Fused layer_norm E[x²]-E[x]²** → FIXED: Reverted NEON and AVX2 to stable
   two-pass variance (sum → mean, then sum (x-mean)²).

4. **Decode softmax Schraudolph mismatch** → FIXED: Added test_decode_softmax_neon_vs_scalar_parity
   with adversarial patterns (all-equal, ramp, one-hot), atol=5e-3, rtol=0.05.

5. **Polynomial exp boundary infinity** → FIXED: Upper clamp reduced from 88.72→88.0,
   doc comment updated to state ~10-15 ULP (not 1 ULP).

## VERIFY blocks for round 2
- [ ] VERIFY each of the 5 fixes actually landed in the code (not just in commit message)
- [ ] VERIFY the dotprod gate falls through to scalar when dotprod is absent
- [ ] VERIFY the -128 assert! is at the right boundary (public functions only)
- [ ] VERIFY the two-pass layer_norm variance is numerically stable for large-offset inputs
- [ ] VERIFY no new issues introduced by the fixes

Be harsh. If the fixes are incomplete or incorrect, REQUEST CHANGES again.
Write findings to codex_review_pr76_round2.md.
"

#!/usr/bin/env bash
# ADR-064 Phase-0: decode slope/intercept fit harness driver.
#
# Builds bench_decode_slopefit (if needed), runs it, pipes stdout through the
# Python post-processor, and emits the final JSON.
#
# Usage:
#   ./scripts/bench_decode_slopefit.sh          # smoke grid {64,256,512}
#   SLOPEFIT_FULL=1 ./scripts/bench_decode_slopefit.sh   # full production grid
#   SLOPEFIT_CONTEXTS="64 512 1024" ./scripts/bench_decode_slopefit.sh
#
# Does NOT touch bench_decode_slope.sh or the bench-decode Make target.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/target/release/bench_decode_slopefit"
PY="$REPO/scripts/bench_decode_slopefit.py"

# Build if stale.
if [[ ! -x "$BIN" ]]; then
    >&2 echo "[slopefit] building bench_decode_slopefit (release)..."
    cargo build --release -p lattice-inference --bin bench_decode_slopefit \
        --features "f16,metal-gpu" 2>&1 | grep -v "^$" >&2 \
        || { echo '{"error":"build failed"}'; exit 1; }
fi

>&2 echo "[slopefit] running measurement binary..."
"$BIN" | tee /dev/stderr | uv run --project "$REPO" python3 "$PY"

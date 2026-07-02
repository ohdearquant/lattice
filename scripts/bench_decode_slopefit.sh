#!/usr/bin/env bash
# ADR-064: decode slope/intercept fit harness driver.
#
# Builds bench_decode_slopefit, runs it, pipes stdout through the
# Python post-processor, and emits the final ADR-064 JSON.
#
# Usage:
#   ./scripts/bench_decode_slopefit.sh          # smoke grid {64,256,512}
#   SLOPEFIT_FULL=1 ./scripts/bench_decode_slopefit.sh   # full production grid
#   SLOPEFIT_CONTEXTS="64 512 1024" ./scripts/bench_decode_slopefit.sh
#   ./scripts/bench_decode_slopefit.sh --out artifacts/adr064-gpu-decode-current.json
#
# Does NOT touch bench_decode_slope.sh or the bench-decode Make target.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/target/release/bench_decode_slopefit"
PY="$REPO/scripts/bench_decode_slopefit.py"

OUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            OUT="$2"
            shift 2
            ;;
        *)
            >&2 echo "[slopefit] unknown arg: $1"
            exit 1
            ;;
    esac
done

# Build before benchmarking; cargo incremental prevents stale release binaries.
>&2 echo "[slopefit] building bench_decode_slopefit (release)..."
cargo build --release -p lattice-inference --bin bench_decode_slopefit \
    --features "f16,metal-gpu" 2>&1 | grep -v "^$" >&2 \
    || { echo '{"error":"build failed"}'; exit 1; }

>&2 echo "[slopefit] running measurement binary..."
if [[ -n "$OUT" ]]; then
    mkdir -p "$(dirname "$OUT")"
    "$BIN" | tee /dev/stderr | uv run --project "$REPO" python3 "$PY" --out "$OUT"
else
    "$BIN" | tee /dev/stderr | uv run --project "$REPO" python3 "$PY"
fi

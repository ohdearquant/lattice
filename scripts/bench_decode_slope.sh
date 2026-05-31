#!/usr/bin/env bash
# ADR-064 Phase-0: Decode slope/intercept measurement harness.
#
# Runs decode at multiple context lengths, fits a linear model:
#   per_tok_ms = slope * ctx + intercept
#
# Output: JSON to stdout with slope_ms, intercept_ms, r_squared, tok_per_sec_64
#
# Usage:
#   ./scripts/bench_decode_slope.sh              # default 5 runs, contexts 64-1024
#   RUNS=3 ./scripts/bench_decode_slope.sh       # fewer runs (faster)
#   CONTEXTS="64 256 512" ./scripts/bench_decode_slope.sh
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LAT_BIN="$REPO/target/release/bench_decode_ab"
Q8_DIR="$HOME/.lattice/models/qwen3.5-0.8b"
RUNS="${RUNS:-5}"
N1=8

if [[ -z "${CONTEXTS:-}" ]]; then
    CONTEXTS_ARR=(64 128 256 512 1024)
else
    read -ra CONTEXTS_ARR <<< "$CONTEXTS"
fi

# Build if needed
if [[ ! -x "$LAT_BIN" ]]; then
    cargo build --release -p lattice-inference --bin bench_decode_ab \
        --features "f16,metal-gpu" 2>/dev/null \
        || { echo '{"error":"build failed"}'; exit 1; }
fi

# Measure at N1 (baseline)
lattice_median() {
    local n=$1
    env BENCH_N="$n" BENCH_RUNS="$RUNS" LATTICE_MODEL_DIR="$Q8_DIR" "$LAT_BIN" 2>/dev/null \
        | awk -F'total_ms=' '/^RESULT/{print $2}' \
        | sort -n \
        | awk "NR==$(( (RUNS + 1) / 2 ))"
}

T1=$(lattice_median $N1)
if [[ -z "$T1" ]]; then
    echo '{"error":"baseline measurement failed"}'
    exit 1
fi

# Collect (ctx, per_tok_ms) pairs
declare -a CTX_VALS=()
declare -a PTM_VALS=()

for CTX in "${CONTEXTS_ARR[@]}"; do
    T2=$(lattice_median "$CTX")
    if [[ -n "$T2" && "$T2" != "0" ]]; then
        # per_tok_ms = (T2 - T1) / (CTX - N1)
        PTM=$(echo "scale=6; ($T2 - $T1) / ($CTX - $N1)" | bc)
        CTX_VALS+=("$CTX")
        PTM_VALS+=("$PTM")
        >&2 echo "  ctx=$CTX: T2=${T2}ms per_tok=${PTM}ms"
    else
        >&2 echo "  ctx=$CTX: FAILED (skipping)"
    fi
done

N=${#CTX_VALS[@]}
if (( N < 2 )); then
    echo '{"error":"insufficient data points","n":'$N'}'
    exit 1
fi

# Linear regression: per_tok_ms = slope * ctx + intercept
# Using least-squares via awk
RESULT=$(awk -v n="$N" '
BEGIN {
    split(ENVIRON["CTX_STR"], xs, " ")
    split(ENVIRON["PTM_STR"], ys, " ")
    sx = 0; sy = 0; sxx = 0; sxy = 0
    for (i = 1; i <= n; i++) {
        x = xs[i] + 0
        y = ys[i] + 0
        sx += x; sy += y; sxx += x*x; sxy += x*y
    }
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    # R-squared
    ybar = sy / n
    sstot = 0; ssres = 0
    for (i = 1; i <= n; i++) {
        y = ys[i] + 0
        yhat = slope * (xs[i] + 0) + intercept
        sstot += (y - ybar)^2
        ssres += (y - yhat)^2
    }
    r2 = (sstot > 0) ? 1 - ssres/sstot : 1
    # tok/s at ctx=64
    ptm64 = slope * 64 + intercept
    tps64 = (ptm64 > 0) ? 1000 / ptm64 : 0
    printf "{\"slope_ms\":%.6f,\"intercept_ms\":%.4f,\"r_squared\":%.6f,\"tok_per_sec_64\":%.1f,\"n_points\":%d,\"contexts\":[", slope, intercept, r2, tps64, n
    for (i = 1; i <= n; i++) {
        if (i > 1) printf ","
        printf "%d", xs[i]
    }
    printf "],\"per_tok_ms\":["
    for (i = 1; i <= n; i++) {
        if (i > 1) printf ","
        printf "%.4f", ys[i]
    }
    printf "]}\n"
}' <<< "" CTX_STR="${CTX_VALS[*]}" PTM_STR="${PTM_VALS[*]}")

# Pretty print to stderr, raw JSON to stdout
>&2 echo ""
>&2 echo "=== Decode Slope Fit ==="
>&2 echo "$RESULT" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f\"  slope:     {d['slope_ms']:.6f} ms/ctx-token\")
print(f\"  intercept: {d['intercept_ms']:.4f} ms\")
print(f\"  R²:        {d['r_squared']:.6f}\")
print(f\"  tok/s@64:  {d['tok_per_sec_64']:.1f}\")
" 2>/dev/null

echo "$RESULT"

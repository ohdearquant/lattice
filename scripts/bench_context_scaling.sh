#!/usr/bin/env bash
# Context-length scaling benchmark — measures decode throughput at multiple
# generation lengths to show how attention cost scales with KV cache size.
#
# Thin wrapper (issue #813 step 2): all methodology (warmup policy, repeat
# count, aggregation, output rendering) now lives in
# scripts/bench_decode_harness.py, configured by the context_scaling profile
# in scripts/bench_decode_profiles.toml. This script execs the harness (via
# scripts/bench_decode_adapters_context_scaling.py, which registers the
# lattice/ollama/mlx engine adapters AND owns the legacy chart-compatible
# TSV rendering + scripts/bench_context_scaling_chart.py invocation this
# script always did). Default parameters are unchanged: N1=8 baseline,
# contexts {64,128,256}, 5 measured repeats, mlx-only 4-token warmup.
#
# Usage (env vars forwarded verbatim, same names as before migration):
#   ./scripts/bench_context_scaling.sh                   # default: 5 runs
#   RUNS=10 ./scripts/bench_context_scaling.sh           # more runs
#   CONTEXTS="64 128 256" ./scripts/bench_context_scaling.sh  # custom lengths
#   CHART_ONLY=1 ./scripts/bench_context_scaling.sh      # regenerate chart from existing data
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/target/bench_results"
mkdir -p "$OUT"

ARGS=(--allow-missing-engine --out "$OUT/context_scaling_raw.jsonl")

if [[ "${CHART_ONLY:-0}" == "1" ]]; then
  ARGS=(--chart-only)
else
  if [[ -n "${CONTEXTS:-}" ]]; then
    ARGS+=(--contexts "$(echo "${CONTEXTS}" | tr ' ' ',')")
  fi
  if [[ -n "${RUNS:-}" ]]; then
    ARGS+=(--runs "$RUNS")
  fi
fi

uv run --quiet --with mlx-lm python3 "$REPO/scripts/bench_decode_adapters_context_scaling.py" "${ARGS[@]}"
exit $?

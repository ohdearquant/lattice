#!/usr/bin/env bash
# Precise apples-to-apples decode benchmark — reduced noise edition.
#
# Thin wrapper (issue #813 step 2): all methodology (warmup policy, repeat
# count, trimmed-mean aggregation, legacy normal-approximation CI, output
# rendering) now lives in scripts/bench_decode_harness.py, configured by the
# apples_precise profile in scripts/bench_decode_profiles.toml. This script
# only execs the harness (via scripts/bench_decode_adapters_apples_precise.py,
# which registers the lattice/ollama/mlx engine adapters). Parameters are
# unchanged from the pre-migration version of this script: N1=64, N2=512,
# 13 measured repeats + 2 warmup repeats per engine (15 total), trimmed
# mean dropping the top/bottom 2 measured values, legacy 95%-normal-
# approximation CI.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/target/bench_results"
mkdir -p "$OUT"

echo "=== Precise decode bench | Qwen3.5-0.8B | N1=64 N2=512 measured=13 warmup=2 (per engine) ==="
echo "  Harness: scripts/bench_decode_harness.py (profile apples_precise)"
echo ""

uv run --quiet --with mlx-lm python3 "$REPO/scripts/bench_decode_adapters_apples_precise.py" \
  run --profile apples_precise --allow-missing-engine --out "$OUT/precise_raw.jsonl"
status=$?

echo ""
echo "Raw observations: $OUT/precise_raw.jsonl"
exit $status

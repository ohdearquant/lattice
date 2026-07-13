#!/usr/bin/env bash
# Quant-MATCHED apples-to-apple decode benchmark — slope (marginal) method, Q4.
#
# Thin wrapper (issue #813 step 2): all methodology (warmup policy, repeat
# count, aggregation, output rendering) now lives in
# scripts/bench_decode_harness.py, configured by the q4_apples profile in
# scripts/bench_decode_profiles.toml. This script only execs the harness
# (via scripts/bench_decode_adapters_q4_apples.py, which registers the
# lattice/ollama/mlx engine adapters). Parameters are unchanged from the
# pre-migration version of this script: N1=32, N2=256, 5 measured repeats,
# the same fixed prompt, mlx-only 8-token warmup.
#
# DISCLOSED CORRECTION (see bench_decode_profiles.toml's [profiles.q4_apples]
# comment): the pre-migration version of this script set env vars
# (BENCH_Q4_DIR/BENCH_TOKENIZER_DIR) that bench_decode_ab never read, so its
# lattice leg silently benchmarked Q8 (the LATTICE_MODEL_DIR default)
# instead of the Q4 directory it claimed. This migration's adapter wires
# LATTICE_MODEL_DIR to the real Q4 directory — a bug fix, not a parameter
# change.
#
#   Lattice Q4 (plain per-row, ~/.lattice/models/qwen3.5-0.8b-q4)
#   vs MLX Q4  (nn.quantize bits=4 group_size=64, same source weights)
#   Ollama     = Q8_0 REFERENCE ONLY — no Q4 tag exists for qwen3.5:0.8b
#               (verified via `ollama show`), so it is NOT a Q4 competitor.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/target/bench_results"
mkdir -p "$OUT"

echo "=== Q4 apples-to-apple decode bench | Qwen3.5-0.8B | N1=32 N2=256 runs=5 ==="
echo "    Lattice=Q4 plain | MLX=Q4 g64 | Ollama=Q8_0 (reference, no Q4 tag)"
echo "  Harness: scripts/bench_decode_harness.py (profile q4_apples)"
echo ""

uv run --quiet --with mlx-lm python3 "$REPO/scripts/bench_decode_adapters_q4_apples.py" \
  run --profile q4_apples --allow-missing-engine --out "$OUT/q4_a2a_raw.jsonl"
status=$?

echo ""
echo "Raw observations: $OUT/q4_a2a_raw.jsonl"
exit $status

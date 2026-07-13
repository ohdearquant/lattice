#!/usr/bin/env bash
# Apples-to-apple decode-throughput benchmark — slope (marginal) method.
#
# Thin wrapper (issue #813 step 2, first migration): all methodology
# (warmup policy, repeat count, aggregation, confidence interval, output
# rendering) now lives in scripts/bench_decode_harness.py, configured by the
# apples_to_apples_q8/apples_to_apples_q4 profiles in
# scripts/bench_decode_profiles.toml. This script only execs the harness
# (via scripts/bench_decode_adapters_apples_to_apples.py, which registers
# the lattice/ollama/mlx engine adapters) once per tier. Parameters are
# unchanged from the pre-migration version of this script: N1=32, N2=256,
# 5 measured repeats, the same fixed prompt, and the same per-engine warmup
# shape (only mlx warms up, once, at 8 tokens).
#
# Q8 tier (lattice default for <2B params, ollama default, mlx explicit):
#   - lattice: F16 safetensors → auto-quantized to Q8_0 on Metal upload
#   - ollama:  qwen3.5:0.8b (registry default is Q8_0)
#   - mlx:     nn.quantize(bits=8, group_size=64)
#
# Q4 tier (lattice's product differentiator — QuaRot-rotated 4-bit):
#   - lattice: .q4 files in qwen3.5-0.8b-q4-quarot dir
#   - mlx:     nn.quantize(bits=4, group_size=64)
#   - ollama:  SKIPPED — no Q4 variant for qwen3.5:0.8b in registry
#
# MLX uses Apple's private MPS/MPSGraph (AMX matrix engines) — strictly a
# different category than public-Metal-compute engines (lattice, ollama).
# Reported for reference, not as the headline comparison.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/docs/bench_results"
mkdir -p "$OUT"

echo "=== Apples-to-apple decode bench | Qwen3.5-0.8B | N1=32 N2=256 runs=5 ==="
echo "  Harness: scripts/bench_decode_harness.py (profiles apples_to_apples_q8 / apples_to_apples_q4)"
echo ""

echo "─── Q8 tier (apples — public Metal compute API; MLX reference uses private AMX) ───"
uv run --quiet --with mlx-lm python3 "$REPO/scripts/bench_decode_adapters_apples_to_apples.py" \
  run --profile apples_to_apples_q8 --allow-missing-engine --out "$OUT/a2a_q8_raw.jsonl"
q8_status=$?
echo ""

echo "─── Q4 tier (lattice differentiator — QuaRot rotation; MLX Q4 g64 reference) ───"
uv run --quiet --with mlx-lm python3 "$REPO/scripts/bench_decode_adapters_apples_to_apples.py" \
  run --profile apples_to_apples_q4 --allow-missing-engine --out "$OUT/a2a_q4_raw.jsonl"
q4_status=$?
echo ""

echo "Raw observations: $OUT/a2a_q8_raw.jsonl, $OUT/a2a_q4_raw.jsonl"

if [[ $q8_status -ne 0 || $q4_status -ne 0 ]]; then
  exit 1
fi

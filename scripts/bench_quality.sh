#!/usr/bin/env bash
# bench_quality.sh — perplexity comparison across lattice tiers + MLX cross-check.
#
# Per-token negative-log-likelihood (NLL) on the standard WikiText-2 raw test
# split. Lower PPL = better. Compares:
#   - lattice Metal Q4-QuaRot (the lattice product differentiator)
#   - lattice Metal Q4 (unrotated, baseline for QuaRot improvement)
#   - MLX Q8 g64 (cross-check: their quantization vs our Q8 implementation)
#   - MLX Q4 g64 (cross-check vs Q4-QuaRot)
#
# Ollama is omitted: no public logprobs API; would need to drop down to
# llama.cpp's llama-perplexity binary which Ollama doesn't expose.
#
# Lattice F16/CPU is omitted by default (slow); enable with FULL_BENCH=1
# for the absolute gold-standard baseline.
#
# Output: docs/bench_results/perplexity.tsv (engine<TAB>tier<TAB>ppl<TAB>tokens)
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_BIN="$REPO/target/release/eval_perplexity"
Q4_DIR="${Q4_DIR:-$HOME/.lattice/models/qwen3.5-0.8b-q4}"
QUAROT_DIR="${QUAROT_DIR:-$HOME/.lattice/models/qwen3.5-0.8b-q4-quarot}"
TOK_DIR="${TOK_DIR:-$HOME/.lattice/models/qwen3.5-0.8b}"
OUT="$REPO/docs/bench_results"
CORPUS="$OUT/wiki.test.raw"
DATA="$OUT/perplexity.tsv"
MAX_TOKENS="${MAX_TOKENS:-2048}"   # ~20s on Metal Q4 (107 tok/s scoring rate)
WINDOW="${WINDOW:-512}"            # Buffer is window*vocab*4 = ~508MB at vocab=248K
STRIDE="${STRIDE:-256}"            # 2x stride overlap → adequate context coverage
SEED="${SEED:-0xC0FFEE}"           # QuaRot rotation seed (artifacts not interchangeable across seeds)
SKIP_MLX="${SKIP_MLX:-0}"          # set 1 to skip the MLX cross-check (no mlx-lm / offline)

mkdir -p "$OUT"

if [[ ! -f "$CORPUS" ]]; then
  echo "  ERROR: $CORPUS not found. Run:" >&2
  echo "    curl -L https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt -o '$CORPUS'" >&2
  exit 1
fi
if [[ ! -d "$Q4_DIR" ]]; then
  echo "  ERROR: lattice Q4 model directory not found: $Q4_DIR" >&2
  exit 1
fi
if [[ ! -d "$QUAROT_DIR" ]]; then
  echo "  ERROR: lattice Q4-QuaRot model directory not found: $QUAROT_DIR" >&2
  exit 1
fi
if [[ ! -d "$TOK_DIR" ]]; then
  echo "  ERROR: tokenizer directory not found: $TOK_DIR" >&2
  exit 1
fi
if [[ ! -x "$EVAL_BIN" ]]; then
  echo "  ERROR: $EVAL_BIN is not executable. Build it with:" >&2
  echo "    cargo build --release -p lattice-inference --bin eval_perplexity" >&2
  exit 1
fi
if [[ "$SKIP_MLX" != "0" ]] && [[ "$SKIP_MLX" != "1" ]]; then
  echo "  ERROR: SKIP_MLX must be 0 or 1, got: $SKIP_MLX" >&2
  exit 1
fi
if [[ "$SKIP_MLX" == "0" ]] && ! command -v uv >/dev/null 2>&1; then
  echo "  ERROR: uv is required for the MLX cross-check (or set SKIP_MLX=1)" >&2
  exit 1
fi

DATA_TMP="$(mktemp "$OUT/.perplexity.tsv.XXXXXX")"
if [[ -z "$DATA_TMP" ]] || [[ ! -f "$DATA_TMP" ]]; then
  echo "  ERROR: failed to create a temporary result file under $OUT" >&2
  exit 1
fi
MLX_TMP=""
cleanup() {
  [[ -n "$DATA_TMP" ]] && rm -f "$DATA_TMP"
  [[ -n "$MLX_TMP" ]] && rm -f "$MLX_TMP"
}
trap cleanup EXIT
if ! chmod 0644 "$DATA_TMP"; then
  echo "  ERROR: failed to set result-file permissions on $DATA_TMP" >&2
  exit 1
fi

write_failed() {
  echo "  ERROR: failed to write $1 to staged output $DATA_TMP" >&2
  exit 1
}

append_lines() {
  context="$1"
  shift
  printf "%s\n" "$@" >> "$DATA_TMP" || write_failed "$context"
}

append_row() {
  context="$1"
  shift
  printf "%s\t%s\t%s\t%s\n" "$@" >> "$DATA_TMP" || write_failed "$context"
}

SHA="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
STAMP="$(date -u +%Y-%m-%dT%H:%MZ)"
MACHINE="${BENCH_MACHINE:-$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)}"
ARCH="$(uname -m 2>/dev/null || echo unknown)"
HARNESS_VERSION="$(git hash-object "$0" 2>/dev/null | cut -c1-12)"
[[ -n "$HARNESS_VERSION" ]] || HARNESS_VERSION=unknown
append_lines "result header" \
  "# perplexity.tsv — Q4 quantization quality, lattice tiers vs MLX cross-check" \
  "# Regenerated $STAMP from $SHA (real GPU). Qwen3.5-0.8B, WikiText-2 raw test," \
  "# Provenance: machine=$MACHINE arch=$ARCH harness=bench_quality.sh@$HARNESS_VERSION." \
  "#   window=$WINDOW stride=$STRIDE max_tokens=$MAX_TOKENS, QuaRot seed=$SEED." \
  "#   PPL is deterministic over a fixed corpus + window schedule (one run is canonical)." \
  "# ADR-044 reconciliation: ADR-044 step-4 recorded QuaRot Q4 as -1.61 PPL BETTER than" \
  "#   unrotated Q4 (full corpus). That was measured on a PRE-RoPE-fix forward path (May 2026)." \
  "#   A code-bisection on identical corpus/slice/seed shows the delta sign is set by code" \
  "#   version, not corpus: ADR-044-era binary = -0.81 (QuaRot better), current binary = +1.90" \
  "#   (QuaRot worse). Forward-path fixes improved the unrotated baseline ~4.5 PPL but QuaRot" \
  "#   only ~1.8 — QuaRot was compensating for baseline bugs since fixed. Offline QuaRot v0 is" \
  "#   net-negative by design (Hadamard forces symmetric Q4, worse fidelity floor); the missing" \
  "#   mechanism is online R3/R4 (issue #703). See also #616. Columns: engine<TAB>tier<TAB>ppl<TAB>tokens"
echo "=== Perplexity bench | Qwen3.5-0.8B | WikiText-2 test | window=$WINDOW stride=$STRIDE max_tokens=$MAX_TOKENS ==="

echo "  Corpus: $CORPUS ($(wc -c < "$CORPUS") bytes)"

extract_ppl() {
  # Match exactly the eval_perplexity output line: "PPL:                NN.NNNNNN"
  awk '/^PPL:[[:space:]]+[0-9]+\.[0-9]+/{ print $2; exit }'
}

# ---- Lattice Q4 (unrotated) ----
echo "─── Lattice Q4 (unrotated) ───"
OUT_TXT=$("$EVAL_BIN" --q4-dir "$Q4_DIR" --tokenizer-dir "$TOK_DIR" \
  --corpus-file "$CORPUS" --window "$WINDOW" --stride "$STRIDE" \
  --max-tokens "$MAX_TOKENS" 2>&1)
EVAL_RC=$?
if [[ "$EVAL_RC" -ne 0 ]]; then
  echo "  ERROR: eval_perplexity failed for lattice/q4 (exit $EVAL_RC)" >&2
  printf "%s\n" "$OUT_TXT" >&2
  exit 1
fi
PPL=$(echo "$OUT_TXT" | extract_ppl | head -1)
if [[ -z "$PPL" ]]; then
  echo "  ERROR: eval_perplexity produced no parseable PPL for lattice/q4" >&2
  exit 1
fi
echo "  PPL: $PPL"
Q4PPL="$PPL"
append_row "lattice/q4 result" "lattice" "q4" "$PPL" "$MAX_TOKENS"

# ---- Lattice Q4-QuaRot ----
echo "─── Lattice Q4-QuaRot (lattice product) ───"
OUT_TXT=$("$EVAL_BIN" --quarot-q4-dir "$QUAROT_DIR" --tokenizer-dir "$TOK_DIR" \
  --corpus-file "$CORPUS" --window "$WINDOW" --stride "$STRIDE" \
  --max-tokens "$MAX_TOKENS" 2>&1)
EVAL_RC=$?
if [[ "$EVAL_RC" -ne 0 ]]; then
  echo "  ERROR: eval_perplexity failed for lattice/q4-quarot (exit $EVAL_RC)" >&2
  printf "%s\n" "$OUT_TXT" >&2
  exit 1
fi
PPL=$(echo "$OUT_TXT" | extract_ppl | head -1)
if [[ -z "$PPL" ]]; then
  echo "  ERROR: eval_perplexity produced no parseable PPL for lattice/q4-quarot" >&2
  exit 1
fi
echo "  PPL: $PPL"
QRPPL="$PPL"
append_row "lattice/q4-quarot result" "lattice" "q4-quarot" "$PPL" "$MAX_TOKENS"

# ---- QuaRot delta (INFORMATIONAL, not a gate — offline v0 is net-negative, see header/#703) ----
if [[ -n "${Q4PPL:-}" ]] && [[ -n "${QRPPL:-}" ]]; then
  DELTA=$(awk -v q="$Q4PPL" -v r="$QRPPL" 'BEGIN{printf "%+.4f", r-q}')
  echo "  QuaRot delta (quarot-unrotated): $DELTA  [informational; positive = QuaRot worse]"
  append_lines "QuaRot delta" "# informational: quarot-unrotated delta = $DELTA at max_tokens=$MAX_TOKENS (offline v0 net-negative, #703)"
fi

# ---- MLX Q8 + Q4 (cross-check) ----
if [[ "$SKIP_MLX" == "1" ]]; then
  echo "─── MLX cross-check: SKIP (SKIP_MLX=1) ───"
else
echo "─── MLX (Q8 + Q4 cross-check) ───"
# Capture stdout to a temp; only clean "mlx<TAB>..." rows are appended to the
# unpublished result so a
# broken mlx-lm (import/tokenizer errors) can never pollute the data file. stderr → log.
MLX_TMP="$(mktemp)"
if [[ -z "$MLX_TMP" ]] || [[ ! -f "$MLX_TMP" ]]; then
  echo "  ERROR: failed to create the MLX output tempfile" >&2
  exit 1
fi
uv run --quiet --with mlx-lm python3 - "$TOK_DIR" "$CORPUS" "$WINDOW" "$STRIDE" "$MAX_TOKENS" > "$MLX_TMP" 2>/tmp/mlx_ppl.log <<'PY'
import sys, math
mdir, corpus, window, stride, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

text = open(corpus, encoding='utf-8').read()

def ppl_at_bits(bits, label):
    try:
        model, tok = load(mdir)
    except Exception as e:
        sys.stderr.write(f"mlx load failed: {e}\n"); return
    nn.quantize(model, bits=bits, group_size=64); mx.eval(model.parameters())
    ids = tok.encode(text)[:max_tokens]
    N = len(ids)
    # Strided sliding-window perplexity (same as eval_perplexity)
    total_nll = 0.0; total_tokens = 0
    start = 0
    while start < N:
        end = min(start + window, N)
        chunk = mx.array(ids[start:end])[None, :]
        logits = model(chunk)                       # (1, T, V)
        log_probs = nn.log_softmax(logits[0, :-1, :], axis=-1)  # (T-1, V)
        targets = chunk[0, 1:]                      # (T-1,)
        nll = -mx.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
        # Score only the new tokens (stride window) for unbiased measurement
        score_start = 0 if start == 0 else (window - stride)
        scores = nll[score_start:]
        total_nll += float(scores.sum())
        total_tokens += int(scores.shape[0])
        if end >= N: break
        start += stride
    ppl = math.exp(total_nll / max(total_tokens, 1))
    sys.stderr.write(f"  {label}: PPL = {ppl:.4f} ({total_tokens} tokens)\n")
    print(f"mlx\t{label}\t{ppl:.4f}\t{total_tokens}")

ppl_at_bits(8, "q8")
ppl_at_bits(4, "q4")
PY
MLX_RC=$?
if [[ "$MLX_RC" -ne 0 ]]; then
  echo "  ERROR: MLX cross-check failed (exit $MLX_RC; see /tmp/mlx_ppl.log)" >&2
  exit 1
fi
for TIER in q8 q4; do
  if ! MLX_ROW="$(awk -F'\t' -v tier="$TIER" '
    $1 == "mlx" && $2 == tier && NF == 4 { row = $0; count++ }
    END {
      if (count != 1) exit 1
      print row
    }
  ' "$MLX_TMP")"; then
    echo "  ERROR: MLX cross-check did not produce exactly one $TIER row (see /tmp/mlx_ppl.log)" >&2
    exit 1
  fi
  if [[ -z "$MLX_ROW" ]]; then
    echo "  ERROR: MLX cross-check produced an empty $TIER row (see /tmp/mlx_ppl.log)" >&2
    exit 1
  fi
  append_lines "MLX/$TIER result" "$MLX_ROW"
done
rm -f "$MLX_TMP"
MLX_TMP=""
fi

if ! awk -F '\t' -v skip_mlx="$SKIP_MLX" '
  /^#/ { next }
  NF != 4 { bad = 1; next }
  $3 !~ /^[0-9]+([.][0-9]+)?$/ || $4 !~ /^[0-9]+$/ { bad = 1; next }
  $1 == "lattice" && $2 == "q4" { lattice_q4++; next }
  $1 == "lattice" && $2 == "q4-quarot" { lattice_quarot++; next }
  $1 == "mlx" && $2 == "q8" { mlx_q8++; next }
  $1 == "mlx" && $2 == "q4" { mlx_q4++; next }
  { bad = 1 }
  END {
    expected_mlx = skip_mlx == "0" ? 1 : 0
    if (bad || lattice_q4 != 1 || lattice_quarot != 1 ||
        mlx_q8 != expected_mlx || mlx_q4 != expected_mlx) {
      exit 1
    }
  }
' "$DATA_TMP"; then
  echo "  ERROR: staged results failed schema/cardinality validation; refusing publication" >&2
  exit 1
fi

echo ""
echo "═══ Perplexity Summary ═══"
if [[ -s "$DATA_TMP" ]]; then
  printf "  %-10s %-10s %10s %10s\n" "engine" "tier" "PPL ↓" "tokens"
  printf "  %s\n" "----------------------------------------"
  awk -F'\t' '/^#/{next} NF>=4{printf "  %-10s %-10s %10s %10s\n", $1, $2, $3, $4}' "$DATA_TMP"
else
  echo "  (no measurements completed)"
fi

if ! mv -f "$DATA_TMP" "$DATA"; then
  echo "  ERROR: failed to publish completed results to $DATA" >&2
  exit 1
fi
DATA_TMP=""

echo ""
echo "Raw data: $DATA"

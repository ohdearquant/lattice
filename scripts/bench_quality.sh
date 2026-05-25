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
Q4_DIR="$HOME/.lattice/models/qwen3.5-0.8b-q4"
QUAROT_DIR="$HOME/.lattice/models/qwen3.5-0.8b-q4-quarot"
TOK_DIR="$HOME/.lattice/models/qwen3.5-0.8b"
OUT="$REPO/docs/bench_results"
CORPUS="$OUT/wiki.test.raw"
DATA="$OUT/perplexity.tsv"
MAX_TOKENS="${MAX_TOKENS:-2048}"   # ~20s on Metal Q4 (107 tok/s scoring rate)
WINDOW="${WINDOW:-512}"            # Buffer is window*vocab*4 = ~508MB at vocab=248K
STRIDE="${STRIDE:-256}"            # 2x stride overlap → adequate context coverage

mkdir -p "$OUT"
: > "$DATA"
echo "=== Perplexity bench | Qwen3.5-0.8B | WikiText-2 test | window=$WINDOW stride=$STRIDE max_tokens=$MAX_TOKENS ==="

# ---- Corpus check ----
if [[ ! -f "$CORPUS" ]]; then
  echo "  ERROR: $CORPUS not found. Run:"
  echo "    curl -L https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt -o '$CORPUS'"
  exit 1
fi
echo "  Corpus: $CORPUS ($(wc -c < "$CORPUS") bytes)"

extract_ppl() {
  # Match exactly the eval_perplexity output line: "PPL:                NN.NNNNNN"
  awk '/^PPL:[[:space:]]+[0-9]+\.[0-9]+/{ print $2; exit }'
}

# ---- Lattice Q4 (unrotated, requires Q4 dir; skip if missing) ----
if [[ -d "$Q4_DIR" ]] && [[ -x "$EVAL_BIN" ]]; then
  echo "─── Lattice Q4 (unrotated) ───"
  OUT_TXT=$("$EVAL_BIN" --q4-dir "$Q4_DIR" --tokenizer-dir "$TOK_DIR" \
    --corpus-file "$CORPUS" --window "$WINDOW" --stride "$STRIDE" \
    --max-tokens "$MAX_TOKENS" 2>&1)
  PPL=$(echo "$OUT_TXT" | extract_ppl | head -1)
  echo "  PPL: ${PPL:-PARSE_FAILED}"
  [[ -n "$PPL" ]] && printf "lattice\tq4\t%s\t%s\n" "$PPL" "$MAX_TOKENS" >> "$DATA"
else
  echo "  lattice/q4: SKIP (no $Q4_DIR; quantize via target/release/quantize_q4 to enable)"
fi

# ---- Lattice Q4-QuaRot ----
if [[ -d "$QUAROT_DIR" ]] && [[ -x "$EVAL_BIN" ]]; then
  echo "─── Lattice Q4-QuaRot (lattice product) ───"
  OUT_TXT=$("$EVAL_BIN" --quarot-q4-dir "$QUAROT_DIR" --tokenizer-dir "$TOK_DIR" \
    --corpus-file "$CORPUS" --window "$WINDOW" --stride "$STRIDE" \
    --max-tokens "$MAX_TOKENS" 2>&1)
  PPL=$(echo "$OUT_TXT" | extract_ppl | head -1)
  echo "  PPL: ${PPL:-PARSE_FAILED}"
  [[ -n "$PPL" ]] && printf "lattice\tq4-quarot\t%s\t%s\n" "$PPL" "$MAX_TOKENS" >> "$DATA"
else
  echo "  lattice/q4-quarot: SKIP (no $QUAROT_DIR)"
fi

# ---- MLX Q8 + Q4 (cross-check) ----
echo "─── MLX (Q8 + Q4 cross-check) ───"
uv run --quiet --with mlx-lm python3 - "$TOK_DIR" "$CORPUS" "$WINDOW" "$STRIDE" "$MAX_TOKENS" <<'PY' >> "$DATA" 2>&1 | tee /tmp/mlx_ppl.log
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

echo ""
echo "═══ Perplexity Summary ═══"
if [[ -s "$DATA" ]]; then
  printf "  %-10s %-10s %10s %10s\n" "engine" "tier" "PPL ↓" "tokens"
  printf "  %s\n" "----------------------------------------"
  awk -F'\t' '{printf "  %-10s %-10s %10s %10s\n", $1, $2, $3, $4}' "$DATA"
else
  echo "  (no measurements completed)"
fi

echo ""
echo "Raw data: $DATA"

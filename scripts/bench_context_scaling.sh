#!/usr/bin/env bash
# Context-length scaling benchmark — measures decode throughput at multiple
# generation lengths to show how attention cost scales with KV cache size.
#
# Slope method: for each context length C, run N1=8 and N2=C, then
#   decode_tok_s = (C - 8) / (T(C) - T(8))
# This gives average marginal decode rate over the [8, C] window, with
# prefill overhead cancelled.
#
# Model EOS limit: Qwen3.5-0.8B generates ~346 tokens max for the bench
# prompt before hitting EOS. Context lengths beyond 256 are unreliable
# without disabling EOS.
#
# Engines: lattice (Q8 + f16 lm_head), ollama (Q8_0), mlx (Q8 g64)
#
# Usage:
#   ./scripts/bench_context_scaling.sh                   # default: 5 runs
#   RUNS=10 ./scripts/bench_context_scaling.sh           # more runs
#   CONTEXTS="64 128 256" ./scripts/bench_context_scaling.sh  # custom lengths
#   CHART_ONLY=1 ./scripts/bench_context_scaling.sh      # regenerate chart from existing data
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LAT_BIN="$REPO/target/release/bench_decode_ab"
Q8_DIR="$HOME/.lattice/models/qwen3.5-0.8b"
RUNS="${RUNS:-5}"
N1=8  # fixed baseline — small enough to not dominate, large enough to warm caches
OUT="$REPO/docs/bench_results"
mkdir -p "$OUT"
DATA="$OUT/context_scaling.tsv"
CHART="$OUT/context_scaling_benchmark.png"
PROMPT="The quick brown fox jumps over the lazy dog. Once upon a time in a land far away, there lived a"

# Default context lengths (override with CONTEXTS env var)
if [[ -z "${CONTEXTS:-}" ]]; then
    CONTEXTS_ARR=(64 128 256)
else
    read -ra CONTEXTS_ARR <<< "$CONTEXTS"
fi

# ─── Chart-only mode ───
if [[ "${CHART_ONLY:-0}" == "1" ]]; then
    echo "=== Regenerating chart from $DATA ==="
    uv run python "$REPO/scripts/bench_context_scaling_chart.py" "$DATA" "$CHART"
    exit $?
fi

# ─── Build lattice if needed ───
if [[ ! -x "$LAT_BIN" ]]; then
    echo "Building bench_decode_ab (release)..."
    cargo build --release -p lattice-inference --bin bench_decode_ab --features "f16,metal-gpu" 2>/dev/null
fi

echo "=== Context-length scaling | Qwen3.5-0.8B Q8 | N1=$N1 | runs=$RUNS ==="
echo "  Contexts: ${CONTEXTS_ARR[*]}"
echo "  Output:   $DATA"
echo ""

# Header
echo -e "engine\tcontext_tokens\tslope_tok_s\tt1_ms\tt2_ms\truns" > "$DATA"

# ─── Helper: run bench_decode_ab, return median total_ms ───
lattice_median() {
    local n=$1
    env BENCH_N="$n" BENCH_RUNS="$RUNS" LATTICE_MODEL_DIR="$Q8_DIR" "$LAT_BIN" 2>/dev/null \
        | awk -F'total_ms=' '/^RESULT/{print $2}' \
        | sort -n \
        | awk "NR==$(( (RUNS + 1) / 2 ))"
}

# ─── Helper: run ollama, return median total_ms ───
ollama_median() {
    local n=$1
    local vals=()
    for ((r=1; r<=RUNS; r++)); do
        local ms
        ms=$(curl -s http://localhost:11434/api/generate \
            -d "{\"model\":\"qwen3.5:0.8b\",\"prompt\":\"$PROMPT\",\"options\":{\"num_predict\":$n,\"temperature\":0,\"seed\":42},\"stream\":false}" \
            | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"total_duration\"]/1e6:.3f}')" 2>/dev/null)
        [[ -n "$ms" ]] && vals+=("$ms")
    done
    if (( ${#vals[@]} == 0 )); then echo ""; return; fi
    printf '%s\n' "${vals[@]}" | sort -n | awk "NR==$(( (${#vals[@]} + 1) / 2 ))"
}

# ─── Lattice ───
echo "─── Lattice (Q8, f16 lm_head) ───"
# Pre-measure N1 once (shared across all context lengths)
LAT_T1=$(lattice_median $N1)
if [[ -z "$LAT_T1" ]]; then
    echo "  FAILED: bench_decode_ab not working"
else
    echo "  N1=$N1 baseline: ${LAT_T1}ms (median of $RUNS)"
    for CTX in "${CONTEXTS_ARR[@]}"; do
        T2=$(lattice_median "$CTX")
        if [[ -n "$T2" ]]; then
            SLOPE=$(echo "scale=1; ($CTX - $N1) / (($T2 - $LAT_T1) / 1000)" | bc)
            echo "  ctx=$CTX: T2=${T2}ms → ${SLOPE} tok/s"
            echo -e "lattice\t$CTX\t$SLOPE\t$LAT_T1\t$T2\t$RUNS" >> "$DATA"
        else
            echo "  ctx=$CTX: FAILED (model may have hit EOS)"
        fi
    done
fi

# ─── Ollama ───
echo ""
echo "─── Ollama (Q8_0, llama.cpp Metal) ───"
if command -v ollama &>/dev/null && curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c "import json,sys; models=[m['name'] for m in json.load(sys.stdin).get('models',[])]; sys.exit(0 if any('qwen3.5:0.8b' in m for m in models) else 1)" 2>/dev/null; then
    OLL_T1=$(ollama_median $N1)
    if [[ -z "$OLL_T1" ]]; then
        echo "  FAILED: ollama not responding"
    else
        echo "  N1=$N1 baseline: ${OLL_T1}ms (median of $RUNS)"
        for CTX in "${CONTEXTS_ARR[@]}"; do
            T2=$(ollama_median "$CTX")
            if [[ -n "$T2" ]]; then
                SLOPE=$(echo "scale=1; ($CTX - $N1) / (($T2 - $OLL_T1) / 1000)" | bc)
                echo "  ctx=$CTX: T2=${T2}ms → ${SLOPE} tok/s"
                echo -e "ollama\t$CTX\t$SLOPE\t$OLL_T1\t$T2\t$RUNS" >> "$DATA"
            else
                echo "  ctx=$CTX: FAILED"
            fi
        done
    fi
else
    echo "  SKIPPED (ollama not running or qwen3.5:0.8b not pulled)"
fi

# ─── MLX ───
echo ""
echo "─── MLX (Q8 g64, private AMX) ───"
MLX_SCRIPT=$(cat <<'PYTHON'
import sys, time, json
import mlx.core as mx
import mlx.nn as nn
import mlx_lm

model, tokenizer = mlx_lm.load("Qwen/Qwen3.5-0.8B",
                                tokenizer_config={"eos_token": "<|endoftext|>"})
nn.quantize(model, bits=8, group_size=64)

prompt = "The quick brown fox jumps over the lazy dog. Once upon a time in a land far away, there lived a"
greedy = lambda logits: mx.argmax(logits, axis=-1)

# warmup
_ = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=4,
                     sampler=greedy, verbose=False)
mx.eval(mx.zeros(1))

n1 = int(sys.argv[1])
contexts = json.loads(sys.argv[2])
runs = int(sys.argv[3])

# Measure N1 baseline
times_n1 = []
for _ in range(runs):
    mx.eval(mx.zeros(1))
    t0 = time.perf_counter()
    _ = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=n1,
                         sampler=greedy, verbose=False)
    mx.eval(mx.zeros(1))
    times_n1.append((time.perf_counter() - t0) * 1000)
t1 = sorted(times_n1)[len(times_n1) // 2]
sys.stderr.write(f"  N1={n1} baseline: {t1:.1f}ms (median of {runs})\n")

for ctx in contexts:
    times_n2 = []
    for _ in range(runs):
        mx.eval(mx.zeros(1))
        t0 = time.perf_counter()
        _ = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=ctx,
                             sampler=greedy, verbose=False)
        mx.eval(mx.zeros(1))
        times_n2.append((time.perf_counter() - t0) * 1000)
    t2 = sorted(times_n2)[len(times_n2) // 2]
    slope = (ctx - n1) / ((t2 - t1) / 1000)
    sys.stderr.write(f"  ctx={ctx}: T2={t2:.1f}ms -> {slope:.1f} tok/s\n")
    print(f"mlx\t{ctx}\t{slope:.1f}\t{t1:.3f}\t{t2:.3f}\t{runs}")
PYTHON
)

uv run python -c "$MLX_SCRIPT" "$N1" "[$(IFS=,; echo "${CONTEXTS_ARR[*]}")]" "$RUNS" 2>&1 \
    | tee >(grep "^mlx" >> "$DATA") | grep -v "^mlx"

# ─── Summary ───
echo ""
echo "═══ Results ═══"
echo ""
printf "%-10s %6s %12s\n" "engine" "ctx" "tok/s"
printf "%-10s %6s %12s\n" "------" "---" "-----"
tail -n +2 "$DATA" | while IFS=$'\t' read -r eng ctx slope rest; do
    printf "%-10s %6s %12s\n" "$eng" "$ctx" "$slope"
done

# ─── Ratios ───
echo ""
echo "═══ Lattice vs Ollama ═══"
for CTX in "${CONTEXTS_ARR[@]}"; do
    LAT=$(awk -F'\t' -v c="$CTX" '$1=="lattice" && $2==c {print $3}' "$DATA")
    OLL=$(awk -F'\t' -v c="$CTX" '$1=="ollama" && $2==c {print $3}' "$DATA")
    if [[ -n "$LAT" && -n "$OLL" ]]; then
        RATIO=$(echo "scale=2; $LAT / $OLL" | bc)
        echo "  ctx=$CTX: ${RATIO}× (${LAT} vs ${OLL} tok/s)"
    fi
done

echo ""
echo "Raw data: $DATA"

# ─── Generate chart ───
echo ""
echo "─── Generating chart ───"
uv run python "$REPO/scripts/bench_context_scaling_chart.py" "$DATA" "$CHART" 2>&1
echo "Chart: $CHART"

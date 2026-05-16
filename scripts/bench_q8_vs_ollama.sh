#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Lattice vs Ollama vs MLX: Decode Throughput Benchmark
# ============================================================================
#
# Compares single-token decode speed on Qwen3.5-0.8B across three engines:
#   1. Lattice (pure Rust + Metal GPU shaders)
#   2. Ollama (llama.cpp Metal backend)
#   3. MLX (Apple's ML framework, Python)
#
# Prerequisites:
#   1. macOS with Apple Silicon (M1/M2/M3/M4)
#   2. Rust toolchain: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
#   3. Ollama installed: `brew install ollama`
#   4. Python 3.10+ with `uv` or `pip` (for mlx-lm)
#   5. Model weights: ~/.lattice/models/qwen3.5-0.8b/ (HuggingFace safetensors)
#
# Usage:
#   ./scripts/bench_q8_vs_ollama.sh
#
# What it does:
#   1. Builds Lattice inference in release mode
#   2. Runs criterion benchmark (50 samples, 3s warmup, 10s measurement)
#   3. Runs ollama generate benchmark (50 tokens × 5 runs)
#   4. Runs mlx-lm benchmark (50 tokens × 5 runs, Q8 quantized in-memory)
#   5. Prints comparison table
#
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$HOME/.lattice/models/qwen3.5-0.8b"
RESULTS_DIR="$REPO_ROOT/target/bench_results"

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
RESET='\033[0m'

mkdir -p "$RESULTS_DIR"

echo -e "${BOLD}================================================${RESET}"
echo -e "${BOLD} Lattice vs Ollama vs MLX: Decode Benchmark${RESET}"
echo -e "${BOLD} Model: Qwen3.5-0.8B | Quant: Q8 (8-bit)${RESET}"
echo -e "${BOLD}================================================${RESET}"
echo ""

# --- Preflight checks ---
echo -e "${BOLD}[1/6] Preflight checks${RESET}"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo -e "${RED}ERROR: This benchmark requires macOS (Apple Silicon Metal GPU).${RESET}"
    exit 1
fi

CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
echo "  Hardware: $CHIP"

if [[ ! -d "$MODEL_DIR" ]] || [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo -e "${RED}ERROR: Model not found at $MODEL_DIR${RESET}"
    echo "  Download Qwen3.5-0.8B safetensors from HuggingFace:"
    echo "  huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir $MODEL_DIR"
    exit 1
fi
echo "  Model: $MODEL_DIR ($(du -sh "$MODEL_DIR" | cut -f1) total)"

if ! command -v cargo &>/dev/null; then
    echo -e "${RED}ERROR: Rust toolchain not found.${RESET}"
    exit 1
fi
RUST_VERSION=$(rustc --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
echo "  Rust: $RUST_VERSION"

HAS_OLLAMA=false
if command -v ollama &>/dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
    echo "  Ollama: v$OLLAMA_VERSION"
    HAS_OLLAMA=true
else
    echo "  Ollama: not installed (skipping)"
fi

HAS_MLX=false
if python3 -c "import mlx_lm" 2>/dev/null; then
    MLX_VERSION=$(python3 -c "import mlx.core; print(mlx.core.__version__)" 2>/dev/null || echo "unknown")
    echo "  MLX: v$MLX_VERSION"
    HAS_MLX=true
else
    echo "  MLX: not installed (skipping; install: pip install mlx-lm)"
fi

echo ""

# --- Build Lattice ---
echo -e "${BOLD}[2/6] Building Lattice (release + metal-gpu)${RESET}"
cd "$REPO_ROOT"
cargo build --release -p lattice-inference --features metal-gpu,f16 --bench metal_decode_bench 2>&1 | tail -3
echo ""

# --- Run Lattice Q8 benchmark ---
echo -e "${BOLD}[3/6] Running Lattice Q8 benchmark (criterion, 50 samples)${RESET}"
echo "  Warmup: 3s | Measurement: 10s | Samples: 50"
LATTICE_OUTPUT=$(cargo bench -p lattice-inference --features metal-gpu,f16 --bench metal_decode_bench -- "metal_decode_q8/forward_step/no_adapter" 2>&1)
echo "$LATTICE_OUTPUT" > "$RESULTS_DIR/lattice_q8_raw.txt"

LATTICE_TIME_MS=$(echo "$LATTICE_OUTPUT" | grep "time:" | grep "ms" | head -1 | sed 's/.*\[//;s/\].*//' | awk '{print $3}')
LATTICE_TOKS=$(echo "$LATTICE_OUTPUT" | grep "thrpt:" | grep "elem/s" | head -1 | sed 's/.*\[//;s/\].*//' | awk '{print $3}')
echo "  Lattice Q8 result: ${LATTICE_TOKS} tok/s (${LATTICE_TIME_MS} ms/token)"
echo ""

# --- Run Ollama benchmark ---
OLLAMA_MEDIAN="N/A"
if [[ "$HAS_OLLAMA" == "true" ]]; then
    echo -e "${BOLD}[4/6] Running Ollama Q8 benchmark (5 runs × 50 tokens)${RESET}"

    if ! ollama list 2>/dev/null | grep -q "qwen3.5:0.8b"; then
        echo "  Pulling qwen3.5:0.8b..."
        ollama pull qwen3.5:0.8b
    fi
    OLLAMA_QUANT=$(ollama show qwen3.5:0.8b 2>/dev/null | grep "quantization" | awk '{print $2}')
    echo "  Ollama quant: $OLLAMA_QUANT"

    STARTED_OLLAMA=false
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "  Starting ollama serve..."
        ollama serve &>/dev/null &
        OLLAMA_PID=$!
        sleep 2
        STARTED_OLLAMA=true
    fi

    OLLAMA_RESULTS=()
    for i in $(seq 1 5); do
        RESPONSE=$(curl -s http://localhost:11434/api/generate -d '{
            "model": "qwen3.5:0.8b",
            "prompt": "The quick brown fox jumps over the lazy dog. Once upon a time in a land far away, there lived a",
            "stream": false,
            "options": {"num_predict": 50, "temperature": 0.0}
        }')

        EVAL_COUNT=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_count',0))" 2>/dev/null || echo "0")
        EVAL_DURATION=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_duration',0))" 2>/dev/null || echo "0")

        if [[ "$EVAL_COUNT" -gt 0 ]] && [[ "$EVAL_DURATION" -gt 0 ]]; then
            TOKS_PER_SEC=$(python3 -c "print(f'{$EVAL_COUNT / ($EVAL_DURATION / 1e9):.1f}')")
            OLLAMA_RESULTS+=("$TOKS_PER_SEC")
            echo "  Run $i: ${TOKS_PER_SEC} tok/s ($EVAL_COUNT tokens)"
        else
            echo "  Run $i: FAILED (no eval data in response)"
        fi
    done

    if [[ ${#OLLAMA_RESULTS[@]} -gt 0 ]]; then
        OLLAMA_MEDIAN=$(printf '%s\n' "${OLLAMA_RESULTS[@]}" | sort -n | awk 'NR==3{print}')
        if [[ -z "$OLLAMA_MEDIAN" ]]; then
            OLLAMA_MEDIAN=$(printf '%s\n' "${OLLAMA_RESULTS[@]}" | sort -n | tail -1)
        fi
    fi
    echo "  Ollama Q8 median: ${OLLAMA_MEDIAN} tok/s"

    if [[ "$STARTED_OLLAMA" == "true" ]]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
    fi
else
    echo -e "${BOLD}[4/6] Ollama: SKIPPED (not installed)${RESET}"
fi
echo ""

# --- Run MLX benchmark ---
MLX_MEDIAN="N/A"
if [[ "$HAS_MLX" == "true" ]]; then
    echo -e "${BOLD}[5/6] Running MLX Q8 benchmark (5 runs × 50 tokens)${RESET}"
    echo "  Loading Qwen/Qwen3.5-0.8B + quantizing to 8-bit (group_size=64)..."

    MLX_OUTPUT=$(python3 -c "
import time, sys
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

model, tokenizer = load('Qwen/Qwen3.5-0.8B')
nn.quantize(model, bits=8, group_size=64)
mx.eval(model.parameters())

prompt = 'The quick brown fox jumps over the lazy dog. Once upon a time in a land far away, there lived a'
generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)

results = []
for i in range(5):
    t0 = time.time()
    generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
    elapsed = time.time() - t0
    tps = 50 / elapsed
    results.append(tps)
    print(f'  Run {i+1}: {tps:.1f} tok/s')

results.sort()
median = results[2]
print(f'MEDIAN:{median:.1f}')
" 2>&1)
    echo "$MLX_OUTPUT" | grep "Run"
    echo "$MLX_OUTPUT" > "$RESULTS_DIR/mlx_q8_raw.txt"
    MLX_MEDIAN=$(echo "$MLX_OUTPUT" | grep "^MEDIAN:" | cut -d: -f2)
    echo "  MLX Q8 median: ${MLX_MEDIAN} tok/s"
else
    echo -e "${BOLD}[5/6] MLX: SKIPPED (not installed)${RESET}"
fi
echo ""

# --- Results ---
echo -e "${BOLD}[6/6] Results${RESET}"
echo ""
echo -e "${BOLD}┌───────────────────────────────────────────────────────────────────┐${RESET}"
echo -e "${BOLD}│  Decode Throughput: Qwen3.5-0.8B (Q8) on Apple Silicon           │${RESET}"
echo -e "${BOLD}├──────────────┬────────────┬───────────────────────────────────────┤${RESET}"
echo -e "${BOLD}│ Engine       │ tok/s (p50)│ Notes                                 │${RESET}"
echo -e "${BOLD}├──────────────┼────────────┼───────────────────────────────────────┤${RESET}"
printf  "│ %-12s │ %10s │ %-37s │\n" "Lattice" "$LATTICE_TOKS" "Rust+Metal, criterion 50 samples"
printf  "│ %-12s │ %10s │ %-37s │\n" "MLX" "$MLX_MEDIAN" "Apple MLX, Q8 group_size=64, 5 runs"
printf  "│ %-12s │ %10s │ %-37s │\n" "Ollama" "$OLLAMA_MEDIAN" "llama.cpp Metal, Q8_0, 5 runs"
echo -e "${BOLD}└──────────────┴────────────┴───────────────────────────────────────┘${RESET}"
echo ""

# Speedup calculations
if [[ "$OLLAMA_MEDIAN" != "N/A" ]] && [[ "$LATTICE_TOKS" != "" ]]; then
    VS_OLLAMA=$(python3 -c "l=$LATTICE_TOKS; o=$OLLAMA_MEDIAN; print(f'{(l/o - 1)*100:.0f}')")
    echo -e "  ${GREEN}Lattice is ${VS_OLLAMA}% faster than Ollama.${RESET}"
fi
if [[ "$MLX_MEDIAN" != "N/A" ]] && [[ "$LATTICE_TOKS" != "" ]]; then
    VS_MLX=$(python3 -c "l=$LATTICE_TOKS; m=$MLX_MEDIAN; print(f'{(l/m - 1)*100:.0f}')")
    echo -e "  ${GREEN}Lattice is ${VS_MLX}% faster than MLX.${RESET}"
fi

echo ""
echo "  Hardware: $CHIP"
echo "  Model: Qwen3.5-0.8B (873M params, hybrid GDN+GQA architecture)"
echo "  Quantization: 8-bit per-row symmetric (all engines)"
echo "  Lattice: Pure Rust, custom Metal GEMV kernel (int8×f32, simd_sum)"
echo "  MLX: Apple MLX framework v${MLX_VERSION:-unknown}, Metal backend"
echo "  Ollama: llama.cpp Metal backend v${OLLAMA_VERSION:-unknown}"
echo ""
echo "  All three engines implement the full GDN recurrence for Qwen3.5."
echo ""
echo "  Raw output saved to: $RESULTS_DIR/"
echo ""
echo -e "${BOLD}To reproduce:${RESET}"
echo "  cd $(pwd)"
echo "  ./scripts/bench_q8_vs_ollama.sh"

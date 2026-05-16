#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Lattice vs Ollama: Q8_0 Decode Throughput Benchmark
# ============================================================================
#
# Compares single-token decode speed on Qwen3.5-0.8B (Q8_0 quantization)
# between Lattice (Metal GPU inference) and Ollama (llama.cpp backend).
#
# Prerequisites:
#   1. macOS with Apple Silicon (M1/M2/M3/M4)
#   2. Rust toolchain: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
#   3. Ollama installed: `brew install ollama`
#   4. Model weights: ~/.lattice/models/qwen3.5-0.8b/ (HuggingFace safetensors)
#
# Usage:
#   ./scripts/bench_q8_vs_ollama.sh
#
# What it does:
#   1. Builds Lattice inference in release mode
#   2. Runs criterion benchmark (50 samples, 3s warmup, 10s measurement)
#   3. Runs ollama generate benchmark (50 tokens × 5 runs)
#   4. Prints comparison table
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

echo -e "${BOLD}============================================${RESET}"
echo -e "${BOLD} Lattice vs Ollama: Q8_0 Decode Benchmark${RESET}"
echo -e "${BOLD}============================================${RESET}"
echo ""

# --- Preflight checks ---
echo -e "${BOLD}[1/5] Preflight checks${RESET}"

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

if ! command -v ollama &>/dev/null; then
    echo -e "${RED}ERROR: ollama not found. Install: brew install ollama${RESET}"
    exit 1
fi
OLLAMA_VERSION=$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
echo "  Ollama: v$OLLAMA_VERSION"

if ! command -v cargo &>/dev/null; then
    echo -e "${RED}ERROR: Rust toolchain not found.${RESET}"
    exit 1
fi
RUST_VERSION=$(rustc --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
echo "  Rust: $RUST_VERSION"

# Verify ollama has the model
if ! ollama list 2>/dev/null | grep -q "qwen3.5:0.8b"; then
    echo "  Pulling qwen3.5:0.8b for ollama..."
    ollama pull qwen3.5:0.8b
fi
OLLAMA_QUANT=$(ollama show qwen3.5:0.8b 2>/dev/null | grep "quantization" | awk '{print $2}')
echo "  Ollama quant: $OLLAMA_QUANT"
echo ""

# --- Build Lattice ---
echo -e "${BOLD}[2/5] Building Lattice (release + metal-gpu)${RESET}"
cd "$REPO_ROOT"
cargo build --release -p lattice-inference --features metal-gpu,f16 --bench metal_decode_bench 2>&1 | tail -3
echo ""

# --- Run Lattice Q8 benchmark ---
echo -e "${BOLD}[3/5] Running Lattice Q8 benchmark (criterion, 50 samples)${RESET}"
echo "  Warmup: 3s | Measurement: 10s | Samples: 50"
LATTICE_OUTPUT=$(cargo bench -p lattice-inference --features metal-gpu,f16 --bench metal_decode_bench -- "metal_decode_q8/forward_step/no_adapter" 2>&1)
echo "$LATTICE_OUTPUT" > "$RESULTS_DIR/lattice_q8_raw.txt"

# Parse criterion output — extract median (middle value in [low med high] triplet)
# Filter for lines with "elem/s" to avoid the percentage-change "thrpt:" line.
LATTICE_TIME_MS=$(echo "$LATTICE_OUTPUT" | grep "time:" | grep "ms" | head -1 | sed 's/.*\[//;s/\].*//' | awk '{print $3}')
LATTICE_TOKS=$(echo "$LATTICE_OUTPUT" | grep "thrpt:" | grep "elem/s" | head -1 | sed 's/.*\[//;s/\].*//' | awk '{print $3}')
echo "  Lattice Q8 result: ${LATTICE_TOKS} tok/s (${LATTICE_TIME_MS} ms/token)"
echo ""

# --- Run Ollama benchmark ---
echo -e "${BOLD}[4/5] Running Ollama Q8 benchmark (5 runs × 50 tokens)${RESET}"

# Ensure ollama is serving
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "  Starting ollama serve..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    sleep 2
    STARTED_OLLAMA=true
else
    STARTED_OLLAMA=false
fi

OLLAMA_RESULTS=()
for i in $(seq 1 5); do
    # Use /api/generate with raw mode to avoid chat template overhead.
    # num_predict=50 gives us 50 decode tokens to measure.
    RESPONSE=$(curl -s http://localhost:11434/api/generate -d '{
        "model": "qwen3.5:0.8b",
        "prompt": "The quick brown fox jumps over the lazy dog. Once upon a time in a land far away, there lived a",
        "stream": false,
        "options": {"num_predict": 50, "temperature": 0.0}
    }')

    EVAL_COUNT=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_count',0))" 2>/dev/null || echo "0")
    EVAL_DURATION=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_duration',0))" 2>/dev/null || echo "0")

    if [[ "$EVAL_COUNT" -gt 0 ]] && [[ "$EVAL_DURATION" -gt 0 ]]; then
        # eval_duration is in nanoseconds
        TOKS_PER_SEC=$(python3 -c "print(f'{$EVAL_COUNT / ($EVAL_DURATION / 1e9):.1f}')")
        OLLAMA_RESULTS+=("$TOKS_PER_SEC")
        echo "  Run $i: ${TOKS_PER_SEC} tok/s ($EVAL_COUNT tokens)"
    else
        echo "  Run $i: FAILED (no eval data in response)"
    fi
done

# Compute median of ollama results
if [[ ${#OLLAMA_RESULTS[@]} -gt 0 ]]; then
    OLLAMA_MEDIAN=$(printf '%s\n' "${OLLAMA_RESULTS[@]}" | sort -n | awk 'NR==3{print}')
    if [[ -z "$OLLAMA_MEDIAN" ]]; then
        OLLAMA_MEDIAN=$(printf '%s\n' "${OLLAMA_RESULTS[@]}" | sort -n | tail -1)
    fi
else
    OLLAMA_MEDIAN="N/A"
fi

echo "  Ollama Q8 median: ${OLLAMA_MEDIAN} tok/s"
echo ""

if [[ "$STARTED_OLLAMA" == "true" ]]; then
    kill "$OLLAMA_PID" 2>/dev/null || true
fi

# --- Results ---
echo -e "${BOLD}[5/5] Results${RESET}"
echo ""
echo -e "${BOLD}┌─────────────────────────────────────────────────────────────┐${RESET}"
echo -e "${BOLD}│  Q8_0 Decode Throughput: Qwen3.5-0.8B on Apple Silicon     │${RESET}"
echo -e "${BOLD}├──────────────┬────────────┬─────────────────────────────────┤${RESET}"
echo -e "${BOLD}│ Engine       │ tok/s (p50)│ Notes                           │${RESET}"
echo -e "${BOLD}├──────────────┼────────────┼─────────────────────────────────┤${RESET}"
printf  "│ %-12s │ %10s │ %-31s │\n" "Lattice" "$LATTICE_TOKS" "criterion 50 samples"
printf  "│ %-12s │ %10s │ %-31s │\n" "Ollama" "$OLLAMA_MEDIAN" "5 runs × 50 tokens, median"
echo -e "${BOLD}└──────────────┴────────────┴─────────────────────────────────┘${RESET}"
echo ""

if [[ "$OLLAMA_MEDIAN" != "N/A" ]] && [[ "$LATTICE_TOKS" != "" ]]; then
    SPEEDUP=$(python3 -c "l=$LATTICE_TOKS; o=$OLLAMA_MEDIAN; print(f'{(l/o - 1)*100:.0f}')")
    echo -e "  ${GREEN}Lattice is ${SPEEDUP}% faster than Ollama at Q8_0 decode.${RESET}"
fi

echo ""
echo "  Hardware: $CHIP"
echo "  Model: Qwen3.5-0.8B (873M params, Q8_0 per-row symmetric)"
echo "  Lattice: Metal GPU GEMV kernel (int8×f32, simd_sum reduction)"
echo "  Ollama: llama.cpp Metal backend v$OLLAMA_VERSION"
echo ""
echo "  Raw output saved to: $RESULTS_DIR/"
echo ""
echo -e "${BOLD}To reproduce:${RESET}"
echo "  cd $(pwd)"
echo "  ./scripts/bench_q8_vs_ollama.sh"

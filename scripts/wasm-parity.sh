#!/usr/bin/env bash
# wasm-parity.sh: build the wasm32 embedding path and gate its output against
# the same HF-reference goldens and native-lattice reference used by the
# native parity test (crates/embed/tests/embed_parity_vs_hf.rs).
#
# Usage:
#   scripts/wasm-parity.sh
#   make wasm-parity
#
# Skip-graceful: if node, wasm-bindgen, the wasm32 target, or the bge-small /
# all-MiniLM-L6-v2 model files (model.safetensors + config.json +
# tokenizer.json under ~/.lattice/models/<name>/) are missing, this prints a
# one-line reason and exits 0 so a plain workspace build is unaffected.
#
# Fail-closed: set LATTICE_WASM_PARITY_ENFORCE=1 to turn every skip above into
# a hard failure (exit 1) instead; used by the CI gate, where a missing
# prerequisite means provisioning failed, not that the check is optional.
#
# Model files: the native embed CLI's --download-only fetches
# model.safetensors and (vocab.txt or tokenizer.json, model-dependent) but
# never config.json, and for WordPiece models (bge-small, MiniLM) never
# tokenizer.json either; the wasm LatticeEmbedder constructor needs both, so
# this script fetches them directly from the Hugging Face repo the same
# --download-only step already resolved, into the same cache directory.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${LATTICE_MODELS_DIR:-$HOME/.lattice/models}"
ENFORCE="${LATTICE_WASM_PARITY_ENFORCE:-}"

skip_or_fail() {
  reason="$1"
  if [ -n "$ENFORCE" ]; then
    echo "wasm-parity: FAIL (LATTICE_WASM_PARITY_ENFORCE=1): $reason" >&2
    exit 1
  fi
  echo "wasm-parity: SKIPPED: $reason"
  exit 0
}

# --- Tool prerequisites -----------------------------------------------------

command -v node >/dev/null 2>&1 || skip_or_fail "node not found on PATH"
command -v wasm-bindgen >/dev/null 2>&1 || skip_or_fail "wasm-bindgen (CLI) not found on PATH; install with: cargo install wasm-bindgen-cli --version 0.2.105"
command -v cargo >/dev/null 2>&1 || skip_or_fail "cargo not found on PATH"

if ! rustup target list --installed 2>/dev/null | grep -q '^wasm32-unknown-unknown$'; then
  rustup target add wasm32-unknown-unknown >/dev/null 2>&1 \
    || skip_or_fail "wasm32-unknown-unknown target not installed and could not be added"
fi

# --- Model file prerequisites ------------------------------------------------
# Each entry: <lattice cache dir name>|<HF repo id>

MODEL_SPECS="bge-small-en-v1.5|BAAI/bge-small-en-v1.5
all-minilm-l6-v2|sentence-transformers/all-MiniLM-L6-v2"

echo "$MODEL_SPECS" | while IFS='|' read -r dir_name hf_id; do
  model_dir="$MODELS_DIR/$dir_name"
  if [ ! -f "$model_dir/model.safetensors" ]; then
    echo "wasm-parity: MISSING $model_dir/model.safetensors, provision with:" >&2
    echo "  cargo run --release -p lattice-embed --bin embed -- --model $dir_name --download-only" >&2
    exit 1
  fi
  for f in config.json tokenizer.json; do
    if [ ! -f "$model_dir/$f" ]; then
      echo "wasm-parity: fetching $f for $hf_id into $model_dir"
      mkdir -p "$model_dir"
      curl -fsSL -o "$model_dir/$f.tmp" "https://huggingface.co/$hf_id/resolve/main/$f"
      mv "$model_dir/$f.tmp" "$model_dir/$f"
    fi
  done
done || skip_or_fail "model weights or config/tokenizer files missing for bge-small-en-v1.5 / all-minilm-l6-v2 (see stderr above)"

for dir_name in bge-small-en-v1.5 all-minilm-l6-v2; do
  model_dir="$MODELS_DIR/$dir_name"
  for f in model.safetensors config.json tokenizer.json; do
    [ -f "$model_dir/$f" ] || skip_or_fail "$model_dir/$f still missing after provisioning attempt"
  done
done

# --- Build wasm --------------------------------------------------------------

echo "=== wasm-parity: building lattice-embed for wasm32 ==="
cargo build --release --target wasm32-unknown-unknown -p lattice-embed \
  --no-default-features --features wasm

WASM_OUT_DIR="$REPO/target/wasm-parity-out"
mkdir -p "$WASM_OUT_DIR"
wasm-bindgen --target nodejs --out-dir "$WASM_OUT_DIR" \
  "$REPO/target/wasm32-unknown-unknown/release/lattice_embed.wasm"

# --- Build native reference dump ---------------------------------------------

echo "=== wasm-parity: building native reference dump ==="
NATIVE_DUMP="$REPO/target/wasm-parity-native-dump.json"
DUMP_OUT="$NATIVE_DUMP" cargo run --release -p lattice-embed \
  --example dump_parity_embeddings

# --- Run the harness ---------------------------------------------------------

echo "=== wasm-parity: running Node harness ==="
LATTICE_WASM_JS="$WASM_OUT_DIR/lattice_embed.js" \
LATTICE_NATIVE_DUMP="$NATIVE_DUMP" \
LATTICE_MODELS_DIR="$MODELS_DIR" \
  node "$REPO/crates/embed/tests/wasm/embed_parity_wasm.mjs"

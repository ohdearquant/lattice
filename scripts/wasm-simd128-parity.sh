#!/usr/bin/env bash
# wasm-simd128-parity.sh: build lattice-embed for wasm32 twice (plain vs
# `-C target-feature=+simd128`) and gate the SIMD128 build's
# dot_product/squared_euclidean_distance/cosine_similarity/normalize output
# against the scalar reference, via
# crates/embed/tests/wasm/simd128_parity_wasm.mjs.
#
# Usage:
#   scripts/wasm-simd128-parity.sh
#
# Skip-graceful: if node, wasm-bindgen, or the wasm32 target are missing,
# prints a one-line reason and exits 0, mirroring scripts/wasm-parity.sh.
# Set LATTICE_WASM_SIMD128_ENFORCE=1 to turn a skip into a hard failure.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENFORCE="${LATTICE_WASM_SIMD128_ENFORCE:-}"

skip_or_fail() {
  reason="$1"
  if [ -n "$ENFORCE" ]; then
    echo "wasm-simd128-parity: FAIL (LATTICE_WASM_SIMD128_ENFORCE=1): $reason" >&2
    exit 1
  fi
  echo "wasm-simd128-parity: SKIPPED: $reason"
  exit 0
}

command -v node >/dev/null 2>&1 || skip_or_fail "node not found on PATH"
command -v wasm-bindgen >/dev/null 2>&1 || skip_or_fail "wasm-bindgen (CLI) not found on PATH; install with: cargo install wasm-bindgen-cli --version 0.2.105"
command -v cargo >/dev/null 2>&1 || skip_or_fail "cargo not found on PATH"

if ! rustup target list --installed 2>/dev/null | grep -q '^wasm32-unknown-unknown$'; then
  rustup target add wasm32-unknown-unknown >/dev/null 2>&1 \
    || skip_or_fail "wasm32-unknown-unknown target not installed and could not be added"
fi

BASELINE_OUT="$REPO/target/wasm-simd128-parity-baseline"
SIMD128_OUT="$REPO/target/wasm-simd128-parity-simd128"
mkdir -p "$BASELINE_OUT" "$SIMD128_OUT"

echo "=== wasm-simd128-parity: building baseline (no simd128) ==="
cargo build --release --target wasm32-unknown-unknown -p lattice-embed \
  --no-default-features --features wasm
wasm-bindgen --target nodejs --out-dir "$BASELINE_OUT" \
  "$REPO/target/wasm32-unknown-unknown/release/lattice_embed.wasm"

echo "=== wasm-simd128-parity: building simd128 (-C target-feature=+simd128) ==="
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown -p lattice-embed \
  --no-default-features --features wasm
wasm-bindgen --target nodejs --out-dir "$SIMD128_OUT" \
  "$REPO/target/wasm32-unknown-unknown/release/lattice_embed.wasm"

echo "=== wasm-simd128-parity: running Node harness ==="
LATTICE_WASM_JS_BASELINE="$BASELINE_OUT/lattice_embed.js" \
LATTICE_WASM_JS_SIMD128="$SIMD128_OUT/lattice_embed.js" \
  node "$REPO/crates/embed/tests/wasm/simd128_parity_wasm.mjs"

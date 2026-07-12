#!/usr/bin/env bash
# Builds the wasm32 embedding core and generates the JS bindings this package
# ships. Run via `npm run build` from this package directory, or directly.
#
# Requires: cargo, the wasm32-unknown-unknown target, and the wasm-bindgen
# CLI (version must match the wasm-bindgen crate version pinned in
# ../../../crates/embed/Cargo.toml). Install with:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.105
set -euo pipefail

PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"
OUT_DIR="$PKG_DIR/wasm"

command -v cargo >/dev/null 2>&1 || { echo "build-wasm: cargo not found on PATH" >&2; exit 1; }
command -v wasm-bindgen >/dev/null 2>&1 || {
  echo "build-wasm: wasm-bindgen CLI not found on PATH." >&2
  echo "  install with: cargo install wasm-bindgen-cli --version 0.2.105" >&2
  exit 1
}

if ! rustup target list --installed 2>/dev/null | grep -q '^wasm32-unknown-unknown$'; then
  rustup target add wasm32-unknown-unknown
fi

echo "=== build-wasm: compiling lattice-embed for wasm32 (SIMD128) ==="
# -C target-feature=+simd128 turns on wasm32 SIMD128 kernels in
# crates/embed/src/simd/{dot_product,distance,cosine,normalize}.rs (the
# dispatch resolvers there pick a SIMD128 kernel automatically when the
# crate is built this way; see README.md "Performance" for measured
# speedups). Runtime floor: WebAssembly SIMD128 is a baseline feature in
# every evergreen browser and in Node >=16 (this package's `engines.node`
# field already requires >=18, so no consumer of this package can be below
# the floor). There is no non-simd128 fallback artifact published from this
# flow; a caller needing a pre-SIMD128 runtime should build from source
# without this flag.
(
  cd "$REPO_ROOT"
  RUSTFLAGS="-C target-feature=+simd128" \
    cargo build --release --target wasm32-unknown-unknown -p lattice-embed \
    --no-default-features --features wasm
)

echo "=== build-wasm: generating JS bindings (--target web) ==="
mkdir -p "$OUT_DIR"
wasm-bindgen --target web --out-dir "$OUT_DIR" \
  "$REPO_ROOT/target/wasm32-unknown-unknown/release/lattice_embed.wasm"

echo "build-wasm: done. Output in $OUT_DIR"

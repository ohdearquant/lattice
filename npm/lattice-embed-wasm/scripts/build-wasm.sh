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
command -v node >/dev/null 2>&1 || {
  echo "build-wasm: node not found on PATH (needed for the post-build SIMD128 dispatch assertion)" >&2
  exit 1
}

if ! rustup target list --installed 2>/dev/null | grep -q '^wasm32-unknown-unknown$'; then
  rustup target add wasm32-unknown-unknown
fi

# CARGO_ENCODED_RUSTFLAGS takes precedence over a plain RUSTFLAGS= assignment
# on the cargo invocation below (documented cargo flag precedence: encoded
# beats plain regardless of which is "more local"), so an inherited encoded
# flag set without +simd128 would silently override this script's
# RUSTFLAGS="-C target-feature=+simd128" and produce a scalar-only artifact
# that otherwise looks like a normal successful build. There is no safe way
# to merge into an already-encoded flag set from a shell script, so fail
# loudly instead of shipping something this script can't verify.
if [ -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]; then
  echo "build-wasm: CARGO_ENCODED_RUSTFLAGS is set in the environment." >&2
  echo "  cargo gives CARGO_ENCODED_RUSTFLAGS precedence over this script's" >&2
  echo "  RUSTFLAGS=\"-C target-feature=+simd128\" assignment, so building" >&2
  echo "  with it set would silently drop the SIMD128 target-feature flag" >&2
  echo "  and ship a scalar-only artifact. Unset CARGO_ENCODED_RUSTFLAGS" >&2
  echo "  (and whatever tool or shell config set it) before running this" >&2
  echo "  script." >&2
  exit 1
fi

# One target-dir variable feeds both the cargo build below and the
# wasm-bindgen consume step further down, so they can never disagree about
# where the artifact landed. Without this, a caller with CARGO_TARGET_DIR set
# (e.g. to share a build cache across worktrees) would have cargo write the
# fresh .wasm somewhere else while wasm-bindgen kept reading the default
# target/wasm32-unknown-unknown/release/lattice_embed.wasm path -- packaging
# whatever stale artifact (or nothing at all) happened to already be sitting
# there instead of the build that just ran.
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
case "$TARGET_DIR" in
  /*) ;;
  *) TARGET_DIR="$REPO_ROOT/$TARGET_DIR" ;;
esac

echo "=== build-wasm: compiling lattice-embed for wasm32 (SIMD128) ==="
echo "    target-dir: $TARGET_DIR"
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
    --no-default-features --features wasm \
    --target-dir "$TARGET_DIR"
)

WASM_ARTIFACT="$TARGET_DIR/wasm32-unknown-unknown/release/lattice_embed.wasm"
if [ ! -f "$WASM_ARTIFACT" ]; then
  echo "build-wasm: expected build artifact missing at $WASM_ARTIFACT" >&2
  exit 1
fi

echo "=== build-wasm: generating JS bindings (--target web) ==="
mkdir -p "$OUT_DIR"
wasm-bindgen --target web --out-dir "$OUT_DIR" "$WASM_ARTIFACT"

echo "=== build-wasm: verifying SIMD128 dispatch in the generated bindings ==="
# Loads the bindings just written to $OUT_DIR and asserts the built .wasm is
# actually dispatching to the SIMD128 kernels (simdSimd128Dispatch(), the
# same probe crates/embed/tests/wasm/simd128_parity_wasm.mjs checks before
# trusting any numeric comparison) rather than silently serving scalar
# because the target-feature flag above didn't take effect on the consumed
# artifact. Fails the build rather than shipping an artifact this check
# can't confirm is SIMD128.
node "$PKG_DIR/scripts/assert-simd128-dispatch.mjs" "$OUT_DIR"

echo "build-wasm: done. Output in $OUT_DIR"

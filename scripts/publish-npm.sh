#!/bin/sh
set -e

# Publish the two npm embedding packages under the @khive-ai scope:
#   @khive-ai/lattice-embed-wasm   portable pure-wasm channel (all platforms)
#   @khive-ai/lattice-embed        native napi channel + per-platform binaries
#
# Mirrors scripts/publish.sh (crates.io) in style: --dry-run support, ordered
# tiers, fail-closed on a missing artifact.
#
# Provenance: each package's publishConfig sets access=public. Sigstore
# provenance requires a supported CI with OIDC (e.g. GitHub Actions), so it is
# NOT baked into publishConfig (that would hard-fail a local publish). This
# script adds --provenance only when running under CI; a local/manual publish
# runs without it.

DRY_RUN=${1:-""}

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "=== Dry Run: npm packages ==="
    FLAG="--dry-run"
else
    echo "=== Publishing to npm ==="
    FLAG=""
fi

PROV=""
if [ -n "$CI" ]; then
    PROV="--provenance"
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WASM_DIR="$ROOT/npm/lattice-embed-wasm"
NATIVE_DIR="$ROOT/npm/lattice-embed-native"

# ---- @khive-ai/lattice-embed-wasm: portable, single artifact, all platforms.
# `prepack` rebuilds wasm/lattice_embed_bg.wasm from source before packing.
echo "--- @khive-ai/lattice-embed-wasm (portable wasm) ---"
( cd "$WASM_DIR" && npm publish $FLAG $PROV )

# ---- @khive-ai/lattice-embed: napi per-platform binaries.
# Gather any locally built .node into npm/<platform>/, then publish every
# platform subpackage that has a binary BEFORE the main package (which lists
# them in optionalDependencies). npm skips optional deps that were never
# published; at runtime the binding reports FL_EMBED_UNSUPPORTED_PLATFORM on a
# platform whose binary was not shipped, and the portable wasm package covers
# those platforms. Cross-platform binaries come from the napi build matrix in
# CI; a local run ships only the current platform.
echo "--- @khive-ai/lattice-embed native binaries ---"
( cd "$NATIVE_DIR" && npm run artifacts >/dev/null 2>&1 || true )

PUBLISHED_PLATFORMS=""
MISSING_PLATFORMS=""
for pkgdir in "$NATIVE_DIR"/npm/*/; do
    [ -d "$pkgdir" ] || continue
    platform=$(basename "$pkgdir")
    if ls "$pkgdir"*.node >/dev/null 2>&1; then
        echo "  publishing @khive-ai/lattice-embed-$platform"
        ( cd "$pkgdir" && npm publish $FLAG $PROV )
        PUBLISHED_PLATFORMS="$PUBLISHED_PLATFORMS $platform"
    else
        MISSING_PLATFORMS="$MISSING_PLATFORMS $platform"
    fi
done

if [ -z "$PUBLISHED_PLATFORMS" ]; then
    echo "ERROR: no native platform binary found under $NATIVE_DIR/npm/*/." >&2
    echo "Build the current platform first: (cd $NATIVE_DIR && npm run build)." >&2
    echo "Cross-platform binaries are produced by the napi build matrix in CI." >&2
    exit 1
fi

if [ -n "$MISSING_PLATFORMS" ]; then
    echo "NOTE: no native binary shipped for:$MISSING_PLATFORMS"
    echo "      users there get FL_EMBED_UNSUPPORTED_PLATFORM; @khive-ai/lattice-embed-wasm covers them."
fi

if [ -z "$DRY_RUN" ]; then
    echo "Waiting for npm indexing before the main package resolves its optional deps..."
    sleep 30
fi

echo "--- @khive-ai/lattice-embed (native main) ---"
( cd "$NATIVE_DIR" && npm publish $FLAG $PROV )

echo "=== Done ==="

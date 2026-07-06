#!/bin/sh
set -e

# Publish the two npm embedding packages under the @khive-ai scope:
#   @khive-ai/lattice-embed-wasm   portable pure-wasm channel (all platforms)
#   @khive-ai/lattice-embed        native napi channel + per-platform binaries
#
# Mirrors scripts/publish.sh (crates.io) in style: --dry-run support, ordered
# tiers, fail-closed on a missing artifact.
#
# Immutability: npm name@version tuples cannot be republished. A real publish
# that fails partway therefore leaves an unrepublishable partial release, so
# this script dry-runs the WHOLE release first and only starts real publishes
# once every package packs and every gate (native prepublishOnly test) passes.
# A mid-flight network failure during the real pass still requires a manual
# version bump — npm has no atomic multi-package publish — but every
# deterministic failure is caught before the first upload.
#
# Provenance: each package's publishConfig sets access=public. Sigstore
# provenance requires an OIDC-capable publish environment (e.g. GitHub Actions
# with `id-token: write`), which a generic `$CI` does not guarantee, so it is
# gated behind an explicit NPM_PROVENANCE=1 opt-in rather than baked into
# publishConfig (which would hard-fail a local publish).

case "${1:-}" in
    "")
        MODE="publish"
        ;;
    --dry-run)
        MODE="dry-run"
        ;;
    *)
        echo "usage: $0 [--dry-run]" >&2
        exit 2
        ;;
esac

PROV=""
if [ "${NPM_PROVENANCE:-}" = "1" ]; then
    PROV="--provenance"
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WASM_DIR="$ROOT/npm/lattice-embed-wasm"
NATIVE_DIR="$ROOT/npm/lattice-embed-native"

# Gather any locally built .node into npm/<platform>/ so the platform
# subpackages have their binary. Cross-platform binaries come from the napi
# build matrix in CI; a local run only produces the current platform.
( cd "$NATIVE_DIR" && npm run artifacts >/dev/null 2>&1 || true )

# Discover the platform subpackages that actually carry a binary.
PLATFORM_DIRS=""
for pkgdir in "$NATIVE_DIR"/npm/*/; do
    [ -d "$pkgdir" ] || continue
    if ls "$pkgdir"*.node >/dev/null 2>&1; then
        PLATFORM_DIRS="$PLATFORM_DIRS $pkgdir"
    fi
done

if [ -z "$PLATFORM_DIRS" ]; then
    echo "ERROR: no native platform binary found under $NATIVE_DIR/npm/*/." >&2
    echo "Build the current platform first: (cd $NATIVE_DIR && npm run build)." >&2
    echo "Cross-platform binaries are produced by the napi build matrix in CI." >&2
    exit 1
fi

# ---- Preflight: dry-run the ENTIRE release before any real publish. This
# packs wasm (prepack rebuild), every present platform subpackage, and the
# native main package (its prepublishOnly runs `napi artifacts && npm test`).
# Any failure here aborts before a single package is published.
echo "=== Preflight (dry-run: pack + gate the full release) ==="
( cd "$WASM_DIR" && npm publish --dry-run )
for pkgdir in $PLATFORM_DIRS; do
    ( cd "$pkgdir" && npm publish --dry-run )
done
( cd "$NATIVE_DIR" && npm publish --dry-run )
echo "Preflight OK."

if [ "$MODE" = "dry-run" ]; then
    echo "=== Dry run complete: nothing published ==="
    exit 0
fi

# ---- Real publish: platform subpackages BEFORE the main package that lists
# them in optionalDependencies, then the portable wasm package.
echo "=== Publishing to npm ==="
for pkgdir in $PLATFORM_DIRS; do
    platform=$(basename "$pkgdir")
    echo "--- @khive-ai/lattice-embed-$platform ---"
    ( cd "$pkgdir" && npm publish $PROV )
done

echo "Waiting for npm indexing before the main package resolves its optional deps..."
sleep 30

echo "--- @khive-ai/lattice-embed (native main) ---"
( cd "$NATIVE_DIR" && npm publish $PROV )

echo "--- @khive-ai/lattice-embed-wasm (portable wasm) ---"
( cd "$WASM_DIR" && npm publish $PROV )

echo
echo "NOTE: on a platform whose native binary was not published, npm resolves"
echo "      that optional dependency as absent and require('@khive-ai/lattice-embed')"
echo "      throws FL_EMBED_NATIVE_LOAD_FAILED (the native package does NOT fall"
echo "      back to wasm on its own). Consumers wanting portable coverage install"
echo "      @khive-ai/lattice-embed-wasm, which runs everywhere."

echo "=== Done ==="

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
# build matrix in CI (npm-prebuild.yml's `publish` job downloads the
# collector-validated `npm-native-prebuilds` artifact before this script
# runs), so this is a no-op in that path and only fills in the current
# platform for a local run.
( cd "$NATIVE_DIR" && npm run artifacts >/dev/null 2>&1 || true )

# Require every platform advertised in the main package's
# optionalDependencies to carry its native binary -- not any nonempty subset.
# The real publish path only ever runs in CI against the full napi build
# matrix; a partial local checkout must fail closed here rather than
# silently release a package whose optionalDependencies point at platforms
# nobody published this round.
EXPECTED_PLATFORMS=$(node -p "Object.keys(require('$NATIVE_DIR/package.json').optionalDependencies).map(n => n.replace('@khive-ai/lattice-embed-', '')).join(' ')")

PLATFORM_DIRS=""
for platform in $EXPECTED_PLATFORMS; do
    pkgdir="$NATIVE_DIR/npm/$platform/"
    if [ ! -d "$pkgdir" ] || ! ls "$pkgdir"*.node >/dev/null 2>&1; then
        echo "ERROR: missing native binary for platform '$platform' under $pkgdir" >&2
        echo "       the release must include every platform listed in" >&2
        echo "       $NATIVE_DIR/package.json optionalDependencies, not a subset." >&2
        echo "       Cross-platform binaries come from the napi build matrix in CI;" >&2
        echo "       run this from npm-prebuild.yml's publish job, which downloads" >&2
        echo "       the npm-native-prebuilds artifact first." >&2
        exit 1
    fi
    PLATFORM_DIRS="$PLATFORM_DIRS $pkgdir"
done

# ---- Preflight: verify none of the packages we're about to publish already
# exist on npm. name@version tuples are immutable, so a collision discovered
# mid-publish (after some platform packages already went out) is unrecoverable
# without a version bump. Fail closed here instead.
echo "=== Preflight (version-exists check against the npm registry) ==="
check_version_available() {
    pkgdir="$1"
    name=$(node -p "require('$pkgdir/package.json').name")
    version=$(node -p "require('$pkgdir/package.json').version")
    view_out=$(npm view "$name@$version" version --json 2>/dev/null)
    view_status=$?
    if [ "$view_status" -eq 0 ]; then
        echo "ERROR: $name@$version is already published on npm and cannot be republished." >&2
        echo "       npm name@version tuples are immutable. Bump the version in" >&2
        echo "       $pkgdir/package.json (and every sibling package that must move in" >&2
        echo "       lockstep) before publishing -- this is a coordinated release decision," >&2
        echo "       not something this script infers automatically." >&2
        exit 1
    fi
    # A nonzero exit means either "not found" (the only case that establishes
    # availability) or a lookup failure (DNS/TLS, registry 5xx, auth) that tells
    # us nothing about whether the version is free. `npm view --json` puts the
    # error body on stdout as {"error":{"code":...}}; only E404 counts as
    # not-found (verified against npm 11.8.0's actual output for a missing
    # package -- see fix_r4_report.md). Anything else must fail the release
    # closed rather than be silently treated as "available".
    error_code=$(printf '%s' "$view_out" | node -e "
        let input = '';
        process.stdin.on('data', c => input += c);
        process.stdin.on('end', () => {
            try {
                console.log(JSON.parse(input).error.code || '');
            } catch {
                console.log('');
            }
        });
    " 2>/dev/null)
    if [ "$error_code" != "E404" ]; then
        echo "ERROR: preflight lookup for $name@$version failed and did not return npm's" >&2
        echo "       not-found response (E404), so availability could not be established." >&2
        echo "       This can be a DNS/TLS failure, a registry 5xx, or an auth problem --" >&2
        echo "       none of which mean the version is free to publish. Raw npm view output:" >&2
        echo "$view_out" >&2
        exit 1
    fi
}
check_version_available "$WASM_DIR"
for pkgdir in $PLATFORM_DIRS; do
    check_version_available "$pkgdir"
done
check_version_available "$NATIVE_DIR"
echo "Preflight OK: no version collisions."

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

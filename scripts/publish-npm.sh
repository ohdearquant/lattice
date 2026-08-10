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
# version bump -- npm has no atomic multi-package publish -- but every
# deterministic failure is caught before the first upload.
#
# Provenance: each package's publishConfig sets access=public. Sigstore
# provenance requires an OIDC-capable publish environment (e.g. GitHub Actions
# with `id-token: write`), which a generic `$CI` does not guarantee, so it is
# gated behind an explicit NPM_PROVENANCE=1 opt-in rather than baked into
# publishConfig (which would hard-fail a local publish).
#
# This file is sourceable as a function library: set PUBLISH_NPM_SH_LIB_ONLY
# to any non-empty value before `. scripts/publish-npm.sh` and every guard
# below is defined as a callable function with no side effects -- main() is
# defined but never invoked. Unset (the normal `sh scripts/publish-npm.sh`
# invocation), the script runs exactly as before. Tests exercise each guard
# by sourcing under the sentinel and calling the function directly, instead
# of cutting a text fragment out of the live source.

# Guard: reject any argv other than "" or "--dry-run". Reads $1 explicitly
# (not the top-level positional parameters) so it can be called standalone
# with an arbitrary value in tests.
parse_argv() {
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
}

# Guard: require the platform package.json for $1 (named $3) to exist under
# $2 before anything downstream unconditionally reads it. Split out of
# platform_matrix_guard's loop into its own function -- it is a single,
# self-contained precondition check (no dependency on matrix-guard-local
# state beyond the three values passed in), and isolating it this way means
# a test can call it directly without also running the require() calls that
# follow it in the loop, rather than having to cut a text fragment that stops
# short of them.
check_platform_pkgjson() {  # $1=platform $2=pkgdir $3=pkgjson
    platform="$1"
    pkgdir="$2"
    pkgjson="$3"
    if [ ! -f "$pkgjson" ]; then
        echo "ERROR: missing native binary for platform '$platform' under $pkgdir" >&2
        echo "       the release must include every platform listed in" >&2
        echo "       $NATIVE_DIR/package.json optionalDependencies, not a subset." >&2
        echo "       Cross-platform binaries come from the napi build matrix in CI;" >&2
        echo "       run this from npm-prebuild.yml's publish job, which downloads" >&2
        echo "       the npm-native-prebuilds artifact first." >&2
        exit 1
    fi
}

# Guard: require every platform advertised in the main package's
# optionalDependencies to carry its native binary -- not any nonempty subset,
# and not a zero-length set either (an empty optionalDependencies object is
# valid JSON and iterates zero times, which would otherwise publish the wasm
# and native-main packages with no platform binaries backing them at all).
# The real publish path only ever runs in CI against the full napi build
# matrix; a partial or empty local checkout must fail closed here rather
# than silently release a package whose optionalDependencies point at
# platforms nobody published this round.
#
# Per platform, require the EXACT .node file that platform package's own
# "main" field names -- not any *.node match, so a stale or misnamed binary
# left over from a previous build fails closed instead of passing an
# existence glob -- and require that package's declared name and version to
# agree with the platform key and the main package's version, AND require
# the main package's own optionalDependencies entry for this platform to be
# pinned at that exact version too (not a range, wildcard, or "latest"),
# since nothing else here reads what the main package advertises to
# installers. Finally replay the same packlist content guard CI's package
# job runs (assert-platform-packlist.mjs) directly here, so a bare `make
# publish-npm` -- which never touches npm-prebuild.yml -- gets it too
# instead of skipping straight to a real publish untested.
#
# Sets PLATFORM_DIRS (space-separated directories, each backing one
# validated platform package) as a plain global, consumed by main() after
# this function returns -- no `local` anywhere in this function, so that
# assignment (and every other one below) keeps exactly the set -e semantics
# it had at top level: a bare `VAR=$(cmd)` is a simple command whose own
# exit status is the command substitution's, and triggers errexit on
# failure. Combining `local` with a command-substitution assignment on one
# line (`local VAR=$(cmd)`) would instead report the exit status of the
# `local` builtin itself -- which succeeds even when `cmd` failed -- and
# silently defeat that propagation. This function relies on it for every
# `node -p` lookup below, none of which has an explicit `if`/`exit` guard of
# its own.
platform_matrix_guard() {
    EXPECTED_PLATFORMS=$(node -p "Object.keys(require('$NATIVE_DIR/package.json').optionalDependencies).map(n => n.replace('@khive-ai/lattice-embed-', '')).join(' ')")

    if [ -z "$EXPECTED_PLATFORMS" ]; then
        echo "ERROR: $NATIVE_DIR/package.json optionalDependencies is empty -- refusing to" >&2
        echo "       publish a native release with zero platform binaries. A real release" >&2
        echo "       always lists every supported napi target there; an empty set means the" >&2
        echo "       checkout or the metadata itself is broken, not that this round has no" >&2
        echo "       platform binaries to ship." >&2
        exit 1
    fi

    NATIVE_VERSION=$(node -p "require('$NATIVE_DIR/package.json').version")

    PLATFORM_DIRS=""
    for platform in $EXPECTED_PLATFORMS; do
        pkgdir="$NATIVE_DIR/npm/$platform/"
        pkgjson="${pkgdir}package.json"
        check_platform_pkgjson "$platform" "$pkgdir" "$pkgjson"

        node_rel=$(node -p "require('$pkgjson').main || ''")
        node_path="${pkgdir}${node_rel}"
        if [ -z "$node_rel" ] || [ ! -f "$node_path" ]; then
            echo "ERROR: missing native binary for platform '$platform': expected $node_path" >&2
            echo "       (the exact filename $pkgjson's \"main\" field names), not any *.node match." >&2
            echo "       Cross-platform binaries come from the napi build matrix in CI;" >&2
            echo "       run this from npm-prebuild.yml's publish job, which downloads" >&2
            echo "       the npm-native-prebuilds artifact first." >&2
            exit 1
        fi

        plat_name=$(node -p "require('$pkgjson').name")
        expected_name="@khive-ai/lattice-embed-$platform"
        if [ "$plat_name" != "$expected_name" ]; then
            echo "ERROR: $pkgjson has name '$plat_name', expected '$expected_name' for" >&2
            echo "       platform '$platform'." >&2
            exit 1
        fi

        plat_version=$(node -p "require('$pkgjson').version")
        if [ "$plat_version" != "$NATIVE_VERSION" ]; then
            echo "ERROR: $pkgjson is at version $plat_version, but $NATIVE_DIR/package.json" >&2
            echo "       is at $NATIVE_VERSION. Platform packages must move in lockstep with" >&2
            echo "       the main native package." >&2
            exit 1
        fi

        # The two checks above only read the platform package's OWN declared
        # name/version -- neither reads what the main package's own
        # optionalDependencies actually advertises for this platform. Those can
        # disagree independently of everything checked so far: the main package
        # could point installers at a different version (or a non-exact range)
        # of this platform package than the one just validated, which every
        # local-directory check above is blind to. Look the dependency value up
        # by $expected_name (the canonical "@khive-ai/lattice-embed-$platform"
        # key optionalDependencies is always keyed by), not by $plat_name -- a
        # platform package whose own "name" field is wrong is already caught by
        # the guard above and must not also determine which optionalDependencies
        # key this check reads, or a defeated name guard would silently borrow
        # this one's failure and this arm would stop isolating its own guard.
        # Compare literal strings rather than semver-parsing the dependency
        # value -- an exact string match against NATIVE_VERSION is
        # simultaneously the "pinned to this release" check and a rejection of
        # every non-exact form (^1.2.3, ~1.2.3, *, latest, a "x.y.z - a.b.c" or
        # ">=x <y" range expression, or an empty/missing value), none of which
        # can literal-equal a bare version string.
        dep_version=$(node -p "
        const deps = require('$NATIVE_DIR/package.json').optionalDependencies || {};
        deps['$expected_name'] || ''
    ")
        if [ "$dep_version" != "$NATIVE_VERSION" ]; then
            echo "ERROR: $NATIVE_DIR/package.json optionalDependencies declares" >&2
            echo "       '$expected_name' at '$dep_version', but the exact version is" >&2
            echo "       '$NATIVE_VERSION'. The main package's optionalDependencies must pin" >&2
            echo "       each platform package to that exact version -- not a range, wildcard," >&2
            echo "       or 'latest' -- so installers cannot resolve a differently-versioned" >&2
            echo "       platform binary than the one validated here." >&2
            exit 1
        fi

        # The lookup's nonzero status must not trigger errexit here either --
        # land it in an `if` condition, same reasoning as check_version_available
        # below.
        if ! pack_json=$(cd "$pkgdir" && npm pack --dry-run --json 2>/dev/null); then
            echo "ERROR: npm pack --dry-run failed for platform '$platform' under $pkgdir" >&2
            exit 1
        fi
        if ! printf '%s' "$pack_json" | node "$NATIVE_DIR/scripts/assert-platform-packlist.mjs" >/dev/null; then
            echo "ERROR: platform packlist guard failed for '$platform' under $pkgdir -- the" >&2
            echo "       packed tarball did not contain exactly package.json + one .node file" >&2
            echo "       (+ optional README). Re-run 'cd $pkgdir && npm pack --dry-run --json |" >&2
            echo "       node $NATIVE_DIR/scripts/assert-platform-packlist.mjs' to see the" >&2
            echo "       assertion detail." >&2
            exit 1
        fi

        PLATFORM_DIRS="$PLATFORM_DIRS $pkgdir"
    done
}

# Guard: verify none of the packages we're about to publish already exist on
# npm. name@version tuples are immutable, so a collision discovered
# mid-publish (after some platform packages already went out) is unrecoverable
# without a version bump. Fail closed here instead.
check_version_available() {
    pkgdir="$1"
    name=$(node -p "require('$pkgdir/package.json').name")
    version=$(node -p "require('$pkgdir/package.json').version")
    # The lookup's nonzero status must not trigger errexit -- it is not an
    # error, it's the "not found" signal this function exists to interpret.
    # A standalone `view_out=$(...)` assignment IS itself a failed command
    # under set -e, so the failure has to land inside an `if` condition
    # instead, where errexit does not apply.
    if view_out=$(npm view "$name@$version" version --json 2>/dev/null); then
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
    # not-found -- verified directly against npm 11.8.0's actual output for a
    # missing package: `npm view <missing-name>@<version> version --json`
    # prints {"error":{"code":"E404","summary":"..."}} to stdout and exits
    # nonzero. Anything else must fail the release closed rather than be
    # silently treated as "available".
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

# Call check_version_available on every package about to be published: the
# portable wasm package, every validated platform subpackage, then the
# native main package. Reads WASM_DIR/PLATFORM_DIRS/NATIVE_DIR as globals so
# a test can stub check_version_available and assert this calls it on the
# right targets in the right order without also running platform_matrix_guard
# first.
run_version_checks() {
    check_version_available "$WASM_DIR"
    for pkgdir in $PLATFORM_DIRS; do
        check_version_available "$pkgdir"
    done
    check_version_available "$NATIVE_DIR"
}

# Guard: dry-run the ENTIRE release before any real publish. This packs wasm
# (prepack rebuild), every present platform subpackage, and the native main
# package (its prepublishOnly runs `napi artifacts && npm test`). Any
# failure here aborts before a single package is published.
#
# No explicit `exit N` here -- this relies entirely on the script's own
# top-level `set -e` to abort on the first nonzero `npm publish --dry-run`.
# That still holds correctly once this is a function: each `( cd ... && npm
# publish --dry-run )` is called as a bare statement (not inside `if`/`&&`),
# both here and at every call site in main(), so a nonzero exit from any of
# them triggers the same errexit abort it would have at top level.
full_dryrun_guard() {
    ( cd "$WASM_DIR" && npm publish --dry-run )
    for pkgdir in $PLATFORM_DIRS; do
        ( cd "$pkgdir" && npm publish --dry-run )
    done
    ( cd "$NATIVE_DIR" && npm publish --dry-run )
}

# Guard: `npm publish --dry-run` above packs the main package but never
# examines its contents -- it only proves the tarball builds, not that the
# tarball's contents are correct. assert-packlist.mjs is the purpose-built
# content guard for this package, in the same shape the platform packages
# already get from assert-platform-packlist.mjs in npm-prebuild.yml's
# package job.
#
# This calls `npm pack --dry-run --json` directly and pipes its captured
# output to the assertion, the same way platform_matrix_guard's packlist
# check above does -- it does NOT run `npm run packlist` (the package.json
# script that wraps the identical pipeline). A shell pipeline's exit status
# is its last command's by default, so `npm pack --dry-run --json | node
# scripts/assert-packlist.mjs` only reports node's exit code: a failing
# `npm pack` that still emits a well-formed manifest on stdout before dying
# lets the assertion parse that manifest, exit 0, and the pipeline reports
# success. Capturing $pack_json in an `if VAR=$(...)` first, as done here,
# makes the command substitution's own exit status observable and land in
# an `if`, without errexit firing on it -- same technique
# check_version_available uses above, and the same reason
# platform_matrix_guard's own npm-pack call already does this rather than
# piping. Fixing the package.json script instead (e.g. adding
# `set -o pipefail` to the "packlist" script body) was rejected: npm invokes
# "scripts" entries through the platform's default shell, which is not
# guaranteed to support `set -o pipefail` (dash does not), so that fix would
# be silently inert on some platforms; capturing and checking status here
# needs no shell feature beyond command substitution, which every #!/bin/sh
# implementation already provides. `npm run packlist` has no other caller in
# this repo (grepped `.github/workflows/*.yml`, `Makefile`, and `scripts/`),
# so bypassing it here does not change behavior anywhere else.
main_packlist_guard() {
    if ! pack_json=$(cd "$NATIVE_DIR" && npm pack --dry-run --json 2>/dev/null); then
        echo "ERROR: npm pack --dry-run failed for the native main package under $NATIVE_DIR" >&2
        exit 1
    fi
    if ! printf '%s' "$pack_json" | node "$NATIVE_DIR/scripts/assert-packlist.mjs" >/dev/null; then
        echo "ERROR: main packlist guard failed for $NATIVE_DIR -- the packed tarball did" >&2
        echo "       not satisfy assert-packlist.mjs's required-file or forbidden-pattern" >&2
        echo "       checks. Re-run 'cd $NATIVE_DIR && npm pack --dry-run --json | node" >&2
        echo "       scripts/assert-packlist.mjs' to see the assertion detail." >&2
        exit 1
    fi
}

main() {
    parse_argv "$1"

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

    platform_matrix_guard

    # ---- Preflight: verify none of the packages we're about to publish
    # already exist on npm.
    echo "=== Preflight (version-exists check against the npm registry) ==="
    run_version_checks
    echo "Preflight OK: no version collisions."

    echo "=== Preflight (dry-run: pack + gate the full release) ==="
    full_dryrun_guard

    main_packlist_guard
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
}

if [ -z "${PUBLISH_NPM_SH_LIB_ONLY:-}" ]; then
    main "$@"
fi

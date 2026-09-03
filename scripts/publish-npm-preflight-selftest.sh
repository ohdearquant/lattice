#!/usr/bin/env bash
# Self-test for the guard functions in scripts/publish-npm.sh.
#
# publish-npm.sh is sourceable as a function library: set
# PUBLISH_NPM_SH_LIB_ONLY to any non-empty value before `. scripts/publish-npm.sh`
# and every guard below is defined as a callable function with no side
# effects (main() is defined but never invoked). This harness sources the
# real script under that sentinel and calls the real functions directly --
# it no longer cuts a text fragment out of the live source with awk and a
# BEGIN/END marker comment pair the way earlier rounds of this file did.
# That extraction mechanism went through two hardening passes (an
# exactly-one-BEGIN/END count, a foreign-marker check, a non-dry-run `npm
# publish` scan) and was still defeated by a marker pair placed inside a
# here-document, a longer identifier that merely CONTAINS a marker name
# (`grep -c` counts substring hits), an END placed before its BEGIN, two
# BEGINs on one physical line, and several textual disguises of a real `npm
# publish` call. Calling the actual function the production path calls
# cannot be fooled by any of those, because none of the concepts an
# extractor has to reason about (a range, a boundary, a foreign marker, a
# disguised call) exist once the guard is just a named block of code
# invoked directly. See the marker-removal self-test at the bottom, which
# proves the retired mechanism is gone rather than merely unused.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/scripts/publish-npm.sh"

SB="$(mktemp -d)"
trap 'rm -rf "$SB"' EXIT

pass=0; fail=0
check() {  # $1=desc $2=expected_exit $3=actual_exit
  if [ "$2" = "$3" ]; then
    echo "  PASS: $1 (exit $3)"; pass=$((pass+1))
  else
    echo "  FAIL: $1 -- expected exit $2 got $3"
    echo "        output: $(printf '%s' "$OUT" | tr '\n' '|')"
    fail=$((fail+1))
  fi
}

echo "=== publish-npm.sh preflight self-test (7 stubbed npm dispositions) ==="

# A minimal package dir check_version_available()'s `node -p require(...)`
# calls can read.
PKGDIR="$SB/pkg"
mkdir -p "$PKGDIR"
cat > "$PKGDIR/package.json" <<'EOF'
{"name": "@khive-ai/lattice-fixture", "version": "9.9.9"}
EOF

STUBDIR="$SB/stub-bin"
mkdir -p "$STUBDIR"

run_case() {  # $1 = stub npm script body
  cat > "$STUBDIR/npm" <<EOF
#!/usr/bin/env bash
$1
EOF
  chmod +x "$STUBDIR/npm"
  RUNNER="$SB/runner.sh"
  {
    echo 'PUBLISH_NPM_SH_LIB_ONLY=1'
    printf '. %q\n' "$SRC"
    printf 'check_version_available %q\n' "$PKGDIR"
  } > "$RUNNER"
  # /bin/sh with set -e: the production shell and options publish-npm.sh
  # actually runs under (it has `#!/bin/sh` and `set -e` at its own top).
  OUT="$(PATH="$STUBDIR:$PATH" /bin/sh -c "set -e; . '$RUNNER'" 2>&1)"
  return $?
}

# (a) npm's real not-found response: E404 JSON on stdout, nonzero exit.
run_case 'cat <<J
{"error":{"code":"E404","summary":"Not Found - GET https://registry.npmjs.org/x - Not found"}}
J
exit 1'
check "(a) not-found (E404) is treated as available" 0 $?
case "$OUT" in
  *"ERROR:"*) echo "  FAIL:   -> unexpected error output on the available path"; fail=$((fail+1)) ;;
  *) echo "  PASS:   -> no error diagnostic printed"; pass=$((pass+1)) ;;
esac

# (b) a network/registry failure: nonzero exit, no E404 body (e.g. a raw
#     curl-style failure or a 5xx with a different code).
run_case 'echo "npm error code ETIMEDOUT" >&2
echo "npm error network timeout contacting registry" >&2
exit 1'
check "(b) network/registry error fails closed (nonzero)" 1 $?
if printf '%s' "$OUT" | grep -qF "did not return npm"; then
  echo "  PASS:   -> diagnostic distinguishes lookup failure from availability"; pass=$((pass+1))
else
  echo "  FAIL:   -> no distinguishing diagnostic in output"; fail=$((fail+1))
fi

# (c) a successful lookup: version exists, exit 0 -> already-published branch
#     (G10, the `if view_out=$(npm view ...)` success arm). The stub body is
#     valid JSON whose parsed error.code is E404 -- an otherwise-nonsensical
#     pairing (a "successful" view call carrying a not-found body) chosen
#     deliberately: if G10's own exit 1 is defeated, execution falls through
#     unconditionally into the E404-parsing branch below it, which reads
#     error_code == "E404" as the AVAILABLE outcome and does not itself
#     reject, so a defeated G10 makes this arm go fully green (rc 0, no
#     diagnostic) instead of holding rc 1 from the E404 guard's own
#     rejection.
run_case 'cat <<J
{"error":{"code":"E404","summary":"Not Found - GET https://registry.npmjs.org/x - Not found"}}
J
exit 0'
check "(c) successful lookup is treated as already-published" 1 $?
case "$OUT" in
  *"already published on npm"*) echo "  PASS:   -> already-published diagnostic printed"; pass=$((pass+1)) ;;
  *) echo "  FAIL:   -> missing already-published diagnostic"; fail=$((fail+1)) ;;
esac

# (d) empty body: nonzero exit, nothing on stdout at all.
run_case 'exit 1'
check "(d) empty body fails closed (nonzero)" 1 $?
if printf '%s' "$OUT" | grep -qF "did not return npm"; then
  echo "  PASS:   -> diagnostic distinguishes lookup failure from availability"; pass=$((pass+1))
else
  echo "  FAIL:   -> no distinguishing diagnostic in output"; fail=$((fail+1))
fi

# (e) invalid JSON body: nonzero exit, unparseable stdout.
run_case 'echo "not json at all {"
exit 1'
check "(e) invalid JSON fails closed (nonzero)" 1 $?
if printf '%s' "$OUT" | grep -qF "did not return npm"; then
  echo "  PASS:   -> diagnostic distinguishes lookup failure from availability"; pass=$((pass+1))
else
  echo "  FAIL:   -> no distinguishing diagnostic in output"; fail=$((fail+1))
fi

# (f) valid JSON but no "error" key: nonzero exit, well-formed but unrelated body.
run_case 'echo "{\"ok\":false}"
exit 1'
check "(f) missing error key fails closed (nonzero)" 1 $?
if printf '%s' "$OUT" | grep -qF "did not return npm"; then
  echo "  PASS:   -> diagnostic distinguishes lookup failure from availability"; pass=$((pass+1))
else
  echo "  FAIL:   -> no distinguishing diagnostic in output"; fail=$((fail+1))
fi

# (g) "error" key present but no "error.code": nonzero exit, ambiguous error shape.
run_case 'echo "{\"error\":{\"summary\":\"something broke\"}}"
exit 1'
check "(g) missing error.code fails closed (nonzero)" 1 $?
if printf '%s' "$OUT" | grep -qF "did not return npm"; then
  echo "  PASS:   -> diagnostic distinguishes lookup failure from availability"; pass=$((pass+1))
else
  echo "  FAIL:   -> no distinguishing diagnostic in output"; fail=$((fail+1))
fi

echo
echo "=== publish-npm.sh platform matrix guard self-test ==="
# The matrix guard (empty-set check, exact .node path, name/version
# agreement, packlist content) is a real function (platform_matrix_guard) in
# the sourced script now -- call it directly against a fixture NATIVE_DIR
# tree. Every command it runs (`node -p`, `npm pack --dry-run --json`) is
# local/offline -- no registry access, so no npm stubbing is needed here the
# way check_version_available's `npm` had to be stubbed.

REAL_PACKLIST_ASSERT="$REPO/npm/lattice-embed-native/scripts/assert-platform-packlist.mjs"
if [ ! -f "$REAL_PACKLIST_ASSERT" ]; then
  echo "FATAL: $REAL_PACKLIST_ASSERT not found -- has it moved?" >&2
  exit 1
fi

# main_packlist_guard (unlike the old npm-run-based version) makes a real
# `node scripts/assert-packlist.mjs` call -- not a call the generic e2e npm
# stubs' `run) exit 0 ;;` case can absorb -- so every build_native_fixture
# fixture needs this file present too, not just the platform-matrix ones
# that only ever call platform_matrix_guard.
REAL_MAIN_PACKLIST_ASSERT="$REPO/npm/lattice-embed-native/scripts/assert-packlist.mjs"
if [ ! -f "$REAL_MAIN_PACKLIST_ASSERT" ]; then
  echo "FATAL: $REAL_MAIN_PACKLIST_ASSERT not found -- has it moved?" >&2
  exit 1
fi

MSB="$SB/matrix"
mkdir -p "$MSB"

# platform_matrix_guard's own `npm pack --dry-run --json` call (see
# run_matrix_case below) is real, unstubbed npm -- give it a cache this
# self-test owns outright, under its own sandbox, so the case never depends
# on who owns (or whether anyone can write to) the ambient ~/.npm cache.
MATRIX_NPM_CACHE="$MSB/npm-cache"
mkdir -p "$MATRIX_NPM_CACHE"

# Build a fixture NATIVE_DIR ($1) whose optionalDependencies match the
# platform/version pairs given as "$2..." (each "platform:version"). Copies
# the real assert-platform-packlist.mjs and assert-packlist.mjs alongside it
# so the packlist guard steps exercise the actual production checks, not a
# stand-in.
build_native_fixture() {
  native="$1"; shift
  mkdir -p "$native/scripts"
  cp "$REAL_PACKLIST_ASSERT" "$native/scripts/assert-platform-packlist.mjs"
  cp "$REAL_MAIN_PACKLIST_ASSERT" "$native/scripts/assert-packlist.mjs"
  deps=""
  sep=""
  for pv in "$@"; do
    p="${pv%%:*}"; v="${pv#*:}"
    deps="${deps}${sep}\"@khive-ai/lattice-embed-$p\": \"$v\""
    sep=", "
  done
  cat > "$native/package.json" <<EOF
{"name": "@khive-ai/lattice-embed", "version": "1.2.3", "optionalDependencies": {$deps}}
EOF
}

# Write a well-formed platform subpackage: correct name, correct main-named
# .node file, requested version.
add_valid_platform() {  # $1=native $2=platform $3=version
  native="$1"; platform="$2"; version="$3"
  d="$native/npm/$platform"
  mkdir -p "$d"
  main="lattice-embed-native.$platform.node"
  cat > "$d/package.json" <<EOF
{"name": "@khive-ai/lattice-embed-$platform", "version": "$version", "main": "$main", "files": ["$main"]}
EOF
  printf 'fake-binary' > "$d/$main"
}

run_matrix_case() {  # $1 = fixture NATIVE_DIR
  RUNNER="$MSB/runner.sh"
  {
    echo 'PUBLISH_NPM_SH_LIB_ONLY=1'
    printf 'export NPM_CONFIG_CACHE=%q\n' "$MATRIX_NPM_CACHE"
    printf '. %q\n' "$SRC"
    printf 'NATIVE_DIR=%q\n' "$1"
    printf 'platform_matrix_guard\n'
    printf 'echo MATRIX_GUARD_PASSED\n'
  } > "$RUNNER"
  OUT="$(/bin/sh -c "set -e; . '$RUNNER'" 2>&1)"
  return $?
}

# (h) empty optionalDependencies -- must fail closed (Defect A).
NATIVE="$MSB/h-empty"; build_native_fixture "$NATIVE"
run_matrix_case "$NATIVE"; rc=$?
check "(h) empty optionalDependencies fails closed" 1 $rc
if printf '%s' "$OUT" | grep -qF "optionalDependencies is empty"; then
  echo "  PASS:   -> empty-matrix diagnostic printed"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing empty-matrix diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

# (i) platform .node present but misnamed -- must fail closed (Defect B,
#     exact-path guard rather than a *.node glob). "files" lists the
#     wrong-name file too (not the name "main" declares), so the packlist
#     guard (G4) still sees exactly one .node file and would NOT catch this
#     on its own -- this arm has to isolate G2 specifically, or a G2 removal
#     could hide behind G4 and this arm would falsely report the guard as
#     load-bearing when it isn't the one doing the rejecting.
NATIVE="$MSB/i-misnamed"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
d="$NATIVE/npm/darwin-arm64"; mkdir -p "$d"
cat > "$d/package.json" <<'EOF'
{"name": "@khive-ai/lattice-embed-darwin-arm64", "version": "1.2.3", "main": "lattice-embed-native.darwin-arm64.node", "files": ["totally-wrong-name.node"]}
EOF
printf 'fake-binary' > "$d/totally-wrong-name.node"
run_matrix_case "$NATIVE"; rc=$?
check "(i) misnamed .node fails closed" 1 $rc
if printf '%s' "$OUT" | grep -qF "expected"; then
  echo "  PASS:   -> exact-path diagnostic printed"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing exact-path diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

# (j) platform .node absent under the exact name "main" declares -- must fail
#     closed via the exact-path guard (G4, the `[ ! -f "$node_path" ]` check).
#     The fixture ships a DIFFERENT .node file (not the main-named one),
#     present on disk and listed in "files", so that once G4 is defeated the
#     packlist guard (G9) accepts the tarball and this arm goes fully green
#     instead of holding rc 1 from G9's own rejection.
NATIVE="$MSB/j-absent"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
d="$NATIVE/npm/darwin-arm64"; mkdir -p "$d"
present="lattice-embed-native-alt.darwin-arm64.node"
cat > "$d/package.json" <<EOF
{"name": "@khive-ai/lattice-embed-darwin-arm64", "version": "1.2.3", "main": "lattice-embed-native.darwin-arm64.node", "files": ["$present"]}
EOF
printf 'fake-binary' > "$d/$present"
run_matrix_case "$NATIVE"; rc=$?
check "(j) absent .node fails closed" 1 $rc
if printf '%s' "$OUT" | grep -qF "missing native binary"; then
  echo "  PASS:   -> missing-binary diagnostic printed"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing missing-binary diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

# (k) platform package version disagrees with the main package -- must fail
#     closed. optionalDependencies is pinned at the CORRECT version (1.2.3,
#     matching NATIVE_VERSION) so only the platform package's own version
#     (9.9.9) is wrong -- otherwise this fixture also trips the
#     optionalDependencies-value guard (same wrong value, same NATIVE_VERSION
#     comparison), and the two guards stop isolating.
NATIVE="$MSB/k-version"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
add_valid_platform "$NATIVE" "darwin-arm64" "9.9.9"
run_matrix_case "$NATIVE"; rc=$?
check "(k) version disagreement fails closed" 1 $rc
if printf '%s' "$OUT" | grep -qF "Platform packages must move in lockstep"; then
  echo "  PASS:   -> version-lockstep diagnostic printed"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing version-lockstep diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

# (l) platform ships two .node files -- passes the exact-path check (the
#     correctly-named one exists) but must fail the packlist content guard.
NATIVE="$MSB/l-twonode"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
d="$NATIVE/npm/darwin-arm64"; mkdir -p "$d"
main="lattice-embed-native.darwin-arm64.node"
extra="lattice-embed-native.other-arch.node"
cat > "$d/package.json" <<EOF
{"name": "@khive-ai/lattice-embed-darwin-arm64", "version": "1.2.3", "main": "$main", "files": ["$main", "$extra"]}
EOF
printf 'fake-binary' > "$d/$main"
printf 'fake-binary' > "$d/$extra"
run_matrix_case "$NATIVE"; rc=$?
check "(l) two .node files fails closed" 1 $rc
if printf '%s' "$OUT" | grep -qF "packlist guard failed"; then
  echo "  PASS:   -> packlist-guard diagnostic printed"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing packlist-guard diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

# (m) a full, legitimate two-platform matrix -- must pass every guard. This
# fixture also doubles as the must-PASS control for the
# optionalDependencies-value guard below.
NATIVE="$MSB/m-valid"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3" "linux-x64-gnu:1.2.3"
add_valid_platform "$NATIVE" "darwin-arm64" "1.2.3"
add_valid_platform "$NATIVE" "linux-x64-gnu" "1.2.3"
run_matrix_case "$NATIVE"; rc=$?
check "(m) full valid matrix passes" 0 $rc
if printf '%s' "$OUT" | grep -qF "MATRIX_GUARD_PASSED"; then
  echo "  PASS:   -> guard reached the end of platform_matrix_guard"; pass=$((pass+1))
else
  echo "  FAIL:   -> did not reach end of function (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

# (n) a platform package.json whose "name" field disagrees with its
#     directory/optionalDependencies key -- passes the exact-.node-path check
#     (right file, right place) so this arm isolates the name guard
#     specifically; the misnamed-.node arm (i) isolates the exact-path guard,
#     not this one.
add_misnamed_platform() {  # $1=native $2=platform $3=version $4=wrong_name
  native="$1"; platform="$2"; version="$3"; wrong_name="$4"
  d="$native/npm/$platform"
  mkdir -p "$d"
  main="lattice-embed-native.$platform.node"
  cat > "$d/package.json" <<EOF
{"name": "$wrong_name", "version": "$version", "main": "$main", "files": ["$main"]}
EOF
  printf 'fake-binary' > "$d/$main"
}
NATIVE="$MSB/n-wrongname"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
add_misnamed_platform "$NATIVE" "darwin-arm64" "1.2.3" "@khive-ai/lattice-embed-totally-wrong"
run_matrix_case "$NATIVE"; rc=$?
check "(n) platform package.json name mismatch fails closed" 1 $rc
if printf '%s' "$OUT" | grep -qF "has name '@khive-ai/lattice-embed-totally-wrong'"; then
  echo "  PASS:   -> name-mismatch diagnostic printed"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing name-mismatch diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

echo
echo "=== publish-npm.sh optionalDependencies value guard self-test ==="
# build_native_fixture's "platform:version" pairs set the optionalDependencies
# VALUE independently of add_valid_platform's own version arg, so pinning
# add_valid_platform at "1.2.3" (== NATIVE_VERSION, always "1.2.3" per
# build_native_fixture) while varying only the optionalDependencies pair
# isolates this guard from the pre-existing exact-path/name/version checks,
# which all read the platform package's own files, never
# optionalDependencies' value.
depvalue_case() {  # $1=label $2=bad_dep_value $3=must_fail_msg_substring
  label="$1"; bad="$2"; needle="$3"
  NATIVE="$MSB/depvalue-$(printf '%s' "$label" | tr -c 'a-zA-Z0-9' '-')"
  build_native_fixture "$NATIVE" "darwin-arm64:$bad"
  add_valid_platform "$NATIVE" "darwin-arm64" "1.2.3"
  run_matrix_case "$NATIVE"; rc=$?
  check "$label fails closed" 1 $rc
  if printf '%s' "$OUT" | grep -qF "$needle"; then
    echo "  PASS:   -> optionalDependencies-value diagnostic printed"; pass=$((pass+1))
  else
    echo "  FAIL:   -> missing optionalDependencies-value diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
  fi
}
# (o) the originally measured defect: dependency value at a stale/different
#     exact version than every platform package actually publishes at.
depvalue_case "(o) optionalDependencies mismatched exact version" "9.9.9" "must pin"
# (p) empty string value. Also stands in for a genuinely absent key: this
#     guard's lookup is `deps['<name>'] || ''`, so a present-but-empty value
#     and a missing key produce the identical dep_version='' and take the
#     identical branch -- true key-absence can't be isolated as a SEPARATE
#     arm here because EXPECTED_PLATFORMS is itself derived from
#     Object.keys(optionalDependencies) inside platform_matrix_guard, so a
#     platform this loop ever reaches by construction always has a key -- an
#     absent key never enters the loop at all and is instead the
#     empty-optionalDependencies shape arm (h) covers, or (for one absent key
#     among several present) the "missing native binary" shape arms (j)
#     already cover, since a key-less platform has no
#     $NATIVE_DIR/npm/<platform>/ directory backing it.
depvalue_case "(p) optionalDependencies empty value" "" "must pin"
# (q) caret range.
depvalue_case "(q) optionalDependencies caret range" "^1.2.3" "must pin"
# (r) tilde range.
depvalue_case "(r) optionalDependencies tilde range" "~1.2.3" "must pin"
# (s) wildcard.
depvalue_case "(s) optionalDependencies wildcard" "*" "must pin"
# (t) "latest" tag.
depvalue_case "(t) optionalDependencies latest tag" "latest" "must pin"
# (u) explicit range expression (comparator range, not just a bare operator).
depvalue_case "(u) optionalDependencies range expression" ">=1.2.3 <2.0.0" "must pin"

echo
echo "=== publish-npm.sh main-package packlist guard self-test ==="
# main_packlist_guard is a real function now -- call it directly against a
# minimal npm project fixture satisfying assert-packlist.mjs's
# required-files list, with an optional extra path appended to "files" to
# trip a forbidden pattern.
REAL_MAIN_PACKLIST_ASSERT="$REPO/npm/lattice-embed-native/scripts/assert-packlist.mjs"
if [ ! -f "$REAL_MAIN_PACKLIST_ASSERT" ]; then
  echo "FATAL: $REAL_MAIN_PACKLIST_ASSERT not found -- has it moved?" >&2
  exit 1
fi

MPB="$SB/main-packlist"
mkdir -p "$MPB"

# main_packlist_guard's own `npm pack --dry-run --json` call (see
# run_main_packlist_case below) is real, unstubbed npm too -- same reasoning
# as MATRIX_NPM_CACHE above: a per-run cache this self-test owns, not the
# ambient ~/.npm.
MAIN_PACKLIST_NPM_CACHE="$MPB/npm-cache"
mkdir -p "$MAIN_PACKLIST_NPM_CACHE"

# Build a fixture NATIVE_DIR ($1) with the minimal file set assert-packlist.mjs
# requires (package.json, README.md, binding.js, index.js, index.d.ts), plus
# a "scripts.packlist" entry matching the real package.json's, plus a copy of
# the real assert-packlist.mjs. Extra args ($2...) are additional paths added
# to "files" and created on disk, to trip the forbidden-pattern check.
build_main_fixture() {
  native="$1"; shift
  mkdir -p "$native/scripts"
  cp "$REAL_MAIN_PACKLIST_ASSERT" "$native/scripts/assert-packlist.mjs"
  printf '# fixture readme\n' > "$native/README.md"
  printf 'module.exports = {}\n' > "$native/binding.js"
  printf 'module.exports = {}\n' > "$native/index.js"
  printf 'export {}\n' > "$native/index.d.ts"
  files_json='"README.md", "binding.js", "index.js", "index.d.ts", "package.json"'
  for extra in "$@"; do
    files_json="${files_json}, \"$extra\""
    extra_dir=$(dirname "$extra")
    [ "$extra_dir" != "." ] && mkdir -p "$native/$extra_dir"
    printf 'extra fixture content\n' > "$native/$extra"
  done
  cat > "$native/package.json" <<EOF
{"name": "@khive-ai/lattice-fixture-main", "version": "1.2.3", "files": [$files_json], "scripts": {"packlist": "npm pack --dry-run --json | node scripts/assert-packlist.mjs"}}
EOF
}

run_main_packlist_case() {  # $1 = fixture NATIVE_DIR
  RUNNER="$MPB/runner.sh"
  {
    echo 'PUBLISH_NPM_SH_LIB_ONLY=1'
    printf 'export NPM_CONFIG_CACHE=%q\n' "$MAIN_PACKLIST_NPM_CACHE"
    printf '. %q\n' "$SRC"
    printf 'NATIVE_DIR=%q\n' "$1"
    printf 'main_packlist_guard\n'
    printf 'echo MAIN_PACKLIST_GUARD_PASSED\n'
  } > "$RUNNER"
  OUT="$(/bin/sh -c "set -e; . '$RUNNER'" 2>&1)"
  return $?
}

# (v) a valid main-package fixture -- must pass the packlist call.
NATIVE="$MPB/v-valid"; build_main_fixture "$NATIVE"
run_main_packlist_case "$NATIVE"; rc=$?
check "(v) main package packlist valid fixture passes" 0 $rc
if printf '%s' "$OUT" | grep -qF "MAIN_PACKLIST_GUARD_PASSED"; then
  echo "  PASS:   -> guard reached the end of main_packlist_guard"; pass=$((pass+1))
else
  echo "  FAIL:   -> did not reach end of function (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

# (w) a forbidden Rust source file included in the tarball -- must fail
#     closed via assert-packlist.mjs's forbidden-pattern check (^src/).
NATIVE="$MPB/w-forbidden"; build_main_fixture "$NATIVE" "src/lib.rs"
run_main_packlist_case "$NATIVE"; rc=$?
check "(w) main package packlist forbidden file fails closed" 1 $rc
if printf '%s' "$OUT" | grep -qF "must not include src/lib.rs"; then
  echo "  PASS:   -> forbidden-pattern diagnostic printed"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing forbidden-pattern diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

echo
echo "=== publish-npm.sh missing-platform-package.json guard self-test ==="
# check_platform_pkgjson is its own function now, factored out of
# platform_matrix_guard's loop specifically so it can be called in isolation
# -- it no longer needs a bounded text extraction to avoid the unconditional
# `node -p require($pkgjson)...` call that follows it in the loop, because
# that call now lives in the CALLER (platform_matrix_guard), not in this
# function. A defeated check_platform_pkgjson (its own `if [ ! -f ... ]`
# neutralized) now simply returns 0 with no crash, so this arm's exit-code
# assertion alone is an unambiguous isolation of this guard -- the earlier
# round's coincidental-rc-match risk (a defeated guard's `exit 1` and a
# downstream crash's `exit 1` producing the same rc for different reasons)
# cannot arise here any more.
run_pkgjson_case() {  # $1=NATIVE_DIR $2=platform
  pkgdir="$1/npm/$2/"
  pkgjson="${pkgdir}package.json"
  RUNNER="$MSB/pkgjson-runner.sh"
  {
    echo 'PUBLISH_NPM_SH_LIB_ONLY=1'
    printf '. %q\n' "$SRC"
    printf 'NATIVE_DIR=%q\n' "$1"
    printf 'check_platform_pkgjson %q %q %q\n' "$2" "$pkgdir" "$pkgjson"
    printf 'echo PKGJSON_GUARD_PASSED\n'
  } > "$RUNNER"
  OUT="$(/bin/sh -c "set -e; . '$RUNNER'" 2>&1)"
  return $?
}

NATIVE="$MSB/x-nopkgjson"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
# Deliberately create no npm/darwin-arm64/ directory at all, so pkgjson can
# never exist.
run_pkgjson_case "$NATIVE" "darwin-arm64"; rc=$?
check "(x) missing platform package.json fails closed" 1 $rc
if printf '%s' "$OUT" | grep -qF "missing native binary for platform 'darwin-arm64' under"; then
  echo "  PASS:   -> missing-package.json diagnostic printed"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing distinguishing diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi
# (x2) must-PASS control: a present package.json must NOT trip this guard --
# without this, (x) alone cannot tell "the guard correctly rejects absence"
# apart from "the fixture always rejects, regardless of input".
NATIVE="$MSB/x2-present"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
add_valid_platform "$NATIVE" "darwin-arm64" "1.2.3"
run_pkgjson_case "$NATIVE" "darwin-arm64"; rc=$?
check "(x2) present platform package.json passes" 0 $rc
if printf '%s' "$OUT" | grep -qF "PKGJSON_GUARD_PASSED"; then
  echo "  PASS:   -> guard reached the end of check_platform_pkgjson"; pass=$((pass+1))
else
  echo "  FAIL:   -> did not reach end of function (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

echo
echo "=== publish-npm.sh platform_binaries_present main-field validation self-test ==="
# platform_binaries_present funnels its per-platform .node resolution
# through platform_node_rel -- the same function platform_matrix_guard's
# exact-path guard uses -- so hardening that one function against an
# absent/empty "main", a path-separator or ".." value, or a non-".node"
# extension protects both callers. These arms exercise
# platform_binaries_present directly, mirroring the JS-side
# platformBinariesPresent coverage in
# npm/lattice-embed-native/__test__/guard-artifacts.spec.mjs.
run_binaries_present_case() {  # $1=NATIVE_DIR
  RUNNER="$MSB/binpresent-runner.sh"
  {
    echo 'PUBLISH_NPM_SH_LIB_ONLY=1'
    printf '. %q\n' "$SRC"
    printf 'NATIVE_DIR=%q\n' "$1"
    printf 'platform_binaries_present\n'
    printf 'echo BINARIES_PRESENT_TRUE\n'
  } > "$RUNNER"
  OUT="$(/bin/sh -c "set -e; . '$RUNNER'" 2>&1)"
  return $?
}

# (bp1) must-PASS control: a fully valid single-platform matrix returns true.
NATIVE="$MSB/bp1-binpresent-valid"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
add_valid_platform "$NATIVE" "darwin-arm64" "1.2.3"
run_binaries_present_case "$NATIVE"; rc=$?
check "(bp1) platform_binaries_present valid matrix returns true" 0 $rc
if printf '%s' "$OUT" | grep -qF "BINARIES_PRESENT_TRUE"; then
  echo "  PASS:   -> function returned success"; pass=$((pass+1))
else
  echo "  FAIL:   -> did not reach success marker (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

# (bp2) empty "main" returns false.
NATIVE="$MSB/bp2-binpresent-emptymain"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
d="$NATIVE/npm/darwin-arm64"; mkdir -p "$d"
cat > "$d/package.json" <<'EOF'
{"name": "@khive-ai/lattice-embed-darwin-arm64", "version": "1.2.3", "main": ""}
EOF
run_binaries_present_case "$NATIVE"; rc=$?
check "(bp2) platform_binaries_present empty main returns false" 1 $rc

# (bp3) a "main" that escapes the platform directory via "..". The escaped
# file is created and DOES exist on disk, so a defeated validation rule
# would make this arm go green for the wrong reason (a missing-file
# coincidence) instead of holding rc 1 from the rejected-format check.
NATIVE="$MSB/bp3-binpresent-evil"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
d="$NATIVE/npm/darwin-arm64"; mkdir -p "$d"
cat > "$d/package.json" <<'EOF'
{"name": "@khive-ai/lattice-embed-darwin-arm64", "version": "1.2.3", "main": "../evil.js"}
EOF
printf 'arbitrary file escaping the platform directory\n' > "$NATIVE/npm/evil.js"
run_binaries_present_case "$NATIVE"; rc=$?
check "(bp3) platform_binaries_present path-traversal main returns false" 1 $rc

# (bp4) a "main" with no separator and no ".." but the wrong extension --
# also present on disk, isolating the extension check from the two above.
NATIVE="$MSB/bp4-binpresent-wrongext"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
d="$NATIVE/npm/darwin-arm64"; mkdir -p "$d"
cat > "$d/package.json" <<'EOF'
{"name": "@khive-ai/lattice-embed-darwin-arm64", "version": "1.2.3", "main": "README.md"}
EOF
printf '# not a binary\n' > "$d/README.md"
run_binaries_present_case "$NATIVE"; rc=$?
check "(bp4) platform_binaries_present non-.node main returns false" 1 $rc

echo
echo "=== publish-npm.sh npm-pack-command-failure guard self-test ==="
# publish-npm.sh's platform_matrix_guard fails closed when `npm pack
# --dry-run --json` itself returns nonzero -- distinct from the packlist
# CONTENT guard (arm (l) above), which requires a pack that succeeds but
# whose contents are wrong. Isolate by shimming a stub `npm` ahead of the
# real one on PATH that fails only on `npm pack`; nothing else in
# platform_matrix_guard invokes `npm` (only `node -p` and this one pack
# call), so the stub cannot accidentally mask a different guard.
#
# The stub ALSO prints a well-formed pack manifest on stdout before exiting
# nonzero, so that a defeated failure-check would let the downstream
# packlist-content guard accept the tarball too and reach
# MATRIX_GUARD_PASSED with rc=0 -- this arm's rc assertion alone, not just
# its message assertion, is what catches the defeat.
YSB="$MSB/y-packfail"
mkdir -p "$YSB/stub-bin"
cat > "$YSB/stub-bin/npm" <<'NPMSTUB'
#!/usr/bin/env bash
if [ "$1" = "pack" ]; then
  echo '[{"files":[{"path":"package.json"},{"path":"lattice-embed-native.darwin-arm64.node"}]}]'
  echo "npm error simulated pack failure for selftest arm (y)" >&2
  exit 1
fi
echo "unexpected npm invocation in arm (y) stub: $*" >&2
exit 99
NPMSTUB
chmod +x "$YSB/stub-bin/npm"

NATIVE="$YSB/native"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
add_valid_platform "$NATIVE" "darwin-arm64" "1.2.3"
RUNNER="$MSB/runner.sh"
{
  echo 'PUBLISH_NPM_SH_LIB_ONLY=1'
  printf '. %q\n' "$SRC"
  printf 'NATIVE_DIR=%q\n' "$NATIVE"
  printf 'platform_matrix_guard\n'
  printf 'echo MATRIX_GUARD_PASSED\n'
} > "$RUNNER"
OUT="$(PATH="$YSB/stub-bin:$PATH" /bin/sh -c "set -e; . '$RUNNER'" 2>&1)"
rc=$?
check "(y) npm pack --dry-run command failure fails closed" 1 $rc
if printf '%s' "$OUT" | grep -qF "npm pack --dry-run failed for platform"; then
  echo "  PASS:   -> pack-command-failure diagnostic printed"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing pack-command-failure diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

echo
echo "=== publish-npm.sh check_version_available call-site coverage self-test ==="
# check_version_available()'s own logic (E404 vs ambiguous vs
# already-published) is exercised above by arms (a)-(g) against one
# synthetic fixture, but nothing above proves the real script actually
# CALLS it on the WASM package, every platform package, AND the native main
# package. Call the real run_version_checks() function with
# check_version_available REDEFINED (after sourcing) to a recording stub --
# shell resolves a function call by name at call time, so the redefinition
# is what run_version_checks actually invokes, and no marker/range anchoring
# is needed to isolate the call sites: they ARE the function body now.
ZSB="$SB/z-callsites"
mkdir -p "$ZSB"
CVA_LOG="$ZSB/calls.log"
rm -f "$CVA_LOG"
RUNNER="$ZSB/runner.sh"
{
  echo 'PUBLISH_NPM_SH_LIB_ONLY=1'
  printf '. %q\n' "$SRC"
  printf 'check_version_available() { printf "%%s\\n" "$1" >> %q; }\n' "$CVA_LOG"
  printf 'WASM_DIR=%q\n' "wasm-dir-marker"
  printf 'NATIVE_DIR=%q\n' "native-dir-marker"
  printf 'PLATFORM_DIRS=%q\n' "plat-a-marker plat-b-marker"
  printf 'run_version_checks\n'
} > "$RUNNER"
OUT="$(/bin/sh -c "set -e; . '$RUNNER'" 2>&1)"
rc=$?
check "(z) call-site harness runs cleanly" 0 $rc
EXPECTED_CALLS="$(printf 'wasm-dir-marker\nplat-a-marker\nplat-b-marker\nnative-dir-marker\n')"
ACTUAL_CALLS="$(cat "$CVA_LOG" 2>/dev/null)"
if [ "$ACTUAL_CALLS" = "$EXPECTED_CALLS" ]; then
  echo "  PASS: (z) check_version_available called on WASM, every platform, and native, in order"; pass=$((pass+1))
else
  echo "  FAIL: (z) call-site set/order wrong -- expected:"
  printf '%s\n' "$EXPECTED_CALLS" | sed 's/^/        /'
  echo "        got:"
  printf '%s\n' "$ACTUAL_CALLS" | sed 's/^/        /'
  fail=$((fail+1))
fi

echo
echo "=== publish-npm.sh full-release dry-run guard self-test ==="
# full_dryrun_guard packs and gates the WHOLE release (wasm, every platform,
# native main) via three `npm publish --dry-run` calls before any real
# publish -- the immutability contract this script exists to protect (npm
# name@version tuples cannot be republished, so a real publish that fails
# partway is unrecoverable). This guard has no explicit `exit N`: it relies
# entirely on the script's own top-level `set -e` to abort on the first
# nonzero `npm publish --dry-run`. Converting it into a function does not
# change that: it is called as a bare statement (not inside `if`/`&&`) both
# here and in main(), and none of its own statements use `local`, so a
# failing subshell inside it still triggers errexit exactly as it did at
# top level.
DSB="$SB/dryrun"
mkdir -p "$DSB/wasm" "$DSB/native" "$DSB/platform-a" "$DSB/stub-bin"

# The stub keys its failure on $PWD (which package's dry-run is running),
# not on the mere fact that `npm publish` was invoked. An unconditional
# "always fail" stub cannot isolate the WASM call specifically: mutating the
# WASM call away still leaves the NATIVE_DIR call later in the same guard,
# which an unconditional stub would ALSO fail -- reddening the arm for the
# wrong reason and masking the very mutation it exists to catch.
#
# The stub also requires the EXACT argv "publish --dry-run" (both the
# command word AND the flag, and nothing beyond them) -- not just "$1 is
# publish and $2 is --dry-run", which would silently accept e.g. an extra
# trailing flag no caller intends. $#=2 is checked alongside $2 so a
# same-length-but-wrong-flag argv and a right-flag-but-extra-args argv are
# both rejected the same way a real accidental non-dry-run publish is.
run_dryrun_case() {  # $1=fail_dir (absolute dir the stub should fail in, or ""
                      # for none)  $2=PLATFORM_DIRS (space-separated, or ""
                      # for none, the default)
  fail_dir="$1"
  platform_dirs="${2:-}"
  : > "$DSB/calls.log"
  cat > "$DSB/stub-bin/npm" <<EOF
#!/usr/bin/env bash
echo "\$*" >> "$DSB/calls.log"
if [ "\$1" = "publish" ]; then
  if [ "\$#" -ne 2 ] || [ "\$2" != "--dry-run" ]; then
    echo "UNEXPECTED PUBLISH ARGV in full-dryrun-guard stub: npm \$* in \$PWD (expected exactly: publish --dry-run)" >&2
    exit 98
  fi
  if [ "\$PWD" = "$fail_dir" ]; then
    echo "npm error simulated dry-run failure for selftest arm in \$PWD" >&2
    exit 1
  fi
  exit 0
fi
echo "unexpected npm invocation in dry-run guard stub: \$*" >&2
exit 99
EOF
  chmod +x "$DSB/stub-bin/npm"
  RUNNER="$DSB/runner.sh"
  {
    echo 'PUBLISH_NPM_SH_LIB_ONLY=1'
    printf '. %q\n' "$SRC"
    printf 'WASM_DIR=%q\n' "$DSB/wasm"
    printf 'PLATFORM_DIRS=%q\n' "$platform_dirs"
    printf 'NATIVE_DIR=%q\n' "$DSB/native"
    printf 'full_dryrun_guard\n'
    printf 'echo FULL_DRYRUN_GUARD_PASSED\n'
  } > "$RUNNER"
  OUT="$(PATH="$DSB/stub-bin:$PATH" /bin/sh -c "set -e; . '$RUNNER'" 2>&1)"
  return $?
}

publish_dryrun_call_count() {  # counts exact "publish --dry-run" lines logged by the last run_dryrun_case
  cat "$DSB/calls.log" 2>/dev/null | grep -cxF -- "publish --dry-run"
}

# (bb) the wasm package's dry-run fails specifically (stub fails only when
#      $PWD is $WASM_DIR) -- must abort before reaching the native
#      package's own dry-run or the PASSED marker.
run_dryrun_case "$DSB/wasm" ""
rc=$?
check "(bb) wasm dry-run failure aborts the release" 1 $rc
if printf '%s' "$OUT" | grep -qF "FULL_DRYRUN_GUARD_PASSED"; then
  echo "  FAIL:   -> guard reached the end of the function despite the failure"; fail=$((fail+1))
else
  echo "  PASS:   -> guard aborted before reaching the end of the function"; pass=$((pass+1))
fi

# (bb2) a platform package's dry-run fails specifically -- full_dryrun_guard
#       runs wasm, then every PLATFORM_DIRS entry, then native, so this must
#       abort after wasm's own (successful) dry-run but before native's ever
#       runs. Neither (bb) (PLATFORM_DIRS empty) nor (cc) below exercises a
#       PLATFORM_DIRS failure at all, so a broken platform-loop failure path
#       would previously have gone completely uncaught.
run_dryrun_case "$DSB/platform-a" "$DSB/platform-a"
rc=$?
check "(bb2) platform dry-run failure aborts the release" 1 $rc
if printf '%s' "$OUT" | grep -qF "FULL_DRYRUN_GUARD_PASSED"; then
  echo "  FAIL:   -> guard reached the end of the function despite the failure"; fail=$((fail+1))
else
  echo "  PASS:   -> guard aborted before reaching the end of the function"; pass=$((pass+1))
fi
call_count=$(publish_dryrun_call_count)
if [ "$call_count" -eq 2 ]; then
  echo "  PASS:   -> exactly wasm + platform-a were attempted (native's dry-run never ran)"; pass=$((pass+1))
else
  echo "  FAIL:   -> expected exactly 2 publish --dry-run calls (wasm, platform-a), got $call_count"; fail=$((fail+1))
fi

# (bb3) the native package's dry-run fails specifically -- the LAST call in
#       full_dryrun_guard's sequence. Runs with a nonempty PLATFORM_DIRS too,
#       so a pass here also confirms wasm and the platform package were each
#       attempted (and succeeded) before the native failure aborted the
#       function -- an exit-code check alone cannot tell "aborted correctly
#       at the last call" apart from "the fixture never ran the earlier
#       calls at all".
run_dryrun_case "$DSB/native" "$DSB/platform-a"
rc=$?
check "(bb3) native dry-run failure aborts the release" 1 $rc
if printf '%s' "$OUT" | grep -qF "FULL_DRYRUN_GUARD_PASSED"; then
  echo "  FAIL:   -> guard reached the end of the function despite the failure"; fail=$((fail+1))
else
  echo "  PASS:   -> guard aborted before reaching the end of the function"; pass=$((pass+1))
fi
call_count=$(publish_dryrun_call_count)
if [ "$call_count" -eq 3 ]; then
  echo "  PASS:   -> wasm, platform-a, and native were all attempted before the abort"; pass=$((pass+1))
else
  echo "  FAIL:   -> expected exactly 3 publish --dry-run calls (wasm, platform-a, native), got $call_count"; fail=$((fail+1))
fi

# (cc) must-PASS control: every dry-run succeeds (no fail_dir), with a
#      nonempty PLATFORM_DIRS too -- the guard must reach the end of the
#      function. Without this, (bb)/(bb2)/(bb3) alone cannot tell "the guard
#      correctly rejects a failure" apart from "the fixture is broken and
#      nothing ever reaches PASSED".
run_dryrun_case "" "$DSB/platform-a"
rc=$?
check "(cc) full-release dry-run success control reaches the end" 0 $rc
if printf '%s' "$OUT" | grep -qF "FULL_DRYRUN_GUARD_PASSED"; then
  echo "  PASS:   -> guard reached the end of the function"; pass=$((pass+1))
else
  echo "  FAIL:   -> did not reach end of function (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi
call_count=$(publish_dryrun_call_count)
if [ "$call_count" -eq 3 ]; then
  echo "  PASS:   -> wasm, platform-a, and native were all attempted"; pass=$((pass+1))
else
  echo "  FAIL:   -> expected exactly 3 publish --dry-run calls (wasm, platform-a, native), got $call_count"; fail=$((fail+1))
fi

# (cc2) must-fail control for the argv-exactness check itself: a stub call
#       with a well-formed "publish --dry-run" PREFIX but an extra trailing
#       argument must be rejected by the stub's own $#-eq-2 check, not
#       silently accepted -- proving the exact-argv guard added above is
#       actually load-bearing and not just cosmetic. Exercised directly
#       against the stub binary (not through full_dryrun_guard, which never
#       emits a third argument itself) so this arm isolates the stub's own
#       argv-validation branch.
run_dryrun_case "" ""
OUT_EXTRA="$("$DSB/stub-bin/npm" publish --dry-run --unexpected-extra-flag 2>&1)"
rc_extra=$?
check "(cc2) stub rejects publish --dry-run with an extra trailing argument" 98 $rc_extra
if printf '%s' "$OUT_EXTRA" | grep -qF "UNEXPECTED PUBLISH ARGV"; then
  echo "  PASS:   -> stub's exact-argv diagnostic printed"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing exact-argv diagnostic (got: $(printf '%s' "$OUT_EXTRA" | tr '\n' '|'))"; fail=$((fail+1))
fi

echo
echo "=== publish-npm.sh argv usage guard self-test ==="
# parse_argv (G1) rejects any argv other than "" or "--dry-run" with
# `usage: $0 [--dry-run]; exit 2`. Call it directly with an arbitrary $1 --
# it takes its argument explicitly (not the top-level positional
# parameters), so calling it standalone cannot reach anything else in the
# script regardless of checkout state. $0 is rigged to $SRC via the
# `sh -c CMD name arg` positional trick (sourcing does not reset $0 or the
# positional parameters unless the `.` command is given explicit arguments),
# so the usage message's `$0` reads exactly as it would in a real
# `./scripts/publish-npm.sh --totally-bogus-flag` invocation.
RUNNER="$MSB/argv-runner.sh"
{
  echo 'PUBLISH_NPM_SH_LIB_ONLY=1'
  printf '. %q\n' "$SRC"
  printf 'parse_argv "$1"\n'
  printf 'echo ARGV_GUARD_PASSED\n'
} > "$RUNNER"
OUT="$(/bin/sh -c "set -e; . '$RUNNER'" "$SRC" --totally-bogus-flag 2>&1)"; rc=$?
check "(aa) unrecognized argv fails closed" 2 $rc
if printf '%s' "$OUT" | grep -qF "usage: $SRC [--dry-run]"; then
  echo "  PASS:   -> usage diagnostic printed"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing usage diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi
# (aa2) must-PASS control: "--dry-run" must NOT trip this guard.
RUNNER="$MSB/argv2-runner.sh"
{
  echo 'PUBLISH_NPM_SH_LIB_ONLY=1'
  printf '. %q\n' "$SRC"
  printf 'parse_argv "$1"\n'
  printf 'echo "MODE=$MODE"\n'
  printf 'echo ARGV_GUARD_PASSED\n'
} > "$RUNNER"
OUT="$(/bin/sh -c "set -e; . '$RUNNER'" "$SRC" --dry-run 2>&1)"; rc=$?
check "(aa2) --dry-run is accepted and sets MODE=dry-run" 0 $rc
if printf '%s' "$OUT" | grep -qF "MODE=dry-run"; then
  echo "  PASS:   -> MODE set to dry-run"; pass=$((pass+1))
else
  echo "  FAIL:   -> MODE not set correctly (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

echo
echo "=== publish-npm.sh sourceable-without-executing self-test ==="
# The whole point of the refactor: sourcing under PUBLISH_NPM_SH_LIB_ONLY
# must define every guard function and must NOT run main()'s side effects
# (no preflight echoes, no npm/node invocations of any kind). Every arm
# above already depends on this holding, but none of them proves it
# directly -- this is the must-PASS control for the sourcing mechanism
# itself. Positive control passed to `command -v` alongside the guard names
# (`echo`, always defined) to prove `command -v` is not silently reporting
# false for everything in this shell.
OUT="$(PUBLISH_NPM_SH_LIB_ONLY=1 /bin/sh -c '. "$1"; echo SOURCED_OK; command -v echo >/dev/null 2>&1 && echo COMMAND_V_WORKS; command -v main >/dev/null 2>&1 && echo MAIN_DEFINED; command -v parse_argv >/dev/null 2>&1 && echo PARSE_ARGV_DEFINED; command -v platform_matrix_guard >/dev/null 2>&1 && echo MATRIX_DEFINED; command -v check_platform_pkgjson >/dev/null 2>&1 && echo PKGJSON_DEFINED; command -v check_version_available >/dev/null 2>&1 && echo CVA_DEFINED; command -v run_version_checks >/dev/null 2>&1 && echo RUNCHECKS_DEFINED; command -v full_dryrun_guard >/dev/null 2>&1 && echo DRYRUN_DEFINED; command -v main_packlist_guard >/dev/null 2>&1 && echo PACKLIST_DEFINED' sh "$SRC" 2>&1)"
rc=$?
check "(ff) sourcing with the sentinel set exits cleanly" 0 $rc
for token in SOURCED_OK COMMAND_V_WORKS MAIN_DEFINED PARSE_ARGV_DEFINED MATRIX_DEFINED PKGJSON_DEFINED CVA_DEFINED RUNCHECKS_DEFINED DRYRUN_DEFINED PACKLIST_DEFINED; do
  if printf '%s' "$OUT" | grep -qF "$token"; then
    echo "  PASS:   -> $token"; pass=$((pass+1))
  else
    echo "  FAIL:   -> missing $token (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
  fi
done
if printf '%s' "$OUT" | grep -qF "Preflight"; then
  echo "  FAIL:   -> sourcing under the sentinel ran main()'s side effects"; fail=$((fail+1))
else
  echo "  PASS:   -> sourcing under the sentinel produced no main() side effects"; pass=$((pass+1))
fi

echo
echo "=== publish-npm.sh full-script dry-run self-test (does-it-still-publish, sentinel unset) ==="
# Every arm above exercises one function in isolation under the sentinel.
# None of them proves the UNSOURCED, sentinel-unset top-level path -- the
# one real `sh scripts/publish-npm.sh --dry-run` invocation actually takes
# in CI and locally -- still wires every guard together and reaches
# completion. Copy the real script into a fixture repo layout (so $0's
# dirname resolves ROOT/WASM_DIR/NATIVE_DIR the same way it would in a real
# checkout) and execute it directly, with a single stub `npm` on PATH
# covering every subcommand the whole script invokes (view/pack/publish/run)
# so no network call or real publish can happen, per this repo's method
# constraints. Real `node` stays on PATH -- every `node -p require(...)`
# and the real assert-platform-packlist.mjs still run for real against the
# fixture tree, so this arm also proves the matrix guard's content checks
# fire in the unsourced path, not just under the per-function arms above.
EFIX="$SB/e2e"
mkdir -p "$EFIX/repo/scripts" "$EFIX/repo/npm/lattice-embed-wasm"
cp "$SRC" "$EFIX/repo/scripts/publish-npm.sh"
chmod +x "$EFIX/repo/scripts/publish-npm.sh"

cat > "$EFIX/repo/npm/lattice-embed-wasm/package.json" <<'EOF'
{"name": "@khive-ai/lattice-fixture-wasm", "version": "1.2.3"}
EOF

NATIVE="$EFIX/repo/npm/lattice-embed-native"
build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
add_valid_platform "$NATIVE" "darwin-arm64" "1.2.3"

EGENSTUB="$SB/e2e-stub-bin"
mkdir -p "$EGENSTUB"
cat > "$EGENSTUB/npm" <<'NPMSTUB'
#!/usr/bin/env bash
echo "$*" >> "$(dirname "$0")/calls.log"
case "$1" in
  run)
    exit 0
    ;;
  view)
    echo '{"error":{"code":"E404","summary":"Not Found"}}'
    exit 1
    ;;
  pack)
    if [ "$(basename "$PWD")" = "lattice-embed-native" ]; then
      echo '[{"files":[{"path":"package.json"},{"path":"README.md"},{"path":"binding.js"},{"path":"index.js"},{"path":"index.d.ts"}]}]'
    else
      echo '[{"files":[{"path":"package.json"},{"path":"lattice-embed-native.darwin-arm64.node"}]}]'
    fi
    exit 0
    ;;
  publish)
    if [ "$#" -ne 2 ] || [ "$2" != "--dry-run" ]; then
      echo "UNEXPECTED PUBLISH ARGV in e2e stub: npm $* in $PWD (expected exactly: publish --dry-run)" >&2
      exit 98
    fi
    exit 0
    ;;
  *)
    echo "unexpected npm invocation in e2e stub: $*" >&2
    exit 99
    ;;
esac
NPMSTUB
chmod +x "$EGENSTUB/npm"

OUT="$(cd "$EFIX/repo" && PATH="$EGENSTUB:$PATH" NPM_CONFIG_CACHE="$SB/npm-cache-e2e" /bin/sh scripts/publish-npm.sh --dry-run 2>&1)"
rc=$?
check "(gg) full script dry-run reaches completion end-to-end" 0 $rc
for banner in \
  "=== Preflight (version-exists check against the npm registry) ===" \
  "Preflight OK: no version collisions." \
  "=== Preflight (dry-run: pack + gate the full release) ===" \
  "Preflight OK." \
  "=== Dry run complete: nothing published ==="
do
  if printf '%s' "$OUT" | grep -qF -- "$banner"; then
    echo "  PASS:   -> saw: $banner"; pass=$((pass+1))
  else
    echo "  FAIL:   -> missing: $banner (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
  fi
done

# (gg2) must-fail control for the same E2E path: break the platform matrix
# (drop the platform package entirely) and confirm the real, unsourced
# script fails closed instead of just reaching completion regardless of
# input -- without this, (gg) alone cannot tell "the wiring correctly
# propagates a guard failure" apart from "the fixture always succeeds".
EFIX2="$SB/e2e-fail"
mkdir -p "$EFIX2/repo/scripts" "$EFIX2/repo/npm/lattice-embed-wasm"
cp "$SRC" "$EFIX2/repo/scripts/publish-npm.sh"
chmod +x "$EFIX2/repo/scripts/publish-npm.sh"
cat > "$EFIX2/repo/npm/lattice-embed-wasm/package.json" <<'EOF'
{"name": "@khive-ai/lattice-fixture-wasm", "version": "1.2.3"}
EOF
NATIVE2="$EFIX2/repo/npm/lattice-embed-native"
build_native_fixture "$NATIVE2" "darwin-arm64:1.2.3"
# Deliberately omit add_valid_platform: optionalDependencies advertises
# darwin-arm64 but no npm/darwin-arm64/ directory backs it.
OUT="$(cd "$EFIX2/repo" && PATH="$EGENSTUB:$PATH" NPM_CONFIG_CACHE="$SB/npm-cache-e2e-fail" /bin/sh scripts/publish-npm.sh --dry-run 2>&1)"
rc=$?
check "(gg2) full script dry-run fails closed on a broken platform matrix" 1 $rc
if printf '%s' "$OUT" | grep -qF "missing native binary for platform 'darwin-arm64' under"; then
  echo "  PASS:   -> matrix-guard diagnostic propagated through the unsourced top-level path"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing matrix-guard diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

echo
echo "=== publish-npm.sh full-script dry-run call-set wiring self-test ==="
# (gg) and (gg2) prove the unsourced path reaches completion and fails closed
# on a broken matrix, but neither one proves WHICH guards actually ran to get
# there -- deleting main_packlist_guard's or full_dryrun_guard's call from
# main() (the wiring gap Finding 4 names) still lets this well-formed fixture
# sail through to "Dry run complete" with nothing above to notice the missing
# call, because both arms only assert banners and an exit code. Rerun the
# same unsourced --dry-run path with a stub that logs "$1 $2 basename($PWD)"
# for every npm invocation it receives, then assert that the three call-site
# families the review named -- version check (npm view), full dry run (npm
# publish --dry-run per package), and main packlist (npm pack --dry-run
# against the native package) -- actually landed in the log. A direct call
# to the guard function (as the arms above (v)-(cc) do) only proves the
# function itself works; this proves main() still reaches it.
HSB="$SB/e2e-callset"
mkdir -p "$HSB/repo/scripts" "$HSB/repo/npm/lattice-embed-wasm"
cp "$SRC" "$HSB/repo/scripts/publish-npm.sh"
chmod +x "$HSB/repo/scripts/publish-npm.sh"
cat > "$HSB/repo/npm/lattice-embed-wasm/package.json" <<'EOF'
{"name": "@khive-ai/lattice-fixture-wasm", "version": "1.2.3"}
EOF
NATIVE3="$HSB/repo/npm/lattice-embed-native"
build_native_fixture "$NATIVE3" "darwin-arm64:1.2.3"
add_valid_platform "$NATIVE3" "darwin-arm64" "1.2.3"

HGENSTUB="$SB/e2e-callset-stub-bin"
mkdir -p "$HGENSTUB"
: > "$HSB/calls.log"
cat > "$HGENSTUB/npm" <<EOF
#!/usr/bin/env bash
# Logs the COMPLETE argv (not just \$1/\$2, which would silently drop e.g. a
# missing trailing --json) alongside the caller's cwd basename, separated by
# " -- " so a call with zero arguments still parses back unambiguously.
echo "\$* -- \$(basename "\$PWD")" >> "$HSB/calls.log"
case "\$1" in
  run)
    exit 0
    ;;
  view)
    echo '{"error":{"code":"E404","summary":"Not Found"}}'
    exit 1
    ;;
  pack)
    if [ "\$(basename "\$PWD")" = "lattice-embed-native" ]; then
      echo '[{"files":[{"path":"package.json"},{"path":"README.md"},{"path":"binding.js"},{"path":"index.js"},{"path":"index.d.ts"}]}]'
    else
      echo '[{"files":[{"path":"package.json"},{"path":"lattice-embed-native.darwin-arm64.node"}]}]'
    fi
    exit 0
    ;;
  publish)
    if [ "\$#" -ne 2 ] || [ "\$2" != "--dry-run" ]; then
      echo "UNEXPECTED PUBLISH ARGV in call-set stub: npm \$* in \$PWD (expected exactly: publish --dry-run)" >&2
      exit 98
    fi
    exit 0
    ;;
  *)
    echo "unexpected npm invocation in call-set stub: \$*" >&2
    exit 99
    ;;
esac
EOF
chmod +x "$HGENSTUB/npm"

OUT="$(cd "$HSB/repo" && PATH="$HGENSTUB:$PATH" NPM_CONFIG_CACHE="$SB/npm-cache-e2e-callset" /bin/sh scripts/publish-npm.sh --dry-run 2>&1)"
rc=$?
check "(hh) full script dry-run with call-logging stub reaches completion" 0 $rc

CALLS="$(cat "$HSB/calls.log" 2>/dev/null)"
# Exact whole-line match (grep -x) against the complete "argv -- cwd" record
# logged above, not a substring search -- a substring needle like "view
# name@1.2.3" would still match if the real call dropped its trailing
# "version --json" arguments (or gained an unexpected extra one), so it
# cannot actually prove the argv is what production sends.
assert_call() {  # $1=label $2=exact expected "argv -- cwd" line
  if printf '%s\n' "$CALLS" | /usr/bin/grep -qxF -- "$2"; then
    echo "  PASS:   -> (hh) saw $1: $2"; pass=$((pass+1))
  else
    echo "  FAIL:   -> (hh) missing $1: $2 (log: $(printf '%s' "$CALLS" | tr '\n' '|'))"; fail=$((fail+1))
  fi
}
# check_version_available runs with no `cd` of its own, so every "view" call
# happens from the top-level cwd the script was invoked from ("repo", per
# `cd "$HSB/repo" && ... /bin/sh scripts/publish-npm.sh` above) -- NOT from
# each package's own directory.
assert_call "version check on the wasm package" "view @khive-ai/lattice-fixture-wasm@1.2.3 version --json -- repo"
assert_call "version check on the platform package" "view @khive-ai/lattice-embed-darwin-arm64@1.2.3 version --json -- repo"
assert_call "version check on the native package" "view @khive-ai/lattice-embed@1.2.3 version --json -- repo"
assert_call "full dry run of the wasm package" "publish --dry-run -- lattice-embed-wasm"
assert_call "full dry run of the platform package" "publish --dry-run -- darwin-arm64"
assert_call "full dry run of the native package" "publish --dry-run -- lattice-embed-native"
assert_call "main packlist run" "pack --dry-run --json -- lattice-embed-native"

echo
echo "=== marker-extraction mechanism removal self-test ==="
# Every arm above calls a real function instead of cutting a text fragment
# out of the live source with a BEGIN/END comment-pair convention read by an
# awk range. Prove that retired mechanism is actually gone from both files,
# not merely unused: no leftover extractor function definition or awk range
# pattern in this file, and no leftover comment-pair markers in the script
# under test (this file's own header prose above is free to name the
# retired mechanism in past tense, so the checks below key on the CODE
# shapes -- a function signature, an awk range operator, a comment-pair
# convention string -- not on any English mention of it). Known positive
# first: grep this very file for its own shebang line, so a zero-hit result
# below is trustworthy rather than a sign the grep invocation itself is
# broken.
if ! /usr/bin/grep -qF -- '#!/usr/bin/env bash' "$0"; then
  echo "FATAL: known-positive grep for this file's own shebang line found nothing -- grep instrument is not trustworthy" >&2
  exit 1
fi
# Patterns are assembled from concatenated halves rather than written as one
# literal token, so this detection code's own string arguments -- which
# necessarily name what they search for -- do not match themselves and
# produce a false FAIL against this very file.
extractor_pat="extract_marker""_block[[:space:]]*\\(\\)"
if /usr/bin/grep -qE -- "$extractor_pat" "$0"; then
  echo "  FAIL: an extractor function definition is still present in $0"; fail=$((fail+1))
else
  echo "  PASS: no extractor function definition remains in $0"; pass=$((pass+1))
fi
awk_range_pat="index(\$0,b){f=1;next}"" index(\$0,e){f=0} f"
if /usr/bin/grep -qF -- "$awk_range_pat" "$0"; then
  echo "  FAIL: the old awk BEGIN/END range program is still present in $0"; fail=$((fail+1))
else
  echo "  PASS: no awk BEGIN/END range program remains in $0"; pass=$((pass+1))
fi
marker_pat="selftest-extraction""-marker"
if /usr/bin/grep -qF -- "$marker_pat" "$SRC"; then
  echo "  FAIL: selftest-extraction-marker comments still present in $SRC"; fail=$((fail+1))
else
  echo "  PASS: no selftest-extraction-marker comments remain in $SRC"; pass=$((pass+1))
fi
marker_pat_colon="${marker_pat}:"
if /usr/bin/grep -qF -- "$marker_pat_colon" "$0"; then
  echo "  FAIL: a live selftest-extraction-marker comment-pair convention string is still present in $0"; fail=$((fail+1))
else
  echo "  PASS: no live selftest-extraction-marker comment-pair convention string remains in $0"; pass=$((pass+1))
fi

echo "=== $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]

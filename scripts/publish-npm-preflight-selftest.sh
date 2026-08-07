#!/usr/bin/env bash
# Self-test for the check_version_available() preflight in scripts/publish-npm.sh.
#
# publish-npm.sh is not a sourceable library (it's a top-to-bottom release
# script gated by a `case "${1:-}"` on argv), so there's no existing home to
# import just this function from. Rather than refactor the release script
# into a lib/ module for one test (out of scope for this fix), this harness
# extracts the live check_version_available() function body verbatim with
# awk and evals it in a subshell per case, with a stub `npm` shimmed onto
# PATH ahead of the real one. Extraction means this test only stays honest
# as long as the function's name and brace-delimited shape don't change
# without the extraction being re-verified -- see step 0 below, which fails
# loudly instead of silently testing stale text.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/scripts/publish-npm.sh"

FN_BODY="$(awk '/^check_version_available\(\) \{/,/^\}$/' "$SRC")"
if [ -z "$FN_BODY" ]; then
  echo "FATAL: could not extract check_version_available() from $SRC -- has it been renamed" >&2
  echo "  or reshaped? The extraction below would otherwise silently test nothing." >&2
  exit 1
fi

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

# A minimal package dir the function's `node -p require(...)` calls can read.
PKGDIR="$SB/pkg"
mkdir -p "$PKGDIR"
cat > "$PKGDIR/package.json" <<'EOF'
{"name": "@khive-ai/lattice-fixture", "version": "9.9.9"}
EOF

STUBDIR="$SB/stub-bin"
mkdir -p "$STUBDIR"

RUNNER="$SB/runner.sh"

run_case() {  # $1 = stub npm script body
  cat > "$STUBDIR/npm" <<EOF
#!/usr/bin/env bash
$1
EOF
  chmod +x "$STUBDIR/npm"
  {
    printf '%s\n' "$FN_BODY"
    printf 'check_version_available %q\n' "$PKGDIR"
  } > "$RUNNER"
  # /bin/sh with set -e: the production shell and options publish-npm.sh
  # actually runs under (it has `#!/bin/sh` and `set -e` at its own top).
  # Testing under bash without -e (as this harness used to) exercises a
  # shell environment the release preflight never runs in.
  OUT="$(PATH="$STUBDIR:$PATH" /bin/sh -c "set -e; . '$RUNNER'" 2>&1)"
  return $?
}

echo "=== publish-npm.sh preflight self-test (7 stubbed npm dispositions) ==="

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
#     rejection. Measured with the old plain-text "9.9.9" stub: disabling
#     G10 still left this arm at rc 1 (invalid JSON -> empty error_code ->
#     the E404 guard's own "did not return npm's not-found response"
#     branch), with only the message assertion catching it -- not isolating.
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
# agreement, packlist content) is inline top-to-bottom script, not a
# function -- there is no brace-delimited body to extract the way
# check_version_available's is above. Instead the guard is bracketed in
# scripts/publish-npm.sh by literal marker comments; extract the text
# between them and eval it under /bin/sh against a fixture NATIVE_DIR tree.
# Every command the extracted text runs (`node -p`, `npm pack --dry-run
# --json`) is local/offline -- no registry access, so no stubbing is needed
# here the way check_version_available's `npm` had to be stubbed.
MATRIX_BODY="$(awk '/^# selftest-extraction-marker: PLATFORM_MATRIX_GUARD_BEGIN$/,/^# selftest-extraction-marker: PLATFORM_MATRIX_GUARD_END$/' "$SRC")"
if [ -z "$MATRIX_BODY" ]; then
  echo "FATAL: could not extract the platform matrix guard from $SRC -- have its" >&2
  echo "  selftest-extraction-marker comments been removed or reshaped? The" >&2
  echo "  extraction below would otherwise silently test nothing." >&2
  exit 1
fi

REAL_PACKLIST_ASSERT="$REPO/npm/lattice-embed-native/scripts/assert-platform-packlist.mjs"
if [ ! -f "$REAL_PACKLIST_ASSERT" ]; then
  echo "FATAL: $REAL_PACKLIST_ASSERT not found -- has it moved?" >&2
  exit 1
fi

MSB="$SB/matrix"
mkdir -p "$MSB"

# Build a fixture NATIVE_DIR ($1) whose optionalDependencies match the
# platform/version pairs given as "$2..." (each "platform:version"). Copies
# the real assert-platform-packlist.mjs alongside it so the packlist guard
# step exercises the actual production check, not a stand-in.
build_native_fixture() {
  native="$1"; shift
  mkdir -p "$native/scripts"
  cp "$REAL_PACKLIST_ASSERT" "$native/scripts/assert-platform-packlist.mjs"
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
    printf 'NATIVE_DIR=%q\n' "$1"
    printf '%s\n' "$MATRIX_BODY"
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
#     instead of holding rc 1 from G9's own rejection. Measured with the
#     original fixture (no .node file on disk at all): disabling G4 still
#     left this arm at rc 1 via G9 (an assert-platform-packlist.mjs
#     assertion failure over zero .node files), with only the
#     "missing native binary" message assertion catching it -- not isolating.
#     (i) already isolates the misnamed-but-present case; this arm now
#     isolates the absent-under-the-declared-name case specifically.
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
#     optionalDependencies-value guard at :155 (same wrong value, same
#     NATIVE_VERSION comparison), and the two guards stop isolating: measured
#     by disabling the :125 branch alone, which left the exit-code assertion
#     GREEN (rc still 1, from :155) while only the message assertion caught
#     it. An arm whose exit assertion survives its own guard being deleted is
#     not isolating that guard.
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

# (m) a full, legitimate two-platform matrix -- must pass every new guard.
# This fixture also doubles as the must-PASS control for the
# optionalDependencies-value guard below: build_native_fixture's
# "platform:version" pairs populate optionalDependencies at the same
# version add_valid_platform gives each platform package, so (m) proves the
# new check does not reject a genuinely matched matrix.
NATIVE="$MSB/m-valid"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3" "linux-x64-gnu:1.2.3"
add_valid_platform "$NATIVE" "darwin-arm64" "1.2.3"
add_valid_platform "$NATIVE" "linux-x64-gnu" "1.2.3"
run_matrix_case "$NATIVE"; rc=$?
check "(m) full valid matrix passes" 0 $rc
if printf '%s' "$OUT" | grep -qF "MATRIX_GUARD_PASSED"; then
  echo "  PASS:   -> guard reached the end of the extracted block"; pass=$((pass+1))
else
  echo "  FAIL:   -> did not reach end of block (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
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
#     Object.keys(optionalDependencies) a few lines above (selftest-extraction-marker:
#     PLATFORM_MATRIX_GUARD_BEGIN), so a platform this loop ever reaches by
#     construction always has a key -- an absent key never enters the loop at
#     all and is instead the empty-optionalDependencies shape arm (h) covers,
#     or (for one absent key among several present) the "missing native
#     binary" shape arms (j) already cover, since a key-less platform has no
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
# The matrix guard above never touches the main package's own packlist call
# (scripts/publish-npm.sh: MAIN_PACKLIST_GUARD_BEGIN/END, `npm run packlist`
# inside NATIVE_DIR) -- it is a separate, later step in the real script, so
# it needs its own extraction and its own fixture: a minimal npm project
# satisfying assert-packlist.mjs's required-files list, with an optional
# extra path appended to "files" to trip a forbidden pattern.
MAIN_PACKLIST_BODY="$(awk '/^# selftest-extraction-marker: MAIN_PACKLIST_GUARD_BEGIN$/,/^# selftest-extraction-marker: MAIN_PACKLIST_GUARD_END$/' "$SRC")"
if [ -z "$MAIN_PACKLIST_BODY" ]; then
  echo "FATAL: could not extract the main-package packlist guard from $SRC -- have its" >&2
  echo "  selftest-extraction-marker comments been removed or reshaped? The" >&2
  echo "  extraction below would otherwise silently test nothing." >&2
  exit 1
fi

REAL_MAIN_PACKLIST_ASSERT="$REPO/npm/lattice-embed-native/scripts/assert-packlist.mjs"
if [ ! -f "$REAL_MAIN_PACKLIST_ASSERT" ]; then
  echo "FATAL: $REAL_MAIN_PACKLIST_ASSERT not found -- has it moved?" >&2
  exit 1
fi

MPB="$SB/main-packlist"
mkdir -p "$MPB"

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
    printf 'NATIVE_DIR=%q\n' "$1"
    printf '%s\n' "$MAIN_PACKLIST_BODY"
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
  echo "  PASS:   -> guard reached the end of the extracted block"; pass=$((pass+1))
else
  echo "  FAIL:   -> did not reach end of block (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
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
# publish-npm.sh:95-102 (G3) rejects an advertised platform whose
# npm/<platform>/package.json is absent entirely -- distinct from the
# exact-.node-path guard a few lines later (G4, arm (j) above), which always
# writes a package.json and omits only the .node file. G3's own check sits
# directly before an UNCONDITIONAL `node -p require($pkgjson)...` call later
# in the same for-loop iteration, so running the whole platform-matrix body
# (as arms (h)-(n) do via run_matrix_case) cannot isolate G3 alone: a
# defeated G3 still crashes on that require() with Node's own
# MODULE_NOT_FOUND (rc 1), which happens to match G3's own expected rc and
# so leaves the exit-code assertion falsely green -- measured: mutating
# :95's condition to `if false` gave 53 passed, 1 failed, with only the
# diagnostic-message assertion catching it, not the exit code. Bounding the
# extraction to JUST the pkgjson-existence check (its own
# PLATFORM_PKGJSON_GUARD marker pair in publish-npm.sh, ending before the
# require() call) keeps that crash out of the tested block entirely, so a
# defeated G3 now falls straight through to the PASSED marker with rc 0 --
# an unambiguous isolation instead of a coincidentally-matching rc.
PLATFORM_PKGJSON_BODY="$(awk '/selftest-extraction-marker: PLATFORM_PKGJSON_GUARD_BEGIN/,/selftest-extraction-marker: PLATFORM_PKGJSON_GUARD_END/' "$SRC")"
if [ -z "$PLATFORM_PKGJSON_BODY" ]; then
  echo "FATAL: could not extract the platform package.json guard from $SRC -- have its" >&2
  echo "  selftest-extraction-marker comments been removed or reshaped? The" >&2
  echo "  extraction below would otherwise silently test nothing." >&2
  exit 1
fi

run_pkgjson_case() {  # $1=NATIVE_DIR $2=platform
  RUNNER="$MSB/pkgjson-runner.sh"
  {
    printf 'NATIVE_DIR=%q\n' "$1"
    printf 'platform=%q\n' "$2"
    printf '%s\n' "$PLATFORM_PKGJSON_BODY"
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

echo
echo "=== publish-npm.sh npm-pack-command-failure guard self-test ==="
# publish-npm.sh:168-170 fails closed when `npm pack --dry-run --json`
# itself returns nonzero -- distinct from the packlist CONTENT guard at
# :172-178 (arm (l) above), which requires a pack that succeeds but whose
# contents are wrong. Measured before this arm existed: replacing :170's
# `exit 1` with `:` left the suite fully green, because every fixture up to
# that point had its `npm pack` call succeed, so nothing exercised this
# branch. That baseline is what motivated adding the arm below, which
# re-creates the same failure directly via a stub and now holds it (verified
# by re-running the same mutation with this arm in place: it goes red on its
# own exit-code assertion).
#
# Isolate by shimming a stub `npm` ahead of the real one on PATH that fails
# only on `npm pack`; nothing else in the extracted matrix guard body
# invokes `npm` (only `node -p` and this one pack call), so the stub cannot
# accidentally mask a different guard.
#
# The stub ALSO prints a well-formed pack manifest on stdout before exiting
# nonzero. This matters: :168's diagnostic echo and :170's `exit 1` are two
# separate statements in the same `if` block, so a mutation that only guts
# :170 (leaving the echo) would still print :170's own message on the way
# through to :172-178's packlist-content guard, which would then fail
# closed on its own over the malformed/empty capture and produce the SAME
# overall rc=1 -- an arm asserting only "message contains :170's text" would
# stay green under that mutation without :170 itself doing any rejecting.
# Handing the downstream check a manifest it accepts closes that gap: if
# :170 is defeated, :172-178 is satisfied too and the whole block reaches
# MATRIX_GUARD_PASSED with rc=0, so this arm's rc assertion alone -- not
# just its message assertion -- is what catches the defeat.
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
  printf 'NATIVE_DIR=%q\n' "$NATIVE"
  printf '%s\n' "$MATRIX_BODY"
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
# package -- publish-npm.sh:236,238,240. Measured before this arm existed:
# replacing :240's `check_version_available "$NATIVE_DIR"` with `:` left the
# suite fully green; a native name@version collision would reach the real
# publish loop undetected. That baseline motivated the arm below, which now
# holds the same mutation (verified by re-running it with this arm in
# place: it goes red on both of its own assertions). A guard that works
# perfectly and is
# never called is indistinguishable from a broken one at runtime, so this
# needs a CALLER-level test: extract the three call sites verbatim and run
# them with check_version_available replaced by a recording stub, then
# assert every expected target was actually passed to it.
# Anchored on marker comments (like the matrix/main-packlist extractions
# above), NOT on the literal text of the first/last call. A range anchored on
# the call sites' own text cannot detect the deletion of the LAST call: awk's
# range pattern only closes when its end pattern matches again, so removing
# the "$NATIVE_DIR" line would leave the range unclosed and silently overrun
# to end-of-file, capturing unrelated later script content instead of
# reporting a clean miss. Measured directly: doing exactly that produced a
# nonsense capture that tried to `cd` into a fixture marker string.
CALL_SITES_BODY="$(awk '/^# selftest-extraction-marker: VERSION_CHECK_CALLSITES_BEGIN$/,/^# selftest-extraction-marker: VERSION_CHECK_CALLSITES_END$/' "$SRC" | grep -v 'selftest-extraction-marker')"
if [ -z "$CALL_SITES_BODY" ]; then
  echo "FATAL: could not extract the check_version_available call sites from $SRC -- have" >&2
  echo "  its selftest-extraction-marker comments been removed or reshaped? The" >&2
  echo "  extraction below would otherwise silently test nothing." >&2
  exit 1
fi

ZSB="$SB/z-callsites"
mkdir -p "$ZSB"
CVA_LOG="$ZSB/calls.log"
rm -f "$CVA_LOG"
RUNNER="$ZSB/runner.sh"
{
  printf 'check_version_available() { printf "%%s\\n" "$1" >> %q; }\n' "$CVA_LOG"
  printf 'WASM_DIR=%q\n' "wasm-dir-marker"
  printf 'NATIVE_DIR=%q\n' "native-dir-marker"
  printf 'PLATFORM_DIRS=%q\n' "plat-a-marker plat-b-marker"
  printf '%s\n' "$CALL_SITES_BODY"
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
# publish-npm.sh:245-254 packs and gates the WHOLE release (wasm, every
# platform, native main) via three `npm publish --dry-run` calls before any
# real publish -- the immutability contract this script exists to protect
# (npm name@version tuples cannot be republished, so a real publish that
# fails partway is unrecoverable). This guard has no explicit `exit N`: it
# relies entirely on the script's own top-level `set -e` to abort on the
# first nonzero `npm publish --dry-run`, so `grep -nE 'exit[[:space:]]+[0-9]+'`
# does not find it at all and it was covered by zero arms. Measured:
# replacing the first dry-run call (:250) with `:` left the suite fully
# green -- a dry-run failure on any package would previously reach the real
# publish loop undetected.
FULL_DRYRUN_BODY="$(awk '/selftest-extraction-marker: FULL_DRYRUN_GUARD_BEGIN/,/selftest-extraction-marker: FULL_DRYRUN_GUARD_END/' "$SRC")"
if [ -z "$FULL_DRYRUN_BODY" ]; then
  echo "FATAL: could not extract the full-release dry-run guard from $SRC -- have its" >&2
  echo "  selftest-extraction-marker comments been removed or reshaped? The" >&2
  echo "  extraction below would otherwise silently test nothing." >&2
  exit 1
fi

DSB="$SB/dryrun"
mkdir -p "$DSB/wasm" "$DSB/native" "$DSB/stub-bin"

# The stub keys its failure on $PWD (which package's dry-run is running),
# not on the mere fact that `npm publish` was invoked. An unconditional
# "always fail" stub cannot isolate the WASM call specifically: mutating the
# WASM call away (as the measured defect above does) still leaves the
# NATIVE_DIR call later in the same chained block, which an unconditional
# stub would ALSO fail -- reddening the arm for the wrong reason and
# masking the very mutation it exists to catch, the same overdetermination
# shape as the (bb)/(c)/(j) fixes above.
run_dryrun_case() {  # $1 = absolute dir the stub should fail in, or "" for none
  fail_dir="$1"
  cat > "$DSB/stub-bin/npm" <<EOF
#!/usr/bin/env bash
if [ "\$1" = "publish" ]; then
  if [ "\$PWD" = "$fail_dir" ]; then
    echo "npm error simulated dry-run failure for selftest arm (bb) in \$PWD" >&2
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
    printf 'WASM_DIR=%q\n' "$DSB/wasm"
    printf 'PLATFORM_DIRS=%q\n' ""
    printf 'NATIVE_DIR=%q\n' "$DSB/native"
    printf '%s\n' "$FULL_DRYRUN_BODY"
    printf 'echo FULL_DRYRUN_GUARD_PASSED\n'
  } > "$RUNNER"
  OUT="$(PATH="$DSB/stub-bin:$PATH" /bin/sh -c "set -e; . '$RUNNER'" 2>&1)"
  return $?
}

# (bb) the wasm package's dry-run fails specifically (stub fails only when
#      $PWD is $WASM_DIR) -- must abort before reaching the native
#      package's own dry-run or the PASSED marker.
run_dryrun_case "$DSB/wasm"
rc=$?
check "(bb) wasm dry-run failure aborts the release" 1 $rc
if printf '%s' "$OUT" | grep -qF "FULL_DRYRUN_GUARD_PASSED"; then
  echo "  FAIL:   -> guard reached the end of the extracted block despite the failure"; fail=$((fail+1))
else
  echo "  PASS:   -> guard aborted before reaching the end of the extracted block"; pass=$((pass+1))
fi

# (cc) must-PASS control: every dry-run succeeds (no fail_dir) -- the guard
#      must reach the end of the extracted block. Without this, (bb) alone
#      cannot tell "the guard correctly rejects a failure" apart from "the
#      fixture is broken and nothing ever reaches PASSED".
run_dryrun_case ""
rc=$?
check "(cc) full-release dry-run success control reaches the end" 0 $rc
if printf '%s' "$OUT" | grep -qF "FULL_DRYRUN_GUARD_PASSED"; then
  echo "  PASS:   -> guard reached the end of the extracted block"; pass=$((pass+1))
else
  echo "  FAIL:   -> did not reach end of block (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

echo
echo "=== publish-npm.sh argv usage guard self-test ==="
# publish-npm.sh:25-36's case statement (G1) rejects any argv other than ""
# or "--dry-run" with `usage: ... ; exit 2` at :34 -- found during
# enumeration (grep -n 'exit [0-9]' scripts/publish-npm.sh), covered by zero
# arms before this one. Invoking the real script directly is not isolated
# from the rest of the script, though: if :34's `exit 2` is defeated, the
# case's `*)` branch falls through with $MODE unset and execution continues
# into the platform-matrix guard against whatever repo state the test
# happens to run in -- measured: replacing :34's `exit 2` with `:` gave
# rc 1 (from the current checkout's own matrix guard) instead of rc 2, which
# this arm's exit-code assertion does catch as a mismatch, but only because
# a coincidentally-different rc came out of unrelated, environment-dependent
# code; the usage-message assertion stays green regardless (the case's own
# echo runs unconditionally before the mutated exit), and a fresh checkout
# with a fully populated matrix could produce yet another rc, including a
# false rc 2. Extracting just the case statement (its own ARGV_GUARD marker
# pair in publish-npm.sh) and running it standalone -- with $0/$1 set via
# `sh -c CMD name arg` so the extracted `$0`/`${1:-}` references still see
# the real script path and the bogus flag -- bounds the test so it cannot
# reach the matrix guard or anything else, regardless of checkout state.
ARGV_BODY="$(awk '/selftest-extraction-marker: ARGV_GUARD_BEGIN/,/selftest-extraction-marker: ARGV_GUARD_END/' "$SRC")"
if [ -z "$ARGV_BODY" ]; then
  echo "FATAL: could not extract the argv usage guard from $SRC -- have its" >&2
  echo "  selftest-extraction-marker comments been removed or reshaped? The" >&2
  echo "  extraction below would otherwise silently test nothing." >&2
  exit 1
fi

RUNNER="$MSB/argv-runner.sh"
{
  printf '%s\n' "$ARGV_BODY"
  printf 'echo ARGV_GUARD_PASSED\n'
} > "$RUNNER"
OUT="$(/bin/sh -c "set -e; . '$RUNNER'" "$SRC" --totally-bogus-flag 2>&1)"; rc=$?
check "(aa) unrecognized argv fails closed" 2 $rc
if printf '%s' "$OUT" | grep -qF "usage: $SRC [--dry-run]"; then
  echo "  PASS:   -> usage diagnostic printed"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing usage diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

echo "=== $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]

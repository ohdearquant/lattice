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

# (c) a successful lookup: version exists, exit 0 -> already-published branch.
run_case 'echo "9.9.9"
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

# (j) platform .node absent entirely -- must fail closed.
NATIVE="$MSB/j-absent"; build_native_fixture "$NATIVE" "darwin-arm64:1.2.3"
d="$NATIVE/npm/darwin-arm64"; mkdir -p "$d"
cat > "$d/package.json" <<'EOF'
{"name": "@khive-ai/lattice-embed-darwin-arm64", "version": "1.2.3", "main": "lattice-embed-native.darwin-arm64.node", "files": ["lattice-embed-native.darwin-arm64.node"]}
EOF
run_matrix_case "$NATIVE"; rc=$?
check "(j) absent .node fails closed" 1 $rc
if printf '%s' "$OUT" | grep -qF "missing native binary"; then
  echo "  PASS:   -> missing-binary diagnostic printed"; pass=$((pass+1))
else
  echo "  FAIL:   -> missing missing-binary diagnostic (got: $(printf '%s' "$OUT" | tr '\n' '|'))"; fail=$((fail+1))
fi

# (k) platform package version disagrees with the main package -- must fail
#     closed.
NATIVE="$MSB/k-version"; build_native_fixture "$NATIVE" "darwin-arm64:9.9.9"
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

echo "=== $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]

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

echo "=== $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]

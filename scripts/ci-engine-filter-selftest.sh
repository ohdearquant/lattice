#!/usr/bin/env bash
# Self-test for the engine/tune change classifier in .github/workflows/e2e-parity.yml.
#
# WHY THIS EXISTS. Most alternations in the engine filter are DIRECTORY PREFIXES
# (tests/fixtures/vision/, crates/embed/tests/wasm/, crates/*/src/, ...). A prefix
# matches any file placed under it, so a README committed inside a fixture
# directory classifies a documentation change as an engine change and demands the
# full macOS parity battery. PR #1159 changed 50 files, none under crates/*/src,
# and was blocked by exactly one path: tests/fixtures/vision/README.md.
#
# WHY IT READS THE WORKFLOW instead of restating the regex. A test carrying its own
# copy of the pattern proves only that the copy behaves; the two drift and nothing
# complains. This extracts the live regex from the YAML, so editing the gate is what
# this test grades.
#
#   bash scripts/ci-engine-filter-selftest.sh
set -uo pipefail

WF="$(cd "$(dirname "$0")/.." && pwd)/.github/workflows/e2e-parity.yml"
[ -r "$WF" ] || { echo "FAIL: cannot read $WF"; exit 1; }

# Extract the two classifier regexes from their `if grep -E '...'` lines. The
# engine line is identified by an anchor unique to it, the tune line by its own.
extract_re() {  # $1 = anchor substring unique to the wanted line
  grep -E "^ *if grep -E '" "$WF" \
    | grep -F "$1" \
    | head -1 \
    | sed -E "s/^ *if grep -E '//; s/' <<<.*$//"
}

ENGINE_RE="$(extract_re 'crates/(inference|embed)/src/')"
TUNE_RE="$(extract_re 'crates/tune/')"

# FAIL CLOSED ON EMPTY. An extraction that resolved nothing would make every
# assertion below pass vacuously (grep against an empty pattern matches all, or
# the harness silently degrades) and this file would report a green gate while
# grading nothing at all.
[ -n "$ENGINE_RE" ] || { echo "FAIL: could not extract the engine regex from $WF"; exit 1; }
[ -n "$TUNE_RE" ]   || { echo "FAIL: could not extract the tune regex from $WF";   exit 1; }

# Does the workflow strip documentation before classifying? Read from the file
# rather than assumed, so removing the fix is what flips this.
STRIPS_DOCS=0
grep -qF "grep -v '\\.md\$' <<<\"\$CHANGED\"" "$WF" && STRIPS_DOCS=1

classify() {  # $1 = regex, stdin = newline-separated paths -> prints true|false
  local re="$1" changed
  changed="$(cat)"
  if [ "$STRIPS_DOCS" -eq 1 ]; then
    changed="$(grep -v '\.md$' <<<"$changed" || true)"
  fi
  if grep -E "$re" <<<"$changed" >/dev/null; then echo true; else echo false; fi
}

FAILED=0
check() {  # $1 = label, $2 = expected, $3 = actual
  if [ "$2" = "$3" ]; then
    printf 'ok    %-58s %s\n' "$1" "$3"
  else
    printf 'FAIL  %-58s expected=%s actual=%s\n' "$1" "$2" "$3"
    FAILED=1
  fi
}

# ── the regression this file exists for ──────────────────────────────────────
# A README inside a fixture directory must NOT demand the parity battery.
check "vision fixture README alone -> engine" false \
  "$(classify "$ENGINE_RE" <<<'tests/fixtures/vision/README.md')"

# The real #1159 shape: docs across the tree, plus that README.
check "docs-only diff (#1159 shape) -> engine" false \
  "$(classify "$ENGINE_RE" <<<'README.md
docs/adr/ADR-064-ci-gate-taxonomy.md
crates/inference/README.md
crates/inference/METAL_TRACE.md
tests/fixtures/vision/README.md')"

# ── positive controls: the gate must still fire on real engine surface ───────
# Without these, deleting the whole regex would make this file pass.
check "inference source -> engine" true \
  "$(classify "$ENGINE_RE" <<<'crates/inference/src/rope.rs')"
check "embed source -> engine" true \
  "$(classify "$ENGINE_RE" <<<'crates/embed/src/simd/mod.rs')"
check "Cargo.lock -> engine" true \
  "$(classify "$ENGINE_RE" <<<'Cargo.lock')"
check "shared Rust setup action -> engine" true \
  "$(classify "$ENGINE_RE" <<<'.github/actions/rust-setup/action.yml')"

# A push to main is filtered before the classifier job exists, so the action
# path must also be present in the workflow trigger. Keeping this assertion in
# the same owning test makes either half of the routing contract fail closed.
PUSH_ROUTES_RUST_SETUP=false
sed -n '1,45p' "$WF" \
  | grep -qF "      - '.github/actions/rust-setup/**'" \
  && PUSH_ROUTES_RUST_SETUP=true
check "shared Rust setup action -> push workflow" true "$PUSH_ROUTES_RUST_SETUP"

# ── the over-exclusion guard ────────────────────────────────────────────────
# Stripping .md must not stop fixture DATA under the same directory from firing.
check "vision fixture DATA -> engine" true \
  "$(classify "$ENGINE_RE" <<<'tests/fixtures/vision/goldens_s3a.json')"
check "ppl fixture DATA -> engine" true \
  "$(classify "$ENGINE_RE" <<<'crates/inference/tests/fixtures/ppl_gate_v1/golden.json')"

# ── mixed diff: docs must not mask a real engine change ─────────────────────
check "docs + engine source -> engine" true \
  "$(classify "$ENGINE_RE" <<<'tests/fixtures/vision/README.md
crates/inference/src/rope.rs')"

# ── tune classifier carries the identical directory-prefix exposure ──────────
check "tune tests README alone -> tune" false \
  "$(classify "$TUNE_RE" <<<'crates/tune/tests/README.md')"
check "tune source -> tune" true \
  "$(classify "$TUNE_RE" <<<'crates/tune/src/lora.rs')"

if [ "$FAILED" -eq 0 ]; then
  echo "PASS: engine/tune change classifier behaves as specified"
else
  echo "FAILURES above"
fi
exit "$FAILED"

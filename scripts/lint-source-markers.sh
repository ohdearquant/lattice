#!/bin/sh
# Syntactic provenance lint (issue #818): every comment-leading marker
# (TODO, FIXME, HACK, or XXX) must carry a same-line numeric issue reference
# in the form `#N` or `latticeN` (e.g. `TODO(#123): ...` or
# `TODO(lattice#123): ...`).
#
# This is a deterministic, offline source lint. It does NOT call the GitHub
# API and cannot tell whether a referenced issue is open, closed, or
# transferred -- it only proves the marker carries *a* numeric reference.
#
# Exit codes follow explicit ripgrep semantics, not implicit pass-through:
#   0 = clean (no violations)
#   1 = violations found (printed to stdout)
#   2 = lint failure (ripgrep itself errored) -- never a silent pass
#
# Usage:
#   scripts/lint-source-markers.sh              # scan crates/ apps/ scripts/
#   scripts/lint-source-markers.sh --selftest    # run the fixture self-test

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Comment-leading marker: the marker token must directly follow the
# comment-opening syntax (mod whitespace), not merely appear somewhere in a
# comment. This is what keeps prose like "// see TODO handling above" or a
# `\uXXXX` escape sequence from tripping the lint -- `\bXXX\b` never matches
# inside `\uXXXX` (four X's; no word boundary between the first three and
# the fourth), and free-floating prose mentions of the word are not
# comment-leading.
#
# Rust: `//` line comments (also matches `///` and `//!` doc comments, since
# both contain `//` as a substring).
# Shell/Python: `#` comments.
MARKER_WORDS='TODO|FIXME|HACK|XXX'
RUST_MARKER_RE="//[[:space:]]*($MARKER_WORDS)\\b"
HASH_MARKER_RE="#[[:space:]]*($MARKER_WORDS)\\b"

# Same-line numeric issue reference: `#123` or `lattice#123`.
ISSUE_REF_RE='(^|[^A-Za-z0-9_])(lattice)?#[0-9]+'

# lint_paths <label> <marker-regex> <glob> <path...>
#
# Finds comment-leading marker lines under the given path(s)/glob, then
# filters out lines that already carry a same-line issue reference. Anything
# left over is a violation. Appends violations (prefixed with file:line) to
# $VIOLATIONS_FILE.
lint_paths() {
    label="$1"
    marker_re="$2"
    glob="$3"
    shift 3
    paths="$*"

    # $paths is an intentional word-split list of directory names.
    # shellcheck disable=SC2086
    matches="$(rg -n --no-heading -e "$marker_re" -g "$glob" $paths 2>"$RG_ERR_FILE")"
    rc=$?

    if [ "$rc" -gt 1 ]; then
        echo "lint-source-markers: ripgrep failed scanning $label (exit $rc)" >&2
        cat "$RG_ERR_FILE" >&2
        return 2
    fi

    if [ "$rc" -eq 1 ] || [ -z "$matches" ]; then
        # No comment-leading markers at all under this path -- clean.
        return 0
    fi

    # rc == 0: at least one comment-leading marker line. Keep only the ones
    # that lack a same-line issue reference.
    offenders="$(printf '%s\n' "$matches" | grep -vE "$ISSUE_REF_RE" || true)"
    if [ -n "$offenders" ]; then
        printf '%s\n' "$offenders" >> "$VIOLATIONS_FILE"
    fi
    return 0
}

run_lint() {
    scan_root="${1:-$REPO_ROOT}"
    tmp_violations="$(mktemp)"
    tmp_rg_err="$(mktemp)"
    VIOLATIONS_FILE="$tmp_violations"
    RG_ERR_FILE="$tmp_rg_err"
    status=0

    (
        cd "$scan_root" || exit 2
        lint_paths "Rust" "$RUST_MARKER_RE" '*.rs' crates apps scripts
        rc1=$?
        lint_paths "shell" "$HASH_MARKER_RE" '*.sh' crates apps scripts
        rc2=$?
        lint_paths "Python" "$HASH_MARKER_RE" '*.py' crates apps scripts
        rc3=$?
        if [ "$rc1" -eq 2 ] || [ "$rc2" -eq 2 ] || [ "$rc3" -eq 2 ]; then
            exit 2
        fi
        exit 0
    )
    status=$?

    if [ "$status" -eq 2 ]; then
        rm -f "$tmp_violations" "$tmp_rg_err"
        return 2
    fi

    if [ -s "$tmp_violations" ]; then
        echo "lint-source-markers: found TODO/FIXME/HACK/XXX markers without a same-line issue reference (#N or latticeN):" >&2
        sort -u "$tmp_violations" >&2
        rm -f "$tmp_violations" "$tmp_rg_err"
        return 1
    fi

    rm -f "$tmp_violations" "$tmp_rg_err"
    return 0
}

selftest() {
    sb="$(mktemp -d)/repo"
    mkdir -p "$sb/crates" "$sb/apps" "$sb/scripts"
    fail=0

    check() {  # $1 = description, $2 = expected rc, $3 = actual rc
        desc="$1"; expected="$2"; actual="$3"
        if [ "$expected" -ne "$actual" ]; then
            echo "FAIL: $desc (expected rc=$expected, got rc=$actual)" >&2
            fail=1
        else
            echo "ok: $desc (rc=$actual)"
        fi
    }

    reset_tree() {
        rm -rf "$sb/crates" "$sb/apps" "$sb/scripts"
        mkdir -p "$sb/crates" "$sb/apps" "$sb/scripts"
    }

    # 1. Clean tree -> rc 0.
    reset_tree
    cat > "$sb/crates/clean.rs" <<'EOF'
fn main() {
    // nothing to see here
}
EOF
    run_lint "$sb"
    check "clean tree" 0 "$?"

    # 2. Valid TODO(#1) -> rc 0.
    reset_tree
    cat > "$sb/crates/valid_hash.rs" <<'EOF'
fn main() {
    // TODO(#1): finish this
}
EOF
    run_lint "$sb"
    check "valid TODO(#1)" 0 "$?"

    # 3. Valid TODO(lattice#1) -> rc 0.
    reset_tree
    cat > "$sb/crates/valid_lattice.rs" <<'EOF'
fn main() {
    // TODO(lattice#1): finish this
}
EOF
    run_lint "$sb"
    check "valid TODO(lattice#1)" 0 "$?"

    # 4. Invalid milestone-name marker (no numeric ref) -> rc 1.
    reset_tree
    cat > "$sb/crates/milestone.rs" <<'EOF'
fn main() {
    // TODO(i2): wire this up later
}
EOF
    run_lint "$sb"
    check "milestone-name marker TODO(i2) flagged" 1 "$?"

    # 5. \uXXXX escape sequence must NOT trigger.
    reset_tree
    cat > "$sb/scripts/escape.py" <<'EOF'
# encode as \uXXXX for the placeholder
value = 1
EOF
    run_lint "$sb"
    check "\\uXXXX escape not flagged" 0 "$?"

    # 6. Prose mentioning "issue #123" that is not a marker -> not flagged.
    reset_tree
    cat > "$sb/scripts/prose.sh" <<'EOF'
#!/bin/sh
# see issue #123 for background, nothing actionable here
echo hi
EOF
    run_lint "$sb"
    check "non-marker prose mentioning issue #123 not flagged" 0 "$?"

    # 7. ripgrep-error case surfaces as a lint failure, not a silent pass.
    reset_tree
    echo "not really a repo" > "$sb/not_a_dir_marker"
    rm -rf "$sb/crates"  # crates/ missing entirely -> ripgrep no-such-path error
    run_lint "$sb"
    check "missing scan path surfaces as failure" 2 "$?"

    rm -rf "$(dirname "$sb")"

    if [ "$fail" -ne 0 ]; then
        echo "lint-source-markers selftest: FAILED" >&2
        exit 1
    fi
    echo "lint-source-markers selftest: all checks passed"
}

if [ "${1:-}" = "--selftest" ]; then
    selftest
    exit $?
fi

run_lint "$REPO_ROOT"
rc=$?
if [ "$rc" -eq 0 ]; then
    echo "lint-source-markers: clean"
fi
exit "$rc"

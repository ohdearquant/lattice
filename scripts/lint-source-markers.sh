#!/bin/sh
# Syntactic provenance lint (issue #818): every comment-leading marker
# (TODO, FIXME, HACK, or XXX) must carry a same-line numeric issue reference
# in the form `#N` or `lattice#N` (e.g. `TODO(#123): ...` or
# `TODO(lattice#123): ...`).
#
# This is a deterministic, offline source lint. It does NOT call the GitHub
# API and cannot tell whether a referenced issue is open, closed, or
# transferred -- it only proves the marker carries *a* numeric reference.
#
# Known approximation (line-based, not a language parser): the lint matches
# comment-OPENING syntax textually, not an actual tokenizer/AST. It does not
# know about Rust/shell/Python string literals, so a marker-shaped substring
# inside a string literal -- e.g. a Rust string literal containing `// ` plus
# a marker word, or a shell/Python string literal containing the hash-comment
# opener plus a marker word -- is indistinguishable from a real comment and
# WILL be flagged. This trade-off avoids a per-language parser
# dependency and, verified against the current tree, produces zero false
# positives from string literals today. If a legitimate marker-shaped string
# literal is ever added, suppress it the same way as any other offender: add
# a same-line issue reference, or break the textual pattern (e.g. string
# concatenation).
#
# Exit codes follow explicit ripgrep semantics, not implicit pass-through:
#   0 = clean (no violations)
#   1 = violations found (printed to stdout)
#   2 = lint failure (ripgrep or the offender filter itself errored) --
#       never a silent pass
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
# Rust: `//` line comments (also `///` and `//!` doc comments, via the
# optional `[!/]` after the opening `//`), `/* ... */` block comments opened
# on the marker's own line, and `*`-prefixed continuation lines inside a
# multiline block comment (the common `/* ...\n * TODO ...\n */` leader
# style). The block-comment forms are recognized positionally (comment
# opener or line-leading `*`), not via a full block-comment-interior parse,
# which is why a line-leading `*` outside an actual block comment (rare in
# this codebase's style) is also treated as comment-leading -- consistent
# with the lint's general line-based-heuristic trade-off (see header).
# Shell/Python: `#` comments.
MARKER_WORDS='TODO|FIXME|HACK|XXX'
RUST_MARKER_RE="(//[!/]?[[:space:]]*|/\\*[[:space:]]*|^[[:space:]]*\\*[[:space:]]*)($MARKER_WORDS)\\b"
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
    # that lack a same-line issue reference. grep's exit status is checked
    # explicitly rather than pass-through-normalized: rc=0 means offenders
    # exist, rc=1 means every line carried a reference (no offenders, not a
    # failure), and rc>1 is a filter-stage failure that must not be silently
    # treated as "clean" (a `|| true` here would normalize rc>1 to success).
    offenders="$(printf '%s\n' "$matches" | grep -vE "$ISSUE_REF_RE")"
    filter_rc=$?
    if [ "$filter_rc" -gt 1 ]; then
        echo "lint-source-markers: offender filter failed scanning $label (exit $filter_rc)" >&2
        return 2
    fi
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
        echo "lint-source-markers: found TODO/FIXME/HACK/XXX markers without a same-line issue reference (#N or lattice#N):" >&2
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

    # 8. Offender-filter failure (grep exit >1) must surface as a lint
    # failure, never a silent "clean" pass. Shadows `grep` on PATH with a
    # deterministic rc=2 stub for the duration of this one run_lint call.
    reset_tree
    cat > "$sb/crates/valid_hash.rs" <<'EOF'
fn main() {
    // TODO(#1): finish this
}
EOF
    fake_bin_dir="$(mktemp -d)"
    cat > "$fake_bin_dir/grep" <<'EOF'
#!/bin/sh
echo "injected grep failure" >&2
exit 2
EOF
    chmod +x "$fake_bin_dir/grep"
    old_path="$PATH"
    PATH="$fake_bin_dir:$PATH"
    export PATH
    run_lint "$sb"
    filter_fail_rc=$?
    PATH="$old_path"
    export PATH
    rm -rf "$fake_bin_dir"
    check "offender-filter failure (grep rc=2) surfaces as lint failure" 2 "$filter_fail_rc"

    # 9. `/* FIXME */` single-line block comment without a reference -> rc 1.
    reset_tree
    cat > "$sb/crates/block_comment.rs" <<'EOF'
fn main() {
    /* FIXME: block comment marker */
}
EOF
    run_lint "$sb"
    check "/* FIXME */ block comment flagged" 1 "$?"

    # 10. `//! TODO` inner doc comment without a reference -> rc 1.
    reset_tree
    cat > "$sb/crates/inner_doc.rs" <<'EOF'
//! TODO: inner doc comment marker
fn main() {}
EOF
    run_lint "$sb"
    check "//! TODO inner doc comment flagged" 1 "$?"

    # 11. `* TODO` multiline block-comment leader line without a reference
    # -> rc 1.
    reset_tree
    cat > "$sb/crates/block_leader.rs" <<'EOF'
/*
 * TODO: multiline block comment leader
 */
fn main() {}
EOF
    run_lint "$sb"
    check "* TODO block-comment leader line flagged" 1 "$?"

    # 12. Referenced forms of the newly covered comment shapes stay clean,
    # confirming the new alternatives don't just flag unconditionally.
    reset_tree
    cat > "$sb/crates/block_comment_ref.rs" <<'EOF'
fn main() {
    /* FIXME(#1): block comment marker */
}
EOF
    run_lint "$sb"
    check "/* FIXME(#1) */ block comment not flagged" 0 "$?"

    reset_tree
    cat > "$sb/crates/inner_doc_ref.rs" <<'EOF'
//! TODO(#1): inner doc comment marker
fn main() {}
EOF
    run_lint "$sb"
    check "//! TODO(#1) inner doc comment not flagged" 0 "$?"

    # 13. Known approximation: marker-shaped text inside string literals is
    # flagged (documented in the script header, not a bug). These are
    # KNOWN-flagged fixtures, not a request for parser-level precision.
    reset_tree
    cat > "$sb/crates/string_literal.rs" <<'EOF'
fn main() {
    let s = "// TODO: this is inside a string literal, not a comment";
    let _ = s;
}
EOF
    run_lint "$sb"
    check "KNOWN approximation: Rust string literal containing // TODO flagged" 1 "$?"

    # These two fixtures build their hash-comment-shaped marker text via
    # variable interpolation (not a literal `#` immediately before the
    # marker word) so this script's own source doesn't self-match when it
    # scans scripts/*.sh under its own shell-marker regex.
    reset_tree
    hash_char='#'
    cat > "$sb/scripts/string_literal.sh" <<EOF
#!/bin/sh
echo "${hash_char} TODO: this is inside a shell string literal, not a comment"
EOF
    run_lint "$sb"
    check "KNOWN approximation: shell string literal containing a hash+TODO marker flagged" 1 "$?"

    reset_tree
    cat > "$sb/scripts/string_literal.py" <<EOF
text = "${hash_char} FIXME: this is inside a Python string literal, not a comment"
EOF
    run_lint "$sb"
    check "KNOWN approximation: Python string literal containing a hash+FIXME marker flagged" 1 "$?"

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

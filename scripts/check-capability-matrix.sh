#!/bin/sh
# Cross-checks docs/capability-matrix.md's Fixture manifest section against
# the actual #[test]/#[tokio::test] fns in the three serve-surface binaries
# plus the shared Metal worker module they both route through (#654, #832).
# A row that cites a fixture ID whose test function has been renamed,
# removed, or never existed fails this check closed instead of silently
# going stale.
#
# Wired into `make lint-docs`. See docs/capability-matrix.md's "Fixture
# manifest (#654)" section for the human-readable row-by-row mapping this
# script verifies mechanically.
#
# POSIX sh only (targets macOS's /bin/sh, bash 3.2): no arrays, no
# associative arrays, no `mapfile`, no `local`.
set -eu

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

lattice_rs="$repo_root/crates/inference/src/bin/lattice.rs"
lattice_serve_rs="$repo_root/crates/inference/src/bin/lattice_serve.rs"
chat_metal_rs="$repo_root/crates/inference/src/bin/chat_metal.rs"
# Shared Metal worker owner (#832): the cancellation/window-check fixtures
# that used to live in lattice_serve.rs's own #[cfg(test)] mod now live here
# instead, ported verbatim and shared by both `lattice` and `lattice_serve`.
metal_worker_rs="$repo_root/crates/inference/src/serve/metal_worker.rs"

# Fixture IDs always start with one of these prefixes (the naming convention
# every `#[test] fn` added for #654, plus the pre-existing per-surface test
# families it reuses, already follows). Restricting to these prefixes -- vs.
# matching any backtick-quoted snake_case token -- avoids false positives on
# the many field-name/type-name backtick references in surrounding prose
# (`max_tokens`, `response_format`, `model_max_context`, etc.), which are not
# fixture IDs and have no `fn` to find.
prefixes='cm_ validate_ render_prompt_ reject_unsupported_ parse_stop_strings_ chat_completions_ build_cfg_ model_context_ check_prompt_fits_window parse_chat_req_ message_content_ message_role_ queued_job_ running_job_ generation_failure_'

# Every #[test]/#[tokio::test] fn name in a file, one per line. Only fns
# immediately preceded by one of those attributes (skipping the `async`
# keyword on `#[tokio::test]` fns) count -- NOT every `fn` whose name happens
# to match a fixture prefix. That distinction matters: `validate_max_tokens`
# is the implementation under test, `validate_max_tokens_rejects_zero` is the
# `#[test]` fixture for it, and a bare `grep "fn $tok("` cannot tell them
# apart (nor can it tell a fixture ID from a stray comment or doc-string
# mentioning the same name).
extract_test_fn_names() {
    grep -A1 -E '#\[test\]|#\[tokio::test\]' "$1" 2>/dev/null \
        | grep -E '^[[:space:]]*(async )?fn [a-zA-Z0-9_]*\(' \
        | sed -E 's/^[[:space:]]*(async )?fn ([a-zA-Z0-9_]*)\(.*/\2/'
}

all_test_fns="$(
    {
        extract_test_fn_names "$lattice_rs"
        extract_test_fn_names "$lattice_serve_rs"
        extract_test_fn_names "$chat_metal_rs"
        extract_test_fn_names "$metal_worker_rs"
    } | sort -u
)"

is_real_test_fn() {
    printf '%s\n' "$all_test_fns" | grep -qx "$1"
}

# Runs the forward check (every cited fixture ID resolves to a real test fn)
# against $1, a path to a capability-matrix-shaped markdown file. Prints
# diagnostics unless $2 is "1" (quiet, used by --selftest so a deliberately
# bogus fixture doesn't spam stderr on every normal run). Sets $tokens and
# $cited as a side effect for the reverse check below; returns 0/1 via exit
# status, not a printed value (no bash-4 local, no subshell-return capture).
check_doc() {
    doc="$1"
    quiet="$2"

    if [ ! -f "$doc" ]; then
        echo "check-capability-matrix: $doc not found" >&2
        return 1
    fi

    # Extract the Fixture manifest section (from its heading to the next
    # `## ` heading) so we never scan the rest of the doc (file:line
    # citations there use the same backtick syntax for unrelated things).
    section="$(awk '/^## Fixture manifest \(#654\)/{flag=1} flag && /^## See also/{flag=0} flag' "$doc")"

    if [ -z "$section" ]; then
        [ "$quiet" = "1" ] || echo "check-capability-matrix: 'Fixture manifest (#654)' section not found in $doc" >&2
        return 1
    fi

    # Every backtick-quoted token in that section's table rows (lines
    # starting with `|`), one per line. Fixture IDs are only ever cited in
    # table cells; the section's own prose paragraphs (e.g. "see
    # `validate_chat_request` in `lattice.rs`") reference real, non-test
    # implementation functions by the same backtick style and would
    # otherwise false-positive against the `validate_`/`chat_completions_`/
    # etc. prefixes below.
    tokens="$(printf '%s\n' "$section" | grep '^|' | grep -o '`[a-zA-Z0-9_]*`' | tr -d '`' | sort -u)"

    fail=0
    cited=""

    for tok in $tokens; do
        matched_prefix=0
        for p in $prefixes; do
            case "$tok" in
            "$p"*) matched_prefix=1 ;;
            esac
        done
        [ "$matched_prefix" -eq 1 ] || continue

        cited="$cited $tok"

        if ! is_real_test_fn "$tok"; then
            [ "$quiet" = "1" ] || echo "check-capability-matrix: fixture '$tok' is cited in $doc but no #[test]/#[tokio::test] fn named '$tok' exists in lattice.rs, lattice_serve.rs, chat_metal.rs, or serve/metal_worker.rs" >&2
            fail=1
        fi
    done

    if [ -z "$(printf '%s' "$cited" | tr -d '[:space:]')" ]; then
        [ "$quiet" = "1" ] || echo "check-capability-matrix: no fixture IDs found in the Fixture manifest section -- check the prefix list or the doc's table formatting" >&2
        fail=1
    fi

    [ "$fail" -eq 0 ]
}

if [ "${1:-}" = "--selftest" ]; then
    # Prove the check actually fails closed: take the real doc, inject one
    # more fixture-prefixed backtick token that is guaranteed not to resolve
    # to any real #[test] fn, and confirm check_doc rejects it. Guards
    # against a future edit accidentally loosening the match (e.g. back to a
    # bare `grep "fn $tok("`, which a bogus name-only citation would still
    # pass if a same-named non-test fn or comment existed).
    doc="$repo_root/docs/capability-matrix.md"
    bogus="cm_this_fixture_does_not_exist_1234"

    if is_real_test_fn "$bogus"; then
        echo "check-capability-matrix: SELFTEST SETUP FAILED -- '$bogus' unexpectedly matches a real #[test] fn; pick a different bogus name" >&2
        exit 1
    fi

    scratch="$(mktemp)"
    trap 'rm -f "$scratch"' EXIT

    # The injected line must itself look like a table row (start with `|`):
    # tokens are only scanned from table rows within the section (see
    # check_doc), so a bogus citation dropped into surrounding prose would
    # silently not exercise the check at all.
    awk -v bogus="$bogus" '
        { print }
        /^## Fixture manifest \(#654\)/ { print "\n| selftest-injected row | `" bogus "` |\n" }
    ' "$doc" >"$scratch"

    if check_doc "$scratch" 1; then
        echo "check-capability-matrix: SELFTEST FAILED -- a bogus fixture ID ('$bogus') did not cause the check to fail" >&2
        exit 1
    fi

    echo "check-capability-matrix: selftest OK -- a bogus fixture ID correctly fails the check"
    exit 0
fi

doc="$repo_root/docs/capability-matrix.md"

if ! check_doc "$doc" 0; then
    exit 1
fi

# Reverse direction: every capability-matrix #[test] fn that exists in the
# three binaries and matches one of the known fixture-naming prefixes should
# be cited somewhere in the doc, or a fixture can silently stop being tracked
# after a rename. Report-only (not yet a hard fail) so pre-existing fixture
# families added before #654's Fixture manifest section don't need every
# single test enumerated by name on day one. Reuses $tokens, set by the
# check_doc call above.
for f in "$lattice_rs" "$lattice_serve_rs" "$chat_metal_rs" "$metal_worker_rs"; do
    extract_test_fn_names "$f" | while read -r name; do
        matched_prefix=0
        for p in $prefixes; do
            case "$name" in
            "$p"*) matched_prefix=1 ;;
            esac
        done
        [ "$matched_prefix" -eq 1 ] || continue
        if ! printf '%s\n' "$tokens" | grep -qx "$name"; then
            echo "check-capability-matrix: NOTE: '$name' in $(basename "$f") matches a fixture-naming prefix but is not cited in docs/capability-matrix.md" >&2
        fi
    done
done

echo "check-capability-matrix: all cited fixtures resolved to a real #[test]/#[tokio::test] fn"

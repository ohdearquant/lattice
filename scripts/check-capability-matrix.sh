#!/bin/sh
# Cross-checks docs/capability-matrix.md's Fixture manifest section against
# the actual #[test] fns in the three serve-surface binaries (#654). A row
# that cites a fixture ID whose test function has been renamed, removed, or
# never existed fails this check closed instead of silently going stale.
#
# Wired into `make lint-docs`. See docs/capability-matrix.md's "Fixture
# manifest (#654)" section for the human-readable row-by-row mapping this
# script verifies mechanically.
set -eu

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
doc="$repo_root/docs/capability-matrix.md"

lattice_rs="$repo_root/crates/inference/src/bin/lattice.rs"
lattice_serve_rs="$repo_root/crates/inference/src/bin/lattice_serve.rs"
chat_metal_rs="$repo_root/crates/inference/src/bin/chat_metal.rs"

if [ ! -f "$doc" ]; then
    echo "check-capability-matrix: $doc not found" >&2
    exit 1
fi

# Fixture IDs always start with one of these prefixes (the naming convention
# every `#[test] fn` added for #654, plus the pre-existing per-surface test
# families it reuses, already follows). Restricting to these prefixes -- vs.
# matching any backtick-quoted snake_case token -- avoids false positives on
# the many field-name/type-name backtick references in surrounding prose
# (`max_tokens`, `response_format`, `model_max_context`, etc.), which are not
# fixture IDs and have no `fn` to find.
prefixes='cm_ validate_ render_prompt_ reject_unsupported_ parse_stop_strings_ chat_completions_ build_cfg_ model_context_ check_prompt_fits_window parse_chat_req_ message_content_ message_role_ queued_job_ running_job_ generation_failure_'

# Extract the Fixture manifest section (from its heading to the next `## `
# heading) so we never scan the rest of the doc (file:line citations there
# use the same backtick syntax for unrelated things).
section="$(awk '/^## Fixture manifest \(#654\)/{flag=1} flag && /^## See also/{flag=0} flag' "$doc")"

if [ -z "$section" ]; then
    echo "check-capability-matrix: 'Fixture manifest (#654)' section not found in $doc" >&2
    exit 1
fi

# Every backtick-quoted token in that section, one per line.
tokens="$(printf '%s\n' "$section" | grep -o '`[a-zA-Z0-9_]*`' | tr -d '`' | sort -u)"

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

    if ! grep -q "fn $tok(" "$lattice_rs" "$lattice_serve_rs" "$chat_metal_rs" 2>/dev/null; then
        echo "check-capability-matrix: fixture '$tok' is cited in docs/capability-matrix.md but no 'fn $tok(' exists in lattice.rs, lattice_serve.rs, or chat_metal.rs" >&2
        fail=1
    fi
done

if [ -z "$(printf '%s' "$cited" | tr -d '[:space:]')" ]; then
    echo "check-capability-matrix: no fixture IDs found in the Fixture manifest section -- check the prefix list or the doc's table formatting" >&2
    fail=1
fi

# Reverse direction: every capability-matrix #[test] fn that exists in the
# three binaries and matches one of the known fixture-naming prefixes should
# be cited somewhere in the doc, or a fixture can silently stop being tracked
# after a rename. Report-only (not yet a hard fail) so pre-existing fixture
# families added before #654's Fixture manifest section don't need every
# single test enumerated by name on day one.
for f in "$lattice_rs" "$lattice_serve_rs" "$chat_metal_rs"; do
    # Only fns immediately preceded by `#[test]` (skipping the `async` keyword
    # on `#[tokio::test]` fns) -- not every helper whose name happens to share
    # a fixture prefix (e.g. `validate_max_tokens`, the implementation `fn`
    # itself, vs. `validate_max_tokens_rejects_zero`, the `#[test]` for it).
    grep -A1 -E '#\[test\]|#\[tokio::test\]' "$f" \
        | grep -E '^\s*(async )?fn [a-zA-Z0-9_]*\(' \
        | sed -E 's/^\s*(async )?fn ([a-zA-Z0-9_]*)\(.*/\2/' | while read -r name; do
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

if [ "$fail" -ne 0 ]; then
    exit 1
fi

echo "check-capability-matrix: all cited fixtures resolved to a real #[test] fn"

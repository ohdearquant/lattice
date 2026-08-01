#!/bin/sh
# Reject developer-specific home paths from tracked documentation, Rust,
# manifests, scripts, workflows, and JSON.
#
# Swift remains outside this check until its existing preview literals are
# normalized under #1102.
#
# The historical benchmark evidence directory is intentionally excluded: those
# immutable reports preserve the exact commands and paths used for their runs.
#
# Exit status:
#   0 = clean
#   1 = developer-specific paths found
#   2 = checker failure
#
# Usage:
#   scripts/lint-absolute-paths.sh
#   scripts/lint-absolute-paths.sh --selftest

set -u

unset GIT_INDEX_FILE GIT_DIR GIT_WORK_TREE

SCRIPT_DIR=$(CDPATH='' cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH='' cd "$SCRIPT_DIR/.." && pwd)
HOME_DIR_NAMES='Users|home'
ABSOLUTE_PATH_RE="/(${HOME_DIR_NAMES})/[^/[:space:]]+/"

run_check() {
    check_root=$1
    matches_file=$(mktemp "${TMPDIR:-/tmp}/lattice-absolute-path-matches.XXXXXX")
    matches_rc=$?
    if [ "$matches_rc" -ne 0 ] || [ -z "$matches_file" ]; then
        echo "lint-absolute-paths: failed to create matches tempfile" >&2
        return 2
    fi

    errors_file=$(mktemp "${TMPDIR:-/tmp}/lattice-absolute-path-errors.XXXXXX")
    errors_rc=$?
    if [ "$errors_rc" -ne 0 ] || [ -z "$errors_file" ]; then
        echo "lint-absolute-paths: failed to create errors tempfile" >&2
        rm -f "$matches_file"
        return 2
    fi

    git -C "$check_root" grep -n -I -E "$ABSOLUTE_PATH_RE" -- \
        '*.md' \
        '*.rs' \
        '*.toml' \
        '*.sh' \
        '*.py' \
        '*.yml' \
        '*.yaml' \
        '*.json' \
        ':(exclude)scripts/bench_evidence/**' \
        >"$matches_file" 2>"$errors_file"
    grep_rc=$?

    case "$grep_rc" in
        0)
            echo "lint-absolute-paths: found developer-specific home paths:" >&2
            if ! cat "$matches_file" >&2; then
                echo "lint-absolute-paths: failed to report findings" >&2
                rm -f "$matches_file" "$errors_file"
                return 2
            fi
            rm -f "$matches_file" "$errors_file"
            return 1
            ;;
        1)
            if [ -s "$errors_file" ]; then
                echo "lint-absolute-paths: git grep reported an error:" >&2
                cat "$errors_file" >&2
                rm -f "$matches_file" "$errors_file"
                return 2
            fi
            rm -f "$matches_file" "$errors_file"
            return 0
            ;;
        *)
            echo "lint-absolute-paths: git grep failed (exit $grep_rc):" >&2
            cat "$errors_file" >&2
            rm -f "$matches_file" "$errors_file"
            return 2
            ;;
    esac
}

selftest() {
    sandbox=$(mktemp -d "${TMPDIR:-/tmp}/lattice-absolute-path-selftest.XXXXXX")
    sandbox_rc=$?
    case "$sandbox_rc:$sandbox" in
        0:"${TMPDIR:-/tmp}"/lattice-absolute-path-selftest.*) ;;
        *)
            echo "lint-absolute-paths selftest: failed to create safe sandbox" >&2
            return 2
            ;;
    esac
    trap 'rm -rf "$sandbox"' 0 1 2 3 15

    mkdir -p "$sandbox/docs" "$sandbox/crates/demo/src" "$sandbox/scripts/bench_evidence/run"
    printf 'Use $%s/model-name.\n' 'LATTICE_MODEL_CACHE' >"$sandbox/docs/clean.md"
    printf '%s\n' 'fn main() {}' >"$sandbox/crates/demo/src/main.rs"
    printf '/%s/%s/projects/archive/model\n' 'Users' 'archived-runner' \
        >"$sandbox/scripts/bench_evidence/run/report.json"
    git -C "$sandbox" init -q
    git -C "$sandbox" add docs crates scripts

    clean_output="$sandbox/clean.out"
    run_check "$sandbox" >"$clean_output" 2>&1
    clean_rc=$?
    if [ "$clean_rc" -ne 0 ]; then
        echo "lint-absolute-paths selftest: clean tree returned $clean_rc" >&2
        cat "$clean_output" >&2
        return 1
    fi

    printf '/%s/%s/projects/demo\n' 'Users' 'developer' >"$sandbox/docs/dirty.md"
    printf '// /%s/%s/src/demo\n' 'home' 'developer' \
        >"$sandbox/crates/demo/src/dirty.rs"
    git -C "$sandbox" add docs/dirty.md crates/demo/src/dirty.rs
    dirty_output="$sandbox/dirty.out"
    run_check "$sandbox" >"$dirty_output" 2>&1
    dirty_rc=$?
    if [ "$dirty_rc" -ne 1 ]; then
        echo "lint-absolute-paths selftest: dirty tree returned $dirty_rc, expected 1" >&2
        cat "$dirty_output" >&2
        return 1
    fi
    if ! grep -F 'docs/dirty.md:1:' "$dirty_output" >/dev/null; then
        echo "lint-absolute-paths selftest: macOS-style dirty finding was not reported" >&2
        cat "$dirty_output" >&2
        return 1
    fi
    if ! grep -F 'crates/demo/src/dirty.rs:1:' "$dirty_output" >/dev/null; then
        echo "lint-absolute-paths selftest: Linux-style dirty finding was not reported" >&2
        cat "$dirty_output" >&2
        return 1
    fi

    missing_output="$sandbox/missing.out"
    run_check "$sandbox/not-a-repository" >"$missing_output" 2>&1
    missing_rc=$?
    if [ "$missing_rc" -ne 2 ]; then
        echo "lint-absolute-paths selftest: discovery failure returned $missing_rc, expected 2" >&2
        cat "$missing_output" >&2
        return 1
    fi

    echo "lint-absolute-paths selftest: clean, dirty, and failure fixtures passed"
}

case "${1:-}" in
    "")
        run_check "$REPO_ROOT"
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "lint-absolute-paths: clean"
        fi
        exit "$rc"
        ;;
    --selftest)
        selftest
        exit $?
        ;;
    *)
        echo "usage: $0 [--selftest]" >&2
        exit 64
        ;;
esac

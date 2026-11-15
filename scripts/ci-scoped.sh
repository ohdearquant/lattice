#!/bin/sh
# The checks of scripts/ci.sh, scoped to what a change can reach. Used for pull
# requests into an integration branch; the integration branch's own pull request
# into main runs the full scripts/ci.sh.
#
# Usage: scripts/ci-scoped.sh <base> [<head>]   (head defaults to HEAD)
#
# Always run: fmt, the fmt-coverage lint and the doc lint (cheap, repo-wide).
# Scoped by scripts/affected_packages.py: clippy and tests run on the packages
# that own a changed file plus their workspace dependents; the lattice-inference
# source-contract tests run whenever anything builds; the tokenizer parity gate
# runs when lattice-inference or lattice-embed is in scope. No release build.
set -e

base=${1:?usage: scripts/ci-scoped.sh <base> [<head>]}
head=${2:-HEAD}

: "${LATTICE_REQUIRE_FIXTURES:=1}"
export LATTICE_REQUIRE_FIXTURES

echo "=== Format Check ==="
cargo fmt --all -- --check

echo "=== Rust Format Coverage ==="
./scripts/lint-rust-fmt-coverage.sh --selftest
./scripts/lint-rust-fmt-coverage.sh

echo "=== Doc Lint ==="
./scripts/lint-docs.sh

echo "=== Scope ==="
scope=$(python3 scripts/affected_packages.py "$base" "$head")
echo "scope: $scope"
if [ "$scope" = "NONE" ]; then
    echo "No Rust package is reached by this change; clippy and tests are skipped."
    echo "=== Scoped CI Passed ==="
    exit 0
fi

echo "=== Clippy ==="
# shellcheck disable=SC2086 # $scope is a list of cargo arguments
cargo clippy $scope -- -D warnings

echo "=== Tests ==="
# shellcheck disable=SC2086
LATTICE_GEMMA4_GATE_SKIP=1 cargo test $scope

in_scope() {
    [ "$scope" = "--workspace" ] && return 0
    case " $scope " in *" -p $1 "*) return 0 ;; esac
    return 1
}

if ! in_scope lattice-inference; then
    echo "=== Source Contract Tests ==="
    # shellcheck disable=SC2046
    cargo test -p lattice-inference $(python3 scripts/affected_packages.py --contract-tests)
fi

if in_scope lattice-model-bert; then
    cargo test -p lattice-model-bert --example embedding_pair_producer
fi

if in_scope lattice-inference || in_scope lattice-embed; then
    echo "=== Tokenizer Parity Gate ==="
    tmp_tokenizer_log="$(mktemp)"
    tmp_cmd_log="$(mktemp)"
    trap 'rm -f "$tmp_tokenizer_log" "$tmp_cmd_log"' EXIT
    for gate in "lattice-inference audit_tokenizer_parity" "lattice-embed tokenizer_parity_e2e"; do
        # shellcheck disable=SC2086 # split "package test" into two arguments
        set -- $gate
        if ! cargo test -p "$1" --test "$2" -- --nocapture >"$tmp_cmd_log" 2>&1; then
            cat "$tmp_cmd_log"
            exit 1
        fi
        cat "$tmp_cmd_log"
        cat "$tmp_cmd_log" >> "$tmp_tokenizer_log"
    done
    if grep -E 'SKIP|LATTICE_.*SKIPPED' "$tmp_tokenizer_log"; then
        echo "Tokenizer parity gate: unexpected skip detected — tokenizer tests must all run" >&2
        exit 1
    fi
fi

echo "=== Scoped CI Passed ==="

#!/usr/bin/env bash
# Bracket the compressed size of each publishable crate against the crates.io
# upload limit, before `cargo publish` gets a chance to fail at the registry.
#
# Why this exists rather than `cargo package`: cargo cannot package a crate whose
# internal path dependencies are not yet live on the registry, so the only crate
# it can self-measure at release time is the leaf tier — the same gap that limits
# `make publish-dry`. `cargo package --list` has no such dependency, builds
# nothing, and names exactly the files that would be archived.
#
# Two of the outcomes below are refusals rather than results, and both exit
# non-zero. A failed `cargo package --list` is reported with cargo's own message
# (a dirty working tree is the usual cause, and `cargo publish` refuses the same
# tree). A listed file that is not on disk is what a measurement taken from the
# wrong directory looks like, and reporting a size then would understate it.
# Neither refusal is an absence of a problem.
#
# Usage:
#   scripts/package-size-check.sh [crate ...]     # defaults to all publishable crates
#
# Env:
#   PKG_SIZE_WARN_FRACTION   warn above this fraction of the limit (default 0.75)
#   PKG_SIZE_ALLOW_DIRTY     set to 1 to measure an uncommitted tree (local use only)
#
# Exit: 0 all crates under the limit, 1 any crate over it or any measurement refused.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/bench-python.sh
. "${REPO_ROOT}/scripts/lib/bench-python.sh"
PYTHON_BIN="$(bench_require_python3 "package-size-check.sh")" || exit 1
BRACKET="${REPO_ROOT}/scripts/lib/package-size-bracket.py"

CRATES=("$@")
if [ ${#CRATES[@]} -eq 0 ]; then
    CRATES=(lattice-fann lattice-transport lattice-inference lattice-embed lattice-tune)
fi

# Expanded below as ${DIRTY_FLAG[@]+"${DIRTY_FLAG[@]}"}: under `set -u`, bash 3.2
# (the macOS system bash) treats a plain "${arr[@]}" on an empty array as an
# unbound variable. That error aborted the command and was reported as a refusal,
# which is the same outcome this check produces for a genuinely dirty tree — so
# the guard read as working while measuring nothing.
DIRTY_FLAG=()
if [ "${PKG_SIZE_ALLOW_DIRTY:-0}" = "1" ]; then
    DIRTY_FLAG=(--allow-dirty)
fi

status=0
for crate in "${CRATES[@]}"; do
    crate_dir="$(cargo metadata --no-deps --format-version 1 --manifest-path "${REPO_ROOT}/Cargo.toml" \
        | "$PYTHON_BIN" -c 'import json, os, sys
name = sys.argv[1]
for pkg in json.load(sys.stdin)["packages"]:
    if pkg["name"] == name:
        print(os.path.dirname(pkg["manifest_path"]))
        break
else:
    sys.exit(f"package-size-check: no workspace member named {name}")' "$crate")"

    # `cargo package --list` prints paths relative to the crate directory, so the
    # measurement runs from there.
    list_file="$(mktemp)"
    err_file="$(mktemp)"
    if ! (cd "$crate_dir" && cargo package --list -p "$crate" ${DIRTY_FLAG[@]+"${DIRTY_FLAG[@]}"}) \
        >"$list_file" 2>"$err_file"; then
        echo "package-size-check: cargo package --list refused for ${crate}:" >&2
        cat "$err_file" >&2
        rm -f "$list_file" "$err_file"
        status=1
        continue
    fi

    (cd "$crate_dir" && "$PYTHON_BIN" "$BRACKET" "$crate" "$list_file" \
        --lockfile "${REPO_ROOT}/Cargo.lock") || status=1
    rm -f "$list_file" "$err_file"
done

if [ "$status" -ne 0 ]; then
    echo >&2
    echo "package-size-check: a crate is over the crates.io upload limit, or a measurement was refused." >&2
    echo "package-size-check: shrink the package with an \`exclude\` list in the crate manifest for" >&2
    echo "package-size-check: test-only fixtures, then re-run. A published crate cannot run its" >&2
    echo "package-size-check: integration tests, so their fixtures are dead weight in the archive." >&2
fi
exit "$status"

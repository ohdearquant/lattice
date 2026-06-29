#!/bin/sh
set -e

DRY_RUN=${1:-""}

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "=== Dry Run: Checking packages ==="
    FLAG="--dry-run"
else
    echo "=== Publishing to crates.io ==="
    FLAG=""
fi

# Publish order follows the internal dependency DAG (deps before dependents):
#   fann, transport   leaves, no internal deps
#     -> inference     depends on fann (via the `mixture` feature)
#       -> embed       depends on inference, transport
#       -> tune        depends on fann, inference
# Each tier waits for crates.io indexing before the next tier resolves it.
# (--dry-run validates only the leaf tier: cargo cannot dry-run a crate whose
#  internal deps are not yet live on the registry.)

echo "--- Tier 1: leaf crates (no internal deps) ---"
cargo publish -p lattice-fann $FLAG
cargo publish -p lattice-transport $FLAG

if [ -z "$DRY_RUN" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 2: inference (depends on fann) ---"
cargo publish -p lattice-inference $FLAG

if [ -z "$DRY_RUN" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 3: dependent crates (depend on inference) ---"
cargo publish -p lattice-embed $FLAG
cargo publish -p lattice-tune $FLAG

echo "=== Done ==="

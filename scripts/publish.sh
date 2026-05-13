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

echo "--- Leaf crates (no internal deps) ---"
cargo publish -p lattice-inference $FLAG
cargo publish -p lattice-fann $FLAG
cargo publish -p lattice-transport $FLAG

if [ -z "$DRY_RUN" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Dependent crates ---"
cargo publish -p lattice-embed $FLAG

if [ -z "$DRY_RUN" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

cargo publish -p lattice-tune $FLAG

echo "=== Done ==="

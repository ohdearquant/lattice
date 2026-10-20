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
#   fann, transport, inference-core   leaves, no internal deps
#     -> tokenizer                    depends on inference-core
#       -> inference                  depends on fann (via `mixture`), tokenizer
#         -> embed                    depends on inference, transport
#         -> tune                     depends on fann, inference
# Each tier waits for crates.io indexing before the next tier resolves it.
# (--dry-run validates only the leaf tier: cargo cannot dry-run a crate whose
#  internal deps are not yet live on the registry.)

# Every tier below uploads an archive the registry will reject outright if it
# exceeds 10 MiB, and a rejection in tier 2 leaves tier 1 already published and
# immutable. `cargo publish --dry-run` does not cover this: it can only validate
# the leaf tier, because a crate whose internal path dependencies are not yet
# live on the registry cannot be packaged at all. Measure all seven first.
echo "--- Pre-publish: package size against the crates.io upload limit ---"
"$(dirname "$0")/package-size-check.sh"

echo "--- Tier 1: leaf crates (no internal deps) ---"
cargo publish -p lattice-fann $FLAG
cargo publish -p lattice-transport $FLAG
cargo publish -p lattice-inference-core $FLAG

if [ -z "$DRY_RUN" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 2: tokenizer (depends on inference-core) ---"
cargo publish -p lattice-tokenizer $FLAG

if [ -z "$DRY_RUN" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 3: inference (depends on fann, inference-core, tokenizer) ---"
cargo publish -p lattice-inference $FLAG

if [ -z "$DRY_RUN" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 4: dependent crates (depend on inference) ---"
cargo publish -p lattice-embed $FLAG
cargo publish -p lattice-tune $FLAG

echo "=== Done ==="

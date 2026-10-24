#!/bin/sh
set -e

# Accept exactly no argument (publish) or --dry-run. Anything else exits before
# the first cargo call: a mistyped flag must never fall through to a real publish.
if [ "$#" -gt 1 ]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi
case "${1:-}" in
    "")
        echo "=== Publishing to crates.io ==="
        FLAG=""
        ;;
    --dry-run)
        echo "=== Dry Run: Checking packages ==="
        FLAG="--dry-run"
        ;;
    *)
        echo "usage: $0 [--dry-run]" >&2
        exit 2
        ;;
esac

# Publish order follows the internal dependency DAG (deps before dependents):
#   fann, transport, inference-core   leaves, no internal deps
#     -> tokenizer, grammar, weights  depend on inference-core
#       -> quant                      depends on inference-core, weights
#         -> inference                depends on fann (via `mixture` or `metal-gpu`), inference-core, tokenizer, grammar, weights, quant
#           -> embed                  depends on inference
#           -> tune                   depends on fann, inference
# Each tier waits for crates.io indexing before the next tier resolves it.
# (--dry-run validates only the leaf tier: cargo cannot dry-run a crate whose
#  internal deps are not yet live on the registry.)

# Every tier below uploads an archive the registry will reject outright if it
# exceeds 10 MiB, and a rejection in tier 2 leaves tier 1 already published and
# immutable. `cargo publish --dry-run` does not cover this: it can only validate
# the leaf tier, because a crate whose internal path dependencies are not yet
# live on the registry cannot be packaged at all. Measure all ten first.
echo "--- Pre-publish: package size against the crates.io upload limit ---"
"$(dirname "$0")/package-size-check.sh"

echo "--- Tier 1: leaf crates (no internal deps) ---"
cargo publish -p lattice-fann $FLAG
cargo publish -p lattice-transport $FLAG
cargo publish -p lattice-inference-core $FLAG

if [ -n "$FLAG" ]; then
    echo "=== Dry run stops after the leaf tier: later tiers need this release's leaf crates live on crates.io ==="
    exit 0
fi
echo "Waiting for crates.io indexing..."
sleep 30

echo "--- Tier 2: tokenizer, grammar and weights (depend on inference-core) ---"
cargo publish -p lattice-tokenizer $FLAG
# lattice-grammar's tests depend on the lattice-tokenizer release just uploaded.
if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi
cargo publish -p lattice-grammar $FLAG
cargo publish -p lattice-weights $FLAG

if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 3: quant (depends on inference-core, weights) ---"
cargo publish -p lattice-quant $FLAG

if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 4: inference (depends on fann, inference-core, tokenizer, grammar, weights, quant) ---"
cargo publish -p lattice-inference $FLAG

if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 5: dependent crates (depend on inference) ---"
cargo publish -p lattice-embed $FLAG
cargo publish -p lattice-tune $FLAG

echo "=== Done ==="

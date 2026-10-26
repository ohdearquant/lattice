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
#     -> tokenizer, grammar, weights, metal-runtime  depend on inference-core
#       -> model-config               depends on inference-core, weights
#         -> quant, compute           quant depends on inference-core, weights, model-config; compute on inference-core, tokenizer, weights, model-config
#           -> inference              depends on fann (via `mixture` or `metal-gpu`), inference-core, tokenizer, grammar, weights, quant, model-config, compute, metal-runtime
#             -> embed                depends on inference
#             -> serve                depends on inference (exact version), fann, tokenizer, weights
#             -> tune                 depends on fann, inference
# Each tier waits for crates.io indexing before the next tier resolves it.
# (--dry-run validates only the leaf tier: cargo cannot dry-run a crate whose
#  internal deps are not yet live on the registry.)

# Every tier below uploads an archive the registry will reject outright if it
# exceeds 10 MiB, and a rejection in tier 2 leaves tier 1 already published and
# immutable. `cargo publish --dry-run` does not cover this: it can only validate
# the leaf tier, because a crate whose internal path dependencies are not yet
# live on the registry cannot be packaged at all. Measure all fourteen first.
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

echo "--- Tier 2: tokenizer, grammar, weights and metal-runtime (depend on inference-core) ---"
cargo publish -p lattice-tokenizer $FLAG
# lattice-grammar's tests depend on the lattice-tokenizer release just uploaded.
if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi
cargo publish -p lattice-grammar $FLAG
cargo publish -p lattice-weights $FLAG
cargo publish -p lattice-metal-runtime $FLAG

if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 3: model-config (depends on inference-core, weights) ---"
cargo publish -p lattice-model-config $FLAG

if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 4: quant and compute (depend on model-config) ---"
cargo publish -p lattice-quant $FLAG
cargo publish -p lattice-compute $FLAG

if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 5: inference (depends on fann, inference-core, tokenizer, grammar, weights, quant, model-config, compute, metal-runtime) ---"
cargo publish -p lattice-inference $FLAG

if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 6: dependent crates (depend on inference) ---"
cargo publish -p lattice-embed $FLAG
cargo publish -p lattice-tune $FLAG
cargo publish -p lattice-serve $FLAG

echo "=== Done ==="

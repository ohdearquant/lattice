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
#     -> tokenizer, grammar, weights, metal-runtime, model-io  depend on inference-core
#       -> arch                       depends on inference-core, grammar, weights
#         -> quant, compute           quant depends on inference-core, weights, arch; compute on inference-core, tokenizer, weights, arch
#           -> embeddinggemma2        depends on arch, compute, inference-core, tokenizer, weights, model-io and optional metal-runtime
#           -> bert                   depends on arch, compute, inference-core, tokenizer, weights, model-io
#           -> qwen3                  depends on arch, compute, inference-core, tokenizer, weights, model-io and optional metal-runtime
#           -> gemma4                 depends on arch, compute, inference-core, tokenizer, weights
#           -> ernie45                depends on arch, compute, inference-core, tokenizer, weights and optional metal-runtime
#           -> qwen35                 depends on arch, compute, fann, grammar, inference-core, model-io, quant, tokenizer, weights and optional metal-runtime
#           -> router                 depends on tokenizer, weights and optional compute, fann, quant, qwen35 (so it waits for qwen35's index)
#             -> inference            depends on every model family and fann (via `mixture` or `metal-gpu`), inference-core, tokenizer, grammar, weights, quant, arch, compute, model-io, metal-runtime and router
#             -> embed                depends on bert, embeddinggemma2, qwen3, qwen35 and lower crates (inference only as a dev-dependency)
#               -> serve              depends on inference (exact version), tokenizer, weights and fann (via `mixture` or `metal-gpu`)
#               -> tune               depends on fann, and on inference only through `inference-hook` or `train-backward`
# Each tier waits for crates.io indexing before the next tier resolves it.
# (--dry-run validates only the leaf tier: cargo cannot dry-run a crate whose
#  internal deps are not yet live on the registry.)

# crates.io accepts at most five new crate names in a burst from one account, then
# one more every ten minutes, and answers HTTP 429 past that; `cargo publish` does
# not retry it. A release that adds more than five crates therefore waits partway
# through. A rate-limited crate is retried after PUBLISH_RETRY_WAIT seconds, up to
# PUBLISH_RETRY_MAX attempts. A crate whose version is already on the registry is
# skipped, so running the script again after a stop resumes where it stopped.
# Any other failure stops the script, as before.
PUBLISH_RETRY_WAIT="${PUBLISH_RETRY_WAIT:-610}"
PUBLISH_RETRY_MAX="${PUBLISH_RETRY_MAX:-12}"
publish_crate() {
    attempt=1
    while :; do
        log=$(mktemp "${TMPDIR:-/tmp}/lattice-publish.XXXXXX")
        if cargo publish -p "$1" $FLAG >"$log" 2>&1; then
            cat "$log"
            rm -f "$log"
            return 0
        fi
        cat "$log"
        if grep -q "^error: crate $1@[^ ]* already exists on" "$log"; then
            rm -f "$log"
            echo "$1: this version is already on the registry; skipping"
            return 0
        fi
        if grep -q -E 'status 429|429 Too Many Requests' "$log" && [ "$attempt" -lt "$PUBLISH_RETRY_MAX" ]; then
            rm -f "$log"
            echo "$1: rate limited by the registry (attempt $attempt of $PUBLISH_RETRY_MAX); retrying in ${PUBLISH_RETRY_WAIT}s"
            sleep "$PUBLISH_RETRY_WAIT"
            attempt=$((attempt + 1))
            continue
        fi
        rm -f "$log"
        echo "$1: cargo publish failed" >&2
        return 1
    done
}

# Every tier below uploads an archive the registry will reject outright if it
# exceeds 10 MiB, and a rejection in tier 2 leaves tier 1 already published and
# immutable. `cargo publish --dry-run` does not cover this: it can only validate
# the leaf tier, because a crate whose internal path dependencies are not yet
# live on the registry cannot be packaged at all. Measure every publishable workspace member first.
echo "--- Pre-publish: package size against the crates.io upload limit ---"
"$(dirname "$0")/package-size-check.sh"

echo "--- Tier 1: leaf crates (no internal deps) ---"
publish_crate lattice-fann
publish_crate lattice-transport
publish_crate lattice-inference-core

if [ -n "$FLAG" ]; then
    echo "=== Dry run stops after the leaf tier: later tiers need this release's leaf crates live on crates.io ==="
    exit 0
fi
echo "Waiting for crates.io indexing..."
sleep 30

echo "--- Tier 2: tokenizer, grammar, weights, metal-runtime and model-io (depend on inference-core) ---"
publish_crate lattice-tokenizer
# lattice-grammar's tests depend on the lattice-tokenizer release just uploaded.
if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi
publish_crate lattice-grammar
publish_crate lattice-weights
publish_crate lattice-metal-runtime
publish_crate lattice-model-io

if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 3: arch (depends on inference-core, grammar, weights) ---"
publish_crate lattice-arch

if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 4: quant and compute (depend on arch) ---"
publish_crate lattice-quant
publish_crate lattice-compute

if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 5: model families, forecasting and the adapter router (depend on arch, compute and lower crates) ---"
publish_crate lattice-forecast
publish_crate lattice-model-embeddinggemma2
publish_crate lattice-model-bert
publish_crate lattice-model-qwen3-embedding
publish_crate lattice-model-gemma4
publish_crate lattice-model-ernie45
publish_crate lattice-model-qwen35
# lattice-router depends on the lattice-model-qwen35 release just uploaded.
if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi
publish_crate lattice-router

if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 6: inference (depends on the model families and lower crates) ---"
publish_crate lattice-inference

if [ -z "$FLAG" ]; then
    echo "Waiting for crates.io indexing..."
    sleep 30
fi

echo "--- Tier 7: dependent crates (depend on inference) ---"
publish_crate lattice-embed
publish_crate lattice-tune
publish_crate lattice-serve

echo "=== Done ==="

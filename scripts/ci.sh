#!/bin/sh
set -e

: "${LATTICE_REQUIRE_FIXTURES:=1}"
export LATTICE_REQUIRE_FIXTURES

echo "=== Format Check ==="
cargo fmt --all -- --check

echo "=== Clippy ==="
cargo clippy --workspace -- -D warnings

echo "=== Doc Lint ==="
if command -v deno >/dev/null 2>&1; then
    deno fmt --check **/*.md
else
    echo "deno not found, skipping doc lint"
fi

echo "=== Tests ==="
# gemma4_e2e_forward_test.rs fails closed by default on a missing checkpoint
# (deliberately -- see the test file's module docs): this general-purpose
# workspace test run is not the checkpoint-gated job, so it opts into the
# explicit, loudly-logged skip rather than failing every runner that lacks
# the multi-GB gemma-4-e2b-it checkpoint.
LATTICE_GEMMA4_GATE_SKIP=1 cargo test --workspace

echo "=== Tokenizer Parity Gate ==="
tmp_tokenizer_log="$(mktemp)"
tmp_cmd_log="$(mktemp)"
trap 'rm -f "$tmp_tokenizer_log" "$tmp_cmd_log"' EXIT

if ! cargo test -p lattice-inference --test audit_tokenizer_parity -- --nocapture >"$tmp_cmd_log" 2>&1; then
    cat "$tmp_cmd_log"
    exit 1
fi
cat "$tmp_cmd_log"
cat "$tmp_cmd_log" >> "$tmp_tokenizer_log"

if ! cargo test -p lattice-embed --test tokenizer_parity_e2e -- --nocapture >"$tmp_cmd_log" 2>&1; then
    cat "$tmp_cmd_log"
    exit 1
fi
cat "$tmp_cmd_log"
cat "$tmp_cmd_log" >> "$tmp_tokenizer_log"

if grep -E 'SKIP|LATTICE_.*SKIPPED' "$tmp_tokenizer_log"; then
    echo "Tokenizer parity gate: unexpected skip detected — tokenizer tests must all run" >&2
    exit 1
fi

echo "=== Embedding Parity vs HF (weights optional) ==="
cargo test -p lattice-embed --test embed_parity_vs_hf -- --nocapture

echo "=== Build (release) ==="
cargo build --workspace --release

echo "=== CI Passed ==="

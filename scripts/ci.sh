#!/bin/sh
set -e

echo "=== Format Check ==="
cargo fmt --all -- --check

echo "=== Clippy ==="
cargo clippy --workspace -- -D warnings

echo "=== Doc Lint ==="
./scripts/lint-docs.sh

echo "=== Tests ==="
cargo test --workspace

echo "=== Build (release) ==="
cargo build --workspace --release

echo "=== CI Passed ==="

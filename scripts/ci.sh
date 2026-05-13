#!/bin/sh
set -e

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
cargo test --workspace

echo "=== Build (release) ==="
cargo build --workspace --release

echo "=== CI Passed ==="

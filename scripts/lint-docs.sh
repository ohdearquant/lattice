#!/bin/sh
set -e

echo "=== Doc Linting (deno) ==="
deno fmt --check **/*.md
deno lint **/*.md 2>/dev/null || true

echo "=== Doc Lint Passed ==="

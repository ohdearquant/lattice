#!/bin/sh
set -e

echo "=== Doc Linting (deno) ==="
deno fmt --check **/*.md
deno lint **/*.md 2>/dev/null || true

echo "=== Capability Matrix Fixture Check (#654) ==="
"$(dirname "$0")/check-capability-matrix.sh" --selftest
"$(dirname "$0")/check-capability-matrix.sh"

echo "=== Doc Lint Passed ==="

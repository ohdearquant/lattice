#!/usr/bin/env bash
#
# Build every binary the macOS Studio app ships (apps/macos/scripts/package-app.sh),
# each with its required-features.
#
# This is the regression gate that `cargo build --workspace` cannot provide:
# --workspace (even with --all-targets) silently SKIPS any [[bin]] that declares
# `required-features` unless those features are explicitly enabled. So a
# required-features binary can stop compiling and never trip fmt / clippy / test
# / build CI — the break only surfaces when the app is packaged.
#
# Two real regressions slipped through exactly this gap and were caught only at
# packaging time:
#   - train_grad_full  E0063  (missing GDN gradient fields under train-backward)
#   - generate_lora    E0599  (generate_streaming method not in scope)
#
# Keep this binary list in sync with apps/macos/scripts/package-app.sh.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FAIL=0
build_bin() {
    echo ""
    echo "==> cargo build --release $*"
    if ! cargo build --release "$@"; then
        echo "!! FAILED: cargo build --release $*"
        FAIL=1
    fi
}

# lattice-inference — default features (already covered by --workspace, built
# here too so the gate reflects the full app binary set).
for BIN in quantize_q4 quantize_quarot lattice qwen35_generate; do
    build_bin -p lattice-inference --bin "$BIN"
done

# required-features binaries — the ones `cargo build --workspace` silently skips.
build_bin -p lattice-tune      --bin train_grad_full --features train-backward
build_bin -p lattice-tune      --bin generate_lora   --features safetensors,inference-hook
build_bin -p lattice-inference --bin eval_perplexity --features f16,metal-gpu
build_bin -p lattice-inference --bin chat_metal      --features f16,metal-gpu

# lattice-embed — `native` is a default feature, so no explicit --features.
build_bin -p lattice-embed     --bin embed

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "ERROR: one or more app-shipped binaries failed to build."
    echo "The macOS Studio app would fail to package. Fix before merging."
    exit 1
fi
echo ""
echo "OK: all app-shipped binaries built."

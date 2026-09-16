#!/bin/sh
# The macOS Metal arms of CI, run locally, with the one bound that CI does not
# need and this laptop does.
#
# The three clippy commands below are copied from the macOS jobs in
# .github/workflows/ci.yml rather than guessed at, so a local pass means the
# same commands passed; re-read them there if that job changes.
#
# THE LIB ARM IS FILTERED, AND THE FILTER IS THE POINT. An unfiltered
# `cargo test -p lattice-inference --features f16,metal-gpu --lib` ends in the
# `forward::metal_qwen35::inner::tests::generate_multimodal_vision_*` oracle
# tests, which load a real checkpoint. Measured 2026-09-16: that run reached
# 36 GB resident and drove the kernel's memory pressure to level 4 (critical),
# starving another build on the same machine. The suite is not wrong; a laptop
# is the wrong place to run it. Those tests run in CI, where the job owns the
# box. Locally the lib arm is scoped to the module a serving change can reach
# and to one thread, which is seconds rather than tens of gigabytes.
#
# THE BOUND IS ABOUT THE SUITE, NOT THE FLAG, and that correction is here because
# the first version of this header got it wrong. `--lib` is one door to those
# oracle tests; `cargo test --workspace` is another, and it was walked through on
# this machine fifteen minutes after the `--lib` bound was written, reaching 31 GB
# and closing admission fleet-wide a second time. `--workspace`, `--all-targets`
# on a test command, and a bare `cargo test -p lattice-inference` all reach them.
# On this kind of host, run a NAMED test or a scoped module, never a suite whose
# membership you have not enumerated.
#
# Run this under whatever machine-wide build lock your environment uses; this
# script takes none of its own.
#
# Usage: scripts/metal-local-gate.sh
set -u

case "$(uname -s)" in
    Darwin) ;;
    *)
        echo "metal-local-gate: refusing on $(uname -s); these arms are macOS-only" >&2
        exit 1
        ;;
esac

overall=0

run() {
    label=$1
    shift
    echo "=== $label"
    "$@"
    rc=$?
    echo "$label rc=$rc"
    [ "$rc" -eq 0 ] || overall=$rc
    return 0
}

run "clippy-default" \
    cargo clippy -p lattice-inference --all-targets -- -D warnings
run "clippy-metal" \
    cargo clippy -p lattice-inference --all-targets --features f16,metal-gpu -- -D warnings
run "clippy-metal-bench-release" \
    cargo clippy --release -p lattice-inference --all-targets \
    --features f16,metal-gpu,bench-internals -- -D warnings
run "test-bin-lattice_serve" \
    cargo test --locked -p lattice-inference --features f16,metal-gpu,test-utils \
    --bin lattice_serve
run "test-bin-lattice" \
    cargo test --locked -p lattice-inference --features f16,metal-gpu,test-utils \
    --bin lattice
# See the header: scoped on purpose, and `--test-threads=1` because these are
# Metal tests sharing one GPU lock.
run "test-lib-serve" \
    cargo test --locked -p lattice-inference --features f16,metal-gpu,test-utils \
    --lib serve:: -- --test-threads=1

echo "== END rc=$overall"
exit "$overall"

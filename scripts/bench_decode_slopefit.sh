#!/usr/bin/env bash
# ADR-064: decode slope/intercept fit harness driver.
#
# Runs bench_decode_slopefit through the GPU handoff (bench-command.sh builds
# it and launches it under the machine locks), pipes its output through the
# Python post-processor, and emits the final ADR-064 JSON. The binary takes the
# Metal GPU lock itself, so this script is not supervised around it: the
# handoff run is the outermost supervisor and the post-processor runs outside
# the locks. The handoff merges the binary's standard error into the stream
# the post-processor reads; it ignores every line that is not a SLOPEFIT record.
# The handoff admits a commit-clean checkout only and runs from the repository
# root.
#
# Usage:
#   ./scripts/bench_decode_slopefit.sh          # smoke grid {64,256,512}
#   SLOPEFIT_FULL=1 ./scripts/bench_decode_slopefit.sh   # full production grid
#   SLOPEFIT_CONTEXTS="64 512 1024" ./scripts/bench_decode_slopefit.sh
#   ./scripts/bench_decode_slopefit.sh --out artifacts/adr064-gpu-decode-current.json
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"

PY="$REPO/scripts/bench_decode_slopefit.py"

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$PWD" "$1" ;;
    esac
}

OUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            OUT="$2"
            shift 2
            ;;
        *)
            >&2 echo "[slopefit] unknown arg: $1"
            exit 1
            ;;
    esac
done

# The handoff runs from the repository root, so a relative path the caller gave
# is resolved against the caller's directory first.
if [[ -n "$OUT" ]]; then
    OUT="$(absolute_path "$OUT")"
fi
for path_var in LATTICE_MODEL_DIR LATTICE_TOKENIZER_DIR; do
    if [[ -n "${!path_var:-}" ]]; then
        export "$path_var=$(absolute_path "${!path_var}")"
    fi
done

# The handoff builds in the repository's .cache. Reapply the marker before every
# build because cleaning .cache removes the protection with it.
"$REPO/scripts/lib/ensure-noindex-marker.sh" "$REPO/.cache"

slopefit_handoff() {
    (cd "$REPO" && "$REPO/scripts/bench-command.sh" --gpu-handoff --durable --label decode-slopefit -- \
        cargo run --locked --release -p lattice-inference --bin bench_decode_slopefit \
        --features "f16,metal-gpu")
}

>&2 echo "[slopefit] building and running bench_decode_slopefit (release, GPU handoff)..."
if [[ -n "$OUT" ]]; then
    mkdir -p "$(dirname "$OUT")"
    slopefit_handoff | tee /dev/stderr | uv run --project "$REPO" python3 "$PY" --out "$OUT"
else
    slopefit_handoff | tee /dev/stderr | uv run --project "$REPO" python3 "$PY"
fi

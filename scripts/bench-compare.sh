#!/usr/bin/env bash
# bench-compare.sh — A/B benchmark comparison across two git refs.
#
# Usage:
#   scripts/bench-compare.sh                        # origin/main vs HEAD (quick)
#   scripts/bench-compare.sh main pr/embed           # explicit base vs head
#   scripts/bench-compare.sh --full main pr/embed    # full Criterion (slow, tight CIs)
#   scripts/bench-compare.sh HEAD~3                  # HEAD~3 vs HEAD
#
# Defaults to --quick (~2 min). Use --full (~15 min) for tight confidence intervals.
# Uses a git worktree for the base ref so your working tree stays untouched.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QUICK_FLAGS="--quick"  # ~10 samples, ~2 min total

# Parse --full flag
if [ "${1:-}" = "--full" ]; then
  QUICK_FLAGS=""  # 100 samples, ~15 min total
  shift
fi

BASE_REF="${1:-origin/main}"
HEAD_REF="${2:-HEAD}"

# Resolve to short SHAs for display
BASE_SHA=$(git -C "$REPO" rev-parse --short "$BASE_REF" 2>/dev/null || echo "$BASE_REF")
HEAD_SHA=$(git -C "$REPO" rev-parse --short "$HEAD_REF" 2>/dev/null || echo "$HEAD_REF")

echo "=== bench-compare: $BASE_REF ($BASE_SHA) vs $HEAD_REF ($HEAD_SHA) ==="

# --- Worktree for base ref ---
WT="$REPO/.cache/bench-compare-base"
if [ -d "$WT" ]; then
  git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
fi
git -C "$REPO" worktree add --detach "$WT" "$BASE_REF" 2>&1 | tail -1

cleanup() {
  git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || true
}
trap cleanup EXIT

# --- Bench list (same as ADR-058 Phase 1) ---
BENCHES_INFERENCE=("elementwise_cpu_bench" "batch_throughput_bench")
BENCHES_EMBED="simd"

# --- Build + bench base ---
echo ""
echo "--- Building + benching BASE ($BASE_SHA) ---"
(
  cd "$WT"
  # Only bench what exists — some benches may not exist on older refs
  for bench in "${BENCHES_INFERENCE[@]}"; do
    if cargo bench -p lattice-inference --bench "$bench" --no-run 2>/dev/null; then
      cargo bench -p lattice-inference --bench "$bench" -- --save-baseline compare-base --noplot $QUICK_FLAGS 2>&1 | grep -E "time:" || true
    else
      echo "  ($bench not present on $BASE_SHA — skipping)"
    fi
  done
  cargo bench -p lattice-embed --bench "$BENCHES_EMBED" -- --save-baseline compare-base --noplot $QUICK_FLAGS 2>&1 | grep -E "time:" || true
)

# --- Copy base criterion data to HEAD's target ---
echo ""
echo "--- Building + benching HEAD ($HEAD_SHA) ---"

# Determine head working dir — if HEAD_REF is HEAD, use $REPO directly
if [ "$HEAD_REF" = "HEAD" ]; then
  HEAD_DIR="$REPO"
else
  HEAD_WT="$REPO/.cache/bench-compare-head"
  if [ -d "$HEAD_WT" ]; then
    git -C "$REPO" worktree remove --force "$HEAD_WT" 2>/dev/null || rm -rf "$HEAD_WT"
  fi
  git -C "$REPO" worktree add --detach "$HEAD_WT" "$HEAD_REF" 2>&1 | tail -1
  HEAD_DIR="$HEAD_WT"
  # Update cleanup to also remove head worktree
  trap 'git -C "$REPO" worktree remove --force "$HEAD_WT" 2>/dev/null || true; cleanup' EXIT
fi

# Copy base's criterion baseline into head's target/criterion
mkdir -p "$HEAD_DIR/target/criterion"
if [ -d "$WT/target/criterion" ]; then
  # Copy the baseline data (compare-base dirs)
  rsync -a "$WT/target/criterion/" "$HEAD_DIR/target/criterion/" --include='**/compare-base/**' --include='*/' --exclude='*' 2>/dev/null || true
  # Also copy the raw estimates for comparison
  rsync -a "$WT/target/criterion/" "$HEAD_DIR/target/criterion/" 2>/dev/null || true
fi

(
  cd "$HEAD_DIR"
  for bench in "${BENCHES_INFERENCE[@]}"; do
    if cargo bench -p lattice-inference --bench "$bench" --no-run 2>/dev/null; then
      cargo bench -p lattice-inference --bench "$bench" -- --baseline compare-base --noplot $QUICK_FLAGS 2>&1 | grep -E "time:|change:" || true
    else
      echo "  ($bench not present on $HEAD_SHA — skipping)"
    fi
  done
  cargo bench -p lattice-embed --bench "$BENCHES_EMBED" -- --baseline compare-base --noplot $QUICK_FLAGS 2>&1 | grep -E "time:|change:" || true
)

# --- Report ---
echo ""
echo "=== Full gate report ==="
if [ -d "$HEAD_DIR/target/criterion" ]; then
  python3 "$REPO/scripts/perf-bench-gate.py" "$HEAD_DIR/target/criterion" "local-compare" 2>&1 || true
fi

echo ""
echo "Done. Base=$BASE_REF ($BASE_SHA), Head=$HEAD_REF ($HEAD_SHA)"

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
# Optional Criterion filters:
#   BENCH_GROUPS_INFERENCE="rms_norm|gelu" scripts/bench-compare.sh
#   BENCH_GROUPS_EMBED="simd_dot_product|int8_raw" scripts/bench-compare.sh
# Unset filters run all groups in the default bench targets:
#   lattice-inference: elementwise_cpu_bench
#   lattice-embed: simd
# Uses a git worktree for the base ref so your working tree stays untouched.
#
# lattice#714: five of the lattice-embed bench targets' groups
# (simd_dot_product, simd_cosine_similarity, int8_batch_cosine,
# int4_cosine_distance, simd_batch_cosine_non_normalized_query) are confirmed
# noise-dominated in --quick mode by same-toolchain A/A reproductions on
# identical refs (the original pair: FAIL/WARN sign flipped across dozens of
# entries run to run; the 2026-07-13 additions: confirmed-CI FAIL rows inside
# exclusive bench windows with DISJOINT failing groups across two same-commit
# runs). In --quick mode (the default), only those named groups are
# measured and reported but excluded from the FAIL/WARN gate and exit code —
# see the "informational" section of the report and INFO_GROUPS_ALLOWLIST
# below. Every other group in the target, including any added later, gates
# normally. --full mode gates every group normally regardless.
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
# BENCHES_INFERENCE / CARGO_FEATURES_INFERENCE are overridable so a PR can
# point this script at a different inference bench target (e.g. one gated
# behind `bench-internals`) without hand-rolling a separate A/B script.
BENCHES_INFERENCE="${BENCHES_INFERENCE:-elementwise_cpu_bench}"
CARGO_FEATURES_INFERENCE="${CARGO_FEATURES_INFERENCE:-}"
BENCHES_EMBED="simd"
BENCH_GROUPS_INFERENCE="${BENCH_GROUPS_INFERENCE:-}"
BENCH_GROUPS_EMBED="${BENCH_GROUPS_EMBED:-}"

# --- Build + bench base ---
echo ""
echo "--- Building + benching BASE ($BASE_SHA) ---"
(
  cd "$WT"
  # Only bench what exists — some benches may not exist on older refs
  if cargo bench -p lattice-inference --bench "$BENCHES_INFERENCE" ${CARGO_FEATURES_INFERENCE:+--features "$CARGO_FEATURES_INFERENCE"} --no-run 2>/dev/null; then
    cargo bench -p lattice-inference --bench "$BENCHES_INFERENCE" ${CARGO_FEATURES_INFERENCE:+--features "$CARGO_FEATURES_INFERENCE"} -- ${BENCH_GROUPS_INFERENCE:+"$BENCH_GROUPS_INFERENCE"} --save-baseline compare-base --noplot $QUICK_FLAGS 2>&1 | grep -E "time:" || true
  else
    echo "  ($BENCHES_INFERENCE not present on $BASE_SHA — skipping)"
  fi
  cargo bench -p lattice-embed --bench "$BENCHES_EMBED" -- ${BENCH_GROUPS_EMBED:+"$BENCH_GROUPS_EMBED"} --save-baseline compare-base --noplot $QUICK_FLAGS 2>&1 | grep -E "time:" || true
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
  if cargo bench -p lattice-inference --bench "$BENCHES_INFERENCE" ${CARGO_FEATURES_INFERENCE:+--features "$CARGO_FEATURES_INFERENCE"} --no-run 2>/dev/null; then
    cargo bench -p lattice-inference --bench "$BENCHES_INFERENCE" ${CARGO_FEATURES_INFERENCE:+--features "$CARGO_FEATURES_INFERENCE"} -- ${BENCH_GROUPS_INFERENCE:+"$BENCH_GROUPS_INFERENCE"} --baseline compare-base --noplot $QUICK_FLAGS 2>&1 | grep -E "time:|change:" || true
  else
    echo "  ($BENCHES_INFERENCE not present on $HEAD_SHA — skipping)"
  fi
  cargo bench -p lattice-embed --bench "$BENCHES_EMBED" -- ${BENCH_GROUPS_EMBED:+"$BENCH_GROUPS_EMBED"} --baseline compare-base --noplot $QUICK_FLAGS 2>&1 | grep -E "time:|change:" || true
)

# --- Quick-mode informational-groups (lattice#714) ---
# Fixed, reviewed allowlist of the issue-evidenced noisy groups only — NOT a
# target-wide dump. lattice#714's quantitative reproduction (same-toolchain
# A/A runs on identical refs, --quick mode) confined every flip-signed FAIL/
# WARN to two lattice-embed `simd` groups; no other group in that target has
# issue-backed noise evidence, so no other group is exempted here. The
# allowlist itself and the intersection-with-the-real-listing logic live in
# scripts/lib/bench-informational-groups.sh, invoked below — kept in its own
# file (rather than inline here) so scripts/perf-bench-gate.py --selftest can
# run the exact same shell code against a controlled listing and catch a
# shell-only regression, not just a Python-classifier one.
# This list is embed-`simd`-specific: names are matched against the Criterion
# top-level group name only (scripts/perf-bench-gate.py), so it must never be
# extended with a name that could collide with a `lattice-inference` group —
# if that ever becomes a concern, namespace by target (e.g. "embed:<group>")
# on both sides of the handoff instead of trusting name uniqueness. Adding a
# group requires the same kind of same-toolchain A/A quantitative evidence
# that justified the current two, reviewed in a PR — never derived
# automatically from `--list`, which would silently exempt every future
# group added to the target.
INFO_GROUPS_FILE="$REPO/.cache/bench-compare-informational-groups.txt"
rm -f "$INFO_GROUPS_FILE"
if [ -n "$QUICK_FLAGS" ] && [ "$BENCHES_EMBED" = "simd" ]; then
  (
    cd "$HEAD_DIR"
    # --list reflects both the built binary and any BENCH_GROUPS_EMBED filter
    # already applied; the helper intersects it against the fixed allowlist
    # so a stale or renamed allowlist entry fails closed (never gates) rather
    # than silently no-op'ing into "everything is informational."
    cargo bench -p lattice-embed --bench "$BENCHES_EMBED" -- ${BENCH_GROUPS_EMBED:+"$BENCH_GROUPS_EMBED"} --list 2>/dev/null \
      | "$REPO/scripts/lib/bench-informational-groups.sh" > "$INFO_GROUPS_FILE"
  )
fi

# --- Report ---
echo ""
echo "=== Full gate report ==="
GATE_ARGS=(--baseline-name compare-base)
if [ -s "$INFO_GROUPS_FILE" ]; then
  GATE_ARGS+=(--informational-groups-file "$INFO_GROUPS_FILE")
fi
if [ -d "$HEAD_DIR/target/criterion" ]; then
  python3 "$REPO/scripts/perf-bench-gate.py" "$HEAD_DIR/target/criterion" "local-compare" "${GATE_ARGS[@]}" 2>&1 || true
fi

echo ""
echo "Done. Base=$BASE_REF ($BASE_SHA), Head=$HEAD_REF ($HEAD_SHA)"

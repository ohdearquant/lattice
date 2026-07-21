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
# VOCABULARY, because these four are separate and get conflated. A group is
# MEASURED if the bench ran and produced numbers; REPORTED if those numbers
# appear in the report; CLASSIFIED GATING (vs informational) if a regression
# in it contributes to the report's FAIL verdict; and ENFORCED only if that
# FAIL verdict reaches the caller as a non-zero exit status. Classification
# is not enforcement: this script computes a verdict and then discards its
# exit status, so it is REPORT-ONLY in both modes. `make bench-gate` is the
# path that enforces. Use these words literally below.
#
# lattice#714 / lattice#1060: the lattice-embed `simd` bench TARGET is
# informational in --quick mode (the default). Same-toolchain, same-commit
# A/A reproductions in exclusive bench windows repeatedly produced
# confirmed-CI FAIL rows (+8% to +17%, 95% CIs) on identical binaries with
# DISJOINT failing groups across runs — machine-level noise above
# quick-mode resolution, so per-group exemptions were a treadmill. Demoted
# targets are named in ONE validated manifest,
# scripts/lib/bench-quick-informational-targets.txt (validated by
# scripts/perf-bench-gate.py --selftest); their groups are still fully
# measured and rendered — the informational section plus the
# all-measurements table record every number — but classified informational,
# so they cannot produce a FAIL verdict. Every non-demoted target this script
# benches (the lattice-inference one) is classified gating in --quick.
#
# --full applies no informational demotion: every group it benches is
# classified gating, simd included, because full resolution is tight enough
# to distinguish a real simd regression from machine noise. Three caveats
# keep that from meaning "a regression cannot get past this".
#
# Enforcement: neither mode enforces. This script ends its gate invocation
# with `|| true`, so a FAIL verdict is printed and then discarded, and the
# script exits 0 either way. `make bench-gate` is the enforcing path: it runs
# the same two default targets unfiltered against the perf-baselines branch
# and returns perf-bench-gate.py's status directly.
#
# Scope: this script benches two targets, not the workspace's full bench set
# — lattice-inference:$BENCHES_INFERENCE (default elementwise_cpu_bench) and
# lattice-embed:simd. The optional BENCH_GROUPS_* filters above narrow it
# further, so a filtered --full run classifies only the selected groups of
# those two.
#
# Automation: bench-update.yml runs those targets at full resolution on main
# — on every push touching the perf paths (which include embed's simd source
# and bench) and weekly by cron — and saves the baselines. It does not invoke
# perf-bench-gate.py and takes no regression-specific fail or alert action;
# its ordinary job steps can of course still fail on their own errors. It is
# not silent about regressions either: its README generator compares each
# snapshot against its predecessor and publishes a "Worst step-regression"
# headline. So full-resolution automation today is regression REPORTING, and
# what is missing is ENFORCEMENT (#1105 tracks that lane).
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

# --- Keep Spotlight out of the bench worktrees ---
# The worktrees created below are full repository checkouts. Indexing them
# produces filesystem churn that lands asymmetrically in whichever measurement
# phase it overlaps, and a base-then-head run turns that asymmetry into an
# apparent code delta. The marker suppresses indexing for the whole directory.
# Recreated every run so a wiped .cache does not silently lose the protection.
# Inert on non-macOS. Fail-closed: refuse to measure without the protection
# rather than emit numbers that look trustworthy and are not.
"$REPO/scripts/lib/ensure-noindex-marker.sh" "$REPO/.cache"

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

# --- Quick-mode informational demotion (lattice#714 / lattice#1060) ---
# Target-level policy: the demoted-target set lives in ONE validated
# manifest, scripts/lib/bench-quick-informational-targets.txt. For each
# bench target this script ran, the helper below is given the target key
# (`<crate>:<bench-target>`) and that target's Criterion `--list` output;
# it prints every top-level group of a demoted target and nothing for any
# other target. Deriving the group set from the listing is deliberate
# under target semantics — the demotion covers the target, so a group
# added to a demoted target later is informational by policy, while every
# non-demoted target stays gated because its key is absent from the
# manifest.
#
# Criterion group names are bare strings once they leave the helper, with
# no target attribution — so before folding a demoted target's groups into
# the flat informational set, resolve-informational-groups.sh checks them
# against every gated target's own listing and drops (gates) any name that
# collides, warning loudly on stderr. This is the composed-path guard: a
# group demoted for lattice-embed:simd must never silently exempt an
# identically-named lattice-inference group from the gate.
#
# Both the helper and the resolver live in scripts/lib/ (rather than inline
# here) so scripts/perf-bench-gate.py --selftest can run the exact same
# shell code against controlled listings and catch a shell-only
# regression, not just a Python-classifier one. --full skips this block
# entirely: every group of every target gates at full resolution.
INFO_GROUPS_FILE="$REPO/.cache/bench-compare-informational-groups.txt"
rm -f "$INFO_GROUPS_FILE"
if [ -n "$QUICK_FLAGS" ]; then
  (
    cd "$HEAD_DIR"
    DEMOTED_GROUPS_FILE="$REPO/.cache/bench-compare-demoted-groups.txt"
    GATED_GROUPS_FILE="$REPO/.cache/bench-compare-gated-groups.txt"
    : > "$DEMOTED_GROUPS_FILE"
    : > "$GATED_GROUPS_FILE"
    DEMOTED_TARGETS=""
    GATED_TARGETS=""

    # --list reflects both the built binary and any BENCH_GROUPS_* filter
    # already applied. Every target's groups go into the demoted-set file
    # or the gated-set file depending on manifest membership; the resolver
    # then reconciles the two before anything is treated as informational.
    route_target_groups() {
      local target="$1" listing="$2"
      if "$REPO/scripts/lib/bench-informational-groups.sh" --print-targets | grep -qxF "$target"; then
        "$REPO/scripts/lib/bench-informational-groups.sh" --list-groups "$listing" >> "$DEMOTED_GROUPS_FILE"
        DEMOTED_TARGETS="${DEMOTED_TARGETS:+$DEMOTED_TARGETS,}$target"
      else
        "$REPO/scripts/lib/bench-informational-groups.sh" --list-groups "$listing" >> "$GATED_GROUPS_FILE"
        GATED_TARGETS="${GATED_TARGETS:+$GATED_TARGETS,}$target"
      fi
    }

    EMBED_LISTING="$REPO/.cache/bench-compare-embed-list.txt"
    cargo bench -p lattice-embed --bench "$BENCHES_EMBED" -- ${BENCH_GROUPS_EMBED:+"$BENCH_GROUPS_EMBED"} --list 2>/dev/null > "$EMBED_LISTING" || true
    route_target_groups "lattice-embed:$BENCHES_EMBED" "$EMBED_LISTING"

    INFERENCE_LISTING="$REPO/.cache/bench-compare-inference-list.txt"
    cargo bench -p lattice-inference --bench "$BENCHES_INFERENCE" ${CARGO_FEATURES_INFERENCE:+--features "$CARGO_FEATURES_INFERENCE"} -- ${BENCH_GROUPS_INFERENCE:+"$BENCH_GROUPS_INFERENCE"} --list 2>/dev/null > "$INFERENCE_LISTING" || true
    route_target_groups "lattice-inference:$BENCHES_INFERENCE" "$INFERENCE_LISTING"

    "$REPO/scripts/lib/resolve-informational-groups.sh" \
      "$DEMOTED_GROUPS_FILE" "${DEMOTED_TARGETS:-none}" \
      "$GATED_GROUPS_FILE" "${GATED_TARGETS:-none}" \
      > "$INFO_GROUPS_FILE"
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

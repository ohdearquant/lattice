#!/usr/bin/env bash
# bench-compare-impl.sh — A/B benchmark comparison across two git refs.
#
# INVOKE scripts/bench-compare.sh, NOT THIS FILE. This is the measurement body;
# the entry point runs it under scripts/lib/bench-locks.py, which holds the
# machine-wide bench-window and Metal GPU locks for the whole run. Running this
# file by accident is refused below. Running it deliberately, by a caller
# willing to prepare the status file, is not prevented — see the comment above
# verify_locks for exactly what that check establishes.
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
# is not enforcement: by default this script computes a verdict, captures the
# gate's exit status, and does not act on it, so --quick and --full are both
# REPORT-ONLY.
# Enforcement is opt-in per invocation via --fail-on-regression, which
# propagates the gate's status instead; `make bench-gate` also enforces.
# Use these words literally below.
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
# Enforcement: neither mode enforces BY DEFAULT. The gate's exit status is
# captured into GATE_RC at the bottom and re-raised only under
# --fail-on-regression, so by default a FAIL verdict is printed and the script
# still exits 0 — which is why the demotion below is a resolution split rather
# than a coverage hole in the default path. Two
# callers do enforce: --fail-on-regression propagates the gate's status (exit
# 1 confirmed regression, exit 2 the measurement itself is broken), and
# `make bench-gate` runs the same two default targets unfiltered against the
# perf-baselines branch and returns perf-bench-gate.py's status directly.
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
# headline. So bench-update.yml itself is regression REPORTING.
#
# Enforcement at full resolution is a separate workflow, perf-postmerge-gate
# .yml: it runs this script with --full --fail-on-regression on perf-path
# merges to main, benching the merged commit against its own parent, and
# fails the job when the gate confirms a regression. Read the two together —
# bench-update.yml maintains the baselines and the trend, the post-merge gate
# is what makes a confirmed regression stop something.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
QUICK_FLAGS="--quick"  # ~10 samples, ~2 min total

# --- Refuse to measure unless the recorded supervisor is one of our ancestors ---
# scripts/bench-compare.sh runs this body under scripts/lib/bench-locks.py,
# which records its own PID here after taking both locks. This requires that PID
# to be one of THIS process's ancestors before measuring.
#
# WHAT THAT ACTUALLY PROVES, stated exactly, because the tempting overclaim is
# one word wider than the truth. The file supplies the PID and the OS supplies
# the chain, so the check establishes a RELATION: the named process is really an
# ancestor of this one. It refuses a status file left over from a finished run,
# a file copied from a different run or machine, and accidental direct
# invocation of this body, which are the ways this actually gets run without
# isolation. It does NOT stop a caller who deliberately records an ancestor's
# PID, their own shell's included: the recorded PID is still caller-supplied,
# and ancestry confirms the relation, not that the named process holds anything.
#
# Closing that needs the lock DESCRIPTOR rather than a PID — an fstat identity
# check on an inherited fd, followed by a non-blocking flock on it, which leaves
# the lock held on that description whichever branch is taken. That arrives with
# the nested-acquirer work, where a child that must hold a lock exists to
# receive the descriptor. Until then this is the strong refusal, not a proof.
LOCK_STATUS_FILE="$REPO/.cache/bench-locks-status.txt"
LOCK_SUMMARY=""
verify_locks() {
  if [ ! -f "$LOCK_STATUS_FILE" ]; then
    echo "bench-compare: no lock status at $LOCK_STATUS_FILE." >&2
    echo "  Run scripts/bench-compare.sh, not this file directly." >&2
    exit 2
  fi
  local sup
  sup="$(sed -n 's/^supervisor_pid=//p' "$LOCK_STATUS_FILE" | head -1)"
  case "$sup" in
    ''|*[!0-9]*)
      echo "bench-compare: lock status names no supervisor PID — refusing." >&2
      exit 2
      ;;
  esac
  local pid="$PPID"
  local hops=0
  local parent
  local walked=1
  while [ "$pid" -gt 1 ] && [ "$hops" -lt 64 ]; do
    if [ "$pid" = "$sup" ]; then
      LOCK_SUMMARY="$(sed -n 's/^lock=/  /p' "$LOCK_STATUS_FILE")"
      return 0
    fi
    # A failing ps must reach the refusal below rather than abort the script.
    # Under `set -o pipefail` the failure propagates out of the assignment and
    # `set -e` exits with ps's own status, skipping the diagnostic entirely: the
    # caller sees a bare 1 or 126 and no message. That is still fail-closed, but
    # silently, and it fires on the ordinary case of an ancestor exiting during
    # the walk, not only where process inspection is denied.
    if ! parent="$(ps -o ppid= -p "$pid" 2>/dev/null)"; then
      walked=0
      break
    fi
    pid="$(printf '%s' "$parent" | tr -d ' ')"
    case "$pid" in ''|*[!0-9]*) walked=0; break ;; esac
    hops=$((hops + 1))
  done
  if [ "$walked" -eq 0 ]; then
    echo "bench-compare: could not walk this process's ancestry to the end" \
         "(ps failed or returned nothing) — refusing to measure." >&2
    echo "  Supervisor $sup was not seen before the walk stopped, so whether it" \
         "is an ancestor is unknown, and unknown is refused." >&2
  else
    echo "bench-compare: lock supervisor $sup is not an ancestor of this run" \
         "(stale or copied $LOCK_STATUS_FILE) — refusing to measure." >&2
  fi
  echo "  Run scripts/bench-compare.sh, not this file directly." >&2
  exit 2
}
verify_locks

# --- Ambient load gate ---
# A lock excludes peers; it says nothing about ambient load. Both are needed.
# Sampled at three points and refused at each: an unquiet machine before the
# base phase invalidates the run before it costs anything, and an unquiet
# machine after the head phase invalidates numbers that were already taken,
# which is exactly when the temptation to keep them is strongest.
QUIET_SAMPLES=""
quiet_gate() {
  local label="$1" line rc=0
  line="$(python3 "$REPO/scripts/lib/quiet-probe.py" --label "$label")" || rc=$?
  echo "$line"
  QUIET_SAMPLES="${QUIET_SAMPLES}${QUIET_SAMPLES:+
}$line"
  if [ "$rc" -ne 0 ]; then
    echo "bench-compare: machine was not quiet at '$label' — refusing to" \
         "certify this A/B. Set BENCH_IDLE_FLOOR to judge against a" \
         "different floor, and say so wherever the numbers are quoted." >&2
    exit 2
  fi
}

# Parse flags. Both are optional and may appear in either order, but they must
# precede the positional BASE/HEAD pair: the first non-flag argument ends flag
# parsing. A flag written AFTER a ref is rejected rather than silently taken as
# a ref — `bench-compare.sh HEAD~1 --full` used to resolve "--full" as HEAD_REF
# and bench against nonsense. Use `--` to pass a ref that legitimately begins
# with a dash.
#
# --fail-on-regression exists because this script is a REPORTER by default: the
# gate invocation at the bottom captures its status into GATE_RC and re-raises
# it only under this flag, so a confirmed regression is rendered in the report
# while the script still exits 0. That is correct for a
# human reading an A/B, and completely wrong for an automated lane, where a
# green exit beside a printed FAIL means the job passes on a real regression.
# Opt in to propagate the gate's exit code instead. Default behavior is
# unchanged so existing callers keep their current semantics.
FAIL_ON_REGRESSION=0
AFTER_DDASH=0
while [ $# -gt 0 ]; do
  case "${1:-}" in
    --full)
      QUICK_FLAGS=""  # 100 samples, ~15 min total
      shift
      ;;
    --fail-on-regression)
      FAIL_ON_REGRESSION=1
      shift
      ;;
    --)
      AFTER_DDASH=1
      shift
      break
      ;;
    -*)
      echo "bench-compare.sh: unknown flag '$1'" >&2
      echo "usage: bench-compare.sh [--full] [--fail-on-regression] [BASE_REF] [HEAD_REF]" >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

BASE_REF="${1:-origin/main}"
HEAD_REF="${2:-HEAD}"

# Reject dash-led leftovers. Without this a misplaced flag becomes a ref and the
# script benches against garbage, which is worse than refusing: it produces a
# confident-looking A/B nobody asked for.
#
# `--` opts out, and that opt-out has to be honored HERE, not just in the parser
# above: the parser only shifts `--` away, so without this guard the very
# arguments `--` was meant to protect land back in "$@" and are rejected by the
# loop below. Advertising an escape hatch that the next ten lines then close is
# worse than having none, because the diagnostic names it as the remedy.
if [ "$AFTER_DDASH" = "0" ]; then
  for arg in "$@"; do
    case "$arg" in
      -*)
        echo "bench-compare.sh: '$arg' looks like a flag but follows a positional" \
             "argument; flags must precede BASE/HEAD (use -- for a literal ref)" >&2
        echo "usage: bench-compare.sh [--full] [--fail-on-regression] [BASE_REF] [HEAD_REF]" >&2
        exit 2
        ;;
    esac
  done
fi
if [ "$#" -gt 2 ]; then
  echo "bench-compare.sh: too many positional arguments ($#); expected at most" \
       "BASE_REF and HEAD_REF" >&2
  exit 2
fi

# Resolve to short SHAs for display
BASE_SHA=$(git -C "$REPO" rev-parse --short --end-of-options "$BASE_REF" 2>/dev/null || echo "$BASE_REF")
HEAD_SHA=$(git -C "$REPO" rev-parse --short --end-of-options "$HEAD_REF" 2>/dev/null || echo "$HEAD_REF")

echo "=== bench-compare: $BASE_REF ($BASE_SHA) vs $HEAD_REF ($HEAD_SHA) ==="
quiet_gate "before base"

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
git -C "$REPO" worktree add --detach --end-of-options "$WT" "$BASE_REF" 2>&1 | tail -1

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

# --- Measurement-integrity helpers (only bite under --fail-on-regression) ---
# `cargo bench ... | grep -E "time:" || true` discards cargo's status TWICE: a
# pipeline reports its LAST command (grep), and `|| true` then resets
# PIPESTATUS to 0. So a bench that failed to build or died mid-run looked
# exactly like a bench that produced no matching lines, and the A/B continued
# with half its measurements missing. Verified: after `p | grep x || true`,
# ${PIPESTATUS[0]} reads 0 even when p exited 7.
#
# Cargo's exit status is necessary and not sufficient. A bench invocation whose
# Criterion filter matches no benchmark exits 0 having measured nothing, and a
# target that emits no Criterion output contributes no comparison for the gate
# to reconcile — absence leaves no artifact to be found missing. So each
# invocation also reports how many measurement lines it actually printed:
# that is the only per-target evidence available at the point where the run's
# INTENT is known. Downstream, `perf-bench-gate.py` sees a directory tree and
# cannot know a second target was ever supposed to be in it.
BENCH_RC=0
BENCH_LINES=0
run_bench() {
  local filter="$1"; shift
  BENCH_RC=0
  BENCH_LINES=0
  local matched
  matched="$(mktemp)"
  { "$@" 2>&1 | grep -E "$filter" | tee "$matched"; BENCH_RC=${PIPESTATUS[0]}; } || true
  BENCH_LINES="$(wc -l < "$matched" | tr -d ' ')"
  rm -f "$matched"
}

# A partial A/B is not weaker evidence that nothing regressed, it is no
# evidence: the target that failed is precisely the one nobody measured. Exit 2
# (measurement broken) rather than 1 (confirmed regression) because the two ask
# the reader for opposite responses. The reporter keeps its tolerant behavior.
require_measured() {
  local what="$1" rc="$2" lines="${3:-}"
  if [ "$FAIL_ON_REGRESSION" != "1" ]; then
    return 0
  fi
  if [ "$rc" -ne 0 ]; then
    echo "bench-compare: $what failed (exit $rc) — refusing to certify a partial A/B." >&2
    exit 2
  fi
  # An invocation that exits 0 having printed no measurement line ran no
  # benchmark (a filter that matches nothing is the ordinary way to get here).
  # Its target then produces no Criterion comparison at all, and a gate that
  # reconciles comparisons found against comparisons judged cannot see it:
  # there is nothing on disk to be missing. Caught here, where the target is
  # still named, or not at all.
  if [ -n "$lines" ] && [ "$lines" -eq 0 ]; then
    echo "bench-compare: $what exited 0 but produced no measurements — refusing to certify a partial A/B." >&2
    exit 2
  fi
}

# --- Build + bench base ---
echo ""
echo "--- Building + benching BASE ($BASE_SHA) ---"
BASE_PHASE_RC=0
(
  cd "$WT"
  # Only bench what exists — some benches may not exist on older refs. That
  # tolerance is right for a human comparing against an old ref and wrong for
  # the enforcing lane, where "absent" and "failed to compile" arrive on the
  # same channel and one of them silently deletes half the comparison.
  if cargo bench -p lattice-inference --bench "$BENCHES_INFERENCE" ${CARGO_FEATURES_INFERENCE:+--features "$CARGO_FEATURES_INFERENCE"} --no-run 2>/dev/null; then
    run_bench "time:" cargo bench -p lattice-inference --bench "$BENCHES_INFERENCE" ${CARGO_FEATURES_INFERENCE:+--features "$CARGO_FEATURES_INFERENCE"} -- ${BENCH_GROUPS_INFERENCE:+"$BENCH_GROUPS_INFERENCE"} --save-baseline compare-base --noplot $QUICK_FLAGS
    require_measured "base lattice-inference:$BENCHES_INFERENCE" "$BENCH_RC" "$BENCH_LINES"
  else
    require_measured "base lattice-inference:$BENCHES_INFERENCE build (--no-run)" 1
    echo "  ($BENCHES_INFERENCE not present on $BASE_SHA — skipping)"
  fi
  run_bench "time:" cargo bench -p lattice-embed --bench "$BENCHES_EMBED" -- ${BENCH_GROUPS_EMBED:+"$BENCH_GROUPS_EMBED"} --save-baseline compare-base --noplot $QUICK_FLAGS
  require_measured "base lattice-embed:$BENCHES_EMBED" "$BENCH_RC" "$BENCH_LINES"
) || BASE_PHASE_RC=$?
# `exit` inside `( ... )` leaves the SUBSHELL, so the status has to be caught
# and re-raised here or the refusal above is itself swallowed.
if [ "$BASE_PHASE_RC" -ne 0 ]; then exit "$BASE_PHASE_RC"; fi

quiet_gate "between phases"

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
  git -C "$REPO" worktree add --detach --end-of-options "$HEAD_WT" "$HEAD_REF" 2>&1 | tail -1
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

HEAD_PHASE_RC=0
(
  cd "$HEAD_DIR"
  if cargo bench -p lattice-inference --bench "$BENCHES_INFERENCE" ${CARGO_FEATURES_INFERENCE:+--features "$CARGO_FEATURES_INFERENCE"} --no-run 2>/dev/null; then
    run_bench "time:|change:" cargo bench -p lattice-inference --bench "$BENCHES_INFERENCE" ${CARGO_FEATURES_INFERENCE:+--features "$CARGO_FEATURES_INFERENCE"} -- ${BENCH_GROUPS_INFERENCE:+"$BENCH_GROUPS_INFERENCE"} --baseline compare-base --noplot $QUICK_FLAGS
    require_measured "head lattice-inference:$BENCHES_INFERENCE" "$BENCH_RC" "$BENCH_LINES"
  else
    require_measured "head lattice-inference:$BENCHES_INFERENCE build (--no-run)" 1
    echo "  ($BENCHES_INFERENCE not present on $HEAD_SHA — skipping)"
  fi
  run_bench "time:|change:" cargo bench -p lattice-embed --bench "$BENCHES_EMBED" -- ${BENCH_GROUPS_EMBED:+"$BENCH_GROUPS_EMBED"} --baseline compare-base --noplot $QUICK_FLAGS
  require_measured "head lattice-embed:$BENCHES_EMBED" "$BENCH_RC" "$BENCH_LINES"
) || HEAD_PHASE_RC=$?
if [ "$HEAD_PHASE_RC" -ne 0 ]; then exit "$HEAD_PHASE_RC"; fi

quiet_gate "after head"

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
# The conditions go in the report, not just in the log. A number that does not
# record what produced it is indistinguishable from one produced under good
# conditions, and a reader weeks later cannot reconstruct the difference. This
# block is what makes a quoted figure auditable: which refs, which targets and
# features (both are overridable, so the defaults are not a guarantee), which
# resolution, whether the machine was isolated, and how quiet it actually was.
echo ""
echo "=== Run conditions ==="
echo "  base: $BASE_REF ($BASE_SHA)   head: $HEAD_REF ($HEAD_SHA)"
echo "  resolution: ${QUICK_FLAGS:---full}"
echo "  targets: lattice-inference:$BENCHES_INFERENCE, lattice-embed:$BENCHES_EMBED"
echo "  inference features: ${CARGO_FEATURES_INFERENCE:-<none>}"
echo "  filters: inference='${BENCH_GROUPS_INFERENCE:-<all>}' embed='${BENCH_GROUPS_EMBED:-<all>}'"
echo "  enforcement: $([ "$FAIL_ON_REGRESSION" = "1" ] && echo "--fail-on-regression (gate status propagated)" || echo "report-only (gate status printed, exit 0)")"
echo "  locks:"
echo "$LOCK_SUMMARY"
echo "  ambient load:"
echo "$QUIET_SAMPLES" | sed 's/^/    /'

echo ""
echo "=== Full gate report ==="
GATE_ARGS=(--baseline-name compare-base)
if [ -s "$INFO_GROUPS_FILE" ]; then
  GATE_ARGS+=(--informational-groups-file "$INFO_GROUPS_FILE")
fi
if [ "$FAIL_ON_REGRESSION" = "1" ]; then
  # Ask the gate to distinguish "nothing regressed" from "nothing was
  # measured". It is the only party that can: it parses the comparisons and
  # knows how many were judgeable. Testing for the criterion DIRECTORY here
  # cannot work — this script creates that directory itself before benching.
  GATE_ARGS+=(--require-measurements)
fi

GATE_RC=0
if [ -d "$HEAD_DIR/target/criterion" ]; then
  python3 "$REPO/scripts/perf-bench-gate.py" "$HEAD_DIR/target/criterion" "local-compare" "${GATE_ARGS[@]}" 2>&1 || GATE_RC=$?
else
  # Cannot happen via the normal path (the directory is created above), but a
  # missing root must not read as a pass under --fail-on-regression.
  GATE_RC=2
fi

echo ""
echo "Done. Base=$BASE_REF ($BASE_SHA), Head=$HEAD_REF ($HEAD_SHA)"

if [ "$FAIL_ON_REGRESSION" = "1" ] && [ "$GATE_RC" -ne 0 ]; then
  # Exit 1 is a confirmed regression; exit 2 is the gate refusing to certify a
  # run it could not judge (no comparison data, or nothing gating). Both must
  # fail the caller: a green exit standing in for evidence that was never
  # produced is the exact defect this flag exists to remove.
  if [ "$GATE_RC" = "2" ]; then
    echo "bench-compare: gate could not judge this run — no usable measurements." >&2
  else
    echo "bench-compare: gate reported a confirmed regression (exit $GATE_RC)." >&2
  fi
  exit "$GATE_RC"
fi

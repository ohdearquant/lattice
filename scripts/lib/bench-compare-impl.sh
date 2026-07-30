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
# Uses detached git worktrees for both refs so your working tree stays untouched.
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
# scripts/perf-bench-gate.py --selftest); their benchmarks are still fully
# measured and rendered — the informational section plus the
# all-measurements table record every number — but classified informational,
# so they cannot produce a FAIL verdict. Every non-demoted target this script
# benches (the lattice-inference one) is classified gating in --quick.
#
# Criterion 0.5 permits `/` in group names and uses the same character to join
# group/function/parameter in `--list`, so a flat listing cannot recover the
# group boundary. It also stores every target under one target/criterion tree
# by default. This script instead assigns each bench target a separate
# CRITERION_HOME and passes the exact `<crate>:<bench-target>` key to the gate.
# Classification therefore follows a real target boundary rather than a
# guessed string prefix, and same-named groups in different targets cannot
# affect one another.
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

# A caller may hand us an inherited git environment (git exports GIT_INDEX_FILE to
# hooks as a RELATIVE path, and GIT_DIR/GIT_WORK_TREE arrive empty). The worktree
# add/remove calls below write a git index, so an inherited relative GIT_INDEX_FILE
# would resolve against our cwd and hit the caller's real index instead. Nothing in
# this script needs the caller's index state.
unset GIT_INDEX_FILE GIT_DIR GIT_WORK_TREE

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
QUICK_FLAGS="--quick"  # adaptive two-point sample, ~2 min total
RUN_STARTED_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
if [ -n "${BENCH_HOST_ID:-}" ]; then
  RUN_HOST_ID="configured:${BENCH_HOST_ID}"
else
  RUN_HOST_ID="$(python3 "$REPO/scripts/lib/bench-host-id.py")"
fi
RUN_OS="$(uname -srm)"
PROVENANCE_FILE="$REPO/.cache/bench-run-provenance.txt"

# Parse flags before any enforcement-sensitive setup. Both are optional and may
# appear in either order, but they must precede the positional BASE/HEAD pair:
# the first non-flag argument ends flag parsing. A flag written AFTER a ref is
# rejected rather than silently taken as a ref. Use `--` to pass a ref that
# legitimately begins with a dash.
FAIL_ON_REGRESSION=0
AFTER_DDASH=0
while [ $# -gt 0 ]; do
  case "${1:-}" in
    --full)
      QUICK_FLAGS=""
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

# --- Machine-state and ambient-load gates ---
# A lock excludes peers; it says nothing about ambient load, thermal pressure,
# power source, or an operator actively using the machine. These checkpoints
# settle the macOS host before every sample, then gate AC power, thermal state,
# HID idle time, and current CPU idle. Linux CI runners have no repository
# equivalent for the macOS hardware probes, so that limitation is recorded
# explicitly while the portable CPU-idle gate still applies.
QUIET_SAMPLES=""
MACHINE_STATE_SAMPLES=""
machine_state_probe() {
  local label="$1" platform record rc=0
  if ! platform="$(uname -s)"; then
    echo "bench-compare: could not identify the host platform — refusing." >&2
    exit 2
  fi
  if [ "$platform" = "Darwin" ]; then
    record="$(
      python3 "$REPO/scripts/perf_governor.py" \
        --checkpoint \
        --label "$label" \
        --cooldown 30 \
        --afk-threshold 30
    )" || rc=$?
  else
    record="$(
      python3 "$REPO/scripts/lib/machine-state-probe.py" --label "$label"
    )" || rc=$?
  fi
  echo "[state] $label: $record"
  MACHINE_STATE_SAMPLES="${MACHINE_STATE_SAMPLES}${MACHINE_STATE_SAMPLES:+
}$record"
  if [ "$rc" -ne 0 ] && [ "$FAIL_ON_REGRESSION" = "1" ]; then
    echo "bench-compare: machine-state checkpoint '$label' failed — refusing to certify this A/B." >&2
    exit 2
  fi
}

quiet_gate() {
  local label="$1" line rc=0
  machine_state_probe "$label"
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

# Resolve both display and audit identities before measuring.
if ! BASE_FULL_SHA="$(
  git -C "$REPO" rev-parse --verify --end-of-options "${BASE_REF}^{commit}" 2>/dev/null
)"; then
  echo "bench-compare: base ref '$BASE_REF' is not a commit — refusing." >&2
  exit 2
fi
if ! HEAD_FULL_SHA="$(
  git -C "$REPO" rev-parse --verify --end-of-options "${HEAD_REF}^{commit}" 2>/dev/null
)"; then
  echo "bench-compare: head ref '$HEAD_REF' is not a commit — refusing." >&2
  exit 2
fi
BASE_SHA="${BASE_FULL_SHA:0:10}"
HEAD_SHA="${HEAD_FULL_SHA:0:10}"

HEAD_WT="$REPO/.cache/bench-compare-head"
if [ "$HEAD_REF" = "HEAD" ]; then
  HEAD_MODE="detached snapshot worktree"
else
  HEAD_MODE="detached worktree"
fi
HEAD_DIR="$HEAD_WT"
GATE_SCRIPT="$REPO/scripts/perf-bench-gate.py"

print_execution_provenance() {
  echo "  head arm: $HEAD_MODE"
  echo "  gate: scripts/perf-bench-gate.py from the invoking checkout"
}

require_commit_clean_head() {
  if [ "$HEAD_REF" != "HEAD" ]; then
    return
  fi
  local current_head status
  if ! current_head="$(
    git -C "$REPO" rev-parse --verify --end-of-options 'HEAD^{commit}' 2>/dev/null
  )"; then
    echo "bench-compare: cannot resolve the invoking HEAD commit — refusing." >&2
    exit 2
  fi
  if [ "$current_head" != "$HEAD_FULL_SHA" ]; then
    echo "bench-compare: the invoking HEAD commit changed during the run — refusing." >&2
    echo "  expected $HEAD_FULL_SHA" >&2
    echo "  observed $current_head" >&2
    exit 2
  fi
  if ! status="$(git -C "$REPO" status --porcelain=v1 --untracked-files=normal)"; then
    echo "bench-compare: cannot inspect the invoking HEAD worktree — refusing." >&2
    exit 2
  fi
  if [ -n "$status" ]; then
    echo "bench-compare: the invoking HEAD worktree is not commit-clean — refusing." >&2
    echo "  Commit or remove tracked, staged, and untracked changes so the measured" >&2
    echo "  source is exactly reconstructible from the recorded head SHA." >&2
    exit 2
  fi
}

echo "=== bench-compare: $BASE_REF ($BASE_SHA) vs $HEAD_REF ($HEAD_SHA) ==="
print_execution_provenance
require_commit_clean_head
quiet_gate "before base"

# --- Keep Spotlight out of the benchmark build trees ---
# .cache protects the detached base/head worktrees. The separate target marker
# protects build artifacts created by callers outside these detached worktrees.
# Recreate both every run because deleting either tree deletes its own marker.
# Inert on non-macOS. Fail-closed: refuse to measure without the protection
# rather than emit numbers that look trustworthy and are not.
"$REPO/scripts/lib/ensure-noindex-marker.sh" "$REPO/.cache"
"$REPO/scripts/lib/ensure-noindex-marker.sh" "$REPO/target"

# --- Worktree for base ref ---
WT="$REPO/.cache/bench-compare-base"
if [ -d "$WT" ]; then
  git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
fi
git -C "$REPO" worktree add --detach --end-of-options "$WT" "$BASE_REF" 2>&1 | tail -1

command_version() {
  local directory="$1"; shift
  local output
  if ! output="$(cd "$directory" && "$@" 2>&1)" || [ -z "$output" ]; then
    printf '%s' "unavailable"
    return
  fi
  printf '%s' "$output" | tr '\n' ';'
}

criterion_version() {
  local lockfile="$1/Cargo.lock"
  if [ ! -f "$lockfile" ]; then
    printf '%s' "unavailable"
    return
  fi
  awk '
    $0 == "name = \"criterion\"" { in_criterion = 1; next }
    in_criterion && /^version = "/ {
      value = $0
      sub(/^version = "/, "", value)
      sub(/"$/, "", value)
      print value
      exit
    }
    in_criterion && /^\[\[package\]\]/ { exit }
  ' "$lockfile"
}

BASE_RUSTC="$(command_version "$WT" rustc --version)"
BASE_CARGO="$(command_version "$WT" cargo --version)"
BASE_CRITERION="$(criterion_version "$WT")"
[ -n "$BASE_CRITERION" ] || BASE_CRITERION="unavailable"

cleanup() {
  git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || true
  git -C "$REPO" worktree remove --force "$HEAD_WT" 2>/dev/null || true
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
BENCH_BASELINE_NAME="compare-base"

# Keep Criterion evidence target-qualified without giving up Cargo's shared
# compilation target. CRITERION_HOME controls only Criterion's report/baseline
# tree; Cargo continues to use each worktree's normal target directory.
BENCH_CRITERION_ROOT="$REPO/.cache/bench-compare-criterion"
BASE_INFERENCE_CRITERION_ROOT="$BENCH_CRITERION_ROOT/base/inference/criterion"
BASE_EMBED_CRITERION_ROOT="$BENCH_CRITERION_ROOT/base/embed/criterion"
INFERENCE_CRITERION_ROOT="$BENCH_CRITERION_ROOT/head/inference/criterion"
EMBED_CRITERION_ROOT="$BENCH_CRITERION_ROOT/head/embed/criterion"
mkdir -p \
  "$BASE_INFERENCE_CRITERION_ROOT" "$BASE_EMBED_CRITERION_ROOT" \
  "$INFERENCE_CRITERION_ROOT" "$EMBED_CRITERION_ROOT"
if [ "$FAIL_ON_REGRESSION" = "1" ]; then
  python3 "$GATE_SCRIPT" "$BASE_INFERENCE_CRITERION_ROOT" \
    --baseline-name "$BENCH_BASELINE_NAME" --prepare-baseline-copy
  python3 "$GATE_SCRIPT" "$BASE_EMBED_CRITERION_ROOT" \
    --baseline-name "$BENCH_BASELINE_NAME" --prepare-baseline-copy
fi

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
# that is the only evidence available at the point where the run's INTENT is
# known. Downstream, each target has its own Criterion root, but an empty root
# still cannot prove a benchmark was supposed to populate it.
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
    run_bench "time:" env CRITERION_HOME="$BASE_INFERENCE_CRITERION_ROOT" cargo bench -p lattice-inference --bench "$BENCHES_INFERENCE" ${CARGO_FEATURES_INFERENCE:+--features "$CARGO_FEATURES_INFERENCE"} -- ${BENCH_GROUPS_INFERENCE:+"$BENCH_GROUPS_INFERENCE"} --save-baseline "$BENCH_BASELINE_NAME" --noplot $QUICK_FLAGS
    require_measured "base lattice-inference:$BENCHES_INFERENCE" "$BENCH_RC" "$BENCH_LINES"
  else
    require_measured "base lattice-inference:$BENCHES_INFERENCE build (--no-run)" 1
    echo "  ($BENCHES_INFERENCE not present on $BASE_SHA — skipping)"
  fi
  run_bench "time:" env CRITERION_HOME="$BASE_EMBED_CRITERION_ROOT" cargo bench -p lattice-embed --bench "$BENCHES_EMBED" -- ${BENCH_GROUPS_EMBED:+"$BENCH_GROUPS_EMBED"} --save-baseline "$BENCH_BASELINE_NAME" --noplot $QUICK_FLAGS
  require_measured "base lattice-embed:$BENCHES_EMBED" "$BENCH_RC" "$BENCH_LINES"
) || BASE_PHASE_RC=$?
# `exit` inside `( ... )` leaves the SUBSHELL, so the status has to be caught
# and re-raised here or the refusal above is itself swallowed.
if [ "$BASE_PHASE_RC" -ne 0 ]; then exit "$BASE_PHASE_RC"; fi

quiet_gate "between phases"

# --- Copy base criterion data to HEAD's target ---
echo ""
echo "--- Building + benching HEAD ($HEAD_SHA) ---"

if [ -d "$HEAD_WT" ]; then
  git -C "$REPO" worktree remove --force "$HEAD_WT" 2>/dev/null || rm -rf "$HEAD_WT"
fi
git -C "$REPO" worktree add --detach --end-of-options \
  "$HEAD_WT" "$HEAD_FULL_SHA" 2>&1 | tail -1

HEAD_RUSTC="$(command_version "$HEAD_DIR" rustc --version)"
HEAD_CARGO="$(command_version "$HEAD_DIR" cargo --version)"
HEAD_CRITERION="$(criterion_version "$HEAD_DIR")"
[ -n "$HEAD_CRITERION" ] || HEAD_CRITERION="unavailable"

copy_base_artifacts() {
  local what="$1"; shift
  local rc=0
  rsync "$@" 2>/dev/null || rc=$?
  if [ "$rc" -ne 0 ] && [ "$FAIL_ON_REGRESSION" = "1" ]; then
    echo "bench-compare: $what failed (rsync exit $rc) — refusing to certify a partial A/B." >&2
    return 2
  fi
  return 0
}

prepare_target_root() {
  local target="$1" base_root="$2" head_root="$3"
  if [ "$FAIL_ON_REGRESSION" = "1" ]; then
    python3 "$GATE_SCRIPT" "$head_root" \
      --baseline-name "$BENCH_BASELINE_NAME" --prepare-baseline-copy
  fi
  copy_base_artifacts "$target selected baseline copy" \
    -a "$base_root/" "$head_root/" \
    --include="**/$BENCH_BASELINE_NAME/**" --include='*/' --exclude='*'
  if [ "$FAIL_ON_REGRESSION" = "1" ]; then
    python3 "$GATE_SCRIPT" "$head_root" \
      --baseline-name "$BENCH_BASELINE_NAME" --prepare-head
  fi
}

prepare_target_root \
  "lattice-inference:$BENCHES_INFERENCE" \
  "$BASE_INFERENCE_CRITERION_ROOT" "$INFERENCE_CRITERION_ROOT"
prepare_target_root \
  "lattice-embed:$BENCHES_EMBED" \
  "$BASE_EMBED_CRITERION_ROOT" "$EMBED_CRITERION_ROOT"

HEAD_PHASE_RC=0
(
  cd "$HEAD_DIR"
  if cargo bench -p lattice-inference --bench "$BENCHES_INFERENCE" ${CARGO_FEATURES_INFERENCE:+--features "$CARGO_FEATURES_INFERENCE"} --no-run 2>/dev/null; then
    run_bench "time:|change:" env CRITERION_HOME="$INFERENCE_CRITERION_ROOT" cargo bench -p lattice-inference --bench "$BENCHES_INFERENCE" ${CARGO_FEATURES_INFERENCE:+--features "$CARGO_FEATURES_INFERENCE"} -- ${BENCH_GROUPS_INFERENCE:+"$BENCH_GROUPS_INFERENCE"} --baseline "$BENCH_BASELINE_NAME" --noplot $QUICK_FLAGS
    require_measured "head lattice-inference:$BENCHES_INFERENCE" "$BENCH_RC" "$BENCH_LINES"
  else
    require_measured "head lattice-inference:$BENCHES_INFERENCE build (--no-run)" 1
    echo "  ($BENCHES_INFERENCE not present on $HEAD_SHA — skipping)"
  fi
  run_bench "time:|change:" env CRITERION_HOME="$EMBED_CRITERION_ROOT" cargo bench -p lattice-embed --bench "$BENCHES_EMBED" -- ${BENCH_GROUPS_EMBED:+"$BENCH_GROUPS_EMBED"} --baseline "$BENCH_BASELINE_NAME" --noplot $QUICK_FLAGS
  require_measured "head lattice-embed:$BENCHES_EMBED" "$BENCH_RC" "$BENCH_LINES"
) || HEAD_PHASE_RC=$?
if [ "$HEAD_PHASE_RC" -ne 0 ]; then exit "$HEAD_PHASE_RC"; fi

quiet_gate "after head"
require_commit_clean_head

write_run_provenance() {
  local finished_utc criterion_mode enforcement
  finished_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if [ -n "$QUICK_FLAGS" ]; then
    criterion_mode="quick"
  else
    criterion_mode="full"
  fi
  if [ "$FAIL_ON_REGRESSION" = "1" ]; then
    enforcement="fail-on-regression"
  else
    enforcement="report-only"
  fi

  mkdir -p "$(dirname "$PROVENANCE_FILE")"
  {
    printf 'schema=lattice-bench-provenance-v1\n'
    printf 'started_utc=%s\n' "$RUN_STARTED_UTC"
    printf 'finished_utc=%s\n' "$finished_utc"
    printf 'host_id=%s\n' "$RUN_HOST_ID"
    printf 'os=%s\n' "$RUN_OS"
    printf 'base_ref=%s\n' "$BASE_REF"
    printf 'base_sha=%s\n' "$BASE_FULL_SHA"
    printf 'head_ref=%s\n' "$HEAD_REF"
    printf 'head_sha=%s\n' "$HEAD_FULL_SHA"
    printf 'head_mode=detached-worktree\n'
    printf 'base_rustc=%s\n' "$BASE_RUSTC"
    printf 'head_rustc=%s\n' "$HEAD_RUSTC"
    printf 'base_cargo=%s\n' "$BASE_CARGO"
    printf 'head_cargo=%s\n' "$HEAD_CARGO"
    printf 'base_criterion=%s\n' "$BASE_CRITERION"
    printf 'head_criterion=%s\n' "$HEAD_CRITERION"
    printf 'criterion_mode=%s\n' "$criterion_mode"
    printf 'baseline_name=%s\n' "$BENCH_BASELINE_NAME"
    printf 'targets=lattice-inference:%s, lattice-embed:%s\n' \
      "$BENCHES_INFERENCE" "$BENCHES_EMBED"
    printf 'inference_features=%s\n' "${CARGO_FEATURES_INFERENCE:-<none>}"
    printf "filters=inference='%s' embed='%s'\n" \
      "${BENCH_GROUPS_INFERENCE:-<all>}" "${BENCH_GROUPS_EMBED:-<all>}"
    printf 'enforcement=%s\n' "$enforcement"
    while IFS= read -r line; do
      [ -n "$line" ] && printf 'lock=%s\n' "$line"
    done <<< "$LOCK_SUMMARY"
    while IFS= read -r line; do
      [ -n "$line" ] && printf 'ambient=%s\n' "$line"
    done <<< "$QUIET_SAMPLES"
    while IFS= read -r line; do
      [ -n "$line" ] && printf 'machine_state=%s\n' "$line"
    done <<< "$MACHINE_STATE_SAMPLES"
  } > "$PROVENANCE_FILE"
}

write_run_provenance

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
print_execution_provenance
echo "  resolution: ${QUICK_FLAGS:---full}"
echo "  targets: lattice-inference:$BENCHES_INFERENCE, lattice-embed:$BENCHES_EMBED"
echo "  inference features: ${CARGO_FEATURES_INFERENCE:-<none>}"
echo "  filters: inference='${BENCH_GROUPS_INFERENCE:-<all>}' embed='${BENCH_GROUPS_EMBED:-<all>}'"
echo "  enforcement: $([ "$FAIL_ON_REGRESSION" = "1" ] && echo "--fail-on-regression (gate status propagated)" || echo "report-only (gate status printed, exit 0)")"
echo "  locks:"
echo "$LOCK_SUMMARY"
echo "  ambient load:"
echo "$QUIET_SAMPLES" | sed 's/^/    /'
echo "  machine state:"
echo "$MACHINE_STATE_SAMPLES" | sed 's/^/    /'

echo ""
echo "=== Target-qualified gate reports ==="
GATE_RC=0
run_target_gate() {
  local target="$1" criterion_root="$2"
  local gate_rc=0 policy_rc=0 policy_invalid=0
  local gate_args=(
    --baseline-name compare-base
    --target "$target"
    --provenance-file "$PROVENANCE_FILE"
  )

  if [ -n "$QUICK_FLAGS" ]; then
    if "$REPO/scripts/lib/bench-informational-targets.sh" \
         --is-informational "$target"; then
      gate_args+=(--informational-target "$target")
    else
      policy_rc=$?
      if [ "$policy_rc" -ne 1 ]; then
        echo "bench-compare: informational-target policy could not classify '$target'." >&2
        policy_invalid=1
      fi
    fi
  fi

  if [ "$FAIL_ON_REGRESSION" = "1" ]; then
    # Each target has its own root, so completeness is checked per intended
    # target rather than inferred from whichever comparisons survived in a
    # shared tree.
    gate_args+=(--require-measurements)
    gate_args+=(--require-provenance)
  fi

  if [ -d "$criterion_root" ]; then
    python3 "$GATE_SCRIPT" \
      "$criterion_root" "local-compare/$target" "${gate_args[@]}" 2>&1 || gate_rc=$?
  else
    gate_rc=2
  fi
  if [ "$policy_invalid" -eq 1 ]; then
    gate_rc=2
  fi

  if [ "$gate_rc" -eq 2 ]; then
    GATE_RC=2
  elif [ "$gate_rc" -ne 0 ] && [ "$GATE_RC" -eq 0 ]; then
    GATE_RC="$gate_rc"
  fi
}

run_target_gate \
  "lattice-inference:$BENCHES_INFERENCE" "$INFERENCE_CRITERION_ROOT"
run_target_gate \
  "lattice-embed:$BENCHES_EMBED" "$EMBED_CRITERION_ROOT"

echo ""
echo "Done. Base=$BASE_REF ($BASE_SHA), Head=$HEAD_REF ($HEAD_SHA)"

if [ "$FAIL_ON_REGRESSION" = "1" ] && [ "$GATE_RC" -ne 0 ]; then
  # Exit 1 is a confirmed regression; exit 2 is the gate refusing to certify a
  # run it could not judge (no usable comparison data or invalid policy). Both must
  # fail the caller: a green exit standing in for evidence that was never
  # produced is the exact defect this flag exists to remove.
  if [ "$GATE_RC" = "2" ]; then
    echo "bench-compare: gate could not judge this run — no usable measurements." >&2
  else
    echo "bench-compare: gate reported a confirmed regression (exit $GATE_RC)." >&2
  fi
  exit "$GATE_RC"
fi

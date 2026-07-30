#!/usr/bin/env bash
# bench-informational-targets.sh — resolve quick-mode informational targets.
#
# Criterion 0.5 accepts any nonempty group name, including `/`. Its terse
# listing joins group, function and parameter with that same character, so a
# listing cannot recover the group boundary without guessing. Classification
# therefore stays at the policy's real unit: the `<crate>:<bench-target>` key.
# bench-compare gives each target its own CRITERION_HOME and passes that exact
# key to perf-bench-gate.py; this helper only answers whether the target is in
# the reviewed quick-mode demotion manifest.
#
# `--print-targets` emits the normalized manifest for the selftest. The
# `--is-informational <target>` predicate exits 0 for a listed target and 1 for
# an unlisted target. Invalid manifest input exits 2, so malformed suppression
# policy cannot silently broaden the informational set.
#
# FULL mode (`bench-compare.sh --full`, or `make bench-gate`) ignores this
# mechanism entirely: every group those paths bench is classified gating,
# with no demotion. Classification is not enforcement — `bench-compare.sh`
# discards its gate's exit status unless the caller passes
# --fail-on-regression, so a FAIL verdict becomes a non-zero exit only under
# that flag or via `make bench-gate`. Both bench the same two targets rather
# than the workspace's full bench set, and --full additionally honors
# bench-compare.sh's BENCH_GROUPS_* filters.
#
# Two automated full-resolution jobs run on main, and they do different
# things. bench-update.yml saves baselines and publishes historical trend
# comparisons, including a "Worst step-regression" headline; it does not
# invoke perf-bench-gate.py and takes no regression-specific fail action, so
# it reports rather than enforces. perf-postmerge-gate.yml is the enforcing
# one: it runs `bench-compare.sh --full --fail-on-regression` against the
# merged commit's own parent and fails on a confirmed regression.
set -euo pipefail

MANIFEST="${INFO_TARGETS_MANIFEST:-$(dirname "$0")/bench-quick-informational-targets.txt}"

manifest_targets() {
  LC_ALL=C awk '
    {
      sub(/\r$/, "")
      sub(/[[:space:]]*#.*/, "")
      sub(/^[[:space:]]+/, "")
      sub(/[[:space:]]+$/, "")
      if ($0 == "") {
        next
      }
      if ($0 ~ /[[:space:]]/) {
        printf "error: malformed informational target entry at line %d: %s\n", NR, $0 > "/dev/stderr"
        invalid = 1
        next
      }
      print
    }
    END {
      if (invalid) {
        exit 2
      }
    }
  ' "$MANIFEST" | LC_ALL=C sort -u
}

if [ "${1:-}" = "--print-targets" ]; then
  manifest_targets
  exit 0
fi

if [ "${1:-}" != "--is-informational" ] || [ "$#" -ne 2 ]; then
  echo "usage: bench-informational-targets.sh --print-targets | --is-informational <crate>:<bench-target>" >&2
  exit 2
fi

target="$2"
case "$target" in
  ''|*[[:space:]]*)
    echo "error: invalid bench target key: '$target'" >&2
    exit 2
    ;;
esac

targets="$(manifest_targets)" || exit $?
if printf '%s\n' "$targets" | LC_ALL=C grep -qxF "$target"; then
  exit 0
fi
exit 1

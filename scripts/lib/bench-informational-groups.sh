#!/usr/bin/env bash
# bench-informational-groups.sh — shell side of the quick-mode informational
# demotion (lattice#714; target-level policy per lattice#1060, 2026-07-19).
#
# The set of demoted bench targets lives in ONE validated manifest:
# scripts/lib/bench-quick-informational-targets.txt. Given a target key
# (`<crate>:<bench-target>`) and that target's Criterion `--list` output (a
# file path as $2, or stdin), this script prints the group names to treat
# as informational in QUICK mode: ALL of the listing's top-level groups
# when the target is in the manifest, nothing otherwise. Deriving the
# group set from the listing is deliberate under target-level semantics —
# the demotion covers the target, so a group added to a demoted target
# later is informational by policy, while every other target stays gated
# because its key is absent from the manifest.
#
# scripts/bench-compare.sh invokes this file for production runs.
# scripts/perf-bench-gate.py --selftest invokes the same file three ways:
# `--print-targets` (compare the manifest against the reviewed
# expectation, so a manifest-only or expectation-only edit fails), a
# demoted-target probe (all listing groups emitted), and a
# non-demoted-target probe (nothing emitted — the cross-target guarantee).
# INFO_TARGETS_MANIFEST overrides the manifest path for those controlled
# probes; production never sets it.
#
# FULL mode (`bench-compare.sh --full`, or `make bench-gate`) ignores this
# mechanism entirely: every group those paths bench gates, with no demotion.
# Both bench the same two targets rather than the workspace's full bench set,
# and --full additionally honors bench-compare.sh's BENCH_GROUPS_* filters.
# Both are manual/local today. bench-update.yml is the one automated
# full-resolution job on main, and it collects baselines without comparing,
# gating, or alerting (#1105 tracks the missing gate lane).
set -euo pipefail

MANIFEST="${INFO_TARGETS_MANIFEST:-$(dirname "$0")/bench-quick-informational-targets.txt}"

manifest_targets() {
  sed 's/#.*//' "$MANIFEST" | awk 'NF { print $1 }' | sort
}

if [ "${1:-}" = "--print-targets" ]; then
  manifest_targets
  exit 0
fi

if [ "${1:-}" = "--list-groups" ]; then
  # Unconditional group extraction, no manifest check — used by
  # resolve-informational-groups.sh to see every target's groups
  # (both demoted and gated) before deciding what stays informational.
  input="${2:-/dev/stdin}"
  awk -F/ '/: benchmark$/{print $1}' "$input" | sort -u
  exit 0
fi

target="${1:?usage: bench-informational-groups.sh <crate>:<bench-target> [listing-file] (or --print-targets|--list-groups)}"
input="${2:-/dev/stdin}"

if manifest_targets | grep -qxF "$target"; then
  awk -F/ '/: benchmark$/{print $1}' "$input" | sort -u
fi

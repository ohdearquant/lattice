#!/usr/bin/env bash
# bench-informational-groups.sh — the shell-side half of lattice#714's
# quick-mode informational-groups contract.
#
# Given Criterion `--list` output (a file path as $1, or stdin when no arg is
# given), print the intersection of the fixed, reviewed allowlist below with
# the top-level group names actually present in that listing — one name per
# line, sorted.
#
# scripts/bench-compare.sh sources this file for the real run so the exact
# same array and intersection logic feeds the gate. scripts/perf-bench-gate.py
# --selftest also invokes this script directly (as a subprocess, against a
# controlled listing) so a shell-only regression here — e.g. a name added
# only to this array — fails the selftest instead of staying invisible. An
# earlier fixture kept its own hardcoded copy of this list in Python and
# never exercised this file at all; extracting the logic here closed that.
#
# Adding a group to the allowlist requires the same kind of same-toolchain
# A/A quantitative evidence that justified these two (see scripts/bench-
# compare.sh's "Quick-mode informational-groups" comment), reviewed in a PR —
# never derived automatically from `--list`, which would silently exempt
# every future group added to the target.
set -euo pipefail

INFO_GROUPS_ALLOWLIST=(
  "simd_dot_product"
  "simd_cosine_similarity"
)

input="${1:-/dev/stdin}"
listed_groups=$(awk -F/ '/: benchmark$/{print $1}' "$input" | sort -u)

for grp in "${INFO_GROUPS_ALLOWLIST[@]}"; do
  if grep -qxF "$grp" <<< "$listed_groups"; then
    echo "$grp"
  fi
done

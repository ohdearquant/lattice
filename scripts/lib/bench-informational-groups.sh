#!/usr/bin/env bash
# bench-informational-groups.sh — the shell-side half of lattice#714's
# quick-mode informational-groups contract.
#
# Given Criterion `--list` output (a file path as $1, or stdin when no arg is
# given), print the intersection of the fixed, reviewed allowlist below with
# the top-level group names actually present in that listing — one name per
# line, sorted.
#
# scripts/bench-compare.sh invokes this file for the real run so the exact
# same array and intersection logic feeds the gate. scripts/perf-bench-gate.py
# --selftest invokes this script two ways: with --print-allowlist (dump the
# raw array, no listing) so the Python-side expected set is compared against
# the ACTUAL array — a name added only here, or only there, fails the
# selftest — and as the intersection subprocess against a controlled listing
# so the production code path is exercised end-to-end. An earlier fixture
# kept its own hardcoded copy of this list in Python and never exercised
# this file at all; extracting the logic here closed that.
#
# Adding a group to the allowlist requires the same kind of same-toolchain
# A/A quantitative evidence that justified every current entry (see scripts/
# bench-compare.sh's "Quick-mode informational-groups" comment), reviewed in
# a PR — never derived automatically from `--list`, which would silently
# exempt every future group added to the target.
set -euo pipefail

INFO_GROUPS_ALLOWLIST=(
  "simd_dot_product"
  "simd_cosine_similarity"
  # Added 2026-07-13: two same-commit A/A runs, each inside an exclusive
  # machine-wide bench window (/tmp/lion-bench-window.lock), still produced
  # confirmed-CI FAIL rows (>7%) in these embed micro-groups — with DISJOINT
  # failing groups across the two runs, the signature of a noise floor above
  # the quick-mode gate rather than a regression. Evidence tables in the PR
  # that added these entries.
  "int8_batch_cosine"
  "int4_cosine_distance"
  "simd_batch_cosine_non_normalized_query"
)

if [ "${1:-}" = "--print-allowlist" ]; then
  printf '%s\n' "${INFO_GROUPS_ALLOWLIST[@]}" | sort
  exit 0
fi

input="${1:-/dev/stdin}"
listed_groups=$(awk -F/ '/: benchmark$/{print $1}' "$input" | sort -u)

for grp in "${INFO_GROUPS_ALLOWLIST[@]}"; do
  if grep -qxF "$grp" <<< "$listed_groups"; then
    echo "$grp"
  fi
done

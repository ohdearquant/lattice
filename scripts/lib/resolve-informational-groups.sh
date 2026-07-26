#!/usr/bin/env bash
# resolve-informational-groups.sh — collision-free informational set
# (lattice#1060 follow-up: target-level demotion, bare-name group namespace).
#
# The quick-mode informational-demotion policy is target-level (a
# `<crate>:<bench-target>` key in bench-quick-informational-targets.txt), but
# once bench-informational-groups.sh emits a target's group names they are
# bare strings with no target attribution. bench-compare.sh runs the helper
# once per bench target and used to concatenate every target's output into
# one flat file — so a group name demoted for one target silently exempted
# any identically-named group produced by a different, gated target from the
# regression gate.
#
# This resolver takes the union of demoted targets' group names and the
# union of gated (non-demoted) targets' group names as two separate files
# and prints the demoted set MINUS any name also present in the gated set —
# fail closed: a colliding name gates rather than staying informational.
# Any collision is reported on stderr, naming the group and both target sets,
# so a real collision is investigable rather than silent.
set -euo pipefail

demoted_file="${1:?usage: resolve-informational-groups.sh <demoted-groups-file> <demoted-target-label> <gated-groups-file> <gated-target-label>}"
demoted_label="${2:?usage: resolve-informational-groups.sh <demoted-groups-file> <demoted-target-label> <gated-groups-file> <gated-target-label>}"
gated_file="${3:?usage: resolve-informational-groups.sh <demoted-groups-file> <demoted-target-label> <gated-groups-file> <gated-target-label>}"
gated_label="${4:?usage: resolve-informational-groups.sh <demoted-groups-file> <demoted-target-label> <gated-groups-file> <gated-target-label>}"

sorted_nonblank() { grep -v '^$' "$1" | sort -u || true; }

collisions="$(comm -12 <(sorted_nonblank "$demoted_file") <(sorted_nonblank "$gated_file") || true)"

if [ -n "$collisions" ]; then
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    echo "warn: group '$name' is informational for '$demoted_label' but also" \
         "produced by gated target '$gated_label' — gating it instead of" \
         "suppressing (fail closed on the namespace collision)" >&2
  done <<< "$collisions"
fi

comm -23 <(sorted_nonblank "$demoted_file") <(sorted_nonblank "$gated_file")

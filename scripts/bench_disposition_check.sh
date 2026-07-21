#!/usr/bin/env bash
# ADR-058 bench-compare disposition check.
#
# Reads a PR description on stdin. Exit 0 = a substantive bench-compare
# disposition is present; exit 1 = absent. Pure text processing, no network,
# so it is unit-testable (tests/test_bench_disposition.py) and runnable
# locally before opening a PR:
#
#   gh pr view <N> --json body -q .body | scripts/bench_disposition_check.sh
#
# The bench-evidence workflow runs this from the BASE checkout under
# pull_request_target, so the logic that decides a merge is never taken from
# the PR's own tree.
set -euo pipefail

BODY=$(cat)

if ! printf '%s' "$BODY" | grep -q '[^[:space:]]'; then
  echo "empty description; ADR-058 requires a bench-compare disposition" >&2
  exit 1
fi

# The section opener must be a Markdown heading containing "bench-compare", not
# any prose mention: a Summary sentence like "run make bench-compare" must not
# satisfy the gate. tolower(), not IGNORECASE (a gawk extension that mawk, the
# Ubuntu runner default, silently ignores — turning the match case-sensitive).
# The section runs until a heading at the same or shallower depth, so a
# subheading under it counts as its content.
SECTION=$(printf '%s' "$BODY" | awk '
  function level(s) { if (match(s, /^#{1,6}[[:space:]]/)) { return RLENGTH - 1 } return 0 }
  {
    lv = level($0)
    if (found && lv > 0 && lv <= start_lv) { exit }
    if (!found && lv > 0 && tolower($0) ~ /bench-?compare/) { found = 1; start_lv = lv }
    if (found) { print }
  }
')

if ! printf '%s' "$SECTION" | grep -q '[^[:space:]]'; then
  echo "no bench-compare disposition heading found in the description" >&2
  echo "put the numbers under a heading, e.g.  ## bench-compare disposition" >&2
  exit 1
fi

# Drop the heading line before measuring content, so a bare "## bench-compare"
# with nothing beneath it cannot satisfy the gate. Keep a spaced copy for
# marker matching (markers like "no change" contain spaces) and a stripped
# copy for the length floor.
BODYTEXT=$(printf '%s' "$SECTION" | tail -n +2)
CONTENT=$(printf '%s' "$BODYTEXT" | tr -d '[:space:]')

# ADR-058's blessed minimal dispositions are legitimately terse and fall well
# under a raw length floor: the no-change one-liner lattice/CLAUDE.md blesses
# ("bench-compare showed no change (p > 0.05 on all groups)."), the one-line
# N/A this workflow's own comment promises for a doc-only change, and the
# compiled-out cfg-gate proof. An 80-char floor rejects all three, making the
# standard's own canonical disposition unsatisfiable. Accept an explicit
# disposition marker regardless of length; otherwise require enough content to
# be a real numeric disposition rather than a bare heading. All markers are
# multi-word phrases or contain "/", so none collides with an English word;
# no \b is used, which keeps the regex portable across GNU grep (the CI runner)
# and BSD grep (a local pre-PR run on macOS).
MARKERS='n/a|not required|not applicable|no( measurable)? change|no perf(ormance)? change|compiled out|identical effective source'
if printf '%s' "$BODYTEXT" | grep -qiE "$MARKERS"; then
  echo "bench-compare disposition present (explicit disposition marker; ${#CONTENT} chars)"
  exit 0
fi

if [ "${#CONTENT}" -lt 80 ]; then
  echo "bench-compare section present but essentially empty (${#CONTENT} chars of content)" >&2
  echo "state the numbers, or an explicit disposition (no change / N/A / compiled out)" >&2
  exit 1
fi

echo "bench-compare disposition present (${#CONTENT} chars of content)"

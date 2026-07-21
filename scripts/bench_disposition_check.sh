#!/usr/bin/env bash
# bench-compare disposition check (enforces the CLAUDE.md contributor rule).
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
  echo "empty description; a bench-compare disposition is required" >&2
  exit 1
fi

# The section opener must be a Markdown heading containing "bench-compare", not
# any prose mention: a Summary sentence like "run make bench-compare" must not
# satisfy the gate. tolower(), not IGNORECASE (a gawk extension that mawk, the
# Ubuntu runner default, silently ignores — turning the match case-sensitive).
# The section runs until a heading at the same or shallower depth, so a
# subheading under it counts as its content.
#
# A heading hidden inside a fenced code block or an HTML comment does NOT open
# the section: a comment renders invisibly and a fence renders as literal code,
# so counting either would let a PR satisfy a reviewer-facing gate with a
# disposition no reviewer sees. Comment state is resolved BEFORE fence state, so a
# fence delimiter that appears inside a comment cannot leak into the fence tracker
# and strand it open. Fences follow CommonMark: up to three leading spaces then a
# run of at least three of the same character (backtick or tilde); a closer must
# repeat the opener character, be at least as long, and carry nothing but trailing
# whitespace, so a shorter or mismatched fence line inside a longer block stays
# content instead of closing it early. Content inside a fence that opens AFTER a
# visible bench-compare heading still counts as section content, because a real
# bench table is normally fenced; the hidden state suppresses only heading
# detection, not membership in an already-open section. Headings are ATX only (a
# leading run of #); Setext underlines are not recognized.
SECTION=$(printf '%s' "$BODY" | awk '
  function level(s) { if (match(s, /^#{1,6}[[:space:]]/)) { return RLENGTH - 1 } return 0 }
  function lead_spaces(s,   n) { n = 0; while (substr(s, n + 1, 1) == " ") n++; return n }
  function run_len(s, ch,   n) { n = 0; while (substr(s, n + 1, 1) == ch) n++; return n }
  {
    line = $0

    was_in_comment = in_comment
    if (in_comment) { if (line ~ /-->/) in_comment = 0 }
    else if (line ~ /<!--/ && line !~ /-->/) { in_comment = 1 }

    if (!was_in_comment && !in_comment) {
      sp = lead_spaces(line)
      if (sp <= 3) {
        fbody = substr(line, sp + 1)
        fch = substr(fbody, 1, 1)
        if (fch == "`" || fch == "~") {
          rl = run_len(fbody, fch)
          if (rl >= 3) {
            if (!in_fence) {
              in_fence = 1; fence_char = fch; fence_len = rl
              if (found) print line
              next
            } else if (fch == fence_char && rl >= fence_len) {
              rest = substr(fbody, rl + 1); gsub(/[ \t]/, "", rest)
              if (rest == "") { in_fence = 0; if (found) print line; next }
            }
          }
        }
      }
    }

    hidden = (in_fence || was_in_comment)
    lv = hidden ? 0 : level(line)
    if (found && lv > 0 && lv <= start_lv) { exit }
    if (!found && lv > 0 && tolower(line) ~ /bench-?compare/) { found = 1; start_lv = lv }
    if (found) print line
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

# The blessed minimal dispositions are legitimately terse and fall well
# under a raw length floor: the no-change one-liner CLAUDE.md blesses
# ("bench-compare showed no change (p > 0.05 on all groups)."), the one-line
# N/A this workflow's own comment promises for a doc-only change, and the
# compiled-out cfg-gate proof. An 80-char floor rejects all three, making the
# standard's own canonical disposition unsatisfiable. Accept an explicit
# disposition marker regardless of length; otherwise require enough content to
# be a real numeric disposition rather than a bare heading. The whole marker
# alternative is anchored on both sides by a portable word boundary
# ([^[:alnum:]_], NOT the GNU-only \b) so a marker cannot fire as the prefix of
# a longer word: unanchored "no change" matched inside "no changelog" and let a
# body with no disposition pass (#1058 round-2 review). The boundary keeps the
# expression portable across GNU grep (the CI runner) and BSD grep (a local
# pre-PR run on macOS); grep -qi tests presence only, so consuming a boundary
# char is harmless.
MARKERS='(^|[^[:alnum:]_])(n/a|not required|not applicable|no( measurable)? change|no perf(ormance)? change|compiled out|identical effective source)($|[^[:alnum:]_])'
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

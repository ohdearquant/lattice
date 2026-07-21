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
# Ubuntu runner default, silently ignores, turning the match case-sensitive).
# The section runs until a heading at the same or shallower depth, so a
# subheading under it counts as its content. Headings are ATX only (a run of one
# to six #, indented at most three spaces per CommonMark); Setext underlines are
# not recognized.
#
# A heading only opens the section when it actually renders as a heading. Several
# constructs render it as something else, and each is tracked as a mutually
# exclusive "masked" region: while inside one, only that region's own terminator
# is inspected, so a delimiter for a different region is literal text and cannot
# desync the tracker.
#   fenced code block  a run of at least three backticks or tildes (indented at
#                      most three spaces); the closer must repeat the opener
#                      character, be at least as long, and carry only trailing
#                      whitespace, so a shorter or mismatched fence line stays
#                      content. Renders as literal code.
#   HTML comment       from <!-- to -->; renders invisibly.
#   raw HTML block     <pre>, <script>, <style>, <textarea>; contents are raw
#                      (not Markdown-parsed) until the matching close tag.
#   processing instr.  from <? to ?> (CommonMark HTML block type 3); raw.
#   declaration        from <! and a letter (e.g. <!DOCTYPE) to > (type 4); raw.
#   CDATA              from <![CDATA[ to ]]> (type 5); raw.
#   block-level HTML   a line beginning (indented at most three spaces) with an
#                      open or close tag from the CommonMark type-6 block-tag list
#                      (div, details, table, section, blockquote, ... the full set)
#                      masks to the next blank line, the CommonMark type-6 end
#                      condition. HTML block types 1 to 6 may interrupt a paragraph,
#                      so this masks unconditionally: a bench-compare heading buried
#                      in a block-level container does not render as a heading whether
#                      the container opens at a block boundary or directly under a run
#                      of prose, and trailing content on the opening line (<div>text)
#                      is masked too.
# CommonMark type-7 blocks -- any OTHER complete tag alone on a line, e.g. a void or
# inline tag such as <br>, <hr>, <img>, <span> -- are deliberately NOT tracked. Type 7
# is the one HTML-block type that cannot interrupt a paragraph, so masking it correctly
# requires tracking open-paragraph state across every other block construct (fences,
# comments, thematic breaks, Setext underlines, indented code, block quotes), which a
# line-oriented shell parser cannot track reliably. The lone residual is a heading hidden directly
# beneath such a tag; wrapping a disposition in a bare <br> takes a motivated author,
# which is outside this gate's threat model (honest mistakes and lazy skips), so it is
# a documented boundary rather than modeled. Because a non-type-6 tag alone on a line is
# left unmasked, an autolink such as <https://example.com> and inline HTML on a prose
# line such as <span>note</span> text still render normally and a heading after them
# opens the section.
# A trailing carriage return is stripped before any state transition, so a fence
# or tag delimiter on a CRLF line still matches. Content inside a masked region
# that opens AFTER a visible bench-compare heading still counts as section
# content, because a real bench table is often fenced; masking suppresses heading
# detection, not membership in an already-open section.
SECTION=$(printf '%s' "$BODY" | awk '
  function level(s,   m) {
    if (match(s, /^ {0,3}#{1,6}([ \t]|$)/)) {
      m = substr(s, 1, RLENGTH); sub(/^ +/, "", m); sub(/[ \t]*$/, "", m); return length(m)
    }
    return 0
  }
  {
    line = $0
    sub(/\r$/, "", line)

    if (in_fence) {
      if (match(line, /^ {0,3}(`{3,}|~{3,})[ \t]*$/)) {
        run = substr(line, RSTART, RLENGTH); sub(/^ +/, "", run); sub(/[ \t]+$/, "", run)
        if (substr(run, 1, 1) == fence_char && length(run) >= fence_len) in_fence = 0
      }
      if (found) print line
      next
    }
    if (in_comment) {
      if (line ~ /-->/) in_comment = 0
      if (found) print line
      next
    }
    if (in_raw_html) {
      if (index(tolower(line), "</" html_tag ">") > 0) in_raw_html = 0
      if (found) print line
      next
    }
    if (in_html_block) {
      if (line ~ /^[ \t]*$/) in_html_block = 0
      if (found) print line
      next
    }
    if (in_cdata) {
      if (index(line, "]]>") > 0) in_cdata = 0
      if (found) print line
      next
    }
    if (in_decl) {
      if (index(line, ">") > 0) in_decl = 0
      if (found) print line
      next
    }
    if (in_pi) {
      if (index(line, "?>") > 0) in_pi = 0
      if (found) print line
      next
    }

    if (match(line, /^ {0,3}(`{3,}|~{3,})/)) {
      run = substr(line, RSTART, RLENGTH); sub(/^ +/, "", run)
      in_fence = 1; fence_char = substr(run, 1, 1); fence_len = length(run)
      if (found) print line
      next
    }
    if (match(tolower(line), "^ {0,3}<(pre|script|style|textarea)([ \t/>]|$)")) {
      html_tag = substr(tolower(line), RSTART, RLENGTH)
      sub(/^ *</, "", html_tag); sub(/[^a-z].*$/, "", html_tag)
      if (index(tolower(line), "</" html_tag ">") == 0) in_raw_html = 1
      if (found) print line
      next
    }
    if (line ~ /<!--/ && line !~ /-->/) {
      in_comment = 1
      if (found) print line
      next
    }
    if (match(line, /^ {0,3}<!\[CDATA\[/)) {
      if (index(line, "]]>") == 0) in_cdata = 1
      if (found) print line
      next
    }
    if (match(line, /^ {0,3}<![a-zA-Z]/)) {
      if (index(line, ">") == 0) in_decl = 1
      if (found) print line
      next
    }
    if (match(line, /^ {0,3}<\?/)) {
      if (index(line, "?>") == 0) in_pi = 1
      if (found) print line
      next
    }
    if (match(tolower(line), "^ {0,3}</?(address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h1|h2|h3|h4|h5|h6|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)([ \t/>]|$)")) {
      in_html_block = 1
      if (found) print line
      next
    }

    lv = level(line)
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
# body with no disposition pass. The boundary keeps the
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

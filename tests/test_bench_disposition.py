#!/usr/bin/env python3
"""Regression tests for scripts/bench_disposition_check.sh (the CLAUDE.md bench-compare gate parser).

Each case pins a behaviour the gate must keep. The one that matters most is
``test_summary_prose_mention_does_not_satisfy``: a bench-compare mention in
Summary prose must not open the disposition section, only a real heading may.
That case fixes a parser that accepted Summary prose as the disposition.
"""
import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_disposition_check.sh"


def has_disposition(body: str) -> bool:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        input=body,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


class BenchDispositionCheck(unittest.TestCase):
    def test_empty_body_fails(self):
        self.assertFalse(has_disposition(""))
        self.assertFalse(has_disposition("   \n\t\n"))

    def test_summary_prose_mention_does_not_satisfy(self):
        body = (
            "## Summary\n"
            "This touches crates/embed. Run make bench-compare to see the numbers.\n"
            "## Test plan\n"
            "ok"
        )
        self.assertFalse(has_disposition(body))

    def test_real_heading_with_content_passes(self):
        body = (
            "## Summary\nx\n"
            "## bench-compare disposition\n"
            "A/B shows no change, p>0.05 on every group; 384=39us 768=73us 1024=96us, "
            "confidence intervals overlap, nothing moved.\n"
            "## Test plan\ny"
        )
        self.assertTrue(has_disposition(body))

    def test_bare_heading_without_content_fails(self):
        self.assertFalse(has_disposition("## bench-compare\n## Test plan\nx"))

    def test_subheadings_count_as_section_content(self):
        body = (
            "## bench-compare\n"
            "### Run 1\n"
            "| size | before | after |\n"
            "| 384 | 39us | 39us | these numbers fill the section well past eighty chars |\n"
            "### Run 2\n"
            "more\n"
            "## Next section\nx"
        )
        self.assertTrue(has_disposition(body))

    def test_mixed_case_heading_passes(self):
        body = (
            "## Bench-Compare Disposition\n"
            "Compiled out of the default bench build via a cfg gate, so base and head "
            "bench binaries have identical effective source.\n"
            "## End\nz"
        )
        self.assertTrue(has_disposition(body))

    def test_canonical_no_change_line_passes(self):
        # lattice/CLAUDE.md blesses this exact terse disposition for a PR that
        # measured no change; it is ~47 stripped chars and must satisfy the gate
        # via the no-change marker, not fail a raw length floor.
        body = (
            "## bench-compare disposition\n"
            "bench-compare showed no change (p > 0.05 on all groups).\n"
            "## Test plan\nx"
        )
        self.assertTrue(has_disposition(body))

    def test_one_line_na_passes(self):
        # The workflow comment promises a doc-only change satisfies the gate with
        # a one-line N/A; that promise must actually hold.
        body = (
            "## bench-compare\nN/A, doc-only change, no code paths touched.\n## End\ny"
        )
        self.assertTrue(has_disposition(body))

    def test_short_content_without_marker_fails(self):
        # Guards the length floor: terse content with no disposition marker is
        # still a bare-heading-in-spirit and must fail.
        self.assertFalse(has_disposition("## bench-compare\nfoo bar baz\n## End\nx"))

    def test_marker_prefix_of_longer_word_does_not_satisfy(self):
        # Marker anchoring: an unanchored "no change" marker matched the
        # prefix of "no changelog", letting a body with no disposition pass. The
        # marker must be word-boundary anchored, so this body (no disposition,
        # no N/A, under the length floor) must fail.
        body = (
            "## bench-compare disposition\n"
            "There is no changelog entry because this PR updates documentation.\n"
            "## Test plan\nx"
        )
        self.assertFalse(has_disposition(body))

    def test_fenced_heading_does_not_satisfy(self):
        # A bench-compare heading hidden inside a code fence renders as literal
        # code, not a disposition section. It must not open the section even with
        # enough content to clear the length floor; the section stays empty and
        # the gate fails, so a disposition no reviewer sees cannot pass.
        body = (
            "## Summary\nx\n"
            "```\n"
            "## bench-compare disposition\n"
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen\n"
            "```\n"
            "## Test plan\ny"
        )
        self.assertFalse(has_disposition(body))

    def test_html_comment_heading_does_not_satisfy(self):
        # Same bypass via a multi-line HTML comment: invisible when rendered, so
        # a heading spanning from <!-- to --> must not open the section.
        body = (
            "## Summary\nx\n"
            "<!--\n"
            "## bench-compare disposition\n"
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen\n"
            "-->\n"
            "## Test plan\ny"
        )
        self.assertFalse(has_disposition(body))

    def test_visible_heading_with_fenced_table_passes(self):
        # A real disposition whose numbers sit in a fenced table. The fence opens
        # AFTER the visible heading, so its content must still count as section
        # content. The prose outside the fence is deliberately under the length
        # floor and carries no marker, so this passes ONLY if the fenced rows are
        # counted. It fails if fenced content stops being counted.
        body = (
            "## bench-compare disposition\n"
            "Numbers:\n"
            "```\n"
            "group        before    after\n"
            "rms_norm     39us      39us\n"
            "gelu         41us      41us\n"
            "silu         44us      44us\n"
            "```\n"
            "Overlapping intervals throughout.\n"
            "## Test plan\nz"
        )
        self.assertTrue(has_disposition(body))

    def test_mismatched_fence_does_not_satisfy(self):
        # A four-backtick fence is not closed by a three-backtick line (a closer
        # must be at least as long as the opener), so the block stays open and the
        # bench-compare heading inside it renders as code, not a disposition.
        body = (
            "## Summary\n"
            "````text\n"
            "innocent code\n"
            "```\n"
            "## bench-compare disposition\n"
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen\n"
            "```\n"
            "## End\nx"
        )
        self.assertFalse(has_disposition(body))

    def test_info_string_closer_does_not_satisfy(self):
        # A fence line carrying an info string (```bench) is not a closing fence
        # (a closer may carry only trailing whitespace), so the block stays open
        # and the heading inside it stays hidden.
        body = (
            "## Summary\n"
            "```\n"
            "```bench\n"
            "## bench-compare disposition\n"
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen\n"
            "```\n"
            "## End\nx"
        )
        self.assertFalse(has_disposition(body))

    def test_comment_with_fence_then_visible_heading_passes(self):
        # A fence delimiter inside an HTML comment must not leak into fence state
        # and hide a later real heading. Comment state is resolved before fence
        # state, so the visible bench-compare heading after the comment opens the
        # section normally (this failed before the ordering fix).
        body = (
            "## Summary\n"
            "<!--\n"
            "```\n"
            "-->\n"
            "## bench-compare disposition\n"
            "A/B shows no change, p>0.05 on every group; nothing moved past eighty characters here yes.\n"
            "## Test plan\ny"
        )
        self.assertTrue(has_disposition(body))


    def test_raw_html_pre_block_does_not_satisfy(self):
        # A heading inside a <pre> raw HTML block renders as literal preformatted
        # text, not a disposition. The content easily clears the length floor, so
        # this proves the block is masked (heading not opened), not that content
        # is too short. Before raw-HTML tracking this exited 0.
        body = (
            "<pre>\n"
            "## bench-compare disposition\n"
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen\n"
            "</pre>\n"
            "## Test plan\ny"
        )
        self.assertFalse(has_disposition(body))

    def test_raw_html_script_block_does_not_satisfy(self):
        # Same class via <script>: raw contents, so a heading between the tags is
        # not a rendered heading and must not open the section.
        body = (
            "<script>\n"
            "## bench-compare disposition\n"
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen\n"
            "</script>\n"
            "## Test plan\ny"
        )
        self.assertFalse(has_disposition(body))

    def test_div_block_heading_does_not_satisfy(self):
        # A heading buried in a block-level <div> (no blank line before the real
        # content resumes) renders as raw HTML, not a heading. The conservative
        # fail-closed HTML-block catch masks it. Content clears the length floor,
        # so this proves masking, not a length failure. Exited 0 before the catch.
        body = (
            "## Summary\nx\n"
            "<div>\n"
            "## bench-compare disposition\n"
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen\n"
            "</div>\n"
            "## Test plan\ny"
        )
        self.assertFalse(has_disposition(body))

    def test_crlf_fenced_body_with_visible_heading_passes(self):
        # GitHub PR bodies commonly use CRLF line endings. A trailing carriage
        # return must be stripped before parsing, or the closing fence keeps a
        # \r, the fence never closes, and the later visible heading is wrongly
        # masked. Before CR normalization this exited 1 (a false reject).
        body = (
            "## Summary\r\n"
            "```text\r\n"
            "code\r\n"
            "```\r\n"
            "## bench-compare disposition\r\n"
            "N/A\r\n"
        )
        self.assertTrue(has_disposition(body))

    def test_comment_delimiter_inside_fence_then_visible_heading_passes(self):
        # A <!-- inside an open fence is literal code, not a comment start. It
        # must not transition comment state and strand the fence open, which would
        # falsely reject the later visible heading. Masked regions are mutually
        # exclusive: while fenced, only a matching closer is inspected. Before the
        # ordered-state fix this exited 1.
        body = (
            "## Summary\n"
            "```\n"
            "<!-- literal code\n"
            "```\n"
            "## bench-compare disposition\n"
            "N/A\n"
        )
        self.assertTrue(has_disposition(body))

    def test_indented_heading_passes(self):
        # CommonMark allows an ATX heading to be indented up to three spaces. Such
        # a heading is visible and must open the section. Before the indent fix
        # the matcher required the # at column zero and exited 1.
        body = (
            "## Summary\n"
            "   ## bench-compare disposition\n"
            "N/A\n"
        )
        self.assertTrue(has_disposition(body))

    def test_autolink_line_start_does_not_hide_disposition(self):
        # A content line beginning with an autolink (<https://...>) must not be
        # mistaken for an HTML block and mask a following heading. The HTML-block
        # catch requires a tag-name delimiter, and a colon follows the scheme, so
        # the autolink is not taken for a block. This pins that false-reject guard.
        body = (
            "<https://example.com/raw-bench-data> is the source below.\n"
            "## bench-compare disposition\n"
            "A/B shows no change, p>0.05 on every group; nothing moved past eighty characters here yes.\n"
            "## Test plan\ny"
        )
        self.assertTrue(has_disposition(body))


    def test_inline_html_span_does_not_false_reject(self):
        # An inline HTML element on a line with prose (a badge, link, or styled
        # note at the start of a line) is paragraph content, not an HTML block:
        # the tag is not alone on the line. The following ATX heading renders
        # normally and must open the section. A too-broad HTML-block predicate
        # masked this and wrongly rejected a visible disposition.
        body = (
            "<span>Measured on macOS.</span>\n"
            "## bench-compare disposition\n"
            "N/A\n"
        )
        self.assertTrue(has_disposition(body))

    def test_processing_instruction_heading_does_not_satisfy(self):
        # A heading inside a <? ... ?> processing instruction is a raw HTML block
        # (CommonMark type 3) and does not render as a heading. Content clears the
        # length floor, so this proves masking, not a length failure.
        body = (
            "<?\n"
            "## bench-compare disposition\n"
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen\n"
            "?>\n"
            "## Test plan\ny"
        )
        self.assertFalse(has_disposition(body))

    def test_cdata_heading_does_not_satisfy(self):
        # A heading inside a <![CDATA[ ... ]]> block is raw (CommonMark type 5)
        # and does not render as a heading.
        body = (
            "<![CDATA[\n"
            "## bench-compare disposition\n"
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen\n"
            "]]>\n"
            "## Test plan\ny"
        )
        self.assertFalse(has_disposition(body))

    def test_declaration_heading_does_not_satisfy(self):
        # A heading inside a <! ... > declaration is raw (CommonMark type 4) and
        # does not render as a heading.
        body = (
            "<!DOCTYPE bench\n"
            "## bench-compare disposition\n"
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen\n"
            ">\n"
            "## Test plan\ny"
        )
        self.assertFalse(has_disposition(body))


if __name__ == "__main__":
    unittest.main()

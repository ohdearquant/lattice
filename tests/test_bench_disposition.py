#!/usr/bin/env python3
"""Regression tests for scripts/bench_disposition_check.sh (the ADR-058 gate parser).

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


if __name__ == "__main__":
    unittest.main()

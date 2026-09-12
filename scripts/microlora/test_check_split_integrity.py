#!/usr/bin/env python3
"""Stdlib regression tests for split integrity diagnostics."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


CHECKER = Path(os.environ.get("SPLIT_CHECKER", Path(__file__).with_name("check_split_integrity.py")))
LEGACY_STDOUT = b"""train 1 rows (1 distinct completions, 1 appearing once)
valid 1 rows
  control (train prompts found in train): 1/1  [must be all]
  PROMPT overlap:     0/1
  COMPLETION overlap: 0/1  (0 distinct strings)
  prompt length median: train 5B, valid 5B (ratio 1.00)

PASS on the condition that voids a run: no held-out prompt appears in train.
"""


class SplitIntegrityTests(unittest.TestCase):
    def run_checker(self, train, valid, *flags):
        directory = Path(tempfile.mkdtemp(prefix="split-integrity-"))
        for name, completion in (("train", train), ("valid", valid)):
            (directory / f"{name}.jsonl").write_text(
                json.dumps({"prompt": name, "completion": completion}) + "\n"
            )
        return subprocess.run(
            ["uv", "run", "--no-project", "python3", str(CHECKER), "--dir", str(directory), *flags],
            capture_output=True,
        )

    def test_crossing_fails_and_names_verb(self):
        result = self.run_checker('memory.recall(query="a")', 'memory.recall(query="b")', "--verb-partition")
        self.assertEqual(result.returncode, 4, result.stderr.decode())
        self.assertIn(b"VERB crossing: 1/1 train, 1/1 valid: memory.recall", result.stdout)
        self.assertIn(b"control (train verbs found in train): 1/1  [must be all]", result.stdout)

    def test_bare_kg_verb_crosses(self):
        result = self.run_checker('search(query="a")', 'search(query="b")', "--verb-partition")
        self.assertEqual(result.returncode, 4, result.stderr.decode())
        self.assertIn(b"valid: search", result.stdout)

    def test_disjoint_and_literals_are_ignored(self):
        literal = json.dumps('escaped " quote, backslash \\, x.y( and search(')
        result = self.run_checker(f'memory.recall(query={literal}) | get(id="a")', f'comm.thread(id={literal})', "--verb-partition")
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertIn(b"VERB count: train 2 distinct/1 rows, valid 1 distinct/1 rows", result.stdout)
        self.assertIn(b"VERB crossing: 0/2 train, 0/1 valid: none", result.stdout)
        self.assertIn(b"control (train verbs found in train): 2/2  [must be all]", result.stdout)

    def test_flag_off_stdout_is_byte_identical(self):
        train, valid = 'memory.recall(query="a")', 'comm.thread(id="b")'
        off = self.run_checker(train, valid)
        on = self.run_checker(train, valid, "--verb-partition")
        self.assertEqual(off.returncode, 0)
        self.assertEqual(on.returncode, 0)
        self.assertEqual(off.stdout, LEGACY_STDOUT)
        self.assertEqual(
            off.stdout,
            b"".join(line for line in on.stdout.splitlines(keepends=True)
                     if not line.startswith((b"  VERB", b"  control (train verbs"))),
        )
        crossing_off = self.run_checker(train, 'memory.recall(query="b")')
        self.assertEqual(crossing_off.returncode, 0)
        self.assertEqual(crossing_off.stdout, LEGACY_STDOUT)

    def test_crossing_takes_priority_over_prompt_overlap(self):
        result = self.run_checker('search(query="a")', 'search(query="b")', "--verb-partition", "--valid", "train.jsonl")
        self.assertEqual(result.returncode, 4, result.stderr.decode())

    def test_no_calls(self):
        result = self.run_checker('"x.y("', '"search("', "--verb-partition")
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertIn(b"VERB crossing: 0/0 train, 0/0 valid: none", result.stdout)


if __name__ == "__main__":
    unittest.main()

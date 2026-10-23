#!/usr/bin/env python3
"""Regression tests for reconciling declared benchmark groups with Criterion's directories.

Criterion writes each group to a directory whose name it makes filename-safe:
it replaces `?"/\\*<>:|^` with `_` and cuts the name to 64 bytes. These cases
live apart from tests/test_bench_compare_measurement.py because that module is
above ADR-102's line cap and may only shrink. The runner and stubs come from
bench_compare_support.
"""
import os
import subprocess
import unittest


if __package__:
    from . import bench_compare_support as support
else:
    import bench_compare_support as support

_run = support.run_compare
GIT = support.GIT
STALE_CHANGE_CARGO = support.STALE_CHANGE_CARGO


class BenchCompareGroupDirectories(unittest.TestCase):
    @staticmethod
    def _declare_inference_groups(*names):
        def setup(root):
            path = root / "crates" / "inference" / "benches" / "elementwise_cpu_bench.rs"
            path.write_text(
                "".join(f'let mut group = c.benchmark_group("{name}");\n' for name in names),
                encoding="utf-8",
            )
            subprocess.run([*GIT, "-C", str(root), "add", "-f", str(path)], check=True)
            subprocess.run(
                [*GIT, "-C", str(root), "commit", "-qm", "declare fixture groups"],
                check=True,
                env={
                    **os.environ,
                    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                },
            )
        return setup

    def test_group_name_with_a_slash_reconciles_against_its_criterion_directory(self):
        """Criterion writes `benchmark_group("backward_stage0/gdn_backward")` to the
        directory `backward_stage0_gdn_backward`. The declared name must be compared
        in that form.

        Mutation-sensitive: compare the raw declared names (drop the
        criterion_group_dir mapping in bench-compare-impl.sh) and the run refuses
        with exit 2, naming backward_stage0_gdn_backward as undeclared.
        """
        result = _run(
            [],
            stub_cargo=STALE_CHANGE_CARGO,
            setup=self._declare_inference_groups("backward_stage0/gdn_backward"),
            extra_env={"STUB_INFERENCE_GROUP_DIR": "backward_stage0_gdn_backward"},
        )
        self.assertEqual(
            result.returncode, 0,
            f"expected exit 0, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertNotIn("does not declare", result.stderr)

    def test_group_name_longer_than_criterion_directory_limit_reconciles(self):
        """Criterion cuts a directory name to 64 bytes; the declared name is cut the
        same way."""
        declared = "long_group/" + "x" * 60
        directory = ("long_group_" + "x" * 60)[:64]
        result = _run(
            [],
            stub_cargo=STALE_CHANGE_CARGO,
            setup=self._declare_inference_groups(declared),
            extra_env={"STUB_INFERENCE_GROUP_DIR": directory},
        )
        self.assertEqual(
            result.returncode, 0,
            f"expected exit 0, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_undeclared_group_directory_still_refuses(self):
        """The mapping must not widen the declared set: a directory the source does
        not declare in any form still refuses the run."""
        result = _run(
            [],
            stub_cargo=STALE_CHANGE_CARGO,
            setup=self._declare_inference_groups("backward_stage0"),
            extra_env={"STUB_INFERENCE_GROUP_DIR": "backward_stage0_gdn_backward"},
        )
        self.assertEqual(
            result.returncode, 2,
            f"expected exit 2, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("reports group(s) this target does not declare", result.stderr)
        self.assertIn("    - backward_stage0_gdn_backward", result.stderr)

    def test_long_non_ascii_group_name_refuses_the_run(self):
        """A declared name over 64 bytes with a non-ASCII byte has no derivable
        directory, so the run refuses even when every reported directory belongs to
        another declared group.

        Mutation-sensitive: let the caller swallow the mapping's failure (the earlier
        `|| true` form) and the declared set reads as empty, so the run refuses for
        that reason instead and the last assertion fails.
        """
        result = _run(
            [],
            stub_cargo=STALE_CHANGE_CARGO,
            setup=self._declare_inference_groups("rms_norm", "\u00e9" * 40),
            extra_env={"STUB_INFERENCE_GROUP_DIR": "rms_norm"},
        )
        self.assertEqual(
            result.returncode, 2,
            f"expected exit 2, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("its Criterion directory cannot be derived", result.stderr)
        self.assertNotIn("found zero declared Criterion groups", result.stderr)


if __name__ == "__main__":
    support.FailOnEmptyTestProgram()

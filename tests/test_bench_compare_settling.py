#!/usr/bin/env python3
"""Post-build settling order, deadline and quiet-admission contracts."""

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

if __package__:
    from . import bench_compare_support as support
    from . import test_bench_compare_prebuild as prebuild
else:
    import bench_compare_support as support
    import test_bench_compare_prebuild as prebuild


ARMS = ("base1", "head1", "head2", "base2")


def recording_cargo(source):
    prefix = r'''if [[ " $* " == *" --no-run "* ]]; then
  package="inference"
  if [[ " $* " == *" -p lattice-embed "* ]]; then
    package="embed"
  fi
  trap 'rc=$?; sampler="not-running"; for marker in "$STUB_REPO_ROOT/.cache/bench-compare-criterion/phase-load/"*.ready; do if [[ -e "$marker" ]]; then sampler="running"; fi; done; printf "build\t%s\t%s\t%s\n" "$package" "$sampler" "$rc" >> "$STUB_SETTLE_EVENTS"; exit "$rc"' EXIT
fi
'''
    return source.replace("#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n" + prefix, 1)


SAMPLER_MARKER = 'pathlib.Path(out + ".ready").touch()'
assert support.STUB_PHASE_SAMPLER.count(SAMPLER_MARKER) == 1
RECORDING_SAMPLER = support.STUB_PHASE_SAMPLER.replace(
    "import json\n", "import json\nimport os\n", 1,
).replace(
    SAMPLER_MARKER,
    'with open(os.environ["STUB_SETTLE_EVENTS"], "a", encoding="utf-8") as output:\n'
    '    interval = sys.argv[sys.argv.index("--interval") + 1]\n'
    '    output.write(f"sample\\t{arm}\\t{interval}\\n")\n' + SAMPLER_MARKER,
)

RECORDING_QUIET = r'''#!/usr/bin/env python3
import json
import os
import sys

label = sys.argv[sys.argv.index("--label") + 1]
with open(os.environ["STUB_SETTLE_EVENTS"], "a", encoding="utf-8") as output:
    output.write("quiet\t" + label + "\n")
failed = label == "after " + os.environ.get("STUB_SETTLE_LOUD_ARM", "") + " settling"
idle = 10.0 if failed else 100.0
if "--jsonl-out" in sys.argv:
    phase = sys.argv[sys.argv.index("--phase") + 1]
    path = sys.argv[sys.argv.index("--jsonl-out") + 1]
    with open(path, "a", encoding="utf-8") as output:
        output.write(json.dumps({"schema": "perf-ambient-sample/v1",
                                 "phase": phase, "idle_pct": idle}) + "\n")
verdict = "BELOW FLOOR" if failed else "ok"
print(f"[quiet] {label}: idle {idle:.1f}% (floor 70.0%) {verdict} | top: fixture 0.0%")
sys.exit(1 if failed else 0)
'''


class SettlingOrdering(unittest.TestCase):
    def run_ordered(self, *, handoff=False, extra_env=None, failing_build=False,
                    args=(), post_run=None, stub_quiet_probe=RECORDING_QUIET):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        events = Path(temporary.name) / "events.tsv"
        environment = {"STUB_SETTLE_EVENTS": str(events), **(extra_env or {})}
        if handoff:
            result, _ = prebuild.PrebuildOrdering._run_handoff(
                self, temporary.name, fail_first_build=failing_build,
                stub_cargo=recording_cargo(prebuild.HANDOFF_CARGO),
                stub_phase_sampler=RECORDING_SAMPLER,
                stub_quiet_probe=stub_quiet_probe, extra_env=environment,
                post_run=post_run,
            )
        else:
            cargo = (prebuild.FAILING_EMBED_BUILD_CARGO if failing_build
                     else support.STALE_CHANGE_CARGO)
            result = support.run_compare(
                list(args), stub_cargo=recording_cargo(cargo),
                stub_phase_sampler=RECORDING_SAMPLER,
                stub_quiet_probe=stub_quiet_probe, extra_env=environment,
                post_run=post_run,
            )
        rows = [line.split("\t") for line in events.read_text().splitlines()] if events.exists() else []
        return result, rows

    def assert_arm_order(self, result, rows):
        samples = [(i, row[1]) for i, row in enumerate(rows) if row[0] == "sample"]
        self.assertEqual([arm for _, arm in samples], list(ARMS), rows)
        if not any(row[0] == "settle-start" for row in rows):
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        previous = -1
        for end, arm in samples:
            window = rows[previous + 1:end]
            start = [i for i, row in enumerate(window) if row == ["settle-start", arm]]
            settled = [i for i, row in enumerate(window) if row[:2] == ["settle-end", arm]]
            quiet = [i for i, row in enumerate(window)
                     if row == ["quiet", f"after {arm} settling"]]
            builds = [i for i, row in enumerate(window)
                      if row[0] == "build" and row[2:] == ["not-running", "0"]]
            self.assertEqual(len(start), 1, f"{arm}: missing unique settling start; {window}")
            self.assertEqual(len(settled), 1, f"{arm}: missing unique settling completion; {window}")
            self.assertTrue(builds, f"{arm}: no successful prebuild; {window}")
            self.assertLess(max(builds), start[0], f"{arm}: settling preceded final prebuild; {window}")
            self.assertLess(start[0], settled[0], f"{arm}: settling did not complete; {window}")
            self.assertEqual(len(quiet), 1, f"{arm}: missing unique post-settle quiet gate; {window}")
            self.assertLess(settled[0], quiet[0], f"{arm}: quiet gate preceded settling; {window}")
            previous = end
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_ordinary_arms_settle_after_final_build_before_quiet_and_sampler(self):
        result, rows = self.run_ordered()
        self.assert_arm_order(result, rows)

    def test_all_handoff_arms_settle_after_final_build_before_quiet_and_sampler(self):
        result, rows = self.run_ordered(handoff=True)
        self.assert_arm_order(result, rows)

    def test_final_build_failure_never_settles_admits_or_samples(self):
        for handoff in (False, True):
            with self.subTest(handoff=handoff):
                result, rows = self.run_ordered(handoff=handoff, failing_build=True)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertTrue(any(row[0] == "build" and row[-1] != "0" for row in rows), rows)
                self.assertFalse(any(row[0] in ("settle-start", "settle-end", "sample") for row in rows), rows)
                self.assertEqual([row[1] for row in rows if row[0] == "quiet"], ["before first arm"])

    def test_invalid_override_or_fractional_default_refuses_before_build(self):
        for environment in ({"BENCH_SETTLE_SECONDS": value} for value in ("", "0", "-1", "0.5", "nan", " 5")):
            with self.subTest(environment=environment):
                result, rows = self.run_ordered(extra_env=environment)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("settling seconds", result.stderr)
                self.assertEqual(rows, [])
        result, rows = self.run_ordered(extra_env={"PHASE_SAMPLE_INTERVAL": "0.25"})
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(rows, [])

    def test_integer_override_preserves_fractional_sampler_cadence(self):
        result, rows = self.run_ordered(extra_env={
            "BENCH_SETTLE_SECONDS": "2", "PHASE_SAMPLE_INTERVAL": "0.25",
        })
        self.assert_arm_order(result, rows)
        self.assertEqual([row[2] for row in rows if row[0] == "sample"], ["0.25"] * 4)
        self.assertEqual([float(row[2]) for row in rows if row[0] == "settle-end"], [2.0] * 4)

    def test_quiet_refusal_prevents_that_arm_sampler(self):
        result, rows = self.run_ordered(extra_env={"STUB_SETTLE_LOUD_ARM": "base2"})
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("machine was not quiet at 'after base2 settling'", result.stderr)
        self.assertEqual([row[1] for row in rows if row[0] == "sample"], list(ARMS[:3]))
        self.assertIn(["quiet", "after base2 settling"], rows)

    def test_status_mode_retains_all_checkpoints_and_defers_low_idle_refusal(self):
        captured = []

        def capture(root):
            captured.append((root / ".cache" / "bench-run-provenance.txt").read_text())

        with tempfile.TemporaryDirectory() as temporary:
            status_dir = Path(temporary) / "status"
            result, rows = self.run_ordered(extra_env={
                "PERF_POSTMERGE_STATUS_DIR": str(status_dir),
                "STUB_SETTLE_LOUD_ARM": "head1", "BENCH_IDLE_FLOOR": "70",
            }, args=("--fail-on-regression",), post_run=capture)
            statuses = [json.loads(path.read_text()) for path in sorted(status_dir.glob("*.json"))]
        self.assertEqual([row[1] for row in rows if row[0] == "sample"], list(ARMS))
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertEqual(len(statuses), 2)
        for status in statuses:
            self.assertEqual(status["ambient"]["assessment"], "not_measurable")
            self.assertEqual(status["ambient"]["samples"]["settled-head1"], 10.0)
            self.assertEqual(len(status["ambient"]["samples"]), 7)
        self.assertEqual(len(captured), 1)
        lines = captured[0].splitlines()
        self.assertIn("schema=lattice-bench-provenance-v3", lines)
        self.assertIn("settle_seconds=5", lines)
        self.assertEqual(sum(line.startswith("ambient=") for line in lines), 7)
        states = [json.loads(line.removeprefix("machine_state=")) for line in lines if line.startswith("machine_state=")]
        self.assertEqual([state["label"] for state in states], [
            "before first arm", "after base1 settling", "after head1 settling",
            "between order strata", "after head2 settling", "after base2 settling", "after final arm",
        ])

    def test_in_arm_journal_write_failure_refuses_certification(self):
        quiet = RECORDING_QUIET.replace("import json\n", "import json\nfrom pathlib import Path\n", 1).replace(
            'verdict = "BELOW FLOOR" if failed else "ok"',
            'if label == "after base1 settling":\n'
            '    paths = list((Path(os.environ["STUB_REPO_ROOT"]) / ".cache").glob("bench-checkpoints.*"))\n'
            '    assert len(paths) == 1, paths\n'
            '    paths[0].unlink()\n'
            '    paths[0].mkdir()\n'
            'verdict = "BELOW FLOOR" if failed else "ok"',
        )
        result, rows = self.run_ordered(stub_quiet_probe=quiet)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("cannot retain quiet checkpoint 'after base1 settling'", result.stderr)
        self.assertIn(["quiet", "after base1 settling"], rows)
        self.assertNotIn("=== Run conditions ===", result.stdout)


class SettlingDeadline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = support.LIB / "bench_settle.py"
        spec = importlib.util.spec_from_file_location("settling_deadline", path)
        cls.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.helper)

    def test_interrupted_wait_reaches_original_deadline(self):
        now = [100.0]
        requests = []

        def sleep(seconds):
            requests.append(seconds)
            self.assertLessEqual(len(requests), 3, "wait extended the original deadline")
            if len(requests) == 1:
                now[0] += 2.0
                raise InterruptedError("fixture interruption")
            now[0] += 1.0 if len(requests) == 2 else seconds

        with mock.patch.object(self.helper.time, "monotonic", side_effect=lambda: now[0]), \
                mock.patch.object(self.helper.time, "sleep", side_effect=sleep):
            self.helper.wait_seconds(5)
        self.assertEqual(now[0], 105.0, "settling returned before its original deadline")
        self.assertEqual(requests, [5.0, 3.0, 2.0])

    def test_effective_duration_uses_interval_or_explicit_positive_integer(self):
        self.assertEqual(self.helper.resolve_seconds("5", {}), 5)
        self.assertEqual(self.helper.resolve_seconds("8", {}), 8)
        self.assertEqual(self.helper.resolve_seconds("0.25", {"BENCH_SETTLE_SECONDS": "2"}), 2)
        for value in ("", "0", "-1", "1.5", "nan", "inf", " 5", "5 ", "1e2"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.helper.resolve_seconds("5", {"BENCH_SETTLE_SECONDS": value})


if __name__ == "__main__":
    support.FailOnEmptyTestProgram(verbosity=2)

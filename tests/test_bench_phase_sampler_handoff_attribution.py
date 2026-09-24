#!/usr/bin/env python3
"""Regression tests for lattice#1732: under --gpu-handoff, the in-phase load
sampler's self-tree never reaches the admitted benchmark executable.

scripts/lib/bench_handoff.py's HandoffService.run() launches bench-compare-
impl.sh (its "outer" command) as one child of the cooperative supervisor
process (the bench_supervision.py "run" invocation that owns the HandoffService
instance). Its background broker thread separately launches the admitted
benchmark executable as ANOTHER child of that same supervisor once
bench_admission.py connects to it (_launch() in bench_handoff.py). The two are
siblings, never ancestor and descendant, so a self-tree rooted at
bench-compare-impl.sh's own "$$" -- which is what scripts/lib/bench-compare-
impl.sh's phase_sampler_start always passed -- can never walk down to the
admitted executable: its CPU always counted as "foreign". This was most
visible once the executable's build artifact was already cached (head2/base2
reuse head1/base1's target_dir) and its own measured run then dominated the
sampled window, but the misattribution was present in every arm.

These tests drive the REAL, unmodified phase_sampler_start function, extracted
verbatim out of the shipping scripts/lib/bench-compare-impl.sh (never a
hand-copied mirror that could silently drift from it -- see CLAUDE.md's
"DERIVED CARRIERS DON'T SHARE THE WORDS"), against a REAL process tree shaped
exactly like the bug: a "supervisor" process (this test) with two kinds of
children -- a harness script standing in for bench-compare-impl.sh, and
several real, CPU-burning processes standing in for the admitted executable,
spawned directly by the test the same way HandoffService._launch() spawns the
real one: as a sibling of bench-compare-impl.sh under the shared parent, never
as its descendant.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "scripts" / "lib"
IMPL = LIB / "bench-compare-impl.sh"

# phase-load-sampler.py's idle_percent() calls `top -l 2 -n 0 -s 1` on macOS,
# so a single sample already costs roughly a second regardless of --interval.
# These durations are sized generously so the busy load is still running for
# the sampler's first sample, at least one in-loop sample, and its closing
# sample (taken after the harness's own EXIT trap SIGTERMs it).
BUSY_SECONDS = 6.0
HARNESS_SLEEP_SECONDS = 3.0


def _extract_phase_sampler_source() -> str:
    """The REAL PHASE_LOAD_ROOT / phase_load_file / phase_sampler_start block,
    verbatim, out of the shipping bench-compare-impl.sh -- not a copy that
    could drift from the fix these tests exist to guard."""
    text = IMPL.read_text()
    root_line = re.search(r"^PHASE_LOAD_ROOT=.*$", text, re.M)
    file_fn = re.search(r"^phase_load_file\(\) \{\n.*?\n\}\n", text, re.M | re.S)
    start_fn = re.search(r"^phase_sampler_start\(\) \{\n.*?\n\}\n", text, re.M | re.S)
    assert root_line and file_fn and start_fn, (
        "bench-compare-impl.sh's phase-sampler block moved or was renamed; "
        "update this extraction"
    )
    return "\n".join([root_line.group(0), file_fn.group(0), start_fn.group(0)])


def _busy_process() -> subprocess.Popen:
    """A real, CPU-pinning child process standing in for the admitted
    benchmark executable. Spawned directly by the TEST process, exactly the
    way HandoffService._launch() spawns the real admitted executable as a
    child of the supervisor rather than of bench-compare-impl.sh."""
    return subprocess.Popen([
        sys.executable, "-c",
        "import time\n"
        f"end = time.monotonic() + {BUSY_SECONDS}\n"
        "while time.monotonic() < end:\n"
        "    pass\n",
    ])


def _drive(handoff_broker: str) -> list[dict]:
    """Run the REAL phase_sampler_start under a process tree shaped like
    --gpu-handoff: this test process is the "supervisor", with the harness
    below (standing in for bench-compare-impl.sh) and several busy processes
    (standing in for the admitted executable) as its two kinds of children --
    siblings of each other, exactly as HandoffService.run()/_launch() launch
    bench-compare-impl.sh and the admitted executable.
    """
    n_busy = max(2, min(8, (os.cpu_count() or 4) // 2))
    busy = [_busy_process() for _ in range(n_busy)]
    try:
        with tempfile.TemporaryDirectory(prefix="phase-sampler-1732-") as tmp:
            root = Path(tmp)
            lib = root / "scripts" / "lib"
            lib.mkdir(parents=True)
            # The REAL, unmodified sampler: this test is about who
            # bench-compare-impl.sh tells it to root at, not about its own
            # ps-tree-walking logic (already covered elsewhere).
            shutil.copy2(LIB / "phase-load-sampler.py", lib / "phase-load-sampler.py")
            harness = root / "harness.sh"
            harness.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"REPO={shlex.quote(str(root))}\n"
                f"PYTHON_BIN={shlex.quote(sys.executable)}\n"
                "PHASE_SAMPLE_INTERVAL=0.2\n"
                f"handoff_broker={shlex.quote(handoff_broker)}\n"
                + _extract_phase_sampler_source()
                + '\nphase_sampler_start "fixture-arm"\n'
                f"sleep {HARNESS_SLEEP_SECONDS}\n"
            )
            harness.chmod(0o755)
            result = subprocess.run(
                ["bash", str(harness)], capture_output=True, text=True, timeout=60,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            out = (
                root / ".cache" / "bench-compare-criterion" / "phase-load"
                / "fixture-arm.jsonl"
            )
            samples = [
                json.loads(line) for line in out.read_text().splitlines() if line.strip()
            ]
            assert samples, "no phase-load samples were written"
            return samples
    finally:
        for proc in busy:
            proc.terminate()
        for proc in busy:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


class PhaseSamplerHandoffSelfAttribution(unittest.TestCase):
    """lattice#1732."""

    @classmethod
    def setUpClass(cls):
        cls.handoff_samples = _drive("unix:/tmp/lattice-1732-fixture-broker")
        cls.non_handoff_samples = _drive("")

    def test_handoff_shaped_tree_lands_admitted_executable_cpu_in_self(self):
        """A handoff-shaped process tree (the busy stand-ins parented outside
        bench-compare-impl.sh, as siblings under the shared parent -- exactly
        as HandoffService.run()/_launch() parent the real admitted executable
        and bench-compare-impl.sh) must attribute the stand-ins' CPU to
        "self", not "foreign", once handoff_broker is set.

        Edit that must turn this red: in scripts/lib/bench-compare-impl.sh's
        phase_sampler_start, delete the
        `if [ -n "$handoff_broker" ]; then self_pid="$PPID"; fi` branch (i.e.
        revert to the unconditional `--self-pid "$$"` this function used
        before lattice#1732). With that reverted, this run's self-pid stays
        rooted at the harness alone; the busy stand-ins are its siblings
        under the shared parent rather than its descendants, and their CPU
        reads back as "foreign" instead of "self".
        """
        self_max = max(s["self_pct"] for s in self.handoff_samples)
        foreign_max = max(s["foreign_pct"] for s in self.handoff_samples)
        self.assertGreater(
            self_max, foreign_max,
            f"handoff samples did not attribute the busy load to self: {self.handoff_samples}",
        )
        # The SAME busy load reads mostly foreign in the non-handoff control
        # below; the swing between the two runs is the signal, not a fixed
        # cross-machine percentage threshold, since it cancels out whatever
        # ambient load happens to sit on the box during either run.
        non_handoff_foreign_max = max(s["foreign_pct"] for s in self.non_handoff_samples)
        self.assertLess(
            foreign_max, non_handoff_foreign_max,
            (self.handoff_samples, self.non_handoff_samples),
        )

    def test_non_handoff_control_keeps_current_attribution(self):
        """Without --gpu-handoff, the admitted-executable stand-ins' CPU must
        keep reading as "foreign", exactly as it always has.

        Edit that must turn this red: any change to phase_sampler_start that
        makes the NON-handoff path pick something other than the
        unconditional "$$" it always used -- e.g. rooting the self-tree at
        "$PPID" unconditionally instead of only when handoff_broker is set.
        That would be a broader fix than lattice#1732 calls for; this test is
        what the packet's "non-handoff attribution is unchanged" constraint
        means concretely.
        """
        self_max = max(s["self_pct"] for s in self.non_handoff_samples)
        foreign_max = max(s["foreign_pct"] for s in self.non_handoff_samples)
        self.assertGreater(
            foreign_max, self_max,
            f"non-handoff samples did not keep the busy load foreign: {self.non_handoff_samples}",
        )


if __name__ == "__main__":
    unittest.main()

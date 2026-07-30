#!/usr/bin/env python3
"""Regression tests for scripts/bench-compare.sh's measurement-integrity guard.

The guard exists because cargo's exit status is necessary and not sufficient. A
bench invocation whose Criterion filter matches nothing exits 0 having measured
nothing, and the target then contributes no Criterion comparison at all — so a
downstream gate that reconciles comparisons FOUND against comparisons JUDGED
cannot see the omission: absence leaves no artifact to be found missing. The
only place the run's intent is still known is the invocation itself.

These tests drive the real script, not an extracted copy of the helper. The
script derives its repo root from its own location, so each case builds a
disposable git repo, copies the shipping script and its lib/ into it, and puts a
stub `cargo` on PATH that exits 0 and prints no measurement lines — exactly the
shape that used to pass.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "bench-compare.sh"
GATE = REPO / "scripts" / "perf-bench-gate.py"
LIB = REPO / "scripts" / "lib"

# Exits 0 for every subcommand and prints nothing a measurement filter matches.
STUB_CARGO = """#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  printf '%s\n' 'cargo 1.94.1 (fixture)'
fi
if [ "${STUB_EMIT_CRITERION_HOME:-0}" = "1" ]; then
  case " $* " in
    *" --save-baseline "*|*" --baseline "*)
      echo "time: criterion-home=${CRITERION_HOME:-<unset>}"
      ;;
  esac
fi
exit 0
"""

STUB_GOVERNOR = """#!/usr/bin/env python3
import json
import sys
from datetime import UTC, datetime

label = sys.argv[sys.argv.index("--label") + 1]
print(json.dumps({
    "schema": "lattice-machine-state-v1",
    "label": label,
    "captured_at_utc": datetime.now(UTC).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z"),
    "power": {"status": "measured", "source": "fixture", "state": "ac"},
    "thermal": {
        "status": "measured",
        "source": "fixture",
        "state": "nominal",
    },
    "idle": {
        "status": "measured",
        "source": "fixture",
        "seconds": 30.0,
    },
    "gate": {
        "status": "passed",
        "cooldown_seconds": 30.0,
        "afk_threshold_seconds": 30.0,
        "kill_switch": "clear",
    },
}, separators=(",", ":"), sort_keys=True))
"""

# Test helpers invoking real Git must disable repository hooks.
GIT = ("git", "-c", "core.hooksPath=/dev/null")

STALE_CHANGE_CARGO = r"""#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--version" ]]; then
  printf '%s\n' 'cargo 1.94.1 (fixture)'
  exit 0
fi

write_baseline() {
  local bench="$1"
  mkdir -p "$CRITERION_HOME/$bench/compare-base"
  printf '%s\n' '{"mean":{"point_estimate":90.0}}' \
    > "$CRITERION_HOME/$bench/compare-base/estimates.json"
  printf '%s\n' \
    '{"sampling_mode":"Linear","iters":[1.0,2.0],"times":[1.0,2.0]}' \
    > "$CRITERION_HOME/$bench/compare-base/sample.json"
}

write_head() {
  local bench="$1"
  mkdir -p "$CRITERION_HOME/$bench/new"
  mkdir -p "$CRITERION_HOME/$bench/change"
  printf '%s\n' '{"mean":{"point_estimate":100.0}}' \
    > "$CRITERION_HOME/$bench/new/estimates.json"
  printf '%s\n' \
    '{"sampling_mode":"Flat","iters":[1.0,2.0],"times":[1.0,2.0]}' \
    > "$CRITERION_HOME/$bench/new/sample.json"
  printf '%s\n' \
    '{"mean":{"point_estimate":0.01,"confidence_interval":{"lower_bound":0.0,"upper_bound":0.02}}}' \
    > "$CRITERION_HOME/$bench/change/estimates.json"
}

args=" $* "
if [[ "$args" == *" --no-run "* ]]; then
  exit 0
fi
if [[ "$args" == *" --list "* ]]; then
  if [[ "$args" == *" lattice-inference "* ]]; then
    printf '%s\n' 'rms_norm/896: benchmark'
  else
    printf '%s\n' 'simd_dot_product/scalar/384: benchmark'
  fi
  exit 0
fi

if [[ "$args" == *" --save-baseline "* ]]; then
  if [[ "$args" == *" lattice-inference "* ]]; then
    write_baseline "rms_norm/896"
    write_baseline "rms_norm/4096"
  else
    write_baseline "simd_dot_product/scalar/384"
  fi
else
  if [[ "$args" == *" lattice-inference "* ]]; then
    write_head "rms_norm/896"
    if [[ "${STUB_REMOVE_RMS_4096:-0}" != "1" ]]; then
      write_head "rms_norm/4096"
    fi
  else
    write_head "simd_dot_product/scalar/384"
  fi
fi

printf '%s\n' 'time: [1.000 ns 1.010 ns 1.020 ns]'
printf '%s\n' 'change: [+0.0% +1.0% +2.0%] (p = 0.50 > 0.05)'
"""

PARTIAL_COPY_RSYNC = r"""#!/usr/bin/env bash
set -euo pipefail

src="$2"
dst="$3"
for bench in rms_norm/896 simd_dot_product/scalar/384; do
  if [[ -d "$src/$bench/compare-base" ]]; then
    mkdir -p "$dst/$bench"
    cp -R "$src/$bench/compare-base" "$dst/$bench/"
  fi
done
exit 23
"""


def _run(
    extra_args,
    *,
    stub_cargo=STUB_CARGO,
    stub_rsync=None,
    setup=None,
    extra_env=None,
    emit_criterion_home=False,
):
    """Run the shipping bench-compare.sh in a throwaway repo with a stub cargo."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        (root / "scripts").mkdir(parents=True)
        shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
        shutil.copy2(GATE, root / "scripts" / GATE.name)
        shutil.copytree(LIB, root / "scripts" / "lib")
        shutil.copy2(REPO / ".gitignore", root / ".gitignore")
        governor = root / "scripts" / "perf_governor.py"
        governor.write_text(STUB_GOVERNOR)
        governor.chmod(0o755)
        quiet_probe = root / "scripts" / "lib" / "quiet-probe.py"
        quiet_probe.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "label = sys.argv[sys.argv.index('--label') + 1]\n"
            "print(f'[quiet] {label}: idle 100.0% (floor 0.0%) ok | top: fixture 0.0%')\n"
        )
        machine_probe = root / "scripts" / "lib" / "machine-state-probe.py"
        machine_probe.write_text(
            "#!/usr/bin/env python3\n"
            "import datetime, json, sys\n"
            "label = sys.argv[sys.argv.index('--label') + 1]\n"
            "print(json.dumps({'schema':'lattice-machine-state-v1','label':label,"
            "'captured_at_utc':datetime.datetime.now(datetime.UTC)"
            ".strftime('%Y-%m-%dT%H:%M:%SZ'),"
            "'power':{'status':'unavailable','reason':'fixture'},"
            "'thermal':{'status':'unavailable','reason':'fixture'},"
            "'idle':{'status':'unavailable','reason':'fixture'}},"
            "separators=(',', ':'), sort_keys=True))\n"
        )

        env_git = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                   "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run([*GIT, "init", "-q", "-b", "main", str(root)], check=True)
        (root / "Cargo.lock").write_text(
            'version = 4\n\n'
            '[[package]]\n'
            'name = "criterion"\n'
            'version = "0.5.1"\n'
        )
        subprocess.run([*GIT, "-C", str(root), "add", "-f", "Cargo.lock"], check=True)
        for i in range(2):
            (root / f"f{i}.txt").write_text(str(i))
            subprocess.run([*GIT, "-C", str(root), "add", "-A"], check=True)
            subprocess.run([*GIT, "-C", str(root), "commit", "-qm", f"c{i}"],
                           check=True, env=env_git)

        if setup is not None:
            setup(root)

        # Redirect the machine-wide lock and pending-marker paths inside the
        # COPIED supervisor. These tests measure nothing, so serializing them
        # against real benches on this machine buys no isolation and costs a
        # wait that can exceed the timeout below. Rewriting path constants in
        # the copy is deliberately weaker than reimplementing the locking:
        # every line of acquisition, refusal and reporting logic is still the
        # shipping one. There is no equivalent knob in the shipping script,
        # which is the point -- a real run cannot redirect its own locks.
        locks = root / "scripts" / "lib" / "bench-locks.py"
        src = locks.read_text()
        for const in ("BENCH_WINDOW", "GPU_LOCK", "PENDING_DIR"):
            before = src
            src = re.sub(
                rf'^{const} = "[^"]*"$',
                f'{const} = "{tmp}/{const.lower()}"',
                src,
                flags=re.M,
            )
            assert src != before, f"{const} constant not found to redirect"
        locks.write_text(src)
        subprocess.run(
            [*GIT, "-C", str(root), "add", "scripts/lib/bench-locks.py"],
            check=True,
        )
        subprocess.run(
            [*GIT, "-C", str(root), "commit", "-qm", "fixture lock paths"],
            check=True,
            env=env_git,
        )

        bindir = Path(tmp) / "bin"
        bindir.mkdir()
        cargo = bindir / "cargo"
        cargo.write_text(stub_cargo)
        cargo.chmod(0o755)
        if stub_rsync is not None:
            rsync = bindir / "rsync"
            rsync.write_text(stub_rsync)
            rsync.chmod(0o755)

        # The ambient-load gate judges whether the MACHINE was quiet enough for
        # a number to be trusted. This run produces no number, so the only
        # thing the gate could do here is fail the test on unrelated load.
        # Zero is honest for a run whose output is never quoted as a
        # measurement; it is not a default anything else should use.
        env = {
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "BENCH_IDLE_FLOOR": "0",
            "LATTICE_BENCH_HOST_ID_FILE": f"{tmp}/bench-host-id",
            **(extra_env or {}),
            "STUB_EMIT_CRITERION_HOME": "1" if emit_criterion_home else "0",
        }
        return subprocess.run(
            ["bash", str(root / "scripts" / SCRIPT.name), *extra_args, "HEAD~1", "HEAD"],
            capture_output=True, text=True, env=env, timeout=300)


class BenchCompareMeasurementGuard(unittest.TestCase):
    def test_enforcing_mode_refuses_a_run_that_measured_nothing(self):
        """A bench that exits 0 having printed no measurement must not certify.

        Mutation-sensitive: drop the line-count argument from the call sites, or
        the zero-line branch from require_measured, and this run exits 0 instead
        of 2 -- which is precisely the partial A/B the flag exists to refuse.
        """
        result = _run(["--fail-on-regression"])
        self.assertEqual(
            result.returncode, 2,
            f"expected exit 2 (measurement broken), got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("produced no measurements", result.stderr)

    def test_reporter_mode_is_unchanged_by_the_guard(self):
        """Without the flag the script stays tolerant: the guard must not bite.

        The default caller is a human reading an A/B against an arbitrary ref,
        where a missing bench target is ordinary. Pinning this stops a later
        tightening from silently becoming the default.
        """
        result = _run([])
        self.assertNotEqual(
            result.returncode, 2,
            f"reporter mode must not exit 2\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}")
        self.assertNotIn("produced no measurements", result.stderr)

    def test_stale_change_cannot_mask_a_benchmark_removed_on_head(self):
        """A stale same-path comparison must not satisfy head completeness.

        The disposable HEAD starts with a valid old rms_norm/4096 new/change
        tree. The base arm measures 896 and 4096, while the head stub measures
        only 896 (plus the independent embed target), all with successful cargo
        exits and measurement lines. Mutation-sensitive: remove the
        --prepare-head call from bench-compare-impl.sh and the stale 4096 change
        survives, the gate sees every base ID in the change set, and this
        enforcing run exits 0 instead of 2.
        """
        def seed_stale_change(root):
            bench = (
                root / ".cache" / "bench-compare-criterion" / "head" /
                "inference" / "criterion" / "rms_norm" / "4096"
            )
            (bench / "new").mkdir(parents=True)
            (bench / "change").mkdir()
            (bench / "new" / "estimates.json").write_text(
                '{"mean":{"point_estimate":100.0}}\n'
            )
            (bench / "change" / "estimates.json").write_text(
                '{"mean":{"point_estimate":0.01,'
                '"confidence_interval":{"lower_bound":0.0,"upper_bound":0.02}}}\n'
            )

        result = _run(
            ["--fail-on-regression"],
            stub_cargo=STALE_CHANGE_CARGO,
            setup=seed_stale_change,
            extra_env={"STUB_REMOVE_RMS_4096": "1"},
        )
        self.assertEqual(
            result.returncode, 2,
            f"expected exit 2 (missing head benchmark), got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("selected baseline 'compare-base'", result.stdout)
        self.assertIn(
            "  - lattice-inference:elementwise_cpu_bench: rms_norm/4096",
            result.stdout,
        )
        self.assertIn("removed 2 stale head artifact directories", result.stdout)

    def test_stale_unrelated_selected_baseline_is_pruned_before_copy(self):
        """A prior alternate-target baseline must not join today's base set.

        Mutation-sensitive: remove the --prepare-baseline-copy call and the
        stale compare-base artifact remains selected. The later head cleanup
        removes its old comparison, so completeness false-fails old_group/42
        even though every benchmark in today's fresh base set ran on HEAD.
        """
        def seed_unrelated_run(root):
            bench = (
                root / ".cache" / "bench-compare-criterion" / "head" /
                "inference" / "criterion" / "old_group" / "42"
            )
            (bench / "compare-base").mkdir(parents=True)
            (bench / "new").mkdir()
            (bench / "change").mkdir()
            (bench / "compare-base" / "estimates.json").write_text(
                '{"mean":{"point_estimate":90.0}}\n'
            )
            (bench / "new" / "estimates.json").write_text(
                '{"mean":{"point_estimate":100.0}}\n'
            )
            (bench / "change" / "estimates.json").write_text(
                '{"mean":{"point_estimate":0.01,'
                '"confidence_interval":{"lower_bound":0.0,"upper_bound":0.02}}}\n'
            )

        result = _run(
            ["--fail-on-regression"],
            stub_cargo=STALE_CHANGE_CARGO,
            setup=seed_unrelated_run,
        )
        self.assertEqual(
            result.returncode, 0,
            f"fresh complete A/B was contaminated by stale unrelated data\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn(
            "removed 1 stale selected-baseline artifact directory before fresh base copy",
            result.stdout,
        )
        self.assertNotIn("old_group/42", result.stdout)

    def test_enforcing_mode_refuses_a_partial_baseline_copy(self):
        """A failed partial copy must not shrink the selected set and certify.

        The rsync stub copies two of the three base measurements, then returns
        rsync's partial-transfer status. Mutation-sensitive: mask that status
        with `|| true` and the head measures all three benches, while the gate
        sees only the two copied baseline IDs and returns 0.
        """
        result = _run(
            ["--fail-on-regression"],
            stub_cargo=STALE_CHANGE_CARGO,
            stub_rsync=PARTIAL_COPY_RSYNC,
        )
        self.assertEqual(
            result.returncode, 2,
            f"expected exit 2 (partial baseline copy), got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn(
            "lattice-inference:elementwise_cpu_bench selected baseline copy "
            "failed (rsync exit 23)",
            result.stderr,
        )

    def test_each_bench_target_gets_a_distinct_criterion_root(self):
        """Target identity must be structural, not reconstructed from group names.

        Mutation-sensitive: point EMBED_CRITERION_ROOT at the inference root (or
        drop either CRITERION_HOME assignment) and the observed path set has one
        member or includes `<unset>`, reproducing the shared namespace behind
        #1090. The stub emits only its inherited CRITERION_HOME; no benchmark
        implementation is duplicated here.
        """
        result = _run([], emit_criterion_home=True)
        self.assertEqual(
            result.returncode, 0,
            f"reporter-mode probe failed\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}")
        roots = set(re.findall(r"criterion-home=(\S+)", result.stdout))
        self.assertEqual(
            len(roots), 4,
            f"expected isolated base/head roots per target, saw {roots}\n"
            f"stdout:\n{result.stdout}")
        self.assertNotIn("<unset>", roots)
        self.assertTrue(any("/inference/criterion" in path for path in roots), roots)
        self.assertTrue(any("/embed/criterion" in path for path in roots), roots)


if __name__ == "__main__":
    unittest.main()

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
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "bench-compare.sh"
GATE = REPO / "scripts" / "perf-bench-gate.py"
LIB = REPO / "scripts" / "lib"

_GATE_SPEC = importlib.util.spec_from_file_location("perf_bench_gate", GATE)
assert _GATE_SPEC is not None and _GATE_SPEC.loader is not None
gate = importlib.util.module_from_spec(_GATE_SPEC)
sys.modules[_GATE_SPEC.name] = gate
_GATE_SPEC.loader.exec_module(gate)

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

FAILING_CARGO = """#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  printf '%s\n' 'cargo 1.94.1 (fixture)'
  exit 0
fi
if [[ " $* " == *" --no-run "* ]]; then
  exit 0
fi
printf '%s\n' 'fixture cargo failed before producing a measurement' >&2
exit 7
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

FAILING_STATE_PROBE = """#!/usr/bin/env python3
print("malformed machine-state fixture")
raise SystemExit(127)
"""

STUB_MACHINE_PROBE = """#!/usr/bin/env python3
import datetime
import json
import sys

label = sys.argv[sys.argv.index("--label") + 1]
print(json.dumps({
    "schema": "lattice-machine-state-v1",
    "label": label,
    "captured_at_utc": datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    ),
    "power": {"status": "unavailable", "reason": "fixture"},
    "thermal": {"status": "unavailable", "reason": "fixture"},
    "idle": {"status": "unavailable", "reason": "fixture"},
}, separators=(",", ":"), sort_keys=True))
"""

def _stub_machine_probe_with_first_failure(failure_statement):
    if not failure_statement:
        raise ValueError("failure statement must be non-empty")
    marker = "print(json.dumps({"
    if marker not in STUB_MACHINE_PROBE:
        raise ValueError("machine-state fixture print marker is missing")
    return STUB_MACHINE_PROBE.replace(
        marker,
        "if label == 'before first arm':\n"
        f"    {failure_statement}\n"
        f"{marker}",
        1,
    )


FAILED_MACHINE_STATE_PROBES = {
    "nonzero": _stub_machine_probe_with_first_failure("raise SystemExit(19)"),
    "empty": _stub_machine_probe_with_first_failure("raise SystemExit(0)"),
}

PYTHON_ENTRYPOINTS_USING_DATETIME_UTC = (
    REPO / "scripts" / "lib" / "machine-state-probe.py",
    REPO / "scripts" / "perf-bench-gate.py",
    REPO / "scripts" / "bench_decode_harness.py",
    REPO / "scripts" / "bench_cpu_flagship_supervisor.py",
)

# Test helpers invoking real Git must disable repository hooks.
GIT = ("git", "-c", "core.hooksPath=/dev/null")

STALE_CHANGE_CARGO = r"""#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--version" ]]; then
  printf '%s\n' 'cargo 1.94.1 (fixture)'
  exit 0
fi

if [[ "${1:-}" == "bench" && "${STUB_REQUIRE_LOCKED:-0}" == "1" ]]; then
  case " $* " in
    *" --locked "*) ;;
    *) exit 86 ;;
  esac
fi

write_baseline() {
  local bench="$1"
  local baseline_name="$2"
  mkdir -p "$CRITERION_HOME/$bench/$baseline_name"
  printf '%s\n' '{"mean":{"point_estimate":90.0}}' \
    > "$CRITERION_HOME/$bench/$baseline_name/estimates.json"
  printf '%s\n' \
    '{"sampling_mode":"Linear","iters":[1.0,2.0],"times":[1.0,2.0]}' \
    > "$CRITERION_HOME/$bench/$baseline_name/sample.json"
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
  baseline_name="${args#* --save-baseline }"
  baseline_name="${baseline_name%% *}"
  if [[ "$args" == *" lattice-inference "* ]]; then
    write_baseline "rms_norm/896" "$baseline_name"
    write_baseline "rms_norm/4096" "$baseline_name"
  else
    write_baseline "simd_dot_product/scalar/384" "$baseline_name"
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

if [[ "${STUB_EMIT_CRITERION_HOME:-0}" == "1" ]]; then
  echo "time: criterion-home=${CRITERION_HOME:-<unset>}"
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
printf '%s\n' 'fixture rsync: partial baseline transfer' >&2
exit 23
"""

ORDER_BALANCE_CARGO = r"""#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--version" ]]; then
  printf '%s\n' 'cargo 1.94.1 (fixture)'
  exit 0
fi

args=" $* "
if [[ "$args" == *" --no-run "* ]]; then
  exit 0
fi

if [[ "$args" == *" lattice-inference "* ]]; then
  bench="rms_norm/512"
  target="inference"
else
  bench="simd_dot_product/scalar/384"
  target="embed"
fi

if [[ "$PWD" == *"/.cache/bench-compare-base" ]]; then
  arm="A"
else
  arm="B"
fi
printf '%s\n' "$target:$arm" >> "$STUB_ORDER_FILE"

write_estimate() {
  local artifact="$1"
  local ns="$2"
  mkdir -p "$CRITERION_HOME/$bench/$artifact"
  printf '{"mean":{"point_estimate":%s}}\n' "$ns" \
    > "$CRITERION_HOME/$bench/$artifact/estimates.json"
  printf '%s\n' \
    '{"sampling_mode":"Flat","iters":[1.0,2.0],"times":[1.0,2.0]}' \
    > "$CRITERION_HOME/$bench/$artifact/sample.json"
}

write_change() {
  local point="$1"
  local low="$2"
  local high="$3"
  mkdir -p "$CRITERION_HOME/$bench/change"
  printf '{"mean":{"point_estimate":%s,"confidence_interval":{"lower_bound":%s,"upper_bound":%s}}}\n' \
    "$point" "$low" "$high" \
    > "$CRITERION_HOME/$bench/change/estimates.json"
}

scenario="${STUB_SCENARIO:-directional-drift}"
if [[ "$target" == "inference" && -n "${STUB_INFERENCE_SCENARIO:-}" ]]; then
  scenario="$STUB_INFERENCE_SCENARIO"
elif [[ "$target" == "embed" && -n "${STUB_EMBED_SCENARIO:-}" ]]; then
  scenario="$STUB_EMBED_SCENARIO"
fi
if [[ "$scenario" == "directional-drift" ]]; then
  a1="100.0"
  b1="110.0"
  b2="121.0"
  a2="133.1"
  forward_point="0.10"
  forward_low="0.095"
  forward_high="0.105"
  reverse_point="0.10"
  reverse_low="0.095"
  reverse_high="0.105"
elif [[ "$scenario" == "true-regression" ]]; then
  a1="100.0"
  b1="122.4"
  b2="124.848"
  a2="106.1208"
  forward_point="0.224"
  forward_low="0.222"
  forward_high="0.226"
  reverse_point="-0.15"
  reverse_low="-0.152"
  reverse_high="-0.148"
else
  printf 'unknown STUB_SCENARIO=%s\n' "$scenario" >&2
  exit 9
fi

if [[ "$args" == *" --save-baseline "* ]]; then
  baseline_name="${args#* --save-baseline }"
  baseline_name="${baseline_name%% *}"
  if [[ "$baseline_name" == "compare-base" ]]; then
    write_estimate "$baseline_name" "$a1"
  elif [[ "$baseline_name" == "compare-head" ]]; then
    write_estimate "$baseline_name" "$b2"
  else
    printf 'unexpected save baseline %s\n' "$baseline_name" >&2
    exit 9
  fi
else
  baseline_name="${args#* --baseline }"
  baseline_name="${baseline_name%% *}"
  if [[ "$baseline_name" == "compare-base" && "$arm" == "B" ]]; then
    write_estimate "new" "$b1"
    write_change "$forward_point" "$forward_low" "$forward_high"
  elif [[ "$baseline_name" == "compare-head" && "$arm" == "A" ]]; then
    write_estimate "new" "$a2"
    write_change "$reverse_point" "$reverse_low" "$reverse_high"
  else
    printf 'unexpected comparison baseline=%s arm=%s\n' "$baseline_name" "$arm" >&2
    exit 9
  fi
fi

printf '%s\n' 'time: [1.000 ns 1.010 ns 1.020 ns]'
if [[ "$args" == *" --baseline "* ]]; then
  printf '%s\n' 'change: [+0.0% +1.0% +2.0%] (p = 0.50 > 0.05)'
fi
"""


def _run(
    extra_args,
    *,
    stub_cargo=STUB_CARGO,
    stub_rsync=None,
    setup=None,
    extra_env=None,
    emit_criterion_home=False,
    stub_machine_state=None,
    post_run=None,
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
        governor.write_text(
            STUB_GOVERNOR if stub_machine_state is None else stub_machine_state
        )
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
            STUB_MACHINE_PROBE
            if stub_machine_state is None
            else stub_machine_state
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
        # The default bench targets' declared-group derivation reads
        # crates/<crate>/benches/<target>.rs from the checked-out worktree, so
        # the fixture needs stub sources declaring the same groups the stub
        # cargo fixtures above write results for (rms_norm, simd_dot_product).
        # Every test defaults to BENCHES_INFERENCE=elementwise_cpu_bench and
        # BENCHES_EMBED=simd (none override them), so one fixed pair covers
        # the whole file.
        inference_benches = root / "crates" / "inference" / "benches"
        inference_benches.mkdir(parents=True)
        (inference_benches / "elementwise_cpu_bench.rs").write_text(
            'let mut group = c.benchmark_group("rms_norm");\n'
        )
        embed_benches = root / "crates" / "embed" / "benches"
        embed_benches.mkdir(parents=True)
        (embed_benches / "simd.rs").write_text(
            'let mut group = c.benchmark_group("simd_dot_product");\n'
        )
        subprocess.run([*GIT, "-C", str(root), "add", "-f", "Cargo.lock", "crates"], check=True)
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
        result = subprocess.run(
            ["bash", str(root / "scripts" / SCRIPT.name), *extra_args, "HEAD~1", "HEAD"],
            capture_output=True, text=True, env=env, timeout=300)
        if post_run is not None:
            post_run(root)
        return result


class ClearSelectedBaselineArtifactsSiblingPrune(unittest.TestCase):
    """Isolated repro for the round-4 fix: clear_selected_baseline_artifacts

    must remove a pruned baseline dir's new/change siblings, and must not
    touch a differently-named baseline tree sharing the same criterion root.
    """

    def _make_root(self, tmp):
        root = Path(tmp) / "criterion"
        pruned = root / "old_group" / "42"
        (pruned / "compare-base").mkdir(parents=True)
        (pruned / "new").mkdir()
        (pruned / "change").mkdir()
        (pruned / "compare-base" / "estimates.json").write_text(
            '{"mean":{"point_estimate":90.0}}\n'
        )
        (pruned / "new" / "estimates.json").write_text(
            '{"mean":{"point_estimate":100.0}}\n'
        )
        (pruned / "change" / "estimates.json").write_text(
            '{"mean":{"point_estimate":0.01,'
            '"confidence_interval":{"lower_bound":0.0,"upper_bound":0.02}}}\n'
        )

        unrelated = root / "other_group" / "7" / "manual-snapshot"
        unrelated.mkdir(parents=True)
        (unrelated / "estimates.json").write_text('{"mean":{"point_estimate":50.0}}\n')
        return root, pruned, unrelated

    def test_removes_pruned_siblings_and_spares_unrelated_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, pruned, unrelated = self._make_root(tmp)

            removed = gate.clear_selected_baseline_artifacts(root, "compare-base")

            self.assertEqual(removed, 1)
            self.assertFalse((pruned / "compare-base").exists())
            self.assertFalse((pruned / "new").exists())
            self.assertFalse((pruned / "change").exists())
            self.assertTrue((unrelated / "estimates.json").exists())


class BenchCompareMeasurementGuard(unittest.TestCase):
    def test_reporter_mode_refuses_failed_machine_state_checkpoints(self):
        """A missing state record must void a report-only A/B."""
        self.assertGreater(len(FAILED_MACHINE_STATE_PROBES), 0)
        control = _run([], stub_cargo=STALE_CHANGE_CARGO)
        self.assertEqual(
            control.returncode,
            0,
            f"valid report-only fixture did not produce a usable A/B\n"
            f"stdout:\n{control.stdout}\nstderr:\n{control.stderr}",
        )
        for name, probe in FAILED_MACHINE_STATE_PROBES.items():
            with self.subTest(probe=name):
                self.assertTrue(name)
                self.assertTrue(probe.strip())
                result = _run(
                    [],
                    stub_cargo=STALE_CHANGE_CARGO,
                    stub_machine_state=probe,
                )
                self.assertEqual(
                    result.returncode,
                    2,
                    f"report-only run accepted {name} state checkpoint\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                )

    def test_reporter_mode_refuses_a_partial_baseline_copy(self):
        """A partial baseline copy must void a report-only A/B."""
        self.assertTrue(STALE_CHANGE_CARGO.strip())
        self.assertTrue(PARTIAL_COPY_RSYNC.strip())
        control = _run([], stub_cargo=STALE_CHANGE_CARGO)
        self.assertEqual(
            control.returncode,
            0,
            f"valid report-only fixture did not produce a usable A/B\n"
            f"stdout:\n{control.stdout}\nstderr:\n{control.stderr}",
        )
        result = _run(
            [],
            stub_cargo=STALE_CHANGE_CARGO,
            stub_rsync=PARTIAL_COPY_RSYNC,
        )
        self.assertEqual(
            result.returncode,
            2,
            "report-only run accepted a partial baseline copy and could "
            "return uncertified A/B output",
        )

    def test_datetime_utc_entrypoints_reject_python_3_9_explicitly(self):
        """The declared Python minimum must fail before datetime.UTC imports."""
        bootstrap = (
            "import runpy,sys;"
            "target=sys.argv[1];"
            "sys.argv=[target];"
            "sys.version_info=(3,9,6);"
            "runpy.run_path(target,run_name='__main__')"
        )
        self.assertTrue(bootstrap)
        self.assertGreater(len(PYTHON_ENTRYPOINTS_USING_DATETIME_UTC), 0)
        for entrypoint in PYTHON_ENTRYPOINTS_USING_DATETIME_UTC:
            with self.subTest(entrypoint=entrypoint.name):
                self.assertTrue(str(entrypoint))
                result = subprocess.run(
                    [sys.executable, "-c", bootstrap, str(entrypoint)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(
                    result.returncode,
                    1,
                    f"unsupported interpreter was not rejected by {entrypoint}\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                )
                self.assertIn("requires Python 3.11 or newer", result.stderr)
                self.assertIn("running Python 3.9.6", result.stderr)
                self.assertIn(sys.executable, result.stderr)

    def test_machine_state_probe_handles_missing_datetime_utc(self):
        """A Python 3.9-shaped datetime module must yield the minimum diagnostic."""
        bootstrap = (
            "import runpy,sys;"
            "target=sys.argv[1];"
            "sys.argv=[target,'--label','compatibility-control'];"
            "sys.version_info=(3,9,6);"
            "runpy.run_path(target,run_name='__main__')"
        )
        datetime_stub = "class datetime:\n    pass\n"
        self.assertTrue(bootstrap)
        self.assertTrue(datetime_stub)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "datetime.py").write_text(datetime_stub)
            python_path = os.pathsep.join(
                part
                for part in (tmp, os.environ.get("PYTHONPATH"))
                if part
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    bootstrap,
                    str(REPO / "scripts" / "lib" / "machine-state-probe.py"),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "PYTHONPATH": python_path},
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("requires Python 3.11 or newer", result.stderr)
        self.assertIn("running Python 3.9.6", result.stderr)
        self.assertIn(sys.executable, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_every_bench_command_requires_the_committed_lockfile(self):
        """Every A/B build and measurement must refuse dependency re-resolution."""
        source = (LIB / "bench-compare-impl.sh").read_text()
        commands = [
            line for line in source.splitlines()
            if re.search(r"\bcargo bench\b", line)
            and not line.lstrip().startswith("#")
        ]
        self.assertTrue(commands, "found 0 cargo bench invocations")
        self.assertEqual(
            len(commands), 10,
            f"found {len(commands)} cargo bench invocations:\n"
            + "\n".join(commands),
        )
        self.assertTrue(
            all(re.search(r"\bcargo bench --locked\b", line) for line in commands),
            f"found {len(commands)} cargo bench invocations; "
            "every command must pass --locked:\n" + "\n".join(commands),
        )

        result = _run(
            ["--fail-on-regression"],
            stub_cargo=STALE_CHANGE_CARGO,
            extra_env={"STUB_REQUIRE_LOCKED": "1"},
        )
        self.assertEqual(
            result.returncode, 0,
            f"locked benchmark harness failed\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )

    def test_noindex_marker_failure_is_not_a_confirmed_regression(self):
        """A pre-measurement integrity failure must not read as a regression.

        scripts/lib/ensure-noindex-marker.sh runs under `set -e` in
        bench-compare-impl.sh before either worktree or benchmark exists
        (bench-compare-impl.sh:396-397). If that guard ever exits 1 again, the
        raw status propagates unchanged through bench-locks.py's
        subprocess.call and this script's exec, and
        perf-postmerge-gate.yml:280-282 would report it as a confirmed
        regression with revert advice -- although no benchmark ever ran.

        Mutation-sensitive: revert the guard's normalization (exit 2 -> exit
        1) in scripts/lib/ensure-noindex-marker.sh and this run's exit code
        flips from 2 to 1, exactly the collision this test exists to catch.
        """
        def occupy_marker(root):
            # Mirrors ensure-noindex-marker-selftest.sh case 7: a directory
            # sitting at the marker path cannot become the marker file and
            # cannot be silently removed, so the guard must refuse.
            occupied = root / ".cache" / ".metadata_never_index" / "occupied"
            occupied.mkdir(parents=True)

        result = _run(["--fail-on-regression"], setup=occupy_marker)
        self.assertEqual(
            result.returncode, 2,
            "a pre-measurement instrumentation failure must exit 2 (input/"
            "instrumentation error), never 1 (confirmed regression); got "
            f"{result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}")
        self.assertIn("[noindex] FATAL", result.stderr)
        self.assertNotIn("gate reported a confirmed regression", result.stderr)

    def test_cache_mkdir_failure_is_not_a_confirmed_regression(self):
        """The entry point's own `mkdir -p "$REPO/.cache"` must not leak raw exit 1.

        scripts/bench-compare.sh runs this mkdir under `set -e` before it ever
        execs bench-locks.py -- before any lock is taken, any worktree exists,
        or any benchmark runs. If it ever regresses to a bare `mkdir -p`, a
        regular file occupying `.cache` makes mkdir fail with an unnormalized
        exit 1, and perf-postmerge-gate.yml would report it as a confirmed
        regression with revert advice for a run that never measured anything.

        Mutation-sensitive: revert the guard's normalization (exit 2 -> the
        bare `mkdir -p "$REPO/.cache"`) in scripts/bench-compare.sh and this
        run's exit code flips from 2 to 1, exactly the collision this test
        exists to catch.
        """
        def occupy_cache(root):
            (root / ".cache").write_text("occupied")

        result = _run(["--fail-on-regression"], setup=occupy_cache)
        self.assertEqual(
            result.returncode, 2,
            "a pre-measurement instrumentation failure must exit 2 (input/"
            "instrumentation error), never 1 (confirmed regression); got "
            f"{result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}")
        self.assertIn("FATAL", result.stderr)
        self.assertNotIn("gate reported a confirmed regression", result.stderr)

    def test_cache_mkdir_failure_with_closed_stderr_still_exits_2(self):
        """A fatal diagnostic write must not itself preempt the exit status.

        Every FATAL echo in scripts/bench-compare.sh writes to fd 2. Under
        `set -e`, a write that fails (fd 2 closed by the caller) is itself a
        failing command, and an unguarded `echo ... >&2` would abort the
        script right there with the shell's own exit 1 -- the status this
        contract reserves for a confirmed regression -- before the script
        ever reaches its explicit `exit 2`.

        Mutation-sensitive: drop the `|| :` from any FATAL echo in the
        mkdir-failure branch and this closed-stderr run flips from 2 to 1.
        """
        def occupy_cache(root):
            (root / ".cache").write_text("occupied")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            (root / "scripts").mkdir(parents=True)
            shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
            shutil.copytree(LIB, root / "scripts" / "lib")
            occupy_cache(root)
            result = subprocess.run(
                ["bash", "-c",
                 f'exec "{root / "scripts" / SCRIPT.name}" HEAD~1 HEAD 2>&-'],
                capture_output=True, text=True, timeout=30)
        self.assertEqual(
            result.returncode, 2,
            "a fatal diagnostic write failing under closed stderr must not "
            f"leak the shell's raw exit 1; got {result.returncode}\n"
            f"stdout:\n{result.stdout}")

    def test_repo_root_resolution_failure_is_not_a_confirmed_regression(self):
        """An unguarded `REPO="$(cd ... && pwd)"` must not leak raw exit 1.

        scripts/bench-compare.sh:27 resolves its own repository root via a
        command substitution before anything else runs. If that `cd` ever
        fails -- e.g. the checkout's parent directory disappeared between
        bash opening the script and this line executing -- the unguarded
        form aborts under `set -e` with the shell's own exit 1, the status
        this contract reserves for a confirmed regression.

        The failure is reproduced deterministically (not via a real,
        inherently racy delete-mid-exec) by handing bash the script's body
        on the command line with $0 set to a path whose parent never
        existed, so the `cd` fails for the same reason a raced deletion
        would: the resolved directory is not there.

        Mutation-sensitive: revert the guard around the REPO= assignment in
        scripts/bench-compare.sh and this run's exit code flips from 2 to a
        raw 1 (or an unhandled `set -e` abort), never a controlled refusal.
        """
        script_body = SCRIPT.read_text()
        fake_path = "/tmp/lattice-repo-root-resolution-never-existed/scripts/bench-compare.sh"
        result = subprocess.run(
            ["bash", "-c", script_body, fake_path],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(
            result.returncode, 2,
            "repository-root resolution failure must exit 2 (input/"
            "instrumentation error), never a raw 1 (confirmed regression); "
            f"got {result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}")
        self.assertIn("FATAL", result.stderr)

    def test_inner_root_resolution_failure_is_not_a_confirmed_regression(self):
        """The measurement body's own root resolution must not leak raw exit 1.

        scripts/lib/bench-compare-impl.sh resolves its own repository root via
        an unguarded `REPO="$(cd "$(dirname "$0")/../.." && pwd)"` before doing
        anything else. This is the merged entry route's own inner boundary,
        distinct from scripts/bench-compare.sh's outer resolution already
        covered above -- the outer script's own guard does not protect this
        body when it (or a caller bypassing the entry point) invokes it with a
        $0 whose parent has disappeared.

        Reproduced deterministically the same way as the outer case: bash gets
        the body's own source on the command line with $0 set to a path whose
        parent never existed.

        Mutation-sensitive: revert the guard around the REPO= assignment in
        bench-compare-impl.sh and this run's exit code flips from 2 to a raw 1
        (or an unhandled `set -e` abort), never a controlled refusal.
        """
        impl_body = (LIB / "bench-compare-impl.sh").read_text()
        fake_path = (
            "/tmp/lattice-inner-root-resolution-never-existed/"
            "scripts/lib/bench-compare-impl.sh"
        )
        result = subprocess.run(
            ["bash", "-c", impl_body, fake_path],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(
            result.returncode, 2,
            "inner repository-root resolution failure must exit 2 (input/"
            "instrumentation error), never a raw 1 (confirmed regression); "
            f"got {result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}")
        self.assertIn("FATAL", result.stderr)

    def test_perf_postmerge_status_dir_regular_file_refuses_with_exit_2(self):
        """The postmerge status-directory setup must not leak raw exit 1.

        bench-compare-impl.sh creates $PERF_POSTMERGE_STATUS_DIR and truncates
        an ambient-samples file inside it before any worktree or benchmark
        exists. A regular file occupying that path makes `mkdir -p` fail with
        an unnormalized exit 1 under `set -e`.

        Mutation-sensitive: revert the `if ! mkdir -p ...; then ... fi` guard
        around $PERF_POSTMERGE_STATUS_DIR and this run's exit code flips from
        2 to 1.
        """
        with tempfile.TemporaryDirectory() as status_tmp:
            status_dir = Path(status_tmp) / "postmerge-status"
            status_dir.write_text("occupied")
            result = _run(
                ["--fail-on-regression"],
                extra_env={"PERF_POSTMERGE_STATUS_DIR": str(status_dir)},
            )
        self.assertEqual(
            result.returncode, 2,
            "a pre-measurement instrumentation failure must exit 2 (input/"
            "instrumentation error), never 1 (confirmed regression); got "
            f"{result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}")
        self.assertIn("FATAL", result.stderr)
        self.assertNotIn("gate reported a confirmed regression", result.stderr)

    def test_perf_postmerge_status_dir_nonwritable_refuses_with_exit_2(self):
        """A non-writable (but existing) status directory must also exit 2.

        `mkdir -p` on an existing directory succeeds regardless of write
        permission, so the ambient-samples file truncation is the operation
        that actually fails here -- a second, independent failure point in
        the same setup block.

        Mutation-sensitive: revert the guard around the
        `: > "$AMBIENT_SAMPLES_FILE"` truncation and this run's exit code
        flips from 2 to 1.
        """
        with tempfile.TemporaryDirectory() as status_tmp:
            status_dir = Path(status_tmp) / "postmerge-status"
            status_dir.mkdir()
            status_dir.chmod(0o555)
            try:
                result = _run(
                    ["--fail-on-regression"],
                    extra_env={"PERF_POSTMERGE_STATUS_DIR": str(status_dir)},
                )
            finally:
                status_dir.chmod(0o755)
        self.assertEqual(
            result.returncode, 2,
            "a pre-measurement instrumentation failure must exit 2 (input/"
            "instrumentation error), never 1 (confirmed regression); got "
            f"{result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}")
        self.assertIn("FATAL", result.stderr)
        self.assertNotIn("gate reported a confirmed regression", result.stderr)

    def test_perf_postmerge_status_filename_strips_colon_and_slash(self):
        """The gate-status filename derived from a bench target must contain
        neither ':' nor '/': actions/upload-artifact rejects both, and
        uploading "lattice-embed:simd.json" fails the run with "Contains the
        following character: Colon :".

        A bracket expression bash never closes (`[:\\/]` -- `[:` opens a
        POSIX character-class token with no matching `:]`) matches nothing,
        so the substitution silently no-ops and the colon survives all the
        way to the upload step. This asserts the OUTPUT the fix must
        produce, not that the substitution merely ran: a no-op substitution
        exits 0 and prints a string too, so only the produced characters
        can tell the two apart.

        Mutation-sensitive: revert the substitution to `[:\\/]` and this
        fails because ':' (and, on the slash fixture, '/') survive in the
        printed filename.
        """
        impl_source = (REPO / "scripts" / "lib" / "bench-compare-impl.sh").read_text()
        sanitizer_line = next(
            line.strip() for line in impl_source.splitlines()
            if line.strip().startswith("local status_name=")
        )
        for target in ("lattice-embed:simd", "lattice-inference/elementwise"):
            script = (
                'set -euo pipefail\n'
                f'target="{target}"\n'
                f'f() {{ {sanitizer_line}; printf "%s" "$status_name"; }}\n'
                'f\n'
            )
            result = subprocess.run(
                ["bash", "-c", script], capture_output=True, text=True, timeout=10)
            self.assertEqual(
                result.returncode, 0,
                f"sanitizer line failed for target {target!r}: {result.stderr}")
            produced = result.stdout
            self.assertNotIn(
                ":", produced,
                f"status filename retains a colon for target {target!r}: "
                f"{produced!r} -- actions/upload-artifact rejects this path")
            self.assertNotIn(
                "/", produced,
                f"status filename retains a slash for target {target!r}: "
                f"{produced!r} -- this would be read as a subdirectory")

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

    def test_default_mode_refuses_a_run_that_measured_nothing(self):
        """Report-only controls regression enforcement, not measurement validity."""
        result = _run([])
        self.assertEqual(
            result.returncode, 2,
            f"expected exit 2 (not measurable), got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("produced no measurements", result.stderr)

    def test_default_mode_surfaces_a_failed_benchmark_command(self):
        result = _run([], stub_cargo=FAILING_CARGO)
        self.assertEqual(
            result.returncode, 2,
            f"expected exit 2 (not measurable), got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("failed (exit 7)", result.stderr)
        self.assertIn(
            "fixture cargo failed before producing a measurement", result.stderr
        )

    def test_default_mode_refuses_a_failed_machine_state_probe_before_base(self):
        result = _run([], stub_machine_state=FAILING_STATE_PROBE)
        self.assertEqual(
            result.returncode, 2,
            f"expected exit 2 (not measurable), got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("machine-state checkpoint 'before first arm' failed", result.stderr)
        self.assertNotIn("--- Building + benching BASE", result.stdout)

    def test_default_mode_completes_a_healthy_measurement_fixture(self):
        result = _run([], stub_cargo=STALE_CHANGE_CARGO)
        self.assertEqual(
            result.returncode, 0,
            f"healthy report-only fixture failed\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )
        self.assertIn("Done.", result.stdout)

    def test_report_only_mode_uses_balanced_order(self):
        """Report-only evidence uses the same balanced ABBA measurement."""
        with tempfile.TemporaryDirectory() as temporary:
            order_file = Path(temporary) / "order.txt"
            result = _run(
                [],
                stub_cargo=ORDER_BALANCE_CARGO,
                extra_env={
                    "STUB_ORDER_FILE": str(order_file),
                    "STUB_SCENARIO": "true-regression",
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            observations = order_file.read_text().splitlines()
            for target in ("inference", "embed"):
                self.assertEqual(
                    [
                        observation.split(":", 1)[1]
                        for observation in observations
                        if observation.startswith(f"{target}:")
                    ],
                    ["A", "B", "B", "A"],
                    observations,
                )
            self.assertIn(
                "arm order: ABBA (base₁ → head₁ → head₂ → base₂)",
                result.stdout,
            )
            self.assertIn("ABBA bound", result.stdout)

    def test_enforcing_abba_refuses_identical_source_directional_drift(self):
        """A gate-sized second-arm drift is NOT_MEASURABLE, never regression.

        Mutation-sensitive in two independent ways: remove the reverse-order
        arms and the old forward +10% interval exits 1; combine the two ratios
        without retaining the order-effect envelope and the run exits 0 instead
        of failing closed with 3.
        """
        with tempfile.TemporaryDirectory() as temporary:
            order_file = Path(temporary) / "order.txt"
            result = _run(
                ["--fail-on-regression"],
                stub_cargo=ORDER_BALANCE_CARGO,
                extra_env={
                    "STUB_ORDER_FILE": str(order_file),
                    "STUB_SCENARIO": "directional-drift",
                },
            )
            self.assertEqual(
                result.returncode,
                3,
                f"expected NOT_MEASURABLE (3), got {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertIn("order-bias bound above", result.stdout)
            self.assertIn("**⏸ NOT MEASURABLE**", result.stdout)
            self.assertNotIn("✅ All 1 gated benches", result.stdout)
            observations = order_file.read_text().splitlines()
            for target in ("inference", "embed"):
                self.assertEqual(
                    [
                        observation.split(":", 1)[1]
                        for observation in observations
                        if observation.startswith(f"{target}:")
                    ],
                    ["A", "B", "B", "A"],
                    observations,
                )
            self.assertIn(
                "arm order: ABBA (base₁ → head₁ → head₂ → base₂)",
                result.stdout,
            )

    def test_enforcing_abba_retains_distinguishable_regression(self):
        """A true 20% source regression under 2% drift still exits 1."""
        with tempfile.TemporaryDirectory() as temporary:
            order_file = Path(temporary) / "order.txt"
            result = _run(
                ["--fail-on-regression"],
                stub_cargo=ORDER_BALANCE_CARGO,
                extra_env={
                    "STUB_ORDER_FILE": str(order_file),
                    "STUB_SCENARIO": "true-regression",
                },
            )
            self.assertEqual(
                result.returncode,
                1,
                f"expected regression (1), got {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertIn("gate reported a confirmed regression", result.stderr)

    def test_confirmed_regression_outranks_unmeasurable_target(self):
        """One target's exit 3 must not suppress another target's exit 1."""
        with tempfile.TemporaryDirectory() as temporary:
            order_file = Path(temporary) / "order.txt"
            result = _run(
                ["--full", "--fail-on-regression"],
                stub_cargo=ORDER_BALANCE_CARGO,
                extra_env={
                    "STUB_ORDER_FILE": str(order_file),
                    "STUB_INFERENCE_SCENARIO": "directional-drift",
                    "STUB_EMBED_SCENARIO": "true-regression",
                },
            )
            self.assertEqual(
                result.returncode,
                1,
                "a confirmed regression was suppressed by an unmeasurable "
                f"target\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertIn("**⏸ NOT MEASURABLE**", result.stdout)
            self.assertIn("**❌ 1 FAIL**", result.stdout)
            self.assertIn("gate reported a confirmed regression", result.stderr)

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
            # Path is target-keyed (lattice#bench-criterion-root-per-target):
            # BENCHES_INFERENCE defaults to elementwise_cpu_bench, so that is
            # the segment between the crate and "criterion" here.
            bench = (
                root / ".cache" / "bench-compare-criterion" / "head" /
                "inference" / "elementwise_cpu_bench" / "criterion" /
                "rms_norm" / "4096"
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
        # The per-target root is now wiped wholesale before the head phase
        # (lattice#bench-criterion-root-per-target), so the selective
        # clear_selected_head_artifacts prune this message reports on always
        # finds the root already empty here — the seeded stale artifact
        # never survives to be found. The invariant this test guards (a stale
        # same-path comparison cannot satisfy head completeness) is proven by
        # the exit-2 assertion above, which fires because the real head stub
        # genuinely never measured rms_norm/4096 in this run, independent of
        # whichever mechanism cleared any prior data.
        self.assertIn("removed 0 stale head artifact directories", result.stdout)

    def test_stale_unrelated_selected_baseline_is_pruned_before_copy(self):
        """A prior alternate-target baseline must not join today's base set.

        Pre-lattice#bench-criterion-root-per-target, this asserted that
        clear_selected_baseline_artifacts selectively pruned only the exact
        selected-baseline dir, leaving an unrelated differently-named
        baseline snapshot in the same root untouched. That fix now wipes the
        whole arm-and-target root before either phase writes into it
        (bench-compare-impl.sh's clear_criterion_root), which is a stronger
        guarantee: a bench-compare-owned per-target root holds nothing but
        this run's own artifacts, so nothing sharing that root — selected
        baseline or not — should outlive one invocation. Both stale seeds
        below (old_group/42 and the differently-named manual-snapshot) are
        gone by construction; this test now asserts that directly rather than
        asserting the old selective-prune's "removed N" message, which always
        reads 0 now that the wipe runs first.
        """
        old_group_dir = None
        unrelated_dir = None

        def seed_unrelated_run(root):
            nonlocal old_group_dir, unrelated_dir
            # Path is target-keyed: BENCHES_INFERENCE defaults to
            # elementwise_cpu_bench, the segment between crate and
            # "criterion".
            target_root = (
                root / ".cache" / "bench-compare-criterion" / "head" /
                "inference" / "elementwise_cpu_bench" / "criterion"
            )
            bench = target_root / "old_group" / "42"
            old_group_dir = bench
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

            # A different target's persistent, differently-named baseline
            # snapshot sharing the same criterion root before this run wipes
            # it. It has no new/change, so it cannot trip the head-ids
            # coverage check either way; it is here to prove the wipe is
            # total rather than scoped to the selected baseline name.
            other = target_root / "other_group" / "7"
            unrelated_dir = other / "manual-snapshot"
            unrelated_dir.mkdir(parents=True)
            (unrelated_dir / "estimates.json").write_text(
                '{"mean":{"point_estimate":50.0}}\n'
            )

        snapshot = {}

        def capture(root):
            snapshot["old_group_new_exists"] = old_group_dir and (old_group_dir / "new").exists()
            snapshot["old_group_change_exists"] = old_group_dir and (old_group_dir / "change").exists()
            snapshot["unrelated_exists"] = (
                unrelated_dir and (unrelated_dir / "estimates.json").exists()
            )

        result = _run(
            ["--fail-on-regression"],
            stub_cargo=STALE_CHANGE_CARGO,
            setup=seed_unrelated_run,
            post_run=capture,
        )
        self.assertEqual(
            result.returncode, 0,
            f"fresh complete A/B was contaminated by stale unrelated data\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn(
            "removed 0 stale selected-baseline artifact directories before fresh base copy",
            result.stdout,
        )
        self.assertNotIn("old_group/42", result.stdout)
        self.assertFalse(
            snapshot["old_group_new_exists"],
            "stale old_group/42/new must not survive the arm-and-target root wipe",
        )
        self.assertFalse(
            snapshot["old_group_change_exists"],
            "stale old_group/42/change must not survive the arm-and-target root wipe",
        )
        self.assertFalse(
            snapshot["unrelated_exists"],
            "a differently-named baseline snapshot must not survive the arm-and-target "
            "root wipe either — the wipe is total, not scoped to the selected baseline",
        )

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
        self.assertIn("fixture rsync: partial baseline transfer", result.stderr)

    def test_each_bench_target_gets_a_distinct_criterion_root(self):
        """Target identity must be structural, not reconstructed from group names.

        Mutation-sensitive: point EMBED_CRITERION_ROOT at the inference root (or
        drop either CRITERION_HOME assignment) and the observed path set has one
        member or includes `<unset>`, reproducing the shared namespace behind
        #1090. The stub emits only its inherited CRITERION_HOME; no benchmark
        implementation is duplicated here.

        Roots are keyed by bench TARGET as well as by crate (lattice#bench-
        criterion-root-per-target): every root's path includes the exact
        target name as its own path component, not just the crate name, so
        two different targets under the same crate can never collide.
        """
        result = _run(
            [], stub_cargo=STALE_CHANGE_CARGO, emit_criterion_home=True
        )
        self.assertEqual(
            result.returncode, 0,
            f"reporter-mode probe failed\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}")
        roots = set(re.findall(r"criterion-home=(\S+)", result.stdout))
        self.assertEqual(
            len(roots), 8,
            f"expected isolated ABBA roots per target, saw {roots}\n"
            f"stdout:\n{result.stdout}")
        self.assertNotIn("<unset>", roots)
        self.assertTrue(
            any("/inference/elementwise_cpu_bench/criterion" in path for path in roots),
            roots,
        )
        self.assertTrue(
            any("/embed/simd/criterion" in path for path in roots), roots
        )

    def test_different_bench_targets_get_different_criterion_roots(self):
        """The actual defect: two runs choosing different BENCHES_INFERENCE
        values must never write into the same Criterion root.

        Mutation-sensitive: drop the target-name path segment (key the root by
        crate alone, as the pre-fix script did) and both invocations report
        the same criterion-home path.
        """
        def add_f16_convert_bench_source(root):
            # STALE_CHANGE_CARGO picks its fabricated group name (rms_norm)
            # from the CRATE in argv, not the --bench target, so the fixture
            # source declares the same group name regardless of target — this
            # test asserts path distinctness, not group-content reconciliation
            # (that is covered separately by the reconciliation tests below).
            # The file must be committed, not just written: bench-compare
            # reads it from a `git worktree add --detach` checkout of HEAD,
            # which only contains tracked, committed content.
            path = root / "crates" / "inference" / "benches" / "f16_convert_bench.rs"
            path.write_text('let mut group = c.benchmark_group("rms_norm");\n')
            subprocess.run([*GIT, "-C", str(root), "add", "-f", str(path)], check=True)
            subprocess.run(
                [*GIT, "-C", str(root), "commit", "-qm", "add f16_convert_bench fixture"],
                check=True,
                env={
                    **os.environ,
                    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                },
            )

        first = _run(
            [], stub_cargo=STALE_CHANGE_CARGO, emit_criterion_home=True
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        second = _run(
            [],
            stub_cargo=STALE_CHANGE_CARGO,
            emit_criterion_home=True,
            extra_env={"BENCHES_INFERENCE": "f16_convert_bench"},
            setup=add_f16_convert_bench_source,
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        first_roots = set(re.findall(r"criterion-home=(\S+)", first.stdout))
        second_roots = set(re.findall(r"criterion-home=(\S+)", second.stdout))
        inference_first = {p for p in first_roots if "/inference/" in p}
        inference_second = {p for p in second_roots if "/inference/" in p}
        self.assertTrue(inference_first, first.stdout)
        self.assertTrue(inference_second, second.stdout)
        self.assertFalse(
            inference_first & inference_second,
            f"different BENCHES_INFERENCE values shared a root: "
            f"{inference_first} vs {inference_second}",
        )


class _FailOnEmptyTestProgram(unittest.TestProgram):
    def runTests(self) -> None:
        if self.test.countTestCases() == 0:
            raise SystemExit("no tests collected")
        super().runTests()


if __name__ == "__main__":
    _FailOnEmptyTestProgram()

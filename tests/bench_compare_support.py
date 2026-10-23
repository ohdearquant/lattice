"""Shared fixtures for the bench-compare measurement and prebuild suites."""

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

# STUB_PHASE_SAMPLER's canonical copy lives in
# tests/fixtures/bench_phase_sampler_stub.py; test_bench_locks.py loads the
# same copy (lattice#1515 Amendment 2 — do not duplicate the stub text).
_PHASE_STUB_SPEC = importlib.util.spec_from_file_location(
    "bench_phase_sampler_stub",
    REPO / "tests" / "fixtures" / "bench_phase_sampler_stub.py",
)
assert _PHASE_STUB_SPEC is not None and _PHASE_STUB_SPEC.loader is not None
_phase_stub = importlib.util.module_from_spec(_PHASE_STUB_SPEC)
_PHASE_STUB_SPEC.loader.exec_module(_phase_stub)
STUB_PHASE_SAMPLER = _phase_stub.STUB_PHASE_SAMPLER

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

# Disable hooks and auto-gc so disposable Git fixtures have no automatic GC writer.
GIT = ("git", "-c", "core.hooksPath=/dev/null", "-c", "gc.auto=0")

STALE_CHANGE_CARGO = r"""#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--version" ]]; then
  printf '%s\n' 'cargo 1.94.1 (fixture)'
  exit 0
fi

if [[ -n "${STUB_CARGO_ARGV_FILE:-}" ]]; then
  {
    printf '%q ' "$@"
    printf '\n'
  } >> "$STUB_CARGO_ARGV_FILE"
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
    printf '%s\n' '${STUB_INFERENCE_GROUP_DIR:-rms_norm}/896: benchmark'
  else
    printf '%s\n' 'simd_dot_product/scalar/384: benchmark'
  fi
  exit 0
fi

if [[ "$args" == *" --save-baseline "* ]]; then
  baseline_name="${args#* --save-baseline }"
  baseline_name="${baseline_name%% *}"
  if [[ "$args" == *" lattice-inference "* ]]; then
    write_baseline "${STUB_INFERENCE_GROUP_DIR:-rms_norm}/896" "$baseline_name"
    write_baseline "${STUB_INFERENCE_GROUP_DIR:-rms_norm}/4096" "$baseline_name"
  else
    write_baseline "simd_dot_product/scalar/384" "$baseline_name"
  fi
else
  if [[ "$args" == *" lattice-inference "* ]]; then
    write_head "${STUB_INFERENCE_GROUP_DIR:-rms_norm}/896"
    if [[ "${STUB_REMOVE_RMS_4096:-0}" != "1" ]]; then
      write_head "${STUB_INFERENCE_GROUP_DIR:-rms_norm}/4096"
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


def run_compare(
    extra_args,
    *,
    stub_cargo=STUB_CARGO,
    stub_rsync=None,
    setup=None,
    extra_env=None,
    emit_criterion_home=False,
    stub_machine_state=None,
    stub_phase_sampler=None,
    stub_quiet_probe=None,
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
            if stub_quiet_probe is None else stub_quiet_probe
        )
        machine_probe = root / "scripts" / "lib" / "machine-state-probe.py"
        machine_probe.write_text(
            STUB_MACHINE_PROBE
            if stub_machine_state is None
            else stub_machine_state
        )
        phase_sampler = root / "scripts" / "lib" / "phase-load-sampler.py"
        phase_sampler.write_text(
            STUB_PHASE_SAMPLER if stub_phase_sampler is None else stub_phase_sampler
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
        # Tests default to BENCHES_INFERENCE=elementwise_cpu_bench and
        # BENCHES_EMBED=simd. Override cases add and commit their selected
        # target's fixture source before the detached worktrees are created.
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
            **{key: value for key, value in os.environ.items()
               if key != "CARGO_BUILD_TARGET_DIR"},
            "CARGO_TARGET_DIR": str(root / "target"),
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "BENCH_IDLE_FLOOR": "0",
            "LATTICE_BENCH_HOST_ID_FILE": f"{tmp}/bench-host-id", "STUB_REPO_ROOT": str(root),
            **(extra_env or {}),
            "STUB_EMIT_CRITERION_HOME": "1" if emit_criterion_home else "0",
        }
        result = subprocess.run(
            ["bash", str(root / "scripts" / SCRIPT.name), *extra_args, "HEAD~1", "HEAD"],
            capture_output=True, text=True, env=env, timeout=300)
        if post_run is not None:
            post_run(root)
        return result


class GpuBenchmarkAdmissionSupport:
    """Admission fixtures shared by suites without contributing test cases."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("bench_admission", LIB / "bench_admission.py")
        assert spec is not None and spec.loader is not None
        cls.admission = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.admission
        spec.loader.exec_module(cls.admission)

    def manifest(self, *, eligible=True, bins=None, bin_tables=None, autobins=None):
        admission = self.admission
        bins = list(admission.BINS) if bins is None else bins
        if bin_tables is None:
            bin_tables = [{"name": name, "path": f"src/bin/{name}.rs", "required-features": ["f16", "metal-gpu"]}
                          for name in ("bench_decode_ab", "bench_logit_dump")]
        return {
            "package": {
                "name": "lattice-inference", "version": {"workspace": True},
                "metadata": {"gpu-bench-handoff": {
                    "version": 1, "targets": list(admission.TARGETS), "bins": bins}}
                if eligible else {},
                **({} if autobins is None else {"autobins": autobins}),
            },
            "bin": bin_tables,
            "features": {
                "default": ["std", "download", "serve"], "std": [],
                "download": ["dep:ureq"], "serve": ["dep:axum"], "f16": [],
                "metal-gpu": ["dep:metal"], "bench-internals": [],
                "metal-bench": ["metal-gpu", "f16"],
            },
            "bench": [{"name": target, "harness": False} for target in admission.TARGETS],
        }

    def compare(self, *, base_eligible=True, target="decode_attn_bench", features="metal-gpu,f16", args=None):
        from unittest import mock
        admission = self.admission
        base, head = "a" * 40, "b" * 40
        def manifest(repo, revision, path):
            if path == "Cargo.toml":
                return {"workspace": {"package": {"version": "0.0.1"}}}
            return self.manifest(eligible=base_eligible or revision != base)
        with mock.patch.object(admission.sys, "platform", "darwin"), \
             mock.patch.object(admission, "_revision", side_effect=lambda repo, ref: base if ref == "base" else head), \
             mock.patch.object(admission, "_manifest", side_effect=manifest), \
             mock.patch.object(admission, "_git", return_value="source"), \
             mock.patch.object(admission, "_clean_revision"):
            return admission.plan_compare(Path("/tmp/admission-fixture"), args or ["base", "head"], {
                "BENCHES_INFERENCE": target, "CARGO_FEATURES_INFERENCE": features,
                "BENCH_GROUPS_INFERENCE": "reference|flash", "BENCHES_EMBED": "simd",
            })


class FailOnEmptyTestProgram(unittest.TestProgram):
    def runTests(self) -> None:
        if self.test.countTestCases() == 0:
            raise SystemExit("no tests collected")
        super().runTests()

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
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
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

# STUB_PHASE_SAMPLER itself is defined once in
# tests/fixtures/bench_phase_sampler_stub.py (imported above) and shared with
# test_bench_locks.py. The LOUD/DEAD/READY-BUT-EMPTY variants below build on
# the same shape but are specific to this file's tests, so they stay local.

# Emits foreign=LOUD_FOREIGN_PCT samples only for the arm named by
# LOUD_ARM (env vars), quiet samples for every other arm. Writes `.ready`
# unconditionally (every arm actually starts and samples), matching the real
# sampler's readiness contract (lattice#1515 Amendment 2).
STUB_PHASE_SAMPLER_LOUD_ON_ONE_ARM = """#!/usr/bin/env python3
import json
import os
import pathlib
import signal
import sys
import time


def _handle(signum, frame):
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _handle)
signal.signal(signal.SIGINT, _handle)

arm = sys.argv[sys.argv.index("--arm") + 1]
out = sys.argv[sys.argv.index("--out") + 1]
pathlib.Path(out + ".ready").touch()
loud_arm = os.environ.get("LOUD_ARM")
foreign = float(os.environ.get("LOUD_FOREIGN_PCT", "45.0")) if arm == loud_arm else 0.0
with open(out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({
        "schema": "perf-phase-sample/v1",
        "arm": arm,
        "captured_utc": "2026-01-01T00:00:00Z",
        "idle_pct": 100.0 - foreign,
        "foreign_pct": foreign,
        "self_pct": 5.0,
        "top_foreign": "loud_stub" if foreign else "none",
        "top_foreign_pct": foreign,
    }) + "\\n")
    fh.flush()
    while True:
        time.sleep(0.05)
"""


STUB_QUIET_STATUS_PROBE = (
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "label = sys.argv[sys.argv.index('--label') + 1]\n"
        "phase = sys.argv[sys.argv.index('--phase') + 1]\n"
        "if '--jsonl-out' in sys.argv:\n"
        "    path = sys.argv[sys.argv.index('--jsonl-out') + 1]\n"
        "    with open(path, 'a') as out:\n"
        "        out.write(json.dumps({'schema': 'perf-ambient-sample/v1', "
        "'phase': phase, 'idle_pct': 100.0}) + '\\n')\n"
        "print(f'[quiet] {label}: idle 100.0% (floor 70.0%) ok | top: fixture 0.0%')\n"
)

# Never touches `.ready` for the arm named by DEAD_ARM: a genuinely dead
# instrument, one that never even signaled it started. Writes `.ready` (and
# a quiet sample) for every other arm. This is the process-never-started
# case (distinct from STARTED_BUT_NO_SAMPLES below, which does signal ready
# but then produces nothing).
STUB_PHASE_SAMPLER_DEAD_ON_ONE_ARM = """#!/usr/bin/env python3
import json
import os
import pathlib
import signal
import sys
import time


def _handle(signum, frame):
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _handle)
signal.signal(signal.SIGINT, _handle)

arm = sys.argv[sys.argv.index("--arm") + 1]
out = sys.argv[sys.argv.index("--out") + 1]
dead_arm = os.environ.get("DEAD_ARM")
if arm != dead_arm:
    pathlib.Path(out + ".ready").touch()
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "schema": "perf-phase-sample/v1",
            "arm": arm,
            "captured_utc": "2026-01-01T00:00:00Z",
            "idle_pct": 100.0,
            "foreign_pct": 0.0,
            "self_pct": 5.0,
            "top_foreign": "none",
            "top_foreign_pct": 0.0,
        }) + "\\n")
        fh.flush()
while True:
    time.sleep(0.05)
"""

# Touches `.ready` for EVERY arm (it started fine) but writes no sample at
# all for the arm named by DEAD_ARM: a started-but-produced-nothing
# instrument, which phase-load-report.py's own zero-records check must
# still refuse ("no usable samples"), distinct from the never-started case
# above ("did not start").
STUB_PHASE_SAMPLER_STARTED_BUT_NO_SAMPLES_ON_ONE_ARM = """#!/usr/bin/env python3
import json
import os
import pathlib
import signal
import sys
import time


def _handle(signum, frame):
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _handle)
signal.signal(signal.SIGINT, _handle)

arm = sys.argv[sys.argv.index("--arm") + 1]
out = sys.argv[sys.argv.index("--out") + 1]
pathlib.Path(out + ".ready").touch()
dead_arm = os.environ.get("DEAD_ARM")
if arm != dead_arm:
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "schema": "perf-phase-sample/v1",
            "arm": arm,
            "captured_utc": "2026-01-01T00:00:00Z",
            "idle_pct": 100.0,
            "foreign_pct": 0.0,
            "self_pct": 5.0,
            "top_foreign": "none",
            "top_foreign_pct": 0.0,
        }) + "\\n")
        fh.flush()
while True:
    time.sleep(0.05)
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
elif [[ "$scenario" == "stable" ]]; then
  a1="100.0"
  b1="100.0"
  b2="100.0"
  a2="100.0"
  forward_point="0.0"
  forward_low="-0.001"
  forward_high="0.001"
  reverse_point="0.0"
  reverse_low="-0.001"
  reverse_high="0.001"
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


def _add_embeddings_bench_source(root):
    # The cargo fixtures fabricate simd_dot_product for every embed target,
    # so the selected source declares that group for the harness's
    # declared-vs-measured reconciliation step.
    path = root / "crates" / "embed" / "benches" / "embeddings.rs"
    path.write_text('let mut group = c.benchmark_group("simd_dot_product");\n')
    subprocess.run(
        [*GIT, "-C", str(root), "add", "-f", str(path)], check=True
    )
    subprocess.run(
        [*GIT, "-C", str(root), "commit", "-qm", "add embeddings fixture"],
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        },
    )


def _add_inference_bench_source(root):
    # The cargo fixtures key on the CRATE (`-p lattice-inference`), not the target name, so
    # they fabricate rms_norm for every inference target. The selected source therefore
    # declares that group for the harness's declared-vs-measured reconciliation step, exactly
    # as the embed helper above does.
    path = root / "crates" / "inference" / "benches" / "f16_convert_bench.rs"
    path.write_text('let mut group = c.benchmark_group("rms_norm");\n')
    subprocess.run(
        [*GIT, "-C", str(root), "add", "-f", str(path)], check=True
    )
    subprocess.run(
        [*GIT, "-C", str(root), "commit", "-qm", "add f16_convert_bench fixture"],
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        },
    )


def _captured_embed_argv(path):
    invocations = [shlex.split(line) for line in path.read_text().splitlines()]
    return [
        argv
        for argv in invocations
        if "-p" in argv
        and argv.index("-p") + 1 < len(argv)
        and argv[argv.index("-p") + 1] == "lattice-embed"
    ]


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

    def test_embed_bench_target_honors_caller_override(self):
        """BENCHES_EMBED selects an existing embed target for the whole run.

        Mutation-sensitive: restore the pre-fix bare ``BENCHES_EMBED="simd"``
        assignment and all four captured embed commands and roots remain keyed
        by ``simd`` rather than the caller-selected ``embeddings`` target.
        """
        with tempfile.TemporaryDirectory() as temporary:
            argv_file = Path(temporary) / "cargo-argv.txt"
            result = _run(
                [],
                stub_cargo=STALE_CHANGE_CARGO,
                emit_criterion_home=True,
                extra_env={
                    "BENCHES_EMBED": "embeddings",
                    "STUB_CARGO_ARGV_FILE": str(argv_file),
                },
                setup=_add_embeddings_bench_source,
            )
            embed_argv = _captured_embed_argv(argv_file)
        self.assertEqual(
            result.returncode, 0,
            f"embed target override failed\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )
        self.assertEqual(len(embed_argv), 4, embed_argv)
        for argv in embed_argv:
            bench_index = argv.index("--bench")
            self.assertEqual(argv[bench_index + 1], "embeddings", argv)
            self.assertNotIn("simd", argv)
            self.assertNotIn("--features", argv)
        roots = set(re.findall(r"criterion-home=(\S+)", result.stdout))
        embed_roots = {path for path in roots if "/embed/" in path}
        self.assertEqual(
            len(embed_roots), 4,
            f"expected one embed root per ABBA phase, saw {embed_roots}\n"
            f"stdout:\n{result.stdout}",
        )
        self.assertTrue(
            all("/embed/embeddings/criterion" in path for path in embed_roots),
            embed_roots,
        )
        self.assertIn(
            "targets: lattice-inference:elementwise_cpu_bench, "
            "lattice-embed:embeddings",
            result.stdout,
        )
        self.assertIn("embed features: <none>", result.stdout)
        self.assertIn("embed_features=<none>", result.stdout)

    def test_embed_features_reach_all_four_abba_commands_and_provenance(self):
        """CARGO_FEATURES_EMBED is one exact argv value in every embed arm."""
        with tempfile.TemporaryDirectory() as temporary:
            argv_file = Path(temporary) / "cargo-argv.txt"
            captured_provenance = []

            def capture_provenance(root):
                captured_provenance.append(
                    (root / ".cache" / "bench-run-provenance.txt").read_text()
                )

            result = _run(
                [],
                stub_cargo=STALE_CHANGE_CARGO,
                extra_env={
                    "BENCHES_EMBED": "embeddings",
                    "CARGO_FEATURES_EMBED": "native,prepared-bench",
                    "STUB_CARGO_ARGV_FILE": str(argv_file),
                },
                setup=_add_embeddings_bench_source,
                post_run=capture_provenance,
            )
            embed_argv = _captured_embed_argv(argv_file)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(embed_argv), 4, embed_argv)
        for argv in embed_argv:
            self.assertEqual(argv.count("--features"), 1, argv)
            feature_index = argv.index("--features")
            self.assertEqual(argv[feature_index + 1], "native,prepared-bench", argv)
        self.assertIn("embed features: native,prepared-bench", result.stdout)
        self.assertIn("embed_features=native,prepared-bench", result.stdout)
        self.assertEqual(len(captured_provenance), 1)
        self.assertTrue(
            captured_provenance[0].startswith(
                "schema=lattice-bench-provenance-v2\n"
            ),
            captured_provenance[0],
        )

    def test_uncalibrated_embed_target_is_informational_at_every_resolution(self):
        """A custom embed target cannot inherit simd's calibrated full gate.

        Mutation-sensitive: classify every selected embed target as calibrated
        and the fabricated +20% custom-target regression exits 1 in both quick
        and full enforcing modes.
        """
        for flags, resolution in (([], "quick"), (["--full"], "full")):
            with self.subTest(resolution=resolution):
                with tempfile.TemporaryDirectory() as temporary:
                    order_file = Path(temporary) / "order.txt"
                    result = _run(
                        [*flags, "--fail-on-regression"],
                        stub_cargo=ORDER_BALANCE_CARGO,
                        extra_env={
                            "BENCHES_EMBED": "embeddings",
                            "STUB_ORDER_FILE": str(order_file),
                            "STUB_INFERENCE_SCENARIO": "stable",
                            "STUB_EMBED_SCENARIO": "true-regression",
                        },
                        setup=_add_embeddings_bench_source,
                    )
                self.assertEqual(
                    result.returncode, 0,
                    f"uncalibrated {resolution} target voted on a regression\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                )
                self.assertIn("**ℹ️ 1 informational**", result.stdout)
                self.assertIn("explicit target policy", result.stdout)
                self.assertNotIn("re-run `--full` for a gated verdict", result.stdout)
                self.assertNotIn("informational-only in quick mode", result.stdout)
                self.assertNotIn("gate reported a confirmed regression", result.stderr)

    def test_feature_changed_default_embed_target_is_not_calibrated(self):
        """Full-gate calibration binds the target and feature selection."""
        with tempfile.TemporaryDirectory() as temporary:
            order_file = Path(temporary) / "order.txt"
            result = _run(
                ["--full", "--fail-on-regression"],
                stub_cargo=ORDER_BALANCE_CARGO,
                extra_env={
                    "CARGO_FEATURES_EMBED": "native,local",
                    "STUB_ORDER_FILE": str(order_file),
                    "STUB_INFERENCE_SCENARIO": "stable",
                    "STUB_EMBED_SCENARIO": "true-regression",
                },
            )
        self.assertEqual(
            result.returncode, 0,
            "a feature-modified simd target inherited the default instrument's "
            f"full gate\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("**ℹ️ 1 informational**", result.stdout)

    def test_uncalibrated_inference_target_is_informational_at_every_resolution(self):
        """A non-default inference target cannot vote at full resolution either.

        The allowlist used to test ``lattice-embed:*`` only, so at full resolution an
        inference target outside the calibrated set reached neither demotion path and
        voted on thresholds nobody calibrated for it. That had no instance while the
        PR-time driver could pass group filters and nothing else; it acquired one the
        moment the target and features became reachable.

        Mutation-sensitive: restore the ``[[ "$target" == lattice-embed:* ]] &&``
        guard on the calibration branch and the fabricated inference regression exits
        1 at full resolution.

        The two resolutions end on different exit codes and the difference carries
        the rule. At full resolution ``lattice-embed:simd`` still gates, so the run
        holds an aggregate verdict and the demoted inference regression is simply
        excluded from it. In quick mode the manifest demotes ``simd`` as well, so no
        arm gated at all and the run has no verdict to render. Neither may exit 1:
        an uncalibrated target must not vote either way.
        """
        for flags, resolution, expected_rc in (
            ([], "quick", 3),
            (["--full"], "full", 0),
        ):
            with self.subTest(resolution=resolution):
                with tempfile.TemporaryDirectory() as temporary:
                    order_file = Path(temporary) / "order.txt"
                    result = _run(
                        [*flags, "--fail-on-regression"],
                        stub_cargo=ORDER_BALANCE_CARGO,
                        extra_env={
                            "BENCHES_INFERENCE": "f16_convert_bench",
                            "STUB_ORDER_FILE": str(order_file),
                            "STUB_INFERENCE_SCENARIO": "true-regression",
                            "STUB_EMBED_SCENARIO": "stable",
                        },
                        setup=_add_inference_bench_source,
                    )
                self.assertEqual(
                    result.returncode, expected_rc,
                    f"uncalibrated {resolution} inference target did not land on "
                    f"the expected disposition\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                )
                self.assertIn("**ℹ️ 1 informational**", result.stdout)
                self.assertNotIn("gate reported a confirmed regression", result.stderr)
                if expected_rc == 3:
                    self.assertIn(
                        "no target in this run held gating authority",
                        result.stderr,
                    )
                    self.assertNotIn(
                        "gated benches within noise band", result.stdout
                    )
                else:
                    self.assertNotIn("held gating authority", result.stderr)

    def test_feature_changed_default_inference_target_is_not_calibrated(self):
        """Calibration binds target AND features on the inference side too.

        Same target, different compiled binary: the features are chosen before any
        benchmark function runs, so the calibrated pair's thresholds say nothing about
        this one.
        """
        with tempfile.TemporaryDirectory() as temporary:
            order_file = Path(temporary) / "order.txt"
            result = _run(
                ["--full", "--fail-on-regression"],
                stub_cargo=ORDER_BALANCE_CARGO,
                extra_env={
                    "CARGO_FEATURES_INFERENCE": "bench-internals",
                    "STUB_ORDER_FILE": str(order_file),
                    "STUB_INFERENCE_SCENARIO": "true-regression",
                    "STUB_EMBED_SCENARIO": "stable",
                },
            )
        self.assertEqual(
            result.returncode, 0,
            "a feature-modified elementwise_cpu_bench inherited the default "
            f"instrument's full gate\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )
        self.assertIn("**ℹ️ 1 informational**", result.stdout)

    def test_default_inference_configuration_still_votes_at_full_resolution(self):
        """The arm that fails if the generalization demoted everything.

        Widening an allowlist to cover a second crate has an obvious failure mode in
        the safe-looking direction: classify the default pair as uncalibrated too and
        every assertion above passes while the gate stops gating. This is the same
        fixture as the two arms above with the default target and no features, and it
        must exit 1.
        """
        with tempfile.TemporaryDirectory() as temporary:
            order_file = Path(temporary) / "order.txt"
            result = _run(
                ["--full", "--fail-on-regression"],
                stub_cargo=ORDER_BALANCE_CARGO,
                extra_env={
                    "STUB_ORDER_FILE": str(order_file),
                    "STUB_INFERENCE_SCENARIO": "true-regression",
                    "STUB_EMBED_SCENARIO": "stable",
                },
            )
        self.assertEqual(
            result.returncode, 1,
            "the default inference configuration stopped voting\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("gate reported a confirmed regression", result.stderr)
        self.assertNotIn("**ℹ️ 1 informational**", result.stdout)


    def test_status_records_loud_arm_with_quiet_boundaries(self):
        documents = {}
        for foreign_pct in (45, 0):
            with self.subTest(foreign_pct=foreign_pct):
                with tempfile.TemporaryDirectory() as status_tmp:
                    result = _run(
                        ["--fail-on-regression"],
                        stub_cargo=STALE_CHANGE_CARGO,
                        stub_phase_sampler=STUB_PHASE_SAMPLER_LOUD_ON_ONE_ARM,
                        stub_quiet_probe=STUB_QUIET_STATUS_PROBE,
                        extra_env={
                            "PERF_POSTMERGE_STATUS_DIR": status_tmp,
                            "LOUD_ARM": "head1",
                            "LOUD_FOREIGN_PCT": str(foreign_pct),
                            "BENCH_IDLE_FLOOR": "70",
                        },
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    statuses = sorted(Path(status_tmp).glob("*.json"))
                    self.assertEqual(len(statuses), 2, result.stdout + result.stderr)
                    documents[foreign_pct] = [json.loads(p.read_text()) for p in statuses]
                    for status in documents[foreign_pct]:
                        self.assertEqual(
                            status.get("phase_load_verdict"),
                            "LOUD" if foreign_pct else "ok",
                            status,
                        )
                        self.assertEqual(status["phase_load"], [
                            {
                                "arm": arm,
                                "verdict": "LOUD" if arm == "head1" and foreign_pct else "ok",
                                "foreign_max_pct": float(foreign_pct) if arm == "head1" else 0.0,
                                "floor_pct": 70.0,
                            }
                            for arm in ("base1", "head1", "head2", "base2")
                        ])
                        self.assertEqual(status["ambient"]["assessment"], "valid")
                        self.assertEqual(status["ambient"]["samples"], {
                            phase: 100.0
                            for phase in ("before", "between", "after")
                        })
        for quiet, loud in zip(documents[0], documents[45]):
            for key in ("phase_load", "phase_load_verdict"):
                quiet.pop(key)
                loud.pop(key)
            self.assertEqual(quiet, loud)

    def test_interactive_last_loud_arm_still_refuses(self):
        result = _run(
            ["--fail-on-regression"],
            stub_cargo=STALE_CHANGE_CARGO,
            stub_phase_sampler=STUB_PHASE_SAMPLER_LOUD_ON_ONE_ARM,
            extra_env={
                "LOUD_ARM": "base2",
                "LOUD_FOREIGN_PCT": "45",
                "BENCH_IDLE_FLOOR": "70",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("in-phase load during 'base2'", result.stderr)
        self.assertNotIn("Done.", result.stdout)

    def test_loud_in_phase_load_refuses_the_run(self):
        """A stub sampler emitting 45% foreign load during head1 only must refuse.

        lattice#1515: the three boundary probes above cannot see a load that
        starts and ends between two of them. This asserts the in-phase gate
        DOES see it.

        Mutation-sensitive: drop (or neutralize, e.g. by forcing the ceiling
        to 100) phase_gate's rc==1 refusal branch in bench-compare-impl.sh
        and this run flips from exit 2 to exit 0 with a rendered report.
        """
        result = _run(
            [],
            stub_cargo=STALE_CHANGE_CARGO,
            stub_phase_sampler=STUB_PHASE_SAMPLER_LOUD_ON_ONE_ARM,
            extra_env={
                "LOUD_ARM": "head1",
                "LOUD_FOREIGN_PCT": "45",
                "BENCH_IDLE_FLOOR": "70",
            },
        )
        self.assertEqual(
            result.returncode, 2,
            f"expected exit 2 (loud in-phase load), got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("head1", result.stderr)
        self.assertIn("in-phase load", result.stderr)
        self.assertNotIn("Done.", result.stdout)
        self.assertNotIn("=== Run conditions ===", result.stdout)

    def test_quiet_in_phase_load_reports_all_four_arms_in_abba_order(self):
        """A healthy run must render one In-phase load line per arm, in ABBA order.

        Mutation-sensitive: skip phase_gate for any arm (or drop the
        PHASE_LOAD_SAMPLES accumulation) and the arm count below drops
        below 4, or the order assertion fails.
        """
        result = _run([], stub_cargo=STALE_CHANGE_CARGO, extra_env={"BENCH_IDLE_FLOOR": "70"})
        self.assertEqual(
            result.returncode, 0,
            f"healthy in-phase fixture failed\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )
        arms = re.findall(r"\[phase-load\] (\S+):", result.stdout)
        # Each arm's line appears four times: once printed live by phase_gate
        # as the arm completes, once again rendered into the "Run conditions"
        # summary block at the end of the run, and once more inside EACH of
        # the two target-qualified gate reports (inference, embed) that
        # render the shared provenance file's phase_load= lines.
        self.assertEqual(
            arms, ["base1", "head1", "head2", "base2"] * 4, result.stdout
        )
        self.assertIn("in-phase load", result.stdout)
        for arm in ("base1", "head1", "head2", "base2"):
            self.assertIn(
                f"[phase-load] {arm}: samples=1 idle min/mean=100.0%/100.0% "
                f"foreign max/mean=0.0%/0.0% self mean=5.0%",
                result.stdout,
                result.stdout,
            )

    def test_dead_in_phase_sampler_never_starts_and_refuses(self):
        """A sampler process that never signals readiness for one arm must refuse.

        The stub never touches `.ready` for base2 (it never even opens the
        output file for that arm), reproducing a sampler that crashed or was
        killed before phase_sampler_start's poll window closed (lattice#1515
        Amendment 2: the same failure mode a fast-ending Linux CI arm hit
        against the real sampler, before the do-while-shape + readiness-
        marker fix). This is refused as "did not start" — an instrument
        that never started is a distinct failure from one that started and
        produced nothing (see test_started_but_no_samples below).

        Mutation-sensitive: drop phase_sampler_start's readiness poll/exit-2
        in bench-compare-impl.sh and this run flips from exit 2 to hanging
        (no cargo/arm ever validates readiness) or, if the poll is merely
        neutered to always succeed, to exit 0 with a fabricated in-phase
        report despite the sampler having produced zero rows for base2.
        """
        result = _run(
            [],
            stub_cargo=STALE_CHANGE_CARGO,
            stub_phase_sampler=STUB_PHASE_SAMPLER_DEAD_ON_ONE_ARM,
            extra_env={"DEAD_ARM": "base2"},
        )
        self.assertEqual(
            result.returncode, 2,
            f"expected exit 2 (in-phase sampler never started), got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("base2", result.stderr)
        self.assertIn("did not start", result.stderr)

    def test_started_in_phase_sampler_with_no_samples_still_refuses(self):
        """A sampler that signals ready but writes zero samples is still a failed instrument.

        Distinct from the never-started case above: here `.ready` appears
        for base2 (phase_sampler_start's poll succeeds, so the harness
        proceeds to run the arm), but the sampler never wrote a row for it.
        phase-load-report.py's own zero-records check must still refuse this
        — starting is necessary but not sufficient evidence.

        Mutation-sensitive: drop the rc==2 branch from phase_gate in
        bench-compare-impl.sh (or the zero-records check in
        phase-load-report.py) and this run flips from exit 2 to exit 0.
        """
        result = _run(
            [],
            stub_cargo=STALE_CHANGE_CARGO,
            stub_phase_sampler=STUB_PHASE_SAMPLER_STARTED_BUT_NO_SAMPLES_ON_ONE_ARM,
            extra_env={"DEAD_ARM": "base2"},
        )
        self.assertEqual(
            result.returncode, 2,
            f"expected exit 2 (in-phase sampler produced no usable samples), "
            f"got {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("base2", result.stderr)
        self.assertIn("no usable samples", result.stderr)
        self.assertNotIn("did not start", result.stderr)

    def test_provenance_carries_four_phase_load_lines_and_preserves_ambient_lines(self):
        """bench-run-provenance.txt gains four phase_load= lines; ambient= is unchanged."""
        captured = []

        def capture_provenance(root):
            captured.append(
                (root / ".cache" / "bench-run-provenance.txt").read_text()
            )

        result = _run(
            [], stub_cargo=STALE_CHANGE_CARGO, post_run=capture_provenance
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(captured), 1)
        provenance = captured[0]
        phase_lines = [
            line for line in provenance.splitlines() if line.startswith("phase_load=")
        ]
        ambient_lines = [
            line for line in provenance.splitlines() if line.startswith("ambient=")
        ]
        self.assertEqual(len(phase_lines), 4, provenance)
        self.assertEqual(len(ambient_lines), 3, provenance)

    def test_phase_sampler_shares_quiet_probes_idle_parser(self):
        """The sampler's idle parser is a pinned COPY of quiet-probe.py's own.

        phase-load-sampler.py cannot `importlib`-import quiet-probe.py (see
        the PARSER SHARING note in its docstring: several existing test
        fixtures stub quiet-probe.py with unguarded module-level argparse
        calls that would fire against this process's own argv on import), so
        it carries a copy of parse_top_idle/parse_proc_stat/linux_idle_pct
        instead. This test is what keeps that copy honest: both must agree on
        the idle percentage for the same `top -l 2` transcript.

        Mutation-sensitive: edit either copy's regex/logic without mirroring
        the change in the other, and this produces a different number for the
        same transcript below.
        """
        quiet_probe_spec = importlib.util.spec_from_file_location(
            "quiet_probe_direct", LIB / "quiet-probe.py"
        )
        assert quiet_probe_spec is not None and quiet_probe_spec.loader is not None
        quiet_probe_direct = importlib.util.module_from_spec(quiet_probe_spec)
        quiet_probe_spec.loader.exec_module(quiet_probe_direct)

        phase_sampler_spec = importlib.util.spec_from_file_location(
            "phase_load_sampler_direct", LIB / "phase-load-sampler.py"
        )
        assert phase_sampler_spec is not None and phase_sampler_spec.loader is not None
        phase_sampler_direct = importlib.util.module_from_spec(phase_sampler_spec)
        phase_sampler_spec.loader.exec_module(phase_sampler_direct)

        transcript = (
            "Processes: 400 total\n"
            "CPU usage: 5.0% user, 3.0% sys, 92.0% idle\n"
            "\n"
            "Processes: 401 total\n"
            "CPU usage: 12.5% user, 4.5% sys, 83.0% idle\n"
        )
        expected = quiet_probe_direct.parse_top_idle(transcript)
        self.assertEqual(expected, 83.0)
        self.assertEqual(
            phase_sampler_direct.parse_top_idle(transcript), expected
        )

    def test_real_sampler_yields_at_least_one_sample_for_a_sub_100ms_arm(self):
        """The REAL sampler (not a stub) must yield >=1 sample even for a
        near-instant arm.

        lattice#1515 Amendment 2's do-while shape is what makes this true: a
        `while not stop:` loop that checks the stop flag BEFORE the first
        sample would yield zero rows if SIGTERM lands between handler
        install and loop entry -- exactly what a fast-ending Linux CI arm
        hit against the real sampler (9/40 test_bench_locks.py failures, all
        "PROBE FAILED (zero usable samples)"). This is the mutation control
        for that fix.

        Runs scripts/lib/phase-load-sampler.py directly (subprocess), waits
        for its `.ready` marker (proving handlers are installed and it is
        safe to signal), then SIGTERMs it as close to immediately as this
        harness can drive it, and asserts exactly one JSONL row landed.
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sample.jsonl"
            proc = subprocess.Popen(
                [
                    sys.executable, str(LIB / "phase-load-sampler.py"),
                    "--arm", "instant", "--self-pid", str(os.getpid()),
                    "--out", str(out), "--interval", "5",
                ],
            )
            try:
                ready = Path(str(out) + ".ready")
                deadline = time.monotonic() + 10
                while not ready.exists():
                    if time.monotonic() > deadline:
                        self.fail("sampler never signaled readiness within 10s")
                    time.sleep(0.02)
                proc.terminate()
                proc.wait(timeout=10)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)
            lines = [
                line for line in out.read_text().splitlines() if line.strip()
            ] if out.exists() else []
            self.assertEqual(
                len(lines), 1,
                "expected exactly one sample for a near-instant arm, got "
                f"{len(lines)}: {lines}",
            )


class GpuBenchmarkAdmission(unittest.TestCase):
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

    def test_explicit_admission_requires_each_historical_revision(self):
        with self.assertRaisesRegex(self.admission.AdmissionError, "base1.*no supported GPU handoff"):
            self.compare(base_eligible=False)

    def test_admission_freezes_all_four_ordered_invocations_and_criterion_arguments(self):
        plan = self.compare()
        self.assertEqual([entry["id"] for entry in plan["entries"]], ["base1", "head1", "head2", "base2"])
        self.assertEqual([entry["revision"] for entry in plan["entries"]], ["a" * 40, "b" * 40, "b" * 40, "a" * 40])
        self.assertEqual([entry["argv"] for entry in plan["entries"]], [
            ["--bench", "reference|flash", "--save-baseline", "compare-base", "--noplot", "--quick"],
            ["--bench", "reference|flash", "--baseline", "compare-base", "--noplot", "--quick"],
            ["--bench", "reference|flash", "--save-baseline", "compare-head", "--noplot", "--quick"],
            ["--bench", "reference|flash", "--baseline", "compare-head", "--noplot", "--quick"],
        ])
        self.assertEqual(len({entry["criterion_home"] for entry in plan["entries"]}), 4)
        self.assertEqual({entry["package"] for entry in plan["entries"]}, {"lattice-inference"})
        self.assertEqual(plan["environment"]["LATTICE_GPU_HANDOFF_BASE_SHA"], "a" * 40)

    def test_every_supported_target_checks_its_active_features(self):
        for target in self.admission.TARGETS:
            with self.subTest(target=target):
                features = "metal-gpu,f16,bench-internals" if target == "lm_head_bench" else "metal-gpu,f16"
                self.assertEqual(self.compare(target=target, features=features)["entries"][0]["target"], target)
                with self.assertRaisesRegex(self.admission.AdmissionError, "lacks features"):
                    self.compare(target=target, features="f16")

    def test_unknown_and_mixed_self_locking_selections_refuse(self):
        for target in ("future_metal", "decode_attn_bench topk_readback", "decode_attn_bench,topk_readback"):
            with self.subTest(target=target), self.assertRaises(self.admission.AdmissionError):
                self.compare(target=target)

    def test_feature_identity_includes_defaults_and_transitive_features(self):
        entry = self.compare(features="metal-bench")["entries"][0]
        self.assertEqual(entry["features"], "metal-bench")
        self.assertEqual(entry["feature_set"], ["default", "download", "f16", "metal-bench", "metal-gpu", "serve", "std"])
        with self.assertRaisesRegex(self.admission.AdmissionError, "unknown or unsupported"):
            self.compare(features="metal-gpu,f16,missing")

    def test_full_resolution_preserves_criterion_arguments_without_quick(self):
        plan = self.compare(args=["--full", "--fail-on-regression", "base", "head"])
        self.assertTrue(all("--quick" not in entry["argv"] for entry in plan["entries"]))
        self.assertTrue(all(entry["argv"][-1] == "--noplot" for entry in plan["entries"]))

    def test_command_grammar_refuses_opaque_or_mixed_cargo_selection(self):
        from unittest import mock
        admission = self.admission
        with mock.patch.object(admission.sys, "platform", "darwin"):
            for command in (["sh", "-c", "cargo bench"],
                            ["cargo", "bench", "--workspace"],
                            ["cargo", "bench", "--locked", "-p", "lattice-inference", "--bench", "topk_readback", "--bench", "mtp_decode"]):
                with self.subTest(command=command), self.assertRaises(admission.AdmissionError):
                    admission.plan_command(Path.cwd(), command, {})

    def test_command_plan_preserves_the_exact_criterion_arguments(self):
        from unittest import mock
        admission = self.admission
        repo = Path.cwd().resolve()
        def manifest(directory, revision, path):
            if path == "Cargo.toml":
                return {"workspace": {"package": {"version": "0.0.1"}}}
            return self.manifest()
        with mock.patch.object(admission.sys, "platform", "darwin"), \
             mock.patch.object(admission, "_revision", return_value="c" * 40), \
             mock.patch.object(admission, "_manifest", side_effect=manifest), \
             mock.patch.object(admission, "_git", return_value="source"), \
             mock.patch.object(admission, "_clean_revision"):
            plan = admission.plan_command(repo, ["cargo", "bench", "--locked", "-p", "lattice-inference",
                "--bench", "topk_readback", "--features", "metal-gpu", "--", "readback", "--quick"],
                {"CRITERION_HOME": "relative-evidence"})
        self.assertEqual(plan["entries"][0]["argv"], ["--bench", "readback", "--quick"])
        self.assertEqual(plan["entries"][0]["revision"], "c" * 40)
        self.assertEqual(plan["command"][-3:], ["--", "readback", "--quick"])
        self.assertEqual(plan["entries"][0]["run_cwd"], str(repo / "crates/inference"))
        self.assertEqual(plan["entries"][0]["criterion_home"],
                         str(repo / "crates/inference/relative-evidence"))

    def test_command_output_directory_preserves_cargo_target_override(self):
        admission = self.admission
        repo = Path("/tmp/admission-output").resolve()
        self.assertEqual(admission.command_criterion_home(repo, {"CARGO_TARGET_DIR": "cache"}),
                         repo / "crates/inference/cache/criterion")
        self.assertEqual(admission.command_criterion_home(repo, {"CARGO_TARGET_DIR": "/tmp/build-cache"}),
                         Path("/tmp/build-cache/criterion").resolve())

    def test_shipping_compatibility_policy_names_the_six_supported_targets(self):
        admission = self.admission
        manifest = admission.tomllib.loads((REPO / "crates/inference/Cargo.toml").read_text())
        policy = manifest["package"]["metadata"]["gpu-bench-handoff"]
        self.assertEqual(policy["version"], admission.PROTOCOL)
        self.assertEqual(sorted(policy["targets"]), sorted(admission.TARGETS))

    def command(self, argv, *, manifest=None, git=None):
        from unittest import mock
        admission = self.admission
        def load(directory, revision, path):
            if path == "Cargo.toml":
                return {"workspace": {"package": {"version": "0.0.1"}}}
            return self.manifest() if manifest is None else manifest
        with mock.patch.object(admission.sys, "platform", "darwin"), \
             mock.patch.object(admission, "_revision", return_value="c" * 40), \
             mock.patch.object(admission, "_manifest", side_effect=load), \
             mock.patch.object(admission, "_git", side_effect=git or (lambda *args: "source")), \
             mock.patch.object(admission, "_clean_revision"):
            return admission.plan_command(Path.cwd().resolve(), argv, {"CRITERION_HOME": "evidence"})

    def bin_command(self, target, features="metal-gpu,f16", *, args=None, locked=True, release=True, **kw):
        if args is None:
            args = ["--q4-dir", "model"] if target == "eval_perplexity" else ["--flag"]
        argv = ["cargo", "run", *(["--locked"] if locked else []), *(["--release"] if release else []),
                "-p", "lattice-inference", "--bin", target, "--features", features, "--", *args]
        return self.command(argv, **kw)

    def test_each_declared_binary_is_admitted_as_a_built_executable(self):
        repo = Path.cwd().resolve()
        for target in self.admission.BINS:
            with self.subTest(target=target):
                args = ["--q4-dir", "model", "--corpus-file", "c.txt"] if target == "eval_perplexity" else ["--flag", "1"]
                plan = self.bin_command(target, args=args)
                entry = plan["entries"][0]
                self.assertEqual((entry["kind"], entry["target"], entry["release"]), ("bin", target, True))
                self.assertEqual(entry["argv"], args)
                self.assertEqual(entry["run_cwd"], str(repo))
                self.assertIsNone(entry["criterion_home"])
                self.assertEqual(entry["source_path"], f"crates/inference/src/bin/{target}.rs")
                self.assertEqual(plan["command"][-(len(args) + 1):], ["--", *args])
                self.assertEqual(plan["command"][plan["command"].index("--kind") + 1], "bin")
                self.assertIn("--release", plan["command"])

    def test_binary_release_flag_is_optional_and_frozen(self):
        entry = self.bin_command("bench_decode_ab", release=False)["entries"][0]
        self.assertFalse(entry["release"])
        self.assertNotIn("--release", self.bin_command("bench_decode_ab", release=False)["command"])

    def test_undeclared_binaries_and_cross_kind_selections_refuse(self):
        admission = self.admission
        for target in ("chat_metal", "ppl_metal", "bench_gdn_prefill_ab"):
            with self.subTest(target=target), self.assertRaisesRegex(admission.AdmissionError, "unsupported self-locking"):
                self.bin_command(target)
        # A bench target is not a binary, and a binary is not a bench target.
        with self.assertRaisesRegex(admission.AdmissionError, "unsupported self-locking"):
            self.bin_command("topk_readback")
        with self.assertRaisesRegex(admission.AdmissionError, "unsupported self-locking"):
            self.command(["cargo", "bench", "--locked", "-p", "lattice-inference", "--bench", "bench_decode_ab",
                          "--features", "metal-gpu,f16"])
        with self.assertRaisesRegex(admission.AdmissionError, "unsupported self-locking"):
            self.compare(target="bench_decode_ab")

    def test_declared_binary_without_its_features_refuses(self):
        admission = self.admission
        for target, needed in admission.BINS.items():
            for missing in sorted(needed):
                features = ",".join(sorted(needed - {missing}))
                with self.subTest(target=target, missing=missing), \
                        self.assertRaisesRegex(admission.AdmissionError, r"lacks features \['%s'\]" % missing):
                    self.bin_command(target, features)

    def test_binary_required_features_include_its_manifest_declaration(self):
        admission = self.admission
        tables = [{"name": "bench_decode_slopefit", "required-features": ["bench-internals"]}]
        manifest = self.manifest(bin_tables=tables)
        with self.assertRaisesRegex(admission.AdmissionError, r"lacks features \['bench-internals'\]"):
            self.bin_command("bench_decode_slopefit", "metal-gpu", manifest=manifest)
        plan = self.bin_command("bench_decode_slopefit", "metal-gpu,bench-internals", manifest=manifest)
        self.assertEqual(plan["entries"][0]["feature_set"][0], "bench-internals")

    def test_binary_admission_requires_a_bins_declaration_in_the_revision(self):
        admission = self.admission
        names = sorted(admission.BINS)
        for label, manifest in (
            ("no policy", self.manifest(eligible=False)),
            ("no bins list", self.manifest(bins=[])),
            ("missing one", self.manifest(bins=names[1:])),
            ("extra one", self.manifest(bins=[*names, "chat_metal"])),
        ):
            with self.subTest(label=label), \
                    self.assertRaisesRegex(admission.AdmissionError, "no supported GPU handoff declaration for bins"):
                self.bin_command("bench_decode_ab", manifest=manifest)

    def test_bench_admission_is_unchanged_by_a_missing_bins_declaration(self):
        # A revision that predates the bins list still admits its declared bench targets.
        from unittest import mock
        admission = self.admission
        historical = self.manifest(bins=[])
        with mock.patch.object(admission, "_manifest", side_effect=lambda repo, rev, path: (
                {"workspace": {"package": {"version": "0.0.1"}}} if path == "Cargo.toml" else historical)), \
             mock.patch.object(admission, "_git", return_value="source"), \
             mock.patch.object(admission.sys, "platform", "darwin"):
            entry = admission._entry(Path("/tmp/x"), "a" * 40, "topk_readback", "metal-gpu", entry_id="command",
                                     cwd=Path("/tmp/x"), criterion_home=Path("/tmp/h"), target_args=[], platform="darwin")
        self.assertEqual(entry["kind"], "bench")

    def test_binary_build_must_be_locked_and_use_the_declared_grammar(self):
        admission = self.admission
        with self.assertRaisesRegex(admission.AdmissionError, "require cargo run --locked"):
            self.bin_command("bench_decode_ab", locked=False)
        base = ["cargo", "run", "--locked", "-p", "lattice-inference", "--bin", "bench_decode_ab",
                "--features", "metal-gpu,f16"]
        for label, argv in (
            ("other package", ["cargo", "run", "--locked", "-p", "lattice-embed", "--bin", "bench_decode_ab"]),
            ("no package", ["cargo", "run", "--locked", "--bin", "bench_decode_ab"]),
            ("no bin", ["cargo", "run", "--locked", "-p", "lattice-inference"]),
            ("two bins", [*base, "--bin", "bench_logit_dump"]),
            ("example", ["cargo", "run", "--locked", "-p", "lattice-inference", "--example", "metal_decode_bench"]),
            ("manifest path", [*base, "--manifest-path", "other/Cargo.toml"]),
            ("repeated release", [*base, "--release", "--release"]),
            ("repeated locked", [*base, "--locked"]),
            ("build, not run", ["cargo", "build", "--locked", "-p", "lattice-inference", "--bin", "bench_decode_ab"]),
            ("shell", ["sh", "-c", "cargo run --locked -p lattice-inference --bin bench_decode_ab"]),
            ("release on bench", ["cargo", "bench", "--locked", "--release", "-p", "lattice-inference", "--bench", "topk_readback",
                                  "--features", "metal-gpu"]),
        ):
            with self.subTest(label=label), self.assertRaises(admission.AdmissionError):
                self.command(argv)

    def test_binary_source_must_be_the_discoverable_default_path(self):
        admission = self.admission
        moved = [{"name": "bench_decode_ab", "path": "src/other.rs", "required-features": ["f16", "metal-gpu"]}]
        with self.assertRaisesRegex(admission.AdmissionError, "unsupported bin source"):
            self.bin_command("bench_decode_ab", manifest=self.manifest(bin_tables=moved))
        with self.assertRaisesRegex(admission.AdmissionError, "not one discoverable binary"):
            self.bin_command("eval_perplexity", manifest=self.manifest(autobins=False))
        twice = [{"name": "eval_perplexity"}, {"name": "eval_perplexity"}]
        with self.assertRaisesRegex(admission.AdmissionError, "not one discoverable binary"):
            self.bin_command("eval_perplexity", manifest=self.manifest(bin_tables=twice))
        def missing(*args):
            raise admission.AdmissionError("git show failed: path does not exist")
        with self.assertRaisesRegex(admission.AdmissionError, "path does not exist"):
            self.bin_command("eval_perplexity", git=missing)

    def test_eval_perplexity_admits_only_one_gpu_lock_acquisition(self):
        admission = self.admission
        for mode in admission.EVAL_PERPLEXITY_METAL_MODES:
            with self.subTest(mode=mode):
                plan = self.bin_command("eval_perplexity", args=[mode, "dir", "--corpus-file", "c.txt"])
                self.assertEqual(plan["entries"][0]["argv"][0], mode)
        for label, args in (
            ("dual Q4", ["--q4-dir", "a", "--quarot-q4-dir", "b", "--tokenizer-dir", "t"]),
            ("metal dir and Q4", ["--metal-model-dir", "m", "--q4-dir", "a"]),
            ("CPU mode takes no lock", ["--model-dir", "m", "--corpus-file", "c.txt"]),
            ("no arguments", []),
        ):
            with self.subTest(label=label), \
                    self.assertRaisesRegex(admission.AdmissionError, "one GPU lock acquisition per process"):
                self.bin_command("eval_perplexity", args=args)

    def test_shipping_binary_policy_matches_the_manifest_and_sources(self):
        admission = self.admission
        manifest = admission.tomllib.loads((REPO / "crates/inference/Cargo.toml").read_text())
        policy = manifest["package"]["metadata"]["gpu-bench-handoff"]
        self.assertEqual(sorted(policy["bins"]), sorted(admission.BINS))
        declared = {item["name"]: item for item in manifest.get("bin", [])}
        for name, features in admission.BINS.items():
            with self.subTest(binary=name):
                source = REPO / "crates/inference/src/bin" / f"{name}.rs"
                self.assertIn("gpu_test_lock()", source.read_text())
                self.assertIn('feature = "metal-gpu"', source.read_text())
                self.assertLessEqual(set(declared.get(name, {}).get("required-features", [])), features)
                self.assertIn("metal-gpu", features)
                self.assertNotEqual(manifest["package"].get("autobins"), False)

    def test_binary_artifact_must_be_the_admitted_binary(self):
        admission = self.admission
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory).resolve()
            executable = cwd / "build" / "release" / "eval_perplexity"
            executable.parent.mkdir(parents=True)
            executable.write_text("fixture")
            executable.chmod(0o755)
            entry = {"target": "eval_perplexity", "kind": "bin",
                     "source_path": "crates/inference/src/bin/eval_perplexity.rs",
                     "feature_set": ["default", "metal-gpu"], "package_version": "0.0.1",
                     "target_dir": str(cwd / "build")}
            artifact = {"reason": "compiler-artifact", "target": {
                "name": "eval_perplexity", "kind": ["bin"], "src_path": str(cwd / entry["source_path"])},
                "features": entry["feature_set"], "executable": str(executable),
                "package_id": f"path+{(cwd / 'crates/inference').as_uri()}#lattice-inference@0.0.1"}
            self.assertEqual(admission.validate_artifact(entry, artifact, cwd), executable)
            changed = {**artifact, "target": {**artifact["target"], "kind": ["bench"]}}
            with self.assertRaisesRegex(admission.AdmissionError, "does not match admitted target"):
                admission.validate_artifact(entry, changed, cwd)

    def test_cargo_environment_has_no_handoff_capabilities(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {
            "LATTICE_GPU_HANDOFF_BROKER": "/tmp/broker",
            "LATTICE_GPU_HANDOFF_BROKER_TOKEN": "token",
            "LATTICE_GPU_HANDOFF_PLAN": "/tmp/plan",
            "LATTICE_BENCH_LOCK_FDS": "3,4",
            "LATTICE_BENCH_SUPERVISOR_FD": "5",
            "LATTICE_MODEL_DIR": "/tmp/model",
        }):
            environment = self.admission._cargo_environment()
        self.assertFalse(any(key.startswith("LATTICE_GPU_HANDOFF_") for key in environment))
        self.assertNotIn("LATTICE_BENCH_LOCK_FDS", environment)
        self.assertNotIn("LATTICE_BENCH_SUPERVISOR_FD", environment)
        self.assertEqual(environment["LATTICE_MODEL_DIR"], "/tmp/model")

    def test_artifact_validation_rejects_changed_source_features_and_output_directory(self):
        import copy
        admission = self.admission
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory).resolve()
            executable = cwd / "build" / "selected"
            executable.parent.mkdir()
            executable.write_text("fixture")
            executable.chmod(0o755)
            entry = {"target": "topk_readback", "kind": "bench",
                     "source_path": "crates/inference/benches/topk_readback.rs",
                     "feature_set": ["default", "metal-gpu"], "package_version": "0.0.1",
                     "target_dir": str(executable.parent)}
            artifact = {"reason": "compiler-artifact", "target": {
                "name": "topk_readback", "kind": ["bench"], "src_path": str(cwd / entry["source_path"])},
                "features": entry["feature_set"], "executable": str(executable),
                "package_id": f"path+{(cwd / 'crates/inference').as_uri()}#lattice-inference@0.0.1"}
            self.assertEqual(admission.validate_artifact(entry, artifact, cwd), executable)
            for field, value in (("features", ["default"]), ("package_id", "other"),
                                 ("executable", "/bin/sh")):
                changed = {**artifact, field: value}
                with self.subTest(field=field), self.assertRaises(admission.AdmissionError):
                    admission.validate_artifact(entry, changed, cwd)
            changed = copy.deepcopy(artifact)
            changed["target"]["src_path"] = str(cwd / "other.rs")
            with self.assertRaises(admission.AdmissionError):
                admission.validate_artifact(entry, changed, cwd)

    def test_request_cannot_change_frozen_arguments(self):
        admission = self.admission
        entry = self.compare()["entries"][0]
        request = {"entry": entry["id"], **{name: entry[name] for name in (
            "cwd", "run_cwd", "revision", "target", "kind", "release", "features", "criterion_home", "argv")}}
        for name in request:
            changed = {**request, name: [] if name == "argv" else "changed"}
            with self.subTest(field=name), self.assertRaisesRegex(admission.AdmissionError, "changed admitted"):
                admission.validate_measurement_request(entry, changed)


class SystemBashEntrypoints(unittest.TestCase):
    def run_wrapper(self, name, args):
        import json

        with tempfile.TemporaryDirectory(prefix="bench-system-bash-") as temporary:
            root = Path(temporary)
            lib = root / "scripts/lib"
            lib.mkdir(parents=True)
            shutil.copy2(REPO / "scripts" / name, root / "scripts" / name)
            shutil.copy2(LIB / "bench-python.sh", lib / "bench-python.sh")
            # Capture the wrapper boundary without entering a measurement window.
            (lib / "bench_supervision.py").write_text(
                "import json,sys\nprint(json.dumps(sys.argv[1:]))\n"
            )
            bindir = root / "bin"
            bindir.mkdir()
            (bindir / "python3.13").symlink_to(sys.executable)
            result = subprocess.run(
                ["/bin/bash", str(root / "scripts" / name), *args],
                env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"},
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            forwarded = json.loads(result.stdout)
            return forwarded, str(root / "scripts/lib/bench-compare-impl.sh")

    def test_compare_without_arguments_reaches_supervisor(self):
        forwarded, implementation = self.run_wrapper("bench-compare.sh", [])
        self.assertEqual(forwarded, ["run", "--label", "bench-compare", "--entrypoint", "--", implementation])

    def test_compare_handoff_without_refs_reaches_supervisor(self):
        forwarded, implementation = self.run_wrapper("bench-compare.sh", ["--gpu-handoff"])
        self.assertEqual(forwarded, ["run", "--label", "bench-compare", "--entrypoint",
                                     "--gpu-handoff", "compare", "--", implementation])

    def test_compare_handoff_preserves_argument_boundaries(self):
        args = ["--full", "--", "base ref", ""]
        forwarded, implementation = self.run_wrapper("bench-compare.sh", ["--gpu-handoff", *args])
        self.assertEqual(forwarded, ["run", "--label", "bench-compare", "--entrypoint",
                                     "--gpu-handoff", "compare", "--", implementation, *args])

    def test_ordinary_command_reaches_supervisor_with_exact_arguments(self):
        # --entrypoint must always be forwarded: it is what lets a wrapped
        # command that is itself a self-supervising Python entry point (one
        # that calls ensure_python_entrypoint) receive the liveness pipe it
        # looks for instead of refusing with "LATTICE_BENCH_SUPERVISOR_FD is
        # not set". An ordinary command never reads that pipe, so the flag
        # is unconditional here, matching bench-compare.sh and
        # bench_supervise_entry, which already pass it unconditionally too.
        command = ["printf", "%s", "two words", ""]
        for durable in (False, True):
            with self.subTest(durable=durable):
                flags = ["--durable"] if durable else []
                forwarded, _ = self.run_wrapper("bench-command.sh", ["--label", "fixture label", *flags, "--", *command])
                quiet = ["--quiet"] if durable else []
                self.assertEqual(forwarded, ["run", "--label", "fixture label", *quiet, "--entrypoint", "--", *command])

    def test_command_handoff_reaches_supervisor_with_exact_arguments(self):
        command = ["cargo", "bench", "--", "two words", ""]
        for durable in (False, True):
            with self.subTest(durable=durable):
                flags = ["--durable"] if durable else []
                forwarded, _ = self.run_wrapper("bench-command.sh", ["--gpu-handoff", "--label", "fixture label", *flags, "--", *command])
                quiet = ["--quiet"] if durable else []
                self.assertEqual(forwarded, ["run", "--label", "fixture label", *quiet, "--entrypoint",
                                             "--gpu-handoff", "command", "--", *command])


# The stand-in for `uv run ...`: it records its stdin, its arguments and whether both machine
# locks are held by someone else while it runs, then prints the rows/diagnostics the test asks for.
UV_STUB = r'''
import fcntl,json,os,pathlib,sys
def held(path):
    with open(path, "r+") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False
events = os.environ.get("FIXTURE_EVENTS")
if events:
    open(events, "a").write("mlx-start\n")
pathlib.Path(os.environ["FIXTURE_UV_STDIN"]).write_bytes(sys.stdin.buffer.read())
pathlib.Path(os.environ["FIXTURE_UV_ARGS"]).write_text(" ".join(sys.argv[1:]) + "\n")
pathlib.Path(os.environ["FIXTURE_UV_RECORD"]).write_text(json.dumps(dict(
    gpu_lock_held=held(os.environ["FIXTURE_GPU"]), window_held=held(os.environ["FIXTURE_WINDOW"]),
    supervisor_marker="LATTICE_GPU_LOCK_SUPERVISOR_PID" in os.environ,
    handoff_env=sorted(name for name in os.environ if name.startswith("LATTICE_GPU_HANDOFF_")),
    lock_fds="LATTICE_BENCH_LOCK_FDS" in os.environ)))
sys.stdout.write(os.environ.get("FIXTURE_UV_OUT", "").replace("\\t", "\t").replace("\\n", "\n"))
sys.stderr.write(os.environ.get("FIXTURE_UV_ERR", ""))
if events:
    open(events, "a").write("mlx-end\n")
sys.exit(int(os.environ["FIXTURE_UV_RC"]))
'''


class GpuHandoffShippingCommand(unittest.TestCase):
    BINS = ["bench_decode_ab", "bench_decode_slopefit", "bench_logit_dump", "eval_perplexity"]

    def run_fixture(self, *, eligibility="valid", artifact_valid=True, kind="bench", bin_args=("--q4-dir", "model"),
                    bin_name="eval_perplexity", second_acquisition=False, launch=None, commit_files=None,
                    extra_env=None, gitignore=None, stdout_text=None, stderr_text=None, logits=False,
                    features=("default", "metal-gpu", "std"), uv_rc=0, read_files=()):
        """`launch(root, temp)` returns (argv, cwd) for a shipping script that runs the
        fixture binary through the real bench-command.sh instead of calling it directly;
        each launch of the fixture binary is then recorded, not asserted against bin_args."""
        import json
        with tempfile.TemporaryDirectory(prefix="admission-command-") as temporary:
            temp = Path(temporary).resolve()
            root = temp / "repo"
            lib = root / "scripts/lib"
            lib.mkdir(parents=True)
            for name in ("bench_admission.py", "bench_handoff.py", "bench_supervision.py",
                         "bench-locks.py", "bench-python.sh", "quiet-probe.py"):
                shutil.copy2(LIB / name, lib / name)
            shutil.copy2(REPO / "scripts/bench-command.sh", root / "scripts/bench-command.sh")
            # The fixture substitutes the host/Metal workload, never admission or RPC validation.
            source = (lib / "bench_admission.py").read_text()
            self.assertIn('if sys.platform != "darwin":', source)
            (lib / "bench_admission.py").write_text(source.replace('if sys.platform != "darwin":', 'if False:', 1))
            lock_names = {"BENCH_WINDOW": temp / "window.lock", "GPU_LOCK": temp / "gpu.lock", "PENDING_DIR": temp / "pending"}
            source = (lib / "bench-locks.py").read_text()
            for name, path in lock_names.items():
                source, count = re.subn(rf'^{name} = "[^"]*"$', f'{name} = "{path}"', source, flags=re.M)
                self.assertEqual(count, 1)
            (lib / "bench-locks.py").write_text(source)
            (lib / "quiet-probe.py").write_text(
                "import os,pathlib,sys\n"
                "assert not pathlib.Path(os.environ['FIXTURE_WORK']).exists()\n"
                "pathlib.Path(os.environ['FIXTURE_QUIET']).write_text('inside guard')\n"
                "open(os.environ['FIXTURE_QUIET'] + '.log', 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
                "if os.environ.get('FIXTURE_QUIET_FAIL') and os.environ['FIXTURE_QUIET_FAIL'] in ' '.join(sys.argv[1:]):\n"
                "    print('fixture: machine not quiet'); sys.exit(1)\n"
                "print('fixture inside-guard CPU idle sample')\n")
            (root / ".gitignore").write_text(".cache/\ntarget/\n__pycache__/\n" if gitignore is None else gitignore)
            (root / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/inference"]\n[workspace.package]\nversion = "0.0.1"\n')
            (root / "Cargo.lock").write_text("version = 4\n")
            package = root / "crates/inference"
            (package / "benches").mkdir(parents=True)
            targets = ["cross_turn_prefix_cache_bench", "decode_attn_bench", "lm_head_bench",
                       "metal_decode_bench", "mtp_decode", "topk_readback"]
            policy = "" if eligibility == "absent" else (
                '[package.metadata.gpu-bench-handoff]\nversion = ' + ('1' if eligibility == "valid" else '2')
                + '\ntargets = ' + json.dumps(targets) + '\nbins = ' + json.dumps(self.BINS) + '\n')
            manifest = '[package]\nname = "lattice-inference"\nversion.workspace = true\n' + policy
            manifest += '[features]\ndefault = ["std"]\nstd = []\nmetal-gpu = []\nf16 = []\nbench-internals = []\n'
            for target in targets:
                manifest += f'[[bench]]\nname = "{target}"\nharness = false\n'
                (package / f"benches/{target}.rs").write_text("fn main() {}\n")
            (package / "Cargo.toml").write_text(manifest)
            (package / "src/bin").mkdir(parents=True)
            for name in self.BINS:
                (package / f"src/bin/{name}.rs").write_text("fn main() {}\n")
            (package / "fixture-model.txt").write_text("package-relative model")
            (root / "fixture-model.txt").write_text("repo-relative model")
            for relative, content in (commit_files or {}).items():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, Path):
                    shutil.copy2(content, root / relative)
                else:
                    (root / relative).write_text(content)
            env_git = {**os.environ, "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
                       "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
            subprocess.run([*GIT, "init", "-q", str(root)], check=True, env=env_git)
            subprocess.run([*GIT, "-C", str(root), "add", "."], check=True, env=env_git)
            subprocess.run([*GIT, "-C", str(root), "commit", "-qm", "fixture"], check=True, env=env_git)
            executable_source = temp / "target-fixture.py"
            executable_source.write_text(f"#!{sys.executable}\n" + r'''
import fcntl,json,os,pathlib,socket,sys
binary = os.environ["FIXTURE_KIND"] == "bin"
repo = pathlib.Path(os.environ["FIXTURE_REPO"])
package = repo / "crates/inference"
# cargo bench runs from the package directory; cargo run keeps the caller's cwd.
expected_cwd = repo if binary else package
assert pathlib.Path.cwd() == expected_cwd, (pathlib.Path.cwd(), expected_cwd)
assert pathlib.Path(os.environ["LATTICE_MODEL_DIR"]).read_text() == ("repo-relative model" if binary else "package-relative model")
assert os.environ.get("FIXTURE_LAUNCHES") or sys.argv[1:] == (os.environ["FIXTURE_ARGS"].split() if binary else ["--bench", "lookup", "--quick"]), sys.argv
assert ("CRITERION_HOME" in os.environ) != binary
assert "LATTICE_GPU_HANDOFF_BROKER_TOKEN" not in os.environ
assert "LATTICE_BENCH_LOCK_FDS" not in os.environ
gpu = pathlib.Path(os.environ["FIXTURE_GPU"])
held, canonical = os.fstat(0), gpu.stat()
assert (held.st_dev, held.st_ino) == (canonical.st_dev, canonical.st_ino)
with gpu.open("r+") as probe:
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        raise AssertionError("GPU lock was not held")
fcntl.flock(0, fcntl.LOCK_EX | fcntl.LOCK_NB)
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    connection.connect(os.environ["LATTICE_GPU_HANDOFF_CONTROL"])
    token = os.environ["LATTICE_GPU_HANDOFF_TOKEN"]
    connection.sendall(json.dumps(dict(protocol=1, token=token, pid=os.getpid(), exe=str(pathlib.Path(__file__).resolve()))).encode() + b"\n")
    with connection.makefile("rb") as stream:
        response = json.loads(stream.readline())
    assert response == dict(protocol=1, token=token, status="ready"), response
home = None if binary else pathlib.Path(os.environ["CRITERION_HOME"])
assert home is None or home == package / "relative-evidence", home
assert pathlib.Path(os.environ["FIXTURE_QUIET"]).exists()
record = dict(cwd=str(pathlib.Path.cwd()), criterion_home=None if home is None else str(home))
if os.environ["FIXTURE_SECOND"] == "1":
    # A second gpu_test_lock() in one process repeats the handshake on the same control path.
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as again:
            again.connect(os.environ["LATTICE_GPU_HANDOFF_CONTROL"])
        record["second"] = "connected"
    except OSError as error:
        record["second"] = type(error).__name__
if os.environ.get("FIXTURE_EVENTS"):
    open(os.environ["FIXTURE_EVENTS"], "a").write("lattice-start\n")
if os.environ.get("FIXTURE_LAUNCHES"):
    with open(os.environ["FIXTURE_LAUNCHES"], "a") as launches:
        launches.write(json.dumps(dict(argv=sys.argv[1:], cwd=str(pathlib.Path.cwd()),
            model=os.environ.get("LATTICE_MODEL_DIR"), tokenizer=os.environ.get("LATTICE_TOKENIZER_DIR"),
            out=os.environ.get("LATTICE_LOGIT_OUT"))) + "\n")
else:
    pathlib.Path(os.environ["FIXTURE_WORK"]).write_text(json.dumps(record))
if os.environ.get("FIXTURE_LOGITS") == "1":
    assert os.path.isabs(os.environ["LATTICE_LOGIT_OUT"]), os.environ["LATTICE_LOGIT_OUT"]
    pathlib.Path(os.environ["LATTICE_LOGIT_OUT"]).write_bytes(bytes(32))
if os.environ.get("FIXTURE_STDERR"):
    print(os.environ["FIXTURE_STDERR"], file=sys.stderr, flush=True)
print(os.environ.get("FIXTURE_STDOUT") or ("RESULT fixture binary" if binary else "lookup time: [1.0 ns 1.1 ns 1.2 ns]"), flush=True)
if os.environ.get("FIXTURE_EVENTS"):
    open(os.environ["FIXTURE_EVENTS"], "a").write("lattice-end\n")
''')
            bindir = temp / "bin"
            bindir.mkdir()
            for version in ("python3.13", "python3.12", "python3.11", "python3"):
                (bindir / version).symlink_to(sys.executable)
            cargo = bindir / "cargo"
            cargo.write_text(f"#!{sys.executable}\n" + r'''
import json,os,pathlib,shutil,sys
binary = os.environ["FIXTURE_KIND"] == "bin"
assert sys.argv[1] == ("build" if binary else "bench"), sys.argv
assert "--locked" in sys.argv and "--message-format=json" in sys.argv
assert ("--no-run" in sys.argv) != binary and ("--bin" in sys.argv) == binary and ("--release" in sys.argv) == binary
assert not any(name.startswith("LATTICE_GPU_HANDOFF_") for name in os.environ)
assert "LATTICE_BENCH_LOCK_FDS" not in os.environ
assert "LATTICE_BENCH_SUPERVISOR_FD" not in os.environ
gpu = pathlib.Path(os.environ["FIXTURE_GPU"]).stat()
for fd in range(256):
    try:
        candidate = os.fstat(fd)
    except OSError:
        continue
    assert (candidate.st_dev, candidate.st_ino) != (gpu.st_dev, gpu.st_ino), fd
with open(os.environ["FIXTURE_CARGO"], "a") as built:
    built.write(json.dumps(sys.argv[1:]) + "\n")
root = pathlib.Path.cwd()
target = sys.argv[sys.argv.index("--bin" if binary else "--bench") + 1]
target_dir = pathlib.Path(sys.argv[sys.argv.index("--target-dir") + 1])
exe = target_dir / ("release" if binary else "release/deps") / target
exe.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(os.environ["FIXTURE_SOURCE"], exe)
exe.chmod(0o755)
source = root / "crates/inference" / ("src/bin" if binary else "benches") / (target + ".rs")
if os.environ["FIXTURE_ARTIFACT_VALID"] != "1":
    source = root / "different.rs"
print(json.dumps(dict(reason="compiler-artifact", package_id="path+" + (root / "crates/inference").as_uri() + "#lattice-inference@0.0.1",
    target=dict(name=target,kind=["bin" if binary else "bench"],src_path=str(source)), features=json.loads(os.environ["FIXTURE_FEATURES"]), executable=str(exe))))
''')
            cargo.chmod(0o755)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("LATTICE_GPU_HANDOFF_") and key not in (
                       "LATTICE_BENCH_LOCK_STATUS", "LATTICE_BENCH_LOCK_FDS", "LATTICE_BENCH_SUPERVISOR_FD",
                       "CARGO_BUILD_TARGET", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS")}
            env.update({"PATH": f"{bindir}:{env['PATH']}", "PYTHONDONTWRITEBYTECODE": "1",
                        "LATTICE_MODEL_DIR": "fixture-model.txt", "FIXTURE_KIND": kind, "FIXTURE_ARGS": " ".join(bin_args), "FIXTURE_SECOND": str(int(second_acquisition)),
                        "FIXTURE_REPO": str(root), "FIXTURE_GPU": str(lock_names["GPU_LOCK"]),
                        "FIXTURE_WORK": str(temp / "worked.json"), "FIXTURE_QUIET": str(temp / "quiet"),
                        "FIXTURE_CARGO": str(temp / "cargo-ran"), "FIXTURE_SOURCE": str(executable_source),
                        "FIXTURE_ARTIFACT_VALID": str(int(artifact_valid)), "FIXTURE_FEATURES": json.dumps(sorted(features))})
            if launch is not None:
                env.update({"FIXTURE_LAUNCHES": str(temp / "launches.jsonl"), "FIXTURE_LOGITS": str(int(logits)),
                            "FIXTURE_UV_STDIN": str(temp / "uv-stdin"), "FIXTURE_UV_ARGS": str(temp / "uv-args"),
                            "FIXTURE_UV_RC": str(uv_rc), "FIXTURE_EVENTS": str(temp / "events.log"),
                            "FIXTURE_UV_RECORD": str(temp / "uv-record.json"),
                            "FIXTURE_WINDOW": str(lock_names["BENCH_WINDOW"])})
                if stdout_text is not None:
                    env["FIXTURE_STDOUT"] = stdout_text
                if stderr_text is not None:
                    env["FIXTURE_STDERR"] = stderr_text
                uv = bindir / "uv"
                uv.write_text(f"#!{sys.executable}\n" + UV_STUB)
                uv.chmod(0o755)
            env.update({key: value.replace("{TEMP}", str(temp)) for key, value in (extra_env or {}).items()})
            if kind == "bench":
                env["CRITERION_HOME"] = "relative-evidence"
                command = ["cargo", "bench", "--locked", "-p", "lattice-inference", "--bench", "topk_readback",
                           "--features", "metal-gpu", "--", "lookup", "--quick"]
            else:
                command = ["cargo", "run", "--locked", "--release", "-p", "lattice-inference", "--bin",
                           bin_name, "--features", "metal-gpu", "--", *bin_args]
            argv, cwd = (["/bin/bash", str(root / "scripts/bench-command.sh"), "--gpu-handoff", "--label", "fixture", "--",
                *command], root) if launch is None else launch(root, temp)
            result = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=60)
            launches_file, quiet_log = temp / "launches.jsonl", temp / "quiet.log"
            return result, {
                "root": root, "temp": temp,
                "launches": [json.loads(line) for line in launches_file.read_text().splitlines()] if launches_file.exists() else [],
                "quiet_log": quiet_log.read_text().splitlines() if quiet_log.exists() else [],
                "cargo_calls": [json.loads(line) for line in (temp / "cargo-ran").read_text().splitlines()] if (temp / "cargo-ran").exists() else [],
                "uv_stdin": (temp / "uv-stdin").read_text() if (temp / "uv-stdin").exists() else None,
                "uv_args": (temp / "uv-args").read_text().strip() if (temp / "uv-args").exists() else None,
                "uv_record": json.loads((temp / "uv-record.json").read_text()) if (temp / "uv-record.json").exists() else None,
                "events": (temp / "events.log").read_text().splitlines() if (temp / "events.log").exists() else [],
                "mlx_loads": [json.loads(line) for line in (temp / "events.log.loads").read_text().splitlines()]
                if (temp / "events.log.loads").exists() else [],
                "files": {name: (root / name).read_text() if (root / name).exists() else None for name in read_files},
                "cargo": (temp / "cargo-ran").exists(), "worked": (temp / "worked.json").exists(),
                "quiet": (temp / "quiet").exists(),
                "locks": [lock_names[name].exists() for name in ("BENCH_WINDOW", "GPU_LOCK")],
                "work": json.loads((temp / "worked.json").read_text()) if (temp / "worked.json").exists() else None,
                "run_cwd": str(root if kind == "bin" else package),
                "criterion_home": None if kind == "bin" else str(package / "relative-evidence"),
            }

    def test_shipping_handoff_preserves_package_cwd_and_relative_paths(self):
        result, evidence = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(evidence["cargo"])
        self.assertTrue(evidence["quiet"])
        self.assertEqual(evidence["work"], {"cwd": evidence["run_cwd"], "criterion_home": evidence["criterion_home"]})
        self.assertIn("lookup time:", result.stdout)

    def test_historical_or_invalid_eligibility_refuses_before_locks_or_cargo(self):
        for eligibility in ("absent", "invalid"):
            with self.subTest(eligibility=eligibility):
                result, evidence = self.run_fixture(eligibility=eligibility)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("no supported GPU handoff declaration", result.stderr)
                self.assertFalse(evidence["cargo"])
                self.assertFalse(evidence["worked"])
                self.assertFalse(evidence["quiet"])
                self.assertEqual(evidence["locks"], [False, False])

    def test_wrong_cargo_artifact_refuses_before_target_or_quiet(self):
        result, evidence = self.run_fixture(artifact_valid=False)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Cargo artifact does not match", result.stderr)
        self.assertTrue(evidence["cargo"])
        self.assertFalse(evidence["worked"])
        self.assertFalse(evidence["quiet"])

    def test_declared_binary_runs_under_the_handoff_from_the_invocation_directory(self):
        result, evidence = self.run_fixture(kind="bin")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(evidence["cargo"])
        self.assertTrue(evidence["quiet"])
        self.assertEqual(evidence["work"], {"cwd": evidence["run_cwd"], "criterion_home": None})
        self.assertIn("RESULT fixture binary", result.stdout)

    def test_binary_from_a_revision_without_a_bins_declaration_refuses_before_locks_or_cargo(self):
        for eligibility in ("absent", "invalid"):
            with self.subTest(eligibility=eligibility):
                result, evidence = self.run_fixture(kind="bin", eligibility=eligibility)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("no supported GPU handoff declaration for bins", result.stderr)
                self.assertFalse(evidence["cargo"])
                self.assertFalse(evidence["worked"])
                self.assertFalse(evidence["quiet"])
                self.assertEqual(evidence["locks"], [False, False])

    def test_second_lock_acquisition_in_one_process_finds_the_control_channel_closed(self):
        # The premise of the eval_perplexity restriction: the supervisor closes the
        # control channel after the first acknowledgement.
        result, evidence = self.run_fixture(kind="bin", bin_name="bench_decode_slopefit", bin_args=(), second_acquisition=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(evidence["work"]["second"], ("ConnectionRefusedError", "FileNotFoundError"))

    def test_dual_q4_binary_invocation_refuses_before_locks_or_cargo(self):
        result, evidence = self.run_fixture(kind="bin", bin_args=("--q4-dir", "a", "--quarot-q4-dir", "b"))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("one GPU lock acquisition per process", result.stderr)
        self.assertFalse(evidence["cargo"])
        self.assertFalse(evidence["worked"])
        self.assertEqual(evidence["locks"], [False, False])

    def test_wrong_binary_artifact_refuses_before_target_or_quiet(self):
        result, evidence = self.run_fixture(kind="bin", artifact_valid=False)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Cargo artifact does not match", result.stderr)
        self.assertTrue(evidence["cargo"])
        self.assertFalse(evidence["worked"])
        self.assertFalse(evidence["quiet"])

    SLOPEFIT_STDOUT = "\n".join([
        "SLOPEFIT_META kv_cache_len=1024 warmup=8 measure=32 repeats=1",
        "SLOPEFIT_META ctx=64 actual_prompt_tokens=70",
        "SLOPEFIT ctx=64 tokens=32 warmup_ms=0.0 measure_ms=96.000 rep=0",
    ])
    SLOPEFIT_STDERR = "\n".join([
        "[slopefit] loading /models/qwen (Q4)",
        "[slopefit] grid=[64] warmup=8 measure=32 repeats=1",
        "[slopefit] kv_cache_len=1024 (deepest_prompt=70 for max_ctx=64 + decode_horizon 32)",
        "[slopefit] ctx=64 actual_prompt_tokens=70",
    ])
    SLOPEFIT_FILES = ("scripts/bench_decode_slopefit.py", "scripts/lib/ensure-noindex-marker.sh")

    def run_script(self, relative, args=(), *, files=(), cwd=None, extra_files=None, **options):
        """Run a shipping script, copied into the fixture repository, from `cwd` (under the repo root)."""
        committed = {name: REPO / name for name in (relative, *files)}
        committed.update(extra_files or {})

        def launch(root, temp):
            workdir = root / cwd if cwd else root
            workdir.mkdir(parents=True, exist_ok=True)
            return ["/bin/bash", str(root / relative), *args], workdir

        return self.run_fixture(kind="bin", launch=launch, commit_files=committed, **options)

    def run_slopefit(self, args=(), **options):
        options.setdefault("stdout_text", self.SLOPEFIT_STDOUT)
        options.setdefault("stderr_text", self.SLOPEFIT_STDERR)
        return self.run_script(
            "scripts/bench_decode_slopefit.sh", args, files=self.SLOPEFIT_FILES,
            bin_name="bench_decode_slopefit", bin_args=(), features=("default", "f16", "metal-gpu", "std"), **options)

    def test_slopefit_script_runs_its_binary_through_the_handoff_and_feeds_the_parser(self):
        result, evidence = self.run_slopefit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        root = evidence["root"]
        # One handoff launch from the repository root, building the release binary with its features.
        self.assertEqual([(item["argv"], item["cwd"]) for item in evidence["launches"]], [([], str(root))])
        self.assertEqual(len(evidence["cargo_calls"]), 1)
        call = evidence["cargo_calls"][0]
        self.assertEqual(call[:1], ["build"])
        self.assertIn("--release", call)
        self.assertEqual(call[call.index("--bin") + 1], "bench_decode_slopefit")
        self.assertEqual(call[call.index("--features") + 1], "f16,metal-gpu")
        # The script was durable before, so the before/after ambient-idle checks survive,
        # and the measured-phase certification runs inside the handoff.
        self.assertEqual(evidence["quiet_log"], [
            "--label decode-slopefit: before",
            "--label command:bench_decode_slopefit: measured guard",
            "--label decode-slopefit: after",
        ])
        self.assertEqual(evidence["uv_args"].split()[:3], ["run", "--project", str(root)])
        # The post-processor reads the binary's stdout and, merged in by the handoff, its stderr and
        # the supervisor's own notices; the parser takes the same records out of either stream.
        stream = evidence["uv_stdin"]
        for line in (*self.SLOPEFIT_STDOUT.splitlines(), *self.SLOPEFIT_STDERR.splitlines(),
                     "fixture inside-guard CPU idle sample"):
            self.assertIn(line + "\n", stream)
        parser = _load_script_with_numpy_stub("slopefit_for_stream_test", "bench_decode_slopefit.py")
        clean = parser.parse_stream(self.SLOPEFIT_STDOUT.splitlines())
        self.assertEqual(parser.parse_stream(stream.splitlines()), clean)
        self.assertEqual(clean[3], {"kv_cache_len": 1024, "warmup": 8, "measure": 32, "repeats": 1})
        self.assertEqual(dict(clean[0]), {64: [3.0]})

    def test_slopefit_script_resolves_caller_relative_paths_before_the_handoff(self):
        # The handoff runs from the repository root; a path the caller wrote against its own
        # directory must still mean the same file.
        result, evidence = self.run_slopefit(
            ["--out", "out/slope.json"], cwd="sub", extra_env={"LATTICE_MODEL_DIR": "../fixture-model.txt"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        root = evidence["root"]
        self.assertEqual(evidence["launches"][0]["cwd"], str(root))
        self.assertEqual(evidence["launches"][0]["model"], f"{root}/sub/../fixture-model.txt")
        self.assertTrue(evidence["uv_args"].endswith(f"--out {root}/sub/out/slope.json"), evidence["uv_args"])

    def test_slopefit_script_refuses_before_locks_or_build_when_the_handoff_refuses(self):
        result, evidence = self.run_slopefit(eligibility="absent", uv_rc=1)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("no supported GPU handoff declaration for bins", result.stderr)
        self.assertEqual((evidence["launches"], evidence["cargo_calls"], evidence["locks"]), ([], [], [False, False]))

    def test_quality_script_runs_each_lattice_tier_as_its_own_handoff_launch(self):
        with tempfile.TemporaryDirectory(prefix="quality-models-") as models:
            dirs = {name: Path(models).resolve() / name for name in ("q4", "quarot", "tokenizer")}
            for directory in dirs.values():
                directory.mkdir()
            result, evidence = self.run_script(
                "scripts/bench_quality.sh", bin_name="eval_perplexity", bin_args=(), cwd="sub",
                # The real ignore file: the staged result is untracked scratch inside the worktree, and
                # the handoff admits only a commit-clean one.
                gitignore=(REPO / ".gitignore").read_text(),
                extra_files={"docs/bench_results/wiki.test.raw": "corpus\n", "docs/bench_results/perplexity.tsv": "canonical\n"},
                extra_env={"Q4_DIR": str(dirs["q4"]), "QUAROT_DIR": str(dirs["quarot"]), "TOK_DIR": str(dirs["tokenizer"]),
                           "SKIP_MLX": "1", "BENCH_MACHINE": "fixture"},
                stdout_text="PPL:                16.589166", read_files=("docs/bench_results/perplexity.tsv",))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            root = evidence["root"]
            self.assertEqual([item["cwd"] for item in evidence["launches"]], [str(root)] * 2)
            first, second = (item["argv"] for item in evidence["launches"])
            # Each tier takes the GPU lock once: exactly one Metal mode per process.
            modes = ("--metal-model-dir", "--q4-dir", "--quarot-q4-dir")
            self.assertEqual([[arg for arg in argv if arg in modes] for argv in (first, second)],
                             [["--q4-dir"], ["--quarot-q4-dir"]])
            self.assertEqual(first[first.index("--q4-dir") + 1], str(dirs["q4"]))
            self.assertEqual(second[second.index("--quarot-q4-dir") + 1], str(dirs["quarot"]))
            self.assertEqual(len(evidence["cargo_calls"]), 2)
            self.assertEqual(sorted(evidence["quiet_log"]), sorted(
                [f"--label {label}" for label in ["quality-perplexity: before", "quality-perplexity: after"] * 2
                 + ["command:eval_perplexity: measured guard"] * 2]))
            published = evidence["files"]["docs/bench_results/perplexity.tsv"]
            self.assertIn("lattice\tq4\t16.589166\t2048\n", published)
            self.assertIn("lattice\tq4-quarot\t16.589166\t2048\n", published)
            # SKIP_MLX=1 skips the cross-check: no MLX launch of any kind.
            self.assertNotIn("mlx-start", evidence["events"])

    def test_quality_script_resolves_caller_relative_model_directories(self):
        # Written against the caller's directory, one level below the repository root, and
        # naming directories next to it; the handoff runs from the root.
        def launch(root, temp):
            for name in ("q4", "quarot", "tokenizer"):
                (temp / "models" / name).mkdir(parents=True)
            workdir = root / "sub"
            workdir.mkdir()
            return ["/bin/bash", str(root / "scripts/bench_quality.sh")], workdir

        result, evidence = self.run_fixture(
            kind="bin", launch=launch, bin_name="eval_perplexity", bin_args=(),
            gitignore=(REPO / ".gitignore").read_text(),
            commit_files={"scripts/bench_quality.sh": REPO / "scripts/bench_quality.sh",
                          "docs/bench_results/wiki.test.raw": "corpus\n", "docs/bench_results/perplexity.tsv": "canonical\n"},
            extra_env={"Q4_DIR": "../../models/q4", "QUAROT_DIR": "../../models/quarot", "TOK_DIR": "../../models/tokenizer",
                       "SKIP_MLX": "1"},
            stdout_text="PPL:                16.589166")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        root = evidence["root"]
        first, second = (item["argv"] for item in evidence["launches"])
        self.assertEqual(first[first.index("--q4-dir") + 1], f"{root}/sub/../../models/q4")
        self.assertEqual(second[second.index("--quarot-q4-dir") + 1], f"{root}/sub/../../models/quarot")
        self.assertEqual(second[second.index("--tokenizer-dir") + 1], f"{root}/sub/../../models/tokenizer")

    def test_logit_dump_helper_runs_the_binary_through_the_handoff_with_absolute_paths(self):
        def launch(root, temp):
            stubs = temp / "pystub"
            stubs.mkdir()
            (stubs / "numpy.py").write_text(NUMPY_STUB)
            workdir = root / "sub"
            (workdir / "out").mkdir(parents=True)
            code = ("import sys; sys.path[:0] = [%r, %r]; import compare_logits as c; from pathlib import Path; "
                    "array = c.run_lattice_logit_dump([5, 6, 7], Path('../fixture-model.txt'), 'out/logits.bin'); "
                    "print('SHAPE', array.shape)" % (str(root / "scripts"), str(stubs)))
            return [sys.executable, "-c", code], workdir

        result, evidence = self.run_fixture(
            kind="bin", launch=launch, commit_files={"scripts/compare_logits.py": REPO / "scripts/compare_logits.py"},
            bin_name="bench_logit_dump", bin_args=(), features=("default", "f16", "metal-gpu", "std"), logits=True,
            stdout_text="VOCAB=4\nNPOS=2\nOUT=elsewhere", stderr_text="[bench_logit_dump] loading model")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SHAPE (2, 4)", result.stdout)
        root = evidence["root"]
        launch_record = evidence["launches"][0]
        self.assertEqual((launch_record["argv"], launch_record["cwd"]), ([], str(root)))
        self.assertEqual(launch_record["model"], f"{root}/fixture-model.txt")
        self.assertEqual(launch_record["out"], f"{root}/sub/out/logits.bin")
        call = evidence["cargo_calls"][0]
        self.assertEqual(call[call.index("--features") + 1], "metal-gpu,f16")
        self.assertIn("--release", call)
        # Lock-only before, as it was: no before/after probes, the measured-phase one only.
        self.assertEqual(evidence["quiet_log"], ["--label command:bench_logit_dump: measured guard"])

    MLX_ROWS = "mlx\tq8\t15.8218\t2041\nmlx\tq4\t18.1839\t2041\n"

    def run_quality_with_mlx(self, *, uv_rc=0, uv_out=MLX_ROWS, env=None):
        """bench_quality.sh with the MLX cross-check enabled, the cross-check stubbed as `uv`."""
        with tempfile.TemporaryDirectory(prefix="quality-models-") as models:
            dirs = {name: Path(models).resolve() / name for name in ("q4", "quarot", "tokenizer")}
            for directory in dirs.values():
                directory.mkdir()
            result, evidence = self.run_script(
                "scripts/bench_quality.sh", bin_name="eval_perplexity", bin_args=(),
                gitignore=(REPO / ".gitignore").read_text(),
                extra_files={"docs/bench_results/wiki.test.raw": "corpus\n", "docs/bench_results/perplexity.tsv": "canonical\n"},
                extra_env={"Q4_DIR": str(dirs["q4"]), "QUAROT_DIR": str(dirs["quarot"]), "TOK_DIR": str(dirs["tokenizer"]),
                           "BENCH_MACHINE": "fixture", "MLX_LOG": str(Path(models).resolve() / "mlx.log"),
                           "FIXTURE_UV_OUT": uv_out, "FIXTURE_UV_ERR": "  q8: PPL = 15.8218\n", **(env or {})},
                stdout_text="PPL:                16.589166", uv_rc=uv_rc, read_files=("docs/bench_results/perplexity.tsv",))
            evidence["dirs"] = dirs
            evidence["mlx_log"] = (Path(models).resolve() / "mlx.log").read_text()
            return result, evidence

    def test_quality_script_runs_the_mlx_cross_check_as_a_durable_plain_launch_under_both_locks(self):
        result, evidence = self.run_quality_with_mlx()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # The cross-check runs after the two lattice tiers and never overlaps either of them.
        self.assertEqual(evidence["events"], ["lattice-start", "lattice-end"] * 2 + ["mlx-start", "mlx-end"])
        # Both machine locks were held by the supervisor around the MLX process, which was not
        # handed any lock or handoff capability.
        self.assertEqual(evidence["uv_record"], {
            "gpu_lock_held": True, "window_held": True, "supervisor_marker": True,
            "handoff_env": [], "lock_fds": False})
        # Plain route: the supervisor ran no admission, so no handoff build happened for it,
        # and the durable before/after ambient-idle checks surround it like the lattice tiers.
        self.assertEqual(len(evidence["cargo_calls"]), 2)
        self.assertEqual(sorted(evidence["quiet_log"]), sorted(
            [f"--label {label}" for label in ["quality-perplexity: before", "quality-perplexity: after"] * 2
             + ["command:eval_perplexity: measured guard"] * 2 + ["quality-mlx: before", "quality-mlx: after"]]))
        # The program still arrives on the cross-check's stdin through the supervisor, with the same arguments.
        dirs = evidence["dirs"]
        root = evidence["root"]
        self.assertEqual(
            evidence["uv_args"],
            f"run --quiet --with mlx-lm python3 - {dirs['tokenizer']} {root}/docs/bench_results/wiki.test.raw 512 256 2048")
        for text in ("from mlx_lm import load", "ppl_at_bits(8, \"q8\")", "ppl_at_bits(4, \"q4\")"):
            self.assertIn(text, evidence["uv_stdin"])
        # Output contract: only the exact MLX rows reach the data file (the supervisor's probe
        # lines share that stdout), and stderr goes to the log.
        published = evidence["files"]["docs/bench_results/perplexity.tsv"]
        for row in ("lattice\tq4\t16.589166\t2048\n", "lattice\tq4-quarot\t16.589166\t2048\n",
                    "mlx\tq8\t15.8218\t2041\n", "mlx\tq4\t18.1839\t2041\n"):
            self.assertIn(row, published)
        self.assertNotIn("fixture inside-guard", published)
        self.assertIn("q8: PPL = 15.8218", evidence["mlx_log"])

    def test_quality_script_treats_a_failing_mlx_cross_check_as_fatal_and_publishes_nothing(self):
        # It was fatal before the handoff migration and stays so: exit 1 after the lattice tiers,
        # with the canonical file untouched, whether or not the failed process had printed rows.
        for uv_out in ("", self.MLX_ROWS):
            with self.subTest(rows_printed=bool(uv_out)):
                result, evidence = self.run_quality_with_mlx(uv_rc=3, uv_out=uv_out)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("MLX cross-check failed (exit 3", result.stderr)
                self.assertEqual(evidence["files"]["docs/bench_results/perplexity.tsv"], "canonical\n")
                self.assertEqual(evidence["events"], ["lattice-start", "lattice-end"] * 2 + ["mlx-start", "mlx-end"])

    def test_quality_script_refuses_an_mlx_cross_check_that_produced_no_rows(self):
        result, evidence = self.run_quality_with_mlx(uv_out="")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("did not produce exactly one q8 row", result.stderr)
        self.assertEqual(evidence["files"]["docs/bench_results/perplexity.tsv"], "canonical\n")

    def test_quality_script_mlx_cross_check_is_gated_by_the_durable_ambient_idle_check(self):
        # A noisy machine before the cross-check: the supervisor refuses (exit 2) and the MLX
        # process never starts; the script fails and publishes nothing.
        result, evidence = self.run_quality_with_mlx(env={"FIXTURE_QUIET_FAIL": "quality-mlx: before"})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("MLX cross-check failed (exit 2", result.stderr)
        self.assertNotIn("mlx-start", evidence["events"])
        self.assertEqual(evidence["files"]["docs/bench_results/perplexity.tsv"], "canonical\n")

    MLX_ENV = {"LATTICE_LOGIT_TMP": ".cache/lattice_logits.bin"}

    def run_logit_script(self, *, mlx_mode="", n_tokens=2):
        """compare_logits.py main() end to end: real bench-command.sh and handoff, stub numpy/MLX/cargo."""
        def launch(root, temp):
            stubs = temp / "pystub"
            (stubs / "mlx").mkdir(parents=True)
            (stubs / "numpy.py").write_text(NUMPY_STUB)
            (stubs / "mlx_lm.py").write_text(MLX_LM_STUB)
            (stubs / "mlx/__init__.py").write_text("")
            (stubs / "mlx/core.py").write_text(MLX_CORE_STUB)
            (stubs / "transformers.py").write_text(TRANSFORMERS_STUB)
            code = "\n".join([
                "import os, sys",
                "print('PARENT_PID', os.getpid())",
                "sys.argv = ['compare_logits.py', '--n-tokens=%d']" % n_tokens,
                "sys.path[:0] = [%r, %r]" % (str(root / "scripts"), str(stubs)),
                "import compare_logits as c",
                "collect = c.collect_mlx_via_child",
                "def wrapped():",
                "    ids, logits = collect()",
                "    print('CHILD', len(ids), logits.shape)",
                "    return ids, logits",
                "c.collect_mlx_via_child = wrapped",
                "c.analyze = lambda lat, mlx, ids: print('ANALYZE', lat.shape, mlx.shape, ids, mlx.flat[:4], mlx.flat[4:8])",
                "c.main()",
            ])
            return [sys.executable, "-c", code], root

        return self.run_fixture(
            kind="bin", launch=launch, bin_name="bench_logit_dump", bin_args=(), features=("default", "f16", "metal-gpu", "std"),
            logits=True, stdout_text="VOCAB=4\nNPOS=2\nOUT=elsewhere",
            commit_files={"scripts/compare_logits.py": REPO / "scripts/compare_logits.py",
                          "docs/bench_results/wiki.test.raw": "corpus\n"},
            extra_env={**self.MLX_ENV, "FIXTURE_MLX_MODE": mlx_mode, "PYTHONPATH": "{TEMP}/pystub"})

    def test_logit_script_runs_all_mlx_work_in_one_child_under_both_locks_before_the_lattice_run(self):
        result, evidence = self.run_logit_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # Tokenizer load, then prefill load (the original two loads), one process exit, and only then
        # the Lattice binary. Nothing of the MLX work happened in the parent or overlapped the Lattice run.
        self.assertEqual(evidence["events"], ["mlx-load", "mlx-load", "mlx-exit", "lattice-start", "lattice-end"])
        loads = evidence["mlx_loads"]
        self.assertEqual(len(loads), 2)
        self.assertEqual({item["pid"] for item in loads}.__len__(), 1)
        # The MLX work is a different process from the one that runs main().
        self.assertNotIn(f"PARENT_PID {loads[0]['pid']}\n", result.stdout)
        self.assertRegex(result.stdout, r"PARENT_PID \d+\n")
        for record in loads:
            self.assertEqual(
                {key: record[key] for key in ("gpu_lock_held", "window_held", "supervisor_marker", "handoff_env", "lock_fds")},
                {"gpu_lock_held": True, "window_held": True, "supervisor_marker": True, "handoff_env": [], "lock_fds": False})
        # Lock-only, as before the migration: no before/after probes for the MLX child.
        self.assertEqual(evidence["quiet_log"], ["--label command:bench_logit_dump: measured guard"])
        # The child's token ids and logits reach the analysis (which runs after both passes, outside the locks).
        self.assertIn("CHILD 2 (2, 4)\n", result.stdout)
        self.assertIn("ANALYZE (2, 4) (2, 4) [101, 102] [0.0, 1.0, 2.0, 3.0] [10.0, 11.0, 12.0, 13.0]", result.stdout)
        # The child's progress lines are not captured.
        self.assertIn("[step 1] tokenizing corpus with MLX tokenizer", result.stdout)
        self.assertEqual(len(evidence["launches"]), 1)

    def test_logit_script_keeps_the_transformers_tokenizer_fallback_inside_the_child(self):
        result, evidence = self.run_logit_script(mlx_mode="first-load-fails")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("mlx_lm tokenizer failed", result.stdout)
        self.assertIn("ANALYZE (2, 4) (2, 4) [201, 202]", result.stdout)
        self.assertEqual(evidence["events"], ["mlx-load", "mlx-load", "mlx-exit", "lattice-start", "lattice-end"])

    def test_logit_script_stops_before_the_lattice_run_when_the_mlx_child_fails(self):
        for mode in ("all-loads-fail", "prefill-fails"):
            with self.subTest(mode=mode):
                result, evidence = self.run_logit_script(mlx_mode=mode)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("MLX child exited 1", result.stderr)
                self.assertNotIn("lattice-start", evidence["events"])
                self.assertEqual((evidence["launches"], evidence["cargo_calls"]), ([], []))
                self.assertNotIn("ANALYZE", result.stdout)


NUMPY_STUB = """
import struct

float32 = "<f4"


class _Array:
    def __init__(self, flat, shape):
        self.flat, self.shape = flat, shape

    def reshape(self, *shape):
        assert len(self.flat) == shape[0] * shape[1], (len(self.flat), shape)
        return _Array(self.flat, shape)

    def copy(self):
        return self

    def astype(self, dtype, **options):
        return self

    def tobytes(self):
        return struct.pack("<%df" % len(self.flat), *self.flat)

    def __getitem__(self, rows):
        width = self.shape[1]
        return _Array(self.flat[rows.start or 0:][: (rows.stop - (rows.start or 0)) * width], (rows.stop - (rows.start or 0), width))


def asarray(rows, dtype=None):
    return _Array([float(value) for row in rows for value in row], (len(rows), len(rows[0])))


def frombuffer(raw, dtype=None):
    count = len(raw) // 4
    return _Array(list(struct.unpack("<%df" % count, raw)), (count,))
"""


MLX_LM_STUB = r"""
import atexit, fcntl, json, os, pathlib

_events = os.environ["FIXTURE_EVENTS"]
_mode = os.environ.get("FIXTURE_MLX_MODE", "")
_marker = pathlib.Path(_events + ".first-load-done")
atexit.register(lambda: open(_events, "a").write("mlx-exit\n"))


def _held(path):
    with open(path, "r+") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [101 + i for i in range(64)]


class _Logits:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        return self

    def tolist(self):
        return [[float(10 * i + j) for j in range(4)] for i in range(self.rows)]


class _Model:
    def eval(self):
        pass

    def __call__(self, ids):
        if _mode == "prefill-fails":
            raise RuntimeError("injected prefill failure")
        return _Logits(len(ids.rows[0]))


def load(path):
    open(_events, "a").write("mlx-load\n")
    with open(_events + ".loads", "a") as sink:
        sink.write(json.dumps(dict(
            pid=os.getpid(), gpu_lock_held=_held(os.environ["FIXTURE_GPU"]), window_held=_held(os.environ["FIXTURE_WINDOW"]),
            supervisor_marker="LATTICE_GPU_LOCK_SUPERVISOR_PID" in os.environ,
            handoff_env=sorted(name for name in os.environ if name.startswith("LATTICE_GPU_HANDOFF_")),
            lock_fds="LATTICE_BENCH_LOCK_FDS" in os.environ)) + "\n")
    if _mode == "all-loads-fail" or (_mode == "first-load-fails" and not _marker.exists()):
        _marker.write_text("1")
        raise RuntimeError("injected mlx load failure")
    return _Model(), _Tok()
"""

MLX_CORE_STUB = """
class array:
    def __init__(self, rows):
        self.rows = rows


def eval(value):
    pass
"""

TRANSFORMERS_STUB = """
import os


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [201 + i for i in range(64)]


class AutoTokenizer:
    @staticmethod
    def from_pretrained(path, **options):
        assert options == {"trust_remote_code": False, "local_files_only": True}, options
        if os.environ.get("FIXTURE_MLX_MODE") == "all-loads-fail":
            raise RuntimeError("injected transformers failure")
        return _Tok()
"""


def _load_script_with_numpy_stub(name, filename):
    """Import a script whose numpy-dependent bodies the test does not run."""
    import types
    from unittest import mock

    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"numpy": types.ModuleType("numpy")}):
        spec.loader.exec_module(module)
    return module


class GpuHandoffScriptStreams(unittest.TestCase):
    """The handoff merges the binary's standard error into the stream its callers parse."""

    def test_slopefit_parser_counts_only_tagged_records(self):
        parser = _load_script_with_numpy_stub("slopefit_stream", "bench_decode_slopefit.py")
        records = [
            "SLOPEFIT_META kv_cache_len=1024 warmup=8 measure=32 repeats=2",
            "SLOPEFIT_META ctx=64 actual_prompt_tokens=70",
            "SLOPEFIT ctx=64 tokens=32 warmup_ms=0.0 measure_ms=96.000 rep=0",
            "SLOPEFIT ctx=64 tokens=32 warmup_ms=0.0 measure_ms=64.000 rep=1",
        ]
        noise = [
            "fixture inside-guard CPU idle sample",
            "[slopefit] loading /models/qwen (Q4)",
            "[slopefit] grid=[64] warmup=8 measure=32 repeats=2",
            "[slopefit] kv_cache_len=1024 (deepest_prompt=70 for max_ctx=64 + decode_horizon 32)",
            "[slopefit] ctx=64 actual_prompt_tokens=70",
            # Progress text that quotes record fields without being a record.
            "[slopefit] ctx=64 actual_prompt_tokens=9999",
            "[slopefit] ctx=64 tokens=1 warmup_ms=0.0 measure_ms=1.000 rep=0",
            "bench-supervision: measured-phase receipt: /repo/.cache/receipt.jsonl",
        ]
        clean = parser.parse_stream(records)
        merged = [noise[0], noise[1], records[0], noise[2], noise[3], noise[4], records[1], noise[5],
                  records[2], noise[6], records[3], noise[7]]
        self.assertEqual(parser.parse_stream(merged), clean)
        self.assertEqual(clean[2], {64: 70})
        self.assertEqual(dict(clean[0]), {64: [3.0, 2.0]})
        # A stream that lost its records yields no run metadata, which the post-processor refuses.
        self.assertIsNone(parser.parse_stream(noise)[3])

    def test_logit_dump_header_counts_only_exact_records(self):
        module = _load_script_with_numpy_stub("compare_logits_stream", "compare_logits.py")
        output = "\n".join([
            "fixture inside-guard CPU idle sample",
            "[bench_logit_dump] loading /models/qwen",
            "VOCAB=4",
            "[bench_logit_dump] writing 2x4 f32",
            "NPOS=2",
            "OUT=/tmp/logits.bin",
            "[bench_logit_dump] VOCAB=999",
            "[bench_logit_dump] NPOS=999",
        ])
        self.assertEqual(module.parse_dump_header(output), (4, 2))
        self.assertEqual(module.parse_dump_header("[bench_logit_dump] VOCAB=4\n[x] NPOS=2"), (None, None))

    def test_logit_dump_helper_refuses_the_retired_prebuilt_binary_override(self):
        module = _load_script_with_numpy_stub("compare_logits_bindir", "compare_logits.py")
        with self.assertRaisesRegex(SystemExit, "LATTICE_BIN_DIR is not supported"):
            module.reject_prebuilt_binary_dir({"LATTICE_BIN_DIR": "/prebuilt"})
        module.reject_prebuilt_binary_dir({})
        module.reject_prebuilt_binary_dir({"LATTICE_BIN_DIR": ""})

    def test_logit_script_rejects_the_override_before_any_work(self):
        """The refusal has to precede the MLX pass, not surface after it."""
        import ast

        source = (REPO / "scripts/compare_logits.py").read_text()
        main = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        first = main.body[0]
        self.assertIsInstance(first, ast.Expr)
        self.assertEqual(ast.unparse(first.value), "reject_prebuilt_binary_dir(os.environ)")


class _FailOnEmptyTestProgram(unittest.TestProgram):
    def runTests(self) -> None:
        if self.test.countTestCases() == 0:
            raise SystemExit("no tests collected")
        super().runTests()


if __name__ == "__main__":
    _FailOnEmptyTestProgram()

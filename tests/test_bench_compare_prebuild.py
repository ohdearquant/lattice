#!/usr/bin/env python3
"""Regression tests for prebuilding benchmarks before an arm's sampler starts.

These cases live apart from tests/test_bench_compare_measurement.py because that
module is already above ADR-102's line cap and may only shrink. The script-level
runner and the shared stubs stay in that module; this one imports them rather
than copying them.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _load_measurement():
    path = Path(__file__).with_name("test_bench_compare_measurement.py")
    spec = importlib.util.spec_from_file_location("bench_compare_measurement", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measurement = _load_measurement()
_run = measurement._run
GIT = measurement.GIT
STALE_CHANGE_CARGO = measurement.STALE_CHANGE_CARGO
STUB_PHASE_SAMPLER = measurement.STUB_PHASE_SAMPLER

_PHASE_READY_MARKER = 'pathlib.Path(out + ".ready").touch()'
assert STUB_PHASE_SAMPLER.count(_PHASE_READY_MARKER) == 1
STUB_PHASE_SAMPLER_RECORDING_START = STUB_PHASE_SAMPLER.replace(
    "import json\n", "import json\nimport os\n", 1,
).replace(
    _PHASE_READY_MARKER,
    _PHASE_READY_MARKER + "\n"
    'with open(os.environ["STUB_SAMPLER_START_LOG"], "a", encoding="utf-8") as fh:\n'
    '    fh.write(arm + "\\n")',
)

SAMPLER_STATE_ANCHOR = 'if [[ "${1:-}" == "bench" && "${STUB_REQUIRE_LOCKED:-0}" == "1" ]]; then'
assert STALE_CHANGE_CARGO.count(SAMPLER_STATE_ANCHOR) == 1
RECORDING_STALE_CHANGE_CARGO = STALE_CHANGE_CARGO.replace(
    SAMPLER_STATE_ANCHOR,
    'if [[ -n "${STUB_SAMPLER_STATE_FILE:-}" ]]; then\n'
    '  sampler="not-running"\n'
    '  for marker in "${STUB_REPO_ROOT:-}/.cache/bench-compare-criterion/phase-load/"*.ready; do\n'
    '    if [[ -e "$marker" ]]; then\n'
    '      sampler="running"\n'
    '    fi\n'
    '  done\n'
    '  package="none"\n'
    '  if [[ " $* " == *" -p lattice-embed "* ]]; then\n'
    '    package="lattice-embed"\n'
    '  elif [[ " $* " == *" -p lattice-inference "* ]]; then\n'
    '    package="lattice-inference"\n'
    '  fi\n'
    '  mode="run"\n'
    '  if [[ " $* " == *" --no-run "* ]]; then\n'
    '    mode="build"\n'
    '  fi\n'
    '  printf \'%s\\t%s\\t%s\\n\' "$package" "$mode" "$sampler" \\\n'
    '    >> "$STUB_SAMPLER_STATE_FILE"\n'
    'fi\n\n' + SAMPLER_STATE_ANCHOR,
    1,
)

FAILING_EMBED_BUILD_CARGO = """#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  printf '%s\n' 'cargo 1.94.1 (fixture)'
  exit 0
fi
if [[ -n "${STUB_SAMPLER_STATE_FILE:-}" ]]; then
  sampler="not-running"
  for marker in "${STUB_REPO_ROOT:-}/.cache/bench-compare-criterion/phase-load/"*.ready; do
    if [[ -e "$marker" ]]; then
      sampler="running"
    fi
  done
  package="none"
  if [[ " $* " == *" -p lattice-embed "* ]]; then
    package="lattice-embed"
  elif [[ " $* " == *" -p lattice-inference "* ]]; then
    package="lattice-inference"
  fi
  mode="run"
  if [[ " $* " == *" --no-run "* ]]; then
    mode="build"
  fi
  printf '%s\t%s\t%s\n' "$package" "$mode" "$sampler" \
    >> "$STUB_SAMPLER_STATE_FILE"
fi
if [[ " $* " == *" -p lattice-embed "* && " $* " == *" --no-run "* ]]; then
  printf '%s\n' 'fixture embed build failed' >&2
  exit 7
fi
exit 0
"""

# The GPU handoff builds into its own target directory, and both `prebuild`
# and `measure` run that identical build. This fixture records the sampler's
# readiness at each such build, then emits the artifact Cargo would emit and
# installs an executable that speaks the measured-phase handshake.
HANDOFF_CARGO = r"""#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--version" ]]; then
  printf '%s\n' 'cargo 1.94.1 (fixture)'
  exit 0
fi

args=" $* "
if [[ "$args" == *" --message-format=json "* && "$args" == *".cache/bench-gpu-handoff-build/"* ]]; then
  sampler="not-running"
  for marker in "${STUB_REPO_ROOT:-}/.cache/bench-compare-criterion/phase-load/"*.ready; do
    if [[ -e "$marker" ]]; then
      sampler="running"
    fi
  done
  {
    printf '%s\t' "$sampler"
    printf '%s\x1f' "$@"
    printf '\n'
  } >> "$STUB_HANDOFF_LOG"
  if [[ "${STUB_FAIL_FIRST_HANDOFF_BUILD:-0}" == "1" && ! -e "${STUB_HANDOFF_LOG}.failed" ]]; then
    : > "${STUB_HANDOFF_LOG}.failed"
    printf '%s\n' 'fixture handoff build failed' >&2
    exit 7
  fi
  target="${args#* --bench }"
  target="${target%% *}"
  target_dir="${args#* --target-dir }"
  target_dir="${target_dir%% *}"
  mkdir -p "$target_dir/release/deps"
  cp "$STUB_HANDOFF_SOURCE" "$target_dir/release/deps/$target"
  chmod 755 "$target_dir/release/deps/$target"
  "$STUB_HANDOFF_PYTHON" - "$target" "$target_dir/release/deps/$target" <<'PY'
import json, os, pathlib, sys
target, executable = sys.argv[1], sys.argv[2]
cwd = pathlib.Path.cwd()
print(json.dumps({
    "reason": "compiler-artifact",
    "package_id": "path+" + (cwd / "crates/inference").as_uri() + "#lattice-inference@0.0.1",
    "target": {
        "name": target,
        "kind": ["bench"],
        "src_path": str(cwd / "crates/inference" / "benches" / (target + ".rs")),
    },
    "features": json.loads(os.environ["STUB_HANDOFF_FEATURES"]),
    "executable": executable,
}))
PY
  exit 0
fi

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
    write_head "rms_norm/4096"
  else
    write_head "simd_dot_product/scalar/384"
  fi
fi

printf '%s\n' 'time: [1.000 ns 1.010 ns 1.020 ns]'
if [[ "$args" == *" --baseline "* ]]; then
  printf '%s\n' 'change: [+0.0% +1.0% +2.0%] (p = 0.50 > 0.05)'
fi
"""


HANDOFF_BENCH_EXECUTABLE = r"""import json
import os
import pathlib
import socket
import sys

args = sys.argv[1:]
home = pathlib.Path(os.environ["CRITERION_HOME"])
token = os.environ["LATTICE_GPU_HANDOFF_TOKEN"]
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    connection.connect(os.environ["LATTICE_GPU_HANDOFF_CONTROL"])
    connection.sendall(json.dumps({
        "protocol": 1,
        "token": token,
        "pid": os.getpid(),
        "exe": str(pathlib.Path(__file__).resolve()),
    }).encode() + b"\n")
    with connection.makefile("rb") as stream:
        response = json.loads(stream.readline())
    assert response == {"protocol": 1, "token": token, "status": "ready"}, response


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


for bench in ("rms_norm/896", "rms_norm/4096"):
    if "--save-baseline" in args:
        baseline = args[args.index("--save-baseline") + 1]
        write(home / bench / baseline / "estimates.json",
              {"mean": {"point_estimate": 90.0}})
        write(home / bench / baseline / "sample.json",
              {"sampling_mode": "Linear", "iters": [1.0, 2.0], "times": [1.0, 2.0]})
    else:
        write(home / bench / "new" / "estimates.json",
              {"mean": {"point_estimate": 100.0}})
        write(home / bench / "new" / "sample.json",
              {"sampling_mode": "Flat", "iters": [1.0, 2.0], "times": [1.0, 2.0]})
        write(home / bench / "change" / "estimates.json",
              {"mean": {"point_estimate": 0.01,
                        "confidence_interval": {"lower_bound": 0.0, "upper_bound": 0.02}}})
print("time: [1.000 ns 1.010 ns 1.020 ns]")
if "--baseline" in args:
    print("change: [+0.0% +1.0% +2.0%] (p = 0.50 > 0.05)")
"""


class PrebuildOrdering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        measurement.GpuBenchmarkAdmission.setUpClass()
        cls.admission = measurement.GpuBenchmarkAdmission.admission

    manifest = measurement.GpuBenchmarkAdmission.manifest
    compare = measurement.GpuBenchmarkAdmission.compare

    def test_failed_embed_build_refuses_before_its_sampler_starts(self):
        """A failed embed prebuild must refuse the run with its own label."""
        with tempfile.TemporaryDirectory() as temporary:
            state_file = Path(temporary) / "cargo-sampler-state.tsv"
            result = _run(
                [],
                stub_cargo=FAILING_EMBED_BUILD_CARGO,
                extra_env={"STUB_SAMPLER_STATE_FILE": str(state_file)},
            )
            records = [
                tuple(line.split("\t"))
                for line in state_file.read_text().splitlines()
            ]
        self.assertEqual(
            result.returncode, 2,
            f"expected exit 2 (embed build failed), got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn(
            "base lattice-embed:simd build (--no-run) failed (exit 1)", result.stderr
        )
        self.assertNotIn("base lattice-embed:simd exited 0", result.stderr)
        self.assertEqual(
            records,
            [("lattice-inference", "build", "not-running"),
             ("lattice-embed", "build", "not-running")],
            records,
        )

    def test_embed_bench_is_built_before_each_arms_sampler_starts(self):
        """Every embed build must finish before that arm's sampler runs.

        The stub cargo records whether a phase sampler readiness marker
        exists at invocation time. A build after phase_sampler_start would
        compile inside the sampled window, which is self-load the sampler's
        decaying process average under-reads.

        Mutation-sensitive: remove the embed --no-run call from any arm and
        that arm contributes no embed build record, so the four-build
        assertion fails. Move the build after phase_sampler_start and its
        record says "running".
        """
        with tempfile.TemporaryDirectory() as temporary:
            state_file = Path(temporary) / "cargo-sampler-state.tsv"
            result = _run(
                [],
                stub_cargo=RECORDING_STALE_CHANGE_CARGO,
                extra_env={"STUB_SAMPLER_STATE_FILE": str(state_file)},
            )
            records = [
                tuple(line.split("\t"))
                for line in state_file.read_text().splitlines()
            ]
        self.assertEqual(
            result.returncode, 0,
            f"embed prebuild fixture failed\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )
        embed = [record for record in records if record[0] == "lattice-embed"]
        builds = [record for record in embed if record[1] == "build"]
        runs = [record for record in embed if record[1] == "run"]
        self.assertEqual(
            len(builds), 4,
            f"expected one embed build per arm, got {builds}\nrecords: {records}",
        )
        self.assertEqual(
            [record[2] for record in builds], ["not-running"] * 4,
            f"an embed build ran inside a sampled window: {builds}",
        )
        self.assertEqual(
            len(runs), 4,
            f"expected one embed measurement per arm, got {runs}\nrecords: {records}",
        )
        self.assertEqual(
            [record[2] for record in runs], ["running"] * 4,
            f"an embed measurement ran outside a sampled window: {runs}",
        )

    def _run_handoff(self, temporary, *, fail_first_build=False,
                     stub_phase_sampler=None, extra_env=None):
        """Run one GPU-handoff A/B whose stub cargo builds the admitted target."""
        source = Path(temporary) / "handoff-bench.py"
        source.write_text(f"#!{sys.executable}\n" + HANDOFF_BENCH_EXECUTABLE)
        log = Path(temporary) / "handoff-builds.jsonl"
        features = ["default", "f16", "metal-gpu", "std"]

        def setup(root):
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["crates/inference"]\n'
                '[workspace.package]\nversion = "0.0.1"\n'
            )
            package = root / "crates" / "inference"
            targets = [
                "cross_turn_prefix_cache_bench", "decode_attn_bench", "lm_head_bench",
                "metal_decode_bench", "mtp_decode", "topk_readback",
            ]
            bins = ["bench_decode_ab", "bench_decode_slopefit",
                    "bench_logit_dump", "eval_perplexity"]
            manifest = (
                '[package]\nname = "lattice-inference"\nversion.workspace = true\n'
                '[package.metadata.gpu-bench-handoff]\nversion = 1\n'
                f'targets = {json.dumps(targets)}\nbins = {json.dumps(bins)}\n'
                '[features]\ndefault = ["std"]\nstd = []\n'
                'metal-gpu = []\nf16 = []\nbench-internals = []\n'
            )
            for target in targets:
                manifest += f'[[bench]]\nname = "{target}"\nharness = false\n'
            (package / "Cargo.toml").write_text(manifest)
            path = package / "benches" / "decode_attn_bench.rs"
            path.write_text('let mut group = c.benchmark_group("rms_norm");\n')
            env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                   "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
            subprocess.run([*GIT, "-C", str(root), "add", "-A"], check=True)
            subprocess.run([*GIT, "-C", str(root), "commit", "-qm", "handoff fixture"],
                           check=True, env=env)

        environment = {
            "BENCHES_INFERENCE": "decode_attn_bench",
            "CARGO_FEATURES_INFERENCE": "metal-gpu,f16",
            "STUB_HANDOFF_LOG": str(log),
            "STUB_HANDOFF_SOURCE": str(source),
            "STUB_HANDOFF_FEATURES": json.dumps(features),
            "STUB_HANDOFF_PYTHON": sys.executable,
            "STUB_FAIL_FIRST_HANDOFF_BUILD": "1" if fail_first_build else "0",
            **(extra_env or {}),
        }
        result = _run(["--gpu-handoff"], stub_cargo=HANDOFF_CARGO, setup=setup,
                      extra_env=environment, stub_phase_sampler=stub_phase_sampler)
        records = [
            tuple(line.split("\t", 1))
            for line in log.read_text().splitlines()
        ] if log.exists() else []
        return result, [
            (state, [arg for arg in argv.rstrip("\x1f").split("\x1f")])
            for state, argv in records
        ]

    def test_gpu_handoff_build_runs_before_each_arms_sampler_starts(self):
        """The admitted build must be warm before its arm's sampled window opens.

        `measure` builds into `.cache/bench-gpu-handoff-build`, which the shell's
        inference `--no-run` never populates, so without a prebuild the first
        measure in each arm compiles inside the window it is measuring.

        Mutation-sensitive: remove the prebuild call from any arm and that arm
        contributes no build record with the sampler stopped.
        """
        with tempfile.TemporaryDirectory() as temporary:
            result, builds = self._run_handoff(temporary)
        self.assertEqual(
            result.returncode, 0,
            f"GPU handoff fixture failed\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )
        self.assertEqual(len(builds), 8, f"expected 8 handoff builds, got {builds}")
        prebuilds, measures = builds[0::2], builds[1::2]
        self.assertEqual(
            [state for state, _ in prebuilds], ["not-running"] * 4,
            f"a handoff build ran inside a sampled window: {prebuilds}",
        )
        self.assertEqual(
            [state for state, _ in measures], ["running"] * 4,
            f"a handoff measurement ran outside a sampled window: {measures}",
        )
        self.assertEqual(
            [argv for _, argv in prebuilds], [argv for _, argv in measures],
            "the measure build did not see the prebuild's exact argv",
        )
        self.assertTrue(
            all(".cache/bench-gpu-handoff-build" in argv[argv.index("--target-dir") + 1]
                for _, argv in builds),
            builds,
        )

    def test_failed_gpu_handoff_prebuild_refuses_before_its_sampler_starts(self):
        """A failed prebuild must refuse the run without a sampled arm."""
        with tempfile.TemporaryDirectory() as temporary:
            starts = Path(temporary) / "sampler-starts.txt"
            result, builds = self._run_handoff(
                temporary, fail_first_build=True,
                stub_phase_sampler=STUB_PHASE_SAMPLER_RECORDING_START,
                extra_env={"STUB_SAMPLER_START_LOG": str(starts)},
            )
            started = starts.read_text().splitlines() if starts.exists() else []
        self.assertEqual(
            result.returncode, 2,
            f"failed prebuild did not refuse\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )
        self.assertIn("lattice-inference:decode_attn_bench prebuild failed (exit 7)",
                      result.stderr)
        self.assertEqual(started, [], f"a sampler started despite the refusal: {started}")
        self.assertEqual(
            [state for state, _ in builds], ["not-running"],
            f"a build ran after the refused prebuild: {builds}",
        )

    def test_prebuild_and_measure_run_identical_build_commands(self):
        """Both actions must resolve one frozen entry into the same Cargo argv.

        `measure` may only add to what `prebuild` warmed if the pair share the
        argv builder; a second copy is the drift this test exists to catch.
        """
        import json as json_module
        import types
        from unittest import mock

        admission = self.admission
        plan = self.compare()
        for frozen in plan["entries"]:
            frozen["cwd"] = str(Path.cwd().resolve())
        entry = plan["entries"][0]
        executable = Path(entry["target_dir"]) / "release" / "deps" / entry["target"]
        artifact = {
            "reason": "compiler-artifact",
            "package_id": f"path+{(Path(entry['cwd']) / 'crates/inference').as_uri()}"
                          f"#lattice-inference@{entry['package_version']}",
            "target": {"name": entry["target"], "kind": ["bench"],
                       "src_path": str(Path(entry["cwd"]) / entry["source_path"])},
            "features": entry["feature_set"], "executable": str(executable),
        }
        completed = mock.Mock(returncode=0, stderr="",
                              stdout=json_module.dumps(artifact) + "\n")
        args = mock.Mock(
            entry="base1", revision="a" * 40, target="decode_attn_bench",
            features="metal-gpu,f16", kind="bench", release=False,
            criterion=entry["argv"][1:],
        )
        calls = []

        def record(command, **kwargs):
            calls.append((command, kwargs))
            return completed

        handoff = types.SimpleNamespace(read_frozen_plan=lambda _repo: plan)
        with mock.patch.dict(os.environ, {
            "CRITERION_HOME": entry["criterion_home"],
        }), mock.patch.dict(sys.modules, {"bench_handoff": handoff}), \
                mock.patch.object(admission, "_clean_revision"), \
                mock.patch.object(admission.subprocess, "run", side_effect=record), \
                mock.patch.object(admission, "validate_artifact", return_value=executable):
            self.assertEqual(admission.prebuild(args), 0)
            # No broker capability: measure gets past the build and then refuses.
            with self.assertRaisesRegex(admission.AdmissionError, "no complete broker"):
                admission.measure(args)
        self.assertEqual(len(calls), 2)
        prebuild_command, prebuild_kwargs = calls[0]
        measure_command, measure_kwargs = calls[1]
        self.assertEqual(prebuild_command, measure_command)
        self.assertEqual(prebuild_kwargs["cwd"], measure_kwargs["cwd"])
        self.assertEqual(prebuild_kwargs["env"], measure_kwargs["env"])
        self.assertEqual(prebuild_command[:3], ["cargo", "bench", "--locked"])
        self.assertIn("--no-run", prebuild_command)
        self.assertEqual(
            prebuild_command[prebuild_command.index("--target-dir") + 1],
            entry["target_dir"],
        )
        self.assertIn(".cache/bench-gpu-handoff-build", entry["target_dir"])


if __name__ == "__main__":
    measurement._FailOnEmptyTestProgram()

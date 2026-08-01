#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bench_target_errors(manifest: Path) -> list[str]:
    document = tomllib.loads(manifest.read_text(encoding="utf-8"))
    benches = document.get("bench", [])
    errors: list[str] = []
    explicit_names: set[str] = set()

    for bench in benches:
        name = bench["name"]
        explicit_names.add(name)
        prefix = f"{manifest.parent.name}:{name}"
        if bench.get("bench", True) is not True:
            errors.append(f"{prefix}: bench targets may not set bench = false")
        if bench.get("harness", True) is not False:
            errors.append(f"{prefix}: stable bench targets must set harness = false")

        source = manifest.parent / bench.get("path", f"benches/{name}.rs")
        if not source.is_file():
            errors.append(f"{prefix}: source does not exist: {source}")
            continue
        text = source.read_text(encoding="utf-8")
        if "criterion_main!" not in text and re.search(r"\bfn\s+main\s*\(", text) is None:
            errors.append(f"{prefix}: source has no executable benchmark entry point")

    bench_dir = manifest.parent / "benches"
    if bench_dir.is_dir():
        implicit = sorted(
            {path.stem for path in bench_dir.glob("*.rs")} - explicit_names
        )
        for name in implicit:
            errors.append(
                f"{manifest.parent.name}:{name}: bench source must have an explicit [[bench]] entry"
            )

    return errors


class BenchTargetPolicyTests(unittest.TestCase):
    def write_fixture(
        self,
        root: Path,
        *,
        enabled: bool = True,
        harness: bool = False,
        source: str = "fn main() {}\n",
    ) -> Path:
        crate = root / "crates" / "fixture"
        (crate / "benches").mkdir(parents=True)
        manifest = crate / "Cargo.toml"
        manifest.write_text(
            '[package]\nname = "fixture"\nversion = "0.0.0"\nedition = "2024"\n'
            '[[bench]]\nname = "measured"\n'
            f"harness = {str(harness).lower()}\n"
            f"bench = {str(enabled).lower()}\n",
            encoding="utf-8",
        )
        (crate / "benches" / "measured.rs").write_text(source, encoding="utf-8")
        return manifest

    def test_workspace_bench_targets_are_executable(self) -> None:
        manifests = sorted((ROOT / "crates").glob("*/Cargo.toml"))
        self.assertGreater(
            len(manifests),
            0,
            "bench-target sweep found zero crate manifests; this is an instrument defect, not a clean result",
        )
        self.assertIn(
            ROOT / "crates" / "inference" / "Cargo.toml",
            manifests,
            "bench-target sweep did not reach crates/inference/Cargo.toml; check the workspace root and manifest glob",
        )
        errors = [
            error
            for manifest in manifests
            for error in bench_target_errors(manifest)
        ]
        self.assertEqual(errors, [])

    def test_disabled_target_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = self.write_fixture(Path(temp), enabled=False)
            self.assertIn("bench targets may not set bench = false", bench_target_errors(manifest)[0])

    def test_libtest_harness_target_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = self.write_fixture(Path(temp), harness=True)
            self.assertIn("must set harness = false", bench_target_errors(manifest)[0])

    def test_missing_entry_point_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = self.write_fixture(Path(temp), source="fn helper() {}\n")
            self.assertIn("no executable benchmark entry point", bench_target_errors(manifest)[0])

    def test_implicit_bench_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = self.write_fixture(Path(temp))
            (manifest.parent / "benches" / "unlisted.rs").write_text(
                "fn main() {}\n", encoding="utf-8"
            )
            self.assertIn(
                "bench source must have an explicit [[bench]] entry",
                bench_target_errors(manifest)[0],
            )

    def test_make_bench_gate_refuses_empty_target_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copy2(ROOT / "Makefile", root / "Makefile")
            (root / "scripts").mkdir()
            shutil.copy2(
                ROOT / "scripts" / "perf-bench-gate.py",
                root / "scripts" / "perf-bench-gate.py",
            )
            gate_script = (ROOT / "scripts" / "bench-gate.sh").read_text()
            gate_script = gate_script.replace(
                'source "$REPO/scripts/lib/bench-supervision.sh"\n'
                "\nbench_gate_measurement() {\n",
                "",
            )
            gate_script = gate_script.replace(
                "\n}\n\n"
                'bench_supervise_entry "bench-gate" durable '
                'bench_gate_measurement "$@"\n',
                "\n",
            )
            gate_script = re.sub(
                r'^bench_quiet_checkpoint "bench-gate: [^"]+"$',
                ":",
                gate_script,
                flags=re.MULTILINE,
            )
            (root / "scripts" / "bench-gate.sh").write_text(gate_script)
            (root / "scripts" / "bench-gate.sh").chmod(0o755)
            baseline = root / ".cache" / "perf-baselines" / "testarch-testos"
            baseline.mkdir(parents=True)
            bindir = root / "bin"
            bindir.mkdir()
            for name, body in {
                "uname": (
                    "#!/usr/bin/env bash\n"
                    'if [ "${1:-}" = "-m" ]; then echo testarch; else echo testos; fi\n'
                ),
                "git": "#!/usr/bin/env bash\nexit 0\n",
                "cargo": "#!/usr/bin/env bash\nexit 0\n",
            }.items():
                script = bindir / name
                script.write_text(body)
                script.chmod(0o755)
            result = subprocess.run(
                ["make", "bench-gate"],
                cwd=root,
                env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"},
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("UNSUITABLE AS BENCHMARK EVIDENCE", result.stdout)
            self.assertEqual(
                result.stderr.count("contains no benchmark estimates"),
                2,
                result.stdout + result.stderr,
            )


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, pattern
    if tests.countTestCases() == 0:
        raise RuntimeError("no tests collected from tests.test_bench_targets")
    return tests


class _FailOnEmptyTestProgram(unittest.TestProgram):
    def runTests(self) -> None:
        if self.test.countTestCases() == 0:
            raise SystemExit("no tests collected")
        super().runTests()


if __name__ == "__main__":
    _FailOnEmptyTestProgram()

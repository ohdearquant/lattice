#!/usr/bin/env python3

from __future__ import annotations

import re
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
        errors = [
            error
            for manifest in sorted((ROOT / "crates").glob("*/Cargo.toml"))
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


if __name__ == "__main__":
    unittest.main()

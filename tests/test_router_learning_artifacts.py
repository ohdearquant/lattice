#!/usr/bin/env python3
"""Failure-injection tests for the ADR-096 Cargo/nm artifact probe."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check-router-learning-artifacts.py"
SPEC = importlib.util.spec_from_file_location("router_learning_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)

TRAINER = (
    "_ZN12lattice_fann8training4rloo11RlooTrainer"
    "32step_batch_with_explored_actions17h0123456789abcdefE"
)
ORDINARY = "_ZN17lattice_inference5model6loader17h0123456789abcdefE"
FINISHED = {"reason": "build-finished", "success": True}


def json_lines(records: list[dict]) -> str:
    return "\n".join(json.dumps(record) for record in records) + "\n"


class ArtifactProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.target = self.root / "build output"
        self.target.mkdir()
        self.records: dict[str, list[dict]] = {}
        self.symbols: dict[str, str] = {}
        for variant in PROBE.VARIANTS:
            crates = ("lattice_inference", "lattice_fann") if variant != "default" else ("lattice_inference",)
            self.records[variant] = [self.artifact(crate, variant) for crate in crates] + [FINISHED.copy()]

    def artifact(self, crate: str, variant: str) -> dict:
        library = self.target / f"lib{crate}-{variant}.rlib"
        library.write_bytes(b"!<arch>\nfixture: nm is mocked\n")
        self.symbols[str(library)] = f"{library}[unit.o]:\n{ORDINARY} T 0 20\n"
        if crate == "lattice_fann" and variant == "router-learning":
            self.symbols[str(library)] += f"{TRAINER} T 20 30\n"
        features = []
        if crate == "lattice_inference":
            features = ["default", "std"]
            if variant != "default":
                features.append("mixture")
            if variant == "router-learning":
                features.append("router-learning")
        elif variant == "router-learning":
            features = ["online-router"]
        crate_dir = self.root / "crates" / PROBE.CRATES[crate]
        return {
            "reason": "compiler-artifact",
            "package_id": f"path+{crate_dir.as_uri()}#lattice-{PROBE.CRATES[crate]}@0.1.0",
            "manifest_path": str(crate_dir / "Cargo.toml"),
            "target": {
                "name": crate, "kind": ["lib"], "crate_types": ["lib"],
                "src_path": str(crate_dir / "src" / "lib.rs"),
            },
            "profile": {"test": False},
            "features": features,
            "filenames": [str(library), str(library.with_suffix(".rmeta"))],
            "executable": None,
            "fresh": True,
        }

    def fake_run(self, command: list[str], **kwargs) -> subprocess.CompletedProcess:
        self.assertEqual(kwargs, {
            "cwd": self.root, "capture_output": True, "text": True, "check": False,
        })
        if command[0] == "cargo":
            variant = command[command.index("--features") + 1] if "--features" in command else "default"
            output = json_lines(self.records[variant])
        else:
            self.assertEqual(command[:2], ["nm", "-P"])
            self.assertEqual(len(command), 3)
            output = self.symbols[command[2]]
        return subprocess.CompletedProcess(command, 0, output, "")

    def check(self, side_effect=None):
        with mock.patch.object(PROBE.subprocess, "run", side_effect=side_effect or self.fake_run) as run:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                PROBE.check(self.root, self.target)
        return run

    def parse(self, records: list[dict] | None = None, variant: str = "mixture"):
        return PROBE.collect_artifacts(
            json_lines(records if records is not None else self.records[variant]),
            self.root, self.target, variant,
        )

    def test_positive_control_precedes_both_off_builds_and_uses_exact_artifacts(self):
        run = self.check()
        commands = [call.args[0] for call in run.call_args_list]
        builds = [command for command in commands if command[0] == "cargo"]
        self.assertEqual([command[-1] if "--features" in command else "default" for command in builds], list(PROBE.VARIANTS))
        for command in builds:
            self.assertIn("--locked", command)
            self.assertIn("--lib", command)
            self.assertIn("--message-format=json", command)
            self.assertEqual(command[command.index("-p") + 1], "lattice-inference")
            self.assertEqual(command[command.index("--target-dir") + 1], str(self.target))
            self.assertNotIn("--no-default-features", command)
        self.assertEqual(commands[0], builds[0])
        self.assertTrue(all(command[0] == "nm" for command in commands[1:3]))
        for variant in PROBE.VARIANTS:
            for record in self.records[variant][:-1]:
                self.assertIn(["nm", "-P", record["filenames"][0]], commands)

    def test_default_allows_fann_when_it_has_no_online_symbols(self):
        self.records["default"].insert(1, self.artifact("lattice_fann", "default"))
        self.check()

    def test_cached_files_not_reported_by_cargo_do_not_count(self):
        self.records["mixture"] = [FINISHED.copy()]
        with self.assertRaisesRegex(PROBE.ProbeError, "missing Cargo library artifacts"):
            self.check()

    def test_missing_inference_or_required_fann_fails(self):
        for variant, crate in (("default", "lattice_inference"), ("mixture", "lattice_fann"), ("router-learning", "lattice_fann")):
            with self.subTest(variant=variant, crate=crate):
                records = [record for record in self.records[variant] if record.get("target", {}).get("name") != crate]
                with self.assertRaisesRegex(PROBE.ProbeError, crate):
                    self.parse(records, variant)

    def test_invalid_json_or_message_fails(self):
        for output in ("not JSON\n", "[]\n", "{}\n", "null\n"):
            with self.subTest(output=output), self.assertRaisesRegex(PROBE.ProbeError, "invalid Cargo"):
                PROBE.collect_artifacts(output, self.root, self.target, "default")

    def test_build_finished_is_required_successful_and_final(self):
        cases = (
            self.records["mixture"][:-1],
            self.records["mixture"][:-1] + [{"reason": "build-finished", "success": False}],
            self.records["mixture"][:-1] + [{"reason": "build-finished", "success": 1}],
            self.records["mixture"] + [FINISHED.copy()],
        )
        for records in cases:
            with self.subTest(records=records[-1]), self.assertRaisesRegex(PROBE.ProbeError, "build-finished"):
                self.parse(records)

    def test_duplicate_artifact_is_rejected(self):
        with self.assertRaisesRegex(PROBE.ProbeError, "duplicate Cargo artifact"):
            self.parse(self.records["mixture"][:1] + self.records["mixture"])

    def test_metadata_does_not_substitute_for_an_rlib(self):
        records = copy.deepcopy(self.records["mixture"])
        records[0]["filenames"] = records[0]["filenames"][1:]
        with self.assertRaisesRegex(PROBE.ProbeError, "exactly one.*rlib"):
            self.parse(records)

    def test_artifact_must_match_source_library_profile_and_features(self):
        mutations = (
            ("manifest_path", str(self.root / "other" / "Cargo.toml")),
            ("target", {"name": "lattice_inference", "kind": ["bin"]}),
            ("profile", {"test": True}),
            ("features", ["mixture", "router-learning"]),
            ("features", []),
            ("features", "mixture"),
            ("filenames", None),
            ("executable", "/tmp/example"),
        )
        for key, value in mutations:
            records = copy.deepcopy(self.records["mixture"])
            records[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(PROBE.ProbeError):
                self.parse(records)

    def test_unexpected_fann_feature_is_rejected(self):
        records = copy.deepcopy(self.records["mixture"])
        records[1]["features"] = ["online-router"]
        with self.assertRaisesRegex(PROBE.ProbeError, "unexpected feature online-router"):
            self.parse(records)

    def test_multiple_rlibs_are_rejected(self):
        records = copy.deepcopy(self.records["mixture"])
        records[0]["filenames"].append(records[0]["filenames"][0])
        with self.assertRaisesRegex(PROBE.ProbeError, "exactly one"):
            self.parse(records)

    def test_artifact_outside_target_is_rejected(self):
        records = copy.deepcopy(self.records["mixture"])
        records[0]["filenames"] = [str(self.root / "elsewhere.rlib")]
        with self.assertRaisesRegex(PROBE.ProbeError, "outside.*target directory"):
            self.parse(records)

    def test_missing_directory_or_nonarchive_artifact_fails(self):
        library = Path(self.records["mixture"][0]["filenames"][0])
        library.unlink()
        with self.assertRaisesRegex(PROBE.ProbeError, "cannot read"):
            self.parse()
        library.mkdir()
        with self.assertRaisesRegex(PROBE.ProbeError, "not a regular file"):
            self.parse()
        library.rmdir()
        library.write_bytes(b"not an archive")
        with self.assertRaisesRegex(PROBE.ProbeError, "not an rlib archive"):
            self.parse()

    def test_each_cargo_failure_propagates_even_with_plausible_stdout(self):
        for failed_variant in PROBE.VARIANTS:
            def fail(command, **kwargs):
                result = self.fake_run(command, **kwargs)
                variant = command[command.index("--features") + 1] if "--features" in command else "default"
                if command[0] == "cargo" and variant == failed_variant:
                    return subprocess.CompletedProcess(command, 101, result.stdout, "compiler failed")
                return result
            with self.subTest(variant=failed_variant), self.assertRaises(PROBE.ProbeError) as caught:
                self.check(fail)
            self.assertEqual(caught.exception.exit_code, 101)
            self.assertIn("compiler failed", str(caught.exception))

    def test_each_nm_failure_propagates_even_with_plausible_symbols(self):
        for failed_path in self.symbols:
            def fail(command, **kwargs):
                result = self.fake_run(command, **kwargs)
                if command[0] == "nm" and command[-1] == failed_path:
                    return subprocess.CompletedProcess(command, 7, result.stdout, "nm failed")
                return result
            with self.subTest(path=failed_path), self.assertRaises(PROBE.ProbeError) as caught:
                self.check(fail)
            self.assertEqual(caught.exception.exit_code, 7)
            self.assertIn("nm failed", str(caught.exception))

    def test_missing_tool_and_signal_failure_are_not_absence(self):
        with mock.patch.object(PROBE.subprocess, "run", side_effect=FileNotFoundError("missing")):
            with self.assertRaisesRegex(PROBE.ProbeError, "cannot run") as caught:
                PROBE.run_command(["nm"], self.root)
        self.assertEqual(caught.exception.exit_code, 2)
        result = subprocess.CompletedProcess(["nm"], -9, "", "killed")
        with mock.patch.object(PROBE.subprocess, "run", return_value=result):
            with self.assertRaises(PROBE.ProbeError) as caught:
                PROBE.run_command(["nm"], self.root)
        self.assertEqual(caught.exception.exit_code, 137)

    def test_undefined_references_do_not_satisfy_positive_control(self):
        library = self.records["router-learning"][1]["filenames"][0]
        for symbol_type in ("U", "w", "v"):
            self.symbols[library] = f"{ORDINARY} T 0 20\n{TRAINER} {symbol_type}\n"
            with self.subTest(symbol_type=symbol_type), self.assertRaisesRegex(PROBE.ProbeError, "positive control missing"):
                self.check()

    def test_failed_positive_control_stops_before_off_builds(self):
        library = self.records["router-learning"][1]["filenames"][0]
        self.symbols[library] = f"{ORDINARY} T 0 20\n"
        with mock.patch.object(PROBE.subprocess, "run", side_effect=self.fake_run) as run:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(PROBE.ProbeError):
                PROBE.check(self.root, self.target)
        builds = [call.args[0] for call in run.call_args_list if call.args[0][0] == "cargo"]
        self.assertEqual(len(builds), 1)

    def test_defined_online_symbols_fail_both_off_variants(self):
        for variant in ("mixture", "default"):
            for module in ("rloo", "router_update", "ewc"):
                library = self.records[variant][0]["filenames"][0]
                before = self.symbols[library]
                self.symbols[library] += f"_ZN12lattice_fann8training{module}function T 50 10\n"
                with self.subTest(variant=variant, module=module), self.assertRaisesRegex(PROBE.ProbeError, "online-router definitions present"):
                    self.check()
                self.symbols[library] = before

    def test_undefined_online_reference_is_not_reported_as_defined(self):
        library = self.records["mixture"][1]["filenames"][0]
        self.symbols[library] += f"{TRAINER} U\n"
        self.check()

    def test_unclassified_online_symbol_is_not_absence(self):
        library = self.records["mixture"][1]["filenames"][0]
        self.symbols[library] += f"{TRAINER} ? 0\n"
        with self.assertRaisesRegex(PROBE.ProbeError, "could not classify symbol"):
            self.check()

    def test_legacy_and_v0_symbols_work_without_demangling(self):
        for symbol in (TRAINER, "_RNvMNtNtCs123_12lattice_fann8training4rlooNtB2_11RlooTrainer32step_batch_with_explored_actions"):
            library = self.records["router-learning"][1]["filenames"][0]
            self.symbols[library] = f"{symbol} T 00000000 30\n"
            with self.subTest(symbol=symbol):
                self.check()

    def test_empty_or_malformed_nm_output_refuses_absence(self):
        for output in ("", "lib.rlib[lib.rmeta]:\n", f"{TRAINER} U\n", "garbage\n", f"{TRAINER} T\n", f"{TRAINER} T nothex\n"):
            with self.subTest(output=output), self.assertRaises(PROBE.ProbeError):
                PROBE.defined_symbols(output, Path("fixture.rlib"))

    def test_cli_propagates_probe_status(self):
        with mock.patch.object(PROBE, "check", side_effect=PROBE.ProbeError("failure", 101)):
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                status = PROBE.main([])
        self.assertEqual(status, 101)
        self.assertIn("failure", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

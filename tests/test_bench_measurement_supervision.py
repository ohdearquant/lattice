#!/usr/bin/env python3
"""Tests for advisory measurement discovery and enforced local supervision."""

from __future__ import annotations

import ast
import fcntl
import os
import re
import shlex
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "scripts" / "bench-measurements.toml"
SPECIAL_ENTRYPOINTS = {
    "scripts/compare_logits.py",
    "scripts/e2e-parity-local.sh",
    "scripts/e2e_parity_check.py",
    "scripts/fake_quant_pilot.py",
    "scripts/perf-bench-gate.py",
    "scripts/perf_governor.py",
}
INTERNAL_OR_TEST = {
    "scripts/ensure-noindex-marker-selftest.sh",
    "scripts/lib/bench-compare-impl.sh",
}
MEASUREMENT_SIGNAL = re.compile(
    r"perf_counter(?:_ns)?\(|hrtime\.bigint\(|cargo bench|tok_per_sec|"
    r"elapsed_ns|elapsed_s|total_ms|tokens/s|tok/s|PPL:"
)
CARGO_BENCH_LITERAL = re.compile(r"\bcargo[ \t\r\n]+bench\b")
CARGO_BENCH_ARGV_LITERAL = re.compile(
    r'''["']cargo["']\s*,\s*(?:\[\s*)?["']bench["']'''
)
PYTHON_TIMING_CALLS = {
    "time.monotonic",
    "time.monotonic_ns",
    "time.perf_counter",
    "time.perf_counter_ns",
    "time.process_time",
    "time.process_time_ns",
    "time.time",
}
PYTHON_COMMAND_CALLS = {
    "Popen",
    "call",
    "check_call",
    "check_output",
    "os.popen",
    "os.system",
    "run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.getoutput",
    "subprocess.getstatusoutput",
    "subprocess.run",
    "system",
}
SCRIPT_EXTENSION_LANGUAGES = {
    ".bash": "shell",
    ".js": "node",
    ".mjs": "node",
    ".py": "python",
    ".sh": "shell",
}
SCRIPT_SHEBANG_LANGUAGES = {
    "#!/bin/bash": "shell",
    "#!/bin/sh": "shell",
    "#!/usr/bin/env bash": "shell",
    "#!/usr/bin/env node": "node",
    "#!/usr/bin/env python3": "python",
    "#!/usr/bin/node": "node",
    "#!/usr/bin/python3": "python",
}
SCRIPT_SUFFIXES = set(SCRIPT_EXTENSION_LANGUAGES)
EXPLICIT_SCRIPT_EXCLUSIONS = frozenset(
    {
        "scripts/bench-measurements.toml",
        "scripts/bench_decode_profiles.toml",
        "scripts/bench_evidence/pr882/CANONICAL_POLICY_IDENTITY.md",
        "scripts/bench_evidence/pr882/POLICY_SHA_TRANSITIONS.md",
        "scripts/bench_evidence/pr882/report_ctx1024.json",
        "scripts/bench_evidence/pr882/report_ctx512.json",
        "scripts/bench_expected_cells.toml",
        "scripts/lib/bench-host-id.py",
        "scripts/lib/bench-informational-targets.sh",
        "scripts/lib/bench-locks.py",
        "scripts/lib/bench-quick-informational-targets.txt",
        "scripts/lib/bench-supervision.sh",
        "scripts/lib/bench_supervision.py",
        "scripts/lib/ensure-noindex-marker.sh",
        "scripts/lib/machine-state-probe.py",
        "scripts/lib/quiet-probe.py",
        "scripts/perf-policy.toml",
        "scripts/perf_governor.README.md",
    }
    | INTERNAL_OR_TEST
)


@dataclass(frozen=True)
class ScriptDecision:
    state: str
    reason: str
    evidence: frozenset[str] = frozenset()


def manifest_entries() -> dict[str, dict[str, str]]:
    data = tomllib.loads(MANIFEST.read_text())
    return {entry["path"]: entry for entry in data["entry"]}


def excluded_measurement_surfaces() -> set[str]:
    data = tomllib.loads(MANIFEST.read_text())
    return {
        path
        for surface in data["excluded_surface"]
        for path in surface["paths"]
    }


def discovered_declared_rust_inventory_paths() -> set[str]:
    paths = {
        str(path.relative_to(REPO))
        for path in (REPO / "crates").glob("*/benches/*.rs")
    }
    paths.update(
        str(path.relative_to(REPO))
        for path in (REPO / "crates/inference/examples").glob("bench*.rs")
    )
    paths.update(
        str(path.relative_to(REPO))
        for path in (REPO / "crates/inference/src/bin").glob("*.rs")
        if path.stem.startswith("bench_")
        or path.stem in {"eval_perplexity", "gramperf_profile", "ppl_metal"}
    )
    paths.add("README.md")
    return paths


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _literal_strings(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [value for element in node.elts for value in _literal_strings(element)]
    return []


def _contains_literal_cargo_bench(node: ast.AST) -> bool:
    values = _literal_strings(node)
    return bool(values and CARGO_BENCH_LITERAL.search(" ".join(values)))


def _python_measurement_evidence(source: str, path: Path) -> set[str]:
    tree = ast.parse(source, filename=str(path))
    evidence: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        if any(module.split(".", 1)[0] in {"mlx", "mlx_lm"} for module in modules):
            evidence.add("MLX runtime import")

        if isinstance(node, ast.Attribute) and _dotted_name(node) in PYTHON_TIMING_CALLS:
            evidence.add("wall or monotonic timer")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "/api/generate" in node.value
        ):
            evidence.add("generation timing API")
        if isinstance(node, (ast.List, ast.Tuple)) and _contains_literal_cargo_bench(
            node
        ):
            evidence.add("cargo bench command")
        if isinstance(node, ast.Call) and _dotted_name(node.func) in PYTHON_COMMAND_CALLS:
            command_nodes = list(node.args[:1])
            command_nodes.extend(
                keyword.value
                for keyword in node.keywords
                if keyword.arg in {"args", "cmd", "command"}
            )
            if any(_contains_literal_cargo_bench(value) for value in command_nodes):
                evidence.add("cargo bench command")
    return evidence


def _shell_or_node_measurement_evidence(source: str) -> set[str]:
    evidence: set[str] = set()
    if CARGO_BENCH_LITERAL.search(source) or CARGO_BENCH_ARGV_LITERAL.search(source):
        evidence.add("cargo bench command")
    if re.search(
        r"(?m)^[ \t]*(?:bench_supervise_entry\b|"
        r"exec[ \t]+python3[^\n]*bench_supervision\.py[^\n]*[ \t]run\b)",
        source,
    ):
        evidence.add("measurement supervisor invocation")
    if re.search(r"process\.hrtime\.bigint\(|performance\.now\(|Date\.now\(", source):
        evidence.add("JavaScript timer")
    if re.search(r"(?m)^[ \t]*(?:from[ \t]+mlx|import[ \t]+mlx)", source):
        evidence.add("MLX runtime import")
    return evidence


def _top_level_python_calls(source: str, path: Path, name: str) -> list[ast.Call]:
    tree = ast.parse(source, filename=str(path))
    calls: list[ast.Call] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Call(self, node: ast.Call) -> None:
            if _dotted_name(node.func) == name:
                calls.append(node)
            self.generic_visit(node)

    Visitor().visit(tree)
    return calls


def _shell_commands(source: str) -> list[tuple[int, list[str]]]:
    commands: list[tuple[int, list[str]]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        candidate = line.rstrip()
        if candidate.endswith("\\"):
            candidate = candidate[:-1]
        try:
            tokens = shlex.split(candidate, comments=True, posix=True)
        except ValueError:
            continue
        if tokens:
            commands.append((line_number, tokens))
    return commands


def _shell_command_argv(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens) and re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[index]
    ):
        index += 1
    return tokens[index:]


def _explicit_script_exclusion(relative: str) -> str | None:
    path = Path(relative)
    if "__pycache__" in path.parts and path.suffix == ".pyc":
        return "generated Python bytecode cache, not source"
    if relative not in EXPLICIT_SCRIPT_EXCLUSIONS:
        return None
    if relative.startswith("scripts/lib/"):
        return "explicit internal helper; supervised callers are classified separately"
    if relative in INTERNAL_OR_TEST:
        return "explicit policy self-test or internal measurement body"
    return "explicit non-executable policy, fixture, or documentation asset"


def _classify_script(path: Path, relative: str) -> ScriptDecision:
    exclusion = _explicit_script_exclusion(relative)
    if exclusion is not None:
        return ScriptDecision("excluded", exclusion)
    try:
        source = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as exc:
        detail = str(exc) or "exception carried no message"
        return ScriptDecision("undecidable", f"source read failed: {type(exc).__name__}: {detail}")

    lines = source.splitlines()
    first_line = lines[0] if lines else ""
    extension_language = SCRIPT_EXTENSION_LANGUAGES.get(path.suffix)
    shebang_language = None
    if first_line.startswith("#!"):
        shebang_language = SCRIPT_SHEBANG_LANGUAGES.get(first_line)
        if shebang_language is None:
            return ScriptDecision("undecidable", f"unsupported shebang {first_line!r}")
    if (
        extension_language is not None
        and shebang_language is not None
        and extension_language != shebang_language
    ):
        return ScriptDecision(
            "undecidable",
            f"extension selects {extension_language}, shebang selects {shebang_language}",
        )
    language = extension_language or shebang_language
    if language is None:
        return ScriptDecision(
            "undecidable",
            f"no supported extension or shebang (suffix {path.suffix or '<none>'!r})",
        )

    try:
        if language == "python":
            evidence = _python_measurement_evidence(source, path)
        else:
            evidence = _shell_or_node_measurement_evidence(source)
    except SyntaxError as exc:
        return ScriptDecision(
            "undecidable",
            f"Python syntax error at line {exc.lineno}: {exc.msg}",
        )
    if evidence:
        return ScriptDecision(
            "measurement",
            f"{language} analysis found direct measurement evidence",
            frozenset(evidence),
        )
    return ScriptDecision(
        "advisory-no-match",
        f"advisory {language} analysis found no direct measurement pattern",
        frozenset(
            {
                f"advisory {language} source analysis completed",
                "zero recognized direct measurement patterns",
            }
        ),
    )


def discovered_script_decisions(repo: Path | None = None) -> dict[str, ScriptDecision]:
    root = repo or REPO
    decisions: dict[str, ScriptDecision] = {}
    for path in sorted((root / "scripts").rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        decisions[relative] = _classify_script(path, relative)
    return decisions


def discovered_measurement_evidence(
    repo: Path | None = None,
) -> dict[str, set[str]]:
    """Collect advisory direct-driver matches without trusting manifest roles.

    The syntax-pattern method has no bounded recall for dynamic construction,
    indirect call graphs, aliases, wrappers, generated code, or shell expansion.
    Read, decode, language-selection, and parse errors fail closed. A successful
    lexical no-match is advisory and does not prove that a script never measures.
    """

    decisions = discovered_script_decisions(repo)
    undecidable = [
        f"{path}: {decision.reason}"
        for path, decision in decisions.items()
        if decision.state == "undecidable"
    ]
    if undecidable:
        raise AssertionError("undecidable script candidates:\n" + "\n".join(undecidable))
    return {
        path: set(decision.evidence)
        for path, decision in decisions.items()
        if decision.state == "measurement"
    }


def validate_direct_measurement_supervision() -> None:
    entries = manifest_entries()
    evidence_by_path = discovered_measurement_evidence()
    if not evidence_by_path:
        raise AssertionError("measurement evidence scan collected zero paths")
    failures: list[str] = []
    for path, evidence in evidence_by_path.items():
        detail = ", ".join(sorted(evidence))
        entry = entries.get(path)
        if entry is None:
            failures.append(f"{path}: unclassified {detail}")
        elif entry["role"] != "measurement":
            failures.append(f"{path}: source contains {detail}; role={entry['role']}")
        elif entry["supervision"] == "none":
            failures.append(f"{path}: source contains {detail}; supervision=none")
    if failures:
        raise AssertionError("\n".join(failures))


class InventoryContract(unittest.TestCase):
    def test_canonical_lock_paths_are_pinned(self):
        source = (REPO / "scripts" / "lib" / "bench-locks.py").read_text()
        self.assertRegex(
            source,
            r'(?m)^BENCH_WINDOW = "/tmp/lion-bench-window\.lock"$',
        )
        self.assertRegex(
            source,
            r'(?m)^GPU_LOCK = "/tmp/lion-metal-gpu-test\.lock"$',
        )

    def test_manifest_schema_is_explicit_and_nonduplicated(self):
        data = tomllib.loads(MANIFEST.read_text())
        self.assertEqual(data["schema"], 1)
        entries = data["entry"]
        paths = [entry["path"] for entry in entries]
        self.assertEqual(len(paths), len(set(paths)))
        for entry in entries:
            with self.subTest(path=entry["path"]):
                self.assertIn(
                    entry["role"],
                    {"measurement", "consumer", "policy-check", "supervisor"},
                )
                self.assertIn(
                    entry["supervision"],
                    {
                        "none",
                        "both-locks",
                        "both-locks+quiet",
                        "both-locks+quiet-baseline",
                        "both-locks+three-phase-quiet",
                    },
                )
                self.assertTrue((REPO / entry["path"]).is_file())
                if entry["role"] == "measurement":
                    self.assertNotEqual(entry["supervision"], "none")

    def test_declared_rust_inventory_grammar_is_exact_and_fail_closed(self):
        """A Rust path matching the declared grammar must be classified."""

        data = tomllib.loads(MANIFEST.read_text())
        contract = data["contract"]
        self.assertEqual(contract["caller_trust"], "cooperative")
        self.assertEqual(
            contract["handoff_check"],
            "instantaneous silent-pipe open-writer and lock-contention "
            "diagnostics; not authenticated ownership, continuous lock-lifetime "
            "proof, or deliberate same-user bypass resistance",
        )
        self.assertEqual(
            contract["rust_inventory_grammar"],
            "crates/*/benches/*.rs; crates/inference/examples/bench*.rs; "
            "crates/inference/src/bin/bench_*.rs plus eval_perplexity.rs, "
            "gramperf_profile.rs, and ppl_metal.rs; README.md",
        )
        self.assertEqual(
            contract["rust_inventory_limitation"],
            "does not discover other Rust examples, binaries, or tests",
        )
        confirmed_outside = {
            "crates/inference/examples/profile_metal_decode.rs",
            "crates/inference/examples/profile_metal.rs",
            "crates/inference/examples/decode_profile.rs",
            "crates/inference/examples/layer_sweep.rs",
            "crates/tune/tests/bench_backward_737.rs",
        }
        self.assertEqual(
            set(contract["confirmed_outside_rust_inventory"]), confirmed_outside
        )
        self.assertTrue(
            confirmed_outside.isdisjoint(discovered_declared_rust_inventory_paths())
        )
        for path in confirmed_outside:
            self.assertTrue((REPO / path).is_file(), path)
        surfaces = data["excluded_surface"]
        self.assertTrue(surfaces)
        for surface in surfaces:
            with self.subTest(family=surface["family"]):
                self.assertRegex(surface["tracking_issue"], r"^#[1-9][0-9]*$")
                self.assertTrue(surface["reason"])
                self.assertTrue(surface["paths"])
                for path in surface["paths"]:
                    self.assertTrue((REPO / path).is_file(), path)
        excluded = excluded_measurement_surfaces()
        self.assertEqual(len(excluded), sum(len(s["paths"]) for s in surfaces))
        self.assertEqual(excluded, discovered_declared_rust_inventory_paths())

    def test_every_benchmark_named_script_is_classified(self):
        """A new bench script cannot appear without an explicit classification."""

        discovered = {
            str(path.relative_to(REPO))
            for path in (REPO / "scripts").iterdir()
            if path.is_file()
            and path.name.startswith("bench")
            and path.suffix in SCRIPT_SUFFIXES
        }
        discovered.update(SPECIAL_ENTRYPOINTS)
        discovered.update(
            str(path.relative_to(REPO))
            for path in (REPO / "scripts").rglob("*")
            if path.is_file()
            and path.suffix in SCRIPT_SUFFIXES
            and MEASUREMENT_SIGNAL.search(path.read_text())
        )
        discovered.difference_update(INTERNAL_OR_TEST)
        self.assertEqual(set(manifest_entries()), discovered)

    def test_direct_measurement_evidence_requires_supervision(self):
        """Source evidence, not manifest claims, determines this requirement."""

        validate_direct_measurement_supervision()

    def test_no_shebang_javascript_measurement_is_discovered(self):
        """Explicit Node invocation does not require a shebang or executable bit."""

        fixture_repo = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scripts = fixture_repo / "scripts"
        scripts.mkdir()
        fixture = scripts / "bench_no_shebang.js"
        fixture.write_text(
            'require("node:child_process").execSync("cargo bench -p lattice-inference");\n'
        )

        evidence = discovered_measurement_evidence(fixture_repo)
        self.assertEqual(
            evidence["scripts/bench_no_shebang.js"], {"cargo bench command"}
        )

    def test_node_argv_cargo_bench_detector_is_mutation_guarded(self):
        fixture_repo = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scripts = fixture_repo / "scripts"
        scripts.mkdir()
        fixture = scripts / "bench_node_argv.js"
        fixture.write_text(
            "const {spawnSync} = require('node:child_process');\n"
            "spawnSync('cargo', ['bench', '-p', 'lattice-inference']);\n"
        )

        evidence = discovered_measurement_evidence(fixture_repo)
        self.assertEqual(
            evidence["scripts/bench_node_argv.js"], {"cargo bench command"}
        )

    def test_no_shebang_bash_measurement_is_discovered(self):
        """Explicit Bash invocation does not require a shebang or executable bit."""

        fixture_repo = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scripts = fixture_repo / "scripts"
        scripts.mkdir()
        fixture = scripts / "bench_no_shebang.bash"
        fixture.write_text('bash -lc "cargo bench -p lattice-inference"\n')

        evidence = discovered_measurement_evidence(fixture_repo)
        self.assertEqual(
            evidence["scripts/bench_no_shebang.bash"], {"cargo bench command"}
        )

    def test_literal_shell_benchmark_consumer_fails_real_supervision(self):
        """A literal subprocess shell command cannot be declared a consumer."""

        fixture_repo = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scripts = fixture_repo / "scripts"
        scripts.mkdir()
        fixture = scripts / "bench_literal_shell.py"
        fixture.write_text(
            "#!/usr/bin/env python3\n"
            "import subprocess\n"
            'subprocess.run("cargo bench -p lattice-inference", shell=True)\n'
        )
        manifest = scripts / "bench-measurements.toml"
        manifest.write_text(
            "[[entry]]\n"
            'path = "scripts/bench_literal_shell.py"\n'
            'role = "consumer"\n'
            'supervision = "none"\n'
        )
        self.enterContext(mock.patch.object(sys.modules[__name__], "REPO", fixture_repo))
        self.enterContext(mock.patch.object(sys.modules[__name__], "MANIFEST", manifest))

        with self.assertRaises(AssertionError) as caught:
            self.test_direct_measurement_evidence_requires_supervision()
        self.assertIn("scripts/bench_literal_shell.py", str(caught.exception))
        self.assertIn("cargo bench command", str(caught.exception))

    def test_literal_shell_benchmark_call_forms_are_detected(self):
        cases = {
            "Popen": 'Popen("cargo bench -p lattice-inference", shell=True)',
            "call": 'call("cargo bench -p lattice-inference", shell=True)',
            "check_call": (
                'check_call("cargo bench -p lattice-inference", shell=True)'
            ),
            "check_output": (
                'check_output("cargo bench -p lattice-inference", shell=True)'
            ),
            "os.popen": 'os.popen("cargo bench -p lattice-inference")',
            "os.system": 'os.system("cargo bench -p lattice-inference")',
            "run": 'run("cargo bench -p lattice-inference", shell=True)',
            "subprocess.Popen": (
                'subprocess.Popen("cargo bench -p lattice-inference", shell=True)'
            ),
            "subprocess.call": (
                'subprocess.call("cargo bench -p lattice-inference", shell=True)'
            ),
            "subprocess.check_call": (
                'subprocess.check_call("cargo bench -p lattice-inference", shell=True)'
            ),
            "subprocess.check_output": (
                'subprocess.check_output("cargo bench -p lattice-inference", shell=True)'
            ),
            "subprocess.getoutput": (
                'subprocess.getoutput("cargo bench -p lattice-inference")'
            ),
            "subprocess.getstatusoutput": (
                'subprocess.getstatusoutput("cargo bench -p lattice-inference")'
            ),
            "subprocess.run": (
                'subprocess.run("cargo bench -p lattice-inference", shell=True)'
            ),
            "system": 'system("cargo bench -p lattice-inference")',
            "shell_driver": (
                'subprocess.run(["bash", "-lc", "cargo bench -p lattice-inference"])'
            ),
        }
        self.assertEqual(set(cases) - {"shell_driver"}, PYTHON_COMMAND_CALLS)
        for name, call in cases.items():
            with self.subTest(name=name):
                evidence = _python_measurement_evidence(
                    f"import os\nimport subprocess\n{call}\n", Path(f"{name}.py")
                )
                self.assertIn("cargo bench command", evidence)

    def test_python_non_command_detectors_are_mutation_guarded(self):
        cases = {
            "mlx_import": ("import mlx.core\n", "MLX runtime import"),
            "mlx_from_import": (
                "from mlx_lm import load\n",
                "MLX runtime import",
            ),
            "generation_api": (
                'endpoint = "http://127.0.0.1/api/generate"\n',
                "generation timing API",
            ),
        }
        for call in sorted(PYTHON_TIMING_CALLS):
            cases[f"timer_{call}"] = (f"{call}()\n", "wall or monotonic timer")
        for name, (source, expected) in cases.items():
            with self.subTest(name=name):
                evidence = _python_measurement_evidence(source, Path(f"{name}.py"))
                self.assertIn(expected, evidence)

    def test_python_literal_sequence_detector_is_mutation_guarded(self):
        for name, source in {
            "list": 'command = ["cargo", "bench", "-p", "lattice-inference"]\n',
            "tuple": 'command = ("cargo", "bench", "-p", "lattice-inference")\n',
        }.items():
            with self.subTest(name=name):
                evidence = _python_measurement_evidence(source, Path(f"{name}.py"))
                self.assertIn("cargo bench command", evidence)

    def test_shell_and_node_auxiliary_detectors_are_mutation_guarded(self):
        cases = {
            "shell_supervisor_function": (
                'bench_supervise_entry "fixture" ordinary measure "$@"\n',
                "measurement supervisor invocation",
            ),
            "shell_supervisor_cli": (
                "exec python3 scripts/lib/bench_supervision.py run --label fixture -- true\n",
                "measurement supervisor invocation",
            ),
            "shell_mlx_import": ("import mlx.core\n", "MLX runtime import"),
            "shell_mlx_from_import": (
                "from mlx_lm import load\n",
                "MLX runtime import",
            ),
            "node_hrtime": (
                "const start = process.hrtime.bigint();\n",
                "JavaScript timer",
            ),
            "node_performance": (
                "const start = performance.now();\n",
                "JavaScript timer",
            ),
            "node_date": ("const start = Date.now();\n", "JavaScript timer"),
        }
        for name, (source, expected) in cases.items():
            with self.subTest(name=name):
                self.assertIn(expected, _shell_or_node_measurement_evidence(source))

    def test_dynamic_python_command_construction_is_advisory_no_match(self):
        fixture_repo = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scripts = fixture_repo / "scripts"
        scripts.mkdir()
        fixture = scripts / "dynamic_dispatch.py"
        fixture.write_text(
            "import subprocess\n"
            "cargo = ''.join(chr(value) for value in (99, 97, 114, 103, 111))\n"
            "bench = ''.join(chr(value) for value in (98, 101, 110, 99, 104))\n"
            "command = [cargo, bench, '-p', 'lattice-inference']\n"
            "target = ''.join(chr(value) for value in (114, 117, 110))\n"
            "getattr(subprocess, target)(command, check=True)\n"
        )

        decision = _classify_script(fixture, "scripts/dynamic_dispatch.py")
        self.assertEqual(decision.state, "advisory-no-match")
        self.assertIn("advisory", decision.reason)

    def test_undecidable_input_fails_real_supervision_with_reason(self):
        """An unanalyzable candidate is a named failure, never a negative."""

        fixture_repo = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scripts = fixture_repo / "scripts"
        scripts.mkdir()
        fixture = scripts / "bench_unknown.rb"
        fixture.write_text('system("cargo bench -p lattice-inference")\n')
        manifest = scripts / "bench-measurements.toml"
        manifest.write_text(
            "[[entry]]\n"
            'path = "scripts/bench_unknown.rb"\n'
            'role = "consumer"\n'
            'supervision = "none"\n'
        )
        self.enterContext(mock.patch.object(sys.modules[__name__], "REPO", fixture_repo))
        self.enterContext(mock.patch.object(sys.modules[__name__], "MANIFEST", manifest))

        with self.assertRaises(AssertionError) as caught:
            self.test_direct_measurement_evidence_requires_supervision()
        self.assertIn("scripts/bench_unknown.rb", str(caught.exception))
        self.assertIn("no supported extension or shebang", str(caught.exception))

        with mock.patch.object(Path, "read_bytes", side_effect=OSError()):
            decision = _classify_script(
                Path("scripts/bench_unreadable.py"),
                "scripts/bench_unreadable.py",
            )
        self.assertEqual(decision.state, "undecidable")
        self.assertIn("exception carried no message", decision.reason)

    def test_empty_measurement_scan_fails_real_supervision(self):
        with mock.patch.object(
            sys.modules[__name__], "discovered_measurement_evidence", return_value={}
        ):
            with self.assertRaisesRegex(
                AssertionError, "measurement evidence scan collected zero paths"
            ):
                validate_direct_measurement_supervision()

    def test_unclassified_measurement_fails_real_supervision(self):
        evidence = {"scripts/new_measurement.py": {"wall or monotonic timer"}}
        with (
            mock.patch.object(sys.modules[__name__], "manifest_entries", return_value={}),
            mock.patch.object(
                sys.modules[__name__],
                "discovered_measurement_evidence",
                return_value=evidence,
            ),
        ):
            with self.assertRaisesRegex(
                AssertionError,
                "scripts/new_measurement.py: unclassified wall or monotonic timer",
            ):
                validate_direct_measurement_supervision()

    def test_measurement_with_no_supervision_fails_real_supervision(self):
        path = "scripts/unsupervised_measurement.py"
        entries = {path: {"role": "measurement", "supervision": "none"}}
        evidence = {path: {"wall or monotonic timer"}}
        with (
            mock.patch.object(
                sys.modules[__name__], "manifest_entries", return_value=entries
            ),
            mock.patch.object(
                sys.modules[__name__],
                "discovered_measurement_evidence",
                return_value=evidence,
            ),
        ):
            with self.assertRaisesRegex(
                AssertionError,
                "scripts/unsupervised_measurement.py: source contains wall or monotonic "
                "timer; supervision=none",
            ):
                validate_direct_measurement_supervision()

    def test_invalid_python_and_unsupported_shebang_are_undecidable(self):
        cases = {
            "scripts/bench_invalid.py": (
                "#!/usr/bin/env python3\nif True print('bad')\n",
                "Python syntax error",
            ),
            "scripts/bench_ruby.py": (
                "#!/usr/bin/env ruby\nputs 'cargo bench'\n",
                "unsupported shebang",
            ),
            "scripts/bench_node.py": (
                '#!/usr/bin/env node\nconsole.log("cargo bench")\n',
                "extension selects python, shebang selects node",
            ),
        }
        for relative, (source, reason) in cases.items():
            with self.subTest(path=relative):
                fixture_repo = Path(self.enterContext(tempfile.TemporaryDirectory()))
                fixture = fixture_repo / relative
                fixture.parent.mkdir(parents=True)
                fixture.write_text(source)
                with self.assertRaises(AssertionError) as caught:
                    discovered_measurement_evidence(fixture_repo)
                self.assertIn(relative, str(caught.exception))
                self.assertIn(reason, str(caught.exception))

    def test_script_discovery_records_every_file_decision(self):
        decisions = discovered_script_decisions()
        searched = {
            str(path.relative_to(REPO))
            for path in (REPO / "scripts").rglob("*")
            if path.is_file()
        }
        self.assertEqual(set(decisions), searched)
        for path, decision in decisions.items():
            with self.subTest(path=path):
                self.assertIn(
                    decision.state,
                    {
                        "measurement",
                        "advisory-no-match",
                        "excluded",
                        "undecidable",
                    },
                )
                self.assertTrue(decision.reason)
                if decision.state in {"measurement", "advisory-no-match"}:
                    self.assertTrue(decision.evidence)

    def test_every_measurement_entry_has_a_live_guard(self):
        """Mutation-sensitive: deleting any entry-point guard fails this scan."""

        for path, entry in manifest_entries().items():
            if entry["role"] != "measurement":
                continue
            source_path = REPO / path
            source = source_path.read_text()
            with self.subTest(path=path):
                if path == "scripts/bench-compare.sh":
                    commands = _shell_commands(source)
                    self.assertTrue(
                        any(
                            tokens[0] == "exec"
                            and any("bench_supervision.py" in token for token in tokens)
                            for _, tokens in commands
                        )
                    )
                elif path.endswith(".py"):
                    calls = _top_level_python_calls(
                        source, source_path, "ensure_python_entrypoint"
                    )
                    self.assertTrue(calls)
                    if entry["supervision"] == "both-locks+quiet":
                        self.assertTrue(
                            any(
                                any(
                                    keyword.arg == "quiet"
                                    and isinstance(keyword.value, ast.Constant)
                                    and keyword.value.value is True
                                    for keyword in call.keywords
                                )
                                for call in calls
                            )
                        )
                elif path.endswith(".mjs"):
                    self.assertIn("bench_supervision.py", source)
                    self.assertIn("'verify'", source)
                else:
                    commands = _shell_commands(source)
                    self.assertTrue(
                        any(
                            tokens[0] == "bench_supervise_entry"
                            for _, tokens in commands
                        )
                    )

    def test_make_delegates_whole_durable_recipes(self):
        """The lock must cover the recipe, not one command inside Make quoting."""

        makefile = (REPO / "Makefile").read_text()
        for target, script in (
            ("bench-ci", "./scripts/bench-ci.sh"),
            ("bench-gate", "./scripts/bench-gate.sh"),
        ):
            match = re.search(
                rf"(?m)^{re.escape(target)}:\n(?P<body>(?:\t.*\n)+)",
                makefile,
            )
            self.assertIsNotNone(match, target)
            body = [
                line.strip()
                for line in match.group("body").splitlines()
                if line.strip()
            ]
            self.assertEqual(body, [script])

    def test_make_has_no_raw_cargo_bench_recipe(self):
        """A new Make measurement must enter through a supervised script."""

        recipes = [
            line.strip()
            for line in (REPO / "Makefile").read_text().splitlines()
            if line.startswith("\t")
        ]
        self.assertFalse(
            [line for line in recipes if re.search(r"\bcargo\s+bench\b", line)]
        )

    def test_durable_multi_target_recipes_probe_between_targets(self):
        """Mutation-sensitive: the outer before/after probes are not a midpoint."""

        for path in ("scripts/bench-ci.sh", "scripts/bench-gate.sh"):
            source = (REPO / path).read_text()
            commands = _shell_commands(source)
            with self.subTest(path=path):
                supervisors = [
                    tokens
                    for _, tokens in commands
                    if tokens[0] == "bench_supervise_entry"
                ]
                self.assertTrue(supervisors)
                self.assertTrue(any("durable" in tokens for tokens in supervisors))
                checkpoints = [
                    (line_number, tokens)
                    for line_number, tokens in commands
                    if tokens[0] == "bench_quiet_checkpoint"
                ]
                self.assertGreaterEqual(len(checkpoints), 2)
                self.assertTrue(
                    any("between targets" in " ".join(tokens) for _, tokens in checkpoints)
                )
                measurements = [
                    line_number
                    for line_number, tokens in commands
                    if _shell_command_argv(tokens)[:2] == ["cargo", "bench"]
                ]
                self.assertTrue(measurements)
                final_probe = max(line_number for line_number, _ in checkpoints)
                final_measurement = max(measurements)
                self.assertGreater(final_probe, final_measurement)
                if path == "scripts/bench-gate.sh":
                    gate_calls = [
                        line_number
                        for line_number, tokens in commands
                        if _shell_command_argv(tokens)[:2]
                        == ["python3", "scripts/perf-bench-gate.py"]
                    ]
                    self.assertTrue(gate_calls)
                    self.assertLess(
                        final_probe,
                        min(gate_calls),
                    )

    def test_fake_quant_does_not_nest_its_old_gpu_only_lock(self):
        """The outer both-lock supervisor and an inner flock would deadlock."""

        source = (REPO / "scripts" / "fake_quant_pilot.py").read_text()
        self.assertNotIn("fcntl.flock", source)
        self.assertNotIn("GPU_LOCK_PATH", source)

    def test_node_measurement_forwards_pipe_to_handoff_sample(self):
        """Node closes extra fds unless the handoff sample maps the pipe."""

        source = (REPO / "scripts" / "bench_wasm_simd.mjs").read_text()
        self.assertIn("LATTICE_BENCH_SUPERVISOR_FD", source)
        self.assertNotIn("LATTICE_BENCH_LOCK_FDS", source)
        self.assertIn("'--require-quiet'", source)
        self.assertIn("stdio: supervisionStdio()", source)
        self.assertIn("closeSync(SUPERVISOR_FD)", source)
        self.assertIn("delete process.env.LATTICE_BENCH_SUPERVISOR_FD", source)
        self.assertGreaterEqual(source.count("[SUPERVISION, 'verify'"), 2)
        self.assertNotIn("verify-retained", source)

    def test_cooperating_shell_entrypoints_recheck_handoff(self):
        for path in (
            "scripts/lib/bench-supervision.sh",
            "scripts/lib/bench-compare-impl.sh",
        ):
            with self.subTest(path=path):
                source = (REPO / path).read_text()
                # The interpreter may be a literal `python3` or a resolved
                # interpreter variable (e.g. `"$python_bin"` from
                # bench-python.sh's version-floor resolver) -- either way,
                # the helper's `verify` subcommand must be re-invoked.
                self.assertGreaterEqual(
                    len(
                        re.findall(
                            r'(?:python3|"\$[A-Za-z_]+") "[^"]+" verify', source
                        )
                    ),
                    2,
                )
                self.assertNotIn("verify-retained", source)


class _SupervisorSandbox:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()

    def __enter__(self):
        self.root = Path(self.tmp.name) / "repo"
        lib = self.root / "scripts" / "lib"
        lib.mkdir(parents=True)
        for name in (
            "bench_supervision.py",
            "bench-locks.py",
            "quiet-probe.py",
            "bench-python.sh",
        ):
            shutil.copy2(REPO / "scripts" / "lib" / name, lib / name)
        shutil.copy2(
            REPO / "scripts" / "lib" / "bench-supervision.sh",
            lib / "bench-supervision.sh",
        )

        self.bench_lock = Path(self.tmp.name) / "bench-window.lock"
        self.gpu_lock = Path(self.tmp.name) / "metal-gpu.lock"
        self.pending = Path(self.tmp.name) / "pending"
        lock_source = (lib / "bench-locks.py").read_text()
        replacements = {
            "BENCH_WINDOW": str(self.bench_lock),
            "GPU_LOCK": str(self.gpu_lock),
            "PENDING_DIR": str(self.pending),
        }
        for name, value in replacements.items():
            lock_source = re.sub(
                rf'^{name} = "[^"]*"$',
                f'{name} = "{value}"',
                lock_source,
                flags=re.M,
            )
        (lib / "bench-locks.py").write_text(lock_source)
        self.helper = lib / "bench_supervision.py"
        return self

    def __exit__(self, *exc):
        self.tmp.cleanup()
        return False

    def run(
        self,
        command: list[str],
        *,
        entrypoint: bool = False,
        **env: str,
    ) -> subprocess.CompletedProcess[str]:
        entrypoint_arg = ["--entrypoint"] if entrypoint else []
        return subprocess.run(
            [
                sys.executable,
                str(self.helper),
                "run",
                "--label",
                "fixture",
                *entrypoint_arg,
                "--",
                *command,
            ],
            capture_output=True,
            text=True,
            env={**os.environ, **env},
            timeout=30,
        )


def _run_bench_ci_fixture(*, fail_first: bool) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    with _SupervisorSandbox() as sb:
        entrypoint = sb.root / "scripts" / "bench-ci.sh"
        shutil.copy2(REPO / "scripts" / "bench-ci.sh", entrypoint)
        quiet_probe = sb.root / "scripts" / "lib" / "quiet-probe.py"
        quiet_probe.write_text("raise SystemExit(0)\n")

        bindir = Path(sb.tmp.name) / "bin"
        bindir.mkdir()
        calls = Path(sb.tmp.name) / "cargo-calls"
        cargo = bindir / "cargo"
        cargo.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "if [[ \" $* \" == *\" lattice-inference \"* ]]; then\n"
            f"  printf '%s\\n' inference >> {str(calls)!r}\n"
            "  if [[ \"${FIXTURE_FAIL_FIRST:-0}\" == 1 ]]; then\n"
            "    printf '%s\\n' 'fixture first target failed' >&2\n"
            "    exit 7\n"
            "  fi\n"
            "else\n"
            f"  printf '%s\\n' embed >> {str(calls)!r}\n"
            "fi\n"
        )
        cargo.chmod(0o755)
        env = {
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "BENCH_IDLE_FLOOR": "0",
            "FIXTURE_FAIL_FIRST": "1" if fail_first else "0",
        }
        for name in (
            "LATTICE_BENCH_LOCK_STATUS",
            "LATTICE_BENCH_LOCK_FDS",
            "LATTICE_BENCH_SUPERVISOR_FD",
        ):
            env.pop(name, None)
        result = subprocess.run(
            ["bash", str(entrypoint)],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        recorded = calls.read_text().splitlines() if calls.exists() else []
        return result, recorded


def _run_wasm_fixture(*, prerequisites: bool) -> subprocess.CompletedProcess[str]:
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node is unavailable")

    with _SupervisorSandbox() as sb:
        entrypoint = sb.root / "scripts" / "bench_wasm_simd.mjs"
        shutil.copy2(REPO / "scripts" / "bench_wasm_simd.mjs", entrypoint)
        quiet_probe = sb.root / "scripts" / "lib" / "quiet-probe.py"
        quiet_probe.write_text("raise SystemExit(0)\n")

        bindir = Path(sb.tmp.name) / "bin"
        bindir.mkdir()
        (bindir / "python3").symlink_to(sys.executable)
        if prerequisites:
            for name, body in {
                "cargo": (
                    "#!/bin/sh\n"
                    "if [ \"${1:-}\" = --version ]; then echo 'cargo fixture'; fi\n"
                    "exit 0\n"
                ),
                "rustup": (
                    "#!/bin/sh\n"
                    "echo wasm32-unknown-unknown\n"
                ),
                "rustc": (
                    "#!/bin/sh\n"
                    "echo 'rustc fixture'\n"
                ),
            }.items():
                executable = bindir / name
                executable.write_text(body)
                executable.chmod(0o755)

            module = (
                "export function simdDotProduct() { return 0; }\n"
                "export function simdSquaredEuclideanDistance() { return 0; }\n"
                "export function simdCosineSimilarity() { return 0; }\n"
                "export function simdNormalize() {}\n"
            )
            bindgen = bindir / "wasm-bindgen"
            bindgen.write_text(
                f"#!{sys.executable}\n"
                "import sys\n"
                "from pathlib import Path\n"
                "if '--version' in sys.argv:\n"
                "    print('wasm-bindgen fixture')\n"
                "    raise SystemExit(0)\n"
                "out = Path(sys.argv[sys.argv.index('--out-dir') + 1])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "(out / 'package.json').write_text('{\"type\":\"module\"}\\n')\n"
                f"(out / 'lattice_embed.js').write_text({module!r})\n"
            )
            bindgen.chmod(0o755)

        env = {
            **os.environ,
            "PATH": str(bindir),
            "BENCH_IDLE_FLOOR": "0",
            "LATTICE_BENCH_WASM_SIMD_ENFORCE": "",
        }
        for name in (
            "LATTICE_BENCH_LOCK_STATUS",
            "LATTICE_BENCH_LOCK_FDS",
            "LATTICE_BENCH_SUPERVISOR_FD",
        ):
            env.pop(name, None)
        return subprocess.run(
            [
                sys.executable,
                str(sb.helper),
                "run",
                "--label",
                "fixture",
                "--quiet",
                "--entrypoint",
                "--",
                node,
                str(entrypoint),
                "--dims",
                "1",
                "--reps",
                "1",
                "--warmup",
                "0",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )


class RuntimeContract(unittest.TestCase):
    def test_first_failed_bench_ci_target_refuses_before_later_targets(self):
        result, calls = _run_bench_ci_fixture(fail_first=True)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(calls, ["inference"])
        self.assertIn("fixture first target failed", result.stderr)
        self.assertIn("NOT MEASURABLE: lattice-inference benchmark failed", result.stderr)

    def test_bench_ci_healthy_targets_complete(self):
        result, calls = _run_bench_ci_fixture(fail_first=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, ["inference", "embed"])

    def test_recipe_outcome_one_is_not_reclassified_as_supervision_failure(self):
        with _SupervisorSandbox() as sb:
            entrypoint = sb.root / "scripts" / "outcome_entrypoint.sh"
            entrypoint.write_text(
                "#!/usr/bin/env bash\n"
                "set -uo pipefail\n"
                f"source {str(sb.root / 'scripts/lib/bench-supervision.sh')!r}\n"
                "measurement() { return 1; }\n"
                'bench_supervise_entry "fixture" ordinary measurement "$@"\n'
            )
            entrypoint.chmod(0o755)
            result = sb.run([str(entrypoint)], entrypoint=True)

        self.assertEqual(result.returncode, 1, result.stderr)

    def test_successful_parent_with_live_descendant_refuses_before_unlock(self):
        with _SupervisorSandbox() as sb:
            pid_file = Path(sb.tmp.name) / "grandchild-pid"
            grandchild = "import time; time.sleep(30)"
            parent = (
                "import subprocess, sys; from pathlib import Path; "
                f"child=subprocess.Popen([sys.executable, '-c', {grandchild!r}], "
                "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
                "stderr=subprocess.DEVNULL); "
                f"Path({str(pid_file)!r}).write_text(str(child.pid))"
            )
            result = sb.run([sys.executable, "-c", parent])
            child_pid = int(pid_file.read_text())
            try:
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("live process-group descendants", result.stderr)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                for path in (sb.bench_lock, sb.gpu_lock):
                    fd = os.open(path, os.O_RDWR)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        os.close(fd)
            finally:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_successful_parent_without_descendants_completes(self):
        with _SupervisorSandbox() as sb:
            result = sb.run([sys.executable, "-c", "raise SystemExit(0)"])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_wasm_prerequisite_is_not_measurable(self):
        result = _run_wasm_fixture(prerequisites=False)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("NOT MEASURABLE", result.stderr)
        self.assertIn("cargo not found on PATH", result.stderr)

    def test_wasm_fixture_with_prerequisites_completes(self):
        result = _run_wasm_fixture(prerequisites=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("| dot_product | 1 |", result.stdout)

    def test_command_waits_for_each_machine_wide_lock(self):
        """Mutation-sensitive: dropping either acquire lets its subtest run early."""

        for which in ("bench", "gpu"):
            with self.subTest(lock=which), _SupervisorSandbox() as sb:
                held_path = sb.bench_lock if which == "bench" else sb.gpu_lock
                held_path.touch()
                marker = Path(sb.tmp.name) / f"{which}.ran"
                with held_path.open("r+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    proc = subprocess.Popen(
                        [
                            sys.executable,
                            str(sb.helper),
                            "run",
                            "--label",
                            "fixture",
                            "--",
                            sys.executable,
                            "-c",
                            f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    time.sleep(0.35)
                    self.assertIsNone(proc.poll())
                    self.assertFalse(marker.exists())
                    fcntl.flock(lock, fcntl.LOCK_UN)
                stdout, stderr = proc.communicate(timeout=30)
                self.assertEqual(proc.returncode, 0, f"{stdout}\n{stderr}")
                self.assertEqual(marker.read_text(), "ran")

    def test_durable_run_refuses_before_command_on_busy_machine(self):
        """Mutation-sensitive: warning instead of refusal creates the marker."""

        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "should-not-run"
            result = subprocess.run(
                [
                    sys.executable,
                    str(sb.helper),
                    "run",
                    "--label",
                    "fixture",
                    "--quiet",
                    "--",
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "BENCH_IDLE_FLOOR": "101"},
                timeout=30,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(marker.exists())
            self.assertIn("refusing to measure", result.stderr)

    def test_unsupervised_command_runs_with_both_lock_receipts(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "receipt"
            code = (
                "import os; from pathlib import Path; "
                "p=Path(os.environ['LATTICE_BENCH_LOCK_STATUS']); "
                f"Path({str(marker)!r}).write_text(p.read_text())"
            )
            result = sb.run([sys.executable, "-c", code])
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = marker.read_text()
            self.assertIn("bench-window", receipt)
            self.assertIn("Metal GPU", receipt)

    def test_arbitrary_command_does_not_inherit_lock_capabilities(self):
        """A raw Cargo command must not leak lock fds into build daemons."""

        with _SupervisorSandbox() as sb:
            code = (
                "import os,sys; "
                "sys.exit(1 if 'LATTICE_BENCH_LOCK_FDS' in os.environ else 0)"
            )
            result = sb.run([sys.executable, "-c", code])
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_entrypoint_cannot_unlock_supervisor_locks_during_measurement(self):
        """LOCK_UN in the child must not reach the supervisor's open files."""

        with _SupervisorSandbox() as sb:
            ready = Path(sb.tmp.name) / "unlock-attempted"
            release = Path(sb.tmp.name) / "release"
            marker = Path(sb.tmp.name) / "measurement-ran"
            entrypoint = sb.root / "scripts" / "unlock_entrypoint.py"
            entrypoint.write_text(
                "import fcntl, os, time\n"
                "from pathlib import Path\n"
                "for fd in range(3, 256):\n"
                "    try:\n"
                "        os.fstat(fd)\n"
                "        fcntl.flock(fd, fcntl.LOCK_UN)\n"
                "    except (OSError, OverflowError):\n"
                "        pass\n"
                f"Path({str(ready)!r}).write_text('attempted')\n"
                f"release = Path({str(release)!r})\n"
                "while not release.exists():\n"
                "    time.sleep(0.01)\n"
                f"Path({str(marker)!r}).write_text('ran')\n"
            )
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(sb.helper),
                    "run",
                    "--label",
                    "fixture",
                    "--entrypoint",
                    "--",
                    sys.executable,
                    str(entrypoint),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and proc.poll() is None:
                    if time.monotonic() >= deadline:
                        self.fail("entrypoint did not attempt LOCK_UN")
                    time.sleep(0.01)
                self.assertIsNone(proc.poll())
                for path in (sb.bench_lock, sb.gpu_lock):
                    fd = os.open(path, os.O_RDWR)
                    try:
                        with self.assertRaises(OSError):
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        os.close(fd)
            finally:
                release.touch()
                stdout, stderr = proc.communicate(timeout=30)
            self.assertEqual(proc.returncode, 0, f"{stdout}\n{stderr}")
            self.assertEqual(marker.read_text(), "ran")

    def test_sigkill_of_supervisor_reaps_measurement_before_unlock(self):
        """An orphaned measurement must not outlive the held lock window."""

        with _SupervisorSandbox() as sb:
            ready = Path(sb.tmp.name) / "measurement-ready"
            release = Path(sb.tmp.name) / "release"
            entrypoint = sb.root / "scripts" / "sigkill_entrypoint.py"
            entrypoint.write_text(
                "import os, sys, time\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import sample_measurement_handoff\n"
                "_, pipe_fd, _ = sample_measurement_handoff()\n"
                "os.close(pipe_fd)\n"
                f"Path({str(ready)!r}).write_text(f'{{os.getppid()}},{{os.getpid()}}')\n"
                f"release = Path({str(release)!r})\n"
                "while not release.exists():\n"
                "    time.sleep(0.01)\n"
            )

            def lock_states() -> list[str]:
                states = []
                for path in (sb.bench_lock, sb.gpu_lock):
                    fd = os.open(path, os.O_RDWR)
                    try:
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except OSError:
                            states.append("blocked")
                        else:
                            states.append("acquired")
                            fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        os.close(fd)
                return states

            outer = subprocess.Popen(
                [
                    sys.executable,
                    str(sb.helper),
                    "run",
                    "--label",
                    "fixture",
                    "--entrypoint",
                    "--",
                    sys.executable,
                    str(entrypoint),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            measurement_pid = None
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and outer.poll() is None:
                    if time.monotonic() >= deadline:
                        self.fail("measurement did not reach the handoff barrier")
                    time.sleep(0.01)
                self.assertIsNone(outer.poll())
                supervisor_pid, measurement_pid = map(
                    int, ready.read_text().split(",")
                )
                self.assertEqual(lock_states(), ["blocked", "blocked"])

                os.kill(supervisor_pid, signal.SIGKILL)
                outer.wait(timeout=30)

                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.kill(measurement_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.01)
                else:
                    self.fail(
                        "measurement remained alive after the lock owner returned"
                    )
                self.assertEqual(lock_states(), ["acquired", "acquired"])
            finally:
                release.touch()
                if measurement_pid is not None:
                    try:
                        os.kill(measurement_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if outer.poll() is None:
                    outer.kill()
                    outer.wait()

    def test_substitute_and_restore_is_outside_cooperative_contract(self):
        """A green result deliberately pins behavior outside the contract."""

        with _SupervisorSandbox() as sb:
            substituted = Path(sb.tmp.name) / "substituted"
            restore = Path(sb.tmp.name) / "restore"
            marker = Path(sb.tmp.name) / "restored"
            entrypoint = sb.root / "scripts" / "substitute_paths.py"
            entrypoint.write_text(
                "import os, time\n"
                "from pathlib import Path\n"
                f"paths = tuple(Path(p) for p in ({str(sb.bench_lock)!r}, {str(sb.gpu_lock)!r}))\n"
                "for path in paths:\n"
                "    path.rename(path.with_name(path.name + '.held'))\n"
                "    path.touch()\n"
                f"Path({str(substituted)!r}).write_text('ready')\n"
                f"restore = Path({str(restore)!r})\n"
                "while not restore.exists():\n"
                "    time.sleep(0.01)\n"
                "for path in paths:\n"
                "    path.unlink()\n"
                "    path.with_name(path.name + '.held').rename(path)\n"
                f"Path({str(marker)!r}).write_text('restored')\n"
            )
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(sb.helper),
                    "run",
                    "--label",
                    "fixture",
                    "--",
                    sys.executable,
                    str(entrypoint),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10
                while not substituted.exists() and proc.poll() is None:
                    if time.monotonic() >= deadline:
                        self.fail("entrypoint did not substitute lock paths")
                    time.sleep(0.01)
                self.assertIsNone(proc.poll())
                for path in (sb.bench_lock, sb.gpu_lock):
                    fd = os.open(path, os.O_RDWR)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        os.close(fd)
            finally:
                restore.touch()
                stdout, stderr = proc.communicate(timeout=30)
            outside_contract = (
                "success deliberately records behavior outside the cooperative contract"
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"{outside_contract}:\n{stdout}\n{stderr}",
            )
            self.assertEqual(marker.read_text(), "restored", outside_contract)

    def test_forged_live_pipe_handoff_is_accepted_but_outside_cooperative_contract(
        self,
    ):
        """Acceptance records the deliberate same-user bypass limitation."""

        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "forged-live-pipe.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.gpu_lock}): fabricated\n"
            )
            handoff_accepted = Path(sb.tmp.name) / "handoff-accepted"
            release = Path(sb.tmp.name) / "release-temporary-holders"
            marker = Path(sb.tmp.name) / "forged-handoff-result"
            entrypoint = sb.root / "scripts" / "forged_handoff.py"
            entrypoint.write_text(
                "import fcntl, os, sys, time\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('fixture')\n"
                f"Path({str(handoff_accepted)!r}).write_text('accepted')\n"
                f"release = Path({str(release)!r})\n"
                "while not release.exists():\n"
                "    time.sleep(0.01)\n"
                "states = []\n"
                f"for path in ({str(sb.bench_lock)!r}, {str(sb.gpu_lock)!r}):\n"
                "    fd = os.open(path, os.O_RDWR)\n"
                "    try:\n"
                "        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                "    except OSError:\n"
                "        states.append('blocked')\n"
                "    else:\n"
                "        states.append('acquired')\n"
                "    finally:\n"
                "        os.close(fd)\n"
                f"Path({str(marker)!r}).write_text('accepted:' + ','.join(states))\n"
            )
            holders = tuple(
                os.open(path, os.O_RDWR) for path in (sb.bench_lock, sb.gpu_lock)
            )
            read_fd, write_fd = os.pipe()
            env = {
                **os.environ,
                "LATTICE_BENCH_LOCK_STATUS": str(status),
                "LATTICE_BENCH_SUPERVISOR_FD": str(read_fd),
            }
            env.pop("LATTICE_BENCH_LOCK_FDS", None)
            proc = None
            try:
                for fd in holders:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                proc = subprocess.Popen(
                    [sys.executable, str(entrypoint)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    pass_fds=(read_fd,),
                )
                deadline = time.monotonic() + 10
                while not handoff_accepted.exists() and proc.poll() is None:
                    if time.monotonic() >= deadline:
                        self.fail("forged handoff was not sampled")
                    time.sleep(0.01)
                self.assertIsNone(proc.poll())
                for fd in holders:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                release.touch()
                stdout, stderr = proc.communicate(timeout=30)
            finally:
                release.touch()
                for fd in (*holders, read_fd, write_fd):
                    os.close(fd)
                if proc is not None and proc.poll() is None:
                    proc.kill()
                    proc.communicate()
            outside_contract = (
                "success deliberately records behavior outside the cooperative contract"
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"{outside_contract}:\n{stdout}\n{stderr}",
            )
            self.assertEqual(
                marker.read_text(),
                "accepted:acquired,acquired",
                outside_contract,
            )

    def test_oversized_descriptor_uses_normal_refusal_diagnostic(self):
        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "oversized.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.gpu_lock}): fabricated\n"
            )
            result = subprocess.run(
                [sys.executable, str(sb.helper), "verify"],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "LATTICE_BENCH_LOCK_STATUS": str(status),
                    "LATTICE_BENCH_LOCK_FDS": f"{2**100},{2**101}",
                },
                timeout=30,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("cannot be matched", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_python_entrypoint_hides_capability_names_during_work(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "python-entrypoint"
            entrypoint = sb.root / "scripts" / "entrypoint.py"
            entrypoint.write_text(
                "import fcntl, os, sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('fixture')\n"
                "states = []\n"
                f"for path in ({str(sb.bench_lock)!r}, {str(sb.gpu_lock)!r}):\n"
                "    fd = os.open(path, os.O_RDWR)\n"
                "    try:\n"
                "        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                "    except OSError:\n"
                "        states.append('blocked')\n"
                "    else:\n"
                "        states.append('acquired')\n"
                "    finally:\n"
                "        os.close(fd)\n"
                f"Path({str(marker)!r}).write_text("
                "('present' if ('LATTICE_BENCH_LOCK_FDS' in os.environ or "
                "'LATTICE_BENCH_SUPERVISOR_FD' in os.environ) "
                "else 'hidden') + ':' + ','.join(states))\n"
            )
            result = subprocess.run(
                [sys.executable, str(entrypoint)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(), "hidden:blocked,blocked")

    def test_python_entrypoint_refuses_pipe_without_open_writer(self):
        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "closed-pipe.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): acquired\n"
                f"lock=Metal GPU ({sb.gpu_lock}): acquired\n"
            )
            marker = Path(sb.tmp.name) / "must-not-run"
            entrypoint = sb.root / "scripts" / "closed_pipe.py"
            entrypoint.write_text(
                "import sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('fixture')\n"
                f"Path({str(marker)!r}).write_text('ran')\n"
            )
            witness_fd, writer_fd = os.pipe()
            os.close(writer_fd)
            try:
                result = subprocess.run(
                    [sys.executable, str(entrypoint)],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_SUPERVISOR_FD": str(witness_fd),
                    },
                    pass_fds=(witness_fd,),
                    timeout=30,
                )
            finally:
                os.close(witness_fd)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("no open writer", result.stderr)
            self.assertFalse(marker.exists())

    def test_replaced_canonical_paths_are_refused_before_measurement(self):
        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "path-replacement.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): acquired\n"
                f"lock=Metal GPU ({sb.gpu_lock}): acquired\n"
            )
            marker = Path(sb.tmp.name) / "replacement-measurement-ran"
            driver = sb.root / "scripts" / "replace_paths.py"
            driver.write_text(
                "import sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "import bench_supervision as supervision\n"
                f"paths = ({str(sb.bench_lock)!r}, {str(sb.gpu_lock)!r})\n"
                "original_flock = supervision.fcntl.flock\n"
                "replaced = False\n"
                "def replace_then_flock(fd, operation):\n"
                "    global replaced\n"
                "    if not replaced:\n"
                "        replaced = True\n"
                "        for raw in paths:\n"
                "            path = Path(raw)\n"
                "            path.rename(path.with_name(path.name + '.held'))\n"
                "            path.touch()\n"
                "    return original_flock(fd, operation)\n"
                "supervision.fcntl.flock = replace_then_flock\n"
                "raise SystemExit(supervision.main([\n"
                "    'run', '--label', 'fixture', '--',\n"
                f"    sys.executable, '-c', \"from pathlib import Path; Path({str(marker)!r}).write_text('ran')\",\n"
                "]))\n"
            )
            inherited = tuple(
                os.open(path, os.O_RDWR) for path in (sb.bench_lock, sb.gpu_lock)
            )
            try:
                for fd in inherited:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    [sys.executable, str(driver)],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_LOCK_FDS": ",".join(map(str, inherited)),
                    },
                    pass_fds=inherited,
                    timeout=30,
                )
            finally:
                for fd in inherited:
                    os.close(fd)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("changed while acquiring", result.stderr)
            self.assertFalse(marker.exists())

    def test_replaced_canonical_paths_during_measurement_are_refused_afterward(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "replacement-during-measurement"
            code = (
                "from pathlib import Path; "
                f"paths=({str(sb.bench_lock)!r}, {str(sb.gpu_lock)!r}); "
                "[(lambda path: (path.rename(path.with_name(path.name + '.held')), "
                "path.touch()))(Path(raw)) for raw in paths]; "
                f"Path({str(marker)!r}).write_text('ran')"
            )
            result = sb.run([sys.executable, "-c", code])
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("changed after measurement", result.stderr)
            self.assertEqual(marker.read_text(), "ran")

    def test_durable_entrypoint_refuses_lock_only_outer_supervisor(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "must-not-run"
            entrypoint = sb.root / "scripts" / "durable_entrypoint.py"
            entrypoint.write_text(
                "import sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('fixture', quiet=True)\n"
                f"Path({str(marker)!r}).write_text('ran')\n"
            )
            result = sb.run(
                [sys.executable, str(entrypoint)],
                entrypoint=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(marker.exists())
            self.assertIn("lock-only", result.stderr)

    def test_shell_measurement_children_do_not_inherit_supervisor_witness(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "shell-entrypoint"
            entrypoint = sb.root / "scripts" / "entrypoint.sh"
            entrypoint.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"source {str(sb.root / 'scripts/lib/bench-supervision.sh')!r}\n"
                'inherited_fd="${LATTICE_BENCH_SUPERVISOR_FD:-}"\n'
                "measurement() {\n"
                '  [[ -z "${LATTICE_BENCH_SUPERVISOR_FD:-}" ]]\n'
                "  if python3 -c \"import os; os.fstat($inherited_fd)\" 2>/dev/null; then\n"
                "      state=inherited\n"
                "    else\n"
                "      state=closed\n"
                "    fi\n"
                f"  printf '%s' \"$state\" > {str(marker)!r}\n"
                "}\n"
                'bench_supervise_entry "fixture" ordinary measurement "$@"\n'
            )
            entrypoint.chmod(0o755)
            result = subprocess.run(
                ["bash", str(entrypoint)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(), "closed")

    def test_shell_handoff_keeps_witness_for_nested_python_guard(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "nested-python-entrypoint"
            cargo_marker = Path(sb.tmp.name) / "cargo-capabilities"
            python_entrypoint = sb.root / "scripts" / "nested.py"
            python_entrypoint.write_text(
                "import os, sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('fixture')\n"
                f"Path({str(marker)!r}).write_text("
                "'present' if 'LATTICE_BENCH_SUPERVISOR_FD' in os.environ else 'hidden')\n"
            )
            shell_entrypoint = sb.root / "scripts" / "nested.sh"
            shell_entrypoint.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"source {str(sb.root / 'scripts/lib/bench-supervision.sh')!r}\n"
                'bench_supervise_entry "fixture" handoff - "$@"\n'
                "(\n"
                "  bench_close_supervisor_witness\n"
                f"  printf '%s' \"${{LATTICE_BENCH_SUPERVISOR_FD:-closed}}\" > {str(cargo_marker)!r}\n"
                ")\n"
                f"exec {sys.executable!r} {str(python_entrypoint)!r}\n"
            )
            shell_entrypoint.chmod(0o755)
            result = subprocess.run(
                ["bash", str(shell_entrypoint)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(cargo_marker.read_text(), "closed")
            self.assertEqual(marker.read_text(), "hidden")

    def test_actual_e2e_entrypoint_hands_off_to_build_and_parity_check(self):
        with _SupervisorSandbox() as sb:
            entrypoint = sb.root / "scripts" / "e2e-parity-local.sh"
            shutil.copy2(REPO / "scripts" / "e2e-parity-local.sh", entrypoint)
            cargo_args = Path(sb.tmp.name) / "cargo-args"
            parity_marker = Path(sb.tmp.name) / "parity-ran"
            bindir = Path(sb.tmp.name) / "bin"
            bindir.mkdir()
            cargo = bindir / "cargo"
            cargo.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                '[[ -z "${LATTICE_BENCH_SUPERVISOR_FD:-}" ]]\n'
                f"printf '%s' \"$*\" > {str(cargo_args)!r}\n"
            )
            cargo.chmod(0o755)
            parity = sb.root / "scripts" / "e2e_parity_check.py"
            parity.write_text(
                "import sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('e2e-parity-local')\n"
                f"Path({str(parity_marker)!r}).write_text('ran')\n"
            )
            env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"}
            for name in (
                "LATTICE_BENCH_LOCK_STATUS",
                "LATTICE_BENCH_LOCK_FDS",
                "LATTICE_BENCH_SUPERVISOR_FD",
            ):
                env.pop(name, None)
            result = subprocess.run(
                ["bash", str(entrypoint)],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                cargo_args.read_text(),
                "build --release --bin qwen35_generate -p lattice-inference "
                "--features f16",
            )
            self.assertEqual(parity_marker.read_text(), "ran")

    def test_durable_shell_refuses_lock_only_outer_without_errexit(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "shell-must-not-run"
            entrypoint = sb.root / "scripts" / "durable_entrypoint.sh"
            entrypoint.write_text(
                "#!/usr/bin/env bash\n"
                "set -uo pipefail\n"
                f"source {str(sb.root / 'scripts/lib/bench-supervision.sh')!r}\n"
                f"measurement() {{ printf ran > {str(marker)!r}; }}\n"
                'bench_supervise_entry "fixture" durable measurement "$@"\n'
            )
            entrypoint.chmod(0o755)
            result = sb.run(
                [str(entrypoint)],
                entrypoint=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(marker.exists())
            self.assertIn("lock-only", result.stderr)

    @unittest.skipUnless(shutil.which("node"), "node is unavailable")
    def test_node_child_can_forward_pipe_to_handoff_sample(self):
        """Mutation-sensitive: ordinary spawn closes the handoff pipe."""

        with _SupervisorSandbox() as sb:
            code = """
const {spawnSync} = require('node:child_process');
const fd = Number(process.env.LATTICE_BENCH_SUPERVISOR_FD);
const stdio = ['ignore', 'pipe', 'pipe'];
while (stdio.length <= fd) stdio.push('ignore');
stdio[fd] = fd;
const child = spawnSync(
  'python3',
  [process.argv[1], 'verify'],
  {env: process.env, encoding: 'utf8', stdio},
);
process.stdout.write(child.stdout ?? '');
process.stderr.write(child.stderr ?? '');
process.exit(child.status ?? 2);
"""
            result = sb.run(
                ["node", "-e", code, str(sb.helper)],
                entrypoint=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_idle_inherited_fds_are_acquired_by_measuring_process(self):
        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "idle.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.gpu_lock}): fabricated\n"
            )
            inherited = tuple(
                os.open(path, os.O_RDWR) for path in (sb.bench_lock, sb.gpu_lock)
            )
            try:
                result = subprocess.run(
                    [sys.executable, str(sb.helper), "verify"],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_LOCK_FDS": ",".join(map(str, inherited)),
                    },
                    pass_fds=inherited,
                    timeout=30,
                )
            finally:
                for fd in inherited:
                    os.close(fd)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_locked_noncanonical_paths_are_refused(self):
        with _SupervisorSandbox() as sb:
            fake_paths = (
                Path(sb.tmp.name) / "fake-window.lock",
                Path(sb.tmp.name) / "fake-gpu.lock",
            )
            inherited = tuple(
                os.open(path, os.O_RDWR | os.O_CREAT) for path in fake_paths
            )
            status = Path(sb.tmp.name) / "fake-path.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({fake_paths[0]}): fabricated\n"
                f"lock=Metal GPU ({fake_paths[1]}): fabricated\n"
            )
            try:
                for fd in inherited:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    [sys.executable, str(sb.helper), "verify"],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_LOCK_FDS": ",".join(map(str, inherited)),
                    },
                    pass_fds=inherited,
                    timeout=30,
                )
            finally:
                for fd in inherited:
                    os.close(fd)
            self.assertEqual(result.returncode, 2)
            self.assertIn(f"expected {sb.bench_lock}", result.stderr)

    def test_swapped_canonical_lock_order_is_refused(self):
        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            inherited = tuple(
                os.open(path, os.O_RDWR) for path in (sb.gpu_lock, sb.bench_lock)
            )
            status = Path(sb.tmp.name) / "swapped.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.gpu_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.bench_lock}): fabricated\n"
            )
            try:
                for fd in inherited:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    [sys.executable, str(sb.helper), "verify"],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_LOCK_FDS": ",".join(map(str, inherited)),
                    },
                    pass_fds=inherited,
                    timeout=30,
                )
            finally:
                for fd in inherited:
                    os.close(fd)
            self.assertEqual(result.returncode, 2)
            self.assertIn(f"expected {sb.bench_lock}", result.stderr)

    def test_duplicate_lock_inode_is_refused(self):
        with _SupervisorSandbox() as sb:
            sb.bench_lock.touch()
            os.link(sb.bench_lock, sb.gpu_lock)
            fd = os.open(sb.bench_lock, os.O_RDWR)
            twin = os.dup(fd)
            status = Path(sb.tmp.name) / "duplicate-inode.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.gpu_lock}): fabricated\n"
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    [sys.executable, str(sb.helper), "verify"],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_LOCK_FDS": f"{fd},{twin}",
                    },
                    pass_fds=(fd, twin),
                    timeout=30,
                )
            finally:
                os.close(twin)
                os.close(fd)
            self.assertEqual(result.returncode, 2)
            self.assertIn("distinct inodes", result.stderr)

    def test_self_held_fds_do_not_replace_supervisor_witness(self):
        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "self-held.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): acquired\n"
                f"lock=Metal GPU ({sb.gpu_lock}): acquired\n"
            )
            marker = Path(sb.tmp.name) / "must-not-run"
            code = (
                "import fcntl, os, sys; from pathlib import Path; "
                f"sys.path.insert(0, {str(sb.helper.parent)!r}); "
                f"paths=({str(sb.bench_lock)!r}, {str(sb.gpu_lock)!r}); "
                "fds=tuple(os.open(path, os.O_RDWR) for path in paths); "
                "[fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB) for fd in fds]; "
                "os.environ['LATTICE_BENCH_LOCK_FDS']=','.join(map(str, fds)); "
                "from bench_supervision import ensure_python_entrypoint; "
                "ensure_python_entrypoint('fixture'); "
                f"Path({str(marker)!r}).write_text('ran')"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "LATTICE_BENCH_LOCK_STATUS": str(status),
                },
                timeout=30,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("LATTICE_BENCH_SUPERVISOR_FD is not set", result.stderr)
            self.assertFalse(marker.exists())

    def test_forged_ancestor_receipt_without_fds_is_refused_before_command(self):
        with _SupervisorSandbox() as sb:
            status = Path(sb.tmp.name) / "forged-ancestor.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.gpu_lock}): fabricated\n"
            )
            marker = Path(sb.tmp.name) / "unsupervised-command-ran"
            result = subprocess.run(
                [
                    sys.executable,
                    str(sb.helper),
                    "run",
                    "--label",
                    "forged-ancestor",
                    "--",
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
                ],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "LATTICE_BENCH_LOCK_STATUS": str(status),
                },
                timeout=30,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("LATTICE_BENCH_LOCK_FDS is not set", result.stderr)
            self.assertFalse(marker.exists())

    def test_unlocked_inherited_fds_do_not_borrow_another_holders_contention(self):
        """Mutation-sensitive: probing only by path accepts the wrong holder."""

        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "wrong-holder.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.gpu_lock}): fabricated\n"
            )
            inherited = tuple(
                os.open(path, os.O_RDWR) for path in (sb.bench_lock, sb.gpu_lock)
            )
            holders = tuple(
                os.open(path, os.O_RDWR) for path in (sb.bench_lock, sb.gpu_lock)
            )
            try:
                for fd in holders:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    [sys.executable, str(sb.helper), "verify"],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_LOCK_FDS": ",".join(map(str, inherited)),
                    },
                    pass_fds=inherited,
                    timeout=30,
                )
            finally:
                for fd in (*inherited, *holders):
                    os.close(fd)
            self.assertEqual(result.returncode, 2)
            self.assertIn("could not acquire canonical lock", result.stderr)


class BenchCompareDirectInvocationRefusal(unittest.TestCase):
    """The bench-compare body requires a descriptor-free supervisor handoff.

    The entry point gives inherited lock capabilities only to the dedicated
    Python supervisor. The shell receives its non-lock handoff pipe. A caller
    that writes a status receipt and invokes the body directly has neither, so
    verify_locks refuses before touching cargo.
    """

    def _build_repo(self, tmp: str) -> Path:
        root = Path(tmp) / "repo"
        lib = root / "scripts" / "lib"
        lib.mkdir(parents=True)
        shutil.copy2(
            REPO / "scripts" / "bench-compare.sh",
            root / "scripts" / "bench-compare.sh",
        )
        (root / "scripts" / "bench-compare.sh").chmod(0o755)
        shutil.copy2(
            REPO / "scripts" / "lib" / "bench-compare-impl.sh",
            lib / "bench-compare-impl.sh",
        )
        (lib / "bench-compare-impl.sh").chmod(0o755)
        shutil.copy2(
            REPO / "scripts" / "lib" / "bench_supervision.py",
            lib / "bench_supervision.py",
        )
        shutil.copy2(
            REPO / "scripts" / "lib" / "bench-python.sh",
            lib / "bench-python.sh",
        )
        self.bench_lock = Path(tmp) / "bench-window.lock"
        self.gpu_lock = Path(tmp) / "metal-gpu.lock"
        self.pending = Path(tmp) / "bench-window-pending"
        lock_source = (REPO / "scripts" / "lib" / "bench-locks.py").read_text()
        for name, value in (
            ("BENCH_WINDOW", self.bench_lock),
            ("GPU_LOCK", self.gpu_lock),
            ("PENDING_DIR", self.pending),
        ):
            lock_source = re.sub(
                rf'^{name} = "[^"]*"$',
                f'{name} = "{value}"',
                lock_source,
                flags=re.M,
            )
        (lib / "bench-locks.py").write_text(lock_source)
        (lib / "quiet-probe.py").write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "label = sys.argv[sys.argv.index('--label') + 1]\n"
            "print(f'[quiet] {label}: idle 100.0% (floor 0.0%) ok | top: fixture 0.0%')\n"
        )
        (lib / "machine-state-probe.py").write_text(
            "#!/usr/bin/env python3\n"
            "import datetime, json, sys\n"
            "label = sys.argv[sys.argv.index('--label') + 1]\n"
            "print(json.dumps({'schema': 'lattice-machine-state-v1', 'label': label,"
            "'captured_at_utc': datetime.datetime.now(datetime.UTC)"
            ".strftime('%Y-%m-%dT%H:%M:%SZ'),"
            "'power': {'status': 'unavailable', 'reason': 'fixture'},"
            "'thermal': {'status': 'unavailable', 'reason': 'fixture'},"
            "'idle': {'status': 'unavailable', 'reason': 'fixture'}}))\n"
        )
        (root / "scripts" / "perf_governor.py").write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "from datetime import UTC, datetime\n"
            "label = sys.argv[sys.argv.index('--label') + 1]\n"
            "print(json.dumps({'schema': 'lattice-machine-state-v1', 'label': label,"
            "'captured_at_utc': datetime.now(UTC).replace(microsecond=0)"
            ".isoformat().replace('+00:00', 'Z'),"
            "'power': {'status': 'measured', 'source': 'fixture', 'state': 'ac'},"
            "'thermal': {'status': 'measured', 'source': 'fixture', 'state': 'nominal'},"
            "'idle': {'status': 'measured', 'source': 'fixture', 'seconds': 30.0},"
            "'gate': {'status': 'passed', 'cooldown_seconds': 30.0,"
            "'afk_threshold_seconds': 30.0, 'kill_switch': 'clear'}}))\n"
        )
        (root / "scripts" / "lib" / "ensure-noindex-marker.sh").write_text(
            "#!/usr/bin/env bash\nexit 0\n"
        )
        (root / "scripts" / "lib" / "ensure-noindex-marker.sh").chmod(0o755)

        shutil.copy2(REPO / ".gitignore", root / ".gitignore")
        env_git = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        }
        git = ("git", "-c", "core.hooksPath=/dev/null")
        subprocess.run([*git, "init", "-q", "-b", "main", str(root)], check=True)
        for i in range(2):
            (root / f"f{i}.txt").write_text(str(i))
            subprocess.run([*git, "-C", str(root), "add", "-A"], check=True)
            subprocess.run(
                [*git, "-C", str(root), "commit", "-qm", f"c{i}"],
                check=True, env=env_git,
            )

        bindir = Path(tmp) / "bin"
        bindir.mkdir()
        cargo_marker = Path(tmp) / "cargo-was-invoked"
        cargo = bindir / "cargo"
        cargo.write_text(
            "#!/usr/bin/env bash\n"
            f"printf ran > {str(cargo_marker)!r}\n"
            "if [[ \"$*\" == *--version* ]]; then printf '%s\\n' 'cargo 1.94.1 (fixture)'; fi\n"
            "exit 0\n"
        )
        cargo.chmod(0o755)
        self.cargo_marker = cargo_marker
        self.bindir = bindir
        return root

    def _write_forged_status(self, root: Path, own_pid: int) -> None:
        cache = root / ".cache"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "bench-locks-status.txt").write_text(
            f"supervisor_pid={own_pid}\n"
            f"lock=bench-window ({self.bench_lock}): fabricated\n"
            f"lock=Metal GPU ({self.gpu_lock}): fabricated\n"
        )

    def test_preflight_helper_cannot_observe_or_release_lock_capabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._build_repo(tmp)
            marker = Path(tmp) / "date-descriptor-probe"
            date = self.bindir / "date"
            date.write_text(
                "#!/usr/bin/env python3\n"
                "import fcntl, os\n"
                "from pathlib import Path\n"
                f"paths = ({str(self.bench_lock)!r}, {str(self.gpu_lock)!r})\n"
                "lock_ids = {\n"
                "    (os.stat(path).st_dev, os.stat(path).st_ino) for path in paths\n"
                "}\n"
                "observed = []\n"
                "for fd in range(3, 256):\n"
                "    try:\n"
                "        fd_stat = os.fstat(fd)\n"
                "    except OSError:\n"
                "        continue\n"
                "    if (fd_stat.st_dev, fd_stat.st_ino) in lock_ids:\n"
                "        observed.append(fd)\n"
                "        fcntl.flock(fd, fcntl.LOCK_UN)\n"
                "states = []\n"
                "for path in paths:\n"
                "    fd = os.open(path, os.O_RDWR)\n"
                "    try:\n"
                "        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                "    except OSError:\n"
                "        states.append('blocked')\n"
                "    else:\n"
                "        states.append('acquired')\n"
                "    finally:\n"
                "        os.close(fd)\n"
                f"Path({str(marker)!r}).write_text(\n"
                "    ('observed' if observed else 'hidden')\n"
                "    + ':' + ','.join(states)\n"
                ")\n"
                "print('2026-07-31T00:00:00Z')\n"
                "raise SystemExit(42)\n"
            )
            date.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{self.bindir}:{os.environ['PATH']}",
                "BENCH_HOST_ID": "fixture",
            }
            for name in (
                "LATTICE_BENCH_LOCK_STATUS",
                "LATTICE_BENCH_LOCK_FDS",
                "LATTICE_BENCH_SUPERVISOR_FD",
            ):
                env.pop(name, None)
            result = subprocess.run(
                [
                    "bash",
                    str(root / "scripts" / "bench-compare.sh"),
                    "HEAD~1",
                    "HEAD",
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            observed = marker.read_text()

        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertEqual(
            observed,
            "hidden:blocked,blocked",
            f"preflight helper inherited usable lock capabilities:\n{result.stderr}",
        )

    def _invoke(self, root: Path) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PATH": f"{self.bindir}:{os.environ['PATH']}",
            "BENCH_HOST_ID": "fixture",
            "BENCH_IDLE_FLOOR": "0",
        }
        env.pop("LATTICE_BENCH_LOCK_FDS", None)
        return subprocess.run(
            ["bash", str(root / "scripts" / "lib" / "bench-compare-impl.sh"),
             "HEAD~1", "HEAD"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

    def test_receipt_only_invocation_is_refused_before_any_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._build_repo(tmp)
            # The receipt contains a parseable caller-controlled PID, but no
            # descriptor-free handoff from the dedicated supervisor.
            self._write_forged_status(root, os.getpid())
            result = self._invoke(root)

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("LATTICE_BENCH_SUPERVISOR_FD", result.stderr)
        self.assertIn("refusing to measure", result.stderr)
        self.assertFalse(self.cargo_marker.exists())
        # Refused before the run-conditions banner, i.e. before base/head
        # resolution and worktree setup ever started.
        self.assertNotIn("=== bench-compare:", result.stdout)


class _FailOnEmptyTestProgram(unittest.TestProgram):
    def runTests(self) -> None:
        if self.test.countTestCases() == 0:
            raise SystemExit("no tests collected")
        super().runTests()


if __name__ == "__main__":
    _FailOnEmptyTestProgram()

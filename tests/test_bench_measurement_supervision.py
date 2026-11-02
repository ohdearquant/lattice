#!/usr/bin/env python3
"""Tests for advisory measurement discovery and the classified measurement inventory.

The supervision runtime tests (locks, idle checks, the GPU handoff) are in
test_bench_measurement_supervision_runtime.py.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import shlex
import sys
import tempfile
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
    # These typed helpers require the frozen plan and private owner service.
    "scripts/lib/bench_admission.py",
    "scripts/lib/bench_handoff.py",
}
MEASUREMENT_SIGNAL = re.compile(
    r"perf_counter(?:_ns)?\(|hrtime\.bigint\(|cargo bench|tok_per_sec|"
    r"elapsed_ns|elapsed_s|total_ms|tokens/s|tok/s|PPL:"
)
CARGO_BENCH_LITERAL = re.compile(r"\bcargo[ \t\r\n]+bench\b")
# A script that delegates its measured binary to the GPU handoff launcher is a
# measurement entry point even though it never calls `cargo bench` itself.
GPU_HANDOFF_LAUNCH_SIGNAL = re.compile(
    r'(?m)^[^#\n]*bench-command\.sh"?[ \t]+--gpu-handoff\b'
)
GPU_HANDOFF_LAUNCH = re.compile(
    r'bench-command\.sh"?\s+--gpu-handoff(?P<durable>\s+--durable)?'
    r"\s+--label\s+(?P<label>\S+)\s+--\s+"
    r"cargo\s+run\s+--locked\s+--release\s+-p\s+lattice-inference"
    r'\s+--bin\s+(?P<bin>\w+)\s+--features\s+"?(?P<features>[\w,-]+)"?'
)
# A plain launch: the launcher without --gpu-handoff, so the supervisor holds both machine
# locks around a command that never takes the GPU lock itself (MLX does not call
# gpu_test_lock()).
PLAIN_LAUNCH = re.compile(
    r'bench-command\.sh"?[ \t]+(?P<durable>--durable[ \t]+)?--label[ \t]+\S+[ \t]+--[ \t]'
)
# What counts as MLX use in a shell script: the package by name, an import of it, or a
# script file named after it. Prose and variable names in capitals are not MLX use.
SHELL_MLX_USE = re.compile(
    r"mlx[_-]lm|\b(?:import|from)[ \t]+mlx\b|\bmlx\.(?:core|nn)\b|\bmlx[\w-]*\.py\b"
)
HEREDOC_OPEN = re.compile(r"<<-?[ \t]*(['\"]?)(\w+)\1")
# Each of these would make a handoff child supervise itself around the binary it
# delegates to the handoff, which can then never receive the GPU lock.
SELF_SUPERVISION_TOKENS = (
    "bench_supervise_entry",
    "bench-supervision.sh",
    "bench_supervision",
    "ensure_python_entrypoint",
)
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
    "#!/usr/bin/env -S uv run --script": "python",
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
        "scripts/cross-encoder-checkpoint.json",
        "scripts/fixtures/quality_compare/candidate.json",
        "scripts/fixtures/quality_compare/reference.json",
        "scripts/lib/bench-host-id.py",
        "scripts/lib/bench-informational-targets.sh",
        "scripts/lib/bench-locks.py",
        "scripts/lib/bench-quick-informational-targets.txt",
        "scripts/lib/bench-supervision.sh",
        "scripts/lib/bench_supervision.py",
        # Typed admission and its private retained-owner service are internal;
        # neither offers a standalone arbitrary measurement entry point.
        "scripts/lib/bench_admission.py",
        "scripts/lib/bench_handoff.py",
        "scripts/lib/ensure-noindex-marker.sh",
        "scripts/lib/machine-state-probe.py",
        "scripts/lib/phase-load-report.py",
        "scripts/lib/phase-load-sampler.py",
        "scripts/lib/quiet-probe.py",
        "scripts/microlora/README.md",
        "scripts/microlora/dsl_validator.rs",
        "scripts/perf-policy.toml",
        "scripts/perf_governor.README.md",
        "scripts/quality_producers.md",
        "scripts/requirements-parity.txt",
        "scripts/rust-fmt-coverage.allow",
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


def non_measurement_surfaces() -> set[str]:
    data = tomllib.loads(MANIFEST.read_text())
    return {surface["path"] for surface in data["non_measurement_surface"]}


# Packages whose examples and binaries the grammar covers. lattice-serve is
# listed before it exists so the extraction moves paths and classifications
# without also changing what discovery can see.
INVENTORY_APPLICATION_PACKAGES = ("crates/inference", "crates/serve")


def discovered_declared_rust_inventory_paths(repo: Path = REPO) -> set[str]:
    """Every Rust path the manifest's grammar must classify.

    Broadened from a bench*-prefix filename match (issue #1273): every
    example and every binary under the inference crate is discovered, not
    just ones whose basename happens to start with `bench`. A path that
    matches this glob and is not in excluded_surface or
    non_measurement_surface fails the fail-closed test below.

    Broadened again to descend into subdirectories of src/bin (multi-file
    binaries such as lattice's chat/serve/doctor/prune-score split): a
    `[[bin]]` target's `path` no longer has to be a single top-level file,
    and a glob that only matched top-level files would let a split binary's
    modules go both unclassified and unseen by this fail-closed discovery.
    """
    paths = {
        str(path.relative_to(repo))
        for path in (repo / "crates").glob("*/benches/*.rs")
    }
    for package in INVENTORY_APPLICATION_PACKAGES:
        for pattern in ("examples/*.rs", "src/bin/*.rs", "src/bin/*/*.rs"):
            paths.update(
                str(path.relative_to(repo))
                for path in (repo / package).glob(pattern)
            )
    paths.add("README.md")
    if (repo / "crates/fann/examples/router_online.rs").is_file():
        paths.add("crates/fann/examples/router_online.rs")
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
        if isinstance(node, ast.Constant) and node.value == "--gpu-handoff":
            evidence.add("GPU handoff measurement launch")
        if isinstance(node, ast.Call) and _dotted_name(node.func) == "harness.run_lattice_handoff":
            evidence.add("decode lattice handoff")
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
    if GPU_HANDOFF_LAUNCH_SIGNAL.search(source):
        evidence.add("GPU handoff measurement launch")
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


def _logical_lines(source: str) -> list[str]:
    """Shell lines with backslash continuations joined and comment lines dropped."""
    lines: list[str] = []
    pending = ""
    for line in source.splitlines():
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        lines.append(pending + stripped)
        pending = ""
    if pending:
        lines.append(pending)
    return [line for line in lines if not line.lstrip().startswith("#")]


def gpu_handoff_launches(path: str, source: str) -> list[dict[str, str]]:
    """Every delegated GPU-handoff launch of a declared binary in one script."""
    if path.endswith(".py"):
        tree = ast.parse(source, filename=path)
        argv: list[str] | None = None
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in {"HANDOFF_ARGV", "LATTICE_HANDOFF_ARGV"}
                    for target in node.targets
                )
                and isinstance(node.value, ast.List)
                and all(
                    isinstance(item, ast.Constant) and isinstance(item.value, str)
                    for item in node.value.elts
                )
            ):
                argv = [item.value for item in node.value.elts]
        # The launcher path is built at run time, so the argv list must be the one
        # handed to it: the script has to splice HANDOFF_ARGV after bench-command.sh.
        argv_name = "LATTICE_HANDOFF_ARGV" if "LATTICE_HANDOFF_ARGV" in source else "HANDOFF_ARGV"
        if (
            argv is None
            or "bench-command.sh" not in source
            or f"*{argv_name}" not in source
        ):
            return []
        text = "bench-command.sh " + " ".join(argv)
    else:
        text = "\n".join(_logical_lines(source))
    return [match.groupdict() for match in GPU_HANDOFF_LAUNCH.finditer(text)]


def _shell_units(source: str) -> list[tuple[str, str]]:
    """(command line, heredoc body) pairs: continuations joined, comment lines dropped."""
    units: list[tuple[str, str]] = []
    pending = ""
    lines = iter(source.splitlines())
    for raw in lines:
        stripped = raw.rstrip()
        if not pending and stripped.lstrip().startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        # A trailing comment is prose, not a command (a `#` inside quotes is a blind spot).
        command = re.sub(r"[ \t]#.*$", "", pending + stripped)
        pending = ""
        body: list[str] = []
        opener = HEREDOC_OPEN.search(command)
        if opener is not None:
            for inner in lines:
                if inner.strip() == opener.group(2):
                    break
                body.append(inner)
        units.append((command, "\n".join(body)))
    return units


def _mlx_call_violations(tree: ast.Module) -> tuple[int, list[str]]:
    """(functions importing mlx, calls that reach MLX from outside the child branch).

    A function reaches MLX if it imports mlx/mlx_lm or makes an unguarded call to one that
    does; a call guarded by an `if` testing MLX_CHILD_DIR does not propagate. A call to a
    function that reaches MLX is allowed when it is guarded, or when its caller is itself
    only ever entered from guarded calls (the child entry and what it calls).
    """

    def imports_mlx(node: ast.AST) -> bool:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Import):
                if any(alias.name.split(".")[0] in {"mlx", "mlx_lm"} for alias in inner.names):
                    return True
            elif isinstance(inner, ast.ImportFrom):
                if (inner.module or "").split(".")[0] in {"mlx", "mlx_lm"}:
                    return True
        return False

    sites: list[tuple[str, str, bool, int]] = []  # (caller, callee, guarded, line)

    def walk(node: ast.AST, owner: str, guarded: bool) -> None:
        if isinstance(node, ast.FunctionDef):
            owner, guarded = node.name, False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            sites.append((owner, node.func.id, guarded, node.lineno))
        for field, value in ast.iter_fields(node):
            for child in value if isinstance(value, list) else [value]:
                if isinstance(child, ast.AST):
                    walk(
                        child,
                        owner,
                        guarded
                        or (
                            isinstance(node, ast.If)
                            and field == "body"
                            and "MLX_CHILD_DIR" in ast.unparse(node.test)
                        ),
                    )

    walk(tree, "<module>", False)
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    importers = {name for name, fn in functions.items() if imports_mlx(fn)}
    reaching = set(importers)
    changed = True
    while changed:
        changed = False
        for caller, callee, guarded, _ in sites:
            if not guarded and callee in reaching and caller in functions and caller not in reaching:
                reaching.add(caller)
                changed = True
    entered_only_from_children = set(reaching)
    changed = True
    while changed:
        changed = False
        for caller, callee, guarded, _ in sites:
            if (
                callee in entered_only_from_children
                and not guarded
                and caller not in entered_only_from_children
            ):
                entered_only_from_children.discard(callee)
                changed = True
    violations = [
        f"line {line}: {callee}() reaches MLX outside the child branch"
        for caller, callee, guarded, line in sites
        if callee in reaching and not guarded and caller not in entered_only_from_children
    ]
    return len(importers), violations


def decode_worker_boundary(source: str) -> bool:
    """Check the shared source for the two worker routes and their ordering."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    launches = gpu_handoff_launches("scripts/bench_decode_harness.py", source)
    if launches != [{"durable": " --durable", "label": "UNIQUE_LABEL", "bin": "bench_decode_ab", "features": "metal-gpu,f16"}]:
        return False
    launch_fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_lattice_handoff"), None)
    plain_fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_launch_plain_worker"), None)
    worker_fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_decode_worker"), None)
    run_fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_profile"), None)
    prompt_fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_worker_prompt"), None)
    unload_fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_ollama_unload"), None)
    url_fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_ollama_url"), None)
    main_fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"), None)
    if not all((launch_fn, plain_fn, worker_fn, run_fn, prompt_fn, unload_fn, url_fn, main_fn)):
        return False
    if not any(isinstance(node, ast.Call) and ast.unparse(node.func) == "subprocess.run" for node in ast.walk(launch_fn)):
        return False
    if "--durable" not in ast.unparse(plain_fn) or "--gpu-handoff" in ast.unparse(plain_fn):
        return False
    worker_dispatch = next(
        (
            node for node in ast.walk(run_fn)
            if isinstance(node, ast.If)
            and "supervised_children" in ast.unparse(node.test)
            and "group.name in {'mlx', 'ollama'}" in ast.unparse(node.test)
            and any(
                isinstance(child, ast.Call) and ast.unparse(child.func) == "_launch_plain_worker"
                for child in ast.walk(node)
            )
        ),
        None,
    )
    if worker_dispatch is None:
        return False
    if not all(text in ast.unparse(prompt_fn) for text in ("prepare_prompt", "mlx", "_launch_plain_worker")):
        return False
    statements = list(worker_fn.body)
    handoff_at = next((i for i, node in enumerate(statements) if "sample_measurement_handoff" in ast.unparse(node)), None)
    import_at = next((i for i, node in enumerate(statements) if "importlib.import_module" in ast.unparse(node)), None)
    if handoff_at is None or import_at is None or handoff_at >= import_at:
        return False
    child_run = next(
        (
            node for node in ast.walk(worker_fn)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "run_profile"
            and "{engine: tracked}" in ast.unparse(node)
        ),
        None,
    )
    if child_run is None or statements.index(next(node for node in statements if child_run in ast.walk(node))) < import_at:
        return False
    if not all(text in ast.unparse(unload_fn) for text in ("keep_alive", "/api/ps", "model in names")):
        return False
    if not all(text in ast.unparse(url_fn) for text in ("localhost", "127.0.0.1", "::1")):
        return False
    plain_source = ast.unparse(plain_fn)
    if plain_source.find("proc.returncode != 0") < 0 or plain_source.find("result_path.read_text()") < plain_source.find("proc.returncode != 0"):
        return False
    if "uuid.uuid4" not in ast.unparse(launch_fn):
        return False
    unload_calls = [
        node for node in ast.walk(worker_fn)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_ollama_unload"
    ]
    if len(unload_calls) != 1:
        return False
    parents = {child: node for node in ast.walk(worker_fn) for child in ast.iter_child_nodes(node)}
    unload_parent = parents.get(unload_calls[0])
    while unload_parent is not None and not isinstance(unload_parent, ast.If):
        unload_parent = parents.get(unload_parent)
    if (
        not isinstance(unload_parent, ast.If)
        or "tracked.started" not in ast.unparse(unload_parent.test)
        or "engine == 'ollama'" not in ast.unparse(unload_parent.test)
    ):
        return False
    host_check = next(
        (
            node for node in ast.walk(url_fn)
            if isinstance(node, ast.Compare)
            and ast.unparse(node.left) == "parsed.hostname"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotIn)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Set)
        ),
        None,
    )
    if host_check is None:
        return False
    hosts = {
        item.value for item in host_check.comparators[0].elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }
    if hosts != {"localhost", "127.0.0.1", "::1"}:
        return False
    coordinator = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "coordinator_is_supervised"),
        None,
    )
    if coordinator is None or "LATTICE_GPU_HANDOFF_CONTROL" not in ast.unparse(coordinator):
        return False
    main_source = ast.unparse(main_fn)
    if "command_line[:1] == ['run']" not in main_source or "coordinator_is_supervised()" not in main_source:
        return False
    if main_source.find("coordinator_is_supervised()") > main_source.find("build_arg_parser()"):
        return False
    run_calls = [
        node for node in ast.walk(main_fn)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "run_profile"
        and any(
            keyword.arg == "supervised_children"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    ]
    if not run_calls:
        return False
    return True


def mlx_launch_report(path: str, source: str, *, durable: bool) -> tuple[int, list[str]]:
    """(MLX use sites found, violations): MLX may be reached only through a plain launch.

    Shell: every command (with its heredoc program) that uses MLX must be a plain
    `bench-command.sh [--durable] --label L --` launch, with --durable present exactly
    when the entry is `+quiet`, and the MLX words must come after the `--`.
    Python: no module-level MLX import; every call of a function that reaches MLX, from
    code that does not itself reach MLX, sits in the body of an `if` testing the child
    flag MLX_CHILD_DIR; and the child flag appears only in one list, MLX_LAUNCH_ARGV,
    which names no --gpu-handoff, carries --durable exactly when the entry is `+quiet`, and is
    spread only into a call whose first element is the bench-command.sh path.
    """
    problems: list[str] = []
    uses = 0
    if path.endswith(".py"):
        tree = ast.parse(source, filename=path)
        if path.startswith("scripts/bench_decode_"):
            parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [(node.module or "").split(".")[0]]
                else:
                    continue
                if any(module in {"mlx", "mlx_lm"} for module in modules):
                    uses += 1
                    owner = node
                    in_mlx_method = False
                    while owner is not None:
                        if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            parent = parents.get(owner)
                            if isinstance(parent, ast.ClassDef) and parent.name == "MlxAdapter":
                                in_mlx_method = True
                            break
                        owner = parents.get(owner)
                    if not in_mlx_method:
                        problems.append(f"line {node.lineno}: MLX import is outside an MlxAdapter method")
            if any(isinstance(node, (ast.Import, ast.ImportFrom)) and
                   ((isinstance(node, ast.Import) and any(a.name.split('.')[0] in {'mlx','mlx_lm'} for a in node.names)) or
                    (isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] in {'mlx','mlx_lm'}))
                   for node in tree.body):
                problems.append("decode coordinator imports MLX at module scope")
            shared_path = REPO / "scripts/bench_decode_harness.py"
            shared = shared_path.read_text()
            if not decode_worker_boundary(shared):
                problems.append("shared decode harness does not constrain MLX to its durable worker entry")
            return uses, problems
        uses, calls = _mlx_call_violations(tree)
        problems.extend(calls)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                if (module or "").split(".")[0] in {"mlx", "mlx_lm"}:
                    problems.append(f"line {node.lineno}: module-level MLX import")
        if uses:
            lists = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.List)
                and any(isinstance(e, ast.Constant) and e.value == "--mlx-child" for e in node.elts)
            ]
            argv = next(
                (
                    [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "MLX_LAUNCH_ARGV" for t in node.targets)
                    and isinstance(node.value, ast.List)
                ),
                None,
            )
            if argv is None or len(lists) != 1:
                problems.append("MLX is reachable but there is not exactly one MLX_LAUNCH_ARGV child launch list")
            else:
                if "--gpu-handoff" in argv:
                    problems.append("the MLX child launch names --gpu-handoff")
                if ("--durable" in argv) != durable:
                    problems.append("the MLX child launch's --durable does not match the entry's +quiet")
                if argv[:1] != ["--label"] or "--" not in argv:
                    problems.append("the MLX child launch is not `--label L -- COMMAND`")
            spreads = [
                argument for call in ast.walk(tree) if isinstance(call, ast.Call)
                for argument in call.args
                if isinstance(argument, ast.List) and any(
                    isinstance(e, ast.Starred) and ast.unparse(e.value) == "MLX_LAUNCH_ARGV" for e in argument.elts)
            ]
            if len(spreads) != 1 or "bench-command.sh" not in ast.unparse(spreads[0].elts[0]):
                problems.append("MLX_LAUNCH_ARGV is not spread exactly once after the bench-command.sh path")
        return uses, problems

    for command, body in _shell_units(source):
        match = SHELL_MLX_USE.search(command + "\n" + body)
        if match is None:
            continue
        uses += 1
        launch = PLAIN_LAUNCH.search(command)
        if launch is None or "--gpu-handoff" in command:
            problems.append(f"MLX used outside a plain bench-command.sh launch: {command.strip()[:100]}")
            continue
        if (launch.group("durable") is not None) != durable:
            problems.append(f"--durable does not match the entry's +quiet: {command.strip()[:100]}")
        in_command = SHELL_MLX_USE.search(command)
        if in_command is not None and in_command.start() < launch.end():
            problems.append(f"MLX words precede the `--` of the launch: {command.strip()[:100]}")
    return uses, problems


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
                        "gpu-handoff-child",
                        "gpu-handoff-child+quiet",
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
            "crates/*/benches/*.rs; crates/inference/examples/*.rs; "
            "crates/inference/src/bin/*.rs; crates/serve/examples/*.rs; "
            "crates/serve/src/bin/*.rs; README.md; "
            "crates/fann/examples/router_online.rs",
        )
        self.assertEqual(
            contract["rust_inventory_limitation"],
            "does not discover Rust tests, except the one tracked entry point below",
        )
        confirmed_outside = {"crates/tune/tests/bench_backward_737.rs"}
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
        non_measurement = data["non_measurement_surface"]
        self.assertTrue(non_measurement)
        for surface in non_measurement:
            with self.subTest(path=surface["path"]):
                self.assertTrue(surface["reason"])
                self.assertTrue((REPO / surface["path"]).is_file(), surface["path"])
        excluded = excluded_measurement_surfaces()
        non_measurement_paths = non_measurement_surfaces()
        self.assertEqual(len(excluded), sum(len(s["paths"]) for s in surfaces))
        self.assertEqual(len(non_measurement_paths), len(non_measurement))
        self.assertTrue(excluded.isdisjoint(non_measurement_paths))
        self.assertEqual(
            excluded | non_measurement_paths, discovered_declared_rust_inventory_paths()
        )

    def test_inventory_discovers_every_application_package_and_no_other(self):
        """An unclassified serve example or binary is discovered, so the
        equality above fails closed; other crates' examples stay outside."""

        fixture_repo = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for relative in (
            "crates/inference/examples/kept.rs",
            "crates/serve/examples/unclassified.rs",
            "crates/serve/src/bin/server.rs",
            "crates/serve/src/bin/split/main.rs",
            "crates/serve/benches/timed.rs",
            "crates/embed/examples/outside.rs",
            "crates/embed/src/bin/outside.rs",
        ):
            path = fixture_repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fn main() {}\n")
        self.assertEqual(
            discovered_declared_rust_inventory_paths(fixture_repo),
            {
                "README.md",
                "crates/inference/examples/kept.rs",
                "crates/serve/benches/timed.rs",
                "crates/serve/examples/unclassified.rs",
                "crates/serve/src/bin/server.rs",
                "crates/serve/src/bin/split/main.rs",
            },
        )

    def test_phase_event_binary_is_classified_as_measurement(self):
        """qwen35_generate.rs's --emit-phase-events mode is a real timing
        surface (the CPU flagship lane driven by
        bench_cpu_flagship_supervisor.py), so classifying it as
        non_measurement_surface is wrong regardless of how the manifest
        entry is worded. Tied to source content, not just the manifest, so
        reverting the classification without also removing the marker
        still fails here."""

        path = "crates/inference/src/bin/qwen35_generate.rs"
        source = (REPO / path).read_text()
        self.assertIn("--emit-phase-events", source)
        self.assertIn(path, excluded_measurement_surfaces())
        self.assertNotIn(path, non_measurement_surfaces())

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

    def test_gpu_handoff_launch_is_measurement_evidence(self):
        """Mutation-sensitive: a script that only delegates to the handoff launcher
        must still be discovered as a measurement entry point, or its role could
        drift to consumer/none without any test noticing."""

        fixture_repo = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scripts = fixture_repo / "scripts"
        scripts.mkdir()
        (scripts / "bench_handoff_child.sh").write_text(
            "#!/usr/bin/env bash\n"
            '"$REPO/scripts/bench-command.sh" --gpu-handoff --label x -- \\\n'
            "    cargo run --locked --release -p lattice-inference --bin eval_perplexity\n"
        )
        (scripts / "bench_handoff_child.py").write_text(
            "#!/usr/bin/env python3\n"
            'HANDOFF_ARGV = ["--gpu-handoff", "--label", "x", "--"]\n'
        )
        (scripts / "bench_handoff_comment.sh").write_text(
            "#!/usr/bin/env bash\n"
            "# bench-command.sh --gpu-handoff is described here, never run\n"
        )

        evidence = discovered_measurement_evidence(fixture_repo)
        self.assertEqual(
            evidence["scripts/bench_handoff_child.sh"],
            {"GPU handoff measurement launch"},
        )
        self.assertEqual(
            evidence["scripts/bench_handoff_child.py"],
            {"GPU handoff measurement launch"},
        )
        self.assertNotIn("scripts/bench_handoff_comment.sh", evidence)

    def test_migrated_scripts_are_measurement_evidence_with_a_handoff_supervision(self):
        """The three scripts that delegate their binary to the handoff launcher are
        found by the source scan and carry a handoff supervision class."""

        evidence = discovered_measurement_evidence()
        entries = manifest_entries()
        for path, supervision in (
            ("scripts/bench_decode_slopefit.sh", "gpu-handoff-child+quiet"),
            ("scripts/bench_quality.sh", "gpu-handoff-child+quiet"),
            ("scripts/compare_logits.py", "gpu-handoff-child"),
            ("scripts/bench_decode_harness.py", "gpu-handoff-child+quiet"),
            ("scripts/bench_decode_adapters_apples_to_apples.py", "gpu-handoff-child+quiet"),
            ("scripts/bench_decode_adapters_apples_precise.py", "gpu-handoff-child+quiet"),
            ("scripts/bench_decode_adapters_q4_apples.py", "gpu-handoff-child+quiet"),
            ("scripts/bench_decode_adapters_context_scaling.py", "gpu-handoff-child+quiet"),
            ("scripts/bench_decode_adapters_agentic.py", "gpu-handoff-child+quiet"),
        ):
            with self.subTest(path=path):
                if path in {
                    "scripts/bench_decode_adapters_apples_to_apples.py",
                    "scripts/bench_decode_adapters_apples_precise.py",
                    "scripts/bench_decode_adapters_q4_apples.py",
                    "scripts/bench_decode_adapters_context_scaling.py",
                    "scripts/bench_decode_adapters_agentic.py",
                }:
                    self.assertIn("decode lattice handoff", evidence[path])
                else:
                    self.assertIn("GPU handoff measurement launch", evidence[path])
                self.assertEqual(entries[path]["role"], "measurement")
                self.assertEqual(entries[path]["supervision"], supervision)

    def test_decode_worker_source_contract(self):
        """Static checks see launch grammar, worker ordering, and the MLX import boundary."""
        harness_source = (REPO / "scripts/bench_decode_harness.py").read_text()
        self.assertTrue(decode_worker_boundary(harness_source))
        for path in (
            "scripts/bench_decode_adapters_agentic.py",
            "scripts/bench_decode_adapters_apples_precise.py",
            "scripts/bench_decode_adapters_apples_to_apples.py",
            "scripts/bench_decode_adapters_context_scaling.py",
            "scripts/bench_decode_adapters_q4_apples.py",
        ):
            source = (REPO / path).read_text()
            self.assertIn("harness.run_lattice_handoff", source)
            uses, violations = mlx_launch_report(path, source, durable=True)
            self.assertGreater(uses, 0)
            self.assertEqual(violations, [])
            self.assertNotIn('if __name__ == "__main__"', source)

    def test_decode_publication_follows_all_requested_handoffs(self):
        agentic_source = (REPO / "scripts/bench_decode_adapters_agentic.py").read_text()
        tree = ast.parse(agentic_source)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        text = ast.unparse(main)
        self.assertLess(text.index("for ctx in contexts"), text.index("for ctx in completed_contexts"))
        self.assertLess(text.index("for ctx in completed_contexts"), text.index("shutil.copy2"))

        context_source = (REPO / "scripts/bench_decode_adapters_context_scaling.py").read_text()
        context_tree = ast.parse(context_source)
        context_main = next(node for node in context_tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        context_text = ast.unparse(context_main)
        self.assertLess(context_text.index("harness.run_profile"), context_text.index("shutil.copy2"))

    def test_linkedin_post_script_is_retired_from_inventory(self):
        self.assertFalse((REPO / "scripts/bench_linkedin_post.py").exists())
        self.assertNotIn("scripts/bench_linkedin_post.py", MANIFEST.read_text())

    def test_handoff_launch_grammar_is_parsed_from_source(self):
        """Mutation-sensitive: the parser behind the live-guard check reads the launch
        words, so a flipped --durable or an unlisted binary is seen."""

        shell = (
            '(cd "$REPO" && "$REPO/scripts/bench-command.sh" --gpu-handoff --durable '
            "--label q -- \\\n"
            "    cargo run --locked --release -p lattice-inference --bin eval_perplexity \\\n"
            '    --features metal-gpu -- "$@")\n'
        )
        self.assertEqual(
            gpu_handoff_launches("scripts/x.sh", shell),
            [
                {
                    "durable": " --durable",
                    "label": "q",
                    "bin": "eval_perplexity",
                    "features": "metal-gpu",
                }
            ],
        )
        self.assertEqual(gpu_handoff_launches("scripts/x.sh", "# " + shell), [])
        plain = shell.replace(" --durable", "")
        self.assertIsNone(gpu_handoff_launches("scripts/x.sh", plain)[0]["durable"])
        python = (
            'HANDOFF_ARGV = ["--gpu-handoff", "--label", "l", "--", "cargo", "run", '
            '"--locked", "--release", "-p", "lattice-inference", "--bin", '
            '"bench_logit_dump", "--features", "metal-gpu,f16"]\n'
            'subprocess.run(["scripts/bench-command.sh", *HANDOFF_ARGV])\n'
        )
        self.assertEqual(
            [launch["bin"] for launch in gpu_handoff_launches("scripts/x.py", python)],
            ["bench_logit_dump"],
        )
        self.assertEqual(
            gpu_handoff_launches("scripts/x.py", python.replace("*HANDOFF_ARGV", "")),
            [],
        )

    def test_handoff_children_reach_mlx_only_through_a_plain_launch(self):
        """Known positives: the check must see MLX in the two scripts that use it, or a
        clean result on them would mean nothing."""

        entries = manifest_entries()
        for path, minimum in (
            ("scripts/bench_decode_slopefit.sh", 0),
            ("scripts/bench_quality.sh", 1),
            ("scripts/compare_logits.py", 1),
            ("scripts/bench_decode_harness.py", 0),
            ("scripts/bench_decode_adapters_apples_to_apples.py", 1),
            ("scripts/bench_decode_adapters_apples_precise.py", 1),
            ("scripts/bench_decode_adapters_q4_apples.py", 1),
            ("scripts/bench_decode_adapters_context_scaling.py", 1),
            ("scripts/bench_decode_adapters_agentic.py", 1),
        ):
            with self.subTest(path=path):
                durable = entries[path]["supervision"].endswith("+quiet")
                uses, violations = mlx_launch_report(path, (REPO / path).read_text(), durable=durable)
                self.assertEqual(violations, [])
                self.assertEqual(uses >= 1, minimum >= 1)

    def test_mlx_reachability_check_flags_the_unlocked_shapes(self):
        """Mutation-sensitive: the earlier shapes (MLX run bare, or in the parent) are reported."""

        compliant_shell = (
            '"$REPO/scripts/bench-command.sh" --durable --label q-mlx -- \\\n'
            '  uv run --quiet --with mlx-lm python3 - "$A" > "$T" 2>"$L" <<\'PY\'\n'
            "import mlx.core as mx\nPY\n"
            "# uv run --with mlx-lm is described here, never run\n"
            'echo "MLX cross-check" ; awk \'$1 == "mlx"\' file\n'
        )
        self.assertEqual(mlx_launch_report("x.sh", compliant_shell, durable=True), (1, []))
        bare = compliant_shell.replace('"$REPO/scripts/bench-command.sh" --durable --label q-mlx -- \\\n  ', "")
        uses, problems = mlx_launch_report("x.sh", bare, durable=True)
        self.assertEqual((uses, len(problems)), (1, 1))
        self.assertIn("outside a plain bench-command.sh launch", problems[0])
        no_durable = compliant_shell.replace("--durable ", "")
        self.assertIn("--durable does not match", mlx_launch_report("x.sh", no_durable, durable=True)[1][0])
        self.assertIn("--durable does not match", mlx_launch_report("x.sh", compliant_shell, durable=False)[1][0])
        handoff = compliant_shell.replace("--durable ", "--gpu-handoff --durable ")
        self.assertIn("outside a plain", mlx_launch_report("x.sh", handoff, durable=True)[1][0])
        body_only = 'python3 - <<\'PY\'\nfrom mlx_lm import load\nPY\n'
        self.assertEqual(mlx_launch_report("x.sh", body_only, durable=False)[0], 1)
        self.assertEqual(len(mlx_launch_report("x.sh", body_only, durable=False)[1]), 1)
        self.assertEqual(mlx_launch_report("x.sh", "# import mlx\n", durable=False), (0, []))

        compliant_python = (
            "import subprocess, sys\n"
            "MLX_CHILD_DIR = None\n"
            "MLX_LAUNCH_ARGV = ['--label', 'l', '--', sys.executable, 'f.py', '--mlx-child']\n"
            "def tokenize():\n    from mlx_lm import load\n"
            "def run_child(d):\n    tokenize()\n"
            "def collect():\n    subprocess.run([str(R / 'scripts' / 'bench-command.sh'), *MLX_LAUNCH_ARGV, 'd'])\n"
            "def main():\n"
            "    if MLX_CHILD_DIR is not None:\n        run_child(MLX_CHILD_DIR)\n        return\n"
            "    collect()\n"
            "if __name__ == '__main__':\n    main()\n"
        )
        self.assertEqual(mlx_launch_report("x.py", compliant_python, durable=False), (1, []))

        def flagged(source, needle, durable=False):
            _, found = mlx_launch_report("x.py", source, durable=durable)
            self.assertTrue(any(needle in item for item in found), (needle, found))

        in_parent = compliant_python.replace(
            "    collect()\n", "    tokenize()\n    collect()\n")
        flagged(in_parent, "tokenize() reaches MLX outside the child branch")
        in_parent_via_child = compliant_python.replace("    collect()\n", "    run_child('d')\n    collect()\n")
        flagged(in_parent_via_child, "run_child() reaches MLX outside the child branch")
        unguarded = compliant_python.replace("if MLX_CHILD_DIR is not None:", "if True:")
        flagged(unguarded, "run_child() reaches MLX outside the child branch")
        flagged(compliant_python.replace("import subprocess, sys\n", "import subprocess, sys\nimport mlx.core\n"), "module-level MLX import")
        without_launcher = compliant_python.replace(
            "[str(R / 'scripts' / 'bench-command.sh'), *MLX_LAUNCH_ARGV, 'd']", "[sys.executable, 'f.py', '--mlx-child', 'd']")
        flagged(without_launcher, "not exactly one MLX_LAUNCH_ARGV")
        no_spread = compliant_python.replace("*MLX_LAUNCH_ARGV, 'd'", "'d'")
        flagged(no_spread, "not spread exactly once")
        flagged(compliant_python.replace("'--label', 'l', '--',", "'--gpu-handoff', '--label', 'l', '--',"), "names --gpu-handoff")
        flagged(compliant_python.replace("'--label', 'l'", "'--durable', '--label', 'l'"), "--durable does not match")
        flagged(compliant_python, "--durable does not match", durable=True)
        other_launcher = compliant_python.replace("'bench-command.sh'", "'run.sh'")
        flagged(other_launcher, "not spread exactly once")

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

    def assert_gpu_handoff_child(self, path: str, source: str, supervision: str) -> None:
        """A handoff child delegates its binary and never supervises around it."""

        for token in SELF_SUPERVISION_TOKENS:
            if token == "bench_supervision" and path == "scripts/bench_decode_harness.py":
                continue  # The shared worker entry samples the outer plain launch witness.
            self.assertNotIn(token, source)
        # Any other GPU pass (MLX) is reachable only through a plain launch under both locks.
        _, violations = mlx_launch_report(path, source, durable=supervision.endswith("+quiet"))
        self.assertEqual(violations, [])
        if path.startswith("scripts/bench_decode_adapters_"):
            self.assertIn("harness.run_lattice_handoff", source)
            shared_source = (REPO / "scripts/bench_decode_harness.py").read_text()
            launches = gpu_handoff_launches("scripts/bench_decode_harness.py", shared_source)
        else:
            launches = gpu_handoff_launches(path, source)
        self.assertTrue(launches, f"{path} has no GPU handoff launch")
        spec = importlib.util.spec_from_file_location(
            "bench_admission_for_inventory", REPO / "scripts/lib/bench_admission.py"
        )
        assert spec is not None and spec.loader is not None
        admission = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(admission)
        manifest = tomllib.loads((REPO / "crates/inference/Cargo.toml").read_text())
        declared = {item["name"]: item for item in manifest.get("bin", [])}
        for launch in launches:
            with self.subTest(binary=launch["bin"]):
                # +quiet means the before/after ambient-idle checks are requested.
                self.assertEqual(
                    launch["durable"] is not None, supervision.endswith("+quiet")
                )
                self.assertIn(launch["bin"], admission.BINS)
                required = admission.BINS[launch["bin"]] | set(
                    declared.get(launch["bin"], {}).get("required-features", [])
                )
                self.assertLessEqual(
                    required,
                    set(admission.feature_closure(manifest, launch["features"])),
                )

    def test_every_measurement_entry_has_a_live_guard(self):
        """Every inventoried measurement or supervisor enters the shared backend."""

        for path, entry in manifest_entries().items():
            if entry["role"] not in {"measurement", "supervisor"}:
                continue
            source_path = REPO / path
            source = source_path.read_text()
            with self.subTest(path=path):
                if entry["supervision"].startswith("gpu-handoff-child"):
                    self.assert_gpu_handoff_child(path, source, entry["supervision"])
                elif entry["role"] == "supervisor" or path == "scripts/bench-compare.sh":
                    commands = _shell_commands(source)
                    backend_variables = {
                        tokens[0].split("=", 1)[0]
                        for _, tokens in commands
                        if "=" in tokens[0]
                        and tokens[0].split("=", 1)[1].endswith("/scripts/lib/bench_supervision.py")
                    }
                    self.assertTrue(
                        any(
                            tokens[0] == "exec"
                            and any(
                                "bench_supervision.py" in token
                                or token in {f"${name}" for name in backend_variables}
                                for token in tokens
                            )
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

    def test_new_supervisor_cannot_skip_the_shared_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            relative = "scripts/bench_new_supervisor.sh"
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_text("#!/bin/bash\nexec python3 measurement.py\n")
            entries = {relative: {"role": "supervisor", "supervision": "both-locks"}}
            with mock.patch(__name__ + ".REPO", root), mock.patch(
                __name__ + ".manifest_entries", return_value=entries
            ):
                result = unittest.TestResult()
                InventoryContract("test_every_measurement_entry_has_a_live_guard").run(result)
            self.assertEqual(result.errors, [])
            self.assertEqual(len(result.failures), 1)
            self.assertIn(relative, result.failures[0][0].id())

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
                    gate_invocations = [
                        (line_number, _shell_command_argv(tokens))
                        for line_number, tokens in commands
                        if len(_shell_command_argv(tokens)) >= 2
                        and _shell_command_argv(tokens)[1]
                        == "scripts/perf-bench-gate.py"
                    ]
                    gate_calls = [line_number for line_number, _ in gate_invocations]
                    self.assertTrue(gate_calls)
                    # The interpreter is resolved by the harness, not taken from
                    # PATH: a bare python3 here is the defect this asserts against.
                    self.assertFalse(
                        [
                            line_number
                            for line_number, argv in gate_invocations
                            if argv[0].strip('"') == "python3"
                        ]
                    )
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


class _FailOnEmptyTestProgram(unittest.TestProgram):
    def runTests(self) -> None:
        if self.test.countTestCases() == 0:
            raise SystemExit("no tests collected")
        super().runTests()


if __name__ == "__main__":
    _FailOnEmptyTestProgram()

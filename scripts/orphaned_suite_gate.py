#!/usr/bin/env python3
"""Refuse when a test module under `tests/` is invoked by no workflow step.

Two suites in this repo were discovered by hand, on the same day, to be running
nowhere. Both looked wired. `tests/test_perf_bench_gate_resolution.py` sat beside
suites that CI runs and was simply never added. `tests/test_e2e_parity_reference.py`
was worse: `e2e-parity.yml` names the file, but only inside the `paths:` regex that
decides whether the parity JOB runs, so a reader grepping the workflow directory for
the filename finds it and stops. A file that is named in a workflow and never
executed by one is the failure this gate exists for, which is why coverage is read
from `run:` bodies alone and from nothing else in the document.

The population is derived rather than globbed: a file under `tests/` counts as a
test module when its AST actually defines tests, so a helper named `test_utils.py`
is not demanded and a suite with an unconventional name is not excused.

Exit 0 when every discovered module is invoked, 1 when any is not (each orphan named
on its own line), 2 when the gate could not read its own inputs.
"""

from __future__ import annotations

import argparse
import ast
import re
import shlex
import sys
import tempfile
from pathlib import Path

EXIT_OK = 0
EXIT_ORPHANED = 1
EXIT_ERROR = 2

RUN_KEY = re.compile(r"^(?P<indent>\s*)(?:-\s+)?run:(?P<rest>.*)$")
BLOCK_SCALAR = re.compile(r"^[|>][+-]?\d*\s*$")
TOKEN = re.compile(r"[A-Za-z0-9_./-]+")


def run_command_text(workflow: str) -> list[str]:
    """Every `run:` value in a workflow, inline and block-scalar, and nothing else.

    `paths:`, `paths-ignore:`, `if:` and step names are deliberately not read: a
    filename mentioned there is precisely the decoy this gate was written for.
    """
    commands: list[str] = []
    lines = workflow.splitlines()
    index = 0
    while index < len(lines):
        match = RUN_KEY.match(lines[index])
        if match is None:
            index += 1
            continue
        rest = match.group("rest").strip()
        key_indent = len(match.group("indent"))
        index += 1
        if not BLOCK_SCALAR.match(rest):
            if rest:
                commands.append(rest)
            continue
        body: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.strip() and len(line) - len(line.lstrip()) <= key_indent:
                break
            body.append(line)
            index += 1
        commands.append("\n".join(body))
    return commands


def tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        # Workflow commands carry `${{ }}` expressions and shell quoting that shlex
        # rejects. A failed split must not read as "no tokens", which would report
        # every module orphaned, so fall back to a lexical scan of the same text.
        return TOKEN.findall(command)


def is_test_module(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                if name.endswith("TestCase"):
                    return True
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            return True
    return False


def discover_modules(tests_dir: Path) -> list[Path]:
    modules = []
    for path in sorted(tests_dir.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as error:
            raise SystemExit(f"orphaned-suite-gate: cannot read {path}: {error}")
        try:
            if is_test_module(source):
                modules.append(path)
        except SyntaxError as error:
            raise SystemExit(f"orphaned-suite-gate: cannot parse {path}: {error}")
    return modules


PY_HEADS = {"python", "python3", "pytest", "py.test", "coverage", "unittest"}
# Leading words that stand in front of the real command without being it.
WRAPPERS = {"uv", "run", "nohup", "time", "env", "exec", "if", "then", "elif", "!", "while"}
SPLIT_OPERATORS = re.compile(r"&&|\|\||;|\|")


def simple_commands(text: str):
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        for part in SPLIT_OPERATORS.split(line):
            part = part.strip()
            if part:
                yield part


def python_arguments(command: str) -> list[str] | None:
    """The arguments of `command` when it invokes Python, else None.

    A filename inside a `grep -E` path filter is the decoy this gate exists for, and
    reading `run:` bodies is not enough to exclude it: e2e-parity.yml implements its
    path filter as a shell grep inside a run step, so the module name really does sit
    in a run body while nothing executes it. Only the arguments of a Python-headed
    command count as an invocation.
    """
    argv = tokens(command)
    index = 0
    while index < len(argv):
        word = argv[index]
        if word in WRAPPERS or word.startswith("-") or ("=" in word and not word.startswith("/")):
            index += 1
            continue
        break
    if index >= len(argv):
        return None
    head = Path(argv[index]).name
    if head not in PY_HEADS:
        return None
    return argv[index + 1 :]


def invoked(
    commands: list[str], tests_dir: Path, root: Path
) -> tuple[set[str], set[str], bool]:
    """Returns script-form hits, module-form hits, and whether a step runs the whole tree."""
    hits: set[str] = set()
    module_hits: set[str] = set()
    whole_tree = False
    rel_tests = tests_dir.relative_to(root).as_posix()
    for command in commands:
        for simple in simple_commands(command):
            argv = python_arguments(simple)
            if argv is None:
                continue
            discovers = "discover" in argv
            for token in argv:
                stripped = token.strip("'\"")
                if stripped.rstrip("/") == rel_tests:
                    whole_tree = True
                    continue
                if stripped.endswith(".py"):
                    candidate = stripped.lstrip("./")
                    if candidate.startswith(f"{rel_tests}/"):
                        hits.add(candidate)
                    continue
                if "/" not in stripped and stripped.startswith(f"{rel_tests}."):
                    module_hits.add(f"{stripped.replace('.', '/')}.py")
            if discovers and any(
                token.strip("'\"").rstrip("/") == rel_tests for token in argv
            ):
                whole_tree = True
    return hits, module_hits, whole_tree


def script_runs_nothing(source: str) -> str | None:
    """Why `python3 <file>` would not run this module's tests, or None when it would.

    A `unittest.main()` guard is what makes the script form execute anything, and this
    repo invokes every Python suite that way. With no guard the run prints nothing and
    exits 0, which reads exactly like a suite that passed; with a guard that is not the
    last top-level statement, `unittest.main()` calls `sys.exit` before the classes
    below it are ever defined, so the run is green over a subset. Both were live in
    this tree when the check was written.
    """
    tree = ast.parse(source)
    guards = [
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.If)
        and "__name__" in ast.dump(node.test)
        and "__main__" in ast.dump(node.test)
    ]
    if not guards:
        return "no `if __name__ == \"__main__\"` block, so running it as a script runs zero tests and exits 0"
    if len(guards) > 1:
        return f"{len(guards)} `__main__` blocks; the first one exits before the rest of the module is defined"
    if guards[0] != len(tree.body) - 1:
        return "its `__main__` block is not the last top-level statement, so it exits before the definitions below it"
    return None


def orphans(root: Path) -> list[str]:
    tests_dir = root / "tests"
    workflow_dir = root / ".github" / "workflows"
    if not tests_dir.is_dir():
        raise SystemExit(f"orphaned-suite-gate: no tests directory at {tests_dir}")
    if not workflow_dir.is_dir():
        raise SystemExit(f"orphaned-suite-gate: no workflow directory at {workflow_dir}")
    workflows = sorted(workflow_dir.glob("*.yml")) + sorted(workflow_dir.glob("*.yaml"))
    if not workflows:
        raise SystemExit(f"orphaned-suite-gate: no workflow files under {workflow_dir}")
    commands: list[str] = []
    for workflow in workflows:
        commands.extend(run_command_text(workflow.read_text(encoding="utf-8")))
    if not commands:
        raise SystemExit("orphaned-suite-gate: parsed no run: steps at all; the parser changed")
    modules = discover_modules(tests_dir)
    if not modules:
        raise SystemExit(f"orphaned-suite-gate: found no test module under {tests_dir}")
    hits, module_hits, whole_tree = invoked(commands, tests_dir, root)
    covered = hits | module_hits
    return [
        module.relative_to(root).as_posix()
        for module in modules
        if not whole_tree and module.relative_to(root).as_posix() not in covered
    ]


def silent_scripts(root: Path) -> list[tuple[str, str]]:
    """Script-invoked modules whose script run would execute nothing, or a subset."""
    tests_dir = root / "tests"
    workflow_dir = root / ".github" / "workflows"
    commands: list[str] = []
    for workflow in sorted(workflow_dir.glob("*.yml")) + sorted(workflow_dir.glob("*.yaml")):
        commands.extend(run_command_text(workflow.read_text(encoding="utf-8")))
    hits, _module_hits, _whole = invoked(commands, tests_dir, root)
    findings = []
    for relative in sorted(hits):
        path = root / relative
        if not path.is_file():
            continue
        reason = script_runs_nothing(path.read_text(encoding="utf-8"))
        if reason is not None:
            findings.append((relative, reason))
    return findings


SELFTEST_WORKFLOW = """\
name: fixture
on:
  pull_request:
    paths:
      - 'tests/test_orphan.py'
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - name: tests/test_orphan.py
        run: echo "named in a step name, which is not an invocation"
      - name: covered
        run: |
          python3 tests/test_covered.py -v
          echo done
"""

SELFTEST_SUITE = """\
import unittest


class Fixture(unittest.TestCase):
    def test_one(self):
        self.assertTrue(True)
"""

SELFTEST_SUITE_RUNNABLE = SELFTEST_SUITE + """

if __name__ == "__main__":
    unittest.main()
"""

SELFTEST_HELPER = """\
def helper():
    return 1
"""

SELFTEST_SUITE_NO_GUARD = SELFTEST_SUITE

SELFTEST_SUITE_EARLY_GUARD = SELFTEST_SUITE + """

if __name__ == "__main__":
    unittest.main()


class Truncated(unittest.TestCase):
    def test_never_defined_under_the_script_form(self):
        self.assertTrue(True)
"""


def selftest() -> int:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        (root / "tests").mkdir()
        (root / ".github" / "workflows").mkdir(parents=True)
        (root / ".github" / "workflows" / "fixture.yml").write_text(SELFTEST_WORKFLOW)
        (root / "tests" / "test_covered.py").write_text(SELFTEST_SUITE_RUNNABLE)
        (root / "tests" / "test_orphan.py").write_text(SELFTEST_SUITE_RUNNABLE)
        (root / "tests" / "test_helper_only.py").write_text(SELFTEST_HELPER)

        found = orphans(root)
        # The decoy is the point: test_orphan.py is named twice in the workflow, once
        # in `paths:` and once as a step name, and executed by neither.
        if found != ["tests/test_orphan.py"]:
            print(f"selftest FAILED: expected the decoy to be the only orphan, got {found}")
            return EXIT_ERROR
        # A file with no tests is not demanded.
        if "tests/test_helper_only.py" in found:
            print("selftest FAILED: a module defining no tests must not be in the population")
            return EXIT_ERROR

        wired = SELFTEST_WORKFLOW.replace(
            "          python3 tests/test_covered.py -v",
            "          python3 tests/test_covered.py -v\n          python3 tests/test_orphan.py -v",
        )
        (root / ".github" / "workflows" / "fixture.yml").write_text(wired)
        if orphans(root):
            print("selftest FAILED: invoking the decoy must clear it")
            return EXIT_ERROR

        (root / ".github" / "workflows" / "fixture.yml").write_text(
            SELFTEST_WORKFLOW.replace(
                "          python3 tests/test_covered.py -v", "          pytest tests/"
            )
        )
        if orphans(root):
            print("selftest FAILED: a whole-tree invocation must cover every module")
            return EXIT_ERROR

        # The second check: a module CI invokes as a script, that runs nothing.
        (root / ".github" / "workflows" / "fixture.yml").write_text(SELFTEST_WORKFLOW)
        if silent_scripts(root):
            print("selftest FAILED: a suite with a trailing main guard must be accepted")
            return EXIT_ERROR
        (root / "tests" / "test_covered.py").write_text(SELFTEST_SUITE_NO_GUARD)
        silent = silent_scripts(root)
        if [path for path, _ in silent] != ["tests/test_covered.py"]:
            print(f"selftest FAILED: a script-invoked module with no main guard must refuse: {silent}")
            return EXIT_ERROR
        (root / "tests" / "test_covered.py").write_text(SELFTEST_SUITE_EARLY_GUARD)
        silent = silent_scripts(root)
        if [path for path, _ in silent] != ["tests/test_covered.py"]:
            print(f"selftest FAILED: a non-final main guard must refuse: {silent}")
            return EXIT_ERROR

    print("orphaned-suite-gate: selftest OK -- a decoy named but never run is refused")
    return EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run the gate against a fixture tree carrying a deliberately unreferenced module",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (default: the repo this script lives in)",
    )
    args = parser.parse_args()
    if args.selftest:
        return selftest()

    root = args.root.resolve()
    found = orphans(root)
    silent = silent_scripts(root)
    if not found and not silent:
        print("orphaned-suite-gate: OK -- every test module under tests/ is invoked and executes")
        return EXIT_OK
    if found:
        print("orphaned-suite-gate: these test modules are invoked by no workflow step:")
        for path in found:
            print(f"  {path}")
        print(
            "Add a step that runs each one, or delete it. A module named only in a "
            "`paths:` filter, a step name, or a shell path-filter regex is not invoked."
        )
    if silent:
        print("orphaned-suite-gate: these modules are invoked as scripts but would run nothing:")
        for path, reason in silent:
            print(f"  {path}: {reason}")
    return EXIT_ORPHANED


if __name__ == "__main__":
    sys.exit(main())

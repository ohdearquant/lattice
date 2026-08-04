"""Tests for the fail-closed CI changed-file range selector."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci-changed-files.sh"
_ROOT = _SCRIPT.parent.parent
_ZERO_SHA = "0" * 40
_GIT = ("git", "-c", "core.hooksPath=/dev/null")
_REQUIRED_WORKFLOWS = (
    ".github/workflows/app-binaries.yml",
    ".github/workflows/cargo-audit.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/e2e-parity.yml",
)
_E2E_PARITY_WORKFLOW = _ROOT / ".github/workflows/e2e-parity.yml"
_CI_WORKFLOW = _ROOT / ".github/workflows/ci.yml"
_CARGO_AUDIT_WORKFLOW = _ROOT / ".github/workflows/cargo-audit.yml"
_APP_BINARIES_WORKFLOW = _ROOT / ".github/workflows/app-binaries.yml"
_DETECTOR_EVENTS = ("pull_request", "merge_group")
_FILTER_SHELL_COMMAND = (
    "/usr/bin/env",
    "-u",
    "BASH_ENV",
    "-u",
    "ENV",
    "PATH=/usr/bin:/bin",
    "/bin/bash",
    "--noprofile",
    "--norc",
    "-e",
    "-u",
    "-o",
    "pipefail",
)
_TRUSTED_FILTER_SHELL = " ".join((*_FILTER_SHELL_COMMAND, "{0}"))
_DETECTOR_LOAD = re.compile(
    r"^if /usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/git "
    r"--no-replace-objects(?: -c core\.quotePath=false)? show "
    r'"\$\{CI_BASE_SHA\}:scripts/ci-changed-files\.sh" '
    r'> "\$TRUSTED_DETECTOR"; then$',
    re.MULTILINE,
)
_TRUSTED_DETECTOR_EXECUTION = (
    "/usr/bin/env -i PATH=/usr/bin:/bin "
    'GITHUB_EVENT_NAME="$GITHUB_EVENT_NAME" '
    'CI_BASE_SHA="$CI_BASE_SHA" CI_HEAD_SHA="$CI_HEAD_SHA" '
    '/bin/sh "$TRUSTED_DETECTOR"'
)
_BASE_FORMAT_BLOCK = (
    'if [[ ! "$CI_BASE_SHA" =~ ^[0-9a-fA-F]{40}$ ]]; then',
    'echo "::error::base revision must be a nonempty '
    '40-character hexadecimal commit ID" >&2',
    "exit 2",
    "fi",
)
_TEMPORARY_FILE_BLOCK = (
    "TRUSTED_DETECTOR=$(mktemp)",
    "CHANGED_FILE=$(mktemp)",
    '''trap 'rm -f "$TRUSTED_DETECTOR" "$CHANGED_FILE"' EXIT''',
)
_EXPECTED_FILTER_PROLOGUES = {
    ".github/workflows/ci.yml": (
        "set -euo pipefail",
        *_BASE_FORMAT_BLOCK,
        f'if [ "$CI_BASE_SHA" = "{_ZERO_SHA}" ]; then',
        'echo "code=true" >> "$GITHUB_OUTPUT"',
        'echo "→ all-zero base revision: full matrix REQUIRED"',
        "exit 0",
        "fi",
        *_TEMPORARY_FILE_BLOCK,
    ),
    ".github/workflows/cargo-audit.yml": (
        "set -euo pipefail",
        'if [ "$GITHUB_EVENT_NAME" = "schedule" ] || '
        '[ "$GITHUB_EVENT_NAME" = "workflow_dispatch" ]; then',
        'echo "deps=true" >> "$GITHUB_OUTPUT"',
        'echo "→ non-diff trigger ($GITHUB_EVENT_NAME): audit REQUIRED"',
        "exit 0",
        "fi",
        *_BASE_FORMAT_BLOCK,
        f'if [ "$CI_BASE_SHA" = "{_ZERO_SHA}" ]; then',
        'echo "deps=true" >> "$GITHUB_OUTPUT"',
        'echo "→ all-zero base revision: audit REQUIRED"',
        "exit 0",
        "fi",
        *_TEMPORARY_FILE_BLOCK,
    ),
    ".github/workflows/app-binaries.yml": (
        "set -euo pipefail",
        'if [ "$GITHUB_EVENT_NAME" = "workflow_dispatch" ]; then',
        'echo "bins=true" >> "$GITHUB_OUTPUT"',
        'echo "swift=true" >> "$GITHUB_OUTPUT"',
        'echo "→ manual dispatch: app-binary and Swift builds REQUIRED"',
        "exit 0",
        "fi",
        *_BASE_FORMAT_BLOCK,
        f'if [ "$CI_BASE_SHA" = "{_ZERO_SHA}" ]; then',
        'echo "bins=true" >> "$GITHUB_OUTPUT"',
        'echo "swift=true" >> "$GITHUB_OUTPUT"',
        'echo "→ all-zero base revision: app-binary and Swift builds REQUIRED"',
        "exit 0",
        "fi",
        *_TEMPORARY_FILE_BLOCK,
    ),
    ".github/workflows/e2e-parity.yml": (
        "set -euo pipefail",
        'if [ "$GITHUB_EVENT_NAME" = "workflow_dispatch" ] || '
        '[ "$GITHUB_EVENT_NAME" = "schedule" ]; then',
        'echo "engine=true" >> "$GITHUB_OUTPUT"',
        'echo "tune=true" >> "$GITHUB_OUTPUT"',
        'echo "→ $GITHUB_EVENT_NAME trigger: full parity suite REQUIRED"',
        "exit 0",
        "fi",
        *_BASE_FORMAT_BLOCK,
        f'if [ "$CI_BASE_SHA" = "{_ZERO_SHA}" ]; then',
        'echo "engine=true" >> "$GITHUB_OUTPUT"',
        'echo "tune=true" >> "$GITHUB_OUTPUT"',
        'echo "→ all-zero base revision: full parity suite REQUIRED"',
        "exit 0",
        "fi",
        *_TEMPORARY_FILE_BLOCK,
    ),
}
_EXPECTED_WORKFLOW_ENV_KEYS = {
    ".github/workflows/ci.yml": ("CARGO_TERM_COLOR", "LATTICE_REQUIRE_FIXTURES"),
    ".github/workflows/cargo-audit.yml": ("CARGO_TERM_COLOR",),
    ".github/workflows/app-binaries.yml": (
        "CARGO_TERM_COLOR",
        "CARGO_INCREMENTAL",
    ),
    ".github/workflows/e2e-parity.yml": (
        "CARGO_TERM_COLOR",
        "CARGO_INCREMENTAL",
    ),
}
_EXPECTED_FILTER_ENVIRONMENT = (
    (
        "CI_BASE_SHA",
        "${{ github.event.pull_request.base.sha || "
        "github.event.merge_group.base_sha || github.event.before }}",
    ),
    (
        "CI_HEAD_SHA",
        "${{ github.event.merge_group.head_sha || github.sha }}",
    ),
)
_CHECKOUT_ACTION = (
    "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
)
_EXPECTED_CHANGE_JOB_KEYS = frozenset(("name", "runs-on", "outputs", "steps"))
_EXPECTED_CHANGE_JOB_NAMES = {
    ".github/workflows/ci.yml": "detect-code-changes",
    ".github/workflows/cargo-audit.yml": "detect-dependency-changes",
    ".github/workflows/app-binaries.yml": "detect-app-bin-changes",
    ".github/workflows/e2e-parity.yml": "detect-engine-changes",
}
_EXPECTED_CHANGE_JOB_OUTPUTS = {
    ".github/workflows/ci.yml": {
        "code": "${{ steps.filter.outputs.code }}",
        "oslist": "${{ steps.oslist.outputs.oslist }}",
        "featureoslist": "${{ steps.oslist.outputs.featureoslist }}",
    },
    ".github/workflows/cargo-audit.yml": {
        "deps": "${{ steps.filter.outputs.deps }}",
    },
    ".github/workflows/app-binaries.yml": {
        "bins": "${{ steps.filter.outputs.bins }}",
        "swift": "${{ steps.filter.outputs.swift }}",
    },
    ".github/workflows/e2e-parity.yml": {
        "engine": "${{ steps.filter.outputs.engine }}",
        "tune": "${{ steps.filter.outputs.tune }}",
    },
}
_EXPECTED_SELECTOR_OUTPUTS = {
    ".github/workflows/ci.yml": ("code",),
    ".github/workflows/cargo-audit.yml": ("deps",),
    ".github/workflows/app-binaries.yml": ("bins", "swift"),
    ".github/workflows/e2e-parity.yml": ("engine", "tune"),
}


class _WorkflowLoader(yaml.SafeLoader):
    pass


_WorkflowLoader.yaml_implicit_resolvers = {
    initial: [
        resolver
        for resolver in resolvers
        if resolver[0] != "tag:yaml.org,2002:bool"
    ]
    for initial, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_WorkflowLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def _construct_unique_mapping(
    loader: _WorkflowLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_WorkflowLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _require_mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise AssertionError(f"{location} must be a string-keyed mapping")
    return value


def _parse_workflow(contents: str, workflow: str) -> dict[str, object]:
    try:
        document = yaml.load(contents, Loader=_WorkflowLoader)
    except yaml.YAMLError as error:
        raise AssertionError(f"{workflow} is not valid workflow YAML: {error}") from error
    return _require_mapping(document, f"{workflow} document")


def _parsed_workflow_job(
    document: dict[str, object],
    job_id: str,
    workflow: str,
) -> dict[str, object]:
    jobs = _require_mapping(document.get("jobs"), f"{workflow} jobs")
    if job_id not in jobs:
        raise AssertionError(f"workflow job {job_id!r} is missing")
    return _require_mapping(jobs[job_id], f"{workflow} job {job_id!r}")


def _parsed_workflow_step_by_id(
    job: dict[str, object],
    step_id: str,
    workflow: str,
) -> tuple[int, dict[str, object]]:
    steps = job.get("steps")
    if not isinstance(steps, list):
        raise AssertionError(f"{workflow} job steps must be a sequence")
    matches = []
    for index, value in enumerate(steps):
        step = _require_mapping(value, f"{workflow} step {index + 1}")
        if step.get("id") == step_id:
            matches.append((index, step))
    if len(matches) != 1:
        raise AssertionError(
            f"workflow step id {step_id!r} must appear exactly once, "
            f"found {len(matches)}"
        )
    return matches[0]


def _workflow_job(contents: str, job_id: str) -> str:
    start_match = re.search(rf"^  {re.escape(job_id)}:\n", contents, re.MULTILINE)
    if start_match is None:
        raise AssertionError(f"workflow job {job_id!r} is missing")
    next_match = re.search(
        r"^  [a-z0-9][a-z0-9-]*:\n",
        contents[start_match.end() :],
        re.MULTILINE,
    )
    end = (
        len(contents)
        if next_match is None
        else start_match.end() + next_match.start()
    )
    return contents[start_match.start() : end]


def _workflow_step(job: str, step_name: str) -> str:
    marker = f"      - name: {step_name}\n"
    start = job.find(marker)
    if start < 0:
        raise AssertionError(f"workflow step {step_name!r} is missing")
    body_start = start + len(marker)
    next_match = re.search(
        r"^(?:      -(?: .*)?| {0,4}\S.*)$",
        job[body_start:],
        re.MULTILINE,
    )
    end = len(job) if next_match is None else body_start + next_match.start()
    return job[start:end]


def _workflow_step_by_id(job: str, step_id: str) -> str:
    marker = f"      - id: {step_id}\n"
    start = job.find(marker)
    if start < 0:
        raise AssertionError(f"workflow step id {step_id!r} is missing")
    body_start = start + len(marker)
    next_match = re.search(
        r"^(?:      -(?: .*)?| {0,4}\S.*)$",
        job[body_start:],
        re.MULTILINE,
    )
    end = len(job) if next_match is None else body_start + next_match.start()
    return job[start:end]


def _workflow_run_script_from_contents(
    contents: str,
    job_id: str,
    step_id: str,
    workflow: str = "workflow",
) -> str:
    document = _parse_workflow(contents, workflow)
    job = _parsed_workflow_job(document, job_id, workflow)
    _, step = _parsed_workflow_step_by_id(job, step_id, workflow)
    script = step.get("run")
    if not isinstance(script, str):
        raise AssertionError(f"workflow step id {step_id!r} has no run script")
    return script


def _workflow_run_script(workflow: Path, job_id: str, step_id: str) -> str:
    contents = workflow.read_text(encoding="utf-8")
    return _workflow_run_script_from_contents(
        contents,
        job_id,
        step_id,
        workflow.name,
    )


def _workflow_contract_key(workflow: str) -> str:
    matches = tuple(
        path
        for path in _REQUIRED_WORKFLOWS
        if workflow == path or workflow == Path(path).name
    )
    if len(matches) != 1:
        raise AssertionError(f"no unique filter contract for {workflow!r}")
    return matches[0]


def _assert_filter_execution_envelope(contents: str, workflow: str) -> str:
    contract_key = _workflow_contract_key(workflow)
    document = _parse_workflow(contents, workflow)
    if "defaults" in document:
        raise AssertionError(f"{workflow} has workflow defaults reaching filter step")

    workflow_environment = _require_mapping(
        document.get("env"),
        f"{workflow} workflow environment",
    )
    workflow_environment_keys = frozenset(workflow_environment)
    expected_workflow_environment_keys = frozenset(
        _EXPECTED_WORKFLOW_ENV_KEYS[contract_key]
    )
    if workflow_environment_keys != expected_workflow_environment_keys:
        raise AssertionError(
            f"{workflow} workflow environment keys must be exactly "
            f"{_EXPECTED_WORKFLOW_ENV_KEYS[contract_key]!r}, found "
            f"{tuple(workflow_environment)!r}"
        )

    job = _parsed_workflow_job(document, "changes", workflow)
    job_keys = frozenset(job)
    if job_keys != _EXPECTED_CHANGE_JOB_KEYS:
        raise AssertionError(
            f"{workflow} changes job schema must contain exactly "
            f"{tuple(sorted(_EXPECTED_CHANGE_JOB_KEYS))!r}, found "
            f"{tuple(sorted(job_keys))!r}"
        )
    if job.get("name") != _EXPECTED_CHANGE_JOB_NAMES[contract_key]:
        raise AssertionError(
            f"{workflow} changes job name must be "
            f"{_EXPECTED_CHANGE_JOB_NAMES[contract_key]!r}"
        )
    if job.get("runs-on") != "ubuntu-latest":
        raise AssertionError(
            f"{workflow} changes job runs-on must be 'ubuntu-latest'"
        )

    outputs = _require_mapping(job.get("outputs"), f"{workflow} changes outputs")
    expected_outputs = _EXPECTED_CHANGE_JOB_OUTPUTS[contract_key]
    if outputs != expected_outputs:
        raise AssertionError(
            f"{workflow} changes job output mappings must be exactly "
            f"{expected_outputs!r}, found {outputs!r}"
        )

    steps = job.get("steps")
    if not isinstance(steps, list):
        raise AssertionError(f"{workflow} changes job steps must be a sequence")
    filter_index, step = _parsed_workflow_step_by_id(job, "filter", workflow)
    if filter_index != 1:
        raise AssertionError(
            f"{workflow} filter step must immediately follow checkout"
        )
    checkout = _require_mapping(steps[0], f"{workflow} checkout step")
    expected_checkout = {
        "uses": _CHECKOUT_ACTION,
        "with": {"fetch-depth": 0},
    }
    if checkout != expected_checkout:
        raise AssertionError(
            f"{workflow} changes job must start with the canonical checkout step"
        )

    metadata_keys = frozenset(step)
    expected_metadata_keys = frozenset(("id", "env", "shell", "run"))
    if metadata_keys != expected_metadata_keys:
        raise AssertionError(
            f"{workflow} filter step metadata keys must be exactly "
            f"{tuple(sorted(expected_metadata_keys))!r}, found "
            f"{tuple(sorted(metadata_keys))!r}"
        )

    step_environment = _require_mapping(
        step.get("env"),
        f"{workflow} filter environment",
    )
    expected_filter_environment = dict(_EXPECTED_FILTER_ENVIRONMENT)
    if step_environment != expected_filter_environment:
        raise AssertionError(
            f"{workflow} filter environment must be exactly "
            f"{expected_filter_environment!r}, found {step_environment!r}"
        )

    if step.get("shell") != _TRUSTED_FILTER_SHELL:
        raise AssertionError(
            f"{workflow} filter shell command must be {_TRUSTED_FILTER_SHELL!r}"
        )
    script = step.get("run")
    if not isinstance(script, str):
        raise AssertionError(f"{workflow} filter step must have a run script")
    return script


def _assert_safe_filter_prologue(script: str, workflow: str) -> tuple[str, ...]:
    """Require the declared ordered grammar before a top-level detector load."""
    detector_load = _DETECTOR_LOAD.search(script)
    if detector_load is None:
        raise AssertionError(
            f"{workflow} does not load the base detector with the trusted "
            "/usr/bin/env and /usr/bin/git form; use the declared load shape"
        )
    prefix = script[: detector_load.start()]
    statements = []
    conditional_depth = 0
    for raw_line in prefix.splitlines():
        statement = raw_line.strip()
        if not statement or statement.startswith("#"):
            continue
        statements.append(statement)
        if statement.startswith("if ") and statement.endswith("; then"):
            conditional_depth += 1
        elif statement == "fi":
            if conditional_depth == 0:
                raise AssertionError(
                    f"{workflow} has an unbalanced conditional terminator "
                    "before detector loading"
                )
            conditional_depth -= 1

    if conditional_depth != 0:
        raise AssertionError(
            f"{workflow} must load the base detector at top level; "
            f"conditional nesting depth is {conditional_depth}"
        )

    contract_key = _workflow_contract_key(workflow)
    expected = _EXPECTED_FILTER_PROLOGUES[contract_key]
    actual = tuple(statements)
    if actual != expected:
        mismatch = next(
            (
                index
                for index in range(max(len(actual), len(expected)))
                if index >= len(actual)
                or index >= len(expected)
                or actual[index] != expected[index]
            ),
            0,
        )
        found = "<missing>" if mismatch >= len(actual) else repr(actual[mismatch])
        required = (
            "<end>" if mismatch >= len(expected) else repr(expected[mismatch])
        )
        raise AssertionError(
            f"{workflow} violates the ordered grammar before detector loading "
            f"at statement {mismatch + 1}: expected {required}, found {found}"
        )

    detector_line = detector_load.group(0)
    if detector_line.startswith((" ", "\t")):
        raise AssertionError(
            f"{workflow} must load the base detector at top level without "
            "indentation"
        )
    return actual


def _assert_trusted_detector_execution(script: str, workflow: str) -> None:
    detector_loads = tuple(_DETECTOR_LOAD.finditer(script))
    if len(detector_loads) != 1:
        raise AssertionError(
            f"{workflow} must load the trusted detector exactly once"
        )
    trusted_execution = (
        f'if {_TRUSTED_DETECTOR_EXECUTION} > "$CHANGED_FILE"; then'
    )
    lines = script.splitlines()
    if lines.count(trusted_execution) != 1:
        raise AssertionError(
            f"{workflow} must execute the trusted detector once with a clean "
            "environment and absolute shell"
        )

    detector_load = detector_loads[0].group(0)
    expected = (
        detector_load,
        "  :",
        "else",
        "  DETECTOR_STATUS=$?",
        '  echo "::error::unable to load change detector from base revision '
        '(status $DETECTOR_STATUS)" >&2',
        '  exit "$DETECTOR_STATUS"',
        "fi",
        trusted_execution,
        "  :",
        "else",
        "  DETECTOR_STATUS=$?",
        '  echo "::error::change detector failed with status '
        '$DETECTOR_STATUS" >&2',
        '  exit "$DETECTOR_STATUS"',
        "fi",
        'CHANGED=$(<"$CHANGED_FILE")',
    )
    start = lines.index(detector_load)
    actual = tuple(lines[start : start + len(expected)])
    if actual != expected:
        mismatch = next(
            (
                index
                for index in range(max(len(actual), len(expected)))
                if index >= len(actual)
                or index >= len(expected)
                or actual[index] != expected[index]
            ),
            0,
        )
        found = "<missing>" if mismatch >= len(actual) else repr(actual[mismatch])
        required = (
            "<end>" if mismatch >= len(expected) else repr(expected[mismatch])
        )
        raise AssertionError(
            f"{workflow} violates ordered detector execution at line "
            f"{mismatch + 1}: expected {required}, found {found}"
        )

    contract_key = _workflow_contract_key(workflow)
    classification_lines = lines[start + len(expected) :]
    if any(
        line.lstrip().startswith("#") and line.rstrip().endswith("\\")
        for line in classification_lines
    ):
        raise AssertionError(
            f"{workflow} violates ordered selector classification with a "
            "continued comment"
        )
    classification = tuple(
        line.strip()
        for line in classification_lines
        if line.strip() and not line.lstrip().startswith("#")
    )
    selector_writes = tuple(
        f'echo "{selector}={value}" >> "$GITHUB_OUTPUT"'
        for selector in _EXPECTED_SELECTOR_OUTPUTS[contract_key]
        for value in ("true", "false")
    )
    for statement in classification:
        if re.search(
            r"(?:^|[\s;&|()])(?:exit|return|exec)(?=$|[\s;&|()])",
            statement,
        ):
            raise AssertionError(
                f"{workflow} violates ordered selector classification with "
                f"an early terminating statement: {statement!r}"
            )
        if "GITHUB_OUTPUT" in statement and statement not in selector_writes:
            raise AssertionError(
                f"{workflow} violates ordered selector classification with "
                f"an unrecognized output statement: {statement!r}"
            )
    for write in selector_writes:
        if classification.count(write) != 1:
            raise AssertionError(
                f"{workflow} selector classification must write {write!r} "
                "exactly once after trusted detector execution"
            )


def _assert_safe_filter_workflow(contents: str, workflow: str) -> None:
    script = _assert_filter_execution_envelope(contents, workflow)
    _assert_safe_filter_prologue(script, workflow)
    _assert_trusted_detector_execution(script, workflow)


def _run_required_gate(
    workflow: Path,
    gate_id: str,
    selector_values: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    contents = workflow.read_text(encoding="utf-8")
    step = _workflow_step(_workflow_job(contents, gate_id), "Resolve gate")
    marker = "        run: |\n"
    if marker not in step:
        raise AssertionError(f"workflow gate {gate_id!r} has no run script")
    script = textwrap.dedent(step.split(marker, maxsplit=1)[1])

    def expression_value(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        selector_prefix = "needs.changes.outputs."
        if expression == "needs.changes.result":
            return "success"
        if expression.startswith(selector_prefix):
            return selector_values.get(expression.removeprefix(selector_prefix), "")
        if expression.startswith("needs.") and expression.endswith(".result"):
            return "success"
        if expression == "github.event.pull_request.draft":
            return "false"
        if expression == "github.event_name":
            return "pull_request"
        raise AssertionError(f"unhandled gate expression: {expression}")

    script = re.sub(r"\$\{\{\s*([^}]+?)\s*\}\}", expression_value, script)
    return subprocess.run(
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-e",
            "-u",
            "-o",
            "pipefail",
            "-c",
            script,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _require_tests_collected(test_suite: unittest.TestSuite) -> None:
    if test_suite.countTestCases() == 0:
        raise SystemExit("ERROR: no tests collected")


class _FailOnEmptyTestProgram(unittest.TestProgram):
    def runTests(self) -> None:
        _require_tests_collected(self.test)
        super().runTests()


def _engine_change_pattern(contents: str) -> re.Pattern[str]:
    script = _workflow_run_script_from_contents(
        contents,
        "changes",
        "filter",
        _E2E_PARITY_WORKFLOW.name,
    )
    match = re.search(
        r"""if\ grep\ -E\ '([^']+)'\ <<<"\$CHANGED"\ >/dev/null;\ then
            \s+echo\ "engine=true" """,
        script,
        re.VERBOSE,
    )
    if match is None:
        raise AssertionError("changes job engine classifier is missing")
    return re.compile(match.group(1))


class ChangedFilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tempdir.name)
        self._git("init", "-q", "-b", "main")
        self._git("config", "user.name", "CI Test")
        self._git("config", "user.email", "ci-test@example.invalid")

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def _git(self, *args: str) -> str:
        # Test helpers invoking real Git must disable repository hooks.
        result = subprocess.run(
            [*_GIT, *args],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _commit(self, path: str, contents: str) -> str:
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")
        self._git("add", path)
        self._git("commit", "-q", "-m", f"write {path}")
        return self._git("rev-parse", "HEAD")

    def _run(
        self,
        event: str,
        base_sha: str,
        head_sha: str,
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            GITHUB_EVENT_NAME=event,
            CI_BASE_SHA=base_sha,
            CI_HEAD_SHA=head_sha,
        )
        return subprocess.run(
            [str(_SCRIPT)],
            cwd=self.repo,
            env=env,
            check=check,
            capture_output=True,
            text=True,
        )

    def test_merge_group_reports_every_change_in_the_event_range(self) -> None:
        base = self._commit("README.md", "base\n")
        self._commit("docs/queue.md", "docs\n")
        head = self._commit("crates/inference/src/queue.rs", "code\n")

        result = self._run("merge_group", base, head)

        self.assertEqual(
            result.stdout.splitlines(),
            ["crates/inference/src/queue.rs", "docs/queue.md"],
        )

    def test_pull_request_uses_the_event_base(self) -> None:
        base = self._commit("README.md", "base\n")
        head = self._commit("crates/embed/src/change.rs", "code\n")

        result = self._run("pull_request", base, head)

        self.assertEqual(result.stdout.splitlines(), ["crates/embed/src/change.rs"])

    def test_multi_commit_push_is_not_reduced_to_the_last_commit(self) -> None:
        base = self._commit("README.md", "base\n")
        self._commit("crates/fann/src/change.rs", "code\n")
        head = self._commit("docs/followup.md", "docs\n")

        result = self._run("push", base, head)

        self.assertEqual(
            result.stdout.splitlines(),
            ["crates/fann/src/change.rs", "docs/followup.md"],
        )

    def test_rename_reports_the_relevant_source_and_destination(self) -> None:
        base = self._commit("crates/embed/src/moved.rs", "code\n")
        (self.repo / "docs").mkdir()
        self._git("mv", "crates/embed/src/moved.rs", "docs/moved.md")
        self._git("commit", "-q", "-m", "move code into docs")
        head = self._git("rev-parse", "HEAD")

        result = self._run("merge_group", base, head)

        self.assertEqual(
            result.stdout.splitlines(),
            ["crates/embed/src/moved.rs", "docs/moved.md"],
        )

    def test_unicode_path_remains_classifiable(self) -> None:
        base = self._commit("README.md", "base\n")
        head = self._commit("crates/inference/src/café.rs", "code\n")

        result = self._run("merge_group", base, head)

        self.assertEqual(
            result.stdout.splitlines(), ["crates/inference/src/café.rs"]
        )

    def test_control_character_path_fails_closed(self) -> None:
        base = self._commit("README.md", "base\n")
        head = self._commit("crates/inference/src/line\nbreak.rs", "code\n")

        result = self._run("merge_group", base, head, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires Git quoting", result.stderr)

    def test_first_push_reports_root_commit_files(self) -> None:
        head = self._commit("Cargo.toml", "[workspace]\n")

        result = self._run("push", _ZERO_SHA, head)

        self.assertEqual(result.stdout.splitlines(), ["Cargo.toml"])

    def test_first_push_with_history_reports_the_entire_tree(self) -> None:
        self._commit("crates/inference/src/change.rs", "code\n")
        head = self._commit("docs/followup.md", "docs\n")

        result = self._run("push", _ZERO_SHA, head)

        self.assertEqual(
            result.stdout.splitlines(),
            ["crates/inference/src/change.rs", "docs/followup.md"],
        )

    def test_checkout_head_mismatch_fails_closed(self) -> None:
        base = self._commit("README.md", "base\n")
        expected_head = self._commit("docs/change.md", "change\n")
        self._git("checkout", "-q", base)

        result = self._run("merge_group", base, expected_head, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match event head", result.stderr)

    def test_unavailable_base_is_a_named_range_resolution_failure(self) -> None:
        head = self._commit("README.md", "base\n")

        result = self._run("pull_request", _ZERO_SHA, head, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn(
            f"event base {_ZERO_SHA} is not available as a commit",
            result.stderr,
        )

    def test_degenerate_equal_commit_range_fails_with_named_error(self) -> None:
        head = self._commit("README.md", "base\n")

        for base in (head, head[:12], head.upper()):
            with self.subTest(base=base):
                result = self._run("pull_request", base, head, check=False)

                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertEqual(
                    result.stderr,
                    f"event base {base} and event head {head} resolve to the "
                    "same commit; refusing degenerate range\n",
                )

    def test_distinct_commit_empty_range_succeeds_without_changed_paths(
        self,
    ) -> None:
        base = self._commit("README.md", "base\n")
        self._git("commit", "--allow-empty", "-q", "-m", "empty change")
        head = self._git("rev-parse", "HEAD")

        result = self._run("pull_request", base, head, check=False)

        self.assertNotEqual(base, head)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "")
        self.assertEqual(result.stderr, "")

    def test_non_ancestor_base_fails_closed(self) -> None:
        base = self._commit("README.md", "base\n")
        self._git("checkout", "-q", "--orphan", "other")
        head = self._commit("other.txt", "unrelated\n")

        result = self._run("merge_group", base, head, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not an ancestor", result.stderr)

    def test_non_hexadecimal_revision_fails_before_git_parsing(self) -> None:
        head = self._commit("README.md", "base\n")

        result = self._run("merge_group", "HEAD^{tree}", head, check=False)

        self.assertEqual(result.returncode, 2)
        self.assertIn("must be hexadecimal commit IDs", result.stderr)

    def test_schedule_requires_explicit_workflow_policy(self) -> None:
        head = self._commit("README.md", "base\n")

        result = self._run("schedule", head, head, check=False)

        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported change-detection event", result.stderr)


class MergeQueueWorkflowTests(unittest.TestCase):
    def test_every_required_workflow_listens_for_merge_group_checks(self) -> None:
        trigger = (
            "  merge_group:\n"
            "    branches: [main]\n"
            "    types: [checks_requested]\n"
        )

        for relative_path in _REQUIRED_WORKFLOWS:
            with self.subTest(workflow=relative_path):
                contents = (_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn(trigger, contents)


class ChangeDetectorWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tempdir.name)
        self._git("init", "-q", "-b", "main")
        self._git("config", "user.name", "CI Test")
        self._git("config", "user.email", "ci-test@example.invalid")

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            [*_GIT, *args],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _commit(self, path: str, contents: str) -> str:
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")
        self._git("add", path)
        self._git("commit", "-q", "-m", f"write {path}")
        return self._git("rev-parse", "HEAD")

    def _run_filter(
        self,
        workflow: Path,
        base_sha: str,
        head_sha: str,
        *,
        event: str,
        environment: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        output = self.repo / "github-output"
        output.unlink(missing_ok=True)
        env = os.environ.copy()
        env.update(
            GITHUB_EVENT_NAME=event,
            GITHUB_OUTPUT=str(output),
            CI_BASE_SHA=base_sha,
            CI_HEAD_SHA=head_sha,
        )
        if environment is not None:
            env.update(environment)
        filter_script = self.repo / "filter-step.sh"
        filter_script.write_text(
            _workflow_run_script(workflow, "changes", "filter"),
            encoding="utf-8",
        )
        result = subprocess.run(
            [*_FILTER_SHELL_COMMAND, str(filter_script)],
            cwd=self.repo,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        contents = output.read_text(encoding="utf-8") if output.exists() else ""
        return result, contents

    def _assert_base_detector_wins(
        self,
        workflow: Path,
        expected_outputs: tuple[str, ...],
    ) -> None:
        base = self._commit(
            "scripts/ci-changed-files.sh",
            _SCRIPT.read_text(encoding="utf-8"),
        )
        head = self._commit(
            "scripts/ci-changed-files.sh",
            "#!/bin/sh\nexit 0\n",
        )

        for event in _DETECTOR_EVENTS:
            with self.subTest(workflow=workflow.name, event=event):
                result, output = self._run_filter(
                    workflow,
                    base,
                    head,
                    event=event,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                for expected_output in expected_outputs:
                    self.assertIn(expected_output, output)

    def _assert_detector_failure_is_observed(self, workflow: Path) -> None:
        base = self._commit(
            "scripts/ci-changed-files.sh",
            "#!/bin/sh\nexit 23\n",
        )
        head = self._commit("README.md", "change\n")

        for event in _DETECTOR_EVENTS:
            with self.subTest(workflow=workflow.name, event=event):
                result, output = self._run_filter(
                    workflow,
                    base,
                    head,
                    event=event,
                )

                self.assertEqual(result.returncode, 23)
                self.assertEqual(output, "")
                self.assertIn(
                    "change detector failed with status 23",
                    result.stderr,
                )

    def _assert_replacement_ref_is_ignored(
        self,
        workflow: Path,
        expected_outputs: tuple[str, ...],
    ) -> None:
        base = self._commit(
            "scripts/ci-changed-files.sh",
            "#!/bin/sh\necho scripts/ci-changed-files.sh\n",
        )
        self._commit("README.md", "head\n")
        replacement = self._commit(
            "scripts/ci-changed-files.sh",
            "#!/bin/sh\nexit 29\n",
        )
        self._git("replace", base, replacement)

        for event in _DETECTOR_EVENTS:
            with self.subTest(workflow=workflow.name, event=event):
                result, output = self._run_filter(
                    workflow,
                    base,
                    replacement,
                    event=event,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                for expected_output in expected_outputs:
                    self.assertIn(expected_output, output)

    def _ci_filter_with_added_prologue(self, statement: str) -> str:
        script = _workflow_run_script(_CI_WORKFLOW, "changes", "filter")
        marker = "TRUSTED_DETECTOR=$(mktemp)"
        self.assertIn(marker, script)
        return script.replace(marker, f"{statement}\n{marker}", 1)

    def _ci_workflow_with_filter_step(self, step: str) -> str:
        contents = _CI_WORKFLOW.read_text(encoding="utf-8")
        job = _workflow_job(contents, "changes")
        current = _workflow_step_by_id(job, "filter")
        mutated_job = job.replace(current, step, 1)
        return contents.replace(job, mutated_job, 1)

    def _ci_filter_step(self) -> str:
        contents = _CI_WORKFLOW.read_text(encoding="utf-8")
        return _workflow_step_by_id(_workflow_job(contents, "changes"), "filter")

    def test_ci_runs_the_base_detector_and_classifies_its_change(self) -> None:
        self._assert_base_detector_wins(_CI_WORKFLOW, ("code=true",))

    def test_ci_observes_detector_failure_status(self) -> None:
        self._assert_detector_failure_is_observed(_CI_WORKFLOW)

    def test_cargo_audit_runs_the_base_detector_and_classifies_its_change(
        self,
    ) -> None:
        self._assert_base_detector_wins(_CARGO_AUDIT_WORKFLOW, ("deps=true",))

    def test_cargo_audit_observes_detector_failure_status(self) -> None:
        self._assert_detector_failure_is_observed(_CARGO_AUDIT_WORKFLOW)

    def test_app_binaries_runs_the_base_detector_and_classifies_its_change(
        self,
    ) -> None:
        self._assert_base_detector_wins(
            _APP_BINARIES_WORKFLOW,
            ("bins=true", "swift=true"),
        )

    def test_app_binaries_observes_detector_failure_status(self) -> None:
        self._assert_detector_failure_is_observed(_APP_BINARIES_WORKFLOW)

    def test_e2e_parity_runs_the_base_detector_and_classifies_its_change(
        self,
    ) -> None:
        self._assert_base_detector_wins(
            _E2E_PARITY_WORKFLOW,
            ("engine=true", "tune=true"),
        )

    def test_e2e_parity_observes_detector_failure_status(self) -> None:
        self._assert_detector_failure_is_observed(_E2E_PARITY_WORKFLOW)

    def test_ci_ignores_replacement_ref_when_loading_detector(self) -> None:
        self._assert_replacement_ref_is_ignored(_CI_WORKFLOW, ("code=true",))

    def test_cargo_audit_ignores_replacement_ref_when_loading_detector(
        self,
    ) -> None:
        self._assert_replacement_ref_is_ignored(
            _CARGO_AUDIT_WORKFLOW,
            ("deps=true",),
        )

    def test_app_binaries_ignores_replacement_ref_when_loading_detector(
        self,
    ) -> None:
        self._assert_replacement_ref_is_ignored(
            _APP_BINARIES_WORKFLOW,
            ("bins=true", "swift=true"),
        )

    def test_e2e_parity_ignores_replacement_ref_when_loading_detector(
        self,
    ) -> None:
        self._assert_replacement_ref_is_ignored(
            _E2E_PARITY_WORKFLOW,
            ("engine=true", "tune=true"),
        )

    def test_filter_shell_removes_startup_and_path_overrides(self) -> None:
        base = self._commit(
            "scripts/ci-changed-files.sh",
            _SCRIPT.read_text(encoding="utf-8"),
        )
        head = self._commit(
            "scripts/ci-changed-files.sh",
            "#!/bin/sh\nexit 0\n",
        )
        startup = self.repo / "bash-env"
        startup.write_text(
            'echo "startup-file-ran=true" >> "$GITHUB_OUTPUT"\n',
            encoding="utf-8",
        )
        shadow_bin = self.repo / "shadow-bin"
        shadow_bin.mkdir()
        shadow_git = shadow_bin / "git"
        shadow_git.write_text(
            "#!/bin/sh\n"
            'echo "shadow-path-ran=true" >> "$GITHUB_OUTPUT"\n'
            'exec /usr/bin/git "$@"\n',
            encoding="utf-8",
        )
        shadow_git.chmod(0o755)
        environment = {
            "BASH_ENV": str(startup),
            "PATH": f"{shadow_bin}:/usr/bin:/bin",
        }
        cases = (
            (_CI_WORKFLOW, ("code=true",)),
            (_CARGO_AUDIT_WORKFLOW, ("deps=true",)),
            (_APP_BINARIES_WORKFLOW, ("bins=true", "swift=true")),
            (_E2E_PARITY_WORKFLOW, ("engine=true", "tune=true")),
        )

        for workflow, expected_outputs in cases:
            for event in _DETECTOR_EVENTS:
                with self.subTest(workflow=workflow.name, event=event):
                    result, output = self._run_filter(
                        workflow,
                        base,
                        head,
                        event=event,
                        environment=environment,
                    )

                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertNotIn("startup-file-ran=true", output)
                    self.assertNotIn("shadow-path-ran=true", output)
                    for expected_output in expected_outputs:
                        self.assertIn(expected_output, output)

    def test_change_jobs_start_filter_immediately_after_checkout(self) -> None:
        for relative_path in _REQUIRED_WORKFLOWS:
            with self.subTest(workflow=relative_path):
                contents = (_ROOT / relative_path).read_text(encoding="utf-8")
                document = _parse_workflow(contents, relative_path)
                job = _parsed_workflow_job(document, "changes", relative_path)
                steps = job.get("steps")
                self.assertIsInstance(steps, list)
                assert isinstance(steps, list)
                self.assertGreaterEqual(len(steps), 2)
                checkout = _require_mapping(
                    steps[0],
                    f"{relative_path} checkout step",
                )
                self.assertRegex(
                    str(checkout.get("uses")),
                    r"^actions/checkout@[0-9a-f]{40}$",
                )
                filter_step = _require_mapping(
                    steps[1],
                    f"{relative_path} filter step",
                )
                self.assertEqual(filter_step.get("id"), "filter")

    def test_change_filter_prologues_match_declared_ordered_grammar(
        self,
    ) -> None:
        self.assertEqual(
            set(_EXPECTED_FILTER_PROLOGUES),
            set(_REQUIRED_WORKFLOWS),
        )
        for relative_path in _REQUIRED_WORKFLOWS:
            with self.subTest(workflow=relative_path):
                contents = (_ROOT / relative_path).read_text(encoding="utf-8")
                _assert_safe_filter_workflow(
                    contents,
                    relative_path,
                )

    def test_filter_execution_rejects_selector_or_exit_before_trusted_execution(
        self,
    ) -> None:
        contents = _CI_WORKFLOW.read_text(encoding="utf-8")
        _assert_safe_filter_workflow(contents, _CI_WORKFLOW.name)
        trusted_execution = (
            f"          if {_TRUSTED_DETECTOR_EXECUTION} "
            '> "$CHANGED_FILE"; then\n'
        )
        self.assertIn(trusted_execution, contents)
        mutations = {
            "selector write": '          echo "code=false" >> "$GITHUB_OUTPUT"\n',
            "exit": "          exit 0\n",
        }

        for name, statement in mutations.items():
            workflow = contents.replace(
                trusted_execution,
                f"{statement}{trusted_execution}",
                1,
            )
            with self.subTest(mutation=name), self.assertRaisesRegex(
                AssertionError,
                "ordered detector execution",
            ):
                _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_filter_execution_rejects_preclassification_output_or_exit(
        self,
    ) -> None:
        contents = _CI_WORKFLOW.read_text(encoding="utf-8")
        _assert_safe_filter_workflow(contents, _CI_WORKFLOW.name)
        changed = '          CHANGED=$(<"$CHANGED_FILE")\n'
        self.assertIn(changed, contents)
        mutations = {
            "selector output": (
                '          printf \'code=false\\n\' >> "$GITHUB_OUTPUT"\n'
            ),
            "exit": "          exit 0\n",
        }

        for name, statement in mutations.items():
            workflow = contents.replace(
                changed,
                f"{changed}{statement}",
                1,
            )
            with self.subTest(mutation=name), self.assertRaisesRegex(
                AssertionError,
                "ordered selector classification",
            ):
                _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_changes_job_rejects_selector_output_literal(self) -> None:
        contents = _CI_WORKFLOW.read_text(encoding="utf-8")
        _assert_safe_filter_workflow(contents, _CI_WORKFLOW.name)
        expected = "      code: ${{ steps.filter.outputs.code }}\n"
        self.assertIn(expected, contents)
        workflow = contents.replace(expected, '      code: "false"\n', 1)

        with self.assertRaisesRegex(AssertionError, "output"):
            _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_changes_job_rejects_container(self) -> None:
        contents = _CI_WORKFLOW.read_text(encoding="utf-8")
        _assert_safe_filter_workflow(contents, _CI_WORKFLOW.name)
        expected = "    runs-on: ubuntu-latest\n"
        self.assertIn(expected, contents)
        workflow = contents.replace(
            expected,
            f"{expected}    container: ubuntu:24.04\n",
            1,
        )

        with self.assertRaisesRegex(AssertionError, "changes job schema"):
            _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_changes_job_rejects_unapproved_execution_schema(self) -> None:
        contents = _CI_WORKFLOW.read_text(encoding="utf-8")
        _assert_safe_filter_workflow(contents, _CI_WORKFLOW.name)
        runner = "    runs-on: ubuntu-latest\n"
        self.assertIn(runner, contents)
        mutations = {
            "runner": contents.replace(
                runner,
                "    runs-on: macos-latest\n",
                1,
            ),
            "services": contents.replace(
                runner,
                f"{runner}    services: {{}}\n",
                1,
            ),
            "defaults": contents.replace(
                runner,
                f"{runner}    defaults: {{run: {{shell: bash}}}}\n",
                1,
            ),
            "condition": contents.replace(
                runner,
                f"{runner}    if: github.event_name == 'pull_request'\n",
                1,
            ),
            "continue on error": contents.replace(
                runner,
                f"{runner}    continue-on-error: true\n",
                1,
            ),
        }

        for name, workflow in mutations.items():
            expected_error = "runs-on" if name == "runner" else "changes job schema"
            with self.subTest(field=name), self.assertRaisesRegex(
                AssertionError,
                expected_error,
            ):
                _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_changes_job_rejects_job_environment_in_block_and_flow_styles(
        self,
    ) -> None:
        contents = _CI_WORKFLOW.read_text(encoding="utf-8")
        _assert_safe_filter_workflow(contents, _CI_WORKFLOW.name)
        expected = "    runs-on: ubuntu-latest\n"
        self.assertIn(expected, contents)
        environments = {
            "block": "    env:\n      PATH: /tmp/filter-bin\n",
            "flow": "    env: {PATH: /tmp/filter-bin}\n",
        }

        for style, environment in environments.items():
            workflow = contents.replace(
                expected,
                f"{expected}{environment}",
                1,
            )
            with self.subTest(style=style), self.assertRaisesRegex(
                AssertionError,
                "changes job schema|job environment",
            ):
                _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_filter_step_accepts_equivalent_flow_style_environment(self) -> None:
        contents = _CI_WORKFLOW.read_text(encoding="utf-8")
        _assert_safe_filter_workflow(contents, _CI_WORKFLOW.name)
        block_environment = (
            "        env:\n"
            "          CI_BASE_SHA: ${{ github.event.pull_request.base.sha || "
            "github.event.merge_group.base_sha || github.event.before }}\n"
            "          CI_HEAD_SHA: ${{ github.event.merge_group.head_sha || "
            "github.sha }}\n"
        )
        flow_environment = (
            '        env: {CI_BASE_SHA: "${{ github.event.pull_request.base.sha || '
            'github.event.merge_group.base_sha || github.event.before }}", '
            'CI_HEAD_SHA: "${{ github.event.merge_group.head_sha || github.sha }}"}\n'
        )
        self.assertIn(block_environment, contents)
        workflow = contents.replace(block_environment, flow_environment, 1)

        _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_detector_reader_event_matrix_is_complete(self) -> None:
        self.assertEqual(_DETECTOR_EVENTS, ("pull_request", "merge_group"))

    def test_filter_prologue_rejects_balanced_event_condition_around_detector(
        self,
    ) -> None:
        step = self._ci_filter_step()
        marker = "          TRUSTED_DETECTOR=$(mktemp)\n"
        self.assertIn(marker, step)
        step = step.replace(
            marker,
            "          if [ \"$GITHUB_EVENT_NAME\" = \"pull_request\" ]; then\n"
            f"{marker}",
            1,
        )
        changed = '          CHANGED=$(<"$CHANGED_FILE")\n'
        self.assertIn(changed, step)
        step = step.replace(changed, f"{changed}          fi\n", 1)
        workflow = self._ci_workflow_with_filter_step(step)

        with self.assertRaisesRegex(AssertionError, "top level"):
            _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_filter_step_rejects_event_dependent_condition(self) -> None:
        step = self._ci_filter_step().replace(
            "      - id: filter\n",
            "      - id: filter\n"
            "        if: github.event_name == 'pull_request'\n",
            1,
        )
        workflow = self._ci_workflow_with_filter_step(step)

        with self.assertRaisesRegex(AssertionError, "step metadata"):
            _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_filter_step_rejects_bash_env_override(self) -> None:
        step = self._ci_filter_step().replace(
            "        env:\n",
            "        env:\n          BASH_ENV: /tmp/filter-startup\n",
            1,
        )
        workflow = self._ci_workflow_with_filter_step(step)

        with self.assertRaisesRegex(AssertionError, "environment"):
            _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_filter_step_rejects_path_override(self) -> None:
        step = self._ci_filter_step().replace(
            "        env:\n",
            "        env:\n          PATH: /tmp/filter-bin\n",
            1,
        )
        workflow = self._ci_workflow_with_filter_step(step)

        with self.assertRaisesRegex(AssertionError, "environment"):
            _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_filter_step_rejects_unapproved_execution_metadata(self) -> None:
        step = self._ci_filter_step()
        mutations = {
            "continue on error": step.replace(
                "      - id: filter\n",
                "      - id: filter\n        continue-on-error: true\n",
                1,
            ),
            "working directory": step.replace(
                "      - id: filter\n",
                "      - id: filter\n        working-directory: /tmp\n",
                1,
            ),
            "alternate shell": step.replace(
                f"        shell: {_TRUSTED_FILTER_SHELL}\n",
                "        shell: bash\n",
                1,
            ),
            "mapping merge": step.replace(
                "      - id: filter\n",
                "      - id: filter\n        <<: *filter-options\n",
                1,
            ),
        }

        for name, mutated_step in mutations.items():
            with self.subTest(metadata=name):
                workflow = self._ci_workflow_with_filter_step(mutated_step)
                with self.assertRaisesRegex(
                    AssertionError,
                    "step metadata|shell command|valid workflow YAML",
                ):
                    _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_filter_step_rejects_inherited_execution_overrides(self) -> None:
        contents = _CI_WORKFLOW.read_text(encoding="utf-8")
        job = _workflow_job(contents, "changes")
        mutations = {
            "workflow defaults": contents.replace(
                "name: CI\n",
                "name: CI\n\ndefaults:\n  run:\n    shell: bash\n",
                1,
            ),
            "job defaults": contents.replace(
                job,
                job.replace(
                    "  changes:\n",
                    "  changes:\n    defaults:\n      run:\n        shell: bash\n",
                    1,
                ),
                1,
            ),
            "job environment": contents.replace(
                job,
                job.replace(
                    "    runs-on: ubuntu-latest\n",
                    "    runs-on: ubuntu-latest\n"
                    "    env:\n"
                    "      PATH: /tmp/filter-bin\n",
                    1,
                ),
                1,
            ),
            "workflow environment": contents.replace(
                "env:\n",
                "env:\n  BASH_ENV: /tmp/filter-startup\n",
                1,
            ),
            "workflow merge": contents.replace(
                "name: CI\n",
                "name: CI\n<<: *workflow-options\n",
                1,
            ),
            "job merge": contents.replace(
                job,
                job.replace(
                    "  changes:\n",
                    "  changes:\n    <<: *job-options\n",
                    1,
                ),
                1,
            ),
        }

        for name, workflow in mutations.items():
            with self.subTest(scope=name), self.assertRaisesRegex(
                AssertionError,
                "defaults|environment|schema|valid workflow YAML",
            ):
                _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_filter_steps_declare_the_sanitized_absolute_shell(self) -> None:
        for relative_path in _REQUIRED_WORKFLOWS:
            with self.subTest(workflow=relative_path):
                contents = (_ROOT / relative_path).read_text(encoding="utf-8")
                document = _parse_workflow(contents, relative_path)
                job = _parsed_workflow_job(document, "changes", relative_path)
                _, step = _parsed_workflow_step_by_id(
                    job,
                    "filter",
                    relative_path,
                )
                self.assertEqual(step.get("shell"), _TRUSTED_FILTER_SHELL)

    def test_filter_workflow_rejects_unsanitized_detector_commands(self) -> None:
        contents = _CI_WORKFLOW.read_text(encoding="utf-8")
        trusted_load = (
            "/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/git "
            "--no-replace-objects show"
        )
        trusted_execution = _TRUSTED_DETECTOR_EXECUTION
        mutations = {
            "relative load command": (
                contents.replace(
                    trusted_load,
                    "git --no-replace-objects show",
                    1,
                ),
                "trusted /usr/bin/env and /usr/bin/git form",
            ),
            "inherited execution environment": (
                contents.replace(
                    trusted_execution,
                    'sh "$TRUSTED_DETECTOR"',
                    1,
                ),
                "clean environment and absolute shell",
            ),
        }

        for name, (workflow, expected_error) in mutations.items():
            with self.subTest(command=name), self.assertRaisesRegex(
                AssertionError,
                expected_error,
            ):
                _assert_safe_filter_workflow(workflow, _CI_WORKFLOW.name)

    def test_filter_prologue_rejects_command_substitution(self) -> None:
        script = self._ci_filter_with_added_prologue(
            "probe=$(git rev-parse --show-toplevel)"
        )

        with self.assertRaisesRegex(AssertionError, "before detector loading"):
            _assert_safe_filter_prologue(script, _CI_WORKFLOW.name)

    def test_filter_prologue_rejects_eval_indirection(self) -> None:
        script = self._ci_filter_with_added_prologue(
            "eval 'git rev-parse --show-toplevel'"
        )

        with self.assertRaisesRegex(AssertionError, "before detector loading"):
            _assert_safe_filter_prologue(script, _CI_WORKFLOW.name)

    def test_filter_prologue_rejects_indirection_in_detector_load(self) -> None:
        script = _workflow_run_script(_CI_WORKFLOW, "changes", "filter")
        script, replacements = re.subn(
            r"/usr/bin/git --no-replace-objects show ",
            '/usr/bin/git "$(scripts/preload-detector)" show ',
            script,
            count=1,
        )
        self.assertEqual(replacements, 1)

        with self.assertRaisesRegex(AssertionError, "base detector"):
            _assert_safe_filter_prologue(script, _CI_WORKFLOW.name)

    def test_filter_prologue_accepts_core_quote_path_git_option(self) -> None:
        script = _workflow_run_script(_CI_WORKFLOW, "changes", "filter")
        script, replacements = re.subn(
            r"/usr/bin/git --no-replace-objects show ",
            "/usr/bin/git --no-replace-objects "
            "-c core.quotePath=false show ",
            script,
            count=1,
        )
        self.assertEqual(replacements, 1)

        _assert_safe_filter_prologue(script, _CI_WORKFLOW.name)

    def test_filter_prologue_rejects_unapproved_git_option_with_guidance(
        self,
    ) -> None:
        script = _workflow_run_script(_CI_WORKFLOW, "changes", "filter")
        script, replacements = re.subn(
            r"/usr/bin/git --no-replace-objects show ",
            "/usr/bin/git --no-replace-objects -C . show ",
            script,
            count=1,
        )
        self.assertEqual(replacements, 1)

        with self.assertRaisesRegex(AssertionError, "use the declared load shape"):
            _assert_safe_filter_prologue(script, _CI_WORKFLOW.name)

    def test_filter_prologue_rejects_unbalanced_conditionals(self) -> None:
        mutations = {
            "unexpected terminator": self._ci_filter_with_added_prologue("fi"),
            "unterminated condition": self._ci_filter_with_added_prologue(
                "if true; then"
            ),
        }

        for name, script in mutations.items():
            expected = "unbalanced" if name == "unexpected terminator" else "top level"
            with self.subTest(structure=name), self.assertRaisesRegex(
                AssertionError,
                expected,
            ):
                _assert_safe_filter_prologue(script, _CI_WORKFLOW.name)

    def test_filter_prologue_rejects_shell_boundary_forms(self) -> None:
        mutations = {
            "semicolon list": "set -euo pipefail; printf unexpected",
            "and list": "set -euo pipefail && printf unexpected",
            "or list": "set -euo pipefail || printf unexpected",
            "pipeline": "set -euo pipefail | printf unexpected",
            "line continuation": "set -euo pipefail \\\nprintf unexpected",
            "trailing comment": "set -euo pipefail # comment",
            "carriage return": "set -euo pipefail\rprintf unexpected",
            "vertical tab": "set -euo pipefail\vprintf unexpected",
            "environment assignment": "MODE=changed printf unexpected",
            "command builtin": "command printf unexpected",
            "brace group": "{ printf unexpected; }",
            "subshell": "(printf unexpected)",
            "coprocess": "coproc printf unexpected",
            "redirection only": '> "$GITHUB_OUTPUT"',
            "for compound": "for x in one; do printf unexpected; done",
            "comment continuation": "# comment \\\nprintf unexpected",
            "unicode lookalike": "ｓｅｔ -euo pipefail",
        }

        for name, statement in mutations.items():
            with self.subTest(form=name):
                script = self._ci_filter_with_added_prologue(statement)
                with self.assertRaisesRegex(
                    AssertionError,
                    "before detector loading",
                ):
                    _assert_safe_filter_prologue(script, _CI_WORKFLOW.name)

    def test_app_binaries_full_selection_exceptions_are_exactly_enumerated(
        self,
    ) -> None:
        script = _workflow_run_script(
            _APP_BINARIES_WORKFLOW,
            "changes",
            "filter",
        )
        prefix, separator, _ = script.partition("TRUSTED_DETECTOR=$(mktemp)")
        self.assertEqual(separator, "TRUSTED_DETECTOR=$(mktemp)")

        exceptions = tuple(
            (
                condition,
                tuple(
                    re.findall(
                        r'^  echo "((?:bins|swift)=true)" '
                        r'>> "\$GITHUB_OUTPUT"$',
                        body,
                        re.MULTILINE,
                    )
                ),
            )
            for condition, body in re.findall(
                r"^if ([^\n]+); then\n(.*?)^fi$",
                prefix,
                re.MULTILINE | re.DOTALL,
            )
            if re.search(r"^  exit 0$", body, re.MULTILINE) is not None
        )

        self.assertEqual(
            exceptions,
            (
                (
                    '[ "$GITHUB_EVENT_NAME" = "workflow_dispatch" ]',
                    ("bins=true", "swift=true"),
                ),
                (
                    f'[ "$CI_BASE_SHA" = "{_ZERO_SHA}" ]',
                    ("bins=true", "swift=true"),
                ),
            ),
        )
        self.assertEqual(prefix.count("  exit 0\n"), 2)

    def test_unavailable_base_fails_without_selector_outputs(self) -> None:
        head = self._commit("README.md", "head\n")

        for workflow in (
            _CI_WORKFLOW,
            _CARGO_AUDIT_WORKFLOW,
            _APP_BINARIES_WORKFLOW,
            _E2E_PARITY_WORKFLOW,
        ):
            for event in _DETECTOR_EVENTS:
                with self.subTest(workflow=workflow.name, event=event):
                    result, output = self._run_filter(
                        workflow,
                        "f" * 40,
                        head,
                        event=event,
                    )

                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(output, "")
                    self.assertIn(
                        "unable to load change detector from base revision",
                        result.stderr,
                    )

    def test_invalid_base_never_executes_checkout_detector(self) -> None:
        head = self._commit(
            "scripts/ci-changed-files.sh",
            "#!/bin/sh\n"
            "echo 'attacker-selected=true' >> \"$GITHUB_OUTPUT\"\n"
            "echo README.md\n",
        )

        for workflow in (
            _CI_WORKFLOW,
            _CARGO_AUDIT_WORKFLOW,
            _APP_BINARIES_WORKFLOW,
            _E2E_PARITY_WORKFLOW,
        ):
            for event in _DETECTOR_EVENTS:
                for base_sha in ("", "a" * 39, "g" * 40):
                    with self.subTest(
                        workflow=workflow.name,
                        event=event,
                        base_sha=base_sha,
                    ):
                        result, output = self._run_filter(
                            workflow,
                            base_sha,
                            head,
                            event=event,
                        )

                        self.assertEqual(result.returncode, 2)
                        self.assertEqual(output, "")
                        self.assertNotIn("attacker-selected=true", output)
                        self.assertIn(
                            "base revision must be a nonempty 40-character "
                            "hexadecimal commit ID",
                            result.stderr,
                        )

    def test_all_zero_base_selects_every_downstream_job(self) -> None:
        head = self._commit(
            "scripts/ci-changed-files.sh",
            "#!/bin/sh\n"
            "echo 'attacker-selected=true' >> \"$GITHUB_OUTPUT\"\n"
            "exit 29\n",
        )
        cases = (
            (_CI_WORKFLOW, "push", ("code=true",)),
            (_CARGO_AUDIT_WORKFLOW, "pull_request", ("deps=true",)),
            (
                _APP_BINARIES_WORKFLOW,
                "push",
                ("bins=true", "swift=true"),
            ),
            (
                _E2E_PARITY_WORKFLOW,
                "push",
                ("engine=true", "tune=true"),
            ),
        )

        for workflow, event, expected_outputs in cases:
            with self.subTest(workflow=workflow.name, event=event):
                result, output = self._run_filter(
                    workflow,
                    _ZERO_SHA,
                    head,
                    event=event,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("attacker-selected=true", output)
                for expected_output in expected_outputs:
                    self.assertIn(expected_output, output)


class RequiredGateSelectorTests(unittest.TestCase):
    _CASES = (
        (_CI_WORKFLOW, "ci-gate", ("code",)),
        (_CARGO_AUDIT_WORKFLOW, "cargo-audit-gate", ("deps",)),
        (
            _APP_BINARIES_WORKFLOW,
            "app-binaries-gate",
            ("bins", "swift"),
        ),
        (_E2E_PARITY_WORKFLOW, "parity-gate", ("engine", "tune")),
    )

    def test_required_gates_reject_missing_selector_outputs(self) -> None:
        for workflow, gate_id, selectors in self._CASES:
            for missing in selectors:
                values = {selector: "false" for selector in selectors}
                values[missing] = ""
                with self.subTest(
                    workflow=workflow.name,
                    selector=missing,
                ):
                    result = _run_required_gate(workflow, gate_id, values)

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        "selector output",
                        f"{result.stdout}\n{result.stderr}",
                    )

    def test_required_gates_reject_non_boolean_selector_outputs(self) -> None:
        for workflow, gate_id, selectors in self._CASES:
            for invalid in selectors:
                values = {selector: "false" for selector in selectors}
                values[invalid] = "unknown"
                with self.subTest(
                    workflow=workflow.name,
                    selector=invalid,
                ):
                    result = _run_required_gate(workflow, gate_id, values)

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        "selector output",
                        f"{result.stdout}\n{result.stderr}",
                    )

    def test_required_gates_accept_explicit_negative_selector_outputs(
        self,
    ) -> None:
        for workflow, gate_id, selectors in self._CASES:
            with self.subTest(workflow=workflow.name):
                result = _run_required_gate(
                    workflow,
                    gate_id,
                    {selector: "false" for selector in selectors},
                )

                self.assertEqual(result.returncode, 0, result.stderr)

    def test_required_gates_accept_explicit_positive_selector_outputs(
        self,
    ) -> None:
        for workflow, gate_id, selectors in self._CASES:
            with self.subTest(workflow=workflow.name):
                result = _run_required_gate(
                    workflow,
                    gate_id,
                    {selector: "true" for selector in selectors},
                )

                self.assertEqual(result.returncode, 0, result.stderr)


class TestRunnerContractTests(unittest.TestCase):
    def test_empty_collection_fails_closed(self) -> None:
        with self.assertRaisesRegex(SystemExit, "no tests collected"):
            _require_tests_collected(unittest.TestSuite())


class E2eParityChangeFilterTests(unittest.TestCase):
    def test_future_rust_setup_action_changes_trigger_engine_parity(self) -> None:
        workflow = (
            _ROOT / ".github/workflows/e2e-parity.yml"
        ).read_text(encoding="utf-8")
        filters = [
            line.strip()
            for line in workflow.splitlines()
            if line.strip().startswith("if grep -E ")
        ]
        self.assertGreaterEqual(len(filters), 1)
        engine_filter = filters[0]

        self.assertIn(
            r"^\.github/actions/rust-setup/",
            engine_filter,
        )
        pattern_match = re.search(r"grep -E '([^']+)'", engine_filter)
        self.assertIsNotNone(pattern_match)
        pattern = pattern_match.group(1)

        for changed_path in (
            ".github/actions/rust-setup/action.yml",
            ".github/actions/rust-setup/scripts/verify.sh",
        ):
            with self.subTest(changed_path=changed_path):
                result = subprocess.run(
                    ["grep", "-E", pattern],
                    input=f"{changed_path}\n",
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0)


class E2eReporterWorkflowTests(unittest.TestCase):
    _REPORTING_CONTRACTS = (
        (
            "parity",
            "Post parity report",
            "e2e-parity",
            "Run e2e parity check",
            "python3 scripts/e2e_parity_check.py",
        ),
        (
            "metal-parity",
            "Post Metal parity report",
            "e2e-metal-parity",
            "Run Metal attention/KV parity check",
            "python3 scripts/e2e_parity_check.py --backend metal",
        ),
        (
            "q4-vision-gates",
            "Post PPL gate report",
            "q4-ppl-gate",
            "Run Q4 PPL regression gate",
            "python3 scripts/ppl_gate_check.py",
        ),
    )

    def setUp(self) -> None:
        self.workflow = (
            _ROOT / ".github/workflows/e2e-parity.yml"
        ).read_text(encoding="utf-8")

    def _assert_reporting_contract(self, workflow: str) -> None:
        for job_id, reporter_name, header, gate_name, command in (
            self._REPORTING_CONTRACTS
        ):
            job = _workflow_job(workflow, job_id)
            reporter = _workflow_step(job, reporter_name)
            gate = _workflow_step(job, gate_name)

            self.assertIn(
                "if: github.event_name == 'pull_request' && always()",
                reporter,
            )
            self.assertIn("continue-on-error: true", reporter)
            self.assertIn("marocchino/sticky-pull-request-comment@", reporter)
            self.assertIn(f"header: {header}", reporter)

            self.assertNotIn("continue-on-error:", gate)
            self.assertNotIn("|| true", gate)
            self.assertRegex(
                gate,
                rf"(?m)^        run: {re.escape(command)}$",
            )

    def _mutate_step(
        self,
        workflow: str,
        job_id: str,
        step_name: str,
        old: str,
        new: str,
    ) -> str:
        job = _workflow_job(workflow, job_id)
        step = _workflow_step(job, step_name)
        self.assertIn(old, step)
        mutated_step = step.replace(old, new, 1)
        mutated_job = job.replace(step, mutated_step, 1)
        return workflow.replace(job, mutated_job, 1)

    def test_reporters_are_informational_and_verification_remains_gating(
        self,
    ) -> None:
        self._assert_reporting_contract(self.workflow)

    def test_contract_rejects_reporter_and_gate_mutations(self) -> None:
        mutations = {}
        for job_id, reporter_name, _, gate_name, command in (
            self._REPORTING_CONTRACTS
        ):
            mutations[f"{reporter_name} made gating"] = self._mutate_step(
                self.workflow,
                job_id,
                reporter_name,
                "continue-on-error: true",
                "continue-on-error: false",
            )
            mutations[f"{gate_name} made non-gating"] = self._mutate_step(
                self.workflow,
                job_id,
                gate_name,
                f"run: {command}",
                f"continue-on-error: true\n        run: {command}",
            )

        for name, mutated in mutations.items():
            with self.subTest(mutation=name), self.assertRaises(AssertionError):
                self._assert_reporting_contract(mutated)


class E2eParityChangeClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        contents = _E2E_PARITY_WORKFLOW.read_text(encoding="utf-8")
        cls.engine_pattern = _engine_change_pattern(contents)

    def test_frozen_reference_fixture_change_is_an_engine_change(self) -> None:
        path = (
            "crates/inference/tests/fixtures/"
            "e2e_parity_reference_v1/reference.json"
        )

        self.assertIsNotNone(self.engine_pattern.search(path))

    def test_frozen_reference_contract_test_change_is_an_engine_change(self) -> None:
        self.assertIsNotNone(
            self.engine_pattern.search("tests/test_e2e_parity_reference.py")
        )


class X86EmbedDriftObservationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contents = _E2E_PARITY_WORKFLOW.read_text(encoding="utf-8")
        cls.job = _workflow_job(cls.contents, "embed-drift-x86")

    def test_job_selects_x86_64_for_engine_changes(self) -> None:
        self.assertIn("name: embed drift gate (x86_64 observation)", self.job)
        self.assertIn("needs: changes", self.job)
        self.assertIn("if: needs.changes.outputs.engine == 'true'", self.job)
        self.assertIn("runs-on: ubuntu-latest", self.job)
        self.assertIn("""run: test "$(uname -m)" = 'x86_64'""", self.job)

    def test_job_reports_failures_without_gating_the_workflow(self) -> None:
        self.assertIn("continue-on-error: true", self.job)
        self._assert_provisions_all_four_models(self.job)
        self.assertIn(
            "cargo run --release -p lattice-embed --bin embed-drift "
            "-- --enforce --json",
            self.job,
        )

    def _assert_provisions_all_four_models(self, job: str) -> None:
        for model in (
            "bge-small-en-v1.5",
            "multilingual-e5-small",
            "all-minilm-l6-v2",
            "paraphrase-multilingual-minilm-l12-v2",
        ):
            self.assertIn(model, job)
        self.assertIn(
            "cargo run --release -p lattice-embed --bin embed \\\n"
            '              -- --model "$model" --download-only',
            job,
        )

    def test_observation_does_not_replace_the_required_arm_gate(self) -> None:
        arm_job = _workflow_job(self.contents, "embed-drift")
        self.assertIn("runs-on: ubuntu-24.04-arm", arm_job)
        self._assert_provisions_all_four_models(arm_job)
        self.assertIn(
            "cargo run --release -p lattice-embed --bin embed-drift "
            "-- --enforce --json",
            arm_job,
        )

        parity_gate = _workflow_job(self.contents, "parity-gate")
        self.assertIn("embed-drift,", parity_gate)
        self.assertNotIn("embed-drift-x86", parity_gate)


class E2eRunnerSpecializationWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = (
            _ROOT / ".github/workflows/e2e-parity.yml"
        ).read_text(encoding="utf-8")

    def _assert_split_is_fail_closed(self, workflow: str) -> None:
        cpu = _workflow_job(workflow, "vision-cpu-gates")
        artifact = _workflow_job(workflow, "quarot-artifact")
        q4_metal = _workflow_job(workflow, "q4-vision-gates")
        gate = _workflow_job(workflow, "parity-gate")
        s5b_step = _workflow_step(
            q4_metal, "Run S5b macOS f16 gate (fail-closed)"
        )
        cpu_code = "\n".join(
            line for line in cpu.splitlines() if not line.lstrip().startswith("#")
        )
        artifact_code = "\n".join(
            line
            for line in artifact.splitlines()
            if not line.lstrip().startswith("#")
        )
        q4_metal_code = "\n".join(
            line
            for line in q4_metal.splitlines()
            if not line.lstrip().startswith("#")
        )

        condition = (
            "if: needs.changes.outputs.engine == 'true' && "
            "github.event.pull_request.draft != true"
        )
        self.assertIn(condition, cpu_code)
        self.assertIn(condition, artifact_code)
        self.assertIn(condition, q4_metal_code)
        self.assertIn("runs-on: ubuntu-24.04-arm", cpu_code)
        self.assertIn('test "$ARCH" = "aarch64"', cpu_code)
        self.assertIn("LATTICE_VISION_CPU_RUNNER", cpu_code)
        self.assertIn("LATTICE_VISION_S3_GATE_ENFORCE: '1'", cpu_code)
        self.assertIn("--features f16 \\", cpu_code)
        self.assertIn("status=$?", cpu_code)
        self.assertIn('exit "$status"', cpu_code)
        self.assertNotIn("metal-gpu", cpu_code)
        self.assertNotIn("LATTICE_METAL_TEST_ENFORCE", cpu_code)
        self.assertNotIn("quantize_", cpu_code)

        test_targets = (
            "vision_s3_vit_forward_test",
            "vision_s4_merger_test",
        )
        proof_markers = (
            "LATTICE_VISION_S3_COSINE",
            "LATTICE_VISION_S3_MUTATION_COSINE",
            "LATTICE_VISION_S4_COSINE",
            "LATTICE_VISION_S4_MUTATION_COSINE",
            "LATTICE_VISION_S4_E2E_COSINE",
            "LATTICE_VISION_S4_E2E_MUTATION_COSINE",
        )
        for target in test_targets:
            self.assertEqual(cpu_code.count(f"--test {target}"), 1)
            self.assertNotIn(target, q4_metal_code)
        for marker in proof_markers:
            self.assertEqual(cpu_code.count(f"grep -q '{marker}'"), 1)
        self.assertNotIn("vision_s5b_e2e_gate_test", cpu_code)
        self.assertNotIn("LATTICE_VISION_S5B_GREEDY_TOKENS", cpu_code)

        self.assertIn("runs-on: ubuntu-24.04-arm", artifact_code)
        self.assertIn('test "$ARCH" = "aarch64"', artifact_code)
        self.assertIn("LATTICE_QUAROT_ARTIFACT_RUNNER", artifact_code)
        self.assertIn(
            "cargo build --release -p lattice-inference "
            "--bin quantize_quarot --features f16",
            artifact_code,
        )
        self.assertIn("target/release/quantize_quarot", artifact_code)
        self.assertIn("--seed 0xCAFE_BABE_DEAD_BEEF", artifact_code)
        self.assertIn("--num-probe-tokens 4", artifact_code)
        self.assertIn("status=$?", artifact_code)
        self.assertIn('exit "$status"', artifact_code)
        self.assertIn("grep -q '^Forward-equiv:'", artifact_code)
        self.assertIn("quantize_index.json", artifact_code)
        self.assertIn("config.json", artifact_code)
        self.assertIn("LATTICE_QUAROT_ARTIFACT_READY", artifact_code)
        self.assertIn('-cf "$RUNNER_TEMP/quarot-q4.tar" .', artifact_code)
        self.assertIn("shasum -a 256 quarot-q4.tar", artifact_code)
        self.assertIn(
            "actions/upload-artifact@"
            "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
            artifact_code,
        )
        self.assertIn(
            "name: quarot-q4-${{ github.run_id }}\n",
            artifact_code,
        )
        self.assertNotIn("github.run_attempt", artifact_code)
        self.assertIn("if-no-files-found: error", artifact_code)
        self.assertIn("retention-days: 1", artifact_code)
        self.assertIn("compression-level: 0", artifact_code)
        self.assertIn("overwrite: true", artifact_code)
        self.assertNotIn("metal-gpu", artifact_code)

        self.assertIn("runs-on: macos-latest", q4_metal_code)
        self.assertIn("needs: [changes, quarot-artifact]", q4_metal_code)
        self.assertIn(
            "actions/download-artifact@"
            "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
            q4_metal_code,
        )
        self.assertIn(
            "name: quarot-q4-${{ github.run_id }}\n",
            q4_metal_code,
        )
        self.assertNotIn("github.run_attempt", q4_metal_code)
        self.assertIn("shasum -a 256 -c quarot-q4.tar.sha256", q4_metal_code)
        self.assertIn("-xf quarot-q4.tar", q4_metal_code)
        self.assertIn("LATTICE_QUAROT_ARTIFACT_CONSUMED", q4_metal_code)
        self.assertNotIn("--bin quantize_quarot", q4_metal_code)
        for metal_gate_anchor in (
            "--test quarot_q4_composed_golden",
            '--features "f16,metal-gpu"',
            "LATTICE_METAL_TEST_ENFORCE: '1'",
            "--lib inject -- --nocapture --test-threads=1",
            "--test vision_s5b_e2e_gate_test",
            "LATTICE_VISION_S5B_GREEDY_TOKENS",
            "--bin eval_perplexity --bin quantize_q4",
            "LATTICE_PPL_GATE_REQUIRE_ARMED: \"1\"",
        ):
            self.assertIn(metal_gate_anchor, q4_metal_code)
        self.assertIn("status=$?", s5b_step)
        self.assertIn('exit "$status"', s5b_step)

        needs = gate.split("    needs:\n", 1)[1].split("    if:", 1)[0]
        self.assertIn("        vision-cpu-gates,\n", needs)
        self.assertIn("        quarot-artifact,\n", needs)
        self.assertIn(
            'VISION_CPU_RESULT="${{ needs.vision-cpu-gates.result }}"',
            gate,
        )
        self.assertIn(
            'if [ "$VISION_CPU_RESULT" != "success" ]; then',
            gate,
        )
        self.assertIn(
            'QUAROT_ARTIFACT_RESULT="${{ needs.quarot-artifact.result }}"',
            gate,
        )
        self.assertIn(
            'if [ "$QUAROT_ARTIFACT_RESULT" != "success" ]; then',
            gate,
        )

    def test_portable_gates_are_arm_only_and_required(self) -> None:
        self._assert_split_is_fail_closed(self.workflow)

    def test_contract_rejects_runner_coverage_and_gate_mutations(self) -> None:
        def mutate_job(job_id: str, old: str, new: str) -> str:
            job = _workflow_job(self.workflow, job_id)
            self.assertIn(old, job)
            return self.workflow.replace(job, job.replace(old, new, 1), 1)

        def mutate_step(
            job_id: str,
            step_name: str,
            old: str,
            new: str,
        ) -> str:
            job = _workflow_job(self.workflow, job_id)
            step = _workflow_step(job, step_name)
            self.assertIn(old, step)
            mutated_step = step.replace(old, new, 1)
            mutated_job = job.replace(step, mutated_step, 1)
            return self.workflow.replace(job, mutated_job, 1)

        mutations = {
            "runner moved back to macOS": mutate_job(
                "vision-cpu-gates",
                "runs-on: ubuntu-24.04-arm",
                "runs-on: macos-latest",
            ),
            "real-path marker removed": mutate_job(
                "vision-cpu-gates",
                "grep -q 'LATTICE_VISION_S4_E2E_MUTATION_COSINE'",
                "true # marker removed",
            ),
            "Metal feature added to CPU job": mutate_job(
                "vision-cpu-gates",
                "--features f16 \\",
                "--features f16,metal-gpu \\",
            ),
            "cargo failure propagation removed": mutate_job(
                "vision-cpu-gates",
                'exit "$status"',
                "true # cargo status discarded",
            ),
            "S5b cargo failure propagation removed": mutate_step(
                "q4-vision-gates",
                "Run S5b macOS f16 gate (fail-closed)",
                'exit "$status"',
                "true # cargo status discarded",
            ),
            "artifact runner moved to macOS": mutate_job(
                "quarot-artifact",
                "runs-on: ubuntu-24.04-arm",
                "runs-on: macos-latest",
            ),
            "artifact converter failure propagation removed": mutate_step(
                "quarot-artifact",
                "Generate and verify QuaRot Q4 artifact",
                'exit "$status"',
                "true # converter status discarded",
            ),
            "artifact equivalence proof removed": mutate_job(
                "quarot-artifact",
                "grep -q '^Forward-equiv:'",
                "true # equivalence proof removed",
            ),
            "artifact ready marker removed": mutate_job(
                "quarot-artifact",
                'echo "LATTICE_QUAROT_ARTIFACT_READY"',
                "true # artifact marker removed",
            ),
            "artifact producer made attempt-dependent": mutate_job(
                "quarot-artifact",
                "name: quarot-q4-${{ github.run_id }}",
                "name: quarot-q4-${{ github.run_id }}-"
                "${{ github.run_attempt }}",
            ),
            "artifact overwrite disabled": mutate_job(
                "quarot-artifact",
                "overwrite: true",
                "overwrite: false",
            ),
            "artifact consumer made attempt-dependent": mutate_job(
                "q4-vision-gates",
                "name: quarot-q4-${{ github.run_id }}",
                "name: quarot-q4-${{ github.run_id }}-"
                "${{ github.run_attempt }}",
            ),
            "artifact checksum verification removed": mutate_job(
                "q4-vision-gates",
                "shasum -a 256 -c quarot-q4.tar.sha256",
                "true # checksum discarded",
            ),
            "macOS artifact dependency removed": mutate_job(
                "q4-vision-gates",
                "needs: [changes, quarot-artifact]",
                "needs: changes",
            ),
            "CPU required needs edge removed": mutate_job(
                "parity-gate",
                "        vision-cpu-gates,\n",
                "",
            ),
            "artifact required needs edge removed": mutate_job(
                "parity-gate",
                "        quarot-artifact,\n",
                "",
            ),
            "CPU required result check disabled": mutate_job(
                "parity-gate",
                'if [ "$VISION_CPU_RESULT" != "success" ]; then',
                "if false; then",
            ),
            "artifact required result check disabled": mutate_job(
                "parity-gate",
                'if [ "$QUAROT_ARTIFACT_RESULT" != "success" ]; then',
                "if false; then",
            ),
        }
        for name, mutated in mutations.items():
            with self.subTest(mutation=name), self.assertRaises(AssertionError):
                self._assert_split_is_fail_closed(mutated)


if __name__ == "__main__":
    _FailOnEmptyTestProgram()

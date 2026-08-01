"""Tests for the fail-closed CI changed-file range selector."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


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
_DETECTOR_LOAD = re.compile(
    r'^[ \t]*if git(?: --no-replace-objects)? show '
    r'"\$\{CI_BASE_SHA\}:scripts/ci-changed-files\.sh" '
    r'> "\$TRUSTED_DETECTOR"; then$',
    re.MULTILINE,
)
_SAFE_FILTER_PROLOGUE_PRIMITIVES = (
    ("strict shell options", re.compile(r"set -euo pipefail")),
    (
        "event policy condition",
        re.compile(
            r'if \[ "\$GITHUB_EVENT_NAME" = "(?:pull_request|workflow_dispatch)" \]'
            r'(?: \|\| \[ "\$GITHUB_EVENT_NAME" = "(?:merge_group|schedule)" \])?; then'
        ),
    ),
    (
        "base format condition",
        re.compile(
            r'if \[\[ ! "\$CI_BASE_SHA" =~ \^\[0-9a-fA-F\]\{40\}\$ \]\]; then'
        ),
    ),
    (
        "all-zero base condition",
        re.compile(r'if \[ "\$CI_BASE_SHA" = "0{40}" \]; then'),
    ),
    (
        "selector output",
        re.compile(
            r'echo "(?:code|deps|bins|swift|engine|tune)=true" '
            r'>> "\$GITHUB_OUTPUT"'
        ),
    ),
    (
        "base validation diagnostic",
        re.compile(
            r'echo "::error::base revision must be a nonempty '
            r'40-character hexadecimal commit ID" >&2'
        ),
    ),
    (
        "selection diagnostic",
        re.compile(
            r'echo "→ (?:'
            r'all-zero base revision: '
            r'(?:full matrix|audit|app-binary and Swift builds|full parity suite) '
            r'REQUIRED'
            r'|manual dispatch: app-binary and Swift builds REQUIRED'
            r'|\$GITHUB_EVENT_NAME trigger: full parity suite REQUIRED)"'
        ),
    ),
    ("early exit", re.compile(r"exit [02]")),
    ("conditional terminator", re.compile(r"fi")),
    (
        "temporary-file allocation",
        re.compile(r"(?:TRUSTED_DETECTOR|CHANGED_FILE)=\$\(mktemp\)"),
    ),
    (
        "temporary-file cleanup",
        re.compile(
            r'''trap 'rm -f "\$TRUSTED_DETECTOR" "\$CHANGED_FILE"' EXIT'''
        ),
    ),
)


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


def _workflow_run_script(workflow: Path, job_id: str, step_id: str) -> str:
    contents = workflow.read_text(encoding="utf-8")
    step = _workflow_step_by_id(_workflow_job(contents, job_id), step_id)
    marker = "        run: |\n"
    if marker not in step:
        raise AssertionError(f"workflow step id {step_id!r} has no run script")
    return textwrap.dedent(step.split(marker, maxsplit=1)[1])


def _assert_safe_filter_prologue(script: str, workflow: str) -> set[str]:
    """Reject every undeclared executable statement before detector loading."""
    detector_load = _DETECTOR_LOAD.search(script)
    if detector_load is None:
        raise AssertionError(f"{workflow} does not load the base detector")
    prefix = script[: detector_load.start()]
    used_primitives = set()
    for line_number, raw_line in enumerate(prefix.splitlines(), start=1):
        statement = raw_line.strip()
        if not statement or statement.startswith("#"):
            continue
        matches = tuple(
            name
            for name, pattern in _SAFE_FILTER_PROLOGUE_PRIMITIVES
            if pattern.fullmatch(statement) is not None
        )
        if len(matches) != 1:
            raise AssertionError(
                f"{workflow} has an undeclared statement before detector "
                f"loading on line {line_number}: {statement!r}"
            )
        used_primitives.add(matches[0])
    return used_primitives


def _require_tests_collected(test_suite: unittest.TestSuite) -> None:
    if test_suite.countTestCases() == 0:
        raise SystemExit("ERROR: no tests collected")


class _FailOnEmptyTestProgram(unittest.TestProgram):
    def runTests(self) -> None:
        _require_tests_collected(self.test)
        super().runTests()


def _engine_change_pattern(contents: str) -> re.Pattern[str]:
    changes = _workflow_job(contents, "changes")
    match = re.search(
        r"""if\ grep\ -E\ '([^']+)'\ <<<"\$CHANGED"\ >/dev/null;\ then
            \s+echo\ "engine=true" """,
        changes,
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
        event: str = "pull_request",
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
        result = subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-o",
                "pipefail",
                "-c",
                _workflow_run_script(workflow, "changes", "filter"),
            ],
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

        result, output = self._run_filter(workflow, base, head)

        self.assertEqual(result.returncode, 0, result.stderr)
        for expected_output in expected_outputs:
            self.assertIn(expected_output, output)

    def _assert_detector_failure_is_observed(self, workflow: Path) -> None:
        base = self._commit(
            "scripts/ci-changed-files.sh",
            "#!/bin/sh\nexit 23\n",
        )
        head = self._commit("README.md", "change\n")

        result, output = self._run_filter(workflow, base, head)

        self.assertEqual(result.returncode, 23)
        self.assertEqual(output, "")
        self.assertIn("change detector failed with status 23", result.stderr)

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

        result, output = self._run_filter(workflow, base, replacement)

        self.assertEqual(result.returncode, 0, result.stderr)
        for expected_output in expected_outputs:
            self.assertIn(expected_output, output)

    def _ci_filter_with_added_prologue(self, statement: str) -> str:
        script = _workflow_run_script(_CI_WORKFLOW, "changes", "filter")
        marker = "TRUSTED_DETECTOR=$(mktemp)"
        self.assertIn(marker, script)
        return script.replace(marker, f"{statement}\n{marker}", 1)

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

    def test_change_jobs_start_filter_immediately_after_checkout(self) -> None:
        for relative_path in _REQUIRED_WORKFLOWS:
            with self.subTest(workflow=relative_path):
                contents = (_ROOT / relative_path).read_text(encoding="utf-8")
                job = _workflow_job(contents, "changes")
                step_entries = re.findall(
                    r"^      -(?: .*)?$",
                    job,
                    re.MULTILINE,
                )

                self.assertGreaterEqual(len(step_entries), 2)
                self.assertRegex(
                    step_entries[0],
                    r"^      - uses: actions/checkout@[0-9a-f]{40}(?: # .*)?$",
                )
                self.assertEqual(step_entries[1], "      - id: filter")

    def test_change_filter_prologues_allow_only_declared_operations(
        self,
    ) -> None:
        used_primitives = set()
        for relative_path in _REQUIRED_WORKFLOWS:
            with self.subTest(workflow=relative_path):
                script = _workflow_run_script(
                    _ROOT / relative_path,
                    "changes",
                    "filter",
                )
                used_primitives.update(
                    _assert_safe_filter_prologue(script, relative_path)
                )

        self.assertEqual(
            used_primitives,
            {name for name, _ in _SAFE_FILTER_PROLOGUE_PRIMITIVES},
        )

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
            r"if git(?: --no-replace-objects)? show ",
            'if git "$(scripts/preload-detector)" show ',
            script,
            count=1,
        )
        self.assertEqual(replacements, 1)

        with self.assertRaisesRegex(AssertionError, "base detector"):
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
            with self.subTest(workflow=workflow.name):
                result, output = self._run_filter(workflow, "f" * 40, head)

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
            for base_sha in ("", "a" * 39, "g" * 40):
                with self.subTest(workflow=workflow.name, base_sha=base_sha):
                    result, output = self._run_filter(workflow, base_sha, head)

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
        self.assertIn("-- --model bge-small-en-v1.5 --download-only", self.job)
        self.assertIn("LATTICE_DRIFT_GATE_ENFORCE: '1'", self.job)
        self.assertIn(
            "-- --nocapture > embed-drift-x86.log 2>&1",
            self.job,
        )
        self.assertIn('exit "$status"', self.job)
        self.assertIn(
            "grep -Fq 'Loaded drift baseline:' embed-drift-x86.log",
            self.job,
        )
        self.assertIn(
            "grep -Fq '[bge-small drift gate] max(1-cosine)=' "
            "embed-drift-x86.log",
            self.job,
        )

    def test_observation_does_not_replace_the_required_arm_gate(self) -> None:
        arm_job = _workflow_job(self.contents, "embed-drift")
        self.assertIn("runs-on: ubuntu-24.04-arm", arm_job)
        self.assertIn("LATTICE_DRIFT_GATE_ENFORCE: '1'", arm_job)

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

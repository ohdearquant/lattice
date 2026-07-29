"""Tests for the fail-closed CI changed-file range selector."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci-changed-files.sh"
_ROOT = _SCRIPT.parent.parent
_ZERO_SHA = "0" * 40
_REQUIRED_WORKFLOWS = (
    ".github/workflows/app-binaries.yml",
    ".github/workflows/cargo-audit.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/e2e-parity.yml",
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
        r"^      - (?:name:|uses:)",
        job[body_start:],
        re.MULTILINE,
    )
    end = len(job) if next_match is None else body_start + next_match.start()
    return job[start:end]


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
        result = subprocess.run(
            ["git", *args],
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
    unittest.main()

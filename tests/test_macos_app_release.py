"""Contract tests for macOS app release packaging and upload (issue #390)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPT = REPO_ROOT / "apps/macos/scripts/package-app.sh"
UPLOAD_SCRIPT = REPO_ROOT / "apps/macos/scripts/upload-release-assets.sh"
RELEASE_WORKFLOW = REPO_ROOT / ".github/workflows/release-binaries.yml"
APP_BINARIES_WORKFLOW = REPO_ROOT / ".github/workflows/app-binaries.yml"
MACOS_SOURCES = REPO_ROOT / "apps/macos/Sources"
RELEASE_TARGETS = (
    "aarch64-apple-darwin",
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
)
TEST_TAG_SHA = "a" * 40


def workspace_version() -> str:
    result = subprocess.run(
        [
            "cargo",
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
            str(REPO_ROOT / "Cargo.toml"),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    members = set(metadata["workspace_members"])
    versions = {
        package["version"]
        for package in metadata["packages"]
        if package["id"] in members
    }
    if len(versions) != 1:
        raise AssertionError(
            f"workspace packages must share exactly one version: {versions}"
        )
    return next(iter(versions))


def release_payload_names(tag: str) -> list[str]:
    version = tag.removeprefix("v")
    return [
        *(f"lattice-{version}-{target}.tar.gz" for target in RELEASE_TARGETS),
        "Lattice.dmg",
        "Lattice.zip",
    ]


def write_release_assets(directory: Path, tag: str, marker: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in release_payload_names(tag):
        payload = f"{marker}:{name}".encode()
        (directory / name).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        (directory / f"{name}.sha256").write_text(
            f"{digest}  {name}\n", encoding="utf-8"
        )


def workflow_step_script(
    workflow: Path, *, step_id: str | None = None, name: str | None = None
) -> str:
    if (step_id is None) == (name is None):
        raise ValueError("exactly one workflow step selector is required")
    marker = f"- id: {step_id}" if step_id is not None else f"- name: {name}"
    lines = workflow.read_text(encoding="utf-8").splitlines()
    try:
        marker_index = next(
            index for index, line in enumerate(lines) if line.strip() == marker
        )
    except StopIteration as error:
        raise AssertionError(f"{workflow.name} has no step {marker}") from error
    try:
        run_index = next(
            index
            for index in range(marker_index + 1, len(lines))
            if lines[index].strip() == "run: |"
        )
    except StopIteration as error:
        raise AssertionError(f"{workflow.name} step {marker} has no run block") from error

    run_indent = len(lines[run_index]) - len(lines[run_index].lstrip())
    block: list[str] = []
    for line in lines[run_index + 1 :]:
        indent = len(line) - len(line.lstrip())
        if line.strip() and indent <= run_indent:
            break
        block.append(line)
    return textwrap.dedent("\n".join(block)).strip() + "\n"


def require_tests_collected(test_suite: unittest.TestSuite) -> None:
    if test_suite.countTestCases() == 0:
        raise SystemExit("ERROR: no tests collected")


class FailOnEmptyTestProgram(unittest.TestProgram):
    def runTests(self) -> None:
        require_tests_collected(self.test)
        super().runTests()


class PackageScriptContractTest(unittest.TestCase):
    def test_app_version_matches_workspace_version(self):
        result = subprocess.run(
            ["bash", str(PACKAGE_SCRIPT), "--print-version"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), workspace_version())

    def test_shared_cargo_target_is_the_binary_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shared-target"
            env = os.environ.copy()
            env["CARGO_TARGET_DIR"] = str(target)
            result = subprocess.run(
                ["bash", str(PACKAGE_SCRIPT), "--print-target-release"],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.stdout.strip(), str(target / "release"))

    def test_shell_scripts_parse(self):
        for script in (PACKAGE_SCRIPT, UPLOAD_SCRIPT):
            subprocess.run(["bash", "-n", str(script)], cwd=REPO_ROOT, check=True)

    def test_preview_macros_are_release_and_toolchain_gated(self):
        required_guard = "#if DEBUG && canImport(PreviewsMacros)"
        preview_count = 0
        for source in sorted(MACOS_SOURCES.rglob("*.swift")):
            conditional_stack: list[str] = []
            for line_number, line in enumerate(
                source.read_text(encoding="utf-8").splitlines(), start=1
            ):
                directive = line.strip()
                if directive.startswith("#if "):
                    conditional_stack.append(directive)
                elif directive.startswith(("#elseif ", "#else")):
                    if conditional_stack:
                        conditional_stack[-1] = directive
                elif directive == "#endif":
                    if conditional_stack:
                        conditional_stack.pop()
                elif directive.startswith("#Preview"):
                    preview_count += 1
                    self.assertIn(
                        required_guard,
                        conditional_stack,
                        f"{source.relative_to(REPO_ROOT)}:{line_number}",
                    )
        self.assertGreater(preview_count, 0)


class UploadContractTest(unittest.TestCase):
    maxDiff = None

    def run_uploader(
        self,
        artifact_dir: Path,
        tag: str,
        remote_dir: Path,
        *,
        draft: bool,
        tag_sha: str = TEST_TAG_SHA,
        remote_tag_sha: str | None = None,
        event_name: str = "workflow_dispatch",
        fail_uploads: int = 0,
        interrupt_upload: bool = False,
        publish_during_upload: bool = False,
        publish_after_view: int = 0,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object], list[list[str]]]:
        root = artifact_dir.parent
        bin_dir = root / "bin"
        bin_dir.mkdir(exist_ok=True)
        fake_gh = bin_dir / "gh"
        fake_gh.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import os
                import signal
                import shutil
                import sys
                from pathlib import Path

                state_path = Path(os.environ["FAKE_GH_STATE"])
                log_path = Path(os.environ["FAKE_GH_LOG"])
                state = json.loads(state_path.read_text(encoding="utf-8"))
                args = sys.argv[1:]
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(args) + "\\n")

                remote = Path(state["remote_dir"])
                remote.mkdir(parents=True, exist_ok=True)

                if args[0] == "api":
                    print(f"commit {state['tag_sha']}")
                elif args[:2] == ["release", "view"]:
                    state["release_views"] += 1
                    if state["release_views"] == state["publish_after_view"]:
                        state["draft"] = False
                    state_path.write_text(json.dumps(state), encoding="utf-8")
                    print("true" if state["draft"] else "false")
                elif args[:2] == ["release", "upload"]:
                    repo_index = args.index("--repo")
                    assets = [Path(value) for value in args[3:repo_index]]
                    if state["publish_during_upload"]:
                        state["draft"] = False
                    inject = state["fail_uploads"] > 0
                    if inject:
                        state["fail_uploads"] -= 1
                    for index, asset in enumerate(assets):
                        destination = remote / asset.name
                        destination.unlink(missing_ok=True)
                        if state["interrupt_upload"] and index == 1:
                            state["interrupted"] = True
                            state_path.write_text(json.dumps(state), encoding="utf-8")
                            os.killpg(os.getpgrp(), signal.SIGTERM)
                        if inject and index == 1:
                            state.setdefault("injected_exit_codes", []).append(42)
                            state_path.write_text(json.dumps(state), encoding="utf-8")
                            raise SystemExit(42)
                        shutil.copy2(asset, destination)
                    state_path.write_text(json.dumps(state), encoding="utf-8")
                elif args[:2] == ["release", "download"]:
                    destination = Path(args[args.index("--dir") + 1])
                    destination.mkdir(parents=True, exist_ok=True)
                    for source in remote.iterdir():
                        if source.is_file():
                            shutil.copy2(source, destination / source.name)
                elif args[:2] == ["release", "edit"]:
                    if "--draft=false" not in args:
                        raise SystemExit("expected --draft=false")
                    state["draft"] = False
                    state_path.write_text(json.dumps(state), encoding="utf-8")
                else:
                    raise SystemExit(f"unsupported fake gh invocation: {args}")
                """
            ),
            encoding="utf-8",
        )
        fake_gh.chmod(0o755)
        remote_dir.mkdir(parents=True, exist_ok=True)
        state_path = root / "gh-state.json"
        log_path = root / "gh-log.jsonl"
        state_path.write_text(
            json.dumps(
                {
                    "draft": draft,
                    "fail_uploads": fail_uploads,
                    "interrupt_upload": interrupt_upload,
                    "publish_during_upload": publish_during_upload,
                    "publish_after_view": publish_after_view,
                    "release_views": 0,
                    "remote_dir": str(remote_dir),
                    "tag_sha": remote_tag_sha or tag_sha,
                }
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["GITHUB_EVENT_NAME"] = event_name
        env["FAKE_GH_LOG"] = str(log_path)
        env["FAKE_GH_STATE"] = str(state_path)
        env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
        result = subprocess.run(
            [
                "bash",
                str(UPLOAD_SCRIPT),
                tag,
                "ohdearquant/lattice",
                tag_sha,
                str(artifact_dir),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            start_new_session=True,
        )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        log = []
        if log_path.exists():
            log = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
        return result, state, log

    def test_draft_upload_verifies_complete_inventory_before_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            remote = root / "remote"
            tag = f"v{workspace_version()}"
            write_release_assets(artifacts, tag, "new")

            result, state, log = self.run_uploader(
                artifacts, tag, remote, draft=True
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(state["draft"])
            self.assertEqual(
                sorted(path.name for path in remote.iterdir()),
                sorted(path.name for path in artifacts.iterdir()),
            )
            actions = [args[:2] for args in log]
            upload_index = actions.index(["release", "upload"])
            verify_index = actions.index(["release", "download"])
            publish_index = actions.index(["release", "edit"])
            self.assertLess(upload_index, verify_index)
            self.assertLess(verify_index, publish_index)

    def test_draft_with_unexpected_remote_asset_stays_unpublished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            remote = root / "remote"
            tag = f"v{workspace_version()}"
            write_release_assets(artifacts, tag, "new")
            remote.mkdir()
            (remote / "unexpected.txt").write_text("unexpected", encoding="utf-8")

            result, state, log = self.run_uploader(
                artifacts, tag, remote, draft=True
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(state["draft"])
            self.assertIn("unexpected release asset", result.stderr)
            self.assertNotIn(
                ["release", "edit"], [args[:2] for args in log]
            )

    def test_moved_release_tag_is_rejected_before_release_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            remote = root / "remote"
            tag = f"v{workspace_version()}"
            write_release_assets(artifacts, tag, "new")

            result, _, log = self.run_uploader(
                artifacts,
                tag,
                remote,
                draft=True,
                remote_tag_sha="b" * 40,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected commit", result.stderr)
            self.assertEqual([args[0] for args in log], ["api"])

    def test_published_release_is_refused_without_remote_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            remote = root / "remote"
            tag = f"v{workspace_version()}"
            write_release_assets(artifacts, tag, "new")
            write_release_assets(remote, tag, "old")
            previous = {
                path.name: path.read_bytes() for path in remote.iterdir()
            }

            result, state, log = self.run_uploader(
                artifacts,
                tag,
                remote,
                draft=False,
                fail_uploads=2,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already published", result.stderr)
            self.assertEqual(state["fail_uploads"], 2)
            self.assertEqual(
                {path.name: path.read_bytes() for path in remote.iterdir()},
                previous,
            )
            mutations = [
                args
                for args in log
                if args[:2] in (["release", "upload"], ["release", "edit"])
            ]
            self.assertEqual(mutations, [])

    def test_interrupted_draft_upload_stays_unpublished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            remote = root / "remote"
            tag = f"v{workspace_version()}"
            write_release_assets(artifacts, tag, "new")

            result, state, log = self.run_uploader(
                artifacts,
                tag,
                remote,
                draft=True,
                interrupt_upload=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(state["draft"])
            self.assertTrue(state["interrupted"])
            self.assertNotIn(
                ["release", "edit"], [args[:2] for args in log]
            )

    def test_failed_draft_upload_stays_unpublished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            remote = root / "remote"
            tag = f"v{workspace_version()}"
            write_release_assets(artifacts, tag, "new")

            result, state, log = self.run_uploader(
                artifacts,
                tag,
                remote,
                draft=True,
                fail_uploads=1,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(state["draft"])
            self.assertEqual(state["injected_exit_codes"], [42])
            self.assertNotIn(
                ["release", "edit"], [args[:2] for args in log]
            )

    def test_publication_during_upload_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            remote = root / "remote"
            tag = f"v{workspace_version()}"
            write_release_assets(artifacts, tag, "new")

            result, state, log = self.run_uploader(
                artifacts,
                tag,
                remote,
                draft=True,
                publish_during_upload=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(state["draft"])
            self.assertIn("may already have changed", result.stderr)
            self.assertNotIn(
                ["release", "edit"], [args[:2] for args in log]
            )
            self.assertEqual(
                {path.name: path.read_bytes() for path in remote.iterdir()},
                {path.name: path.read_bytes() for path in artifacts.iterdir()},
            )

    def test_publication_before_upload_is_refused_without_remote_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            remote = root / "remote"
            tag = f"v{workspace_version()}"
            write_release_assets(artifacts, tag, "new")

            result, state, log = self.run_uploader(
                artifacts,
                tag,
                remote,
                draft=True,
                publish_after_view=2,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(state["draft"])
            self.assertIn("before upload", result.stderr)
            self.assertEqual(list(remote.iterdir()), [])
            self.assertNotIn(
                ["release", "upload"], [args[:2] for args in log]
            )

    def test_publication_during_verification_stops_before_publish_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            remote = root / "remote"
            tag = f"v{workspace_version()}"
            write_release_assets(artifacts, tag, "new")

            result, state, log = self.run_uploader(
                artifacts,
                tag,
                remote,
                draft=True,
                publish_after_view=4,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(state["draft"])
            self.assertIn("became published during verification", result.stderr)
            actions = [args[:2] for args in log]
            self.assertIn(["release", "download"], actions)
            self.assertNotIn(["release", "edit"], actions)

    def test_real_packaged_assets_when_requested(self):
        artifact_dir_value = os.environ.get(
            "LATTICE_MACOS_RELEASE_ARTIFACT_DIR"
        )
        if artifact_dir_value is None:
            self.skipTest("real package artifact directory not requested")

        artifact_dir = Path(artifact_dir_value).resolve()
        dmg = artifact_dir / "Lattice.dmg"
        zip_file = artifact_dir / "Lattice.zip"
        for asset in (dmg, zip_file):
            self.assertTrue(asset.is_file(), asset)
            self.assertGreater(asset.stat().st_size, 0, asset)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged = root / "dist"
            remote = root / "remote"
            tag = f"v{workspace_version()}"
            write_release_assets(staged, tag, "fixture")
            for asset in (dmg, zip_file):
                staged_asset = staged / asset.name
                staged_asset.write_bytes(asset.read_bytes())
                digest = hashlib.sha256(asset.read_bytes()).hexdigest()
                (staged / f"{asset.name}.sha256").write_text(
                    f"{digest}  {asset.name}\n", encoding="utf-8"
                )
            result, _, _ = self.run_uploader(
                staged, tag, remote, draft=True
            )

            self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_malformed_release_tag_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            remote = root / "remote"
            write_release_assets(artifacts, "0.0.0", "new")

            result, _, log = self.run_uploader(
                artifacts, "0.0.0", remote, draft=True
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must use vMAJOR.MINOR.PATCH form", result.stderr)
            self.assertEqual(log, [])

    def test_rejects_missing_asset_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            remote = root / "remote"
            tag = f"v{workspace_version()}"
            write_release_assets(artifacts, tag, "new")
            (artifacts / "Lattice.zip").unlink()

            result, _, log = self.run_uploader(
                artifacts, tag, remote, draft=True
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Lattice.zip", result.stderr)
            self.assertEqual(log, [])

    def test_rejects_non_release_workflow_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            remote = root / "remote"
            tag = f"v{workspace_version()}"
            write_release_assets(artifacts, tag, "new")

            result, _, log = self.run_uploader(
                artifacts,
                tag,
                remote,
                draft=True,
                event_name="pull_request",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("only upload from a release workflow", result.stderr)
            self.assertEqual(log, [])


class WorkflowContractTest(unittest.TestCase):
    def test_release_workflow_stages_all_assets_before_single_publish(self):
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("types: [published]", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("Existing draft release tag", workflow)
        self.assertIn("Require an existing draft release", workflow)
        self.assertIn('if [ "$IS_DRAFT" != "true" ]; then', workflow)
        self.assertIn("group: release-binaries-${{ inputs.tag }}", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("\n  prepare:\n", workflow)
        marker = "\n  macos-app:\n"
        self.assertIn(marker, workflow)
        macos_job = workflow.split(marker, maxsplit=1)[1]
        self.assertIn("runs-on: macos-latest", macos_job)
        self.assertIn("timeout-minutes: 90", macos_job)
        self.assertIn("package-app.sh --print-version", macos_job)
        self.assertIn(
            "run: ./apps/macos/scripts/package-app.sh",
            macos_job,
        )
        self.assertIn("actions/upload-artifact", macos_job)
        self.assertNotIn("gh release", macos_job.split("\n  publish:\n")[0])
        publish_job = workflow.split("\n  publish:\n", maxsplit=1)[1]
        self.assertIn("needs: [prepare, build, macos-app]", publish_job)
        self.assertIn("actions/download-artifact", publish_job)
        self.assertIn("upload-release-assets.sh", publish_job)
        self.assertIn("environment: release", publish_job)

    def test_release_workflow_scopes_write_permission_to_publish_job(self):
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("permissions:\n  contents: read", workflow)
        before_publish, publish_job = workflow.split("\n  publish:\n", maxsplit=1)
        self.assertNotIn("contents: write", before_publish)
        self.assertIn("permissions:\n      contents: write", publish_job)
        self.assertNotIn("secrets.GITHUB_TOKEN", before_publish)
        builder_jobs = before_publish.split("\n  build:\n", maxsplit=1)[1]
        self.assertNotIn("GH_TOKEN", builder_jobs)
        self.assertGreaterEqual(workflow.count("persist-credentials: false"), 4)

    def test_pull_request_gate_executes_this_contract(self):
        workflow = APP_BINARIES_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "run: python3 tests/test_macos_app_release.py -v",
            workflow,
        )
        for path in (
            "apps/macos/scripts/package-app.sh",
            "apps/macos/scripts/upload-release-assets.sh",
            ".github/workflows/release-binaries.yml",
            "tests/test_macos_app_release.py",
        ):
            self.assertIn(f"- '{path}'", workflow)

    def test_pull_request_gate_builds_swift_release_for_macos_sources(self):
        workflow = APP_BINARIES_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("swift: ${{ steps.filter.outputs.swift }}", workflow)
        self.assertIn("^apps/macos/", workflow)
        self.assertIn("\n  build-macos-swift:\n", workflow)
        swift_job = workflow.split("\n  build-macos-swift:\n", maxsplit=1)[1]
        swift_job = swift_job.split("\n  app-binaries-gate:\n", maxsplit=1)[0]
        self.assertIn("needs: changes", swift_job)
        self.assertIn("needs.changes.outputs.swift == 'true'", swift_job)
        self.assertIn(
            "swift build -c release --package-path apps/macos", swift_job
        )
        gate = workflow.split("\n  app-binaries-gate:\n", maxsplit=1)[1]
        self.assertIn("build-macos-swift", gate)
        self.assertIn("SWIFT_RESULT", gate)
        self.assertIn("SWIFT", gate)

    def test_release_workflow_rejects_published_release_behaviorally(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_gh = bin_dir / "gh"
            fake_gh.write_text(
                "#!/bin/sh\n" 'printf "false\\n"\n', encoding="utf-8"
            )
            fake_gh.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "GH_TOKEN": "fixture-token",
                    "TAG": "v0.0.0",
                    "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
                }
            )
            script = workflow_step_script(
                RELEASE_WORKFLOW, name="Require an existing draft release"
            ).replace("${{ github.repository }}", "ohdearquant/lattice")

            result = subprocess.run(
                ["bash", "-c", script],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already published", result.stdout + result.stderr)

    def test_release_docs_create_drafts(self):
        documents = (
            (REPO_ROOT / "docs/RELEASE.md", "VERSION"),
            (REPO_ROOT / "docs/_templates/RELEASE.md", "VERSION"),
        )
        for document, version in documents:
            with self.subTest(document=document.relative_to(REPO_ROOT)):
                contents = document.read_text(encoding="utf-8")
                draft = f"gh release create v{{{version}}} --draft"
                self.assertIn(draft, contents)

    def test_release_docs_dispatch_main_workflow_after_release_creation(self):
        documents = (
            (REPO_ROOT / "docs/RELEASE.md", "VERSION"),
            (REPO_ROOT / "docs/_templates/RELEASE.md", "VERSION"),
        )
        for document, version in documents:
            with self.subTest(document=document.relative_to(REPO_ROOT)):
                contents = document.read_text(encoding="utf-8")
                create = f"gh release create v{{{version}}}"
                dispatch = (
                    "gh workflow run release-binaries.yml "
                    "--repo ohdearquant/lattice --ref main "
                    f"-f tag=v{{{version}}}"
                )
                self.assertIn(create, contents)
                self.assertIn(dispatch, contents)
                self.assertLess(contents.index(create), contents.index(dispatch))

    def test_release_recovery_uses_new_draft_before_yanking(self):
        documents = (
            (REPO_ROOT / "docs/RELEASE.md", "## Bump-and-Yank Recovery"),
            (REPO_ROOT / "docs/_templates/RELEASE.md", "## Rollback"),
        )
        for document, heading in documents:
            with self.subTest(document=document.relative_to(REPO_ROOT)):
                contents = document.read_text(encoding="utf-8")
                recovery = contents.split(heading, maxsplit=1)[1]
                draft = "gh release create v{NEW_VERSION} --draft"
                dispatch = (
                    "gh workflow run release-binaries.yml "
                    "--repo ohdearquant/lattice --ref main "
                    "-f tag=v{NEW_VERSION}"
                )
                self.assertIn(draft, recovery)
                self.assertIn(dispatch, recovery)
                self.assertLess(recovery.index(draft), recovery.index(dispatch))
                self.assertLess(recovery.index(dispatch), recovery.index("cargo yank"))
                self.assertRegex(recovery, r"new (?:version.*tag|tag.*version)")


class RunnerContractTest(unittest.TestCase):
    def test_empty_collection_fails_closed(self):
        with self.assertRaisesRegex(SystemExit, "no tests collected"):
            require_tests_collected(unittest.TestSuite())

    def test_entry_point_fails_when_filter_collects_no_tests(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__)), "-k", "no_test_can_match_this"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no tests collected", result.stdout + result.stderr)


if __name__ == "__main__":
    FailOnEmptyTestProgram()

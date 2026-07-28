"""Contract tests for macOS app release packaging and upload (issue #390)."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPT = REPO_ROOT / "apps/macos/scripts/package-app.sh"
UPLOAD_SCRIPT = REPO_ROOT / "apps/macos/scripts/upload-release-assets.sh"
RELEASE_WORKFLOW = REPO_ROOT / ".github/workflows/release-binaries.yml"
APP_BINARIES_WORKFLOW = REPO_ROOT / ".github/workflows/app-binaries.yml"


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


class UploadContractTest(unittest.TestCase):
    def run_uploader(
        self,
        artifact_dir: Path,
        tag: str,
        capture: Path,
        *,
        event_name: str = "release",
    ) -> subprocess.CompletedProcess[str]:
        bin_dir = capture.parent / "bin"
        bin_dir.mkdir(exist_ok=True)
        fake_gh = bin_dir / "gh"
        fake_gh.write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\nprintf "%s\\n" "$@" > "$GH_CAPTURE"\n',
            encoding="utf-8",
        )
        fake_gh.chmod(0o755)
        env = os.environ.copy()
        env["GITHUB_EVENT_NAME"] = event_name
        env["GH_CAPTURE"] = str(capture)
        env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
        return subprocess.run(
            [
                "bash",
                str(UPLOAD_SCRIPT),
                tag,
                "ohdearquant/lattice",
                str(artifact_dir),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_uploads_exact_versioned_dmg_zip_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            artifacts.mkdir()
            dmg = artifacts / "Lattice.dmg"
            zip_file = artifacts / "Lattice.zip"
            dmg.write_bytes(b"dmg")
            zip_file.write_bytes(b"zip")
            capture = root / "gh-args"

            result = self.run_uploader(
                artifacts, f"v{workspace_version()}", capture
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                capture.read_text(encoding="utf-8").splitlines(),
                [
                    "release",
                    "upload",
                    f"v{workspace_version()}",
                    str(dmg),
                    f"{dmg}.sha256",
                    str(zip_file),
                    f"{zip_file}.sha256",
                    "--repo",
                    "ohdearquant/lattice",
                    "--clobber",
                ],
            )
            for asset in (dmg, zip_file):
                checksum = Path(f"{asset}.sha256")
                self.assertTrue(checksum.is_file())
                self.assertIn(str(asset), checksum.read_text(encoding="utf-8"))

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
            capture = Path(tmp) / "gh-args"
            result = self.run_uploader(
                artifact_dir, f"v{workspace_version()}", capture
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                capture.read_text(encoding="utf-8").splitlines(),
                [
                    "release",
                    "upload",
                    f"v{workspace_version()}",
                    str(dmg),
                    f"{dmg}.sha256",
                    str(zip_file),
                    f"{zip_file}.sha256",
                    "--repo",
                    "ohdearquant/lattice",
                    "--clobber",
                ],
            )

    def test_rejects_release_tag_that_does_not_match_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            artifacts.mkdir()
            (artifacts / "Lattice.dmg").write_bytes(b"dmg")
            (artifacts / "Lattice.zip").write_bytes(b"zip")
            capture = root / "gh-args"

            result = self.run_uploader(artifacts, "v0.0.0", capture)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match workspace version", result.stderr)
            self.assertFalse(capture.exists())

    def test_rejects_missing_asset_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            artifacts.mkdir()
            (artifacts / "Lattice.dmg").write_bytes(b"dmg")
            capture = root / "gh-args"

            result = self.run_uploader(
                artifacts, f"v{workspace_version()}", capture
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Lattice.zip", result.stderr)
            self.assertFalse(capture.exists())

    def test_rejects_non_release_workflow_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "dist"
            artifacts.mkdir()
            (artifacts / "Lattice.dmg").write_bytes(b"dmg")
            (artifacts / "Lattice.zip").write_bytes(b"zip")
            capture = root / "gh-args"

            result = self.run_uploader(
                artifacts,
                f"v{workspace_version()}",
                capture,
                event_name="pull_request",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("only upload from a release workflow", result.stderr)
            self.assertFalse(capture.exists())


class WorkflowContractTest(unittest.TestCase):
    def test_existing_published_release_workflow_builds_and_uploads_app(self):
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("release:\n    types: [published]", workflow)
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
        self.assertIn("upload-release-assets.sh", macos_job)
        self.assertIn("GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}", macos_job)

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


if __name__ == "__main__":
    unittest.main()

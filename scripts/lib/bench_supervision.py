#!/usr/bin/env python3
"""Run a local measurement under the repository's machine-wide supervisor.

Measurement entry points call :func:`ensure_python_entrypoint` before doing
work. Shell and Node entry points invoke the ``run``/``verify`` CLI below.
The outer process is always ``bench-locks.py``; this module only supplies the
common re-exec convention and, for durable evidence, ambient-idle checks.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCKS = REPO / "scripts" / "lib" / "bench-locks.py"
QUIET_PROBE = REPO / "scripts" / "lib" / "quiet-probe.py"
STATUS_ENV = "LATTICE_BENCH_LOCK_STATUS"
FDS_ENV = "LATTICE_BENCH_LOCK_FDS"
QUIET_ENV = "LATTICE_BENCH_QUIET_SUPERVISION"
REFUSAL_EXIT = 2


class SupervisionError(RuntimeError):
    """The claimed lock capabilities do not prove benchmark supervision."""


def _canonical_lock_paths() -> tuple[Path, Path]:
    config = runpy.run_path(str(LOCKS))
    try:
        paths = (Path(config["BENCH_WINDOW"]), Path(config["GPU_LOCK"]))
    except (KeyError, TypeError) as exc:
        raise SupervisionError("bench-locks.py has no canonical lock paths") from exc
    return paths


def _lock_paths(lock_lines: list[str]) -> list[Path]:
    paths: list[Path] = []
    for line in lock_lines:
        match = re.search(r"\((/[^)]*)\):", line)
        if match is None:
            raise SupervisionError(f"lock receipt line has no absolute path: {line!r}")
        paths.append(Path(match.group(1)))
    return paths


def _validated_receipt_paths(lock_lines: list[str]) -> tuple[Path, Path]:
    receipt_paths = _lock_paths(lock_lines)
    canonical_paths = _canonical_lock_paths()
    for receipt_path, canonical_path in zip(
        receipt_paths, canonical_paths, strict=True
    ):
        if receipt_path != canonical_path:
            raise SupervisionError(
                f"lock receipt names {receipt_path}; expected {canonical_path}"
            )
    return canonical_paths


def _verify_inherited_fds(paths: tuple[Path, Path]) -> tuple[int, int]:
    raw = os.environ[FDS_ENV]
    try:
        fds = tuple(int(value) for value in raw.split(","))
    except ValueError as exc:
        raise SupervisionError(f"{FDS_ENV} contains a non-integer descriptor") from exc
    if len(fds) != 2 or len(set(fds)) != 2:
        raise SupervisionError(f"{FDS_ENV} must name exactly two descriptors")

    probes: list[int] = []
    inode_pairs: list[tuple[int, int]] = []
    try:
        for fd, path in zip(fds, paths, strict=True):
            try:
                probe = os.open(path, os.O_RDWR)
                probes.append(probe)
                fd_stat = os.fstat(fd)
                path_stat = os.fstat(probe)
            except OSError as exc:
                raise SupervisionError(
                    f"inherited lock descriptor {fd} cannot be matched to {path}: {exc}"
                ) from exc
            fd_pair = (fd_stat.st_dev, fd_stat.st_ino)
            path_pair = (path_stat.st_dev, path_stat.st_ino)
            if fd_pair != path_pair:
                raise SupervisionError(
                    f"inherited descriptor {fd} does not refer to canonical path {path}"
                )
            inode_pairs.append(path_pair)

        if len(set(inode_pairs)) != 2:
            raise SupervisionError("canonical benchmark locks must use distinct inodes")

        for probe, path in zip(probes, paths, strict=True):
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    continue
                raise SupervisionError(
                    f"could not probe inherited lock {path}: {exc}"
                ) from exc
            else:
                fcntl.flock(probe, fcntl.LOCK_UN)
                raise SupervisionError(
                    f"canonical lock {path} was not already held before verification"
                )

        for fd, path in zip(fds, paths, strict=True):
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise SupervisionError(
                        f"inherited descriptor {fd} does not carry the lock on {path}"
                    ) from exc
                raise SupervisionError(
                    f"could not verify inherited lock descriptor {fd}: {exc}"
                ) from exc
    finally:
        for probe in probes:
            os.close(probe)
    return fds[0], fds[1]


def _read_receipt() -> tuple[Path, list[str]]:
    raw_path = os.environ.get(STATUS_ENV)
    if not raw_path:
        raise SupervisionError(f"{STATUS_ENV} is not set")
    status = Path(raw_path)
    try:
        lines = status.read_text().splitlines()
    except OSError as exc:
        raise SupervisionError(f"cannot read lock receipt {status}: {exc}") from exc

    pid_lines = [line for line in lines if line.startswith("supervisor_pid=")]
    if len(pid_lines) != 1:
        raise SupervisionError(
            f"lock receipt {status} must contain exactly one supervisor_pid"
        )
    try:
        int(pid_lines[0].split("=", 1)[1])
    except ValueError as exc:
        raise SupervisionError(
            f"lock receipt {status} has an invalid supervisor PID"
        ) from exc

    lock_lines = [line for line in lines if line.startswith("lock=")]
    if len(lock_lines) != 2:
        raise SupervisionError(
            f"lock receipt {status} must contain both machine-wide locks"
        )
    return status, lock_lines


def verify_supervision() -> tuple[Path, tuple[int, ...]]:
    """Return the validated status path/capabilities or raise."""

    status, lock_lines = _read_receipt()
    if FDS_ENV not in os.environ:
        raise SupervisionError(f"{FDS_ENV} is not set")
    paths = _validated_receipt_paths(lock_lines)
    return status, _verify_inherited_fds(paths)


def verify_retained_supervision() -> Path:
    """Prove another descriptor still holds both locks after capability retirement."""

    status, lock_lines = _read_receipt()
    paths = _validated_receipt_paths(lock_lines)
    for path in paths:
        try:
            probe = os.open(path, os.O_RDWR)
        except OSError as exc:
            raise SupervisionError(f"could not open retained lock {path}: {exc}") from exc
        try:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    continue
                raise SupervisionError(
                    f"could not probe retained lock {path}: {exc}"
                ) from exc
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise SupervisionError(
                f"canonical lock {path} was released after inherited descriptors closed"
            )
        finally:
            os.close(probe)
    return status


def _slug(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")
    return slug or "measurement"


def _status_path(label: str) -> Path:
    return REPO / ".cache" / "bench-supervision" / f"{_slug(label)}.status"


def _quiet(label: str) -> bool:
    result = subprocess.run(
        [sys.executable, str(QUIET_PROBE), "--label", label],
        check=False,
    )
    return result.returncode == 0


def run_supervised(
    label: str,
    command: list[str],
    *,
    quiet: bool,
    entrypoint: bool = False,
) -> int:
    """Acquire both locks if needed, then run ``command`` under their receipt."""

    if not command:
        raise SupervisionError("no measurement command supplied")

    if STATUS_ENV not in os.environ:
        status = _status_path(label)
        status.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env[STATUS_ENV] = str(status)
        if quiet:
            env[QUIET_ENV] = "1"
        else:
            env.pop(QUIET_ENV, None)
        argv = [
            sys.executable,
            str(LOCKS),
            "--label",
            label,
            "--status-file",
            str(status),
            "--pass-lock-fds",
            "--",
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
            "--label",
            label,
        ]
        if quiet:
            argv.append("--quiet")
        if entrypoint:
            argv.append("--entrypoint")
        argv.extend(["--", *command])
        os.execvpe(sys.executable, argv, env)

    _, inherited_fds = verify_supervision()
    if quiet and not _quiet(f"{label}: before"):
        print(
            f"bench-supervision: machine was not quiet before {label}; "
            "refusing to measure",
            file=sys.stderr,
        )
        return REFUSAL_EXIT

    child_env = os.environ.copy()
    if quiet:
        child_env[QUIET_ENV] = "1"
    if not entrypoint:
        child_env.pop(FDS_ENV, None)
    result = subprocess.run(
        command,
        check=False,
        env=child_env,
        pass_fds=inherited_fds if entrypoint else (),
    )
    if result.returncode != 0:
        return result.returncode

    if quiet and not _quiet(f"{label}: after"):
        print(
            f"bench-supervision: machine was not quiet after {label}; "
            "refusing to certify the result",
            file=sys.stderr,
        )
        return REFUSAL_EXIT
    return 0


def ensure_python_entrypoint(label: str, *, quiet: bool = False) -> None:
    """Re-exec this Python entry point under supervision when necessary."""

    if STATUS_ENV not in os.environ:
        command = [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
        rc = run_supervised(label, command, quiet=quiet, entrypoint=True)
        raise SystemExit(rc)
    try:
        _, inherited_fds = verify_supervision()
        for fd in inherited_fds:
            os.close(fd)
        os.environ.pop(FDS_ENV, None)
        verify_retained_supervision()
    except SupervisionError as exc:
        print(f"bench-supervision: {exc}; refusing to measure", file=sys.stderr)
        raise SystemExit(REFUSAL_EXIT) from exc
    if quiet and os.environ.get(QUIET_ENV) != "1":
        print(
            f"bench-supervision: {label} requires ambient-idle gating, but its "
            "existing supervisor is lock-only; refusing to measure",
            file=sys.stderr,
        )
        raise SystemExit(REFUSAL_EXIT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--require-quiet", action="store_true")
    verify.set_defaults(action="verify")

    retained = sub.add_parser("verify-retained")
    retained.set_defaults(action="verify-retained")

    run = sub.add_parser("run")
    run.add_argument("--label", required=True)
    run.add_argument("--quiet", action="store_true")
    run.add_argument(
        "--entrypoint",
        action="store_true",
        help="pass lock capabilities to a self-verifying measurement entrypoint",
    )
    run.add_argument("measurement", nargs=argparse.REMAINDER)
    run.set_defaults(action="run")

    args = parser.parse_args(argv)
    if args.action == "verify":
        try:
            verify_supervision()
        except SupervisionError as exc:
            print(f"bench-supervision: {exc}; refusing to measure", file=sys.stderr)
            return REFUSAL_EXIT
        if args.require_quiet and os.environ.get(QUIET_ENV) != "1":
            print(
                "bench-supervision: this entrypoint requires ambient-idle "
                "gating, but its existing supervisor is lock-only; refusing "
                "to measure",
                file=sys.stderr,
            )
            return REFUSAL_EXIT
        return 0

    if args.action == "verify-retained":
        try:
            verify_retained_supervision()
        except SupervisionError as exc:
            print(f"bench-supervision: {exc}; refusing to measure", file=sys.stderr)
            return REFUSAL_EXIT
        return 0

    measurement = args.measurement
    if measurement and measurement[0] == "--":
        measurement = measurement[1:]
    try:
        return run_supervised(
            args.label,
            measurement,
            quiet=args.quiet,
            entrypoint=args.entrypoint,
        )
    except SupervisionError as exc:
        print(f"bench-supervision: {exc}; refusing to measure", file=sys.stderr)
        return REFUSAL_EXIT


if __name__ == "__main__":
    sys.exit(main())

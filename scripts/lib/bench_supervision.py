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
    """The claimed lock receipt does not belong to this process tree."""


def _ancestors() -> set[int]:
    pid = os.getppid()
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            result = subprocess.run(
                ["ps", "-o", "ppid=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SupervisionError(f"could not inspect ancestor {pid}: {exc}") from exc
        if result.returncode != 0:
            raise SupervisionError(
                f"could not inspect ancestor {pid}: ps exited {result.returncode}"
            )
        raw = result.stdout.strip()
        if not raw:
            break
        try:
            pid = int(raw)
        except ValueError as exc:
            raise SupervisionError(f"invalid parent PID reported by ps: {raw!r}") from exc
    return seen


def _lock_paths(lock_lines: list[str]) -> list[Path]:
    paths: list[Path] = []
    for line in lock_lines:
        match = re.search(r"\((/[^)]*)\):", line)
        if match is None:
            raise SupervisionError(f"lock receipt line has no absolute path: {line!r}")
        paths.append(Path(match.group(1)))
    return paths


def _verify_inherited_fds(lock_lines: list[str]) -> tuple[int, ...] | None:
    raw = os.environ.get(FDS_ENV)
    if raw is None:
        return None
    try:
        fds = tuple(int(value) for value in raw.split(","))
    except ValueError as exc:
        raise SupervisionError(f"{FDS_ENV} contains a non-integer descriptor") from exc
    if len(fds) != 2 or len(set(fds)) != 2:
        raise SupervisionError(f"{FDS_ENV} must name exactly two descriptors")

    paths = _lock_paths(lock_lines)
    for fd, path in zip(fds, paths, strict=True):
        try:
            fd_stat = os.fstat(fd)
            path_stat = path.stat()
        except OSError as exc:
            raise SupervisionError(
                f"inherited lock descriptor {fd} cannot be matched to {path}: {exc}"
            ) from exc
        if (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
            raise SupervisionError(
                f"inherited descriptor {fd} does not refer to receipt path {path}"
            )

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

        probe = os.open(path, os.O_RDWR)
        try:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise SupervisionError(
                        f"could not probe inherited lock {path}: {exc}"
                    ) from exc
            else:
                fcntl.flock(probe, fcntl.LOCK_UN)
                raise SupervisionError(
                    f"inherited descriptor {fd} names {path}, but no lock is held"
                )
        finally:
            os.close(probe)
    return fds


def verify_supervision() -> tuple[Path, tuple[int, ...]]:
    """Return the validated status path/capabilities or raise."""

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
        supervisor_pid = int(pid_lines[0].split("=", 1)[1])
    except ValueError as exc:
        raise SupervisionError(
            f"lock receipt {status} has an invalid supervisor PID"
        ) from exc

    lock_lines = [line for line in lines if line.startswith("lock=")]
    if len(lock_lines) != 2:
        raise SupervisionError(
            f"lock receipt {status} must contain both machine-wide locks"
        )
    rendered = "\n".join(lock_lines)
    for required in ("bench-window", "Metal GPU"):
        if required not in rendered:
            raise SupervisionError(
                f"lock receipt {status} does not name the {required} lock"
            )

    inherited = _verify_inherited_fds(lock_lines)
    if inherited is not None:
        return status, inherited

    if supervisor_pid not in _ancestors():
        raise SupervisionError(
            f"lock supervisor {supervisor_pid} is not an ancestor of this run"
        )
    return status, ()


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
    except SupervisionError as exc:
        print(f"bench-supervision: {exc}; refusing to measure", file=sys.stderr)
        raise SystemExit(REFUSAL_EXIT) from exc
    for fd in inherited_fds:
        os.close(fd)
    os.environ.pop(FDS_ENV, None)
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

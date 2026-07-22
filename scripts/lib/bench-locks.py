#!/usr/bin/env python3
"""Hold the machine-wide bench locks for the duration of a command.

    bench-locks.py --label <label> --status-file <path> -- <cmd> [args...]

Acquires BOTH machine-wide advisory locks, runs <cmd> while holding them, and
releases them when <cmd> exits. Exit status is <cmd>'s, except when a lock
cannot be acquired, which exits 75 without running anything.

UNCONDITIONAL, AND THAT IS THE POINT. Neither lock is conditional on detecting
CI, and neither is conditional on classifying the selected bench target as
GPU-driving. A classifier is where the fail-open would live: it would have to
recognize a bench name nobody has seen yet, a feature combination nobody
enumerated, and a transitive dependency that pulls Metal in without announcing
itself in the target name. Every miss passes the check while the GPU spins.
Taking both locks always deletes the classifier, and there is then nothing left
to get wrong. The cost is that a CPU-only bench serializes against GPU work,
which is correct rather than merely tolerable: GPU work running during a CPU
bench IS ambient load, and excluding the largest controllable consumer on the
machine costs nothing on an uncontended box.

The same reasoning covers taking the Metal GPU lock on a platform that has no
Metal. An uncontended flock costs microseconds, and a platform test here would
be one more absence-based conditional in a guard whose whole purpose is to stop
being conditional.

flock(1) IS NOT USED, because it does not exist on macOS (it is util-linux) and
every local run of this script happens there. A shell lock written from Linux
habit ships something that is silently absent on the machine that matters, and
command-not-found inside a backgrounded leg is invisible. This uses fcntl.flock
directly, which is what the fleet bench-window helper already does.

THE LOCK FDS ARE NOT INHERITED by <cmd> or its descendants (subprocess's
close_fds default). A leaked lock fd in a long-lived build daemon holds the
window open machine-wide long after the run that took it, because a flock is
released only when every descriptor referring to that open file description is
closed. Children that must hold a lock get the descriptor passed on purpose;
none do today.

WHY lsof IS ONLY A DIAGNOSTIC HERE. lsof lists processes that have the lock
file OPEN, which is a superset of those holding a flock on it, and it does not
distinguish an exclusive hold from a shared one. That is enough to tell a
waiting operator who to look at, and it is NOT enough to conclude "a parent
already holds this, so I can skip acquiring". Nothing in this file draws that
conclusion; if lsof is unavailable the wait is simply less informative, never
shorter.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time

BENCH_WINDOW = "/tmp/lion-bench-window.lock"
GPU_LOCK = "/tmp/lion-metal-gpu-test.lock"

# Matches the fleet bench-window helper (1800s) and the in-repo GPU test lock
# (30 minutes). A bench that has waited half an hour is not queued behind a
# peer, it is behind something wedged, and the operator needs to be told rather
# than kept waiting.
TIMEOUT_S = 1800

# The fleet pending-marker protocol: a marker named by the acquiring PID is
# dropped BEFORE the blocking acquire starts and removed on exit, so it covers
# both "waiting" and "running". Shared-side consumers poll this directory
# instead of probing the lock, because a non-blocking probe cannot tell an
# exclusive bench apart from an unrelated shared holder.
PENDING_DIR = "/tmp/lion-bench-window-pending"

LOCK_EXIT = 75


def _log(msg: str) -> None:
    sys.stderr.write(f"[bench-locks] {msg}\n")
    sys.stderr.flush()


def ancestors() -> list[int]:
    """This process's ancestor PIDs, nearest first."""
    chain: list[int] = []
    pid = os.getppid()
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        chain.append(pid)
        try:
            out = subprocess.run(
                ["ps", "-o", "ppid=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            pid = int(out) if out else 0
        except (subprocess.SubprocessError, ValueError):
            break
    return chain


def _openers(path: str) -> list[tuple[int, str]]:
    """(pid, command) for processes with `path` open. Diagnostic only."""
    if shutil.which("lsof") is None:
        return []
    try:
        out = subprocess.run(
            ["lsof", "-t", path], capture_output=True, text=True, timeout=10
        ).stdout.split()
    except (subprocess.SubprocessError, OSError):
        return []
    found = []
    for tok in out:
        try:
            pid = int(tok)
        except ValueError:
            continue
        try:
            cmd = subprocess.run(
                ["ps", "-o", "command=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            cmd = "?"
        found.append((pid, cmd[:120] or "?"))
    return found


def _describe_contention(path: str) -> str:
    openers = _openers(path)
    if not openers:
        return f"holder unknown (lsof {path} reported nothing)"
    mine = set(ancestors())
    parts = []
    for pid, cmd in openers:
        tag = " <- AN ANCESTOR OF THIS RUN" if pid in mine else ""
        parts.append(f"pid {pid}{tag}: {cmd}")
    return "; ".join(parts)


def acquire(path: str, name: str) -> tuple[int, str]:
    """Take an exclusive flock on `path`, waiting up to TIMEOUT_S.

    Returns (fd, disposition). Exits LOCK_EXIT rather than returning on
    timeout: a bench that could not be isolated must not run, because a number
    produced under unknown conditions is indistinguishable from one produced
    under good conditions and the reader will assume the good case.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd, "acquired immediately (uncontended)"
    except OSError:
        pass

    # Contended. Say so at once and name who to look at: a script that goes
    # silent for up to half an hour reads as a hang, and an operator who cannot
    # see why cannot decide whether to wait or interrupt.
    _log(f"waiting for the {name} lock at {path}")
    _log(f"  currently open by: {_describe_contention(path)}")
    _log(
        "  if one of those is an ancestor of this run, a caller-side bench-window "
        "wrapper is the likely cause: this script now takes both locks itself and "
        "must not be wrapped."
    )
    started = time.time()
    deadline = started + TIMEOUT_S
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            waited = time.time() - started
            return fd, f"acquired after waiting {waited:.0f}s"
        except OSError:
            if time.time() >= deadline:
                _log(f"FATAL: no exclusive {name} lock within {TIMEOUT_S}s.")
                _log(f"  open by: {_describe_contention(path)}")
                _log(f"  inspect with: lsof {path}")
                sys.exit(LOCK_EXIT)
            time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument(
        "--status-file",
        required=True,
        help="where to record this supervisor's PID and lock dispositions",
    )
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("no command given after --")

    os.makedirs(PENDING_DIR, exist_ok=True)
    marker = os.path.join(PENDING_DIR, str(os.getpid()))
    dispositions: list[str] = []
    fds: list[int] = []
    try:
        with open(marker, "w") as fh:
            fh.write(args.label + "\n")

        for path, name in ((BENCH_WINDOW, "bench-window"), (GPU_LOCK, "Metal GPU")):
            fd, how = acquire(path, name)
            fds.append(fd)
            dispositions.append(f"{name} ({path}): {how}")

        os.makedirs(os.path.dirname(os.path.abspath(args.status_file)), exist_ok=True)
        with open(args.status_file, "w") as fh:
            fh.write(f"supervisor_pid={os.getpid()}\n")
            for line in dispositions:
                fh.write(f"lock={line}\n")

        # close_fds defaults True, so neither lock fd reaches cmd or anything
        # cmd spawns. Both stay held here for cmd's whole lifetime.
        return subprocess.call(cmd)
    finally:
        try:
            os.unlink(marker)
        except OSError:
            pass
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())

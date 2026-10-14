#!/usr/bin/env python3
"""Measurement-lock timestamp regressions, using temporary paths only."""

from __future__ import annotations

import contextlib
import errno
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "bench_locks_refresh", REPO / "scripts" / "lib" / "bench-locks.py"
)
assert SPEC is not None and SPEC.loader is not None
bench_locks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench_locks)


class LockFileRefreshTests(unittest.TestCase):
    def _check_refresh(self, *, contend_once: bool) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurement.lock"
            contents = b"retained lock inode\n"
            path.write_bytes(contents)
            old_ns = time.time_ns() - 4 * 24 * 60 * 60 * 1_000_000_000
            os.utime(path, ns=(old_ns, old_ns))
            before = path.stat()
            self.assertLess(before.st_atime_ns, time.time_ns() - 3 * 86400 * 10**9)
            self.assertLess(before.st_mtime_ns, time.time_ns() - 3 * 86400 * 10**9)
            real_flock = bench_locks.fcntl.flock
            calls = []

            def flock(fd: int, operation: int) -> None:
                calls.append(operation)
                if contend_once and len(calls) == 1:
                    raise BlockingIOError(errno.EWOULDBLOCK, "fixture contention")
                real_flock(fd, operation)

            start_ns = time.time_ns()
            with (
                mock.patch.object(bench_locks.fcntl, "flock", side_effect=flock),
                mock.patch.object(bench_locks, "_describe_contention", return_value="fixture"),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                fd, disposition = bench_locks.acquire(str(path), "temporary measurement")
            try:
                after = os.fstat(fd)
                end_ns = time.time_ns()
                self.assertEqual((after.st_dev, after.st_ino), (before.st_dev, before.st_ino))
                for field in ("st_atime_ns", "st_mtime_ns", "st_ctime_ns"):
                    timestamp = getattr(after, field)
                    self.assertGreaterEqual(timestamp, start_ns - 2 * 10**9, field)
                    self.assertLessEqual(timestamp, end_ns + 2 * 10**9, field)
                self.assertEqual(len(calls), 2 if contend_once else 1)
                self.assertIn("after waiting" if contend_once else "immediately", disposition)
                self.assertEqual(path.read_bytes(), contents)
            finally:
                os.close(fd)

    def test_acquire_refreshes_old_timestamps(self) -> None:
        self._check_refresh(contend_once=False)

    def test_retry_acquire_refreshes_old_timestamps(self) -> None:
        self._check_refresh(contend_once=True)

    def test_refresh_failure_refuses_and_closes_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurement.lock"
            real_open = os.open
            opened = []

            def record_open(*args, **kwargs):
                fd = real_open(*args, **kwargs)
                opened.append(fd)
                return fd

            try:
                with (
                    mock.patch.object(bench_locks.os, "open", side_effect=record_open),
                    mock.patch.object(
                        bench_locks.os, "utime", side_effect=PermissionError("refresh denied")
                    ),
                ):
                    with self.assertRaisesRegex(PermissionError, "refresh denied"):
                        bench_locks.acquire(str(path), "temporary measurement")
                self.assertEqual(len(opened), 1)
                with self.assertRaises(OSError) as raised:
                    os.fstat(opened[0])
                self.assertEqual(raised.exception.errno, errno.EBADF)
            finally:
                for fd in opened:
                    try:
                        os.close(fd)
                    except OSError:
                        pass


if __name__ == "__main__":
    unittest.main()

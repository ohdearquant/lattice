#!/usr/bin/env python3
"""The in-phase load sampler's Linux per-process figure is interval-exact.

The implied foreign load the report judges is `100 - idle - self`. Idle comes
from two /proc/stat reads one second apart; self used to come from `ps` pcpu,
which on Linux is the process's average over its whole life. A benchmark that
spends most of its time on one core and briefly fans a matmul out over every
core therefore read as ~6% self while idle dropped to ~33%, and the ~60%
difference was booked as foreign load: every inference_bench arm on a 16-core
host was refused at the same sample. These tests pin the replacement: on Linux
the self and foreign shares are utime+stime deltas over the same window as the
idle share, divided by the same /proc/stat delta.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LIB = Path(__file__).resolve().parent.parent / "scripts" / "lib"

_spec = importlib.util.spec_from_file_location("phase_load_sampler_interval", LIB / "phase-load-sampler.py")
assert _spec is not None and _spec.loader is not None
sampler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sampler)


def _stat(pid: int, comm: str, ppid: int, utime: int, stime: int, start: int) -> str:
    # Fields 3..23 of proc(5): state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt cmajflt
    # utime stime cutime cstime priority nice num_threads itrealvalue starttime vsize
    rest = ["S", ppid, pid, pid, 0, -1, 4194560, 100, 0, 0, 0, utime, stime, 7, 9, 20, 0, 17, 0, start, 1000]
    return f"{pid} ({comm}) " + " ".join(str(x) for x in rest) + "\n"


# 16 CPUs x 100 ticks: the aggregate advances 1600 ticks over the window, 512 of them idle (32%).
LINE0 = "cpu  50000 0 20000 900000 1000 0 0 0 0 0\n"
LINE1 = "cpu  51000 0 20088 900412 1100 0 0 0 0 0\n"


class ParsePidStat(unittest.TestCase):
    def test_fields_are_read_after_the_last_paren(self):
        data = _stat(4242, "a (b) c", 17, 250, 50, 123456)
        self.assertEqual(sampler.parse_pid_stat(data), (17, 300, 123456, "a (b) c"))


class IntervalRows(unittest.TestCase):
    def test_window_totals_and_idle(self):
        idle, _rows = sampler.interval_rows(LINE0, LINE1, {}, {}, 0)
        self.assertAlmostEqual(idle, 32.0)

    def test_a_burst_inside_the_window_is_charged_to_its_process(self):
        snap0 = {100: (1, 6000, 50, "bench"), 200: (1, 400, 60, "cron")}
        snap1 = {100: (1, 6960, 50, "bench"), 200: (1, 408, 60, "cron")}
        idle, rows = sampler.interval_rows(LINE0, LINE1, snap0, snap1, 900)
        self.assertAlmostEqual(rows[100][1], 60.0)
        self.assertAlmostEqual(rows[200][1], 0.5)
        self.assertAlmostEqual(100.0 - idle - rows[100][1], 8.0)

    def test_a_process_born_in_the_window_is_charged_its_whole_time(self):
        snap1 = {300: (100, 80, 905, "worker")}
        _idle, rows = sampler.interval_rows(LINE0, LINE1, {}, snap1, 900)
        self.assertAlmostEqual(rows[300][1], 5.0)

    def test_a_process_with_no_first_read_that_started_earlier_is_charged_nothing(self):
        snap1 = {301: (1, 90000, 10, "old")}
        _idle, rows = sampler.interval_rows(LINE0, LINE1, {}, snap1, 900)
        self.assertEqual(rows[301][1], 0.0)

    def test_a_reused_pid_is_not_charged_its_predecessors_history(self):
        snap0 = {400: (1, 90000, 10, "gone")}
        snap1 = {400: (1, 16, 950, "new")}
        _idle, rows = sampler.interval_rows(LINE0, LINE1, snap0, snap1, 900)
        self.assertAlmostEqual(rows[400][1], 1.0)

    def test_a_window_where_proc_stat_did_not_advance_is_an_error(self):
        with self.assertRaises(RuntimeError):
            sampler.interval_rows(LINE0, LINE0, {}, {}, 0)


class SampleOnceLinux(unittest.TestCase):
    def test_the_bench_burst_lands_in_self_and_the_implied_figure_stays_low(self):
        rows = {
            100: (1, 54.0, "bench"),
            101: (100, 6.0, "bench-child"),
            200: (1, 0.5, "cron"),
        }
        original = sampler._linux_interval_sample
        sampler._linux_interval_sample = lambda: (32.0, rows)
        try:
            row = sampler.sample_once(100, 16, linux=True)
        finally:
            sampler._linux_interval_sample = original
        self.assertAlmostEqual(row["self_pct"], 60.0)
        self.assertAlmostEqual(row["foreign_pct"], 0.5)
        self.assertEqual(row["top_foreign"], "cron")
        self.assertAlmostEqual(row["top_foreign_pct"], 8.0)
        self.assertEqual(row["cpu_source"], "proc-stat-interval")
        self.assertLess(100.0 - row["idle_pct"] - row["self_pct"], 30.0)


class ReportNamesTheSource(unittest.TestCase):
    def _report(self, rows: list[dict]) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arm.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            done = subprocess.run(
                [sys.executable, str(LIB / "phase-load-report.py"), "--arm", "base1", "--in", str(path)],
                capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout

    @staticmethod
    def _row(**extra) -> dict:
        row = {"idle_pct": 92.0, "foreign_pct": 0.0, "self_pct": 6.0, "top_foreign": "none",
               "top_foreign_pct": 0.0, "schema": "perf-phase-sample/v1", "arm": "base1",
               "captured_utc": "2026-10-26T05:09:17Z"}
        row.update(extra)
        return row

    def test_rows_carrying_a_source_are_reported_by_it(self):
        out = self._report([self._row(cpu_source="proc-stat-interval")])
        self.assertIn("per-process=proc-stat-interval", out)

    def test_rows_from_before_the_field_read_as_ps(self):
        out = self._report([self._row()])
        self.assertIn("per-process=ps-pcpu-decaying-avg", out)


@unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc")
class RealProc(unittest.TestCase):
    def test_this_process_appears_with_a_share_and_its_parent(self):
        import os

        idle, rows = sampler._linux_interval_sample()
        self.assertGreaterEqual(idle, 0.0)
        self.assertLessEqual(idle, 100.0)
        self.assertIn(os.getpid(), rows)
        self.assertEqual(rows[os.getpid()][0], os.getppid())
        self.assertGreaterEqual(rows[os.getpid()][1], 0.0)


if __name__ == "__main__":
    unittest.main()

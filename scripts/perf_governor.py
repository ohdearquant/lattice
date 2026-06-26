#!/usr/bin/env python3
"""
perf_governor.py — Resource guardrail for perf benchmarking on macOS.

Pure stdlib, no pip deps. macOS-only. Uses pmset and ioreg (no sudo).

Six guards:
  1. AC-GATE     : refuse unless on AC power
  2. THERMAL     : refuse/pause+cooldown on CPU_Speed_Limit < 100
  3. BOUNDED     : hard wall-clock cap per measurement (default 90 s)
  4. COOLDOWN    : mandatory idle gap between runs (default 30 s)
  5. KILL-SWITCH : sentinel file .khive/loop/PERF_STOP aborts immediately
  6. AFK-ONLY    : refuse if machine is active (HIDIdleTime < threshold, default 300 s)

Dependency-injection seams on PerfGovernor:
  ._thermal_reader  callable() -> {'speed_limit': int, 'nominal': bool}
  ._ac_reader       callable() -> bool
  ._idle_reader     callable() -> float  (seconds)

Override these in tests / --selftest to trip guards without real hardware stress.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS_FILE.parent       # scripts/ (tracked)
_REPO_ROOT = _SCRIPTS_DIR.parent       # one level up: repo root

# Kill-switch sentinel. DECOUPLED from this module's own location (Leo f5aa3305):
# the emergency-stop path must stay at a stable, repo-rooted location even if
# this script moves. Resolution precedence (applied in PerfGovernor.__init__):
#   --sentinel arg  >  $PERF_GOVERNOR_SENTINEL env  >  this default.
DEFAULT_SENTINEL_FILE = _REPO_ROOT / ".khive" / "loop" / "PERF_STOP"
ENV_SENTINEL_VAR = "PERF_GOVERNOR_SENTINEL"


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[governor {ts}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class GovernorAbort(Exception):
    """Raised when any guard trips. .reason carries a human-readable explanation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Hardware readers (overridable via ._thermal_reader / ._ac_reader / ._idle_reader)
# ---------------------------------------------------------------------------

def _read_thermal() -> dict:
    """
    Parse `pmset -g therm`.

    Nominal (no pressure): all three 'No ... has been recorded' notes, no
    CPU_Speed_Limit line.  speed_limit=100, nominal=True.

    Under pressure: CPU_Speed_Limit = N line present.  speed_limit=N,
    nominal=(N >= 100).  Also catches non-'No ...' recorded-warning lines.
    """
    try:
        r = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True, text=True, timeout=5,
        )
        output = r.stdout
    except Exception as exc:
        # DELIBERATE fail-OPEN: a pmset read error assumes nominal so a flaky
        # thermal probe never blocks a run. Safe because BOUNDED + AFK-ONLY +
        # KILL-SWITCH still bound every run (AC and AFK readers fail CLOSED).
        # Verified + accepted by Leo (f5aa3305).
        _log(f"WARNING: pmset -g therm failed ({exc}); assuming nominal")
        return {"speed_limit": 100, "nominal": True}

    speed_limit: Optional[int] = None
    has_recorded_warning = False

    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith("CPU_Speed_Limit"):
            # e.g. "CPU_Speed_Limit = 80"
            try:
                speed_limit = int(line.split("=", 1)[1].strip())
            except (IndexError, ValueError):
                pass
        elif any(kw in line for kw in ("warning level", "performance warning", "CPU power")):
            # Nominal markers look like "Note: No thermal warning level has been recorded"
            if not (line.startswith("Note:") and "has been recorded" in line):
                has_recorded_warning = True

    if speed_limit is not None:
        return {"speed_limit": speed_limit, "nominal": speed_limit >= 100}
    # No CPU_Speed_Limit line: nominal unless a non-nominal warning line was seen
    return {"speed_limit": 100, "nominal": not has_recorded_warning}


def _read_ac() -> bool:
    """Return True iff on AC power. Parses `pmset -g batt`. Fail-closed (False)."""
    try:
        r = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            if "Now drawing from" in line:
                return "AC Power" in line
    except Exception as exc:
        _log(f"WARNING: pmset -g batt failed ({exc}); assuming battery (fail-closed)")
    return False


def _read_idle_s() -> float:
    """
    Return idle seconds via HIDIdleTime from ioreg.
    Fail-closed: returns 0.0 on error (assume machine is active).
    """
    try:
        r = subprocess.run(
            ["ioreg", "-c", "IOHIDSystem"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            if "HIDIdleTime" in line:
                # '    | | |   "HIDIdleTime" = 499077973041'
                _, _, rhs = line.partition("=")
                return int(rhs.strip()) / 1e9
    except Exception as exc:
        _log(f"WARNING: ioreg HIDIdleTime failed ({exc}); assuming active (0 s)")
    return 0.0


# ---------------------------------------------------------------------------
# PerfGovernor
# ---------------------------------------------------------------------------

class PerfGovernor:
    """
    Resource guardrail for perf benchmarking on macOS.

    All six guards are enforced at preflight and/or during a guarded run.
    Readers are injectable callables so tests can override them without
    touching real hardware.
    """

    def __init__(
        self,
        max_window_s: float = 90.0,
        cooldown_s: float = 30.0,
        afk_only: bool = True,
        afk_threshold_s: float = 300.0,
        max_thermal_cooldowns: int = 3,
        poll_interval_s: float = 5.0,
        sentinel_path: "Optional[Path | str]" = None,
    ) -> None:
        self.max_window_s = max_window_s
        self.cooldown_s = cooldown_s
        self.afk_only = afk_only
        self.afk_threshold_s = afk_threshold_s
        self.max_thermal_cooldowns = max_thermal_cooldowns
        self.poll_interval_s = poll_interval_s

        # Kill-switch sentinel: explicit arg > env > repo-rooted default.
        resolved = sentinel_path or os.environ.get(ENV_SENTINEL_VAR) or DEFAULT_SENTINEL_FILE
        self.sentinel_path = Path(resolved).expanduser()

        # Injectable readers — override in tests
        self._thermal_reader: Callable[[], dict] = _read_thermal
        self._ac_reader: Callable[[], bool] = _read_ac
        self._idle_reader: Callable[[], float] = _read_idle_s

    # ------------------------------------------------------------------
    def _check_kill_switch(self) -> bool:
        return self.sentinel_path.exists()

    # ------------------------------------------------------------------
    def status(self) -> dict:
        """Read current system state. Safe to call any time."""
        thermal = self._thermal_reader()
        on_ac = self._ac_reader()
        idle_s = self._idle_reader()
        kill_sw = self._check_kill_switch()
        afk_ok = (idle_s >= self.afk_threshold_s) if self.afk_only else True
        return {
            "on_ac": on_ac,
            "thermal_speed_limit": thermal["speed_limit"],
            "thermal_nominal": thermal["nominal"],
            "idle_s": round(idle_s, 1),
            "kill_switch": kill_sw,
            "afk_idle_ok": afk_ok,
            "sentinel_path": str(self.sentinel_path),
        }

    # ------------------------------------------------------------------
    def preflight(self) -> None:
        """
        Run all guards as a pre-run gate.
        Raises GovernorAbort if any check fails. Logs each verdict to stderr.
        Returns None if everything is clear.
        """
        _log("=== PREFLIGHT START ===")

        # Guard 5 first — highest-priority abort signal
        if self._check_kill_switch():
            reason = f"KILL-SWITCH: sentinel exists at {self.sentinel_path}"
            _log(f"BLOCK: {reason}")
            raise GovernorAbort(reason)
        _log("PASS: kill-switch clear")

        # Guard 1: AC-GATE
        if not self._ac_reader():
            reason = "AC-GATE: not on AC power (running on battery)"
            _log(f"BLOCK: {reason}")
            raise GovernorAbort(reason)
        _log("PASS: AC power confirmed")

        # Guard 2: THERMAL
        thermal = self._thermal_reader()
        if not thermal["nominal"]:
            reason = f"THERMAL: pressure present (speed_limit={thermal['speed_limit']})"
            _log(f"BLOCK: {reason}")
            raise GovernorAbort(reason)
        _log(f"PASS: thermal nominal (speed_limit={thermal['speed_limit']})")

        # Guard 6: AFK-ONLY
        if self.afk_only:
            idle_s = self._idle_reader()
            if idle_s < self.afk_threshold_s:
                reason = (
                    f"AFK-ONLY: machine active "
                    f"(idle_s={idle_s:.1f} < threshold={self.afk_threshold_s}s); "
                    "step away or use afk_only=False"
                )
                _log(f"BLOCK: {reason}")
                raise GovernorAbort(reason)
            _log(f"PASS: AFK idle_s={idle_s:.1f} >= {self.afk_threshold_s}s")
        else:
            _log("SKIP: afk_only=False")

        _log("=== PREFLIGHT PASS ===")

    # ------------------------------------------------------------------
    def cooldown(self, seconds: Optional[float] = None) -> None:
        """
        Guard 4: mandatory idle gap between runs (default self.cooldown_s).
        Interruptible: checks kill-switch every second and raises GovernorAbort
        if the sentinel appears.
        """
        gap = self.cooldown_s if seconds is None else seconds
        _log(f"COOLDOWN: {gap}s gap starting")
        deadline = time.monotonic() + gap
        while time.monotonic() < deadline:
            if self._check_kill_switch():
                raise GovernorAbort("KILL-SWITCH tripped during cooldown")
            remaining = deadline - time.monotonic()
            time.sleep(min(1.0, max(0.0, remaining)))
        _log("COOLDOWN: complete")

    # ------------------------------------------------------------------
    def _kill_pg(self, proc: subprocess.Popen, reason: str) -> None:
        """Send SIGTERM then SIGKILL to the process group. Logs the reason."""
        _log(f"KILL [{reason}]: pid={proc.pid}")
        try:
            pgid = os.getpgid(proc.pid)
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                return
            time.sleep(0.25)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass

    # ------------------------------------------------------------------
    def run_guarded(self, label: str, argv: List[str]) -> int:
        """
        Guard 3 (BOUNDED) + live poll of guards 2 (THERMAL) and 5 (KILL-SWITCH).

        Spawn argv as a new process group. Enforce max_window_s wall-clock cap.
        On thermal pressure: SIGSTOP child, count cooldown cycles. If pressure
        persists beyond max_thermal_cooldowns cycles: hard abort (SIGKILL).
        If thermal clears: SIGCONT child and reset cycle counter.

        Raises GovernorAbort (kills child first) on any guard trip.
        Returns the child's exit code on clean completion.

        This is preflight-free — call preflight() separately if needed.
        """
        _log(f"RUN_GUARDED [{label}]: {argv}")
        start = time.monotonic()
        abort_msgs: List[str] = []

        proc = subprocess.Popen(argv, start_new_session=True)
        _log(f"RUN_GUARDED [{label}]: pid={proc.pid}")

        thermal_paused = [False]
        thermal_cycle = [0]

        def _poller() -> None:
            while True:
                time.sleep(self.poll_interval_s)
                if proc.poll() is not None:
                    return  # process already done

                elapsed = time.monotonic() - start

                # Guard 5: KILL-SWITCH
                if self._check_kill_switch():
                    msg = "KILL-SWITCH tripped during run"
                    abort_msgs.append(msg)
                    self._kill_pg(proc, msg)
                    return

                # Guard 3: BOUNDED wall-clock cap
                if elapsed >= self.max_window_s:
                    msg = (
                        f"BOUNDED: elapsed {elapsed:.1f}s "
                        f"exceeds max_window_s={self.max_window_s}s"
                    )
                    abort_msgs.append(msg)
                    self._kill_pg(proc, msg)
                    return

                # Guard 2: THERMAL
                thermal = self._thermal_reader()
                if not thermal["nominal"]:
                    thermal_cycle[0] += 1
                    _log(
                        f"THERMAL: speed_limit={thermal['speed_limit']} "
                        f"(cooldown cycle {thermal_cycle[0]}/{self.max_thermal_cooldowns})"
                    )
                    if thermal_cycle[0] > self.max_thermal_cooldowns:
                        msg = (
                            f"THERMAL: pressure persisted beyond "
                            f"{self.max_thermal_cooldowns} cooldown cycles — hard abort"
                        )
                        abort_msgs.append(msg)
                        # SIGCONT first so SIGTERM is receivable, then SIGKILL
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGCONT)
                        except ProcessLookupError:
                            pass
                        self._kill_pg(proc, msg)
                        return
                    # Pause child on first pressure detection
                    if not thermal_paused[0]:
                        try:
                            pgid = os.getpgid(proc.pid)
                            os.killpg(pgid, signal.SIGSTOP)
                            thermal_paused[0] = True
                            _log(
                                f"THERMAL: sent SIGSTOP to pgid={pgid}; "
                                "waiting for thermal to clear"
                            )
                        except ProcessLookupError:
                            return
                else:
                    # Thermal cleared — resume if paused
                    if thermal_paused[0]:
                        thermal_paused[0] = False
                        thermal_cycle[0] = 0
                        try:
                            pgid = os.getpgid(proc.pid)
                            os.killpg(pgid, signal.SIGCONT)
                            _log(f"THERMAL: cleared; sent SIGCONT to pgid={pgid}")
                        except ProcessLookupError:
                            return

        poller = threading.Thread(target=_poller, daemon=True)
        poller.start()

        # Main thread waits; generous timeout — poller enforces the real cap
        try:
            proc.wait(timeout=self.max_window_s * 2 + 5)
        except subprocess.TimeoutExpired:
            msg = f"BOUNDED: last-resort main-thread timeout (>{self.max_window_s * 2 + 5}s)"
            abort_msgs.append(msg)
            self._kill_pg(proc, msg)
            proc.wait()

        poller.join(timeout=2.0)

        if abort_msgs:
            reason = abort_msgs[0]
            _log(f"RUN_GUARDED [{label}]: ABORTED — {reason}")
            raise GovernorAbort(reason)

        elapsed = time.monotonic() - start
        rc = proc.returncode
        _log(f"RUN_GUARDED [{label}]: done rc={rc} elapsed={elapsed:.2f}s")
        return rc

    # ------------------------------------------------------------------
    def guard_window(self, label: str) -> "_GuardWindow":
        """
        Context manager for guarded measurement windows.

        IMPORTANT LIMITATION: This runs a watchdog thread that sets abort_event
        and kills any registered child PID on a guard trip. It cannot preempt
        arbitrary in-process Python code. Cooperative callers should poll
        abort_event. The primary hard-kill path is run_guarded (subprocess).
        """
        return _GuardWindow(self, label)


# ---------------------------------------------------------------------------
# _GuardWindow context manager
# ---------------------------------------------------------------------------

class _GuardWindow:
    """Returned by PerfGovernor.guard_window(). See its docstring."""

    def __init__(self, gov: PerfGovernor, label: str) -> None:
        self.gov = gov
        self.label = label
        self.abort_event = threading.Event()
        self._abort_reason: Optional[str] = None
        self._child_pid: Optional[int] = None
        self._lock = threading.Lock()
        self._start: float = 0.0

    def register_child(self, pid: int) -> None:
        """Register a child PID to be killed when a guard trips."""
        with self._lock:
            self._child_pid = pid

    def _trip(self, reason: str) -> None:
        self._abort_reason = reason
        _log(f"GUARD_WINDOW TRIP [{self.label}]: {reason}")
        with self._lock:
            if self._child_pid is not None:
                try:
                    pgid = os.getpgid(self._child_pid)
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(0.2)
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.abort_event.set()

    def _watchdog(self) -> None:
        g = self.gov
        while not self.abort_event.is_set():
            time.sleep(g.poll_interval_s)
            if self.abort_event.is_set():
                break
            if g._check_kill_switch():
                self._trip(f"KILL-SWITCH in guard_window [{self.label}]")
                return
            elapsed = time.monotonic() - self._start
            if elapsed >= g.max_window_s:
                self._trip(
                    f"BOUNDED: {elapsed:.1f}s > {g.max_window_s}s "
                    f"in guard_window [{self.label}]"
                )
                return
            t = g._thermal_reader()
            if not t["nominal"]:
                self._trip(
                    f"THERMAL: speed_limit={t['speed_limit']} "
                    f"in guard_window [{self.label}]"
                )
                return

    def __enter__(self) -> "_GuardWindow":
        self._start = time.monotonic()
        _log(f"GUARD_WINDOW [{self.label}]: enter (max={self.gov.max_window_s}s)")
        threading.Thread(target=self._watchdog, daemon=True).start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.abort_event.set()
        elapsed = time.monotonic() - self._start
        _log(f"GUARD_WINDOW [{self.label}]: exit elapsed={elapsed:.2f}s")
        if self._abort_reason and exc_type is None:
            raise GovernorAbort(self._abort_reason)
        return False


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _cmd_status(gov: PerfGovernor) -> int:
    s = gov.status()
    print("=== perf_governor status ===")
    print(f"  on_ac             : {s['on_ac']}")
    print(f"  thermal_nominal   : {s['thermal_nominal']}")
    print(f"  thermal_speed_lim : {s['thermal_speed_limit']}")
    print(f"  idle_s            : {s['idle_s']}")
    print(f"  afk_idle_ok       : {s['afk_idle_ok']}")
    print(f"  kill_switch       : {s['kill_switch']}")
    print(f"  sentinel_path     : {s['sentinel_path']}")
    print()
    print(json.dumps(s, indent=2))
    return 0


def _cmd_preflight(gov: PerfGovernor) -> int:
    try:
        gov.preflight()
        print("PREFLIGHT: PASS")
        return 0
    except GovernorAbort as e:
        print(f"PREFLIGHT: BLOCKED — {e.reason}", file=sys.stderr)
        return 2


def _cmd_selftest(gov: PerfGovernor) -> int:
    """
    Demonstrate every guard tripping WITHOUT running a real benchmark.
    Uses dependency injection (overriding _thermal_reader / _ac_reader /
    _idle_reader) to simulate conditions without actual hardware stress.
    Exits 0 only if every sub-demo tripped as designed.
    """
    results: List[tuple] = []

    def demo(name: str, fn) -> None:
        print(f"\n--- {name} ---")
        try:
            fn()
            results.append((name, True))
            print(f"PASS: {name}")
        except Exception as exc:
            results.append((name, False))
            print(f"FAIL: {name}: {exc}")

    # (a) Current real status
    def demo_a() -> None:
        s = gov.status()
        print("  Current real status (live hardware reads):")
        for k, v in s.items():
            print(f"    {k}: {v}")

    demo("(a) real status read", demo_a)

    # (b) Normal run: sleep 1 should complete under a 10 s window
    def demo_b() -> None:
        g = PerfGovernor(max_window_s=10, cooldown_s=0, afk_only=False,
                         poll_interval_s=0.5)
        g._thermal_reader = lambda: {"speed_limit": 100, "nominal": True}
        g._ac_reader = lambda: True
        g._idle_reader = lambda: 999.0
        rc = g.run_guarded("demo_b", ["sleep", "1"])
        assert rc == 0, f"expected rc=0, got {rc}"
        print(f"  sleep 1 completed cleanly (rc={rc}) within 10 s window")

    demo("(b) normal bounded run completes", demo_b)

    # (c) BOUNDED cap: sleep 999 must be killed at ~2 s cap
    def demo_c() -> None:
        g = PerfGovernor(max_window_s=2, cooldown_s=0, afk_only=False,
                         poll_interval_s=0.4)
        g._thermal_reader = lambda: {"speed_limit": 100, "nominal": True}
        g._ac_reader = lambda: True
        g._idle_reader = lambda: 999.0
        try:
            g.run_guarded("demo_c", ["sleep", "999"])
            raise AssertionError("run_guarded returned without raising — expected GovernorAbort")
        except GovernorAbort as e:
            print(f"  GovernorAbort raised as expected: {e.reason}")

    demo("(c) BOUNDED cap kills long process", demo_c)

    # (d) KILL-SWITCH: create sentinel → preflight aborts → remove → passes
    def demo_d() -> None:
        g = PerfGovernor(afk_only=False)
        g._thermal_reader = lambda: {"speed_limit": 100, "nominal": True}
        g._ac_reader = lambda: True
        g._idle_reader = lambda: 999.0

        sentinel = g.sentinel_path
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("stop\n")
        print(f"  Created sentinel: {sentinel}")
        try:
            g.preflight()
            sentinel.unlink(missing_ok=True)
            raise AssertionError("preflight should have raised GovernorAbort")
        except GovernorAbort as e:
            print(f"  GovernorAbort raised as expected: {e.reason}")

        sentinel.unlink(missing_ok=True)
        print(f"  Removed sentinel: {sentinel}")
        g.preflight()  # must pass now
        print("  preflight passed after sentinel removal — kill-switch is two-way")

    demo("(d) KILL-SWITCH sentinel create/remove", demo_d)

    # (e) THERMAL trip via injected fake reader
    def demo_e() -> None:
        always_hot = lambda: {"speed_limit": 70, "nominal": False}

        # Part 1: preflight refuses on thermal pressure
        g1 = PerfGovernor(afk_only=False, poll_interval_s=0.2)
        g1._thermal_reader = always_hot
        g1._ac_reader = lambda: True
        g1._idle_reader = lambda: 999.0
        try:
            g1.preflight()
            raise AssertionError("preflight should have raised GovernorAbort on thermal")
        except GovernorAbort as e:
            print(f"  Thermal preflight block: {e.reason}")

        # Part 2: mid-run thermal abort via poller
        # First call (tick 1): nominal — process starts OK.
        # Subsequent calls: hot — triggers cooldown cycle then hard abort.
        call_count = [0]
        def delayed_hot() -> dict:
            call_count[0] += 1
            return (
                {"speed_limit": 100, "nominal": True}
                if call_count[0] <= 1
                else {"speed_limit": 70, "nominal": False}
            )

        g2 = PerfGovernor(
            max_window_s=30, cooldown_s=0, afk_only=False,
            max_thermal_cooldowns=1, poll_interval_s=0.3,
        )
        g2._thermal_reader = delayed_hot
        g2._ac_reader = lambda: True
        g2._idle_reader = lambda: 999.0
        try:
            g2.run_guarded("demo_e_thermal", ["sleep", "30"])
            raise AssertionError("run_guarded should have raised on mid-run thermal")
        except GovernorAbort as e:
            print(f"  Mid-run thermal abort: {e.reason}")

    demo("(e) THERMAL injection (preflight block + mid-run hard abort)", demo_e)

    # (f) AC-GATE and AFK-ONLY trips via injected fakes
    def demo_f() -> None:
        # AC trip
        g_ac = PerfGovernor(afk_only=False)
        g_ac._thermal_reader = lambda: {"speed_limit": 100, "nominal": True}
        g_ac._ac_reader = lambda: False   # fake battery
        g_ac._idle_reader = lambda: 999.0
        try:
            g_ac.preflight()
            raise AssertionError("preflight should have raised on AC-GATE")
        except GovernorAbort as e:
            print(f"  AC-GATE block: {e.reason}")

        # AFK trip (machine too active: 10 s idle < 300 s threshold)
        g_afk = PerfGovernor(afk_only=True, afk_threshold_s=300)
        g_afk._thermal_reader = lambda: {"speed_limit": 100, "nominal": True}
        g_afk._ac_reader = lambda: True
        g_afk._idle_reader = lambda: 10.0  # fake: Ocean is typing
        try:
            g_afk.preflight()
            raise AssertionError("preflight should have raised on AFK-ONLY")
        except GovernorAbort as e:
            print(f"  AFK-ONLY block: {e.reason}")

    demo("(f) AC-GATE + AFK-ONLY injection", demo_f)

    # Summary
    print("\n" + "=" * 50)
    print("SELFTEST SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, passed in results:
        mark = "PASS" if passed else "FAIL"
        print(f"  {mark}: {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nSELFTEST: all guards tripped as designed")
        return 0
    else:
        print("\nSELFTEST: FAILED (see items above)")
        return 1


def _cmd_run(gov: PerfGovernor, label: str, argv: List[str]) -> int:
    try:
        gov.preflight()
    except GovernorAbort as e:
        print(f"PREFLIGHT BLOCKED: {e.reason}", file=sys.stderr)
        return 2
    try:
        rc = gov.run_guarded(label, argv)
        gov.cooldown()
        return rc
    except GovernorAbort as e:
        print(f"GUARD ABORT: {e.reason}", file=sys.stderr)
        return 2


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    # Split at '--' to capture pass-through command for --run
    if "--" in sys.argv:
        split = sys.argv.index("--")
        our_argv = sys.argv[1:split]
        cmd_argv = sys.argv[split + 1:]
    else:
        our_argv = sys.argv[1:]
        cmd_argv = []

    parser = argparse.ArgumentParser(
        prog="perf_governor",
        description="macOS resource guardrail for perf benchmarking (6 guards)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true",
                      help="Print current system status and exit 0")
    mode.add_argument("--preflight", action="store_true",
                      help="Run preflight gates; exit 0 if clear, 2 if blocked")
    mode.add_argument("--selftest", action="store_true",
                      help="Demonstrate all guards without running a real bench")
    mode.add_argument("--run", action="store_true",
                      help="preflight + run_guarded(cmd) + cooldown; needs -- <cmd>")

    parser.add_argument("--label", default="run",
                        help="Label for --run (default: 'run')")
    parser.add_argument("--max-window", type=float, default=90.0, metavar="S",
                        help="Wall-clock cap in seconds (default: 90)")
    parser.add_argument("--cooldown", type=float, default=30.0, metavar="S",
                        help="Cooldown gap in seconds (default: 30)")
    parser.add_argument("--no-afk", action="store_true",
                        help="Disable AFK-only gate (allow foreground runs)")
    parser.add_argument("--afk-threshold", type=float, default=300.0, metavar="S",
                        help="AFK threshold in seconds (default: 300)")
    parser.add_argument("--max-thermal-cooldowns", type=int, default=3, metavar="N",
                        help="Thermal cooldown cycles before hard abort (default: 3)")
    parser.add_argument("--poll-interval", type=float, default=5.0, metavar="S",
                        help="Poller tick interval in seconds (default: 5)")
    parser.add_argument("--sentinel", default=None, metavar="PATH",
                        help="Kill-switch sentinel file path (default: repo-rooted "
                             ".khive/loop/PERF_STOP; also settable via "
                             f"${ENV_SENTINEL_VAR})")

    args = parser.parse_args(our_argv)

    gov = PerfGovernor(
        max_window_s=args.max_window,
        cooldown_s=args.cooldown,
        afk_only=not args.no_afk,
        afk_threshold_s=args.afk_threshold,
        max_thermal_cooldowns=args.max_thermal_cooldowns,
        poll_interval_s=args.poll_interval,
        sentinel_path=args.sentinel,
    )

    if args.status:
        return _cmd_status(gov)
    if args.preflight:
        return _cmd_preflight(gov)
    if args.selftest:
        return _cmd_selftest(gov)
    if args.run:
        if not cmd_argv:
            print("ERROR: --run requires a command after '--', e.g.: "
                  "perf_governor --run --label foo -- cargo bench",
                  file=sys.stderr)
            return 2
        return _cmd_run(gov, args.label, cmd_argv)

    return 0


if __name__ == "__main__":
    sys.exit(main())

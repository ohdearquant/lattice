#!/usr/bin/env python3
"""
perf_governor.py — Resource guardrail for perf benchmarking on macOS.

Pure stdlib, no pip deps. macOS-only. Uses the shared machine-state probe (no sudo).

Six guards:
  1. AC-GATE     : refuse unless on AC power
  2. THERMAL     : refuse/pause+cooldown on non-nominal macOS thermal state
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
import importlib.util
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
_MACHINE_STATE_PROBE_PATH = _SCRIPTS_DIR / "lib" / "machine-state-probe.py"

# Kill-switch sentinel. DECOUPLED from this module's own location (see commit f5aa3305):
# the emergency-stop path must stay at a stable, repo-rooted location even if
# this script moves. Resolution precedence (applied in PerfGovernor.__init__):
#   --sentinel arg  >  $PERF_GOVERNOR_SENTINEL env  >  this default.
DEFAULT_SENTINEL_FILE = _REPO_ROOT / ".khive" / "loop" / "PERF_STOP"
ENV_SENTINEL_VAR = "PERF_GOVERNOR_SENTINEL"


def _load_machine_state_probe():
    spec = importlib.util.spec_from_file_location(
        "lattice_machine_state_probe",
        _MACHINE_STATE_PROBE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load machine-state probe at {_MACHINE_STATE_PROBE_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MACHINE_STATE_PROBE = _load_machine_state_probe()


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
    """Adapt the shared probe's thermal record to the governor interface."""
    thermal = _MACHINE_STATE_PROBE.read_macos_thermal()
    measured = thermal.get("status") == "measured"
    state = thermal.get("state", "unavailable")
    return {
        "speed_limit": thermal.get("cpu_speed_limit_percent"),
        "nominal": measured and state == "nominal",
        "state": state,
        "source": thermal.get("source", thermal.get("reason", "unavailable")),
    }


def _read_ac() -> bool:
    """Return true only for a measured AC-power record."""
    power = _MACHINE_STATE_PROBE.read_macos_power()
    return power.get("status") == "measured" and power.get("state") == "ac"


def _read_idle_s() -> float:
    """Return measured HID idle seconds, or zero when unavailable."""
    idle = _MACHINE_STATE_PROBE.read_macos_idle()
    if idle.get("status") != "measured":
        return 0.0
    return float(idle["seconds"])


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
            "thermal_state": thermal.get(
                "state", "nominal" if thermal["nominal"] else "pressured"
            ),
            "thermal_source": thermal.get("source", "injected reader"),
            "idle_s": round(idle_s, 1),
            "kill_switch": kill_sw,
            "afk_idle_ok": afk_ok,
            "sentinel_path": str(self.sentinel_path),
        }

    # ------------------------------------------------------------------
    def preflight(self, snapshot: Optional[dict] = None) -> None:
        """
        Run all guards as a pre-run gate.
        Raises GovernorAbort if any check fails. Logs each verdict to stderr.
        Returns None if everything is clear.
        """
        _log("=== PREFLIGHT START ===")

        state = self.status() if snapshot is None else snapshot

        # Guard 5 first — highest-priority abort signal
        if state["kill_switch"]:
            reason = f"KILL-SWITCH: sentinel exists at {state['sentinel_path']}"
            _log(f"BLOCK: {reason}")
            raise GovernorAbort(reason)
        _log("PASS: kill-switch clear")

        # Guard 1: AC-GATE
        if not state["on_ac"]:
            reason = "AC-GATE: not on AC power (running on battery)"
            _log(f"BLOCK: {reason}")
            raise GovernorAbort(reason)
        _log("PASS: AC power confirmed")

        # Guard 2: THERMAL
        if not state["thermal_nominal"]:
            reason = (
                "THERMAL: state is "
                f"{state['thermal_state']} "
                f"(speed_limit={state['thermal_speed_limit']}, "
                f"source={state['thermal_source']})"
            )
            _log(f"BLOCK: {reason}")
            raise GovernorAbort(reason)
        _log(
            "PASS: thermal nominal "
            f"(speed_limit={state['thermal_speed_limit']}, "
            f"source={state['thermal_source']})"
        )

        # Guard 6: AFK-ONLY
        if self.afk_only:
            idle_s = state["idle_s"]
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
        """Terminate the child (SIGTERM then SIGKILL), confirm it is dead.

        The child is spawned with ``start_new_session=True`` so its pgid equals
        its pid; the group signal reaches the whole subtree. But ``killpg`` can
        raise ``PermissionError`` (EPERM) on macOS — e.g. when the group leader
        has exited but the group still holds a zombie, or a GPU-helper member
        runs under a security context the sender cannot signal. A bare
        ``except ProcessLookupError`` lets that EPERM escape and crash the poller
        thread, leaving the SIGKILL undelivered. Since this governor IS the
        bounded-window safety guarantee, the kill path must never depend on the
        group signal landing: on ANY failure, fall back to a direct ``proc.kill``
        (the child is one we definitely own) and reap to CONFIRM death.
        """
        _log(f"KILL [{reason}]: pid={proc.pid}")

        def _group_signal(sig: int) -> None:
            # Best-effort group signal. ProcessLookupError = already gone (fine).
            # PermissionError = EPERM on the group; swallow and let the direct
            # proc.kill() fallback below carry the guarantee.
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except (ProcessLookupError, PermissionError):
                pass

        if proc.poll() is not None:
            return  # already dead — nothing to do

        _group_signal(signal.SIGTERM)
        try:
            # Short grace, not a generous one: this is a bounded-window / thermal
            # safety kill, and a GPU bench mid-command-buffer won't service SIGTERM
            # promptly, so dwelling here only adds load at the cap. Escalate fast.
            proc.wait(timeout=0.3)
            return  # SIGTERM was enough
        except subprocess.TimeoutExpired:
            pass

        _group_signal(signal.SIGKILL)
        # Direct kill of the child we own — independent of the group signal,
        # so an EPERM on killpg cannot leave the process running.
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _log(
                f"KILL [{reason}]: pid={proc.pid} STILL ALIVE after SIGKILL — "
                "manual intervention required"
            )

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
                        # SIGCONT first so SIGTERM is receivable, then SIGKILL.
                        # Swallow EPERM too — _kill_pg's direct fallback carries
                        # the kill even if this group signal is denied.
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGCONT)
                        except (ProcessLookupError, PermissionError):
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
                            return  # child already gone — nothing to throttle
                        except PermissionError:
                            # SIGSTOP denied on a still-live, thermally-hot process.
                            # Abandoning the poller would also drop BOUNDED watch and
                            # leave it running hot, so escalate to a hard abort.
                            msg = "THERMAL: SIGSTOP denied (EPERM) — hard abort"
                            abort_msgs.append(msg)
                            self._kill_pg(proc, msg)
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
                        except PermissionError:
                            # Resume denied; the next poll re-evaluates thermal and
                            # will hard-abort if pressure persists. Don't crash here.
                            pass

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
    print(f"  thermal_state     : {s['thermal_state']}")
    print(f"  thermal_source    : {s['thermal_source']}")
    print(f"  thermal_speed_lim : {s['thermal_speed_limit']}")
    print(f"  idle_s            : {s['idle_s']}")
    print(f"  afk_idle_ok       : {s['afk_idle_ok']}")
    print(f"  kill_switch       : {s['kill_switch']}")
    print(f"  sentinel_path     : {s['sentinel_path']}")
    print()
    print(json.dumps(s, indent=2))
    return 0


def _checkpoint_snapshot(gov: PerfGovernor, record: dict) -> dict:
    """Map one shared-probe record into the governor's fail-closed policy."""
    power = record.get("power", {})
    thermal = record.get("thermal", {})
    idle = record.get("idle", {})

    power_measured = power.get("status") == "measured"
    thermal_measured = thermal.get("status") == "measured"
    idle_measured = idle.get("status") == "measured"
    idle_seconds = float(idle.get("seconds", 0.0)) if idle_measured else 0.0
    return {
        "on_ac": power_measured and power.get("state") == "ac",
        "thermal_speed_limit": thermal.get("cpu_speed_limit_percent"),
        "thermal_nominal": (
            thermal_measured and thermal.get("state") == "nominal"
        ),
        "thermal_state": thermal.get("state", "unavailable"),
        "thermal_source": thermal.get(
            "source", thermal.get("reason", "unavailable")
        ),
        "idle_s": idle_seconds,
        "kill_switch": gov._check_kill_switch(),
        "afk_idle_ok": (
            idle_seconds >= gov.afk_threshold_s if gov.afk_only else True
        ),
        "sentinel_path": str(gov.sentinel_path),
    }


def _print_checkpoint_record(record: dict) -> None:
    print(json.dumps(record, separators=(",", ":"), sort_keys=True))


def _cmd_checkpoint(gov: PerfGovernor, label: str) -> int:
    try:
        gov.cooldown()
    except GovernorAbort as exc:
        print(f"CHECKPOINT BLOCKED: {exc.reason}", file=sys.stderr)
        return 2

    record = _MACHINE_STATE_PROBE.collect_record(label, sys.platform)
    state = _checkpoint_snapshot(gov, record)
    try:
        gov.preflight(state)
        record["gate"] = {
            "status": "passed",
            "cooldown_seconds": gov.cooldown_s,
            "afk_threshold_seconds": (
                gov.afk_threshold_s if gov.afk_only else None
            ),
            "kill_switch": "clear",
        }
        _print_checkpoint_record(record)
        return 0
    except GovernorAbort as exc:
        record["gate"] = {
            "status": "blocked",
            "cooldown_seconds": gov.cooldown_s,
            "afk_threshold_seconds": (
                gov.afk_threshold_s if gov.afk_only else None
            ),
            "reason": exc.reason,
        }
        _print_checkpoint_record(record)
        print(f"CHECKPOINT BLOCKED: {exc.reason}", file=sys.stderr)
        return 2


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

    # (c2) killpg EPERM: when the process-group signal is denied (the real macOS
    # failure observed on a GPU bench at the bounded cap), the child must STILL
    # die via the direct proc.kill() fallback. Regression lock — reverting
    # _kill_pg to a bare `except ProcessLookupError` lets the EPERM escape, the
    # SIGKILL never lands, and `sleep 999` survives this assertion.
    def demo_c2() -> None:
        g = PerfGovernor(max_window_s=2, cooldown_s=0, afk_only=False,
                         poll_interval_s=0.4)
        g._thermal_reader = lambda: {"speed_limit": 100, "nominal": True}
        g._ac_reader = lambda: True
        g._idle_reader = lambda: 999.0

        captured: dict = {}
        real_popen = subprocess.Popen
        orig_killpg = os.killpg

        def capturing_popen(*a, **k):
            p = real_popen(*a, **k)
            captured["proc"] = p
            return p

        def eperm_killpg(_pgid, _sig):
            # Simulate the denied group signal. proc.kill() uses os.kill on the
            # single owned child, not killpg, so the fallback remains effective.
            raise PermissionError(1, "Operation not permitted")

        subprocess.Popen = capturing_popen  # type: ignore[assignment]
        os.killpg = eperm_killpg  # type: ignore[assignment]
        try:
            try:
                g.run_guarded("demo_c2", ["sleep", "999"])
                raise AssertionError("expected GovernorAbort under killpg-EPERM")
            except GovernorAbort as e:
                print(f"  GovernorAbort raised under killpg-EPERM: {e.reason}")
        finally:
            subprocess.Popen = real_popen  # type: ignore[assignment]
            os.killpg = orig_killpg  # type: ignore[assignment]

        proc = captured["proc"]
        for _ in range(30):
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        assert proc.poll() is not None, (
            "child survived killpg-EPERM — direct proc.kill() fallback failed"
        )
        print(f"  child pid={proc.pid} killed via direct fallback (rc={proc.returncode})")

    demo("(c2) killpg-EPERM falls back to direct child kill", demo_c2)

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
        g_afk._idle_reader = lambda: 10.0  # fake: user is actively typing
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
    mode.add_argument(
        "--checkpoint",
        action="store_true",
        help="cooldown, emit one auditable state record, and fail if a guard blocks",
    )
    mode.add_argument("--selftest", action="store_true",
                      help="Demonstrate all guards without running a real bench")
    mode.add_argument("--run", action="store_true",
                      help="preflight + run_guarded(cmd) + cooldown; needs -- <cmd>")

    parser.add_argument(
        "--label",
        default="run",
        help="Label for --run or --checkpoint (default: 'run')",
    )
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
    if args.checkpoint:
        return _cmd_checkpoint(gov, args.label)
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

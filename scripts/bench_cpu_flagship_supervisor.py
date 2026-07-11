#!/usr/bin/env python3
"""bench_cpu_flagship_supervisor.py — macOS supervisor/driver for the CPU
flagship load->prefill->decode smoke harness (benchmark-overhaul program,
PR 2 of the phased implementation plan: "Real CPU flagship smoke",
DESIGN.md section 2 "Measurement boundary").

This is the Python-side half of the adapter whose Rust half is
`crates/inference/src/bin/qwen35_generate.rs --emit-phase-events`. This
module:

  1. Spawns one fresh child process per trial session (that binary), reads
     its `@@bench `-prefixed phase-event JSON lines from stdout as they
     stream, and drives a background `proc_pid_rusage` (`ri_phys_footprint`)
     sampler at a 10ms interval plus an immediate on-demand sample at every
     phase marker (DESIGN.md section 2: "samples the child ... every 10 ms
     and immediately at each phase marker"). This is the one place in the
     benchmark-overhaul program that touches macOS `proc_pid_rusage` FFI --
     kept out of Rust `unsafe` per DESIGN.md's explicit instruction.
  2. Computes the five mandated per-cell metrics from one trial's phase
     events + resource samples: model-load time, prefill tok/s, TTFT,
     decode tok/s (tokens 2..N), per-token latency distribution
     (p50/p95/p99), and peak `phys_footprint` (overall and phase-tagged).
  3. Drives a balanced AB/BA paired session sweep (`n` complete pairs,
     seeded order) and folds the per-pair decode-tok/s paired log-slowdowns
     into `bench_gate_math`'s statistics (order-stratified bootstrap of the
     mean, an informational corrected bound), producing the paired inputs
     DESIGN.md section 4's gate math expects.
  4. Assembles a schema-v2 `CellRecord` (PR 1 / #877's
     `bench_decode_harness.CellRecord`) and runs it through
     `validate_run_record` -- the fail-closed round-trip proof this PR's
     acceptance criteria require. A record whose paired `n` is smaller than
     `bench_gate_math.required_n` for its OWN measured same-session CV is
     downgraded to `verdict="unsupported"` BEFORE validation rather than
     submitted as an underpowered PASS/WARN/FAIL -- `validate_run_record`'s
     low-valid-n check only applies to non-`unsupported` verdicts, so an
     honest "not enough pairs yet" record validates cleanly; it never
     needs the validator to reject it. `--n-pairs` below and the
     module-level SHADOW-MODE note control when this downgrade fires.

SHADOW MODE (DESIGN.md section 5, row 2: "initially shadow"): this PR wires
no CI gate and asserts no PASS/FAIL verdict on real hardware noise -- the
deliverable is a REAL run report with real model/path identities, exact
token counts, all five metrics, and a genuine (not fabricated) paired
comparison, not a promoted gate. A demonstration run with `n` smaller than
the registered smoke `n=7` (DESIGN.md: "seven balanced pairs") on a
contended machine is expected and is labeled DEMONSTRATION, not baseline,
in the emitted report's `"run_kind"` field.

Non-macOS fallback: `proc_pid_rusage` and `NSProcessInfo.thermalState` are
both macOS-only. On any other platform (or if the FFI probe fails for any
reason) this module still runs the full timing measurement but reports
`phys_footprint_bytes: null` / `thermal_state: null` with an explicit
`"informational_only": true` + reason string, per DESIGN.md's fallback
clause -- never a fabricated number.

Run with: uv run scripts/bench_cpu_flagship_supervisor.py --model-dir ...
Unit tests (deterministic, fake child output, no engine/GPU/macOS
required): python3 -m pytest tests/test_bench_cpu_flagship_supervisor.py -v
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib.util
import json
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_sibling(name: str):
    """Loads a sibling `scripts/*.py` module by file path -- `scripts/` is
    not a package, matching the convention `tests/test_bench_run_record.py`
    and `bench_decode_harness.py`'s own `import bench_gate_math` already
    use."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load_sibling("bench_decode_harness")
gate_math = _load_sibling("bench_gate_math")

PHASE_EVENT_NAMES = harness.PHASE_EVENT_NAMES
BENCH_LINE_PREFIX = "@@bench "

# The one canonical CPU flagship cell this supervisor drives, matching
# scripts/bench_expected_cells.toml's "decode_cpu" group's anchor axis
# (f16, contexts 512/1024/2048, required_in hosted_pr_smoke).
CELL_PATH = "decode"
CELL_MODEL_TIER = "qwen3.5-small"
CELL_QUANT_TIER = "f16"
CELL_DEVICE = "cpu"
ANCHOR_CONTEXT_POINTS = (512, 1024, 2048)


# ---------------------------------------------------------------------------
# macOS proc_pid_rusage / ri_phys_footprint sampler
# ---------------------------------------------------------------------------

RUSAGE_INFO_V4 = 4
_UUID_BYTES = 16
# rusage_info_v4 (bsd/sys/resource.h) is a uuid_t followed by ~35
# uint64_t fields. ri_phys_footprint is the 8th uint64_t field
# (0-indexed 7), after ri_uuid, ri_user_time, ri_system_time,
# ri_pkg_idle_wkups, ri_interrupt_wkups, ri_pageins, ri_wired_size,
# ri_resident_size -- offset 72 is corroborated by other open-source
# readers of this struct (e.g. htop's DarwinProcess.c).
#
# The buffer size is intentionally NOT computed as a tight fit from a
# hand-counted field total: `proc_pid_rusage` writes the FULL kernel-side
# `struct rusage_info_v4` into the caller's buffer with no bounds check,
# so an undercount here is silent heap corruption, not a clean error --
# exactly what happened here during testing: a 34-field guess was one
# field short of the real struct, and the kernel's 8-byte overwrite past
# the buffer end corrupted the heap, surfacing later as an unrelated
# segfault inside `ctypes.create_string_buffer`. A generous fixed
# over-allocation (4x any plausible rusage_info_v* size) makes this
# safe regardless of the exact field count.
_PHYS_FOOTPRINT_OFFSET = _UUID_BYTES + 8 * 7
_RUSAGE_V4_STRUCT_SIZE = 1024


class RusageSampler:
    """Thin ctypes wrapper around macOS `proc_pid_rusage(pid,
    RUSAGE_INFO_V4, &buf)`, reading only `ri_phys_footprint` out of the
    returned struct by its known byte offset (kept minimal rather than
    modeling the whole struct with `ctypes.Structure`, since this is the
    only field DESIGN.md's contract requires). Never raises: `sample()`
    returns `None` on any platform/FFI failure, and `.error` carries the
    disclosed reason (DESIGN.md's "non-macOS fallback ... informational
    only" clause).

    `sample()` is called concurrently from `ResourceMonitor`'s background
    10ms-interval thread AND on-demand from the line-reader thread at each
    phase marker (`record_phase`) -- it allocates a FRESH `ctypes` buffer
    on every call rather than reusing one shared instance buffer, since a
    shared buffer written from two threads at once corrupted memory and
    crashed the interpreter with a segfault (caught by this module's own
    test suite: `test_trial_end_to_end_via_shim` before this fix)."""

    def __init__(self) -> None:
        self._libc = None
        self._error: str | None = None
        if platform.system() != "Darwin":
            self._error = "proc_pid_rusage is macOS-only; phys_footprint is informational-only on this platform"
            return
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            if not hasattr(libc, "proc_pid_rusage"):
                raise AttributeError("proc_pid_rusage symbol not found in the process image")
            libc.proc_pid_rusage.restype = ctypes.c_int
            libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            self._libc = libc
        except Exception as exc:  # noqa: BLE001 -- best-effort platform probe, never fatal
            self._error = f"proc_pid_rusage unavailable: {exc!r}"

    @property
    def available(self) -> bool:
        return self._libc is not None

    @property
    def error(self) -> str | None:
        return self._error

    def sample(self, pid: int) -> int | None:
        if self._libc is None:
            return None
        buf = ctypes.create_string_buffer(_RUSAGE_V4_STRUCT_SIZE)
        rc = self._libc.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.cast(buf, ctypes.c_void_p))
        if rc != 0:
            return None
        raw = buf.raw
        return int.from_bytes(raw[_PHYS_FOOTPRINT_OFFSET : _PHYS_FOOTPRINT_OFFSET + 8], "little", signed=False)


# ---------------------------------------------------------------------------
# NSProcessInfo.thermalState (cheap objc-runtime call, no PyObjC dependency)
# ---------------------------------------------------------------------------

THERMAL_STATE_NAMES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def thermal_state() -> tuple[int | None, str | None]:
    """Returns `(NSProcessInfoThermalState value, error)`. `(None,
    reason)` on any non-macOS platform or FFI failure -- DESIGN.md: "keep
    it cheap" (no `powermetrics` subprocess, no PyObjC)."""
    if platform.system() != "Darwin":
        return None, "thermal_state is macOS-only (NSProcessInfo.thermalState)"
    try:
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc_path = ctypes.util.find_library("objc") or "/usr/lib/libobjc.dylib"
        objc = ctypes.CDLL(objc_path)
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        send_ptr = objc.objc_msgSend
        send_ptr.restype = ctypes.c_void_p
        send_ptr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls = objc.objc_getClass(b"NSProcessInfo")
        sel_process_info = objc.sel_registerName(b"processInfo")
        process_info = send_ptr(cls, sel_process_info)
        if not process_info:
            return None, "NSProcessInfo.processInfo returned nil"

        send_long = objc.objc_msgSend
        send_long.restype = ctypes.c_long
        send_long.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        sel_thermal = objc.sel_registerName(b"thermalState")
        value = send_long(process_info, sel_thermal)
        return int(value), None
    except Exception as exc:  # noqa: BLE001 -- best-effort platform probe, never fatal
        return None, f"thermal_state probe failed: {exc!r}"


# ---------------------------------------------------------------------------
# Resource monitor: 10ms background sampling + on-demand phase-tagged samples
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawResourceSample:
    parent_ns: int
    phys_footprint_bytes: int | None
    phase: str | None  # None = ambient 10ms sample, not tied to a phase marker


class ResourceMonitor:
    """Samples `sampler` at `interval_s` in a background thread for the
    lifetime of the child at `pid`, plus accepts an immediate on-demand
    `record_phase()` call from the line-reader thread the moment a phase
    marker is observed (DESIGN.md: "every 10 ms and immediately at each
    phase marker"). Thread-safe: `_samples` is only ever mutated under
    `_lock`."""

    def __init__(self, sampler: RusageSampler, pid: int, interval_s: float = 0.01) -> None:
        self._sampler = sampler
        self._pid = pid
        self._interval_s = interval_s
        self._samples: list[RawResourceSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._record(phase=None)
            self._stop.wait(self._interval_s)

    def record_phase(self, phase: str) -> None:
        self._record(phase=phase)

    def _record(self, phase: str | None) -> None:
        footprint = self._sampler.sample(self._pid)
        parent_ns = time.monotonic_ns()
        with self._lock:
            self._samples.append(RawResourceSample(parent_ns=parent_ns, phys_footprint_bytes=footprint, phase=phase))

    def samples(self) -> list[RawResourceSample]:
        with self._lock:
            return list(self._samples)


# ---------------------------------------------------------------------------
# One trial: spawn the child, stream its phase events, run the monitor
# ---------------------------------------------------------------------------


class TrialFailure(RuntimeError):
    """The child process failed, or produced output that does not satisfy
    the phase-event contract (missing an expected marker, no summary
    line, ...). Always fail closed -- never silently substitute a partial
    trial."""


def run_one_trial(
    binary: Path,
    model_dir: Path,
    context: int,
    max_tokens: int,
    warmup_tokens: int,
    seed: int,
    sampler: RusageSampler,
    timeout_s: float = 300.0,
) -> dict:
    """Spawns exactly one fresh `qwen35_generate --emit-phase-events` child
    (DESIGN.md: "a fresh child process per trial session"), streams its
    `@@bench ` lines as they arrive, and returns the raw
    phase_events/summary/resource_samples for `compute_trial_metrics` to
    aggregate. Raises `TrialFailure` on any non-zero exit or malformed/
    incomplete output -- callers must not treat a partial trial as valid.
    """
    cmd = [
        str(binary),
        "--emit-phase-events",
        "--model-dir",
        str(model_dir),
        "--context",
        str(context),
        "--max-tokens",
        str(max_tokens),
        "--warmup-tokens",
        str(warmup_tokens),
        "--seed",
        str(seed),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    monitor = ResourceMonitor(sampler, proc.pid)
    monitor.start()
    phase_events: list[dict] = []
    summary: dict | None = None
    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            if not line.startswith(BENCH_LINE_PREFIX):
                continue
            parent_ns = time.monotonic_ns()
            try:
                payload = json.loads(line[len(BENCH_LINE_PREFIX) :])
            except json.JSONDecodeError as exc:
                raise TrialFailure(f"malformed @@bench line: {line!r} ({exc})") from exc
            ev = payload.get("ev")
            if ev == "phase":
                name = payload["name"]
                phase_events.append(
                    {
                        "name": name,
                        "child_ns": payload["monotonic_ns"],
                        "token_index": payload.get("token_index"),
                        "parent_ns": parent_ns,
                    }
                )
                if name in PHASE_EVENT_NAMES:
                    monitor.record_phase(name)
            elif ev == "summary":
                summary = payload
        returncode = proc.wait(timeout=timeout_s)
    finally:
        monitor.stop()
        if proc.stdout is not None:
            proc.stdout.close()

    stderr_text = proc.stderr.read() if proc.stderr is not None else ""
    if proc.stderr is not None:
        proc.stderr.close()

    if returncode != 0:
        raise TrialFailure(f"child exited {returncode}: {stderr_text.strip()[-2000:]}")
    if summary is None:
        raise TrialFailure(f"child produced no @@bench summary line (stderr: {stderr_text.strip()[-2000:]})")
    for required_name in ("load_start", "backend_ready", "prefill_start", "prefill_end"):
        if not any(e["name"] == required_name for e in phase_events):
            raise TrialFailure(f"child never emitted required phase event {required_name!r}")
    if not any(e["name"] == "token_available" for e in phase_events):
        raise TrialFailure("child emitted no token_available events")

    return {
        "phase_events": phase_events,
        "summary": summary,
        "resource_samples": monitor.samples(),
        "stderr": stderr_text,
    }


# ---------------------------------------------------------------------------
# Metric aggregation (DESIGN.md section 2's five mandated metrics)
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile (the same convention `numpy.percentile`
    defaults to) over a copy of `values`. `pct` in `[0, 1]`."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * pct
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _first_phase_ns(phase_events: list[dict], name: str) -> int:
    for e in phase_events:
        if e["name"] == name:
            return e["child_ns"]
    raise TrialFailure(f"phase event {name!r} missing from trial")


def compute_trial_metrics(trial: dict) -> dict:
    """Aggregates one trial's raw phase events + resource samples into the
    five mandated metrics (DESIGN.md section 2 "Every result cell
    carries..."): load ms, prefill tok/s, TTFT ms, decode tok/s (tokens
    2..N), per-token latency distribution (raw ms list + p50/p95/p99), and
    peak `phys_footprint` (overall + phase-tagged: load/prefill/decode).

    Phase-tagging resource samples into load/prefill/decode buckets needs a
    common clock: phase-event timestamps are the CHILD's own
    `Instant::now()`-relative nanoseconds, while resource samples are taken
    on the PARENT's `time.monotonic_ns()` clock. Both are OS monotonic
    clocks on the same machine (same underlying clock source), so a single
    constant offset -- anchored at the `load_start` event, whose immediate
    on-demand resource sample was taken at essentially the same instant the
    child printed `monotonic_ns~=0` -- suffices to map every later parent
    timestamp into child-relative time; no continuous re-synchronization is
    needed. This IS a sub-millisecond-scale approximation (Python readline
    + JSON-decode latency between the child's `print!` and the parent's
    `time.monotonic_ns()` call), disclosed here rather than assumed exact
    (DESIGN.md's own standard for `phys_footprint`: "the interval and
    possible sub-interval undercount are part of the method metadata").
    """
    events = trial["phase_events"]
    summary = trial["summary"]

    load_start = _first_phase_ns(events, "load_start")
    backend_ready = _first_phase_ns(events, "backend_ready")
    prefill_start = _first_phase_ns(events, "prefill_start")
    prefill_end = _first_phase_ns(events, "prefill_end")

    token_events = sorted(
        (e for e in events if e["name"] == "token_available"),
        key=lambda e: e["token_index"],
    )
    if not token_events:
        raise TrialFailure("no token_available events observed")

    prompt_tokens = summary["prompt_tokens"]
    generated_tokens = len(token_events)

    load_ms = (backend_ready - load_start) / 1e6

    prefill_seconds = (prefill_end - prefill_start) / 1e9
    if prefill_seconds <= 0:
        raise TrialFailure(f"non-positive prefill duration ({prefill_seconds}s); cannot compute prefill_tok_s")
    prefill_tok_s = prompt_tokens / prefill_seconds

    first_token_ns = token_events[0]["child_ns"]
    ttft_ms = (first_token_ns - prefill_start) / 1e6

    decode_tok_s: float | None = None
    per_token_latencies_ms: list[float] = []
    if generated_tokens >= 2:
        last_token_ns = token_events[-1]["child_ns"]
        decode_seconds = (last_token_ns - first_token_ns) / 1e9
        timed_decode_tokens = generated_tokens - 1  # tokens 2..N (DESIGN.md section 2)
        if decode_seconds > 0:
            decode_tok_s = timed_decode_tokens / decode_seconds
        per_token_latencies_ms = [
            (token_events[i]["child_ns"] - token_events[i - 1]["child_ns"]) / 1e6
            for i in range(1, len(token_events))
        ]

    resource_samples: list[RawResourceSample] = trial["resource_samples"]
    footprints = [s.phys_footprint_bytes for s in resource_samples if s.phys_footprint_bytes is not None]
    peak_overall = max(footprints) if footprints else None

    peak_by_phase: dict[str, int | None] = {"load": None, "prefill": None, "decode": None}
    anchor = next((e for e in events if e["name"] == "load_start"), None)
    if anchor is not None and footprints:
        offset = anchor["parent_ns"] - anchor["child_ns"]
        decode_end_ns = token_events[-1]["child_ns"]
        bounds = {
            "load": (load_start, backend_ready),
            "prefill": (prefill_start, prefill_end),
            "decode": (prefill_end, decode_end_ns),
        }
        for phase_name, (lo, hi) in bounds.items():
            vals = [
                s.phys_footprint_bytes
                for s in resource_samples
                if s.phys_footprint_bytes is not None and lo <= (s.parent_ns - offset) <= hi
            ]
            peak_by_phase[phase_name] = max(vals) if vals else None

    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "requested_max_tokens": summary.get("requested_max_tokens"),
        "load_ms": load_ms,
        "prefill_tok_s": prefill_tok_s,
        "ttft_ms": ttft_ms,
        "decode_tok_s": decode_tok_s,
        "per_token_latencies_ms": per_token_latencies_ms,
        "per_token_p50_ms": _percentile(per_token_latencies_ms, 0.50),
        "per_token_p95_ms": _percentile(per_token_latencies_ms, 0.95),
        "per_token_p99_ms": _percentile(per_token_latencies_ms, 0.99),
        "peak_phys_footprint_bytes": peak_overall,
        "peak_phys_footprint_by_phase": peak_by_phase,
        "delta_matches_generated_tokens": summary.get("delta_matches_generated_tokens"),
        "stopped": summary.get("stopped"),
    }


# ---------------------------------------------------------------------------
# Paired A/B session driver (DESIGN.md section 2's fleet-replayable
# experiment contract: AB/BA balance, seeded order, thermal capture)
# ---------------------------------------------------------------------------


def run_paired_sessions(
    binary: Path,
    model_dir: Path,
    context: int,
    max_tokens: int,
    warmup_tokens: int,
    n_pairs: int,
    base_seed: int,
    sampler: RusageSampler,
) -> dict:
    """Runs `n_pairs` complete AB/BA-balanced session pairs at one context
    point. Both arms invoke the IDENTICAL binary/config in this PR (there is
    no candidate-vs-base code diff to compare in a benchmark-*infrastructure*
    PR) -- this is a same-build A/A self-consistency demonstration proving
    the harness produces a genuine paired comparison whose measured
    log-slowdown should center near zero, not a claim about a code change.
    A future PR wiring this into hosted CI (DESIGN.md section 5, row 3)
    supplies the actual candidate-vs-merge-base arms.

    Pair order alternates AB/BA by pair index (`pair_index % 2`) -- a fixed,
    reproducible balance given `base_seed`, matching DESIGN.md's "AB/BA
    balance distributes process/file-cache drift" requirement without
    depending on an unseeded RNG for the order itself (only per-trial
    `--seed` values are drawn from `base_seed`).

    Thermal state (`NSProcessInfo.thermalState`) is captured before and
    after each pair and recorded on every pair's entry, but this PR does
    NOT implement the registered thermal-envelope invalidation/replacement
    protocol (DESIGN.md section 3's "Availability, security, and variance
    policy") -- that envelope is defined alongside the protected M2 Max
    lane (PR 4) this PR does not provision. Captured values are disclosed
    in the report for visibility; no pair is silently dropped or replaced
    here.
    """
    arm_a: list[dict] = []
    arm_b: list[dict] = []
    pairs: list[dict] = []
    order_ab = 0
    order_ba = 0

    for pair_index in range(n_pairs):
        ab_first = pair_index % 2 == 0
        seed_a = base_seed + 2 * pair_index
        seed_b = base_seed + 2 * pair_index + 1

        thermal_before, thermal_before_err = thermal_state()

        def _run_a() -> dict:
            return compute_trial_metrics(
                run_one_trial(binary, model_dir, context, max_tokens, warmup_tokens, seed_a, sampler)
            )

        def _run_b() -> dict:
            return compute_trial_metrics(
                run_one_trial(binary, model_dir, context, max_tokens, warmup_tokens, seed_b, sampler)
            )

        if ab_first:
            m_a, m_b = _run_a(), _run_b()
            order_ab += 1
            order = "AB"
        else:
            m_b, m_a = _run_b(), _run_a()
            order_ba += 1
            order = "BA"

        thermal_after, thermal_after_err = thermal_state()

        arm_a.append(m_a)
        arm_b.append(m_b)
        pairs.append(
            {
                "pair_index": pair_index,
                "order": order,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "arm_a": m_a,
                "arm_b": m_b,
                "thermal_state_before": thermal_before,
                "thermal_state_before_name": THERMAL_STATE_NAMES.get(thermal_before) if thermal_before is not None else None,
                "thermal_state_after": thermal_after,
                "thermal_state_after_name": THERMAL_STATE_NAMES.get(thermal_after) if thermal_after is not None else None,
                "thermal_error": thermal_before_err or thermal_after_err,
            }
        )

    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "pairs": pairs,
        "order_balance": (order_ab, order_ba),
    }


# ---------------------------------------------------------------------------
# Gate-math folding: paired log-slowdowns -> CellAggregate
# ---------------------------------------------------------------------------


def build_decode_cell_aggregate(session: dict, policy: dict, rng_seed: int) -> tuple[harness.CellAggregate, dict]:
    """Folds one paired session's `decode_tok_s` values into a
    `bench_gate_math`-backed `CellAggregate` (correction 1's `measured_cv`/
    `required_n`, correction 2's bootstrap-of-the-mean). Returns the
    aggregate plus a diagnostics dict (measured_cv, required_n, cell_class,
    n_pairs) for the report.

    `measured_cv` is the same-session residual coefficient of variation of
    arm A's `decode_tok_s` values (`stdev / mean`) -- DESIGN.md's "measured
    same-session residual CV" the first null A/A calibration sessions are
    meant to establish; this module computes it directly from the arms it
    already ran rather than requiring a separate calibration pass, since
    this PR's demonstration IS an A/A session (see `run_paired_sessions`).
    """
    arm_a = session["arm_a"]
    arm_b = session["arm_b"]
    order_ab, order_ba = session["order_balance"]
    n_pairs = len(arm_a)
    if n_pairs == 0:
        raise ValueError("build_decode_cell_aggregate requires at least one pair")

    a_values = [m["decode_tok_s"] for m in arm_a]
    b_values = [m["decode_tok_s"] for m in arm_b]
    if any(v is None for v in a_values) or any(v is None for v in b_values):
        raise ValueError("every trial must report a decode_tok_s (generated_tokens >= 2) to fold into the gate")

    log_slowdowns = [
        gate_math.log_slowdown(base=a, candidate=b, higher_is_better=True) for a, b in zip(a_values, b_values, strict=True)
    ]
    order_ab_flags = [p["order"] == "AB" for p in session["pairs"]]

    point_estimate = statistics.fmean(log_slowdowns)
    if len(a_values) >= 2:
        measured_cv = statistics.stdev(a_values) / statistics.fmean(a_values)
    else:
        measured_cv = None

    cv_bands = gate_math.parse_cv_bands(policy["cv_bands"])
    cell_class = "A"  # decode family, per perf-policy.toml's [families.decode] noise_class
    required_n: int | None = None
    if measured_cv is not None:
        required_n, _fail_margin = gate_math.required_n(measured_cv, cv_bands, cell_class)

    rng = random.Random(rng_seed)
    b = policy["gate_math"]["bootstrap_replicates"]
    if n_pairs >= 2:
        boot_means = gate_math.order_stratified_bootstrap_means(log_slowdowns, order_ab_flags, b, rng)
        ci_low = _percentile(boot_means, 0.025) or point_estimate
        ci_high = _percentile(boot_means, 0.975) or point_estimate
        alpha_familywise = policy["gate_math"]["alpha_familywise"]
        corrected_lower_bound = gate_math.bootstrap_upper_bound(
            log_slowdowns, order_ab_flags, b, random.Random(rng_seed + 1), corrected_alpha=alpha_familywise
        )
    else:
        # A single pair has no bootstrap distribution worth reporting;
        # collapse CI to the point estimate rather than fabricating spread.
        ci_low = ci_high = point_estimate
        corrected_lower_bound = None

    aggregate = harness.parse_cell_aggregate(
        {
            "point_estimate": point_estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "corrected_lower_bound": corrected_lower_bound,
            "n_valid": n_pairs,
            "n_invalid": 0,
            "measured_cv": measured_cv,
            "required_n": required_n,
        }
    )
    diagnostics = {
        "log_slowdowns": log_slowdowns,
        "measured_cv": measured_cv,
        "required_n": required_n,
        "cell_class": cell_class,
        "n_pairs": n_pairs,
        "order_balance": [order_ab, order_ba],
    }
    return aggregate, diagnostics


# ---------------------------------------------------------------------------
# Provenance / identity helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()


def git_sha() -> str:
    return _run(["git", "rev-parse", "HEAD"])


def git_dirty() -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(status.stdout.strip())


def hardware_fingerprint() -> str:
    brand = ""
    if platform.system() == "Darwin":
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:  # noqa: BLE001 -- best-effort identity string
            brand = ""
    parts = [platform.system(), platform.machine(), brand or platform.processor()]
    return "-".join(p.replace(" ", "_") for p in parts if p)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def resolve_weight_files(model_dir: Path) -> list[Path]:
    candidates = sorted(model_dir.glob("model.safetensors*"))
    return [p.resolve() for p in candidates if p.is_file() or p.resolve().is_file()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_cpu_flagship_supervisor.py",
        description="CPU flagship load->prefill->decode paired smoke driver (benchmark-overhaul PR 2).",
    )
    p.add_argument("--binary", type=Path, default=REPO_ROOT / "target" / "release" / "qwen35_generate")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--context", type=int, default=512, choices=list(ANCHOR_CONTEXT_POINTS))
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--warmup-tokens", type=int, default=16)
    p.add_argument("--n-pairs", type=int, default=7, help="registered smoke n=7 (DESIGN.md); pass a smaller n for a disclosed DEMONSTRATION run on a contended machine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None, help="write the full report JSON here")
    p.add_argument(
        "--run-kind",
        choices=["baseline", "demonstration"],
        default="baseline",
        help="label the report's run_kind; use 'demonstration' when n-pairs is below the registered smoke n=7",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.binary.exists():
        print(f"FAIL: binary not found at {args.binary} (build it first: cargo build --release --bin qwen35_generate -p lattice-inference --features f16)", file=sys.stderr)
        return 1
    if not args.model_dir.exists():
        print(f"FAIL: --model-dir {args.model_dir} does not exist", file=sys.stderr)
        return 1

    sampler = RusageSampler()
    if not sampler.available:
        print(f"NOTE: {sampler.error} -- phys_footprint will be reported as null (informational_only)", file=sys.stderr)

    policy = gate_math.load_policy()
    policy_sha = gate_math.policy_sha()

    session = run_paired_sessions(
        binary=args.binary,
        model_dir=args.model_dir,
        context=args.context,
        max_tokens=args.max_tokens,
        warmup_tokens=args.warmup_tokens,
        n_pairs=args.n_pairs,
        base_seed=args.seed,
        sampler=sampler,
    )

    aggregate, diagnostics = build_decode_cell_aggregate(session, policy, rng_seed=args.seed)

    weight_files = resolve_weight_files(args.model_dir)
    weight_hashes = {p.name: sha256_file(p) for p in weight_files}

    cid = harness.cell_id(CELL_PATH, CELL_MODEL_TIER, CELL_QUANT_TIER, CELL_DEVICE, args.context)
    repo_sha = git_sha()
    provenance = harness.parse_provenance(
        {
            "repo_sha": repo_sha,
            "candidate_sha": repo_sha,
            "base_sha": None,
            "dirty": git_dirty(),
            "profile_name": "cpu_flagship_smoke_v1",
            "profile_version": 1,
            "profile_sha": hashlib.sha256(
                json.dumps(
                    {
                        "max_tokens": args.max_tokens,
                        "warmup_tokens": args.warmup_tokens,
                        "temperature": 0.0,
                        "top_k": 1,
                        "top_p": 1.0,
                        "repetition_penalty": 1.0,
                        "disable_eos": True,
                        "enable_thinking": False,
                        "enable_mtp": False,
                        "grammar": None,
                        "stop_strings": [],
                        "reasoning_budget": None,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "policy_version": policy["policy_version"],
            "policy_sha": policy_sha,
            "script_sha": sha256_file(Path(__file__)),
            "hardware_fingerprint": hardware_fingerprint(),
            "collected_at": datetime.now(UTC).isoformat(),
            "workflow_run_id": None,
        }
    )

    n_valid = diagnostics["n_pairs"]
    required_n = diagnostics["required_n"]
    measured_cv = diagnostics["measured_cv"]
    fail_pct = policy["families"]["decode"]["fail_pct"]
    warn_pct = policy["families"]["decode"]["warn_pct"]
    gate_eligible = measured_cv is not None and required_n is not None and n_valid >= required_n
    if not gate_eligible:
        verdict = "unsupported"
        unsupported_reason = (
            f"n_valid={n_valid} < required_n={required_n} for measured_cv={measured_cv} "
            "(bench_gate_math.required_n, correction 1) -- shadow/demonstration run, not a gated cell"
        )
    else:
        cb = aggregate.corrected_lower_bound
        if cb is not None and cb > fail_pct:
            verdict = "FAIL"
        elif aggregate.point_estimate > warn_pct:
            verdict = "WARN"
        else:
            verdict = "PASS"
        unsupported_reason = None

    record = harness.CellRecord(
        cell_id=cid,
        metric_family="decode",
        metric_name="decode_tok_s",
        path=CELL_PATH,
        model_tier=CELL_MODEL_TIER,
        quant_tier=CELL_QUANT_TIER,
        device=CELL_DEVICE,
        context_point=args.context,
        aggregate=aggregate,
        verdict=verdict,
        unsupported_reason=unsupported_reason,
        path_proof=("cpu:qwen35_generate::generate_streaming_with_cancel",),
        lock_receipts=(),
        phase_events=(),
        resource_samples=(),
        provenance=provenance,
        order_balance=tuple(diagnostics["order_balance"]),
        raw_artifact_digest=None,
    )

    # NOTE: an "unsupported" verdict (underpowered n) is deliberately
    # constructed to validate cleanly (see build_decode_cell_aggregate /
    # verdict assignment above) -- validate_run_record's low-valid-n check
    # only fires for PASS/WARN/FAIL. A RunRecordValidationError here means
    # a genuine defect (SHA mismatch, non-finite metric, ...), not an
    # expected outcome of a small demonstration n.
    validator_error: str | None = None
    try:
        harness.validate_run_record(record, expected_repo_sha=repo_sha, current_policy_sha=policy_sha)
        validator_verdict = "PASS (validate_run_record raised nothing)"
    except harness.RunRecordValidationError as exc:
        validator_error = str(exc)
        validator_verdict = f"FAIL (validate_run_record fail-closed, unexpected): {exc}"

    report = {
        "run_kind": args.run_kind,
        "cell_id": cid,
        "context": args.context,
        "n_pairs_requested": args.n_pairs,
        "model": {
            "model_dir": str(args.model_dir.resolve()),
            "weight_files": {name: {"sha256": digest} for name, digest in weight_hashes.items()},
        },
        "command": " ".join(
            [
                str(args.binary),
                "--emit-phase-events",
                "--model-dir",
                str(args.model_dir),
                "--context",
                str(args.context),
                "--max-tokens",
                str(args.max_tokens),
                "--warmup-tokens",
                str(args.warmup_tokens),
                "--seed",
                "<per-trial, derived from --seed base>",
            ]
        ),
        "resource_sampler": {"available": sampler.available, "error": sampler.error},
        "session": {
            "pairs": session["pairs"],
            "order_balance": session["order_balance"],
        },
        "aggregate": {
            "point_estimate_log_slowdown": aggregate.point_estimate,
            "ci_low": aggregate.ci_low,
            "ci_high": aggregate.ci_high,
            "corrected_lower_bound": aggregate.corrected_lower_bound,
            "n_valid": aggregate.n_valid,
            "n_invalid": aggregate.n_invalid,
            "measured_cv": aggregate.measured_cv,
            "required_n": aggregate.required_n,
        },
        "diagnostics": diagnostics,
        "verdict": verdict,
        "unsupported_reason": unsupported_reason,
        "validate_run_record": validator_verdict,
        "provenance": {
            "repo_sha": provenance.repo_sha,
            "dirty": provenance.dirty,
            "policy_version": provenance.policy_version,
            "policy_sha": provenance.policy_sha,
            "hardware_fingerprint": provenance.hardware_fingerprint,
            "collected_at": provenance.collected_at,
        },
    }

    text = json.dumps(report, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.write_text(text)
        print(f"Report written to {args.out}")
    print(text)

    return 0 if validator_error is None else 1


if __name__ == "__main__":
    sys.exit(main())

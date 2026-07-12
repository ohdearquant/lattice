"""Unit tests for scripts/bench_cpu_flagship_supervisor.py (bench-overhaul
PR 2, "Real CPU flagship smoke").

Deterministic fixtures only -- no real `qwen35_generate` binary, GPU, or
heavy lane required. A tiny stand-in "child" (a short Python script printing
the same `@@bench ` phase-event / summary line contract the real Rust
binary's `--emit-phase-events` mode emits) drives `run_one_trial` end to
end, proving the parsing/aggregation logic against fake-but-contract-shaped
output. Loaded by file path, matching test_bench_run_record.py's
convention: scripts/ is not a package.

Covers this PR's acceptance row F: unit tests for the supervisor's parsing/
aggregation with fake child output (deterministic), and a validator
round-trip proof that records this module assembles pass PR #877's
`validate_run_record`.

Run with: python3 -m pytest tests/test_bench_cpu_flagship_supervisor.py -v
"""

from __future__ import annotations

import importlib.util
import math
import os
import statistics
import sys
import textwrap
import threading
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load("bench_decode_harness")
gate_math = _load("bench_gate_math")
supervisor = _load("bench_cpu_flagship_supervisor")


# ---------------------------------------------------------------------------
# A fake child: a short Python script that speaks the exact `@@bench `
# contract `qwen35_generate --emit-phase-events` emits, with fully
# deterministic (caller-supplied) phase-event timestamps and token count.
# `run_one_trial` spawns real subprocesses regardless of what's inside them,
# so driving it against this fake child is a real end-to-end exercise of
# the parsing/streaming/resource-monitor code, without needing the engine.
# ---------------------------------------------------------------------------

_FAKE_CHILD_TEMPLATE = textwrap.dedent(
    """\
    import sys, time

    def emit(line):
        print(line, flush=True)

    t0 = time.monotonic_ns()

    def ns(offset_ms):
        # Deterministic, monotonically increasing relative to t0 -- mirrors
        # the real binary's `Instant::now()`-relative `monotonic_ns` field.
        return t0 + int(offset_ms * 1_000_000)

    emit('@@bench {{"ev":"phase","name":"load_start","monotonic_ns":%d}}' % ns(0))
    emit('@@bench {{"ev":"phase","name":"backend_ready","monotonic_ns":%d}}' % ns({load_ms}))
    emit('@@bench {{"ev":"phase","name":"prefill_start","monotonic_ns":%d}}' % ns({load_ms}))
    emit('@@bench {{"ev":"phase","name":"prefill_end","monotonic_ns":%d}}' % ns({prefill_end_ms}))
    {token_lines}
    emit('@@bench {{"ev":"summary","prompt_tokens":{prompt_tokens},"generated_tokens":{n_tokens},"requested_max_tokens":{n_tokens},"delta_call_count":{n_tokens},"delta_matches_generated_tokens":true,"stopped":"length","model_dir":"/fake/model","seed":42,"disable_eos":true}}')
    sys.exit(0)
    """
)


def write_fake_child(
    path: Path,
    *,
    prompt_tokens: int = 512,
    load_ms: float = 50.0,
    prefill_end_ms: float = 60.0,
    token_gap_ms: float = 5.0,
    n_tokens: int = 8,
) -> Path:
    """Writes a deterministic fake child script to `path` and returns it.
    Token N is emitted at `prefill_end_ms + N * token_gap_ms` (child-relative
    ns), so decode tok/s and per-token latencies are exactly
    `1000 / token_gap_ms` and `token_gap_ms` respectively -- values a test
    can assert on exactly, not just "roughly"."""
    # NOTE: _FAKE_CHILD_TEMPLATE is textwrap.dedent()-ed before this join
    # runs, which strips the template's common 4-space indent (so the
    # {token_lines} placeholder line sits at column 0) -- the join
    # separator must be a bare "\n", not "\n    ", or every line after the
    # first ends up with a mismatched leading indent and the generated
    # child script fails to parse.
    # token_index is 1-based (matching the real binary's contract and
    # `bench_decode_harness.parse_phase_event`'s `token_index >= 1`
    # requirement -- `_validate_trial_trace` now runs `parse_phase_event`
    # over every event `run_one_trial` returns, including this fixture's).
    token_lines = "\n".join(
        f'emit(\'@@bench {{"ev":"phase","name":"token_available","monotonic_ns":%d,"token_index":{i + 1}}}\' % ns({prefill_end_ms + (i + 1) * token_gap_ms}))'
        for i in range(n_tokens)
    )
    text = _FAKE_CHILD_TEMPLATE.format(
        load_ms=load_ms,
        prefill_end_ms=prefill_end_ms,
        token_lines=token_lines,
        prompt_tokens=prompt_tokens,
        n_tokens=n_tokens,
    )
    path.write_text(text)
    return path


class FakeChildTrialTest(unittest.TestCase):
    """Drives `run_one_trial` against a real subprocess (the fake child),
    proving the streaming @@bench parser, phase-event bookkeeping, and
    resource monitor lifecycle work end to end without the real engine."""

    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.child_path = Path(self._tmpdir.name) / "fake_child.py"
        write_fake_child(self.child_path)

    def test_trial_end_to_end_via_shim(self):
        # run_one_trial spawns a single `binary` path with flags appended;
        # the fake child ignores all CLI args (it is fully parameterized by
        # the script text baked in by write_fake_child), so a tiny shell
        # shim that execs `python3 fake_child.py "$@"` lets run_one_trial's
        # real subprocess-spawn/argv-building code run unmodified against
        # deterministic fake output.
        shim = Path(self._tmpdir.name) / "shim.sh"
        shim.write_text(f"#!/bin/sh\nexec {sys.executable} {self.child_path} \"$@\"\n")
        shim.chmod(0o755)

        sampler = supervisor.RusageSampler()
        trial = supervisor.run_one_trial(
            binary=shim,
            model_dir=Path("/fake/model"),
            context=512,
            max_tokens=8,
            warmup_tokens=16,
            seed=42,
            sampler=sampler,
            timeout_s=30.0,
        )
        self.assertEqual(trial["summary"]["prompt_tokens"], 512)
        self.assertEqual(trial["summary"]["generated_tokens"], 8)
        names = [e["name"] for e in trial["phase_events"]]
        for required in ("load_start", "backend_ready", "prefill_start", "prefill_end"):
            self.assertIn(required, names)
        self.assertEqual(sum(1 for n in names if n == "token_available"), 8)
        # The resource monitor should have collected at least the four
        # on-demand phase samples, even if the 10ms background thread
        # barely ticks during this short-lived fake child.
        self.assertGreaterEqual(len(trial["resource_samples"]), 4)

        metrics = supervisor.compute_trial_metrics(trial)
        self.assertEqual(metrics["prompt_tokens"], 512)
        self.assertEqual(metrics["generated_tokens"], 8)
        self.assertAlmostEqual(metrics["load_ms"], 50.0, delta=1.0)
        # prefill_start == load_ms (50ms); first token_available fires at
        # prefill_end_ms + 1 * token_gap_ms = 60 + 5 = 65ms -> TTFT = 15ms.
        self.assertAlmostEqual(metrics["ttft_ms"], 15.0, delta=1.0)
        # decode tok/s over tokens 2..8 (7 timed tokens) at a fixed 5ms gap
        # between consecutive token_available events -> 1000/5 = 200 tok/s.
        self.assertAlmostEqual(metrics["decode_tok_s"], 200.0, delta=5.0)
        self.assertEqual(len(metrics["per_token_latencies_ms"]), 7)
        for lat in metrics["per_token_latencies_ms"]:
            self.assertAlmostEqual(lat, 5.0, delta=1.0)
        self.assertAlmostEqual(metrics["per_token_p50_ms"], 5.0, delta=1.0)
        self.assertTrue(metrics["delta_matches_generated_tokens"])

    def test_missing_required_phase_event_fails_closed(self):
        broken = Path(self._tmpdir.name) / "broken_child.py"
        broken.write_text(
            textwrap.dedent(
                """\
                print('@@bench {"ev":"phase","name":"load_start","monotonic_ns":0}', flush=True)
                print('@@bench {"ev":"summary","prompt_tokens":1,"generated_tokens":0,"requested_max_tokens":1,"delta_call_count":0,"delta_matches_generated_tokens":true,"stopped":"length","model_dir":"/fake","seed":1,"disable_eos":true}', flush=True)
                """
            )
        )
        shim = Path(self._tmpdir.name) / "broken_shim.sh"
        shim.write_text(f"#!/bin/sh\nexec {sys.executable} {broken} \"$@\"\n")
        shim.chmod(0o755)

        sampler = supervisor.RusageSampler()
        with self.assertRaises(supervisor.TrialFailure):
            supervisor.run_one_trial(
                binary=shim,
                model_dir=Path("/fake/model"),
                context=1,
                max_tokens=1,
                warmup_tokens=0,
                seed=1,
                sampler=sampler,
                timeout_s=10.0,
            )

    def test_nonzero_exit_fails_closed(self):
        broken = Path(self._tmpdir.name) / "crash_child.py"
        broken.write_text("import sys\nsys.exit(3)\n")
        shim = Path(self._tmpdir.name) / "crash_shim.sh"
        shim.write_text(f"#!/bin/sh\nexec {sys.executable} {broken} \"$@\"\n")
        shim.chmod(0o755)

        sampler = supervisor.RusageSampler()
        with self.assertRaises(supervisor.TrialFailure):
            supervisor.run_one_trial(
                binary=shim,
                model_dir=Path("/fake/model"),
                context=1,
                max_tokens=1,
                warmup_tokens=0,
                seed=1,
                sampler=sampler,
                timeout_s=10.0,
            )


def _run_with_bound(fn, bound_s: float):
    """Runs `fn()` on a background daemon thread and joins it with a hard
    wall-clock bound. A genuine regression of the timeout/reap fix (e.g.
    dropping `proc.kill()` on the timeout path) does not merely make
    `run_one_trial` slow -- it deadlocks it forever (the reader thread's
    blocking `read()` holds the stdlib io object's internal lock, so the
    main thread's later `proc.stdout.close()` blocks waiting for that same
    lock, and the lock never releases because the un-killed child never
    sends EOF). Calling `run_one_trial` directly from the test body would
    therefore hang the ENTIRE test process on a regression, defeating the
    purpose of a test that is supposed to catch it. Running it on a joined,
    bounded, daemon thread instead means a regression shows up as a clean,
    prompt test FAILURE ("did not complete within Ns") instead of a wedged
    CI job -- the outer test process itself is never at risk of hanging.
    Returns `(completed: bool, result, exc)`.
    """
    result_holder: list = []
    exc_holder: list = []

    def _target():
        try:
            result_holder.append(fn())
        except BaseException as exc:  # noqa: BLE001 -- captured for the joining thread to inspect
            exc_holder.append(exc)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=bound_s)
    completed = not t.is_alive()
    result = result_holder[0] if result_holder else None
    exc = exc_holder[0] if exc_holder else None
    return completed, result, exc


class RunOneTrialTimeoutAndReapTest(unittest.TestCase):
    """Covers codex round-1 major: `timeout_s` must actually bound a child
    that hangs with its stdout pipe still open, and the killed child must
    be reaped (never left a zombie/leaked process), and a child that fills
    its stderr pipe while holding stdout open must not deadlock the
    parent (fixed by concurrent background-thread draining of both pipes
    instead of a single blocking `for raw_line in proc.stdout` loop)."""

    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def test_hanging_child_times_out_and_is_reaped(self):
        # A silent child: no @@bench output at all, sleeps far longer than
        # the trial's timeout, with stdout left open the whole time -- the
        # exact shape that could never reach the pre-fix code's
        # `proc.wait(timeout=...)` call because the blocking stdout-read
        # loop ran first and never returned. Sleeps long enough that if
        # the fix regresses and the child is never killed, this test's own
        # bounded join (not the child's sleep) is what ends the test.
        pid_file = Path(self._tmpdir.name) / "child.pid"
        hang = Path(self._tmpdir.name) / "hang_child.py"
        # `exec`-based shims (like every other fake child in this file)
        # replace the shell process image in place, so the shim's own pid
        # IS the actual running child's pid -- the hang child records its
        # own pid via os.getpid() before sleeping, no extra process-tree
        # bookkeeping needed.
        hang.write_text(
            f"import os, time\n"
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
            f"time.sleep(120)\n"
        )
        shim = Path(self._tmpdir.name) / "hang_shim.sh"
        shim.write_text(f"#!/bin/sh\nexec {sys.executable} {hang} \"$@\"\n")
        shim.chmod(0o755)

        sampler = supervisor.RusageSampler()
        completed, _result, exc = _run_with_bound(
            lambda: supervisor.run_one_trial(
                binary=shim,
                model_dir=Path("/fake/model"),
                context=1,
                max_tokens=1,
                warmup_tokens=0,
                seed=1,
                sampler=sampler,
                timeout_s=2.0,
            ),
            bound_s=30.0,
        )
        self.assertTrue(completed, "run_one_trial did not honor timeout_s for a hanging child (it hung instead)")
        self.assertIsInstance(exc, supervisor.TrialFailure)

        # Confirmed reaped: the child process (an `exec`-based shim, so its
        # pid IS the actual sleeping python process, recorded via its own
        # os.getpid()) must no longer exist. `os.kill(pid, 0)` raises
        # ProcessLookupError once a pid is fully reaped/gone; it never
        # raises for a pid that is still alive (or a zombie still
        # occupying the process table).
        if pid_file.exists():
            child_pid = int(pid_file.read_text().strip())
            deadline = time.monotonic() + 5.0
            reaped = False
            while time.monotonic() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    reaped = True
                    break
                time.sleep(0.1)
            self.assertTrue(reaped, f"child pid {child_pid} was never reaped after timeout")

    def test_stderr_filling_child_does_not_deadlock(self):
        # Writes far more to stderr than a typical OS pipe buffer (64KB on
        # macOS) while stdout stays open and silent, then exits cleanly.
        # Without concurrent stderr draining, the child blocks on its own
        # full stderr pipe, the parent is blocked reading stdout, and
        # neither side ever makes progress.
        noisy = Path(self._tmpdir.name) / "noisy_child.py"
        noisy.write_text(
            "import sys\n"
            "for _ in range(20000):\n"
            "    sys.stderr.write('x' * 100 + '\\n')\n"
            "sys.stderr.flush()\n"
            "sys.exit(0)\n"
        )
        shim = Path(self._tmpdir.name) / "noisy_shim.sh"
        shim.write_text(f"#!/bin/sh\nexec {sys.executable} {noisy} \"$@\"\n")
        shim.chmod(0o755)

        sampler = supervisor.RusageSampler()
        # This child never emits a @@bench summary line, so run_one_trial
        # is expected to fail closed (no summary) -- the point of this
        # test is that it fails PROMPTLY (no deadlock), not what it fails
        # with. Bounded the same way as the hang test: a deadlock
        # regression here must not be able to wedge the whole test run.
        completed, _result, exc = _run_with_bound(
            lambda: supervisor.run_one_trial(
                binary=shim,
                model_dir=Path("/fake/model"),
                context=1,
                max_tokens=1,
                warmup_tokens=0,
                seed=1,
                sampler=sampler,
                timeout_s=15.0,
            ),
            bound_s=20.0,
        )
        self.assertTrue(completed, "run_one_trial deadlocked on a stderr-filling child")
        self.assertIsInstance(exc, supervisor.TrialFailure)


def _phase_events_dicts(
    *,
    load_start_ns=0,
    backend_ready_ns=10_000_000,
    prefill_start_ns=10_000_000,
    prefill_end_ns=20_000_000,
    token_gap_ns=5_000_000,
    n_tokens=4,
    token_indices=None,
) -> list[dict]:
    """Builds the `run_one_trial`-shaped raw phase-event dict list (the
    exact shape `_validate_trial_trace` consumes: name/child_ns/token_index/
    parent_ns) in normal, valid, arrival order. `token_indices`, when given,
    overrides the default contiguous 1..n_tokens sequence -- used to
    construct reversed/duplicate/gapped malformed traces for the rejection
    tests below."""
    indices = token_indices if token_indices is not None else list(range(1, n_tokens + 1))
    events = [
        {"name": "load_start", "child_ns": load_start_ns, "token_index": None, "parent_ns": load_start_ns},
        {"name": "backend_ready", "child_ns": backend_ready_ns, "token_index": None, "parent_ns": backend_ready_ns},
        {"name": "prefill_start", "child_ns": prefill_start_ns, "token_index": None, "parent_ns": prefill_start_ns},
        {"name": "prefill_end", "child_ns": prefill_end_ns, "token_index": None, "parent_ns": prefill_end_ns},
    ]
    t = prefill_end_ns
    for idx in indices:
        t += token_gap_ns
        events.append({"name": "token_available", "child_ns": t, "token_index": idx, "parent_ns": t})
    return events


def _valid_summary(n_tokens=4, *, delta_matches=True) -> dict:
    return {
        "prompt_tokens": 512,
        "generated_tokens": n_tokens,
        "requested_max_tokens": n_tokens,
        "delta_matches_generated_tokens": delta_matches,
        "stopped": "length",
    }


class TrialTraceValidationTest(unittest.TestCase):
    """Direct unit tests of `_validate_trial_trace` -- the always-on, every-
    trial, pre-aggregation trace validator added for codex round-1 blocker
    #2 ("the supervisor accepts malformed phase traces ... reorders token
    events, and therefore lets the n=2 demonstration validate without
    proving its measurement boundary"). Each rejection case here is one
    codex explicitly asked for by name; every one must raise
    `supervisor.TrialFailure`, never silently repair the trace."""

    def test_valid_trace_passes(self):
        events = _phase_events_dicts(n_tokens=4)
        supervisor._validate_trial_trace(events, _valid_summary(4), "t")  # must not raise

    def test_reversed_token_events_rejected(self):
        events = _phase_events_dicts(n_tokens=4)
        # Reverse only the token_available tail (the four raw dicts appended
        # after the four single-shot phases), preserving load/backend/
        # prefill order -- this is the "in-memory trace with its four token
        # events reversed" case codex reported the pre-fix code accepted.
        head, tail = events[:4], events[4:]
        supervisor_events = head + list(reversed(tail))
        with self.assertRaises(supervisor.TrialFailure):
            supervisor._validate_trial_trace(supervisor_events, _valid_summary(4), "t")

    def test_repeated_single_shot_phase_rejected(self):
        events = _phase_events_dicts(n_tokens=1)
        events.insert(1, dict(events[0]))  # duplicate load_start
        with self.assertRaises(supervisor.TrialFailure):
            supervisor._validate_trial_trace(events, _valid_summary(1), "t")

    def test_missing_required_phase_rejected(self):
        events = [e for e in _phase_events_dicts(n_tokens=1) if e["name"] != "prefill_end"]
        with self.assertRaises(supervisor.TrialFailure):
            supervisor._validate_trial_trace(events, _valid_summary(1), "t")

    def test_decreasing_timestamp_rejected(self):
        events = _phase_events_dicts(n_tokens=1)
        # Force backend_ready's own child_ns before load_start's.
        events[1] = dict(events[1], child_ns=-1)
        with self.assertRaises(supervisor.TrialFailure):
            supervisor._validate_trial_trace(events, _valid_summary(1), "t")

    def test_duplicate_token_index_rejected(self):
        events = _phase_events_dicts(n_tokens=3, token_indices=[1, 2, 2])
        with self.assertRaises(supervisor.TrialFailure):
            supervisor._validate_trial_trace(events, _valid_summary(3), "t")

    def test_gapped_token_index_rejected(self):
        # Strictly increasing (passes _validate_phase_sequence's own check)
        # but not contiguous from 1 -- the explicit contiguity check added
        # on top of the reused harness validator must catch this.
        events = _phase_events_dicts(n_tokens=3, token_indices=[1, 3, 4])
        with self.assertRaises(supervisor.TrialFailure):
            supervisor._validate_trial_trace(events, _valid_summary(3), "t")

    def test_delta_matches_generated_tokens_false_rejected(self):
        events = _phase_events_dicts(n_tokens=4)
        with self.assertRaises(supervisor.TrialFailure):
            supervisor._validate_trial_trace(events, _valid_summary(4, delta_matches=False), "t")

    def test_summary_event_count_mismatch_rejected(self):
        # "a summary claiming 9 generated tokens with only 4 events" --
        # codex's own reported false-accept.
        events = _phase_events_dicts(n_tokens=4)
        with self.assertRaises(supervisor.TrialFailure):
            supervisor._validate_trial_trace(events, _valid_summary(9), "t")

    def test_underpowered_record_still_gets_trace_validated(self):
        """Proves the fix for the second half of blocker #2: trace
        validation now runs unconditionally inside `run_one_trial`, well
        before `main()`'s later `verdict = "unsupported"` downgrade
        decision even exists -- so an n=2 demonstration record's trial
        traces ARE validated, unlike the pre-fix path where
        `validate_run_record`'s `_validate_phase_sequence` call was itself
        skipped for `verdict == "unsupported"` records. A malformed trace
        from what will become an unsupported-verdict session must still be
        rejected at `run_one_trial` time."""
        events = _phase_events_dicts(n_tokens=2, token_indices=[2, 1])  # reversed, tiny n
        with self.assertRaises(supervisor.TrialFailure):
            supervisor._validate_trial_trace(events, _valid_summary(2), "underpowered-demo-trial")


# ---------------------------------------------------------------------------
# compute_trial_metrics against hand-built phase events (no subprocess) --
# exact-value assertions on the aggregation math itself.
# ---------------------------------------------------------------------------


def _trial(
    *,
    prompt_tokens=1024,
    load_start=0,
    backend_ready=40_000_000,
    prefill_start=40_000_000,
    prefill_end=55_000_000,
    token_gaps_ns=(4_000_000, 4_000_000, 4_000_000, 4_000_000),
    generated_tokens=None,
    stopped="length",
) -> dict:
    events = [
        {"name": "load_start", "child_ns": load_start, "token_index": None, "parent_ns": load_start},
        {"name": "backend_ready", "child_ns": backend_ready, "token_index": None, "parent_ns": backend_ready},
        {"name": "prefill_start", "child_ns": prefill_start, "token_index": None, "parent_ns": prefill_start},
        {"name": "prefill_end", "child_ns": prefill_end, "token_index": None, "parent_ns": prefill_end},
    ]
    t = prefill_end
    for i, gap in enumerate(token_gaps_ns):
        t += gap
        events.append({"name": "token_available", "child_ns": t, "token_index": i, "parent_ns": t})
    n_tok = generated_tokens if generated_tokens is not None else len(token_gaps_ns)
    return {
        "phase_events": events,
        "summary": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": n_tok,
            "requested_max_tokens": n_tok,
            "delta_matches_generated_tokens": True,
            "stopped": stopped,
        },
        "resource_samples": [
            supervisor.RawResourceSample(parent_ns=load_start, phys_footprint_bytes=1_000_000, phase="load_start"),
            supervisor.RawResourceSample(parent_ns=backend_ready, phys_footprint_bytes=1_500_000, phase="backend_ready"),
            supervisor.RawResourceSample(parent_ns=prefill_end, phys_footprint_bytes=1_800_000, phase="prefill_end"),
            supervisor.RawResourceSample(parent_ns=t, phys_footprint_bytes=2_000_000, phase="token_available"),
        ],
    }


class ComputeTrialMetricsTest(unittest.TestCase):
    def test_exact_metric_values(self):
        trial = _trial()
        m = supervisor.compute_trial_metrics(trial)
        self.assertEqual(m["prompt_tokens"], 1024)
        self.assertEqual(m["generated_tokens"], 4)
        self.assertEqual(m["load_ms"], 40.0)
        # prefill: 1024 tokens / 0.015s = 68266.67 tok/s
        self.assertAlmostEqual(m["prefill_tok_s"], 1024 / 0.015, places=3)
        # TTFT: first token at prefill_end + 4ms, measured from prefill_start
        self.assertAlmostEqual(m["ttft_ms"], 19.0, places=6)
        # decode: tokens 2..4 (3 timed tokens) over 3 * 4ms = 12ms -> 250 tok/s
        self.assertAlmostEqual(m["decode_tok_s"], 3 / 0.012, places=3)
        self.assertEqual(m["per_token_latencies_ms"], [4.0, 4.0, 4.0])
        self.assertEqual(m["per_token_p50_ms"], 4.0)
        self.assertEqual(m["peak_phys_footprint_bytes"], 2_000_000)

    def test_single_generated_token_reports_no_decode_rate(self):
        trial = _trial(token_gaps_ns=(4_000_000,), generated_tokens=1)
        m = supervisor.compute_trial_metrics(trial)
        self.assertEqual(m["generated_tokens"], 1)
        self.assertIsNone(m["decode_tok_s"])
        self.assertEqual(m["per_token_latencies_ms"], [])
        self.assertIsNone(m["per_token_p50_ms"])

    def test_missing_prefill_end_fails_closed(self):
        trial = _trial()
        trial["phase_events"] = [e for e in trial["phase_events"] if e["name"] != "prefill_end"]
        with self.assertRaises(supervisor.TrialFailure):
            supervisor.compute_trial_metrics(trial)

    def test_phase_tagged_peak_memory_buckets(self):
        trial = _trial()
        m = supervisor.compute_trial_metrics(trial)
        by_phase = m["peak_phys_footprint_by_phase"]
        self.assertEqual(by_phase["load"], 1_500_000)
        self.assertEqual(by_phase["decode"], 2_000_000)


class PercentileTest(unittest.TestCase):
    def test_empty(self):
        self.assertIsNone(supervisor._percentile([], 0.5))

    def test_single_value(self):
        self.assertEqual(supervisor._percentile([7.0], 0.95), 7.0)

    def test_matches_known_values(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(supervisor._percentile(xs, 0.5), 3.0)
        self.assertEqual(supervisor._percentile(xs, 0.0), 1.0)
        self.assertEqual(supervisor._percentile(xs, 1.0), 5.0)


# ---------------------------------------------------------------------------
# Paired-session gate-math folding + validate_run_record round-trip
# ---------------------------------------------------------------------------


def _fake_raw_phase_events() -> list[dict]:
    """A minimal, valid load->prefill->decode phase-event sequence
    satisfying bench_decode_harness._validate_phase_sequence: exactly one
    each of the four single-shot phases in rank order, plus one
    token_available. Shape-matches compute_trial_metrics's
    "raw_phase_events" output (name/monotonic_ns/token_index).
    token_index is 1-indexed (>= 1), matching both parse_phase_event's
    requirement and the real qwen35_generate binary's own convention
    (its first emitted token_available carries token_index=1, not 0)."""
    return [
        {"name": "load_start", "monotonic_ns": 0, "token_index": None},
        {"name": "backend_ready", "monotonic_ns": 40_000_000, "token_index": None},
        {"name": "prefill_start", "monotonic_ns": 40_000_000, "token_index": None},
        {"name": "prefill_end", "monotonic_ns": 55_000_000, "token_index": None},
        {"name": "token_available", "monotonic_ns": 59_000_000, "token_index": 1},
    ]


def _fake_session(decode_values_a: list[float], decode_values_b: list[float]) -> dict:
    assert len(decode_values_a) == len(decode_values_b)
    n = len(decode_values_a)
    arm_a = [{"decode_tok_s": v, "raw_phase_events": _fake_raw_phase_events()} for v in decode_values_a]
    arm_b = [{"decode_tok_s": v, "raw_phase_events": _fake_raw_phase_events()} for v in decode_values_b]
    pairs = [
        {
            "pair_index": i,
            "order": "AB" if i % 2 == 0 else "BA",
            "thermal_state_before": None,
            "thermal_state_after": None,
        }
        for i in range(n)
    ]
    order_ab = sum(1 for p in pairs if p["order"] == "AB")
    order_ba = n - order_ab
    return {"arm_a": arm_a, "arm_b": arm_b, "pairs": pairs, "order_balance": (order_ab, order_ba)}


class BuildDecodeCellAggregateTest(unittest.TestCase):
    def setUp(self):
        self.policy = gate_math.load_policy()

    def test_low_noise_seven_pairs_is_gate_eligible(self):
        # Near-identical A/B values -> tiny measured_cv -> class-A cv_bands
        # first band (max_cv=0.015) requires n=7, matching exactly.
        values = [200.0, 200.1, 199.9, 200.2, 199.8, 200.0, 200.1]
        session = _fake_session(values, values)
        aggregate, diag = supervisor.build_decode_cell_aggregate(session, self.policy, rng_seed=1)
        self.assertEqual(diag["n_pairs"], 7)
        self.assertIsNotNone(diag["measured_cv"])
        self.assertLess(diag["measured_cv"], 0.015)
        self.assertEqual(diag["required_n"], 7)
        self.assertGreaterEqual(aggregate.n_valid, aggregate.required_n)
        self.assertAlmostEqual(aggregate.point_estimate, 0.0, delta=0.01)

    def test_undersized_n_yields_required_n_greater_than_n_valid(self):
        values = [200.0, 200.1]
        session = _fake_session(values, values)
        aggregate, diag = supervisor.build_decode_cell_aggregate(session, self.policy, rng_seed=1)
        self.assertEqual(diag["n_pairs"], 2)
        self.assertIsNotNone(aggregate.required_n)
        self.assertLess(aggregate.n_valid, aggregate.required_n)

    def test_missing_decode_rate_raises(self):
        session = _fake_session([200.0], [200.0])
        session["arm_a"][0]["decode_tok_s"] = None
        with self.assertRaises(ValueError):
            supervisor.build_decode_cell_aggregate(session, self.policy, rng_seed=1)


class DecodeVerdictTest(unittest.TestCase):
    """PR #898 review, round 1 finding #2: `decode_verdict` must compare
    `aggregate.corrected_lower_bound`/`aggregate.point_estimate` (both
    LOG-space, since `build_decode_cell_aggregate` bootstraps
    `log_slowdowns`) against LOG-space thresholds (`math.log1p(fail_pct)`/
    `math.log1p(warn_pct)`), never the raw `perf-policy.toml` fractions
    directly. These fixtures pin a concrete boundary window
    `(log1p(x), x]` where a raw-fraction comparison and the correct
    log-space comparison DISAGREE, so the fix is provably load-bearing
    rather than a no-op relabeling -- not just "some value below 0.07",
    but specifically a value the pre-fix comparison would have missed."""

    def _agg(self, *, point_estimate, corrected_lower_bound, n_valid=7, required_n=7, measured_cv=0.01):
        return harness.CellAggregate(
            point_estimate=point_estimate,
            ci_low=point_estimate,
            ci_high=point_estimate,
            corrected_lower_bound=corrected_lower_bound,
            n_valid=n_valid,
            n_invalid=0,
            measured_cv=measured_cv,
            required_n=required_n,
        )

    def test_bound_in_log1p_boundary_window_is_fail(self):
        fail_pct = 0.07
        tau_fail = math.log1p(fail_pct)
        cb = (tau_fail + fail_pct) / 2.0  # strictly inside (tau_fail, fail_pct]
        self.assertGreater(cb, tau_fail)
        self.assertLessEqual(cb, fail_pct)
        # Precondition proving this window actually disagrees with a raw-
        # fraction comparison: cb does NOT clear the raw fail_pct itself,
        # so the pre-fix `cb > fail_pct` comparison would have reported
        # WARN/PASS here instead of the genuine policy FAIL.
        self.assertFalse(cb > fail_pct)
        aggregate = self._agg(point_estimate=0.01, corrected_lower_bound=cb)
        diagnostics = {"n_pairs": 7, "required_n": 7, "measured_cv": 0.01}
        verdict, reason = supervisor.decode_verdict(aggregate, diagnostics, fail_pct=fail_pct, warn_pct=0.03)
        self.assertEqual(verdict, "FAIL")
        self.assertIsNone(reason)

    def test_point_estimate_in_log1p_boundary_window_is_warn(self):
        warn_pct = 0.03
        tau_warn = math.log1p(warn_pct)
        pe = (tau_warn + warn_pct) / 2.0  # strictly inside (tau_warn, warn_pct]
        self.assertGreater(pe, tau_warn)
        self.assertLessEqual(pe, warn_pct)
        self.assertFalse(pe > warn_pct)
        aggregate = self._agg(point_estimate=pe, corrected_lower_bound=None)
        diagnostics = {"n_pairs": 7, "required_n": 7, "measured_cv": 0.01}
        verdict, reason = supervisor.decode_verdict(aggregate, diagnostics, fail_pct=0.07, warn_pct=warn_pct)
        self.assertEqual(verdict, "WARN")
        self.assertIsNone(reason)

    def test_undersampled_session_reports_unsupported_regardless_of_bound(self):
        aggregate = self._agg(point_estimate=10.0, corrected_lower_bound=10.0, n_valid=2, required_n=7)
        diagnostics = {"n_pairs": 2, "required_n": 7, "measured_cv": 0.01}
        verdict, reason = supervisor.decode_verdict(aggregate, diagnostics, fail_pct=0.07, warn_pct=0.03)
        self.assertEqual(verdict, "unsupported")
        self.assertIsNotNone(reason)

    def test_gate_eligible_session_can_reach_a_real_verdict(self):
        """Reconciliation check (PR #898 review, round 1 finding #2): a
        sufficiently-powered session is NOT unconditionally downgraded to
        'unsupported' -- it reaches a genuine PASS/WARN/FAIL verdict, so
        the module docstring/comments must never claim shadow-mode safety
        comes from an always-unsupported verdict."""
        aggregate = self._agg(point_estimate=0.0, corrected_lower_bound=0.0)
        diagnostics = {"n_pairs": 7, "required_n": 7, "measured_cv": 0.01}
        verdict, reason = supervisor.decode_verdict(aggregate, diagnostics, fail_pct=0.07, warn_pct=0.03)
        self.assertEqual(verdict, "PASS")
        self.assertIsNone(reason)


class ValidateRunRecordRoundTripTest(unittest.TestCase):
    """Proves records this supervisor assembles pass PR #877's fail-closed
    `validate_run_record` -- both the sufficiently-powered PASS-shaped path
    and the deliberately-underpowered 'unsupported' path (which must
    validate cleanly, never raise, by design; see the module docstring in
    bench_cpu_flagship_supervisor.py)."""

    def setUp(self):
        self.policy = gate_math.load_policy()
        self.policy_sha = "d" * 64  # arbitrary fixed sha for this test's provenance

    def _provenance(self) -> harness.ProvenanceRecord:
        return harness.parse_provenance(
            {
                "repo_sha": "a" * 40,
                "candidate_sha": "a" * 40,
                "base_sha": None,
                "dirty": False,
                "profile_name": "cpu_flagship_smoke_v1",
                "profile_version": 1,
                "profile_sha": "c" * 64,
                "policy_version": self.policy["policy_version"],
                "policy_sha": self.policy_sha,
                "script_sha": "e" * 40,
                "hardware_fingerprint": "Darwin-arm64-test",
                "collected_at": "2026-07-11T00:00:00+00:00",
                "workflow_run_id": None,
            }
        )

    def _record_from_session(self, session: dict) -> harness.CellRecord:
        aggregate, diag = supervisor.build_decode_cell_aggregate(session, self.policy, rng_seed=7)
        gate_eligible = (
            aggregate.measured_cv is not None
            and aggregate.required_n is not None
            and aggregate.n_valid >= aggregate.required_n
        )
        if not gate_eligible:
            verdict = "unsupported"
            unsupported_reason = "n_valid < required_n (test fixture)"
            phase_events: tuple = ()
        else:
            verdict = "PASS"
            unsupported_reason = None
            phase_events = tuple(harness.parse_phase_event(ev) for ev in session["arm_a"][0]["raw_phase_events"])
        return harness.CellRecord(
            cell_id=harness.cell_id("decode", "qwen3.5-small", "f16", "cpu", 512),
            metric_family="decode",
            metric_name="decode_tok_s",
            path="decode",
            model_tier="qwen3.5-small",
            quant_tier="f16",
            device="cpu",
            context_point=512,
            aggregate=aggregate,
            verdict=verdict,
            unsupported_reason=unsupported_reason,
            path_proof=("cpu:qwen35_generate::generate_streaming_with_cancel",),
            lock_receipts=(),
            phase_events=phase_events,
            resource_samples=(),
            provenance=self._provenance(),
            order_balance=tuple(diag["order_balance"]),
            raw_artifact_digest=None,
        )

    def test_sufficiently_powered_session_validates(self):
        values = [200.0, 200.1, 199.9, 200.2, 199.8, 200.0, 200.1]
        session = _fake_session(values, values)
        record = self._record_from_session(session)
        self.assertEqual(record.verdict, "PASS")
        harness.validate_run_record(record, expected_repo_sha="a" * 40, current_policy_sha=self.policy_sha)

    def test_underpowered_demonstration_session_is_marked_unsupported_and_validates(self):
        values = [200.0, 200.1]
        session = _fake_session(values, values)
        record = self._record_from_session(session)
        self.assertEqual(record.verdict, "unsupported")
        self.assertIsNotNone(record.unsupported_reason)
        # The whole point of downgrading to "unsupported" before submission:
        # validate_run_record's low-valid-n check is skipped for
        # unsupported verdicts, so this record validates cleanly rather
        # than raising -- an honest, self-disclosing demonstration record.
        harness.validate_run_record(record, expected_repo_sha="a" * 40, current_policy_sha=self.policy_sha)

    def test_wrong_repo_sha_fails_closed(self):
        values = [200.0, 200.1, 199.9, 200.2, 199.8, 200.0, 200.1]
        session = _fake_session(values, values)
        record = self._record_from_session(session)
        with self.assertRaises(harness.RunRecordValidationError):
            harness.validate_run_record(record, expected_repo_sha="b" * 40, current_policy_sha=self.policy_sha)

    def test_stale_policy_sha_fails_closed(self):
        values = [200.0, 200.1, 199.9, 200.2, 199.8, 200.0, 200.1]
        session = _fake_session(values, values)
        record = self._record_from_session(session)
        with self.assertRaises(harness.RunRecordValidationError):
            harness.validate_run_record(record, expected_repo_sha="a" * 40, current_policy_sha="stale" * 16)


class HardwareFingerprintAndHashTest(unittest.TestCase):
    def test_hardware_fingerprint_nonempty(self):
        self.assertTrue(supervisor.hardware_fingerprint())

    def test_sha256_file_matches_known_digest(self):
        import hashlib
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(b"hello world")
            path = Path(fh.name)
        try:
            expected = hashlib.sha256(b"hello world").hexdigest()
            self.assertEqual(supervisor.sha256_file(path), expected)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()

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
import statistics
import sys
import textwrap
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
    token_lines = "\n".join(
        f'emit(\'@@bench {{"ev":"phase","name":"token_available","monotonic_ns":%d,"token_index":{i}}}\' % ns({prefill_end_ms + (i + 1) * token_gap_ms}))'
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


def _fake_session(decode_values_a: list[float], decode_values_b: list[float]) -> dict:
    assert len(decode_values_a) == len(decode_values_b)
    n = len(decode_values_a)
    arm_a = [{"decode_tok_s": v} for v in decode_values_a]
    arm_b = [{"decode_tok_s": v} for v in decode_values_b]
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
        else:
            verdict = "PASS"
            unsupported_reason = None
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
            phase_events=(),
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

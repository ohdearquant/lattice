"""Unit tests for scripts/bench_gate_math.py (bench-overhaul PR 1).

Deterministic, seeded-RNG tests only -- no engine binary, GPU, or heavy
lane required. Loaded by file path (matching test_bench_decode_harness.py's
convention: scripts/ is not a package).

Run with: python3 -m pytest tests/test_bench_gate_math.py -v
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_gate_math.py"
_SPEC = importlib.util.spec_from_file_location("bench_gate_math", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
gm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gm
_SPEC.loader.exec_module(gm)


class LogSlowdownTest(unittest.TestCase):
    def test_higher_is_better_regression(self):
        # candidate throughput drops -> positive log-slowdown
        val = gm.log_slowdown(base=100.0, candidate=90.0, higher_is_better=True)
        self.assertGreater(val, 0)
        self.assertAlmostEqual(val, math.log(100 / 90))

    def test_lower_is_better_regression(self):
        # candidate latency rises -> positive log-slowdown
        val = gm.log_slowdown(base=10.0, candidate=12.0, higher_is_better=False)
        self.assertGreater(val, 0)
        self.assertAlmostEqual(val, math.log(12 / 10))

    def test_improvement_is_negative(self):
        val = gm.log_slowdown(base=100.0, candidate=110.0, higher_is_better=True)
        self.assertLess(val, 0)

    def test_non_positive_values_rejected(self):
        with self.assertRaises(gm.GateMathError):
            gm.log_slowdown(base=0.0, candidate=1.0, higher_is_better=True)
        with self.assertRaises(gm.GateMathError):
            gm.log_slowdown(base=1.0, candidate=-1.0, higher_is_better=True)


# --------------------------------------------------------------------------
# Correction 2: bootstrap the MEAN, never the median, at small n
# --------------------------------------------------------------------------


class BootstrapMeanTest(unittest.TestCase):
    def test_mean_bootstrap_has_more_than_seven_distinct_values_at_n7(self):
        """The whole point of correction 2: at n=7 the bootstrap
        distribution of the MEDIAN is capped at <= 7 distinct point
        masses (median of an odd-n sample is always one of the original
        observations), but the MEAN is not so capped -- this is the
        concrete, checkable difference the correction buys."""
        rng = random.Random(42)
        values = [1.0, 1.2, 0.9, 1.5, 0.7, 1.1, 1.3]
        order_ab = [True, False, True, False, True, False, True]
        means = gm.order_stratified_bootstrap_means(values, order_ab, 500, rng)
        distinct_means = {round(m, 9) for m in means}
        # 500 resamples of 7 elements -> many more than 7 distinct mean values
        self.assertGreater(len(distinct_means), 7)

    def test_median_of_odd_n_is_capped_at_n_distinct_values(self):
        """Companion check proving the CONTRAST is real: bootstrapping the
        median (not what this module does, but the thing correction 2
        rejects) over the same data is capped at 7 distinct values."""
        import statistics as _stats

        rng = random.Random(42)
        values = [1.0, 1.2, 0.9, 1.5, 0.7, 1.1, 1.3]
        n = len(values)
        medians = set()
        for _ in range(500):
            sample = [values[rng.randrange(n)] for _ in range(n)]
            medians.add(round(_stats.median(sample), 9))
        self.assertLessEqual(len(medians), n)

    def test_mismatched_lengths_rejected(self):
        rng = random.Random(1)
        with self.assertRaises(gm.GateMathError):
            gm.order_stratified_bootstrap_means([1.0, 2.0], [True], 10, rng)

    def test_empty_values_rejected(self):
        rng = random.Random(1)
        with self.assertRaises(gm.GateMathError):
            gm.order_stratified_bootstrap_means([], [], 10, rng)

    def test_degenerate_stratification_falls_back(self):
        rng = random.Random(7)
        values = [1.0, 2.0, 3.0, 4.0]
        order_ab = [True, True, True, True]  # no BA stratum
        means = gm.order_stratified_bootstrap_means(values, order_ab, 100, rng)
        self.assertEqual(len(means), 100)

    def test_bootstrap_pvalue_floored_at_one_over_b(self):
        rng = random.Random(3)
        # All slowdowns far above tau -> p should floor at 1/b, never hit 0
        values = [5.0] * 7
        order_ab = [True, False, True, False, True, False, True]
        p = gm.one_sided_bootstrap_pvalue(values, order_ab, tau_log=0.01, b=200, rng=rng)
        self.assertEqual(p, 1.0 / 200)

    def test_bootstrap_pvalue_high_when_null(self):
        rng = random.Random(9)
        values = [0.0] * 7  # exactly at tau=0
        order_ab = [True, False, True, False, True, False, True]
        p = gm.one_sided_bootstrap_pvalue(values, order_ab, tau_log=0.5, b=500, rng=rng)
        self.assertGreater(p, 0.5)

    def test_upper_bound_is_between_min_and_max(self):
        rng = random.Random(11)
        values = [0.1, 0.2, 0.15, 0.3, 0.05, 0.25, 0.18]
        order_ab = [True, False, True, False, True, False, True]
        bound = gm.bootstrap_upper_bound(values, order_ab, 1000, rng, corrected_alpha=0.01)
        self.assertGreaterEqual(bound, min(values) - 1e-6)
        self.assertLessEqual(bound, max(values) + 1e-6)

    def test_invalid_alpha_rejected(self):
        rng = random.Random(1)
        with self.assertRaises(gm.GateMathError):
            gm.bootstrap_upper_bound([1.0], [True], 10, rng, corrected_alpha=1.5)


class BootstrapLowerBoundTest(unittest.TestCase):
    """The FAIL rule needs a one-sided LOWER bound, not
    `bootstrap_upper_bound`'s `(1 - alpha)` upper tail. These fixtures
    pin the orientation."""

    def test_near_null_lower_bound_at_or_below_threshold(self):
        # point estimate (mean=0.07) sits just above threshold
        # log(1.07)=0.067659, but the sample has enough spread that the
        # corrected LOWER bound must sit at or below the threshold --
        # i.e. this is NOT a confirmed FAIL.
        tau = math.log(1.07)
        values = [0.05, 0.09, 0.06, 0.08, 0.04, 0.10, 0.07]
        order_ab = [True, False, True, False, True, False, True]
        rng = random.Random(123)
        lower = gm.bootstrap_lower_bound(values, order_ab, 2000, rng, corrected_alpha=0.01)
        self.assertLessEqual(lower, tau)

    def test_stable_ten_percent_slowdown_lower_bound_above_threshold(self):
        # a deterministic (zero-variance) 10% slowdown must clear the 7%
        # FAIL threshold at the corrected LOWER bound.
        tau = math.log(1.07)
        values = [math.log(1.10)] * 7
        order_ab = [True, False, True, False, True, False, True]
        rng = random.Random(123)
        lower = gm.bootstrap_lower_bound(values, order_ab, 2000, rng, corrected_alpha=0.01)
        self.assertGreater(lower, tau)

    def test_lower_bound_never_exceeds_upper_bound(self):
        rng_lo = random.Random(11)
        rng_hi = random.Random(11)
        values = [0.1, 0.2, 0.15, 0.3, 0.05, 0.25, 0.18]
        order_ab = [True, False, True, False, True, False, True]
        lower = gm.bootstrap_lower_bound(values, order_ab, 1000, rng_lo, corrected_alpha=0.01)
        upper = gm.bootstrap_upper_bound(values, order_ab, 1000, rng_hi, corrected_alpha=0.01)
        self.assertLessEqual(lower, upper)

    def test_invalid_alpha_rejected(self):
        rng = random.Random(1)
        with self.assertRaises(gm.GateMathError):
            gm.bootstrap_lower_bound([1.0], [True], 10, rng, corrected_alpha=0.0)


# --------------------------------------------------------------------------
# Holm step-down
# --------------------------------------------------------------------------


class HolmRejectTest(unittest.TestCase):
    def test_all_significant_all_rejected(self):
        pvals = [0.001, 0.002, 0.003]
        self.assertEqual(gm.holm_reject(pvals, alpha=0.05), [True, True, True])

    def test_none_significant_none_rejected(self):
        pvals = [0.9, 0.8, 0.7]
        self.assertEqual(gm.holm_reject(pvals, alpha=0.05), [False, False, False])

    def test_step_down_stops_at_first_failure(self):
        # smallest p rejected (0.01 <= 0.05/3), second p 0.02 <= 0.05/2 rejected,
        # third p 0.5 > 0.05/1 not rejected
        pvals = [0.5, 0.02, 0.01]
        result = gm.holm_reject(pvals, alpha=0.05)
        self.assertEqual(result, [False, True, True])

    def test_input_order_preserved(self):
        pvals = [0.5, 0.001]
        result = gm.holm_reject(pvals, alpha=0.05)
        self.assertEqual(len(result), 2)
        self.assertFalse(result[0])
        self.assertTrue(result[1])

    def test_empty_rejected(self):
        with self.assertRaises(gm.GateMathError):
            gm.holm_reject([])

    def test_single_hypothesis(self):
        self.assertEqual(gm.holm_reject([0.01], alpha=0.05), [True])
        self.assertEqual(gm.holm_reject([0.5], alpha=0.05), [False])


# --------------------------------------------------------------------------
# Correction 3: exact Clopper-Pearson bound
# --------------------------------------------------------------------------


class ClopperPearsonTest(unittest.TestCase):
    def test_zero_failures_twenty_sessions_matches_independent_check(self):
        # cross-checked against an independent Monte-Carlo simulation:
        # n=20, k=0 -> 95% upper CP bound = 13.91%
        bound = gm.clopper_pearson_upper(k=0, n=20, conf=0.95)
        self.assertAlmostEqual(bound, 0.1391, places=3)

    def test_zero_failures_three_hundred_sessions_below_one_percent(self):
        # independent check: n=300, k=0 -> 95% upper CP bound = 0.99%
        bound = gm.clopper_pearson_upper(k=0, n=300, conf=0.95)
        self.assertLess(bound, 0.01)
        self.assertAlmostEqual(bound, 0.0099, places=3)

    def test_bound_widens_with_more_failures(self):
        b0 = gm.clopper_pearson_upper(k=0, n=20, conf=0.95)
        b1 = gm.clopper_pearson_upper(k=1, n=20, conf=0.95)
        b2 = gm.clopper_pearson_upper(k=2, n=20, conf=0.95)
        self.assertLess(b0, b1)
        self.assertLess(b1, b2)

    def test_bound_tightens_with_more_sessions_at_same_k(self):
        b20 = gm.clopper_pearson_upper(k=0, n=20, conf=0.95)
        b100 = gm.clopper_pearson_upper(k=0, n=100, conf=0.95)
        b500 = gm.clopper_pearson_upper(k=0, n=500, conf=0.95)
        self.assertGreater(b20, b100)
        self.assertGreater(b100, b500)

    def test_k_equals_n_returns_one(self):
        self.assertEqual(gm.clopper_pearson_upper(k=5, n=5, conf=0.95), 1.0)

    def test_invalid_k_n_rejected(self):
        with self.assertRaises(gm.GateMathError):
            gm.clopper_pearson_upper(k=-1, n=10)
        with self.assertRaises(gm.GateMathError):
            gm.clopper_pearson_upper(k=11, n=10)
        with self.assertRaises(gm.GateMathError):
            gm.clopper_pearson_upper(k=0, n=0)

    def test_twenty_sessions_cannot_prove_sub_one_percent(self):
        """The core honesty check correction 3 encodes: 20 null A/A
        sessions with zero failures does NOT prove <1% false-FAIL --
        it only bounds it to <=13.91%."""
        bound = gm.clopper_pearson_upper(k=0, n=20, conf=0.95)
        self.assertGreater(bound, 0.01)


# --------------------------------------------------------------------------
# Correction 1: measured-CV -> required-n bands
# --------------------------------------------------------------------------


class CvBandsTest(unittest.TestCase):
    def _bands(self):
        return gm.parse_cv_bands(
            [
                {"max_cv": 0.015, "required_n_class_a": 7, "required_n_class_b": 9, "fail_margin_multiplier": 1.0},
                {"max_cv": 0.05, "required_n_class_a": 25, "required_n_class_b": 25, "fail_margin_multiplier": 1.0},
                {"max_cv": 1.0, "required_n_class_a": 25, "required_n_class_b": 25, "fail_margin_multiplier": 2.0},
            ]
        )

    def test_low_cv_keeps_n7(self):
        bands = self._bands()
        n, margin = gm.required_n(0.01, bands, "A")
        self.assertEqual(n, 7)
        self.assertEqual(margin, 1.0)

    def test_moderate_cv_requires_n25(self):
        bands = self._bands()
        n, margin = gm.required_n(0.03, bands, "A")
        self.assertEqual(n, 25)

    def test_high_cv_widens_margin_not_n(self):
        bands = self._bands()
        n, margin = gm.required_n(0.08, bands, "A")
        self.assertEqual(n, 25)  # not inflated further
        self.assertEqual(margin, 2.0)  # margin widened instead

    def test_class_b_uses_class_b_column(self):
        bands = self._bands()
        n, _ = gm.required_n(0.01, bands, "B")
        self.assertEqual(n, 9)

    def test_negative_cv_rejected(self):
        bands = self._bands()
        with self.assertRaises(gm.GateMathError):
            gm.required_n(-0.01, bands, "A")

    def test_invalid_class_rejected(self):
        bands = self._bands()
        with self.assertRaises(gm.GateMathError):
            gm.required_n(0.01, bands, "Z")

    def test_empty_bands_rejected(self):
        with self.assertRaises(gm.PolicyConfigError):
            gm.parse_cv_bands([])

    def test_bands_must_be_strictly_increasing(self):
        with self.assertRaises(gm.PolicyConfigError):
            gm.parse_cv_bands(
                [
                    {"max_cv": 0.05, "required_n_class_a": 7, "required_n_class_b": 9, "fail_margin_multiplier": 1.0},
                    {"max_cv": 0.02, "required_n_class_a": 25, "required_n_class_b": 25, "fail_margin_multiplier": 1.0},
                ]
            )

    def test_bands_must_reach_one(self):
        with self.assertRaises(gm.PolicyConfigError):
            gm.parse_cv_bands(
                [{"max_cv": 0.1, "required_n_class_a": 7, "required_n_class_b": 9, "fail_margin_multiplier": 1.0}]
            )

    def test_missing_key_rejected(self):
        with self.assertRaises(gm.PolicyConfigError):
            gm.parse_cv_bands([{"max_cv": 1.0, "required_n_class_a": 7, "required_n_class_b": 9}])


# --------------------------------------------------------------------------
# perf-policy.toml load (real, shipped file)
# --------------------------------------------------------------------------


class LoadPolicyTest(unittest.TestCase):
    def test_shipped_policy_file_loads(self):
        doc = gm.load_policy()
        self.assertEqual(doc["policy_version"], 1)
        self.assertIn("families", doc)
        self.assertIn("decode", doc["families"])

    def test_policy_sha_is_stable_hex_digest(self):
        sha1 = gm.policy_sha()
        sha2 = gm.policy_sha()
        self.assertEqual(sha1, sha2)
        self.assertEqual(len(sha1), 64)
        int(sha1, 16)  # must be valid hex

    def test_missing_file_rejected(self):
        with self.assertRaises(gm.PolicyConfigError):
            gm.load_policy(Path("/nonexistent/perf-policy.toml"))


# --------------------------------------------------------------------------
# Registered per-metric policy lookup (never a family heuristic)
# --------------------------------------------------------------------------


class ResolveMetricPolicyTest(unittest.TestCase):
    def test_flat_family_metric_resolves_class_a(self):
        doc = gm.load_policy()
        table = gm.resolve_metric_policy(doc, "decode", "decode_tok_s")
        self.assertEqual(table["noise_class"], "A")

    def test_nested_family_metric_resolves_class_b(self):
        # embed.batch_p95 is registered class B (min nine pairs) -- a
        # family-name heuristic that lumped every embed metric into class
        # A family-name heuristic would silently under-require this cell.
        doc = gm.load_policy()
        table = gm.resolve_metric_policy(doc, "embed", "batch_p95")
        self.assertEqual(table["noise_class"], "B")

    def test_nested_family_sibling_metric_resolves_class_a(self):
        doc = gm.load_policy()
        table = gm.resolve_metric_policy(doc, "embed", "texts_s")
        self.assertEqual(table["noise_class"], "A")

    def test_unregistered_family_rejected(self):
        doc = gm.load_policy()
        with self.assertRaises(gm.PolicyConfigError):
            gm.resolve_metric_policy(doc, "bogus_family", "decode_tok_s")

    def test_unregistered_metric_in_flat_family_rejected(self):
        doc = gm.load_policy()
        with self.assertRaises(gm.PolicyConfigError):
            gm.resolve_metric_policy(doc, "decode", "bogus_metric")

    def test_unregistered_metric_in_nested_family_rejected(self):
        doc = gm.load_policy()
        with self.assertRaises(gm.PolicyConfigError):
            gm.resolve_metric_policy(doc, "embed", "bogus_metric")

    def test_class_c_metric_resolves(self):
        doc = gm.load_policy()
        table = gm.resolve_metric_policy(doc, "memory", "model_load_time")
        self.assertEqual(table["noise_class"], "C")


# --------------------------------------------------------------------------
# The complete per-family gate evaluator (Holm + fail_margin_multiplier,
# integrated -- not just unit-tested helpers)
# --------------------------------------------------------------------------


class EvaluateFamilyGateTest(unittest.TestCase):
    def _bands(self):
        return gm.parse_cv_bands(
            [
                {"max_cv": 0.015, "required_n_class_a": 7, "required_n_class_b": 9, "fail_margin_multiplier": 1.0},
                {"max_cv": 0.05, "required_n_class_a": 25, "required_n_class_b": 25, "fail_margin_multiplier": 1.0},
                {"max_cv": 1.0, "required_n_class_a": 25, "required_n_class_b": 25, "fail_margin_multiplier": 2.0},
            ]
        )

    def test_empty_family_rejected(self):
        with self.assertRaises(gm.GateMathError):
            gm.evaluate_family_gate([], self._bands())

    def test_single_cell_strong_regression_fails(self):
        order_ab = (True, False, True, False, True, False, True)
        values = tuple([math.log(1.10)] * 7)  # deterministic 10% slowdown
        cell = gm.CellGateInput(
            cell_id="decode:strong", values=values, order_ab=order_ab,
            measured_cv=0.01, cell_class="A", warn_pct=0.03, fail_pct=0.07,
        )
        rng = random.Random(1)
        results = gm.evaluate_family_gate([cell], self._bands(), bootstrap_replicates=500, rng=rng)
        self.assertEqual(results[0].verdict, "FAIL")
        self.assertTrue(results[0].holm_reject)

    def test_single_cell_null_passes(self):
        order_ab = (True, False, True, False, True, False, True)
        values = (0.0,) * 7  # no slowdown at all
        cell = gm.CellGateInput(
            cell_id="decode:null", values=values, order_ab=order_ab,
            measured_cv=0.01, cell_class="A", warn_pct=0.03, fail_pct=0.07,
        )
        rng = random.Random(1)
        results = gm.evaluate_family_gate([cell], self._bands(), bootstrap_replicates=500, rng=rng)
        self.assertEqual(results[0].verdict, "PASS")
        self.assertFalse(results[0].holm_reject)

    def test_holm_ordering_changes_the_verdict(self):
        """Adversarial fixture: two
        cells whose bootstrap p-values are BOTH individually below the
        naive uncorrected alpha=0.05, but whose Holm step-down across the
        2-cell family rejects NEITHER, because the smaller of the two
        p-values fails to clear the first (tighter, alpha/2) Holm step --
        which halts the step-down, so the second cell is never rejected
        either, regardless of its own p-value. A naive per-cell 0.05
        check would have flagged both as FAIL; the family evaluator does
        not. This is the concrete, checkable case where ONLY the Holm
        step-down (never a per-cell threshold) determines the verdict."""
        tau = math.log(1.07)

        def cell(cid):
            base = tau + 0.005
            spread = 0.02
            values = (
                base - spread, base + spread, base - spread * 0.5, base + spread * 0.5,
                base - spread * 0.2, base + spread * 0.2, base,
            )
            order_ab = (True, False, True, False, True, False, True)
            return gm.CellGateInput(
                cell_id=cid, values=values, order_ab=order_ab,
                measured_cv=0.01, cell_class="A", warn_pct=0.03, fail_pct=0.07,
            )

        cell_a, cell_b = cell("a"), cell("b")
        rng = random.Random(0)
        results = gm.evaluate_family_gate([cell_a, cell_b], self._bands(), bootstrap_replicates=2000, rng=rng)
        p_a, p_b = results[0].p_value, results[1].p_value
        # Precondition: a naive, uncorrected per-cell check would flag
        # both (this is what makes the fixture adversarial -- ordering,
        # not magnitude, is what changes the verdict).
        self.assertLess(p_a, 0.05)
        self.assertLess(p_b, 0.05)
        # Holm, evaluated over the family, rejects neither.
        self.assertFalse(results[0].holm_reject)
        self.assertFalse(results[1].holm_reject)
        self.assertNotEqual(results[0].verdict, "FAIL")
        self.assertNotEqual(results[1].verdict, "FAIL")

    def test_holm_naive_mutation_flips_a_real_verdict(self):
        """Adversarial-review finding: the prior fixture only proved
        `holm_reject`'s internal flag was mutation-sensitive, not the
        actual verdict -- every cell's lower bound was fixed at the
        strictest step-0 Bonferroni tail regardless of Holm's per-cell
        step-down assignment, so Holm's reject flag was never the thing
        separating FAIL from non-FAIL. This fixture reuses the SAME
        two-cell family as `test_holm_ordering_changes_the_verdict` and
        directly swaps `holm_reject` for a naive uncorrected `p <= alpha`
        check via `mock.patch.object`, re-running the evaluator with an
        identically-seeded RNG so the bootstrap draws line up. Under the
        real, coherent step-down evaluator neither cell reaches FAIL (the
        family-level Holm step halts at the first cell); under the naive
        mutation, the cell whose own rank-assigned tail already clears
        its fail threshold DOES flip to FAIL -- the decision-effective
        proof that Holm's step-down, not a fixed bound, drives the
        verdict."""
        tau = math.log(1.07)

        def cell(cid):
            base = tau + 0.005
            spread = 0.02
            values = (
                base - spread, base + spread, base - spread * 0.5, base + spread * 0.5,
                base - spread * 0.2, base + spread * 0.2, base,
            )
            order_ab = (True, False, True, False, True, False, True)
            return gm.CellGateInput(
                cell_id=cid, values=values, order_ab=order_ab,
                measured_cv=0.01, cell_class="A", warn_pct=0.03, fail_pct=0.07,
            )

        cell_a, cell_b = cell("a"), cell("b")

        real_results = gm.evaluate_family_gate(
            [cell_a, cell_b], self._bands(), bootstrap_replicates=2000, rng=random.Random(0)
        )
        # Precondition matching test_holm_ordering_changes_the_verdict:
        # the real, coherent evaluator confirms neither cell as FAIL.
        self.assertFalse(real_results[0].holm_reject)
        self.assertFalse(real_results[1].holm_reject)
        self.assertNotEqual(real_results[0].verdict, "FAIL")
        self.assertNotEqual(real_results[1].verdict, "FAIL")

        def naive_reject(pvalues, alpha=0.05):
            return [p <= alpha for p in pvalues]

        with mock.patch.object(gm, "holm_reject", naive_reject):
            mutated_results = gm.evaluate_family_gate(
                [cell_a, cell_b], self._bands(), bootstrap_replicates=2000, rng=random.Random(0)
            )

        # The mutation must change at least one cell's ACTUAL verdict
        # (not merely its internal holm_reject flag) from non-FAIL to
        # FAIL -- proof that the shipped evaluator's FAIL rule depends on
        # the real Holm step-down, not just on crossing a fixed bound.
        flips = [
            (real.cell_id, real.verdict, mutated.verdict)
            for real, mutated in zip(real_results, mutated_results)
            if real.verdict != "FAIL" and mutated.verdict == "FAIL"
        ]
        self.assertTrue(flips, f"expected at least one non-FAIL->FAIL flip, got {flips!r}")

    def test_undersampled_cell_rejected_before_resampling(self):
        """Major 3: a cell reporting fewer raw pairs than its own
        policy-derived `required_n` must fail closed before any bootstrap
        resampling, never reach a PASS/WARN/FAIL verdict on insufficient
        evidence. One pair at measured_cv=0.01 (class A, required_n=7)."""
        cell = gm.CellGateInput(
            cell_id="undersampled", values=(math.log(1.10),), order_ab=(True,),
            measured_cv=0.01, cell_class="A", warn_pct=0.03, fail_pct=0.07,
        )
        with self.assertRaisesRegex(gm.GateMathError, "requires >= 7"):
            gm.evaluate_family_gate([cell], self._bands())

    def test_non_finite_value_rejected_before_resampling(self):
        """Major 3: seven raw pairs, one of them NaN, must fail closed
        before any bootstrap resampling rather than propagate into a
        NaN-bounded PASS verdict."""
        values = (math.log(1.10),) * 6 + (float("nan"),)
        order_ab = (True, False, True, False, True, False, True)
        cell = gm.CellGateInput(
            cell_id="nonfinite", values=values, order_ab=order_ab,
            measured_cv=0.01, cell_class="A", warn_pct=0.03, fail_pct=0.07,
        )
        with self.assertRaisesRegex(gm.GateMathError, "non-finite"):
            gm.evaluate_family_gate([cell], self._bands())

    def test_high_cv_margin_flips_fail_to_non_fail(self):
        """Adversarial fixture: the SAME
        effect size (a deterministic 8% slowdown) is a confirmed FAIL at
        the low-CV band's raw fail_pct margin (multiplier 1.0) but is NOT
        a FAIL once the same cell is measured at high CV and the
        registered `fail_margin_multiplier` (2.0) widens the threshold --
        proving `fail_margin_multiplier` is actually consumed by the
        gate decision, not just unit-tested in isolation."""
        order_ab_7 = (True, False, True, False, True, False, True)
        values_7 = tuple([math.log(1.08)] * 7)  # deterministic 8% slowdown
        # measured_cv=0.08 falls in the third band (required_n_class_a=25,
        # fail_margin_multiplier=2.0) -- must supply >= 25 raw pairs or the
        # under-sampled-cell guard (major 3) rejects before any bound is
        # even computed.
        order_ab_25 = tuple(i % 2 == 0 for i in range(25))
        values_25 = tuple([math.log(1.08)] * 25)

        low_cv_cell = gm.CellGateInput(
            cell_id="lowcv", values=values_7, order_ab=order_ab_7,
            measured_cv=0.01, cell_class="A", warn_pct=0.03, fail_pct=0.07,
        )
        high_cv_cell = gm.CellGateInput(
            cell_id="highcv", values=values_25, order_ab=order_ab_25,
            measured_cv=0.08, cell_class="A", warn_pct=0.03, fail_pct=0.07,
        )

        low_result = gm.evaluate_family_gate([low_cv_cell], self._bands(), bootstrap_replicates=500, rng=random.Random(5))
        high_result = gm.evaluate_family_gate([high_cv_cell], self._bands(), bootstrap_replicates=500, rng=random.Random(5))

        self.assertEqual(low_result[0].fail_margin_multiplier, 1.0)
        self.assertEqual(low_result[0].verdict, "FAIL")

        self.assertEqual(high_result[0].fail_margin_multiplier, 2.0)
        self.assertNotEqual(high_result[0].verdict, "FAIL")

    def test_mismatched_lengths_rejected(self):
        cell = gm.CellGateInput(
            cell_id="bad", values=(1.0, 2.0), order_ab=(True,),
            measured_cv=0.01, cell_class="A", warn_pct=0.03, fail_pct=0.07,
        )
        with self.assertRaises(gm.GateMathError):
            gm.evaluate_family_gate([cell], self._bands())

    def test_results_preserve_input_order(self):
        order_ab = (True, False, True, False, True, False, True)
        cell_first = gm.CellGateInput(
            cell_id="first", values=(0.0,) * 7, order_ab=order_ab,
            measured_cv=0.01, cell_class="A", warn_pct=0.03, fail_pct=0.07,
        )
        cell_second = gm.CellGateInput(
            cell_id="second", values=tuple([math.log(1.10)] * 7), order_ab=order_ab,
            measured_cv=0.01, cell_class="A", warn_pct=0.03, fail_pct=0.07,
        )
        results = gm.evaluate_family_gate([cell_first, cell_second], self._bands(), bootstrap_replicates=500, rng=random.Random(2))
        self.assertEqual([r.cell_id for r in results], ["first", "second"])


if __name__ == "__main__":
    unittest.main()

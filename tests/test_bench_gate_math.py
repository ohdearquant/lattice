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
    def test_zero_failures_twenty_sessions_matches_mathcheck(self):
        # Leo's mathcheck sim.py: n=20, k=0 -> 95% upper CP bound = 13.91%
        bound = gm.clopper_pearson_upper(k=0, n=20, conf=0.95)
        self.assertAlmostEqual(bound, 0.1391, places=3)

    def test_zero_failures_three_hundred_sessions_below_one_percent(self):
        # mathcheck: n=300, k=0 -> 95% upper CP bound = 0.99%
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


if __name__ == "__main__":
    unittest.main()

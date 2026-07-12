#!/usr/bin/env python3
"""Regression-gate statistics for the benchmark-overhaul contract (PR 1,
"canonical contract and policy, in days" — DESIGN.md section 4
"Regression-gating semantics").

Stdlib-only, matching `bench_decode_harness.py`'s convention: no numpy/scipy
dependency for the statistics a CI gate evaluator has to trust. Uses
`random.Random` (not numpy's RNG) and a manual bisection Clopper-Pearson
bound (not `scipy.stats.beta.ppf`) so this module never grows a third-party
dependency the gate can't audit.

Three of this module's gate-math choices were adversarially checked against
an independent Monte-Carlo simulation (power curves for the paired gate at
several CV levels, bootstrap tail behavior at small n, and exact binomial
bounds for promotion evidence) before the module was written.
All three corrections are encoded as executable functions,
not just prose, so the validator in `bench_decode_harness.py` can enforce
them instead of merely documenting them:

  1. `required_n` — minimum `n` is a function of the MEASURED same-session
     residual CV, not a fixed constant. At cv~=1% n=7 already has ~99%
     power against a true 10% regression (Class-A decode family, FAIL=7%).
     At cv~=3%, n=7 has only ~47-63% power; n~=23-25 restores ~80-84%. At
     cv>=8%, even n=30 only reaches ~31% power against the same true
     regression — inflating n further is not cost-effective, so that band
     widens the FAIL margin instead (see `perf-policy.toml` `[[cv_bands]]`
     `fail_margin_multiplier`). Bands are DATA in `perf-policy.toml`; this
     module only looks them up and refuses to guess a required n for a
     cell with no measured CV on record.

  2. `order_stratified_bootstrap_means` bootstraps the MEAN of paired
     log-slowdowns, never the median. At n=7 the bootstrap distribution of
     the MEDIAN is a discrete distribution over AT MOST 7 possible values
     (the 7 original order statistics — median of an odd-n sample is
     always one of the original observations), regardless of bootstrap
     replicate count B. A Holm-corrected per-cell tail cut at
     alpha/5=0.01 needs finer resolution than 7 point masses can express;
     the bootstrap mean does not have this ceiling.

  3. `clopper_pearson_upper` computes the EXACT one-sided upper confidence
     bound on a Bernoulli false-FAIL rate from (failures, sessions). 20
     null A/A sessions with k=0 observed failures bounds the true rate to
     only <=13.91% at 95% confidence — nowhere near "<1%". Reaching a
     bound that itself proves <1% at 95% confidence from zero observed
     failures needs ~298 sessions (`ln(0.05)/ln(0.99)`). DESIGN.md's
     promotion criterion ("an observed false-FAIL rate below 1% per
     workflow" after >=20 sessions) is a REPORTING requirement, not a
     pre-registered upfront session count: the promotion record must
     report the bound this function actually computes, never assert a
     tighter one, and null sessions keep accruing nightly so the bound
     tightens over time. See `PromotionRecord`/`validate_promotion_record`
     in `bench_decode_harness.py`.

Run with: python3 -m pytest tests/test_bench_gate_math.py -v
(stdlib-only; no engine binaries, GPU, or third-party packages required).
"""

from __future__ import annotations

import math
import random
import statistics
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_FILE = REPO_ROOT / "scripts" / "perf-policy.toml"

CELL_CLASSES = ("A", "B", "C")


class PolicyConfigError(ValueError):
    """`perf-policy.toml` is malformed or fails a fail-closed structural check."""


class GateMathError(ValueError):
    """A gate-math computation was asked to do something statistically unsound
    (e.g. compute a required-n for a CV with no matching band, or a Holm
    correction over zero hypotheses)."""


# --------------------------------------------------------------------------
# Log-slowdown
# --------------------------------------------------------------------------


def log_slowdown(base: float, candidate: float, *, higher_is_better: bool) -> float:
    """Paired log-slowdown for one A/B block (DESIGN.md section 4):

    `log(candidate/base)` when lower is better (e.g. TTFT, per-token
    latency), `log(base/candidate)` when higher is better (e.g. decode
    tok/s). Positive = regression (candidate worse than base) in both
    orientations.
    """
    if base <= 0 or candidate <= 0:
        raise GateMathError(f"log_slowdown requires positive values, got base={base!r} candidate={candidate!r}")
    if higher_is_better:
        return math.log(base / candidate)
    return math.log(candidate / base)


# --------------------------------------------------------------------------
# Correction 2: bootstrap the MEAN, never the median, at small n
# --------------------------------------------------------------------------


def order_stratified_bootstrap_means(
    values: Sequence[float],
    order_ab: Sequence[bool],
    b: int,
    rng: random.Random,
) -> list[float]:
    """Order-stratified bootstrap of the MEAN (correction 2 — never the
    median at small n, see module docstring). Resamples within the AB
    stratum and the BA stratum independently (so AB/BA balance in each
    resample tracks the registered balanced design), concatenates, and
    returns the mean of each of `b` resamples.

    Degenerate stratification (all one order label) falls back to an
    ordinary bootstrap over the whole sample — the paired design still
    holds, only the AB/BA balance guarantee is unavailable.
    """
    if len(values) != len(order_ab):
        raise GateMathError(
            f"values and order_ab must be the same length, got {len(values)} and {len(order_ab)}"
        )
    if not values:
        raise GateMathError("order_stratified_bootstrap_means requires at least one value")
    if b < 1:
        raise GateMathError("b (bootstrap replicate count) must be >= 1")

    ab_idx = [i for i, ab in enumerate(order_ab) if ab]
    ba_idx = [i for i, ab in enumerate(order_ab) if not ab]

    means: list[float] = []
    if not ab_idx or not ba_idx:
        n = len(values)
        for _ in range(b):
            sample = [values[rng.randrange(n)] for _ in range(n)]
            means.append(statistics.fmean(sample))
        return means

    for _ in range(b):
        sample = [values[ab_idx[rng.randrange(len(ab_idx))]] for _ in range(len(ab_idx))]
        sample += [values[ba_idx[rng.randrange(len(ba_idx))]] for _ in range(len(ba_idx))]
        means.append(statistics.fmean(sample))
    return means


def _pvalue_from_boot_means(boot_means: Sequence[float], tau_log: float, b: int) -> float:
    """Extraction-only half of `one_sided_bootstrap_pvalue`: turns an
    ALREADY-GENERATED bootstrap-mean sample into the one-sided p-value,
    without drawing any new resamples. Factored out so `evaluate_family_gate`
    can derive the p-value and (later, at Holm's assigned tail) the lower
    bound from the SAME retained sample instead of two independent bootstrap
    draws — see `_lower_bound_from_sorted_boot_means` and the round-4
    adversarial-review finding at `evaluate_family_gate`'s call sites for why
    two independent draws break the percentile duality the module claims.
    """
    p = sum(1 for m in boot_means if m <= tau_log) / len(boot_means)
    return max(p, 1.0 / b)


def one_sided_bootstrap_pvalue(
    values: Sequence[float],
    order_ab: Sequence[bool],
    tau_log: float,
    b: int,
    rng: random.Random,
) -> float:
    """One-sided percentile-bootstrap p-value for H0: mean_slowdown <= tau vs
    Ha: mean_slowdown > tau. `p = P*(boot_mean <= tau)`, floored at `1/b` so a
    genuinely-zero empirical count never makes a downstream Holm correction
    trivially always-reject.
    """
    boot_means = order_stratified_bootstrap_means(values, order_ab, b, rng)
    return _pvalue_from_boot_means(boot_means, tau_log, b)


def _diagnostic_bootstrap_upper_bound(
    values: Sequence[float],
    order_ab: Sequence[bool],
    b: int,
    rng: random.Random,
    *,
    corrected_alpha: float,
) -> float:
    """DIAGNOSTIC-ONLY. The `(1 - corrected_alpha)` percentile of the
    bootstrap mean distribution — a two-sided-diagnostic UPPER confidence
    bound, useful for reporting/inspection only. This is NOT the bound the
    FAIL decision consumes: `evaluate_family_gate`'s FAIL rule needs a
    one-sided LOWER bound (a slowdown is positive and the claimed
    hypothesis is `mean_slowdown > threshold`; confirming FAIL requires the
    LOWER bound to itself exceed the threshold).

    NAMING IS THE GUARD (bench-gate math audit, finding #8): this function
    used to be named `bootstrap_upper_bound`, public, with the same call
    signature as `bootstrap_lower_bound` — a one-character-different name
    that round 1's original defect (see `bootstrap_lower_bound`'s
    docstring) and a live `bench_cpu_flagship_supervisor.py` call site
    (fixed alongside this rename — it was assigning this function's result
    to a variable literally named `corrected_lower_bound`) both independently
    mis-wired into a FAIL/lower-bound role. The leading underscore plus the
    `_DIAGNOSTIC` name are deliberate: this symbol must never look like a
    drop-in replacement for `bootstrap_lower_bound` at a glance, and must
    never be assigned to anything named `*lower_bound*` or wired into a FAIL
    rule or `CellAggregate.corrected_lower_bound`. Use `bootstrap_lower_bound`
    for that.
    """
    if not (0.0 < corrected_alpha < 1.0):
        raise GateMathError(f"corrected_alpha must be in (0, 1), got {corrected_alpha!r}")
    boot_means = sorted(order_stratified_bootstrap_means(values, order_ab, b, rng))
    idx = min(len(boot_means) - 1, math.ceil((1.0 - corrected_alpha) * len(boot_means)) - 1)
    idx = max(idx, 0)
    return boot_means[idx]


def _lower_bound_from_sorted_boot_means(
    sorted_boot_means: Sequence[float], *, corrected_alpha: float
) -> float:
    """Extraction-only half of `bootstrap_lower_bound`: turns an
    ALREADY-GENERATED, ALREADY-SORTED bootstrap-mean sample into the
    `corrected_alpha`-percentile lower bound, without drawing any new
    resamples. See `_pvalue_from_boot_means` for why this split exists.
    """
    if not (0.0 < corrected_alpha < 1.0):
        raise GateMathError(f"corrected_alpha must be in (0, 1), got {corrected_alpha!r}")
    idx = min(len(sorted_boot_means) - 1, math.ceil(corrected_alpha * len(sorted_boot_means)) - 1)
    idx = max(idx, 0)
    return sorted_boot_means[idx]


def bootstrap_lower_bound(
    values: Sequence[float],
    order_ab: Sequence[bool],
    b: int,
    rng: random.Random,
    *,
    corrected_alpha: float,
) -> float:
    """The `corrected_alpha` percentile of the bootstrap mean distribution
    — the corrected ONE-SIDED LOWER confidence bound DESIGN.md's FAIL rule
    actually needs alongside the p-value ("FAIL requires both effect size
    and corrected confidence bound cross its threshold"). A slowdown is
    positive and the claimed hypothesis is `mean_slowdown > threshold`; a
    confirmed FAIL needs a LOWER one-sided bound above the threshold, not
    an upper one (`_diagnostic_bootstrap_upper_bound` computes the wrong
    tail for this purpose — see its docstring). The percentile-interval lower endpoint
    sits at the alpha percentile, not `1 - alpha`
    (https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=917303).

    This is a standalone-resample convenience wrapper; `evaluate_family_gate`
    does NOT call this function (it would draw a second, independent
    bootstrap distribution from the one `one_sided_bootstrap_pvalue` already
    drew for the same cell, breaking percentile duality between the p-value
    and the bound — the round-4 adversarial-review finding). Instead it
    calls `_lower_bound_from_sorted_boot_means` directly on the SAME sample
    it already generated for the p-value.
    """
    if not (0.0 < corrected_alpha < 1.0):
        raise GateMathError(f"corrected_alpha must be in (0, 1), got {corrected_alpha!r}")
    boot_means = sorted(order_stratified_bootstrap_means(values, order_ab, b, rng))
    return _lower_bound_from_sorted_boot_means(boot_means, corrected_alpha=corrected_alpha)


# --------------------------------------------------------------------------
# Holm step-down correction
# --------------------------------------------------------------------------


def holm_reject(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Standard Holm step-down correction. Returns a same-length list of
    reject flags (True = this hypothesis's cell is a statistically
    confirmed FAIL under familywise alpha), aligned to the INPUT order of
    `pvalues` (not the internal sort order)."""
    m = len(pvalues)
    if m == 0:
        raise GateMathError("holm_reject requires at least one p-value")
    order = sorted(range(m), key=lambda i: pvalues[i])
    reject = [False] * m
    still_rejecting = True
    for k, i in enumerate(order):
        crit = alpha / (m - k)
        if still_rejecting and pvalues[i] <= crit:
            reject[i] = True
        else:
            still_rejecting = False
    return reject


def _holm_tested_tails(pvalues: Sequence[float], alpha: float) -> tuple[list[bool], list[float]]:
    """Internal: replicates `holm_reject`'s own ascending-p-value order and
    step-down stopping rule to determine, for each cell in INPUT order,
    whether the executed procedure actually evaluated that rank's `alpha /
    (m - k)` tail before the first failed comparison halted it, and that
    tail itself (`0.0` where untested -- callers must gate on the `tested`
    flag, never use an untested tail).

    The rank at which the procedure stops (its first failed comparison) WAS
    tested -- it just did not reject. Only ranks STRICTLY AFTER that one
    were never evaluated at their less-stringent tail; `holm_reject` never
    ran a comparison for them, so no bound computed "at their tail" is an
    executed Holm bound. Deliberately duplicates `holm_reject`'s loop
    (rather than calling it) so a test that mocks/mutates the public
    `holm_reject` symbol cannot also change which ranks this function
    reports as tested -- testedness is a property of the real step-down
    order, independent of whatever reject-decision function a caller
    plugs in.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    tested = [False] * m
    tail = [0.0] * m
    still_rejecting = True
    for k, i in enumerate(order):
        if not still_rejecting:
            break
        crit = alpha / (m - k)
        tested[i] = True
        tail[i] = crit
        if pvalues[i] > crit:
            still_rejecting = False
    return tested, tail


# --------------------------------------------------------------------------
# Correction 3: exact Clopper-Pearson bound, reported not asserted
# --------------------------------------------------------------------------


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), via direct summation (n is small
    enough here — a few hundred sessions at most — that this is exact and
    fast without needing the incomplete-beta machinery scipy provides)."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
    return total


def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson upper confidence bound on a Bernoulli
    rate, given `k` observed failures in `n` trials, at confidence `conf`.

    Computed by bisection on `P(X <= k; n, p) = 1 - conf` (this CDF is
    monotonically non-increasing in `p` for fixed `k, n`, so bisection is
    well-posed) rather than `scipy.stats.beta.ppf`, keeping this module
    stdlib-only. `k == n` (every session failed) has no finite bound below
    1.0 and returns 1.0 directly.
    """
    if n <= 0:
        raise GateMathError("clopper_pearson_upper requires n >= 1")
    if k < 0 or k > n:
        raise GateMathError(f"clopper_pearson_upper requires 0 <= k <= n, got k={k} n={n}")
    if not (0.0 < conf < 1.0):
        raise GateMathError(f"conf must be in (0, 1), got {conf!r}")
    if k == n:
        return 1.0

    alpha = 1.0 - conf
    lo, hi = 0.0, 1.0
    for _ in range(100):  # bisection to well below float64 precision
        mid = (lo + hi) / 2
        if _binomial_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------
# Correction 1: measured-CV -> required-n policy lookup
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CvBand:
    max_cv: float
    required_n_class_a: int
    required_n_class_b: int
    fail_margin_multiplier: float
    note: str = ""


def parse_cv_bands(raw_bands: Sequence[dict]) -> list[CvBand]:
    if not raw_bands:
        raise PolicyConfigError("perf-policy.toml: [[cv_bands]] must define at least one band")
    bands: list[CvBand] = []
    prev_max = 0.0
    for i, raw in enumerate(raw_bands):
        for key in ("max_cv", "required_n_class_a", "required_n_class_b", "fail_margin_multiplier"):
            if key not in raw:
                raise PolicyConfigError(f"perf-policy.toml: cv_bands[{i}] missing required key {key!r}")
        max_cv = raw["max_cv"]
        if not isinstance(max_cv, (int, float)) or isinstance(max_cv, bool) or max_cv <= 0:
            raise PolicyConfigError(f"perf-policy.toml: cv_bands[{i}] max_cv must be a positive number")
        if max_cv <= prev_max:
            raise PolicyConfigError(
                f"perf-policy.toml: cv_bands must have strictly increasing max_cv, "
                f"band[{i}] max_cv={max_cv} <= previous {prev_max}"
            )
        prev_max = max_cv
        for n_key in ("required_n_class_a", "required_n_class_b"):
            n_val = raw[n_key]
            if isinstance(n_val, bool) or not isinstance(n_val, int) or n_val < 1:
                raise PolicyConfigError(f"perf-policy.toml: cv_bands[{i}] {n_key} must be a positive int")
        margin = raw["fail_margin_multiplier"]
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin < 1.0:
            raise PolicyConfigError(
                f"perf-policy.toml: cv_bands[{i}] fail_margin_multiplier must be a number >= 1.0"
            )
        bands.append(
            CvBand(
                max_cv=float(max_cv),
                required_n_class_a=raw["required_n_class_a"],
                required_n_class_b=raw["required_n_class_b"],
                fail_margin_multiplier=float(margin),
                note=raw.get("note", ""),
            )
        )
    if bands[-1].max_cv < 1.0:
        raise PolicyConfigError(
            f"perf-policy.toml: cv_bands must cover up to CV=1.0 (100%) with a catch-all band, "
            f"last band tops out at {bands[-1].max_cv} -- an unbounded/uncovered CV must never "
            f"silently fall through to 'no band matched'"
        )
    return bands


def required_n(measured_cv: float, cv_bands: Sequence[CvBand], cell_class: str) -> tuple[int, float]:
    """Look up `(required_n, fail_margin_multiplier)` for a measured CV and
    cell class ("A" or "B"). The FIRST band whose `max_cv >= measured_cv`
    wins (bands are pre-sorted ascending by `parse_cv_bands`). Raises
    `GateMathError` for a negative CV or an unrecognized class — this
    function is never asked to guess in the absence of a measured CV; the
    CALLER (the run-record validator) is responsible for refusing
    promotion of any cell lacking a `measured_cv` record in the first
    place (correction 1's fail-closed requirement)."""
    if measured_cv < 0:
        raise GateMathError(f"measured_cv must be >= 0, got {measured_cv!r}")
    if cell_class not in ("A", "B"):
        raise GateMathError(f"cell_class must be 'A' or 'B', got {cell_class!r}")
    for band in cv_bands:
        if measured_cv <= band.max_cv:
            return (
                (band.required_n_class_a, band.fail_margin_multiplier)
                if cell_class == "A"
                else (band.required_n_class_b, band.fail_margin_multiplier)
            )
    raise GateMathError(
        f"measured_cv={measured_cv!r} exceeds every registered cv_bands upper bound "
        f"(highest is {cv_bands[-1].max_cv!r}) -- perf-policy.toml's catch-all band should have "
        "prevented this; treat as a policy-config bug, not a valid cell"
    )


def resolve_metric_policy(policy_doc: dict, metric_family: str, metric_name: str) -> dict:
    """Look up the registered per-metric policy table for
    `(metric_family, metric_name)` in a loaded `perf-policy.toml` document
    (`load_policy`'s return value). `[[families]]` entries come in two
    shapes: a FLAT family table carrying its own `noise_class`/`warn_pct`/
    `fail_pct` plus a `metrics` list of the metric names it covers (e.g.
    `[families.decode]`), or a NESTED family table keyed by metric name,
    each sub-table carrying its own `noise_class`/`warn_pct`/`fail_pct`
    (e.g. `[families.prefill_ttft.ttft]`). This is the registered lookup
    a validator MUST use to derive a cell's noise class -- never a
    family-name heuristic: an earlier revision used a hard-coded
    `family in (...)` shortcut that silently mis-classified
    `embed.batch_p95` as class A when the policy registers it class B.

    Raises `PolicyConfigError` when the family or metric is not
    registered, or the resolved table has no valid `noise_class` -- an
    unregistered metric can never be assumed class A (the cheapest,
    least-scrutinized band). Also validates that a table declaring
    `fail_abs_mib` (the absolute-MiB AND-gate leg, e.g.
    `families.memory.warm_peak_phys_footprint`) registers it as a positive
    number -- structurally the same fail-closed treatment `parse_cv_bands`
    gives every other numeric threshold field (bench-gate math audit
    finding #2). `fail_abs_mib` is otherwise optional; its absence means
    the metric has no absolute-floor leg at all.
    """
    families = policy_doc.get("families")
    if not isinstance(families, dict):
        raise PolicyConfigError("perf-policy.toml: [families] must be a table")
    entry = families.get(metric_family)
    if not isinstance(entry, dict):
        raise PolicyConfigError(
            f"perf-policy.toml: family {metric_family!r} is not registered under [families] "
            "-- an unregistered family can never be assumed a default noise class"
        )
    if "metrics" in entry:
        metrics = entry.get("metrics")
        if not isinstance(metrics, list) or metric_name not in metrics:
            raise PolicyConfigError(
                f"perf-policy.toml: metric {metric_name!r} is not registered under family "
                f"{metric_family!r} (registered metrics: {metrics!r})"
            )
        table = entry
    else:
        table = entry.get(metric_name)
        if not isinstance(table, dict):
            raise PolicyConfigError(
                f"perf-policy.toml: metric {metric_name!r} is not registered under family "
                f"{metric_family!r}"
            )
    noise_class = table.get("noise_class")
    if noise_class not in CELL_CLASSES:
        raise PolicyConfigError(
            f"perf-policy.toml: family {metric_family!r} metric {metric_name!r} has no valid "
            f"noise_class (must be one of {CELL_CLASSES}, got {noise_class!r})"
        )
    if "fail_abs_mib" in table:
        fail_abs_mib = table["fail_abs_mib"]
        if isinstance(fail_abs_mib, bool) or not isinstance(fail_abs_mib, (int, float)) or fail_abs_mib <= 0:
            raise PolicyConfigError(
                f"perf-policy.toml: family {metric_family!r} metric {metric_name!r} declares "
                f"fail_abs_mib={fail_abs_mib!r}, must be a positive number"
            )
    return table


# --------------------------------------------------------------------------
# Per-family gate evaluator (Holm + fail_margin_multiplier, integrated)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellGateInput:
    """One required cell's raw paired log-slowdown samples plus its
    registered family-policy thresholds (already resolved via
    `resolve_metric_policy`), the input `evaluate_family_gate` needs to
    reach a PASS/WARN/FAIL verdict for that cell within its family.

    `fail_abs_mib`/`abs_delta_mib` (bench-gate math audit finding #2):
    some policy-registered metrics (e.g.
    `families.memory.warm_peak_phys_footprint`) declare an explicit AND of
    a relative threshold (`fail_pct`, evaluated via the paired log-slowdown
    bootstrap below) and an absolute-MiB floor
    (`perf-policy.toml`'s `fail_abs_mib`) -- "FAIL requires both >5% AND
    >64 MiB." `values`/`order_ab` carry only the RELATIVE (log-ratio)
    samples the bootstrap needs; they cannot express an absolute-MiB
    delta, so a cell whose policy declares `fail_abs_mib` must also supply
    the already-computed absolute delta (candidate-minus-base, in MiB) via
    `abs_delta_mib`. Both default to `None` (no absolute-floor leg,
    matching every family that does not declare one) -- `fail_abs_mib`
    without a matching `abs_delta_mib` fails closed in
    `evaluate_family_gate` rather than silently skipping the AND
    condition."""

    cell_id: str
    values: tuple[float, ...]
    order_ab: tuple[bool, ...]
    measured_cv: float
    cell_class: str  # "A" or "B" -- selects the cv_bands column
    warn_pct: float
    fail_pct: float
    fail_abs_mib: float | None = None
    abs_delta_mib: float | None = None


@dataclass(frozen=True)
class CellGateResult:
    """One cell's gate-evaluator verdict, alongside the numbers that
    produced it (queryable/loggable without re-running the bootstrap).

    `corrected_lower_bound` is `None` when the Holm step-down procedure
    never reached this cell's rank (it stopped at an earlier, more
    significant rank first) -- there is no executed-at-this-tail bound to
    report, and a cell in that state can never reach FAIL regardless of
    its `holm_reject` flag (see `evaluate_family_gate`)."""

    cell_id: str
    point_estimate: float
    p_value: float
    corrected_lower_bound: float | None
    holm_reject: bool
    measured_cv: float
    required_n: int
    fail_margin_multiplier: float
    verdict: str  # "PASS" | "WARN" | "FAIL"
    fail_abs_mib: float | None = None
    abs_delta_mib: float | None = None


def evaluate_family_gate(
    cells: Sequence[CellGateInput],
    cv_bands: Sequence[CvBand],
    *,
    alpha_familywise: float = 0.05,
    bootstrap_replicates: int = 2000,
    rng: random.Random | None = None,
) -> list[CellGateResult]:
    """The complete per-family gate evaluator: computes a one-sided
    bootstrap p-value per required cell against that cell's own
    (CV-band-margin-scaled) FAIL threshold, Holm-corrects the p-values
    across every cell in the SAME required family at `alpha_familywise`
    (never per-cell alpha -- an uncorrected per-cell 0.05 across N
    required cells inflates the familywise false-FAIL rate to
    `1 - (1 - 0.05)**N`), and combines the Holm-reject decision with the
    corrected one-sided LOWER bound (`bootstrap_lower_bound`, never
    `_diagnostic_bootstrap_upper_bound` -- see that function's docstring)
    to reach a verdict. FAIL additionally requires the cell's declared
    absolute-delta floor (`fail_abs_mib`, when the policy registers one) to
    clear, alongside the relative bound above -- see the `fail_abs_mib`/
    `abs_delta_mib` handling below (bench-gate math audit finding #2).

    FAIL requires ALL of:
      - Holm rejects H0 for this cell at `alpha_familywise` across the
        family, AND
      - that SAME cell's own Holm-assigned step-down tail (`alpha /
        (m - rank)`, the exact level `holm_reject` tested it at, never a
        single alpha/m Bonferroni tail fixed across the whole family)
        exceeds `fail_pct * fail_margin_multiplier` (the high-CV band
        widens the margin instead of inflating `n` further -- correction
        1). Fixing every cell's bound at the strictest step-0 tail would
        make Holm decision-inert (its reject flag could never be the
        thing separating FAIL from non-FAIL, since a fixed, uniformly
        stricter bound already gates every cell); the bound and the
        decision must be the SAME test, evaluated at the SAME per-cell
        alpha.
      - A cell's rank is only assigned a bound if Holm's step-down
        procedure actually EXECUTED a comparison at that rank. The
        step-down stops at the first failed comparison (`holm_reject`);
        every rank strictly after that stop was never tested at its
        `alpha / (m - k)` tail, so `corrected_lower_bound` is `None` for
        those cells -- reporting a bound at an untested, less-stringent
        tail would expose FAIL-level-looking evidence for a hypothesis
        the procedure never actually evaluated at that level, silently
        disagreeing with its own `holm_reject` flag. A cell with a `None`
        bound can never reach FAIL, regardless of its `holm_reject` value.
    Anything short of both, but with a point estimate past `warn_pct` (or
    past `fail_pct` without confirmed statistical significance), is WARN
    -- DESIGN.md: "a point estimate crossing with an inconclusive bound
    is WARN, never PASS." Everything else is PASS.

    Fails closed (raises `GateMathError`) before any resampling if a
    cell's raw pair count is short of its own policy-derived `required_n`
    or contains a non-finite value -- this mirrors
    `bench_decode_harness.validate_run_record`'s INFRA-FAIL treatment of
    the same two conditions on an already-aggregated record, so a cell
    can never reach a PASS/WARN/FAIL verdict on malformed or
    under-sampled raw evidence via either entry point.

    Order-preserving: `results[i]` corresponds to `cells[i]`, matching
    `holm_reject`'s own input-order-preserving contract.

    Percentile duality (round-4 adversarial-review fix): the p-value and the
    lower bound for a given cell are extracted from ONE retained
    bootstrap-mean sample (`order_stratified_bootstrap_means` is called
    exactly once per cell, up front), never from two independent bootstrap
    draws. Drawing the bound from a second, independent resample -- the
    prior shape of this function -- does not preserve the docstring's "SAME
    test, evaluated at the SAME per-cell alpha" claim: a tested-but-not-
    rejected cell could still expose a threshold-clearing bound purely from
    resampling noise between the two draws (round-4 finding: single cell,
    `base=log(1.07)+0.005`, `spread=0.02`, `bootstrap_replicates=2000`,
    `random.Random(108)` gave `p_value=0.051` (correctly not rejected) but
    an independently-drawn `corrected_lower_bound=0.0680872...`, above the
    7% threshold `log(1.07)=0.06765864847381486`). With one shared sample,
    `p_final > corrected_alpha` (not rejected) algebraically forces the
    `corrected_alpha`-percentile bound to sit at or below tau: not-rejected
    means `p_final = max(count(<=tau)/b, 1/b) > corrected_alpha`, so
    (assuming the floor did not fire, guaranteed below) MORE than
    `corrected_alpha * b` of the shared sample's means are already `<= tau`
    -- at least as many as the bound's own rank `ceil(corrected_alpha *
    b)`, so the bound (that rank's order statistic) is itself `<= tau`.

    This duality only holds if the p-value floor (`max(p, 1/b)`, module
    docstring correction 2) never sits ABOVE the tightest tail Holm will
    actually test (`alpha_familywise / len(cells)`, at step-down rank 0):
    a floored p-value that exceeds that tail could report not-rejected for
    a cell whose true (unfloored, zero-count) bootstrap distribution is
    entirely past tau -- which the shared sample would then also report as
    a bound above tau. Rather than let that reintroduce the same class of
    incoherence, this function fails closed (before any resampling) when
    `bootstrap_replicates` is too small for the family size at
    `alpha_familywise` to guarantee the floor can never sit above that
    tightest tail.
    """
    if not cells:
        raise GateMathError("evaluate_family_gate requires at least one cell")
    if rng is None:
        rng = random.Random()

    smallest_tail = alpha_familywise / len(cells)
    if bootstrap_replicates * smallest_tail < 1.0:
        min_replicates = math.ceil(1.0 / smallest_tail)
        raise GateMathError(
            f"bootstrap_replicates={bootstrap_replicates} is too low for a family of {len(cells)} "
            f"cell(s) at alpha_familywise={alpha_familywise}: the smallest tail Holm's step-down "
            f"will ever test is alpha_familywise/{len(cells)}={smallest_tail!r}, and the p-value "
            "floor `max(p, 1/b)` can only resolve tails of size >= 1/bootstrap_replicates. Fewer "
            f"than {min_replicates} replicates lets the floor sit ABOVE that tightest tail, so a "
            "cell whose true (unfloored) bootstrap count at that tail is genuinely zero -- every "
            "bootstrap mean already past tau -- would be floored to a p-value that reports "
            "not-rejected while its own shared-sample bound still clears the fail threshold, "
            "reintroducing the exact p-value/bound incoherence this function otherwise fixes for "
            "every executed rank. Increase bootstrap_replicates, or evaluate fewer cells per family."
        )

    prelim: list[tuple[CellGateInput, int, float, float, float, float, list[float]]] = []
    for cell in cells:
        if len(cell.values) != len(cell.order_ab):
            raise GateMathError(f"cell {cell.cell_id!r}: values and order_ab must be the same length")
        req_n, margin = required_n(cell.measured_cv, cv_bands, cell.cell_class)
        if len(cell.values) < req_n:
            raise GateMathError(
                f"cell {cell.cell_id!r}: only {len(cell.values)} raw pairs, requires >= {req_n} "
                f"at measured_cv={cell.measured_cv!r} class {cell.cell_class!r} -- an under-sampled cell "
                "fails closed before any resampling, never PASSes on insufficient evidence"
            )
        if any(not math.isfinite(v) for v in cell.values):
            raise GateMathError(
                f"cell {cell.cell_id!r}: raw values contain a non-finite entry -- a malformed cell "
                "fails closed before any resampling, never reaches a PASS/WARN/FAIL verdict"
            )
        if cell.fail_abs_mib is not None:
            if not math.isfinite(cell.fail_abs_mib) or cell.fail_abs_mib <= 0:
                raise GateMathError(
                    f"cell {cell.cell_id!r}: fail_abs_mib must be a positive finite number, "
                    f"got {cell.fail_abs_mib!r}"
                )
            if cell.abs_delta_mib is None:
                raise GateMathError(
                    f"cell {cell.cell_id!r}: declares fail_abs_mib={cell.fail_abs_mib!r} (an AND-gate "
                    "absolute-MiB floor) but supplies no abs_delta_mib -- fails closed rather than "
                    "silently evaluating only the relative leg of a declared two-condition AND rule "
                    "(bench-gate math audit finding #2)"
                )
            if not math.isfinite(cell.abs_delta_mib):
                raise GateMathError(
                    f"cell {cell.cell_id!r}: abs_delta_mib must be finite, got {cell.abs_delta_mib!r}"
                )
        tau_fail = math.log(1.0 + cell.fail_pct * margin)
        point_estimate = statistics.fmean(cell.values)
        # ONE retained bootstrap-mean sample per cell -- both the p-value
        # (now) and, if this rank is tested by the real step-down (below),
        # the lower bound (later) are extracted from THIS SAME sample. Never
        # call `one_sided_bootstrap_pvalue`/`bootstrap_lower_bound` here:
        # each draws its own independent sample from `rng`, which is exactly
        # the percentile-duality bug this restructure fixes.
        boot_means = order_stratified_bootstrap_means(
            cell.values, cell.order_ab, bootstrap_replicates, rng
        )
        p = _pvalue_from_boot_means(boot_means, tau_fail, bootstrap_replicates)
        prelim.append((cell, req_n, margin, tau_fail, point_estimate, p, sorted(boot_means)))

    pvalues = [p for *_rest, p, _boot in prelim]
    rejects = holm_reject(pvalues, alpha=alpha_familywise)
    # The exact step-down tail each cell was ACTUALLY TESTED at in the real
    # Holm procedure -- `tested[i]` is False for any rank strictly after the
    # step-down's first failed comparison (see `_holm_tested_tails`).
    # Computed independently of `rejects` above (never derived from a
    # caller-supplied/mocked `holm_reject`) so testedness always reflects
    # the real step-down order.
    tested, cell_alpha = _holm_tested_tails(pvalues, alpha_familywise)

    results: list[CellGateResult] = []
    for (
        cell,
        req_n,
        margin,
        tau_fail,
        point_estimate,
        p,
        sorted_boot_means,
    ), reject, corrected_alpha, was_tested in zip(prelim, rejects, cell_alpha, tested):
        lower_bound: float | None = None
        if was_tested:
            # Same retained sample as the p-value above -- NOT a fresh
            # `bootstrap_lower_bound` draw (see the docstring's percentile-
            # duality note and that function's own docstring).
            lower_bound = _lower_bound_from_sorted_boot_means(
                sorted_boot_means, corrected_alpha=corrected_alpha
            )
        tau_warn = math.log(1.0 + cell.warn_pct)
        # AND-gate (bench-gate math audit finding #2): a cell whose policy
        # declares fail_abs_mib must ALSO clear the absolute-MiB floor to
        # FAIL, matching perf-policy.toml's declared "FAIL requires both
        # >X% AND >Y MiB" rule -- the relative bound crossing tau_fail is
        # necessary but no longer sufficient when an absolute floor is
        # registered. `abs_ok` is vacuously True when the cell's policy
        # declares no absolute floor at all (fail_abs_mib is None).
        abs_ok = cell.fail_abs_mib is None or (
            cell.abs_delta_mib is not None and abs(cell.abs_delta_mib) > cell.fail_abs_mib
        )
        if reject and lower_bound is not None and lower_bound > tau_fail and abs_ok:
            verdict = "FAIL"
        elif point_estimate > tau_warn:
            verdict = "WARN"
        else:
            verdict = "PASS"
        results.append(
            CellGateResult(
                cell_id=cell.cell_id,
                point_estimate=point_estimate,
                p_value=p,
                corrected_lower_bound=lower_bound,
                holm_reject=reject,
                measured_cv=cell.measured_cv,
                required_n=req_n,
                fail_margin_multiplier=margin,
                verdict=verdict,
                fail_abs_mib=cell.fail_abs_mib,
                abs_delta_mib=cell.abs_delta_mib,
            )
        )
    return results


# --------------------------------------------------------------------------
# Policy file loading
# --------------------------------------------------------------------------


def load_policy(path: Path = DEFAULT_POLICY_FILE) -> dict:
    """Load and structurally validate `perf-policy.toml`. Returns the raw
    parsed dict (callers pull specific sub-tables); `parse_cv_bands` is
    applied separately since most callers only need the CV-band lookup."""
    try:
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise PolicyConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError as exc:
        raise PolicyConfigError(f"{path}: could not read policy file: {exc}") from exc

    for key in ("policy_version", "cv_bands", "families", "promotion", "gate_math"):
        if key not in doc:
            raise PolicyConfigError(f"{path}: missing required top-level key {key!r}")
    if isinstance(doc["policy_version"], bool) or not isinstance(doc["policy_version"], int):
        raise PolicyConfigError(f"{path}: policy_version must be an int")
    parse_cv_bands(doc["cv_bands"])  # raises PolicyConfigError on structural issues
    return doc


def policy_sha(path: Path = DEFAULT_POLICY_FILE) -> str:
    """SHA-256 of the raw policy file bytes — the value a `ProvenanceRecord`
    pins so a later re-validation can detect a post-run threshold change
    (the policy content changed under the same or a different
    `policy_version` after the run was gated)."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()

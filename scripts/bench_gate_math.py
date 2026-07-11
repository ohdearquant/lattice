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
    p = sum(1 for m in boot_means if m <= tau_log) / len(boot_means)
    return max(p, 1.0 / b)


def bootstrap_upper_bound(
    values: Sequence[float],
    order_ab: Sequence[bool],
    b: int,
    rng: random.Random,
    *,
    corrected_alpha: float,
) -> float:
    """The `(1 - corrected_alpha)` percentile of the bootstrap mean
    distribution — a two-sided-diagnostic UPPER confidence bound, useful
    for reporting/inspection only. This is NOT the bound the FAIL decision
    consumes: `evaluate_family_gate`'s FAIL rule needs a one-sided LOWER
    bound (a slowdown is positive and the claimed hypothesis is
    `mean_slowdown > threshold`; confirming FAIL requires the LOWER bound
    to itself exceed the threshold). Do not wire this function's result
    into `CellAggregate.corrected_lower_bound` or any FAIL rule — use
    `bootstrap_lower_bound` for that; an earlier revision of this module
    conflated the two tails, which silently overstated regression
    evidence.
    """
    if not (0.0 < corrected_alpha < 1.0):
        raise GateMathError(f"corrected_alpha must be in (0, 1), got {corrected_alpha!r}")
    boot_means = sorted(order_stratified_bootstrap_means(values, order_ab, b, rng))
    idx = min(len(boot_means) - 1, math.ceil((1.0 - corrected_alpha) * len(boot_means)) - 1)
    idx = max(idx, 0)
    return boot_means[idx]


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
    an upper one (`bootstrap_upper_bound` computes the wrong tail for this
    purpose — see its docstring). The percentile-interval lower endpoint
    sits at the alpha percentile, not `1 - alpha`
    (https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=917303).

    This is the function `evaluate_family_gate` calls to populate
    `CellGateResult.corrected_lower_bound`, which is in turn what a
    validator must require to exceed the (margin-scaled) fail threshold
    before confirming FAIL.
    """
    if not (0.0 < corrected_alpha < 1.0):
        raise GateMathError(f"corrected_alpha must be in (0, 1), got {corrected_alpha!r}")
    boot_means = sorted(order_stratified_bootstrap_means(values, order_ab, b, rng))
    idx = min(len(boot_means) - 1, math.ceil(corrected_alpha * len(boot_means)) - 1)
    idx = max(idx, 0)
    return boot_means[idx]


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
    least-scrutinized band).
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
    return table


# --------------------------------------------------------------------------
# Per-family gate evaluator (Holm + fail_margin_multiplier, integrated)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellGateInput:
    """One required cell's raw paired log-slowdown samples plus its
    registered family-policy thresholds (already resolved via
    `resolve_metric_policy`), the input `evaluate_family_gate` needs to
    reach a PASS/WARN/FAIL verdict for that cell within its family."""

    cell_id: str
    values: tuple[float, ...]
    order_ab: tuple[bool, ...]
    measured_cv: float
    cell_class: str  # "A" or "B" -- selects the cv_bands column
    warn_pct: float
    fail_pct: float


@dataclass(frozen=True)
class CellGateResult:
    """One cell's gate-evaluator verdict, alongside the numbers that
    produced it (queryable/loggable without re-running the bootstrap)."""

    cell_id: str
    point_estimate: float
    p_value: float
    corrected_lower_bound: float
    holm_reject: bool
    measured_cv: float
    required_n: int
    fail_margin_multiplier: float
    verdict: str  # "PASS" | "WARN" | "FAIL"


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
    `bootstrap_upper_bound` -- see that function's docstring) to reach a
    verdict.

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
    """
    if not cells:
        raise GateMathError("evaluate_family_gate requires at least one cell")
    if rng is None:
        rng = random.Random()

    prelim: list[tuple[CellGateInput, int, float, float, float, float]] = []
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
        tau_fail = math.log(1.0 + cell.fail_pct * margin)
        point_estimate = statistics.fmean(cell.values)
        p = one_sided_bootstrap_pvalue(cell.values, cell.order_ab, tau_fail, bootstrap_replicates, rng)
        prelim.append((cell, req_n, margin, tau_fail, point_estimate, p))

    pvalues = [p for *_rest, p in prelim]
    m = len(prelim)
    rejects = holm_reject(pvalues, alpha=alpha_familywise)
    # The exact step-down tail each cell was tested at in `holm_reject`:
    # ascending-p-value rank k (0-indexed) -> alpha/(m-k), matching that
    # function's own `crit = alpha / (m - k)` at each step. Cell-specific,
    # not the fixed alpha/m Bonferroni tail -- see the docstring above.
    rank_order = sorted(range(m), key=lambda i: pvalues[i])
    cell_alpha = [0.0] * m
    for k, i in enumerate(rank_order):
        cell_alpha[i] = alpha_familywise / (m - k)

    results: list[CellGateResult] = []
    for (cell, req_n, margin, tau_fail, point_estimate, p), reject, corrected_alpha in zip(
        prelim, rejects, cell_alpha
    ):
        lower_bound = bootstrap_lower_bound(
            cell.values, cell.order_ab, bootstrap_replicates, rng, corrected_alpha=corrected_alpha
        )
        tau_warn = math.log(1.0 + cell.warn_pct)
        if reject and lower_bound > tau_fail:
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

#!/usr/bin/env python3
"""Parse Criterion change reports and apply the ADR-058 regression gate.

For every Criterion bench under target/criterion/, read change/estimates.json
(produced when running with --baseline <name>). A caller may also supply the
reverse-order half of an ABBA comparison. In that mode the gate combines the
forward and reverse ratios in log space and widens the resulting interval by
the measured order-bias envelope before applying the rule:

  decision lower bound in (-inf, +3%]  : pass silently
  decision lower bound in (+3%, +7%]   : warn (PR-comment only, no fail)
  decision lower bound in (+7%, +inf)  : FAIL
  Point estimate < -3% AND CI-upper<0% : celebrate

The decision bound is Criterion's two-sided-95%-CI lower endpoint in raw
two-arm mode and the order-bias-widened ABBA lower bound in balanced mode.

Usage:
  perf-bench-gate.py <criterion_root> <arch_label> [--out report.md]
  perf-bench-gate.py <criterion_root> --prepare-baseline-copy
  perf-bench-gate.py <criterion_root> --prepare-head
  perf-bench-gate.py <criterion_root> <arch_label> --target <crate>:<bench-target>
                      [--informational-target <crate>:<bench-target>]
  perf-bench-gate.py <criterion_root> <arch_label> --target <crate>:<bench-target>
                      --ambient-samples <jsonl> --status-out <json>
                      --order-control-root <criterion_root>
                      --require-order-balance

Exit codes:
  0 — pass (no gated FAILs)
  1 — at least one gated FAIL (regression > 7% by the selected decision
      lower bound; see the WARN_PCT/FAIL_PCT note below for raw Criterion
      mode, and the ABBA note above for balanced mode)
  2 — parse error / bad input, or (with --require-measurements) the gate
      refusing to certify a run it could not judge: no comparison data, or
      no gating comparison among the parsed results, or a benchmark in the
      selected baseline set with no head comparison, or a head benchmark with
      no selected baseline. Also covers invalid ambient instrumentation for an
      automated lane: the before/between/after sample stream is missing,
      malformed, or duplicated. An automated lane must not read "nothing was
      measured" as "nothing regressed".
  3 — not measurable: the automated lane's before/between/after ambient sample
      stream parsed and was complete, but idle was below BENCH_IDLE_FLOOR in
      at least one phase; or the measured AB/BA order-bias envelope is itself
      larger than the FAIL margin, so the run cannot distinguish a gate-sized
      source effect. This is a non-success state: enforcing consumers must
      keep it nonzero. Across independent targets, a confirmed regression (1)
      outranks this state.

Status mode is opt-in. It requires exactly one perf-ambient-sample/v1 record for
each voting phase (before, between, after), ignores other phase labels, and
writes an atomic perf-bench-gate-status/v1 document. Ambient refusal is checked
before the final exit/status verdict, so exit 3 outranks pass or regression for
every target in the run. Excessive order bias is a second exit-3 cause, and a
gated set left empty by target policy is the third: with nothing able to vote
there is no verdict to render, so the run refuses instead of ticking.

With --phase-load-summaries, status also carries four ordered phase_load entries
and phase_load_verdict (ok or LOUD). This records in-phase contamination without
changing existing verdicts or boundary samples. The fields are absent on legacy
calls without the option; absence is not evidence of quiet in-phase conditions.

--informational-target: bench-compare isolates each bench target in its own
Criterion output root and passes that root's exact key through --target. A
reviewed caller policy may mark that same key informational: the quick-mode
noise-demotion manifest does this for lattice-embed's `simd` target (#714),
and the target/feature calibration allowlist -- which covers both crates -- does
it at either resolution for selected configurations that have not calibrated the
gate. Every result in
an informational root is still measured and reported, but excluded from the
FAIL/WARN verdict. Because the policy is target-wide, demoting the target empties
the gated set: such a run exits 3 with its rows rendered beside the refusal, and
never reports a pass. This classifier validates exact target identity; the caller
owns the policy decision to withhold gating authority.

Run provenance is schema-versioned. New bench-compare runs write
lattice-bench-provenance-v2, where embed_features is required. The reader also
accepts legacy v1, which predates embed feature selection and therefore
normalizes that absent concept to embed_features=<none> for display. A v1 record
that contains embed_features is rejected rather than silently redefining v1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

_RUNNING_PYTHON = ".".join(str(part) for part in sys.version_info[:3])
_PYTHON_REQUIREMENT_ERROR = (
    f"{sys.argv[0]} requires Python 3.11 or newer; "
    f"running Python {_RUNNING_PYTHON} at {sys.executable}"
)
try:
    from datetime import UTC, datetime
except ImportError:
    raise SystemExit(_PYTHON_REQUIREMENT_ERROR) from None
if sys.version_info[:2] < (3, 11):
    raise SystemExit(_PYTHON_REQUIREMENT_ERROR)

PROVENANCE_SCHEMA_V1 = "lattice-bench-provenance-v1"
PROVENANCE_SCHEMA_V2 = "lattice-bench-provenance-v2"
MACHINE_STATE_SCHEMA = "lattice-machine-state-v1"
PHASE_LABELS = ("before first arm", "between order strata", "after final arm")
# lattice#1515: the ABBA measurement arms themselves, sampled IN-PHASE (not
# just at the PHASE_LABELS boundaries above) for foreign ambient load.
PHASE_LOAD_ARMS = ("base1", "head1", "head2", "base2")
PROVENANCE_FIELDS = (
    "started_utc",
    "finished_utc",
    "host_id",
    "os",
    "base_ref",
    "base_sha",
    "head_ref",
    "head_sha",
    "head_mode",
    "base_rustc",
    "head_rustc",
    "base_cargo",
    "head_cargo",
    "base_criterion",
    "head_criterion",
    "criterion_mode",
    "baseline_name",
    "targets",
    "inference_features",
    "embed_features",
    "filters",
    "enforcement",
)
PROVENANCE_V1_FIELDS = tuple(
    field for field in PROVENANCE_FIELDS if field != "embed_features"
)
PROVENANCE_FIELDS_BY_SCHEMA = {
    PROVENANCE_SCHEMA_V1: PROVENANCE_V1_FIELDS,
    PROVENANCE_SCHEMA_V2: PROVENANCE_FIELDS,
}

# Thresholds — ADR-058 §D3. Edit here; the workflow imports nothing else.
#
# Precision note (bench-gate math audit, finding #4): `ci_low`/`ci_high` are
# Criterion's own TWO-SIDED 95% CI endpoints. Using `ci_low` as a one-sided
# FAIL cutoff is directionally sound (a slowdown is a one-sided hypothesis,
# and gating on the lower endpoint of a two-sided interval can only be MORE
# conservative than a properly-computed one-sided 95% bound, never less —
# assuming Criterion's CI is symmetric two-sided at 0.95, `ci_low` sits at
# roughly a 97.5%-one-sided-confidence level, not 95%). This raises the bar
# for FAIL (fewer true positives caught), never lowers it, so it cannot by
# itself produce a false FAIL — but "regression >7% confirmed by 95% CI" in
# this module's docstring/comments overstates precision and should be read
# as "confirmed at approximately a 97.5% one-sided level via the two-sided
# 95% CI's lower bound," not a calibrated one-sided-95% test. Verify against
# the `criterion` crate's own CI-construction source before relying on the
# exact number if it ever matters (not independently checked here).
#
# These thresholds read the order-bias-widened CI LOWER bound, never the point
# estimate the report prints beside it. Measured on an A/A comparison (both arms
# built from identical source, quick resolution, 255 rows on one target): point
# estimates ranged to +7.97% with a widest interval upper bound of +16.74%, while
# the lower bound these constants are compared against crossed 3.0% on one row and
# 7.0% on none. The point column and these constants are differently-computed
# numbers that share a percent sign, so reading a report's point column against
# WARN_PCT/FAIL_PCT overstates what the gate would have decided.
WARN_PCT = 3.0   # CI-lower above this => warning
FAIL_PCT = 7.0   # CI-lower above this => FAIL
CELEBRATE_PCT = -3.0  # point estimate below this AND CI-upper<0 => celebrate
QUICK_UNATTRIBUTABLE = "UNATTRIBUTABLE (quick resolution)"

EXIT_PASS = 0
EXIT_REGRESSION = 1
EXIT_ERROR = 2
EXIT_NOT_MEASURABLE = 3
AMBIENT_SAMPLE_SCHEMA = "perf-ambient-sample/v1"
STATUS_SCHEMA = "perf-bench-gate-status/v1"
REQUIRED_AMBIENT_PHASES = ("before", "between", "after")
DEFAULT_IDLE_FLOOR = 70.0


@dataclass(frozen=True)
class AmbientAssessment:
    samples: dict[str, float]
    floor_pct: float
    outcome: str
    reason: str | None = None

    @property
    def measurable(self) -> bool:
        return self.outcome == "valid"

    @property
    def instrumentation_valid(self) -> bool:
        return self.outcome != "invalid"

    @property
    def minimum_pct(self) -> float | None:
        return min(self.samples.values()) if self.samples else None


def idle_floor_from_env() -> float:
    """Return the shared benchmark idle floor, rejecting unusable values."""
    raw = os.environ.get("BENCH_IDLE_FLOOR", str(DEFAULT_IDLE_FLOOR))
    try:
        floor = float(raw)
    except ValueError as error:
        raise ValueError(
            f"BENCH_IDLE_FLOOR must be numeric, got {raw!r}"
        ) from error
    if not math.isfinite(floor) or not 0.0 <= floor <= 100.0:
        raise ValueError(
            f"BENCH_IDLE_FLOOR must be finite and between 0 and 100, got {raw!r}"
        )
    return floor


def assess_ambient_samples(path: Path | None, floor_pct: float) -> AmbientAssessment:
    """Validate the three voting ambient samples for an automated run."""
    if path is None:
        return AmbientAssessment(
            {}, floor_pct, "invalid", "ambient sample input was not provided"
        )
    try:
        records = [line for line in path.read_text().splitlines() if line.strip()]
    except OSError as error:
        return AmbientAssessment(
            {},
            floor_pct,
            "invalid",
            f"ambient sample input could not be read: {error}",
        )
    if not records:
        return AmbientAssessment(
            {}, floor_pct, "invalid", "ambient sample input was empty"
        )

    samples: dict[str, float] = {}
    for line_number, line in enumerate(records, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            return AmbientAssessment(
                samples,
                floor_pct,
                "invalid",
                f"ambient sample line {line_number} is malformed JSON: {error.msg}",
            )
        if not isinstance(record, dict):
            return AmbientAssessment(
                samples,
                floor_pct,
                "invalid",
                f"ambient sample line {line_number} must be a JSON object",
            )
        if record.get("schema") != AMBIENT_SAMPLE_SCHEMA:
            return AmbientAssessment(
                samples,
                floor_pct,
                "invalid",
                f"ambient sample line {line_number} has an unsupported schema",
            )
        phase = record.get("phase")
        if not isinstance(phase, str) or not phase.strip():
            return AmbientAssessment(
                samples,
                floor_pct,
                "invalid",
                f"ambient sample line {line_number} has no valid phase",
            )
        idle_pct = record.get("idle_pct")
        if (
            isinstance(idle_pct, bool)
            or not isinstance(idle_pct, (int, float))
            or not math.isfinite(float(idle_pct))
            or not 0.0 <= float(idle_pct) <= 100.0
        ):
            return AmbientAssessment(
                samples,
                floor_pct,
                "invalid",
                f"ambient sample line {line_number} has no valid idle_pct",
            )
        phase = phase.strip()
        if phase not in REQUIRED_AMBIENT_PHASES:
            continue
        if phase in samples:
            return AmbientAssessment(
                samples,
                floor_pct,
                "invalid",
                f"ambient sample phase {phase!r} was recorded more than once",
            )
        samples[phase] = float(idle_pct)

    missing = [phase for phase in REQUIRED_AMBIENT_PHASES if phase not in samples]
    if missing:
        return AmbientAssessment(
            samples,
            floor_pct,
            "invalid",
            "ambient sample input is incomplete; missing " + ", ".join(missing),
        )
    busy = [
        f"{phase}={samples[phase]:.1f}%"
        for phase in REQUIRED_AMBIENT_PHASES
        if samples[phase] < floor_pct
    ]
    if busy:
        return AmbientAssessment(
            samples,
            floor_pct,
            "not_measurable",
            f"ambient idle was below the {floor_pct:.1f}% floor at "
            + ", ".join(busy),
        )
    return AmbientAssessment(samples, floor_pct, "valid")


def load_phase_load_summaries(path: Path, floor_pct: float) -> list[dict]:
    """Read a complete ABBA verdict stream; never certify absent instrumentation."""
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read in-phase load summaries: {error}") from error
    if len(records) != len(PHASE_LOAD_ARMS):
        raise ValueError("in-phase load summaries require exactly four ABBA arms")
    summaries = []
    for arm, record in zip(PHASE_LOAD_ARMS, records):
        if (not isinstance(record, dict)
                or record.get("schema") != "perf-phase-load/v1"
                or record.get("arm") != arm):
            raise ValueError(f"invalid in-phase load summary for {arm}")
        foreign = record.get("foreign_max_pct")
        floor = record.get("floor_pct")
        # The implied figure is load the idle reading saw but `ps` could not
        # attribute to a process (a PID namespace hides the host's processes,
        # so foreign reads 0.0 however busy the machine is). It is absent from
        # streams written before the implied rule existed; 0.0 there restores
        # the foreign-only verdict exactly.
        implied = record.get("implied_foreign_max_pct", 0.0)
        try:
            finite_foreign = math.isfinite(foreign)
        except (TypeError, OverflowError):
            finite_foreign = False
        try:
            finite_implied = math.isfinite(implied)
        except (TypeError, OverflowError):
            finite_implied = False
        if (isinstance(foreign, bool) or not isinstance(foreign, (int, float))
                or not finite_foreign or foreign < 0
                or isinstance(implied, bool) or not isinstance(implied, (int, float))
                or not finite_implied or implied < 0
                or isinstance(floor, bool) or not isinstance(floor, (int, float))
                or floor != floor_pct):
            raise ValueError(f"invalid in-phase load figures for {arm}")
        verdict = record.get("verdict")
        ceiling = 100.0 - floor
        if verdict != ("LOUD" if foreign > ceiling or implied > ceiling else "ok"):
            raise ValueError(f"inconsistent in-phase load verdict for {arm}")
        summaries.append({
            "arm": arm,
            "verdict": verdict,
            "foreign_max_pct": foreign,
            "floor_pct": floor,
        })
    return summaries


def write_gate_status(
    path: Path,
    *,
    arch: str,
    target: str | None,
    verdict: str,
    exit_code: int,
    reason: str,
    ambient: AmbientAssessment | None,
    phase_load: list[dict] | None = None,
    measurement_count: int | None = None,
    regression_count: int | None = None,
) -> None:
    """Atomically publish the machine-readable gate status."""
    payload: dict[str, object] = {
        "schema": STATUS_SCHEMA,
        "verdict": verdict,
        "exit_code": exit_code,
        "arch": arch,
        "target": target,
        "reason": reason,
    }
    if ambient is not None:
        payload["ambient"] = {
            "schema": AMBIENT_SAMPLE_SCHEMA,
            "assessment": ambient.outcome,
            "floor_pct": ambient.floor_pct,
            "minimum_idle_pct": ambient.minimum_pct,
            "samples": {
                phase: ambient.samples[phase]
                for phase in REQUIRED_AMBIENT_PHASES
                if phase in ambient.samples
            },
        }
    if measurement_count is not None:
        payload["measurement_count"] = measurement_count
    if regression_count is not None:
        payload["regression_count"] = regression_count
    if phase_load is not None:
        payload["phase_load"] = phase_load
        payload["phase_load_verdict"] = (
            "LOUD" if any(arm["verdict"] == "LOUD" for arm in phase_load) else "ok"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass
class BenchResult:
    name: str            # e.g. "rms_norm/4096"
    point: float         # change.estimates.mean.point_estimate (fraction, not %)
    ci_low: float
    ci_high: float
    new_ns: float        # new median time, nanoseconds
    old_ns: float        # baseline median time, nanoseconds
    head_sample_count: int | None = None
    head_sampling_mode: str | None = None
    base_sample_count: int | None = None
    base_sampling_mode: str | None = None
    order_bias_bound: float | None = None

    @property
    def point_pct(self) -> float: return self.point * 100.0
    @property
    def ci_low_pct(self) -> float: return self.ci_low * 100.0
    @property
    def ci_high_pct(self) -> float: return self.ci_high * 100.0
    @property
    def order_bias_bound_pct(self) -> float | None:
        if self.order_bias_bound is None:
            return None
        return self.order_bias_bound * 100.0
    def is_informational(self, target: str | None, informational_target: str | None) -> bool:
        return target is not None and target == informational_target

    def verdict(self, resolution: str = "full") -> str:
        if self.ci_low_pct > FAIL_PCT:
            return "FAIL"
        if self.ci_low_pct > WARN_PCT:
            if resolution == "quick":
                return QUICK_UNATTRIBUTABLE
            return "WARN"
        if self.point_pct < CELEBRATE_PCT and self.ci_high_pct < 0:
            return "WIN"
        return "PASS"


def _checked_relative_log(value: float, field: str) -> float:
    """Return log(1 + relative change), rejecting malformed Criterion evidence."""
    if not math.isfinite(value) or value <= -1.0:
        raise ValueError(f"{field} must be finite and greater than -1, got {value!r}")
    return math.log1p(value)


def _combined_sample_count(first: int | None, second: int | None) -> int | None:
    if first is None or second is None:
        return None
    return first + second


def _combined_sampling_mode(first: str | None, second: str | None) -> str | None:
    if first is None or second is None:
        return None
    if first == second:
        return first
    return f"{first}+{second}"


def _geometric_mean(first: float, second: float) -> float:
    """Compute a positive geometric mean without overflowing the product."""
    mean = math.exp((math.log(first) + math.log(second)) / 2.0)
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError("order-balanced timing mean is not positive and finite")
    return mean


def order_balance_pair(forward: BenchResult, reverse: BenchResult) -> BenchResult:
    """Combine A→B and B→A Criterion comparisons from an ABBA run.

    `forward` is head₁/base₁ and `reverse` is base₂/head₂. The model is
    `f = s + d_f`, `r = -s + d_r`, where `s` is the source effect and
    `d_f`/`d_r` are within-stratum order increments in log space. The split
    identifies `s` exactly when `d_f == d_r`; that is a stable multiplicative
    timing effect on the raw scale. Widening by the order envelope brackets
    unequal same-sign increments and a disturbance confined to one stratum.

    Opposite-sign or source-dependent increments can cancel in the order term
    and be misattributed to the source. One ABBA block cannot detect that model
    violation; replication, interleaving, or randomized order blocks are needed.
    """
    if forward.name != reverse.name:
        raise ValueError(
            f"order-balance benchmark mismatch: {forward.name!r} != {reverse.name!r}"
        )

    for result_name, result in (("forward", forward), ("reverse", reverse)):
        if not result.ci_low <= result.point <= result.ci_high:
            raise ValueError(
                f"{forward.name}: {result_name} change interval does not contain "
                f"its point estimate"
            )
        for field_name, value in (
            ("new_ns", result.new_ns),
            ("old_ns", result.old_ns),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{forward.name}: {result_name} {field_name} must be positive "
                    f"and finite, got {value!r}"
                )

    f_point = _checked_relative_log(forward.point, "forward point")
    f_low = _checked_relative_log(forward.ci_low, "forward CI lower")
    f_high = _checked_relative_log(forward.ci_high, "forward CI upper")
    r_point = _checked_relative_log(reverse.point, "reverse point")
    r_low = _checked_relative_log(reverse.ci_low, "reverse CI lower")
    r_high = _checked_relative_log(reverse.ci_high, "reverse CI upper")

    effect_point_log = (f_point - r_point) / 2.0
    effect_low_log = (f_low - r_high) / 2.0
    effect_high_log = (f_high - r_low) / 2.0

    order_low_log = (f_low + r_low) / 2.0
    order_high_log = (f_high + r_high) / 2.0
    order_bias_log_bound = max(abs(order_low_log), abs(order_high_log))

    effect_point = math.expm1(effect_point_log)
    effect_low = math.expm1(effect_low_log - order_bias_log_bound)
    effect_high = math.expm1(effect_high_log + order_bias_log_bound)
    order_bias_bound = math.expm1(order_bias_log_bound)
    if not all(
        math.isfinite(value)
        for value in (effect_point, effect_low, effect_high, order_bias_bound)
    ):
        raise ValueError(f"{forward.name}: order-balanced result is not finite")

    return BenchResult(
        name=forward.name,
        point=effect_point,
        ci_low=effect_low,
        ci_high=effect_high,
        new_ns=_geometric_mean(forward.new_ns, reverse.old_ns),
        old_ns=_geometric_mean(forward.old_ns, reverse.new_ns),
        # Forward base is A₁; reverse candidate/new is A₂.
        base_sample_count=_combined_sample_count(
            forward.base_sample_count, reverse.head_sample_count
        ),
        base_sampling_mode=_combined_sampling_mode(
            forward.base_sampling_mode, reverse.head_sampling_mode
        ),
        # Forward head is B₁; reverse baseline/old is B₂.
        head_sample_count=_combined_sample_count(
            forward.head_sample_count, reverse.base_sample_count
        ),
        head_sampling_mode=_combined_sampling_mode(
            forward.head_sampling_mode, reverse.base_sampling_mode
        ),
        order_bias_bound=order_bias_bound,
    )


def order_balance_results(
    forward_results: list[BenchResult],
    reverse_results: list[BenchResult],
) -> list[BenchResult]:
    """Reconcile exact benchmark identities, then order-balance every pair."""
    forward = {result.name: result for result in forward_results}
    reverse = {result.name: result for result in reverse_results}
    if len(forward) != len(forward_results):
        raise ValueError("forward Criterion root contains duplicate benchmark identities")
    if len(reverse) != len(reverse_results):
        raise ValueError("reverse Criterion root contains duplicate benchmark identities")
    if forward.keys() != reverse.keys():
        missing = sorted(forward.keys() - reverse.keys())
        extra = sorted(reverse.keys() - forward.keys())
        raise ValueError(
            "order-control benchmark set differs from the forward set: "
            f"missing={missing}, extra={extra}"
        )
    return [
        order_balance_pair(forward[name], reverse[name])
        for name in sorted(forward)
    ]

@dataclass(frozen=True)
class RunProvenance:
    schema: str
    fields: dict[str, str]
    locks: tuple[str, ...]
    ambient_samples: tuple[str, ...]
    machine_states: tuple[dict[str, object], ...]
    phase_load_samples: tuple[str, ...] = ()


def parse_utc_timestamp(value: str, field: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError(f"{field} must be a UTC timestamp ending in 'Z'")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{field} is not a valid ISO-8601 timestamp: {value!r}") from error
    if parsed.tzinfo != UTC:
        raise ValueError(f"{field} must be UTC")
    return parsed


def validate_capability(
    value: object,
    field: str,
    measured_states: frozenset[str],
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    status = value.get("status")
    if status == "unavailable":
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{field} unavailable state must name a reason")
        return
    if status != "measured":
        raise ValueError(f"{field} status must be 'measured' or 'unavailable'")
    state = value.get("state")
    if state not in measured_states:
        allowed = ", ".join(sorted(measured_states))
        raise ValueError(f"{field} measured state must be one of: {allowed}")
    source = value.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError(f"{field} measured state must name its source")


def validate_idle(value: object, field: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    if value.get("status") == "unavailable":
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{field} unavailable state must name a reason")
        return
    if value.get("status") != "measured":
        raise ValueError(f"{field} status must be 'measured' or 'unavailable'")
    source = value.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError(f"{field} measured state must name its source")
    seconds = value.get("seconds")
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds < 0
    ):
        raise ValueError(f"{field} seconds must be a finite non-negative number")


def validate_checkpoint_gate(value: object, field: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    status = value.get("status")
    if status not in ("passed", "blocked"):
        raise ValueError(f"{field} status must be 'passed' or 'blocked'")
    for key in ("cooldown_seconds", "afk_threshold_seconds"):
        seconds = value.get(key)
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError(
                f"{field} {key} must be a finite non-negative number"
            )
    if status == "passed" and value.get("kill_switch") != "clear":
        raise ValueError(f"{field} passed state must record kill_switch=clear")
    if status == "blocked":
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{field} blocked state must name a reason")


def validate_machine_state(value: str, expected_label: str) -> dict[str, object]:
    try:
        record = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"machine_state for {expected_label!r} is not valid JSON: {error}"
        ) from error
    if not isinstance(record, dict):
        raise ValueError(f"machine_state for {expected_label!r} must be an object")
    if record.get("schema") != MACHINE_STATE_SCHEMA:
        raise ValueError(
            f"machine_state for {expected_label!r} has unknown schema "
            f"{record.get('schema')!r}"
        )
    if record.get("label") != expected_label:
        raise ValueError(
            f"machine_state labels must be {PHASE_LABELS!r} in order; "
            f"got {record.get('label')!r} where {expected_label!r} was required"
        )
    captured = record.get("captured_at_utc")
    if not isinstance(captured, str):
        raise ValueError(f"machine_state {expected_label!r} lacks captured_at_utc")
    parse_utc_timestamp(captured, f"machine_state[{expected_label}].captured_at_utc")
    validate_capability(
        record.get("power"),
        f"machine_state[{expected_label}].power",
        frozenset(("ac", "battery")),
    )
    validate_capability(
        record.get("thermal"),
        f"machine_state[{expected_label}].thermal",
        frozenset(("nominal", "throttled", "fair", "serious", "critical")),
    )
    validate_idle(
        record.get("idle"),
        f"machine_state[{expected_label}].idle",
    )
    if "gate" in record:
        validate_checkpoint_gate(
            record["gate"],
            f"machine_state[{expected_label}].gate",
        )
    thermal = record["thermal"]
    if isinstance(thermal, dict) and "cpu_speed_limit_percent" in thermal:
        limit = thermal["cpu_speed_limit_percent"]
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError(
                f"machine_state[{expected_label}].thermal CPU speed limit "
                "must be an integer from 0 through 100"
            )
    return record


def load_run_provenance(path: Path) -> RunProvenance:
    """Load the strict key-value handoff produced by bench-compare."""
    fields: dict[str, str] = {}
    repeated: dict[str, list[str]] = {
        "lock": [],
        "ambient": [],
        "machine_state": [],
        "phase_load": [],
    }
    allowed = {"schema", *PROVENANCE_FIELDS, *repeated}

    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"cannot read provenance file {path}: {error}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line or "=" not in line:
            raise ValueError(
                f"{path}:{line_number}: expected a non-empty key=value record"
            )
        key, value = line.split("=", 1)
        if key not in allowed:
            raise ValueError(f"{path}:{line_number}: unknown provenance field {key!r}")
        if not value:
            raise ValueError(f"{path}:{line_number}: provenance field {key!r} is empty")
        if key in repeated:
            repeated[key].append(value)
        elif key in fields:
            raise ValueError(f"{path}:{line_number}: duplicate provenance field {key!r}")
        else:
            fields[key] = value

    schema = fields.get("schema")
    if schema not in PROVENANCE_FIELDS_BY_SCHEMA:
        raise ValueError(
            f"{path}: schema must be one of "
            f"{tuple(PROVENANCE_FIELDS_BY_SCHEMA)!r}, got {schema!r}"
        )
    schema_fields = PROVENANCE_FIELDS_BY_SCHEMA[schema]
    unexpected = sorted(set(fields) - {"schema", *schema_fields})
    if unexpected:
        raise ValueError(
            f"{path}: {schema} does not allow provenance fields: "
            f"{', '.join(unexpected)}"
        )
    missing = [field for field in schema_fields if field not in fields]
    if missing:
        raise ValueError(f"{path}: missing provenance fields: {', '.join(missing)}")
    if schema == PROVENANCE_SCHEMA_V1:
        # v1 predates embed target feature selection. Its only representable
        # meaning is the legacy default, so normalize that historical meaning
        # for the v2-shaped report without accepting a v2 field under v1.
        fields["embed_features"] = "<none>"
    started = parse_utc_timestamp(fields["started_utc"], "started_utc")
    finished = parse_utc_timestamp(fields["finished_utc"], "finished_utc")
    if finished < started:
        raise ValueError(f"{path}: finished_utc precedes started_utc")
    if not re.fullmatch(
        r"(?:local-random:[0-9a-f]{32}|hostname-sha256:[0-9a-f]{16}|"
        r"configured:[A-Za-z0-9._:-]+)",
        fields["host_id"],
    ):
        raise ValueError(
            f"{path}: host_id must be a local random identifier, configured label, "
            "or legacy hostname digest"
        )
    for field in ("base_sha", "head_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", fields[field]):
            raise ValueError(f"{path}: {field} must be a full lowercase commit SHA")
    if fields["head_mode"] not in ("in-place", "detached-worktree"):
        raise ValueError(f"{path}: head_mode must be in-place or detached-worktree")
    if fields["criterion_mode"] not in ("quick", "full"):
        raise ValueError(f"{path}: criterion_mode must be quick or full")
    if fields["enforcement"] not in ("report-only", "fail-on-regression"):
        raise ValueError(
            f"{path}: enforcement must be report-only or fail-on-regression"
        )
    baseline_path = Path(fields["baseline_name"])
    if (
        baseline_path.is_absolute()
        or not baseline_path.parts
        or any(part in (".", "..") for part in baseline_path.parts)
    ):
        raise ValueError(f"{path}: baseline_name must be a safe relative path")
    if len(repeated["ambient"]) != 3:
        raise ValueError(
            f"{path}: expected 3 ambient samples, got {len(repeated['ambient'])}"
        )
    if len(repeated["machine_state"]) != 3:
        raise ValueError(
            f"{path}: expected 3 machine-state samples, "
            f"got {len(repeated['machine_state'])}"
        )
    if len(repeated["lock"]) < 2:
        raise ValueError(f"{path}: expected both acquired-lock records")

    ambient_pattern = re.compile(
        r"^\[quiet\] (?P<label>.+): idle (?P<idle>\d+(?:\.\d+)?)% "
        r"\(floor (?P<floor>\d+(?:\.\d+)?)%\) ok \| top: .+$"
    )
    for expected_label, sample in zip(PHASE_LABELS, repeated["ambient"], strict=True):
        match = ambient_pattern.fullmatch(sample)
        if match is None or match.group("label") != expected_label:
            raise ValueError(
                f"{path}: ambient labels must be {PHASE_LABELS!r} in order"
            )
        idle = float(match.group("idle"))
        floor = float(match.group("floor"))
        if not 0.0 <= idle <= 100.0 or not 0.0 <= floor <= 100.0:
            raise ValueError(f"{path}: ambient idle/floor percentages are out of range")

    machine_states = tuple(
        validate_machine_state(value, label)
        for label, value in zip(
            PHASE_LABELS, repeated["machine_state"], strict=True
        )
    )
    captured_times = []
    for record in machine_states:
        captured = parse_utc_timestamp(
            str(record["captured_at_utc"]),
            f"machine_state[{record['label']}].captured_at_utc",
        )
        captured_times.append(captured)
        if not started <= captured <= finished:
            raise ValueError(
                f"{path}: machine_state[{record['label']}] timestamp is outside "
                "the run interval"
            )
    if captured_times != sorted(captured_times):
        raise ValueError(f"{path}: machine-state timestamps are out of phase order")
    if (
        fields["os"].split(maxsplit=1)[0] == "Darwin"
        and fields["enforcement"] == "fail-on-regression"
    ):
        for record in machine_states:
            gate = record.get("gate")
            if not isinstance(gate, dict) or gate.get("status") != "passed":
                raise ValueError(
                    f"{path}: macOS machine_state[{record['label']}] lacks a "
                    "passed fail-closed gate"
                )
            if gate["cooldown_seconds"] < 30:
                raise ValueError(
                    f"{path}: macOS machine_state[{record['label']}] cooldown "
                    "must be at least 30 seconds"
                )
            if gate["afk_threshold_seconds"] < 30:
                raise ValueError(
                    f"{path}: macOS machine_state[{record['label']}] AFK floor "
                    "must be at least 30 seconds"
                )

    # lattice#1515: phase_load is optional at the schema level (a provenance
    # file from before this fix, or a hand-built test fixture, carries none)
    # but when present must be the full ABBA set in order -- bench-compare
    # itself never writes a partial set: phase_gate refuses the whole run
    # before write_run_provenance runs if any arm's sampler failed.
    if repeated["phase_load"] and len(repeated["phase_load"]) != len(PHASE_LOAD_ARMS):
        raise ValueError(
            f"{path}: expected {len(PHASE_LOAD_ARMS)} phase_load samples if any "
            f"are present, got {len(repeated['phase_load'])}"
        )
    phase_load_pattern = re.compile(r"^\[phase-load\] (?P<arm>\S+): .*$")
    for expected_arm, sample in zip(PHASE_LOAD_ARMS, repeated["phase_load"]):
        match = phase_load_pattern.fullmatch(sample)
        if match is None or match.group("arm") != expected_arm:
            raise ValueError(
                f"{path}: phase_load arms must be {PHASE_LOAD_ARMS!r} in order"
            )

    fields.pop("schema")
    return RunProvenance(
        schema=schema,
        fields=fields,
        locks=tuple(repeated["lock"]),
        ambient_samples=tuple(repeated["ambient"]),
        machine_states=machine_states,
        phase_load_samples=tuple(repeated["phase_load"]),
    )


def find_change_files(root: Path) -> list[Path]:
    """Find every change/estimates.json under root (Criterion's per-bench output)."""
    return sorted(root.rglob("change/estimates.json"))


def _baseline_parts(baseline_name: str) -> tuple[str, ...]:
    """Return a safe relative path for a Criterion named baseline."""
    path = Path(baseline_name)
    if path.is_absolute() or not path.parts or any(part in (".", "..") for part in path.parts):
        raise ValueError(
            f"baseline name must be a non-empty relative path without '.' or '..': "
            f"{baseline_name!r}"
        )
    return path.parts


def find_selected_baseline_files(root: Path, baseline_name: str) -> list[Path]:
    """Find estimates files belonging to the exact CLI-selected baseline.

    Do not use parse_bench's tolerant `base/` or sole-sibling fallbacks here:
    those can be useful when rendering an ad-hoc local report, but they are not
    evidence about the named base arm selected by `--baseline-name`.
    """
    baseline_parts = _baseline_parts(baseline_name)
    suffix = (*baseline_parts, "estimates.json")
    matches = []
    for estimate_file in root.rglob("estimates.json"):
        relative_parts = estimate_file.relative_to(root).parts
        if len(relative_parts) > len(suffix) and relative_parts[-len(suffix):] == suffix:
            matches.append(estimate_file)
    return sorted(matches)


def artifact_bench_id(estimate_file: Path, root: Path, artifact_parts: int) -> str:
    """Return a platform-independent Criterion bench ID for an artifact file."""
    relative_parts = estimate_file.relative_to(root).parts
    trim = artifact_parts + 1  # artifact directory component(s) + estimates.json
    if len(relative_parts) <= trim:
        raise ValueError(f"{estimate_file} is not below a Criterion benchmark directory")
    return Path(*relative_parts[:-trim]).as_posix()


def _below_benchmark_dir(estimate_file: Path, root: Path, artifact_parts: int) -> bool:
    """True if estimate_file sits below a Criterion benchmark directory.

    Root-level debris (e.g. a bare `cargo bench` writing `new/estimates.json`
    or `change/estimates.json` directly under the Criterion root) has no
    benchmark directory to attribute to and must be excluded before calling
    artifact_bench_id, mirroring the length guard find_selected_baseline_files
    already applies to the selected-baseline inventory.
    """
    relative_parts = estimate_file.relative_to(root).parts
    trim = artifact_parts + 1
    return len(relative_parts) > trim


def selected_baseline_bench_ids(root: Path, baseline_name: str) -> set[str]:
    """Return bench IDs measured by the exact named base arm."""
    artifact_parts = len(_baseline_parts(baseline_name))
    return {
        artifact_bench_id(estimate_file, root, artifact_parts)
        for estimate_file in find_selected_baseline_files(root, baseline_name)
    }


def _checked_criterion_root(root: Path) -> Path:
    """Resolve an invariant-checked Criterion cleanup root."""
    if root.name != "criterion":
        raise ValueError(f"refusing cleanup outside a directory named 'criterion': {root}")
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"criterion cleanup root must be a real directory: {root}")
    return root.resolve(strict=True)


def _require_within_root(path: Path, root_resolved: Path, description: str) -> Path:
    """Resolve path and reject traversal outside the cleanup root."""
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{description} resolves outside criterion root: {path}") from error
    return resolved


def clear_selected_baseline_artifacts(root: Path, baseline_name: str) -> int:
    """Validate, then remove exact baseline dirs before copying a fresh base set.

    A pruned baseline dir drops its bench out of the selected set, so any
    `new/`/`change/` siblings left behind become unreachable debris that the
    head-side inventory later rglobs up as a phantom, unbaselined head
    measurement. Remove those siblings here, atomically with the baseline dir,
    since this is the only prune step that runs before the fresh base copy.
    Siblings are validated like the baseline dir itself (real dir, no
    symlinks, resolves under root) but excluded from the returned count, which
    stays the exact-baseline-dir total callers already print. Only baseline
    dirs found by find_selected_baseline_files enter the plan, so an unrelated
    target's Criterion data sharing this root is never touched.
    """
    root_resolved = _checked_criterion_root(root)
    baseline_parts = _baseline_parts(baseline_name)
    baseline_files = find_selected_baseline_files(root, baseline_name)
    deletion_plan: set[Path] = set()
    sibling_plan: set[Path] = set()

    for baseline_file in baseline_files:
        if baseline_file.is_symlink() or not baseline_file.is_file():
            raise ValueError(f"selected baseline estimate is not a real file: {baseline_file}")
        _require_within_root(
            baseline_file, root_resolved, "selected baseline estimate"
        )
        baseline_dir = baseline_file.parent
        if baseline_dir.is_symlink() or not baseline_dir.is_dir():
            raise ValueError(
                f"selected baseline artifact is not a real directory: {baseline_dir}"
            )
        _require_within_root(
            baseline_dir, root_resolved, "selected baseline artifact"
        )
        deletion_plan.add(baseline_dir)

        bench_dir = baseline_file.parents[len(baseline_parts)]
        _require_within_root(bench_dir, root_resolved, "selected baseline benchmark")
        for artifact_name in ("new", "change"):
            artifact_dir = bench_dir / artifact_name
            if artifact_dir.is_symlink():
                raise ValueError(f"refusing to remove symlinked Criterion artifact: {artifact_dir}")
            if not artifact_dir.exists():
                continue
            if not artifact_dir.is_dir():
                raise ValueError(
                    f"Criterion artifact is not a directory: {artifact_dir}"
                )
            _require_within_root(artifact_dir, root_resolved, "Criterion artifact")
            sibling_plan.add(artifact_dir)

    for path in sorted(deletion_plan | sibling_plan):
        shutil.rmtree(path)

    return len(deletion_plan)


def clear_selected_head_artifacts(root: Path, baseline_name: str) -> tuple[int, int]:
    """Remove stale new/change siblings for exact selected-baseline benches.

    This is intentionally narrower than cleaning a Criterion tree: it derives
    each benchmark directory from a selected baseline estimates file, checks
    that the resolved directory stays under an actual `criterion/` root, and
    removes only that benchmark's `new/` and `change/` directories. The full
    deletion plan is validated before any artifact is removed.
    """
    root_resolved = _checked_criterion_root(root)
    baseline_parts = _baseline_parts(baseline_name)
    baseline_files = find_selected_baseline_files(root, baseline_name)
    deletion_plan: set[Path] = set()

    for baseline_file in baseline_files:
        if baseline_file.is_symlink() or not baseline_file.is_file():
            raise ValueError(f"selected baseline estimate is not a real file: {baseline_file}")
        _require_within_root(
            baseline_file, root_resolved, "selected baseline estimate"
        )

        bench_dir = baseline_file.parents[len(baseline_parts)]
        _require_within_root(
            bench_dir, root_resolved, "selected baseline benchmark"
        )

        for artifact_name in ("new", "change"):
            artifact_dir = bench_dir / artifact_name
            if artifact_dir.is_symlink():
                raise ValueError(f"refusing to remove symlinked Criterion artifact: {artifact_dir}")
            if not artifact_dir.exists():
                continue
            if not artifact_dir.is_dir():
                raise ValueError(
                    f"Criterion artifact is not a directory: {artifact_dir}"
                )
            _require_within_root(
                artifact_dir, root_resolved, "Criterion artifact"
            )
            deletion_plan.add(artifact_dir)

    for artifact_dir in sorted(deletion_plan):
        shutil.rmtree(artifact_dir)

    return len(baseline_files), len(deletion_plan)


def find_baseline_estimates(bench_dir: Path, baseline_name: str) -> Path | None:
    """Locate the baseline estimates.json for a bench directory.

    Criterion writes the pre-run comparison snapshot under a directory named
    after the baseline: the default (unnamed) rotation uses `base/`, while a
    named baseline (`--save-baseline <name>` / `--baseline <name>`, as used by
    bench-compare.sh's `compare-base` leg) writes under `<name>/` instead —
    `base/` is never created in that flow. Prefer the caller-supplied baseline
    name FIRST: Criterion computed change/ against that baseline, so a stale
    `base/` left in a dirty local tree must not shadow it (review of
    PR #548 reproduced exactly that wrong-baseline report). Then try the
    default `base/` (covers CI's default-rotation runs). As a last resort,
    accept a sibling directory holding an estimates.json that isn't
    `new`/`change` — but only when it is unambiguous: Criterion supports
    multiple named baselines side by side, and guessing among several would
    silently gate against the wrong one.
    """
    candidates = [baseline_name, "base"]
    for candidate in candidates:
        p = bench_dir / candidate / "estimates.json"
        if p.exists():
            return p

    fallbacks = [
        child for child in sorted(bench_dir.iterdir())
        if child.is_dir()
        and child.name not in ("new", "change")
        and (child / "estimates.json").exists()
    ]
    if len(fallbacks) == 1:
        print(f"note: {bench_dir.name}: using sole sibling baseline dir "
              f"'{fallbacks[0].name}/' (neither '{baseline_name}/' nor 'base/' found)",
              file=sys.stderr)
        return fallbacks[0] / "estimates.json"
    if len(fallbacks) > 1:
        names = ", ".join(f.name for f in fallbacks)
        print(f"warn: {bench_dir.name}: multiple candidate baseline dirs ({names}) "
              f"and none match '{baseline_name}/' or 'base/' — refusing to guess",
              file=sys.stderr)
    return None


def load_sample_metadata(path: Path, bench_name: str, arm: str) -> tuple[str, int] | None:
    """Read Criterion's actual sampling mode and sample count."""
    try:
        sample = json.loads(path.read_text())
        mode = sample["sampling_mode"]
        iters = sample["iters"]
        times = sample["times"]
        if mode not in ("Linear", "Flat"):
            raise ValueError(f"unknown sampling_mode {mode!r}")
        if not isinstance(iters, list) or not isinstance(times, list):
            raise ValueError("iters and times must be arrays")
        if not times or len(iters) != len(times):
            raise ValueError(
                f"iters/times lengths must be equal and non-zero, got "
                f"{len(iters)}/{len(times)}"
            )
        for array_name, values in (("iters", iters), ("times", times)):
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in values
            ):
                raise ValueError(
                    f"{array_name} must contain only positive finite numbers"
                )
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(
            f"warn: {bench_name}: cannot read {arm} Criterion sample metadata "
            f"from {path}: {error}",
            file=sys.stderr,
        )
        return None
    return mode, len(times)


def criterion_number(value: object, field: str, *, positive: bool = False) -> float:
    """Validate one numeric Criterion field without accepting JSON booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a JSON number, got {value!r}")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} is outside the finite float range") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    if positive and number <= 0.0:
        raise ValueError(f"{field} must be positive, got {value!r}")
    return number


def parse_bench(change_file: Path, root: Path, baseline_name: str) -> BenchResult | None:
    """Parse one change/estimates.json + sibling new/estimates.json + baseline estimates.json.

    Returns None if files are malformed (bench skipped, not failed).
    """
    bench_dir = change_file.parent.parent  # .../<bench>/<test>/
    name = artifact_bench_id(change_file, root, artifact_parts=1)

    try:
        change = json.loads(change_file.read_text())
        mean = change["mean"]
        point = criterion_number(
            mean["point_estimate"], f"{name} change point estimate"
        )
        ci_low = criterion_number(
            mean["confidence_interval"]["lower_bound"],
            f"{name} change CI lower bound",
        )
        ci_high = criterion_number(
            mean["confidence_interval"]["upper_bound"],
            f"{name} change CI upper bound",
        )
        if ci_low <= -1.0 or point <= -1.0 or ci_high <= -1.0:
            raise ValueError(f"{name} relative-change estimates must be greater than -1")
        if not ci_low <= point <= ci_high:
            raise ValueError(f"{name} change interval does not contain its point estimate")

        new_path = bench_dir / "new" / "estimates.json"
        new_ns = criterion_number(
            json.loads(new_path.read_text())["mean"]["point_estimate"],
            f"{name} head mean",
            positive=True,
        )

        base_path = find_baseline_estimates(bench_dir, baseline_name)
        if base_path is None:
            print(f"warn: {name}: change/estimates.json present but no resolvable "
                  f"baseline dir (tried base/, {baseline_name}/, and other siblings) "
                  f"— skipping", file=sys.stderr)
            return None
        old_ns = criterion_number(
            json.loads(base_path.read_text())["mean"]["point_estimate"],
            f"{name} base mean",
            positive=True,
        )
    except (
        KeyError,
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as e:
        print(f"warn: skipping {name}: {e}", file=sys.stderr)
        return None

    head_sample = load_sample_metadata(new_path.with_name("sample.json"), name, "head")
    base_sample = load_sample_metadata(base_path.with_name("sample.json"), name, "base")
    return BenchResult(name=name, point=point, ci_low=ci_low, ci_high=ci_high,
                       new_ns=new_ns, old_ns=old_ns,
                       head_sample_count=head_sample[1] if head_sample else None,
                       head_sampling_mode=head_sample[0] if head_sample else None,
                       base_sample_count=base_sample[1] if base_sample else None,
                       base_sampling_mode=base_sample[0] if base_sample else None)


def parse_order_control(
    root: Path,
    baseline_name: str,
    expected_ids: set[str],
    *,
    require_samples: bool,
) -> list[BenchResult]:
    """Parse the B→A half of an ABBA run, refusing partial evidence."""
    if not root.exists():
        raise ValueError(f"order-control Criterion root does not exist: {root}")

    selected_ids = selected_baseline_bench_ids(root, baseline_name)
    if selected_ids != expected_ids:
        missing = sorted(expected_ids - selected_ids)
        extra = sorted(selected_ids - expected_ids)
        raise ValueError(
            "order-control selected-baseline set differs from the forward set: "
            f"missing={missing}, extra={extra}"
        )

    change_files = find_change_files(root)
    change_by_id = {}
    for change_file in change_files:
        bench_id = artifact_bench_id(change_file, root, artifact_parts=1)
        change_by_id[bench_id] = change_file
    if change_by_id.keys() != expected_ids:
        missing = sorted(expected_ids - change_by_id.keys())
        extra = sorted(change_by_id.keys() - expected_ids)
        raise ValueError(
            "order-control comparison set differs from the forward set: "
            f"missing={missing}, extra={extra}"
        )

    results = []
    unjudged = []
    for bench_id in sorted(expected_ids):
        result = parse_bench(change_by_id[bench_id], root, baseline_name)
        if result is None:
            unjudged.append(bench_id)
        else:
            results.append(result)
    if unjudged:
        raise ValueError(
            f"{len(unjudged)} order-control comparison(s) could not be judged: "
            + ", ".join(unjudged)
        )

    if require_samples:
        incomplete = [
            result.name
            for result in results
            if result.base_sample_count is None
            or result.base_sampling_mode is None
            or result.head_sample_count is None
            or result.head_sampling_mode is None
        ]
        if incomplete:
            raise ValueError(
                f"{len(incomplete)} order-control comparison(s) lack Criterion "
                f"sample metadata: {', '.join(incomplete[:5])}"
            )
    return results


def sample_shape_summary(results: list[BenchResult], arm: str) -> str:
    """Summarize actual Criterion sample.json shapes for one A/B arm."""
    shapes: dict[tuple[str, int], int] = {}
    missing = 0
    for result in results:
        if arm == "base":
            mode = result.base_sampling_mode
            count = result.base_sample_count
        else:
            mode = result.head_sampling_mode
            count = result.head_sample_count
        if mode is None or count is None:
            missing += 1
            continue
        shapes[(mode, count)] = shapes.get((mode, count), 0) + 1

    parts = [
        f"{count} {mode} ({bench_count} "
        f"{'benchmark' if bench_count == 1 else 'benchmarks'})"
        for (mode, count), bench_count in sorted(shapes.items())
    ]
    if missing:
        parts.append(
            f"unrecorded ({missing} {'benchmark' if missing == 1 else 'benchmarks'})"
        )
    return ", ".join(parts) if parts else "unrecorded"


def sample_cell(result: BenchResult, arm: str) -> str:
    if arm == "base":
        count = result.base_sample_count
        mode = result.base_sampling_mode
    else:
        count = result.head_sample_count
        mode = result.head_sampling_mode
    if count is None or mode is None:
        return "unrecorded"
    return f"{count} {mode}"


def capability_summary(value: object) -> str:
    if not isinstance(value, dict):
        return "invalid"
    if value.get("status") == "unavailable":
        return f"unavailable ({value.get('reason')})"
    state = str(value.get("state"))
    if "cpu_speed_limit_percent" in value:
        state += f" (CPU speed limit {value['cpu_speed_limit_percent']}%)"
    return f"{state} via {value.get('source')}"


def idle_summary(value: object) -> str:
    if not isinstance(value, dict):
        return "invalid"
    if value.get("status") == "unavailable":
        return f"unavailable ({value.get('reason')})"
    return f"{float(value['seconds']):.1f}s via {value.get('source')}"


def checkpoint_gate_summary(value: object) -> str:
    if not isinstance(value, dict):
        return "not enforced on this platform"
    if value.get("status") == "blocked":
        return f"blocked ({value.get('reason')})"
    return (
        f"passed (cooldown {float(value['cooldown_seconds']):.1f}s, "
        f"AFK floor {float(value['afk_threshold_seconds']):.1f}s, "
        f"kill-switch {value.get('kill_switch')})"
    )


def render_run_provenance(
    provenance: RunProvenance | None,
    results: list[BenchResult],
) -> list[str]:
    """Render measurement conditions inside the stored Markdown artifact."""
    lines = ["<details><summary>Run provenance</summary>", ""]
    if provenance is None:
        lines.append(
            "⚠️ Caller supplied no `--provenance-file`; machine and phase "
            "conditions are unavailable. This report is unsuitable as "
            "benchmark evidence."
        )
        lines.append("")
    else:
        lines.append(f"    schema={provenance.schema}")
        blocked_checkpoints = [
            state for state in provenance.machine_states
            if state.get("gate", {}).get("status") == "blocked"
        ]
        if blocked_checkpoints:
            labels = ", ".join(str(state["label"]) for state in blocked_checkpoints)
            lines.append(
                "⚠️ One or more machine-state checkpoints were blocked "
                f"({labels}). This report is unsuitable as benchmark evidence."
            )
            lines.append("")
        for field in PROVENANCE_FIELDS:
            lines.append(f"    {field}={provenance.fields[field]}")
        for lock in provenance.locks:
            lines.append(f"    lock={lock}")
        for sample in provenance.ambient_samples:
            lines.append(f"    ambient={sample}")
        for sample in provenance.phase_load_samples:
            lines.append(f"    phase_load={sample}")
        for state in provenance.machine_states:
            lines.append(
                f"    machine_state[{state['label']}]="
                f"captured {state['captured_at_utc']}; "
                f"power {capability_summary(state['power'])}; "
                f"thermal {capability_summary(state['thermal'])}; "
                f"HID idle {idle_summary(state['idle'])}; "
                f"gate {checkpoint_gate_summary(state.get('gate'))}"
            )
    lines.append(f"    criterion_base_samples={sample_shape_summary(results, 'base')}")
    lines.append(f"    criterion_head_samples={sample_shape_summary(results, 'head')}")
    lines.extend(["", "</details>", ""])
    return lines


def empty_gate_reason(
    results: list[BenchResult],
    target: str | None,
    informational_target: str | None,
) -> str | None:
    """Why nothing could vote, or None when at least one measurement can.

    EMPTY is not CLEAN. With every row demoted by target policy the noise band
    has no population left to be a statement about, so the run owes a refusal
    rather than a tick: a reader cannot tell "nothing regressed" from "nothing
    was allowed to say".
    """
    if not results:
        return None
    if any(not r.is_informational(target, informational_target) for r in results):
        return None
    return (
        f"target policy classified all {len(results)} measurement(s) of "
        f"`{target}` informational, so no benchmark could vote; a pass over an "
        "empty gated set says nothing about this change"
    )


def render_report(results: list[BenchResult], arch: str, target: str | None = None,
                  informational_target: str | None = None,
                  provenance: RunProvenance | None = None,
                  resolution: str = "full",
                  decision_suppressed_reason: str | None = None) -> str:
    if not results:
        raise ValueError("classifier requires at least one measured row")
    if resolution not in ("quick", "full"):
        raise ValueError(f"unsupported benchmark resolution: {resolution}")

    order_balanced = any(r.order_bias_bound is not None for r in results)
    interval_label = "ABBA bound" if order_balanced else "95% CI"
    gated = [
        r for r in results
        if not r.is_informational(target, informational_target)
    ]
    info = [
        r for r in results
        if r.is_informational(target, informational_target)
    ]
    gate_empty = empty_gate_reason(results, target, informational_target)
    # One predicate for every verdict-bearing branch below: a tick, a WARN/FAIL
    # headline and the rule footer all assert a classification, and neither a
    # suppressed decision nor an empty gated set has one to assert.
    suppressed = decision_suppressed_reason is not None or gate_empty is not None

    fails = [r for r in gated if r.verdict(resolution) == "FAIL"]
    warns = [r for r in gated if r.verdict(resolution) == "WARN"]
    unattributable = [
        r for r in gated if r.verdict(resolution) == QUICK_UNATTRIBUTABLE
    ]
    wins = [r for r in gated if r.verdict(resolution) == "WIN"]

    info_fails = [r for r in info if r.verdict(resolution) == "FAIL"]
    info_warns = [r for r in info if r.verdict(resolution) == "WARN"]
    info_unattributable = [
        r for r in info if r.verdict(resolution) == QUICK_UNATTRIBUTABLE
    ]
    info_wins = [r for r in info if r.verdict(resolution) == "WIN"]

    lines = [f"### `{arch}` — perf regression report\n"]
    if gate_empty is not None:
        lines.append(
            "**⛔ NO GATED MEASUREMENT** — "
            f"{gate_empty}. No source-performance verdict was rendered."
        )
    if decision_suppressed_reason is not None:
        lines.append(
            "**⏸ NOT MEASURABLE** — "
            f"{decision_suppressed_reason}. No source-performance verdict was rendered."
        )
    elif fails:
        if order_balanced:
            lines.append(
                f"**❌ {len(fails)} FAIL** (regression >{FAIL_PCT}% — lower bound "
                "of the ABBA log-ratio effect after widening by the measured "
                "order-bias envelope)"
            )
        else:
            lines.append(
                f"**❌ {len(fails)} FAIL** (regression >{FAIL_PCT}% — lower bound "
                "of Criterion's two-sided 95% CI, i.e. about a 97.5% one-sided "
                "level, not a calibrated one-sided 95% test)"
            )
    if not suppressed and warns:
        qualifier = (
            "order-bias-widened ABBA lower bound"
            if order_balanced
            else "two-sided-95%-CI lower bound"
        )
        lines.append(
            f"**⚠ {len(warns)} WARN** (regression {WARN_PCT}-{FAIL_PCT}% by "
            f"the same {qualifier})"
        )
    if not suppressed and unattributable:
        lines.append(
            f"**◻ {len(unattributable)} UNATTRIBUTABLE** (quick-resolution "
            f"{WARN_PCT}-{FAIL_PCT}% warn band is narrower than the harness's "
            "measured arm-order bias)"
        )
    if not suppressed and wins:
        lines.append(f"**🚀 {len(wins)} confirmed improvement**")
    if not suppressed and not (
        fails or warns or unattributable or wins
    ):
        lines.append(f"✅ All {len(gated)} gated benches within noise band (±{WARN_PCT}%)")
    lines.append("")
    lines.extend(render_run_provenance(provenance, results))

    if not suppressed and (
        fails or warns or unattributable or wins
    ):
        bias_column = " | order bias ≤" if order_balanced else ""
        lines.append(
            f"| Bench | Δ point | {interval_label}{bias_column} | new ns | "
            "base ns | base n/mode | head n/mode | verdict |"
        )
        lines.append(
            "|---|---:|---|---:|---:|---:|---|---|---|"
            if order_balanced
            else "|---|---:|---|---:|---:|---|---|---|"
        )
        for r in sorted(
            fails + warns + unattributable + wins, key=lambda r: -r.ci_low_pct
        ):
            verdict = r.verdict(resolution)
            icon = {
                "FAIL": "❌",
                "WARN": "⚠",
                QUICK_UNATTRIBUTABLE: "◻",
                "WIN": "🚀",
            }[verdict]
            bias_cell = (
                f"| {r.order_bias_bound_pct:+.2f}% "
                if r.order_bias_bound_pct is not None
                else ""
            )
            lines.append(
                f"| `{r.name}` | {r.point_pct:+.2f}% | [{r.ci_low_pct:+.2f}%, {r.ci_high_pct:+.2f}%] "
                f"{bias_cell}| {r.new_ns:.1f} | {r.old_ns:.1f} | {sample_cell(r, 'base')} "
                f"| {sample_cell(r, 'head')} | {icon} {verdict} |"
            )
        lines.append("")

    if info:
        lines.append(
            f"**ℹ️ {len(info)} informational** (explicit target policy; "
            f"measured and reported, excluded from the verdict)"
        )
        if gate_empty is not None:
            # Every row was demoted, so the informational table IS the report.
            # Render all of them beside the refusal, PASS rows included: the
            # reader's question here is what was measured at all, and the
            # all-measurements <details> below carries no would-be verdict.
            info_rows = sorted(info, key=lambda r: -r.ci_low_pct)
        elif not suppressed:
            info_rows = sorted(
                info_fails + info_warns + info_unattributable + info_wins,
                key=lambda r: -r.ci_low_pct,
            )
        else:
            info_rows = []
        if info_rows:
            bias_column = " | order bias ≤" if order_balanced else ""
            lines.append(
                f"| Bench | Δ point | {interval_label}{bias_column} | new ns | "
                "base ns | base n/mode | head n/mode | (would-be verdict) |"
            )
            lines.append(
                "|---|---:|---|---:|---:|---:|---|---|---|"
                if order_balanced
                else "|---|---:|---|---:|---:|---|---|---|"
            )
            for r in info_rows:
                verdict = r.verdict(resolution)
                icon = {
                    "FAIL": "❌",
                    "WARN": "⚠",
                    QUICK_UNATTRIBUTABLE: "◻",
                    "WIN": "🚀",
                    "PASS": "✅",
                }[verdict]
                bias_cell = (
                    f"| {r.order_bias_bound_pct:+.2f}% "
                    if r.order_bias_bound_pct is not None
                    else ""
                )
                lines.append(
                    f"| `{r.name}` | {r.point_pct:+.2f}% | [{r.ci_low_pct:+.2f}%, {r.ci_high_pct:+.2f}%] "
                    f"{bias_cell}| {r.new_ns:.1f} | {r.old_ns:.1f} | {sample_cell(r, 'base')} "
                    f"| {sample_cell(r, 'head')} | {icon} {verdict} (informational) |"
                )
        lines.append("")

    lines.append(
        f"<details><summary>All {len(results)} measurements</summary>\n\n"
        f"| Bench | Δ point | bound-lower | bound-upper"
        f"{' | order bias ≤' if order_balanced else ''} | base n/mode | head n/mode |\n"
        + (
            "|---|---:|---:|---:|---:|---|---|"
            if order_balanced
            else "|---|---:|---:|---:|---|---|"
        )
    )
    for r in sorted(results, key=lambda r: r.name):
        bias_cell = (
            f"| {r.order_bias_bound_pct:+.2f}% "
            if r.order_bias_bound_pct is not None
            else ""
        )
        lines.append(
            f"| `{r.name}` | {r.point_pct:+.2f}% | {r.ci_low_pct:+.2f}% "
            f"| {r.ci_high_pct:+.2f}% {bias_cell}| {sample_cell(r, 'base')} "
            f"| {sample_cell(r, 'head')} |"
        )
    lines.append("\n</details>\n")
    lower_bound_name = (
        "order-bias-widened ABBA lower bound"
        if order_balanced
        else "CI-lower of change"
    )
    if not suppressed:
        if resolution == "quick":
            lines.append(
                f"_Rule: {lower_bound_name} ≤{WARN_PCT}% passes silently; "
                f"({WARN_PCT}%, {FAIL_PCT}%] is unattributable at quick resolution "
                "because measured arm-order bias is wider than this band; "
                f">{FAIL_PCT}% fails._"
            )
        else:
            lines.append(
                f"_Rule: {lower_bound_name} ≤{WARN_PCT}% passes silently; "
                f"({WARN_PCT}%, {FAIL_PCT}%] warns; >{FAIL_PCT}% fails._"
            )
    elif gate_empty is not None:
        lines.append(
            "_The measurements are shown for diagnosis only; no row held "
            "gating authority, so none was classified._"
        )
    else:
        lines.append(
            "_The measurements are shown for diagnosis only; WARN/FAIL "
            "classification was suppressed._"
        )
    if informational_target is not None:
        lines.append(
            f"_Target `{informational_target}` classified informational-only by explicit "
            f"target policy; its {len(info)} measurement(s) are excluded from the verdict._\n"
        )
    else:
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("criterion_root", type=Path, nargs="?",
                    help="Path to target/criterion (or per-bench root)")
    ap.add_argument("arch", nargs="?",
                    help="Arch label for the report header (e.g. aarch64-linux)")
    ap.add_argument("--out", type=Path, help="Write markdown report to this path")
    ap.add_argument("--baseline-name", default="compare-base",
                    help="Named-baseline dir to look for when base/ is absent "
                         "(default: compare-base, matching bench-compare.sh)")
    ap.add_argument("--target",
                    help="Exact <crate>:<bench-target> identity for this isolated Criterion "
                         "root. bench-compare passes this for every target.")
    ap.add_argument("--informational-target",
                    help="Classify this exact target as informational-only. Must equal --target; "
                         "the caller owns the reviewed policy decision.")
    ap.add_argument("--resolution", choices=("quick", "full"), default="full",
                    help="Measurement resolution used for verdict classification "
                         "(default: full).")
    ap.add_argument("--require-measurements", action="store_true",
                    help="Fail (exit 2) instead of passing when the run produced no gating "
                         "comparison to judge, a selected-base benchmark has no head "
                         "comparison, or a head benchmark has no selected baseline. "
                         "Without this, an absent baseline exits 0, which is right for a "
                         "first run but wrong for an automated lane: it cannot tell "
                         "'nothing regressed' from 'nothing was measured'.")
    ap.add_argument("--ambient-samples", type=Path,
                    help="JSONL before/between/after ambient samples. Requires "
                         "--status-out; a missing, malformed, or duplicated "
                         "stream exits 2 (bad input), a busy-below-floor "
                         "reading on a valid stream exits 3.")
    ap.add_argument("--status-out", type=Path,
                    help="Atomically write pass/regression/error/not_measurable JSON.")
    ap.add_argument("--phase-load-summaries", type=Path,
                    help="Optional ABBA load-verdict JSONL; requires --status-out. "
                         "Records phase_load and phase_load_verdict without changing "
                         "the performance verdict or boundary assessment.")
    ap.add_argument("--prepare-head", action="store_true",
                    help="Before a head measurement, remove only new/ and change/ siblings "
                         "of benches in the exact selected baseline set. The root must be a "
                         "real directory named criterion. No report is generated.")
    ap.add_argument("--prepare-baseline-copy", action="store_true",
                    help="Before copying a fresh base arm, remove only the exact selected "
                         "baseline artifact dirs from an invariant-checked criterion root. "
                         "No report is generated.")
    ap.add_argument("--provenance-file", type=Path,
                    help="Strict bench-compare key=value provenance handoff to embed in the "
                         "Markdown report. Accepts legacy v1 and current v2; v1 defaults "
                         "the then-unrepresented embed_features field to <none>.")
    ap.add_argument("--require-provenance", action="store_true",
                    help="Fail (exit 2) unless complete run provenance and actual Criterion "
                         "base/head sample metadata are available.")
    ap.add_argument("--order-control-root", type=Path,
                    help="Criterion root for the reverse B→A half of an ABBA run. "
                         "The selected baseline is head₂ and the new arm is base₂; "
                         "benchmark identities must exactly match the forward root.")
    ap.add_argument("--order-control-baseline-name", default="compare-head",
                    help="Named baseline in --order-control-root (default: compare-head).")
    ap.add_argument("--require-order-balance", action="store_true",
                    help="Fail (exit 2) unless --order-control-root supplies a complete "
                         "reverse-order comparison for every forward benchmark.")
    ap.add_argument("--selftest", action="store_true",
                    help="Run the fixture self-test (no criterion_root/arch needed) and exit")
    args = ap.parse_args()

    selected_modes = sum((
        args.selftest,
        args.prepare_head,
        args.prepare_baseline_copy,
    ))
    if selected_modes > 1:
        ap.error("--selftest, --prepare-head, and --prepare-baseline-copy are "
                 "mutually exclusive")
    if args.selftest:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
        import perf_bench_gate_selftest
        return perf_bench_gate_selftest.run(globals())

    if args.prepare_baseline_copy:
        if args.criterion_root is None:
            ap.error("criterion_root is required with --prepare-baseline-copy")
        if args.arch is not None:
            ap.error("arch must be omitted with --prepare-baseline-copy")
        try:
            removed = clear_selected_baseline_artifacts(
                args.criterion_root, args.baseline_name
            )
        except (OSError, ValueError) as error:
            print(f"error: cannot prepare Criterion baseline copy: {error}", file=sys.stderr)
            return 2
        print(
            f"removed {removed} stale selected-baseline artifact "
            f"{'directory' if removed == 1 else 'directories'} before fresh base copy"
        )
        return 0

    if args.prepare_head:
        if args.criterion_root is None:
            ap.error("criterion_root is required with --prepare-head")
        if args.arch is not None:
            ap.error("arch must be omitted with --prepare-head")
        try:
            selected, removed = clear_selected_head_artifacts(
                args.criterion_root, args.baseline_name
            )
        except (OSError, ValueError) as error:
            print(f"error: cannot prepare Criterion head artifacts: {error}", file=sys.stderr)
            return 2
        print(
            f"prepared {selected} selected-baseline benchmark(s); "
            f"removed {removed} stale head artifact "
            f"{'directory' if removed == 1 else 'directories'}"
        )
        return 0

    if args.criterion_root is None or args.arch is None:
        ap.error("criterion_root and arch are required unless --selftest or "
                 "a preparation mode is passed")

    if args.ambient_samples is not None and args.status_out is None:
        print("error: --ambient-samples requires --status-out", file=sys.stderr)
        return EXIT_ERROR
    if args.phase_load_summaries is not None and args.status_out is None:
        print("error: --phase-load-summaries requires --status-out", file=sys.stderr)
        return EXIT_ERROR

    ambient: AmbientAssessment | None = None
    phase_load: list[dict] | None = None

    def finish(
        verdict: str,
        exit_code: int,
        reason: str,
        *,
        measurement_count: int | None = None,
        regression_count: int | None = None,
    ) -> int:
        if args.status_out is None:
            return exit_code
        try:
            write_gate_status(
                args.status_out,
                arch=args.arch,
                target=args.target,
                verdict=verdict,
                exit_code=exit_code,
                reason=reason,
                ambient=ambient,
                phase_load=phase_load,
                measurement_count=measurement_count,
                regression_count=regression_count,
            )
        except OSError as error:
            print(
                f"error: could not write gate status {args.status_out}: {error}",
                file=sys.stderr,
            )
            return EXIT_ERROR
        return exit_code

    if args.status_out is not None:
        try:
            floor_pct = idle_floor_from_env()
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return finish("error", EXIT_ERROR, str(error))
        ambient = assess_ambient_samples(args.ambient_samples, floor_pct)
        if args.phase_load_summaries is not None:
            try:
                phase_load = load_phase_load_summaries(args.phase_load_summaries, floor_pct)
            except ValueError as error:
                print(f"error: {error}", file=sys.stderr)
                return finish("error", EXIT_ERROR, str(error))

    require_measurements = args.require_measurements or args.status_out is not None

    if args.require_order_balance and args.order_control_root is None:
        reason = (
            "--require-order-balance needs --order-control-root; a single "
            "fixed-order Criterion comparison cannot be enforced"
        )
        print(f"error: {reason}.", file=sys.stderr)
        return finish("error", EXIT_ERROR, reason)

    if args.informational_target is not None:
        if args.target is None:
            print("error: --informational-target requires --target", file=sys.stderr)
            return finish(
                "error", EXIT_ERROR, "--informational-target requires --target"
            )
        if args.informational_target != args.target:
            print(f"error: informational target '{args.informational_target}' does not match "
                  f"this Criterion root's target '{args.target}' — refusing to demote",
                  file=sys.stderr)
            return finish(
                "error",
                EXIT_ERROR,
                "the informational target does not match the selected target",
            )
    informational = (
        args.target is not None and args.target == args.informational_target
    )

    if args.require_provenance and args.provenance_file is None:
        print(
            "error: --require-provenance needs --provenance-file; a stored verdict "
            "without machine and phase conditions is not auditable.",
            file=sys.stderr,
        )
        return finish(
            "error",
            EXIT_ERROR,
            "--require-provenance needs --provenance-file",
        )
    provenance = None
    if args.provenance_file is not None:
        try:
            provenance = load_run_provenance(args.provenance_file)
        except ValueError as error:
            if args.require_provenance:
                print(f"error: invalid run provenance: {error}", file=sys.stderr)
                return finish(
                    "error", EXIT_ERROR, f"invalid run provenance: {error}"
                )
            print(
                f"warn: invalid run provenance: {error}; rendering an unsuitable "
                "report without provenance",
                file=sys.stderr,
            )
    if args.require_provenance and provenance is not None:
        missing_toolchain = [
            field
            for field in (
                "base_rustc",
                "head_rustc",
                "base_cargo",
                "head_cargo",
                "base_criterion",
                "head_criterion",
            )
            if provenance.fields[field] == "unavailable"
        ]
        if missing_toolchain:
            print(
                "error: --require-provenance needs complete base/head toolchain "
                f"identity; unavailable: {', '.join(missing_toolchain)}.",
                file=sys.stderr,
            )
            return finish(
                "error",
                EXIT_ERROR,
                "required base/head toolchain identity is unavailable",
            )

    if not args.criterion_root.exists():
        print(f"error: {args.criterion_root} does not exist", file=sys.stderr)
        return finish(
            "error",
            EXIT_ERROR,
            f"criterion root {args.criterion_root} does not exist",
        )

    try:
        baseline_ids = selected_baseline_bench_ids(
            args.criterion_root, args.baseline_name
        )
    except ValueError as error:
        print(f"error: invalid --baseline-name: {error}", file=sys.stderr)
        return finish("error", EXIT_ERROR, f"invalid baseline name: {error}")

    all_change_files = [
        change_file
        for change_file in find_change_files(args.criterion_root)
        if _below_benchmark_dir(change_file, args.criterion_root, artifact_parts=1)
    ]
    change_file_ids = {
        change_file: artifact_bench_id(
            change_file, args.criterion_root, artifact_parts=1
        )
        for change_file in all_change_files
    }
    # Root-level new/estimates.json (a bare `cargo bench` writing directly under
    # the Criterion root, or stray debris) has no benchmark directory to
    # attribute to; silently excluded here, the same disposition
    # find_selected_baseline_files already gives the mirror-image root-level
    # case in the selected-baseline inventory, rather than crashing the gate.
    head_ids = {
        artifact_bench_id(head_file, args.criterion_root, artifact_parts=1)
        for head_file in args.criterion_root.rglob("new/estimates.json")
        if _below_benchmark_dir(head_file, args.criterion_root, artifact_parts=1)
    }
    # A stale/non-selected change/estimates.json is not evidence of coverage: the
    # enforcing filter below only trusts a change file whose id is in baseline_ids,
    # so exempting on any change/ presence here would fail open on exactly the
    # artifact that filter later excludes. Reconcile against baseline_ids directly,
    # the same selected-baseline inventory the enforcing path trusts.
    missing_baseline_ids = sorted(head_ids - baseline_ids)

    def refuse_missing_baseline() -> int:
        print(
            f"error: NO COVERAGE: --require-measurements set but "
            f"{len(missing_baseline_ids)} benchmark(s) ran in the head with no "
            f"estimate in selected baseline {args.baseline_name!r}:",
            file=sys.stderr,
        )
        for bench_id in missing_baseline_ids:
            print(
                f"  - {args.target + ': ' if args.target else ''}{bench_id}",
                file=sys.stderr,
            )
        print(
            "Record a baseline for these benchmarks before enforcing the A/B gate.",
            file=sys.stderr,
        )
        return finish(
            "error",
            EXIT_ERROR,
            f"NO COVERAGE: {len(missing_baseline_ids)} head benchmarks have no selected baseline",
        )

    # An entirely empty selected baseline gets its own clearer message: check
    # it first, since a fully empty baseline_ids would otherwise put every
    # head id into missing_baseline_ids and mask the more specific diagnosis.
    if require_measurements and not baseline_ids:
        print(
            f"error: --require-measurements set but selected baseline "
            f"{args.baseline_name!r} contains no benchmark estimates under "
            f"{args.criterion_root} for target {args.target or '<unspecified>'}. "
            "A run with no selected base set cannot "
            "certify an A/B comparison.",
            file=sys.stderr,
        )
        return finish("error", EXIT_ERROR, "the selected baseline set is empty")

    if require_measurements and missing_baseline_ids:
        return refuse_missing_baseline()

    # Enforcing A/B runs judge only the exact selected base set. A persistent
    # Criterion root can contain change output from other named baselines or
    # bench targets; letting those rows into this run can create either a stale
    # regression or a reassuring comparison unrelated to the selected base.
    if require_measurements:
        change_files = [
            change_file
            for change_file in all_change_files
            if change_file_ids[change_file] in baseline_ids
        ]
    else:
        change_files = all_change_files
    change_ids = {change_file_ids[change_file] for change_file in change_files}
    missing_head_ids = sorted(baseline_ids - change_ids)

    def refuse_missing_head() -> int:
        print(
            f"error: --require-measurements set but {len(missing_head_ids)} "
            f"benchmark(s) in selected baseline {args.baseline_name!r} produced no "
            "head comparison:",
            file=sys.stderr,
        )
        for bench_id in missing_head_ids:
            print(
                f"  - {args.target + ': ' if args.target else ''}{bench_id}",
                file=sys.stderr,
            )
        print(
            "A partial A/B is not evidence that nothing regressed.",
            file=sys.stderr,
        )
        return finish(
            "error",
            EXIT_ERROR,
            f"{len(missing_head_ids)} selected baseline benchmarks have no head comparison",
        )

    if not change_files:
        if all_change_files:
            print(
                f"warn: no change/estimates.json for selected baseline "
                f"{args.baseline_name!r}; ignored {len(all_change_files)} unrelated "
                "comparison artifact(s)",
                file=sys.stderr,
            )
        else:
            print(f"warn: no change/estimates.json under {args.criterion_root}; "
                  "baseline missing?", file=sys.stderr)
        if require_measurements:
            if missing_head_ids:
                return refuse_missing_head()
            print(f"error: --require-measurements set but no change/estimates.json under "
                  f"{args.criterion_root}: the run produced no comparison, so it is not "
                  f"evidence that nothing regressed.", file=sys.stderr)
            return finish(
                "error", EXIT_ERROR, "the run produced no Criterion comparisons"
            )
        # Treat missing baseline as pass — first run on a bench has no comparison.
        report = (
            f"### `{args.arch}` — no baseline to compare\n\n"
            f"No `change/estimates.json` found; this is expected on the first "
            f"run for a bench. Future runs will gate against this run.\n\n"
            + "\n".join(render_run_provenance(provenance, []))
        )
        print(report)
        if args.out:
            args.out.write_text(report)
        return finish("pass", EXIT_PASS, "no baseline exists for this first run")

    results = []
    unjudged = []
    for cf in change_files:
        r = parse_bench(cf, args.criterion_root, args.baseline_name)
        if r is not None:
            results.append(r)
        else:
            # parse_bench warns and returns None for both malformed data and an
            # unresolvable baseline. Keep the file, not just the warning: the
            # count of comparisons the run INTENDED is the only thing that makes
            # a missing one visible downstream.
            unjudged.append(cf)

    if not results:
        if require_measurements and missing_head_ids:
            return refuse_missing_head()
        print("error: change files found but all failed to parse", file=sys.stderr)
        return finish(
            "error",
            EXIT_ERROR,
            "Criterion change files were present but none could be judged",
        )

    if args.require_provenance:
        incomplete_samples = [
            result.name
            for result in results
            if result.base_sample_count is None
            or result.base_sampling_mode is None
            or result.head_sample_count is None
            or result.head_sampling_mode is None
        ]
        if incomplete_samples:
            listed = ", ".join(incomplete_samples[:5])
            more = (
                f" (+{len(incomplete_samples) - 5} more)"
                if len(incomplete_samples) > 5
                else ""
            )
            print(
                f"error: --require-provenance set but {len(incomplete_samples)} "
                f"comparison(s) lack actual Criterion base/head sample metadata: "
                f"{listed}{more}.",
                file=sys.stderr,
            )
            return finish(
                "error",
                EXIT_ERROR,
                f"{len(incomplete_samples)} comparison(s) lack sample metadata",
            )

    # Completeness, not existence. Reconcile the selected base set against the
    # comparisons that were actually judged before attempting to pair them
    # with the reverse-order control.
    if require_measurements and missing_head_ids:
        return refuse_missing_head()

    if require_measurements and unjudged:
        listed = ", ".join(str(cf.parent.parent.relative_to(args.criterion_root))
                           for cf in unjudged[:5])
        more = f" (+{len(unjudged) - 5} more)" if len(unjudged) > 5 else ""
        print(f"error: --require-measurements set but {len(unjudged)} of "
              f"{len(change_files)} comparison(s) could not be judged "
              f"(unresolvable baseline or malformed data): {listed}{more}. A partial "
              f"A/B is not evidence that nothing regressed.", file=sys.stderr)
        return finish(
            "error",
            EXIT_ERROR,
            f"{len(unjudged)} of {len(change_files)} comparisons could not be judged",
        )

    if args.order_control_root is not None:
        try:
            reverse_results = parse_order_control(
                args.order_control_root,
                args.order_control_baseline_name,
                {result.name for result in results},
                require_samples=args.require_provenance,
            )
            results = order_balance_results(results, reverse_results)
        except (ArithmeticError, OSError, TypeError, ValueError) as error:
            print(f"error: invalid order-control evidence: {error}", file=sys.stderr)
            return finish(
                "error",
                EXIT_ERROR,
                f"invalid order-control evidence: {error}",
            )

    gating = [
        r for r in results
        if not r.is_informational(args.target, args.informational_target)
    ]
    gate_empty = empty_gate_reason(
        results, args.target, args.informational_target
    )

    excessive_order_bias = [
        result
        for result in gating
        if result.order_bias_bound_pct is not None
        and result.order_bias_bound_pct > FAIL_PCT
    ]
    order_bias_reason = None
    if excessive_order_bias:
        worst = max(
            excessive_order_bias,
            key=lambda result: result.order_bias_bound_pct or 0.0,
        )
        order_bias_reason = (
            f"{len(excessive_order_bias)} gated benchmark(s) have an AB/BA "
            f"order-bias bound above the {FAIL_PCT:.1f}% fail margin; worst is "
            f"{worst.name} at {worst.order_bias_bound_pct:.2f}%"
        )

    report = render_report(
        results,
        args.arch,
        args.target,
        args.informational_target,
        provenance,
        args.resolution,
        decision_suppressed_reason=order_bias_reason,
    )
    print(report)
    if args.out:
        args.out.write_text(report)

    if ambient is not None and not ambient.instrumentation_valid:
        reason = ambient.reason or "ambient instrumentation was invalid"
        print(f"error: {reason}", file=sys.stderr)
        return finish("error", EXIT_ERROR, reason)

    if ambient is not None and not ambient.measurable:
        reason = ambient.reason or "ambient load could not be assessed"
        print(
            f"NOT MEASURABLE: {reason}; no performance verdict rendered",
            file=sys.stderr,
        )
        return finish("not_measurable", EXIT_NOT_MEASURABLE, reason)

    if order_bias_reason is not None:
        print(
            f"NOT MEASURABLE: {order_bias_reason}; no performance verdict rendered",
            file=sys.stderr,
        )
        return finish(
            "not_measurable",
            EXIT_NOT_MEASURABLE,
            order_bias_reason,
            measurement_count=len(results),
        )

    if gate_empty is not None:
        # Reached only with measurements in hand: the empty-results case already
        # refused above. So this is policy, not a broken read, and the exit has
        # to be non-zero for the same reason exit 3 exists at all — the run
        # produced no verdict, and a zero here would be one.
        print(f"NO GATED MEASUREMENT: {gate_empty}", file=sys.stderr)
        return finish(
            "not_measurable",
            EXIT_NOT_MEASURABLE,
            gate_empty,
            measurement_count=len(results),
        )

    fails = sum(1 for r in gating if r.verdict(args.resolution) == "FAIL")
    if fails:
        return finish(
            "regression",
            EXIT_REGRESSION,
            f"{fails} gated regression(s) exceeded the fail threshold",
            measurement_count=len(results),
            regression_count=fails,
        )
    return finish(
        "pass",
        EXIT_PASS,
        "no gated regression exceeded the fail threshold",
        measurement_count=len(results),
        regression_count=0,
    )


if __name__ == "__main__":
    sys.exit(main())

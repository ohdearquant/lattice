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
      selected baseline set with no head comparison. An automated lane must not
      read "nothing was measured" as "nothing regressed".
  3 — not measurable: the automated lane's before/between/after ambient sample
      stream is missing, malformed, duplicated, or below BENCH_IDLE_FLOOR; or
      the measured AB/BA order-bias envelope is itself larger than the FAIL
      margin, so the run cannot distinguish a gate-sized source effect.

Status mode is opt-in. It requires exactly one perf-ambient-sample/v1 record for
each voting phase (before, between, after), ignores other phase labels, and
writes an atomic perf-bench-gate-status/v1 document. Ambient refusal is checked
before the final exit/status verdict, so exit 3 outranks pass or regression for
every target in the run. Excessive order bias is the other exit-3 cause.

--informational-target (lattice#714): quick-mode Criterion runs on
sub-microsecond micro-benches (lattice-embed's `simd` bench target) are
dominated by scheduler/thermal jitter rather than code changes — confirmed
by two same-toolchain quick-mode A/A runs on identical refs flipping FAIL/
WARN sign across dozens of entries (lattice#714). bench-compare isolates each
bench target in its own Criterion output root, passes that root's exact target
key through --target, and marks the key informational only when the reviewed
manifest says so. Every result in that root is still measured and reported,
but excluded from the FAIL/WARN gate and exit code. Full mode omits
--informational-target, so every result gates normally.
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
from datetime import UTC, datetime
from pathlib import Path

PROVENANCE_SCHEMA = "lattice-bench-provenance-v1"
MACHINE_STATE_SCHEMA = "lattice-machine-state-v1"
PHASE_LABELS = ("before first arm", "between order strata", "after final arm")
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
    "filters",
    "enforcement",
)

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


def write_gate_status(
    path: Path,
    *,
    arch: str,
    target: str | None,
    verdict: str,
    exit_code: int,
    reason: str,
    ambient: AmbientAssessment | None,
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

    `forward` is head₁/base₁. `reverse` is base₂/head₂. In log-ratio space,
    half their difference estimates the source effect while half their sum
    estimates the directional order effect. The Criterion endpoint envelope
    is transformed the same way, then widened by the complete order-effect
    envelope so within-run CIs cannot hide the run-order term.
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
    fields: dict[str, str]
    locks: tuple[str, ...]
    ambient_samples: tuple[str, ...]
    machine_states: tuple[dict[str, object], ...]


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

    if fields.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError(
            f"{path}: schema must be {PROVENANCE_SCHEMA!r}, got "
            f"{fields.get('schema')!r}"
        )
    missing = [field for field in PROVENANCE_FIELDS if field not in fields]
    if missing:
        raise ValueError(f"{path}: missing provenance fields: {', '.join(missing)}")
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

    fields.pop("schema")
    return RunProvenance(
        fields=fields,
        locks=tuple(repeated["lock"]),
        ambient_samples=tuple(repeated["ambient"]),
        machine_states=machine_states,
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
    """Validate, then remove exact baseline dirs before copying a fresh base set."""
    root_resolved = _checked_criterion_root(root)
    baseline_files = find_selected_baseline_files(root, baseline_name)
    deletion_plan: set[Path] = set()

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

    for baseline_dir in sorted(deletion_plan):
        shutil.rmtree(baseline_dir)

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
    """Parse the B→A half of an enforcing ABBA run, refusing partial evidence."""
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
        if bench_id in expected_ids:
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
    if decision_suppressed_reason is None and warns:
        qualifier = (
            "order-bias-widened ABBA lower bound"
            if order_balanced
            else "two-sided-95%-CI lower bound"
        )
        lines.append(
            f"**⚠ {len(warns)} WARN** (regression {WARN_PCT}-{FAIL_PCT}% by "
            f"the same {qualifier})"
        )
    if decision_suppressed_reason is None and unattributable:
        lines.append(
            f"**◻ {len(unattributable)} UNATTRIBUTABLE** (quick-resolution "
            f"{WARN_PCT}-{FAIL_PCT}% warn band is narrower than the harness's "
            "measured arm-order bias)"
        )
    if decision_suppressed_reason is None and wins:
        lines.append(f"**🚀 {len(wins)} confirmed improvement**")
    if decision_suppressed_reason is None and not (
        fails or warns or unattributable or wins
    ):
        lines.append(f"✅ All {len(gated)} gated benches within noise band (±{WARN_PCT}%)")
    lines.append("")
    lines.extend(render_run_provenance(provenance, results))

    if decision_suppressed_reason is None and (
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
            f"**ℹ️ {len(info)} informational** (below quick-mode resolution — "
            f"lattice-embed SIMD micro-benches, tracked in #714; not gated here, "
            f"re-run `--full` for a gated verdict)"
        )
        if decision_suppressed_reason is None and (
            info_fails or info_warns or info_unattributable or info_wins
        ):
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
            for r in sorted(
                info_fails + info_warns + info_unattributable + info_wins,
                key=lambda r: -r.ci_low_pct,
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
    if decision_suppressed_reason is None:
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
    else:
        lines.append(
            "_The measurements are shown for diagnosis only; WARN/FAIL "
            "classification was suppressed._"
        )
    if informational_target is not None:
        lines.append(
            f"_Target `{informational_target}` classified informational-only in quick mode "
            f"(lattice#714); its {len(info)} measurement(s) are excluded from the verdict._\n"
        )
    else:
        lines.append("")
    return "\n".join(lines)


def _fabricate_baseline(bench_dir: Path, baseline_dirname: str,
                        base_ns: float = 90.0) -> None:
    """Write one fake Criterion named-baseline estimate for --selftest."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / baseline_dirname).mkdir(parents=True, exist_ok=True)
    (bench_dir / baseline_dirname / "estimates.json").write_text(
        json.dumps({"mean": {"point_estimate": base_ns}}))


def _fabricate_bench(bench_dir: Path, baseline_dirname: str,
                     point: float = 0.10, ci_low: float = 0.05, ci_high: float = 0.15,
                     new_ns: float = 100.0, base_ns: float = 90.0) -> None:
    """Write a fake Criterion bench dir (new/, <baseline_dirname>/, change/) for --selftest."""
    _fabricate_baseline(bench_dir, baseline_dirname, base_ns)
    (bench_dir / "new").mkdir(exist_ok=True)
    (bench_dir / "new" / "estimates.json").write_text(
        json.dumps({"mean": {"point_estimate": new_ns}}))
    (bench_dir / "change").mkdir(exist_ok=True)
    (bench_dir / "change" / "estimates.json").write_text(json.dumps({
        "mean": {
            "point_estimate": point,
            "confidence_interval": {"lower_bound": ci_low, "upper_bound": ci_high},
        }
    }))


def _fabricate_sample(artifact_dir: Path, count: int, mode: str) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "sample.json").write_text(json.dumps({
        "sampling_mode": mode,
        "iters": [float(index + 1) for index in range(count)],
        "times": [float(index + 1) for index in range(count)],
    }))


def run_selftest() -> int:
    """Fabricate both baseline layouts + an orphan case; assert the parser handles each.

    Regression coverage for #545: a default-rotation `base/` layout, a named-baseline
    `compare-base/` layout (what bench-compare.sh actually produces), and a `change/`
    dir with no resolvable baseline at all (must WARN by bench name, not silently skip).
    Plus the findings on PR #548: when BOTH base/ and the named baseline exist,
    the named baseline must win (dirty local tree with stale base/); when multiple
    unrelated sibling baselines exist and none match, the gate must refuse to guess.
    """
    import contextlib
    import io
    import os
    import subprocess
    import tempfile

    failures: list[str] = []

    # lattice#1137: a fixed-order single Criterion pair can report a narrow,
    # gate-crossing regression on byte-identical source when both arms drift in
    # the same direction. ABBA supplies the mirrored B→A ratio. The balanced
    # source effect must collapse to zero while retaining the observed order
    # term as an explicit uncertainty bound.
    directional_drift = BenchResult(
        name="softmax_attention/512",
        point=0.10,
        ci_low=0.095,
        ci_high=0.105,
        new_ns=110.0,
        old_ns=100.0,
        head_sample_count=20,
        head_sampling_mode="Flat",
        base_sample_count=20,
        base_sampling_mode="Flat",
    )
    reverse_directional_drift = BenchResult(
        name="softmax_attention/512",
        point=0.10,
        ci_low=0.095,
        ci_high=0.105,
        new_ns=133.1,
        old_ns=121.0,
        head_sample_count=20,
        head_sampling_mode="Flat",
        base_sample_count=20,
        base_sampling_mode="Flat",
    )
    if directional_drift.verdict() != "FAIL":
        failures.append(
            "order-balance fixture no longer reproduces the old fixed-order false FAIL"
        )
    balanced_drift = order_balance_pair(
        directional_drift, reverse_directional_drift
    )
    if balanced_drift.verdict() == "FAIL" or abs(balanced_drift.point) > 1e-12:
        failures.append(
            "order-balance: identical-source directional drift still produced "
            f"{balanced_drift.verdict()} at {balanced_drift.point_pct:+.6f}%"
        )
    if (
        balanced_drift.order_bias_bound_pct is None
        or balanced_drift.order_bias_bound_pct <= FAIL_PCT
    ):
        failures.append(
            "order-balance: gate-sized directional drift was not retained as "
            "a fail-closed order-bias bound"
        )

    # A real 20% source slowdown under a 2% directional drift remains
    # distinguishable after the same correction. This prevents the control
    # envelope from becoming an unconditional regression exemption.
    true_regression_forward = BenchResult(
        name="softmax_attention/512",
        point=0.224,
        ci_low=0.222,
        ci_high=0.226,
        new_ns=122.4,
        old_ns=100.0,
        head_sample_count=20,
        head_sampling_mode="Flat",
        base_sample_count=20,
        base_sampling_mode="Flat",
    )
    true_regression_reverse = BenchResult(
        name="softmax_attention/512",
        point=-0.15,
        ci_low=-0.152,
        ci_high=-0.148,
        new_ns=106.1208,
        old_ns=124.848,
        head_sample_count=20,
        head_sampling_mode="Flat",
        base_sample_count=20,
        base_sampling_mode="Flat",
    )
    balanced_regression = order_balance_pair(
        true_regression_forward, true_regression_reverse
    )
    if balanced_regression.verdict() != "FAIL":
        failures.append(
            "order-balance: a true 20% regression under 2% drift did not remain "
            f"a FAIL: point={balanced_regression.point_pct:+.2f}%, "
            f"bound=[{balanced_regression.ci_low_pct:+.2f}%, "
            f"{balanced_regression.ci_high_pct:+.2f}%]"
        )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        default_dir = root / "grp_a" / "bench_default"
        _fabricate_bench(default_dir, "base")

        named_dir = root / "grp_b" / "bench_named"
        _fabricate_bench(named_dir, "compare-base")

        orphan_dir = root / "grp_c" / "bench_orphan"
        orphan_dir.mkdir(parents=True)
        (orphan_dir / "new").mkdir()
        (orphan_dir / "new" / "estimates.json").write_text(
            json.dumps({"mean": {"point_estimate": 100.0}}))
        (orphan_dir / "change").mkdir()
        (orphan_dir / "change" / "estimates.json").write_text(json.dumps({
            "mean": {"point_estimate": 0.1,
                     "confidence_interval": {"lower_bound": 0.05, "upper_bound": 0.15}}
        }))

        # Finding 1: both base/ and compare-base/ present with different
        # values — the named baseline must win over stale base/.
        both_dir = root / "grp_d" / "bench_both"
        _fabricate_bench(both_dir, "compare-base", base_ns=100.0)
        (both_dir / "base").mkdir()
        (both_dir / "base" / "estimates.json").write_text(
            json.dumps({"mean": {"point_estimate": 1.0}}))  # stale decoy

        # Finding 2: multiple unrelated sibling baselines, none matching
        # the requested name — must skip loudly, not guess.
        multi_dir = root / "grp_e" / "bench_multi"
        _fabricate_bench(multi_dir, "old-run-1")
        (multi_dir / "old-run-2").mkdir()
        (multi_dir / "old-run-2" / "estimates.json").write_text(
            json.dumps({"mean": {"point_estimate": 42.0}}))

        change_files = find_change_files(root)
        if len(change_files) != 5:
            failures.append(f"expected 5 change/estimates.json, found {len(change_files)}")

        stderr_buf = io.StringIO()
        results: dict[str, BenchResult] = {}
        with contextlib.redirect_stderr(stderr_buf):
            for cf in change_files:
                r = parse_bench(cf, root, baseline_name="compare-base")
                if r is not None:
                    results[r.name] = r
        stderr_text = stderr_buf.getvalue()

        if "grp_a/bench_default" not in results:
            failures.append("default base/ layout: bench not parsed")
        if "grp_b/bench_named" not in results:
            failures.append("named compare-base/ layout: bench not parsed")
        if "grp_c/bench_orphan" in results:
            failures.append("orphan bench (no resolvable baseline) was parsed instead of skipped")
        if "grp_c/bench_orphan" not in stderr_text:
            failures.append("orphan bench did not emit a warning naming the bench")

        both = results.get("grp_d/bench_both")
        if both is None:
            failures.append("both-dirs layout: bench not parsed")
        elif both.old_ns != 100.0:
            failures.append(f"both-dirs layout: expected named-baseline old_ns=100.0 "
                            f"(compare-base/), got {both.old_ns} (stale base/ shadowed it)")

        if "grp_e/bench_multi" in results:
            failures.append("multi-sibling layout: gate guessed a baseline instead of refusing")
        if "bench_multi" not in stderr_text or "refusing to guess" not in stderr_text:
            failures.append("multi-sibling layout: no loud refusal warning emitted")

        # lattice#714 / lattice#1090: classification follows the target-qualified
        # Criterion root, never a slash-derived group prefix. Criterion 0.5 permits
        # `/` in group names and also uses `/` between group/function/parameter, so
        # splitting the rendered ID cannot recover the group boundary.
        noisy_dir = root / "grp_f" / "noisy_fail"
        _fabricate_bench(noisy_dir, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)
        real_dir = root / "grp_g" / "real_fail"
        _fabricate_bench(real_dir, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)
        slash_dir = root / "slash" / "is" / "legal" / "noisy_fail"
        _fabricate_bench(slash_dir, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)

        for cf in find_change_files(root):
            r = parse_bench(cf, root, baseline_name="compare-base")
            if r is not None:
                results[r.name] = r

        embed_target = "lattice-embed:simd"
        inference_target = "lattice-inference:elementwise_cpu_bench"
        needed = {"grp_f/noisy_fail", "grp_g/real_fail", "slash/is/legal/noisy_fail"}
        if not needed.issubset(results):
            failures.append("target-qualified fixture: benches not parsed")
        else:
            noisy = results["grp_f/noisy_fail"]
            real = results["grp_g/real_fail"]
            slash = results["slash/is/legal/noisy_fail"]
            if not noisy.is_informational(embed_target, embed_target):
                failures.append("target-qualified: demoted target did not classify informational")
            if real.is_informational(inference_target, embed_target):
                failures.append("target-qualified: gated target crossed into demotion")
            if not slash.is_informational(embed_target, embed_target):
                failures.append("target-qualified: slash-containing group was misclassified")

            embed_report = render_report(
                [noisy, slash], "selftest-embed", embed_target, embed_target)
            inference_report = render_report(
                [real], "selftest-inference", inference_target, embed_target)
            if "ℹ️" not in embed_report or "slash/is/legal/noisy_fail" not in embed_report:
                failures.append("target-qualified: informational report lost slash-group bench")
            if "❌" not in inference_report or "grp_g/real_fail" not in inference_report:
                failures.append("target-qualified: gated FAIL missing from inference report")
            if "bench-allow-regression" in embed_report + inference_report:
                failures.append("rendered report advertises an unsupported label override")

        # The shell owns manifest membership. Python receives only the exact
        # target identity selected by that shell path, so there is no second
        # parser whose group-name semantics can drift. The controlled manifest
        # probes cover CRLF, surrounding whitespace, comments, duplicate
        # entries and C-locale sorting; malformed or whitespace-bearing keys
        # must refuse rather than demote.
        helper = Path(__file__).resolve().parent / "lib" / "bench-informational-targets.sh"
        # The reviewed demoted-target set, duplicated here on purpose:
        # the selftest compares this against the manifest itself (via
        # --print-targets), so a target added to only ONE side —
        # manifest or this expectation — fails the selftest.
        approved_targets = frozenset({"lattice-embed:simd"})
        if not helper.exists():
            failures.append(f"manifest-handoff: shell helper missing at {helper}")
        else:
            raw_proc = subprocess.run(
                ["bash", str(helper), "--print-targets"],
                capture_output=True, text=True, timeout=30,
            )
            raw_targets = frozenset(
                ln.strip() for ln in raw_proc.stdout.splitlines() if ln.strip()
            )
            if raw_proc.returncode != 0:
                failures.append(
                    f"manifest-handoff: --print-targets exited "
                    f"{raw_proc.returncode}: {raw_proc.stderr}"
                )
            elif raw_targets != approved_targets:
                failures.append(
                    "manifest-handoff: manifest and selftest expectation "
                    f"disagree — manifest-only: {sorted(raw_targets - approved_targets)}, "
                    f"expectation-only: {sorted(approved_targets - raw_targets)}. "
                    "Every demotion change must update both sides in one PR."
                )
            demoted_proc = subprocess.run(
                ["bash", str(helper), "--is-informational", embed_target],
                capture_output=True, text=True, timeout=30,
            )
            gated_proc = subprocess.run(
                ["bash", str(helper), "--is-informational", inference_target],
                capture_output=True, text=True, timeout=30,
            )
            if demoted_proc.returncode != 0:
                failures.append(
                    f"manifest-handoff: demoted-target probe exited "
                    f"{demoted_proc.returncode}: {demoted_proc.stderr}"
                )
            elif gated_proc.returncode != 1:
                failures.append(
                    "manifest-handoff: non-demoted target did not return the "
                    f"expected predicate miss (1): {gated_proc.returncode}"
                )

            policy_dir = root / "manifest-policy"
            policy_dir.mkdir(parents=True, exist_ok=True)
            normalized_manifest = policy_dir / "normalized.txt"
            normalized_manifest.write_bytes(
                b"  z-target  # comment\r\n"
                b"\r\n"
                b"a-target \r\n"
                b"z-target\r\n"
            )
            policy_env = {
                **os.environ,
                "INFO_TARGETS_MANIFEST": str(normalized_manifest),
            }
            normalized = subprocess.run(
                ["bash", str(helper), "--print-targets"],
                capture_output=True, text=True, timeout=30, env=policy_env,
            )
            if normalized.returncode != 0 or normalized.stdout != "a-target\nz-target\n":
                failures.append(
                    "manifest-normalization: CRLF/whitespace/dedup/C-sort drifted: "
                    f"rc={normalized.returncode}, stdout={normalized.stdout!r}, "
                    f"stderr={normalized.stderr!r}"
                )

            malformed_manifest = policy_dir / "malformed.txt"
            malformed_manifest.write_text("one-target two-targets\n")
            malformed_env = {
                **os.environ,
                "INFO_TARGETS_MANIFEST": str(malformed_manifest),
            }
            malformed = subprocess.run(
                ["bash", str(helper), "--print-targets"],
                capture_output=True, text=True, timeout=30, env=malformed_env,
            )
            if malformed.returncode != 2:
                failures.append(
                    "manifest-normalization: malformed policy did not fail closed "
                    f"(rc={malformed.returncode})"
                )
            bad_key = subprocess.run(
                ["bash", str(helper), "--is-informational", "bad target"],
                capture_output=True, text=True, timeout=30,
            )
            if bad_key.returncode != 2:
                failures.append(
                    "manifest-normalization: whitespace-bearing target key did not refuse"
                )

    # --require-measurements: the lane must not read "nothing measured" as
    # "nothing regressed" (#1105). bench-compare.sh creates the criterion
    # directory itself before benching, and the cargo pipelines swallow bench
    # failures, so an EMPTY-but-present root is the realistic failure shape.
    with tempfile.TemporaryDirectory() as td:
        gate = Path(__file__).resolve()
        empty_root = Path(td) / "empty" / "criterion"
        empty_root.mkdir(parents=True)

        def _run(root: Path, *extra: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, str(gate), str(root), "selftest-arch", *extra],
                capture_output=True, text=True, timeout=60,
            )

        def _prepare(root: Path, mode: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [
                    sys.executable,
                    str(gate),
                    str(root),
                    "--baseline-name",
                    "compare-base",
                    mode,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

        # Without the flag, an absent baseline stays a pass (first-run semantics).
        if _run(empty_root).returncode != 0:
            failures.append("require-measurements: empty root without the flag "
                            "must still exit 0 (first-run semantics changed)")
        # With it, the same empty root must refuse to certify.
        if _run(empty_root, "--require-measurements").returncode != 2:
            failures.append("require-measurements: empty criterion root exited "
                            "0 with the flag set — the lane can go green having "
                            "measured nothing (the #1105 fail-open)")
        empty_informational = _run(
            empty_root,
            "--require-measurements",
            "--target",
            "lattice-embed:simd",
            "--informational-target",
            "lattice-embed:simd",
        )
        if empty_informational.returncode != 2:
            failures.append(
                "require-measurements: an empty informational target exited "
                "0 with enforcement enabled"
            )
        if "for target lattice-embed:simd" not in empty_informational.stderr:
            failures.append(
                "require-measurements: empty informational target was not "
                "reported with its target identity"
            )

        # A stale comparison against a DIFFERENT baseline is not a substitute
        # for the exact selected base set. Mutation-sensitive: condition
        # selected-set filtering on a non-empty set (or omit the empty-set
        # refusal) and this valid base/new/change tree exits 0.
        wrong_baseline_root = Path(td) / "wrong-baseline" / "criterion"
        _fabricate_bench(
            wrong_baseline_root / "rms_norm" / "4096", "base"
        )
        wrong_baseline = _run(
            wrong_baseline_root, "--require-measurements"
        )
        if wrong_baseline.returncode != 2:
            failures.append(
                "require-measurements: stale default-base comparison certified "
                "an empty selected compare-base set"
            )
        if "selected baseline 'compare-base' contains no benchmark estimates" not in (
            wrong_baseline.stderr
        ):
            failures.append(
                "require-measurements: empty selected-set refusal did not name "
                "the selected baseline"
            )

        # A real gating comparison must still pass, or the flag is just a brake.
        ok_root = Path(td) / "ok" / "criterion"
        _fabricate_bench(ok_root / "grp_ok" / "bench_ok", "compare-base")
        if _run(ok_root, "--require-measurements").returncode != 0:
            failures.append("require-measurements: a parsed gating comparison "
                            "was rejected — the flag over-fails")
        missing_order_control = _run(
            ok_root,
            "--require-measurements",
            "--require-order-balance",
        )
        if (
            missing_order_control.returncode != EXIT_ERROR
            or "needs --order-control-root" not in missing_order_control.stderr
        ):
            failures.append(
                "order-balance: enforcing request without reverse evidence did "
                "not fail closed"
            )

        drift_root = Path(td) / "directional-drift" / "criterion"
        drift_control = Path(td) / "directional-drift-control" / "criterion"
        _fabricate_bench(
            drift_root / "softmax_attention" / "512",
            "compare-base",
            point=0.10,
            ci_low=0.095,
            ci_high=0.105,
            new_ns=110.0,
            base_ns=100.0,
        )
        _fabricate_bench(
            drift_control / "softmax_attention" / "512",
            "compare-head",
            point=0.10,
            ci_low=0.095,
            ci_high=0.105,
            new_ns=133.1,
            base_ns=121.0,
        )
        drift_gate = _run(
            drift_root,
            "--require-measurements",
            "--require-order-balance",
            "--order-control-root",
            str(drift_control),
        )
        if drift_gate.returncode != EXIT_NOT_MEASURABLE:
            failures.append(
                "order-balance: identical-source directional drift exited "
                f"{drift_gate.returncode}, expected NOT_MEASURABLE (3)"
            )
        if "order-bias bound above" not in drift_gate.stderr:
            failures.append(
                "order-balance: directional-drift refusal did not identify the "
                "order-bias evidence"
            )

        regression_root = Path(td) / "true-regression" / "criterion"
        regression_control = (
            Path(td) / "true-regression-control" / "criterion"
        )
        _fabricate_bench(
            regression_root / "softmax_attention" / "512",
            "compare-base",
            point=0.224,
            ci_low=0.222,
            ci_high=0.226,
            new_ns=122.4,
            base_ns=100.0,
        )
        _fabricate_bench(
            regression_control / "softmax_attention" / "512",
            "compare-head",
            point=-0.15,
            ci_low=-0.152,
            ci_high=-0.148,
            new_ns=106.1208,
            base_ns=124.848,
        )
        regression_gate = _run(
            regression_root,
            "--require-measurements",
            "--require-order-balance",
            "--order-control-root",
            str(regression_control),
        )
        if regression_gate.returncode != EXIT_REGRESSION:
            failures.append(
                "order-balance: true 20% regression under 2% drift exited "
                f"{regression_gate.returncode}, expected regression (1)"
            )

        incomplete_control = (
            Path(td) / "incomplete-order-control" / "criterion"
        )
        _fabricate_bench(
            incomplete_control / "different_bench",
            "compare-head",
        )
        incomplete_order = _run(
            drift_root,
            "--require-measurements",
            "--require-order-balance",
            "--order-control-root",
            str(incomplete_control),
        )
        if incomplete_order.returncode != EXIT_ERROR:
            failures.append(
                "order-balance: mismatched reverse benchmark set did not fail "
                "closed with exit 2"
            )

        for malformed_name, malformed_value, malformed_field in (
            ("string-point", "corrupt", "point_estimate"),
            ("boolean-ci", True, "lower_bound"),
        ):
            malformed_control = (
                Path(td) / f"malformed-order-control-{malformed_name}" / "criterion"
            )
            malformed_bench = malformed_control / "softmax_attention" / "512"
            _fabricate_bench(
                malformed_bench,
                "compare-head",
                point=0.10,
                ci_low=0.095,
                ci_high=0.105,
                new_ns=133.1,
                base_ns=121.0,
            )
            malformed_change = malformed_bench / "change" / "estimates.json"
            malformed_payload = json.loads(malformed_change.read_text())
            if malformed_field == "point_estimate":
                malformed_payload["mean"]["point_estimate"] = malformed_value
            else:
                malformed_payload["mean"]["confidence_interval"][
                    malformed_field
                ] = malformed_value
            malformed_change.write_text(json.dumps(malformed_payload))
            malformed_order = _run(
                drift_root,
                "--require-measurements",
                "--require-order-balance",
                "--order-control-root",
                str(malformed_control),
            )
            if malformed_order.returncode != EXIT_ERROR:
                failures.append(
                    f"order-balance: malformed reverse {malformed_name} exited "
                    f"{malformed_order.returncode}, expected input error (2)"
                )
            if "Traceback" in malformed_order.stderr:
                failures.append(
                    f"order-balance: malformed reverse {malformed_name} escaped "
                    "as an uncaught exception"
                )

        # A target can be intentionally all-informational only when the exact
        # target key accompanies both the root and the demotion. This remains a
        # measured target and is valid in an enforcing multi-target run; another
        # target supplies the gating comparisons. A mismatched key must refuse.
        info_root = Path(td) / "info" / "criterion"
        _fabricate_bench(info_root / "grp_info" / "bench_info", "compare-base")
        if _run(info_root, "--require-measurements",
                "--target", "lattice-embed:simd",
                "--informational-target", "lattice-embed:simd").returncode != 0:
            failures.append("require-measurements: an explicitly qualified "
                            "informational target was rejected")
        if _run(info_root, "--require-measurements",
                "--target", "lattice-inference:elementwise_cpu_bench",
                "--informational-target", "lattice-embed:simd").returncode != 2:
            failures.append("target-qualified: mismatched informational target "
                            "did not fail closed")

        # PARTIAL is the shape the first fix missed: asking whether ANY judgeable
        # comparison exists passes a run where one target compared cleanly and the
        # other produced malformed selected-baseline data. The unmeasured target
        # is exactly the one a green exit would be vouching for.
        mixed_root = Path(td) / "mixed" / "criterion"
        _fabricate_bench(mixed_root / "grp_ok" / "bench_ok", "compare-base")
        mixed_orphan = mixed_root / "grp_bad" / "bench_orphan"
        (mixed_orphan / "compare-base").mkdir(parents=True)
        (mixed_orphan / "compare-base" / "estimates.json").write_text("{}")
        (mixed_orphan / "new").mkdir(parents=True)
        (mixed_orphan / "new" / "estimates.json").write_text(
            json.dumps({"mean": {"point_estimate": 100.0}}))
        (mixed_orphan / "change").mkdir(parents=True)
        (mixed_orphan / "change" / "estimates.json").write_text(json.dumps({
            "mean": {"point_estimate": 0.10,
                     "confidence_interval": {"lower_bound": 0.05, "upper_bound": 0.15}}
        }))
        mixed = _run(mixed_root, "--require-measurements")
        if mixed.returncode != 2:
            failures.append("require-measurements: a run with one judgeable and one "
                            "unresolvable comparison exited "
                            f"{mixed.returncode} — a partial A/B was certified")
        if "bench_orphan" not in mixed.stderr:
            failures.append("require-measurements: the partial-run refusal did not "
                            "name the unjudged bench")
        # Without the flag the same mixed root stays a pass: the reporter is
        # allowed to render what it has.
        if _run(mixed_root).returncode != 0:
            failures.append("require-measurements: the mixed root without the flag "
                            "must still exit 0 (reporter behavior changed)")

        # lattice#1204: completeness is selected-base minus head-change, not
        # merely change-found minus change-parsed. These use actual IDs from
        # the shipping inference/embed targets, including Criterion's nested
        # function/parameter path. Mutation-sensitive: remove the set
        # subtraction (or exempt informational groups from it) and this exits
        # 0 even though two base measurements vanished from the head.
        coverage_root = Path(td) / "coverage" / "criterion"
        _fabricate_bench(coverage_root / "rms_norm" / "896", "compare-base")
        _fabricate_baseline(coverage_root / "rms_norm" / "4096", "compare-base")
        _fabricate_baseline(
            coverage_root / "simd_dot_product" / "scalar" / "384",
            "compare-base",
        )
        _fabricate_bench(
            coverage_root / "renamed_rms_norm" / "4096", "compare-base"
        )
        _fabricate_bench(
            coverage_root / "stale_default" / "bench", "base"
        )
        _fabricate_bench(
            coverage_root / "stale_named" / "bench", "previous-run"
        )
        # An estimates file directly under the selected baseline directory has
        # no benchmark ID and is unrelated Criterion-tree debris, not a bench.
        (coverage_root / "compare-base").mkdir(parents=True)
        (coverage_root / "compare-base" / "estimates.json").write_text("{}")

        coverage = _run(
            coverage_root,
            "--require-measurements",
            "--target",
            "lattice-inference:elementwise_cpu_bench",
        )
        if coverage.returncode != 2:
            failures.append(
                "baseline-completeness: missing head benchmarks exited "
                f"{coverage.returncode} instead of 2"
            )
        for expected_id in ("rms_norm/4096", "simd_dot_product/scalar/384"):
            qualified = (
                "  - lattice-inference:elementwise_cpu_bench: "
                f"{expected_id}"
            )
            if qualified not in coverage.stderr:
                failures.append(
                    "baseline-completeness: refusal did not name exact missing "
                    f"bench ID {expected_id!r}"
                )
        for unrelated_id in ("stale_default/bench", "stale_named/bench"):
            if unrelated_id in coverage.stderr:
                failures.append(
                    "baseline-completeness: unrelated baseline tree leaked into "
                    f"selected set: {unrelated_id}"
                )
            if unrelated_id in coverage.stdout:
                failures.append(
                    "baseline-completeness: unrelated stale comparison was "
                    f"judged by the enforcing run: {unrelated_id}"
                )
        if _run(
            coverage_root,
        ).returncode != 0:
            failures.append(
                "baseline-completeness: reporter mode rejected missing head "
                "benchmarks without --require-measurements"
            )

        informational_missing_root = (
            Path(td) / "coverage-informational" / "criterion"
        )
        _fabricate_baseline(
            informational_missing_root / "simd_dot_product" / "scalar" / "384",
            "compare-base",
        )
        informational_missing = _run(
            informational_missing_root,
            "--require-measurements",
            "--target",
            "lattice-embed:simd",
            "--informational-target",
            "lattice-embed:simd",
        )
        if informational_missing.returncode != 2:
            failures.append(
                "baseline-completeness: a missing informational target "
                "did not fail measurement completeness"
            )
        if (
            "  - lattice-embed:simd: simd_dot_product/scalar/384"
            not in informational_missing.stderr
        ):
            failures.append(
                "baseline-completeness: missing informational benchmark was "
                "not reported with its target identity"
            )

        # The CLI-selected name can itself be a relative path. Only that exact
        # suffix contributes to completeness; a stale compare-base tree must
        # not contaminate a run selecting snapshot/arm.
        custom_root = Path(td) / "custom-baseline" / "criterion"
        _fabricate_bench(custom_root / "rms_norm" / "896", "snapshot/arm")
        _fabricate_baseline(custom_root / "gelu" / "4096", "snapshot/arm")
        _fabricate_baseline(
            custom_root / "stale_compare_base" / "4096", "compare-base"
        )
        custom = _run(
            custom_root,
            "--require-measurements",
            "--baseline-name",
            "snapshot/arm",
        )
        if custom.returncode != 2 or "  - gelu/4096" not in custom.stderr:
            failures.append(
                "baseline-completeness: nested --baseline-name did not select "
                "and report the missing gelu/4096 benchmark"
            )
        if "stale_compare_base/4096" in custom.stderr:
            failures.append(
                "baseline-completeness: stale non-selected compare-base tree "
                "contaminated custom selected baseline"
            )

        baseline_copy_root = Path(td) / "baseline-copy" / "criterion"
        stale_selected = baseline_copy_root / "old_target" / "old_bench"
        _fabricate_bench(stale_selected, "compare-base")
        preserved_other = baseline_copy_root / "other_target" / "other_bench"
        _fabricate_bench(preserved_other, "previous-run")
        prepared_copy = _prepare(
            baseline_copy_root, "--prepare-baseline-copy"
        )
        if prepared_copy.returncode != 0:
            failures.append(
                f"baseline-copy freshness: prepare step failed: {prepared_copy.stderr}"
            )
        if (stale_selected / "compare-base").exists():
            failures.append(
                "baseline-copy freshness: stale exact selected baseline survived"
            )
        if not (stale_selected / "change" / "estimates.json").exists():
            failures.append(
                "baseline-copy freshness: comparison output was removed before "
                "the fresh selected set existed"
            )
        if not (preserved_other / "previous-run" / "estimates.json").exists():
            failures.append(
                "baseline-copy freshness: unrelated named baseline was removed"
            )

        atomic_baseline_root = Path(td) / "atomic-baseline-copy" / "criterion"
        atomic_safe_baseline = atomic_baseline_root / "a_safe" / "1"
        _fabricate_baseline(atomic_safe_baseline, "compare-base")
        atomic_unsafe_baseline = atomic_baseline_root / "z_unsafe" / "1"
        (atomic_unsafe_baseline / "compare-base").mkdir(parents=True)
        outside_baseline = Path(td) / "outside-baseline-estimates.json"
        outside_baseline.write_text('{"mean":{"point_estimate":90.0}}')
        (atomic_unsafe_baseline / "compare-base" / "estimates.json").symlink_to(
            outside_baseline
        )
        atomic_baseline_prepare = _prepare(
            atomic_baseline_root, "--prepare-baseline-copy"
        )
        if atomic_baseline_prepare.returncode != 2:
            failures.append(
                "baseline-copy freshness: unsafe later baseline was not refused"
            )
        if not (
            atomic_safe_baseline / "compare-base" / "estimates.json"
        ).exists():
            failures.append(
                "baseline-copy freshness: cleanup partially deleted a safe "
                "baseline before refusing a later unsafe path"
            )

        # The in-place HEAD freshness step removes only new/change siblings of
        # exact selected-baseline benches, preserving that baseline and
        # unrelated Criterion trees. Mutation-sensitive: stop removing the
        # stale same-path change below and the deleted rms_norm/4096 appears
        # judged, defeating the completeness check.
        freshness_root = Path(td) / "freshness" / "criterion"
        removed_bench = freshness_root / "rms_norm" / "4096"
        _fabricate_bench(removed_bench, "compare-base")
        unrelated_bench = freshness_root / "legacy_group" / "legacy_bench"
        _fabricate_bench(unrelated_bench, "previous-run")
        sibling_root = Path(td) / "freshness-sibling" / "criterion"
        sibling_bench = sibling_root / "simd_dot_product" / "scalar" / "384"
        _fabricate_bench(sibling_bench, "compare-base")
        prepared = _prepare(freshness_root, "--prepare-head")
        if prepared.returncode != 0:
            failures.append(
                f"head-freshness: prepare step failed: {prepared.stderr}"
            )
        if not (removed_bench / "compare-base" / "estimates.json").exists():
            failures.append("head-freshness: selected baseline was removed")
        if (removed_bench / "new").exists() or (removed_bench / "change").exists():
            failures.append(
                "head-freshness: stale selected-bench new/change artifacts survived"
            )
        if not (unrelated_bench / "new" / "estimates.json").exists():
            failures.append(
                "head-freshness: unrelated named-baseline benchmark was cleaned"
            )
        if not (sibling_bench / "change" / "estimates.json").exists():
            failures.append(
                "head-freshness: preparing one target root touched its sibling"
            )
        freshness = _run(freshness_root, "--require-measurements")
        if freshness.returncode != 2 or "  - rms_norm/4096" not in freshness.stderr:
            failures.append(
                "head-freshness: deleted head benchmark was masked by stale "
                "same-path output"
            )

        unsafe_root = Path(td) / "not-the-criterion-root"
        _fabricate_bench(
            unsafe_root / "rms_norm" / "4096", "compare-base"
        )
        unsafe_prepare = _prepare(unsafe_root, "--prepare-head")
        if unsafe_prepare.returncode != 2:
            failures.append(
                "head-freshness: cleanup outside a criterion-named root was not refused"
            )
        if not (unsafe_root / "rms_norm" / "4096" / "change").exists():
            failures.append(
                "head-freshness: refused cleanup still removed an artifact"
            )

        symlink_root = Path(td) / "symlink-safety" / "criterion"
        symlink_bench = symlink_root / "rms_norm" / "4096"
        _fabricate_baseline(symlink_bench, "compare-base")
        outside = Path(td) / "outside-change"
        outside.mkdir()
        outside_sentinel = outside / "keep.txt"
        outside_sentinel.write_text("must survive")
        (symlink_bench / "change").symlink_to(outside, target_is_directory=True)
        symlink_prepare = _prepare(symlink_root, "--prepare-head")
        if symlink_prepare.returncode != 2:
            failures.append(
                "head-freshness: symlinked cleanup artifact was not refused"
            )
        if not outside_sentinel.exists():
            failures.append(
                "head-freshness: cleanup followed a symlink outside criterion root"
            )

        atomic_head_root = Path(td) / "atomic-head-cleanup" / "criterion"
        atomic_safe_head = atomic_head_root / "a_safe" / "1"
        _fabricate_bench(atomic_safe_head, "compare-base")
        atomic_unsafe_head = atomic_head_root / "z_unsafe" / "1"
        _fabricate_baseline(atomic_unsafe_head, "compare-base")
        (atomic_unsafe_head / "new").mkdir()
        (atomic_unsafe_head / "new" / "estimates.json").write_text("{}")
        (atomic_unsafe_head / "change").symlink_to(
            outside, target_is_directory=True
        )
        atomic_head_prepare = _prepare(atomic_head_root, "--prepare-head")
        if atomic_head_prepare.returncode != 2:
            failures.append(
                "head-freshness: unsafe later head artifact was not refused"
            )
        if not (atomic_safe_head / "new" / "estimates.json").exists():
            failures.append(
                "head-freshness: cleanup partially deleted a safe artifact "
                "before refusing a later unsafe path"
            )

        # #1107: stored Markdown must carry the machine/phase handoff and the
        # actual sample shape Criterion persisted for both A/B arms.
        provenance_path = Path(td) / "provenance.txt"
        provenance_fields = {
            "started_utc": "2026-07-29T12:00:00Z",
            "finished_utc": "2026-07-29T12:00:40Z",
            "host_id": "hostname-sha256:0123456789abcdef",
            "os": "fixture-os",
            "base_ref": "fixture-base",
            "base_sha": "a" * 40,
            "head_ref": "fixture-head",
            "head_sha": "b" * 40,
            "head_mode": "detached-worktree",
            "base_rustc": "rustc fixture-base",
            "head_rustc": "rustc fixture-head",
            "base_cargo": "cargo fixture-base",
            "head_cargo": "cargo fixture-head",
            "base_criterion": "0.5.1",
            "head_criterion": "0.5.1",
            "criterion_mode": "quick",
            "baseline_name": "compare-base",
            "targets": "lattice-inference:fixture",
            "inference_features": "<none>",
            "filters": "inference='<all>' embed='<all>'",
            "enforcement": "fail-on-regression",
        }
        machine_states = [
            {
                "schema": MACHINE_STATE_SCHEMA,
                "label": "before first arm",
                "captured_at_utc": "2026-07-29T12:00:10Z",
                "power": {"status": "measured", "source": "pmset", "state": "ac"},
                "thermal": {
                    "status": "measured",
                    "source": "pmset",
                    "state": "nominal",
                    "cpu_speed_limit_percent": 100,
                },
                "idle": {
                    "status": "measured",
                    "source": "IOHIDSystem.HIDIdleTime",
                    "seconds": 30.0,
                },
            },
            {
                "schema": MACHINE_STATE_SCHEMA,
                "label": "between order strata",
                "captured_at_utc": "2026-07-29T12:00:20Z",
                "power": {"status": "unavailable", "reason": "fixture unsupported"},
                "thermal": {
                    "status": "unavailable",
                    "reason": "fixture unsupported",
                },
                "idle": {
                    "status": "unavailable",
                    "reason": "fixture unsupported",
                },
            },
            {
                "schema": MACHINE_STATE_SCHEMA,
                "label": "after final arm",
                "captured_at_utc": "2026-07-29T12:00:30Z",
                "power": {
                    "status": "measured",
                    "source": "pmset",
                    "state": "battery",
                },
                "thermal": {
                    "status": "measured",
                    "source": "pmset",
                    "state": "throttled",
                    "cpu_speed_limit_percent": 90,
                },
                "idle": {
                    "status": "measured",
                    "source": "IOHIDSystem.HIDIdleTime",
                    "seconds": 31.0,
                },
            },
        ]
        provenance_lines = [
            f"schema={PROVENANCE_SCHEMA}",
            *[f"{field}={provenance_fields[field]}" for field in PROVENANCE_FIELDS],
            "lock=bench-window: acquired",
            "lock=Metal GPU: acquired",
            "ambient=[quiet] before first arm: idle 99.0% (floor 70.0%) ok | top: none",
            "ambient=[quiet] between order strata: idle 98.0% (floor 70.0%) ok | top: none",
            "ambient=[quiet] after final arm: idle 97.0% (floor 70.0%) ok | top: none",
            *[
                f"machine_state={json.dumps(state, separators=(',', ':'), sort_keys=True)}"
                for state in machine_states
            ],
        ]
        provenance_path.write_text("\n".join(provenance_lines) + "\n")

        provenance_root = Path(td) / "provenance" / "criterion"
        provenance_bench = provenance_root / "grp_provenance" / "bench_provenance"
        _fabricate_bench(provenance_bench, "compare-base")
        _fabricate_sample(provenance_bench / "new", count=2, mode="Flat")
        _fabricate_sample(provenance_bench / "compare-base", count=4, mode="Linear")
        provenance_run = _run(
            provenance_root,
            "--require-measurements",
            "--provenance-file",
            str(provenance_path),
            "--require-provenance",
        )
        if provenance_run.returncode != 0:
            failures.append(
                "run-provenance: complete provenance and sample metadata were "
                f"rejected: {provenance_run.stderr}"
            )
        for expected in (
            "<summary>Run provenance</summary>",
            "host_id=hostname-sha256:0123456789abcdef",
            "criterion_base_samples=4 Linear (1 benchmark)",
            "criterion_head_samples=2 Flat (1 benchmark)",
            "ambient=[quiet] after final arm",
            "machine_state[between order strata]=captured",
            "power unavailable (fixture unsupported)",
            "HID idle unavailable (fixture unsupported)",
            "gate not enforced on this platform",
            "| 4 Linear | 2 Flat |",
        ):
            if expected not in provenance_run.stdout:
                failures.append(
                    f"run-provenance: stored report omitted {expected!r}"
                )

        darwin_ungated = Path(td) / "darwin-ungated-provenance.txt"
        darwin_ungated.write_text(
            "\n".join(
                "os=Darwin 24.6.0 arm64"
                if line.startswith("os=")
                else line
                for line in provenance_lines
            )
            + "\n"
        )
        darwin_ungated_run = _run(
            provenance_root,
            "--provenance-file",
            str(darwin_ungated),
            "--require-provenance",
        )
        if (
            darwin_ungated_run.returncode != 2
            or "lacks a passed fail-closed gate" not in darwin_ungated_run.stderr
        ):
            failures.append(
                "run-provenance: macOS handoff without machine-state gates "
                "did not fail closed"
            )

        legacy_report = _run(provenance_root)
        if legacy_report.returncode != 0:
            failures.append(
                "run-provenance: existing report-only CLI without provenance changed"
            )
        if "unsuitable as benchmark evidence" not in legacy_report.stdout:
            failures.append(
                "run-provenance: absent handoff was not disclosed in the report"
            )

        if _run(provenance_root, "--require-provenance").returncode != 2:
            failures.append(
                "run-provenance: --require-provenance accepted no provenance file"
            )

        incomplete_provenance = Path(td) / "incomplete-provenance.txt"
        incomplete_provenance.write_text(
            "\n".join(
                line
                for line in provenance_lines
                if not line.startswith("host_id=")
            )
            + "\n"
        )
        incomplete_report = _run(
            provenance_root,
            "--provenance-file",
            str(incomplete_provenance),
        )
        if (
            incomplete_report.returncode != 0
            or "unsuitable as benchmark evidence" not in incomplete_report.stdout
        ):
            failures.append(
                "run-provenance: incomplete report-only handoff did not render unsuitable"
            )
        else:
            print(
                "MUTATION PROOF: missing host_id report-only exit=0; "
                "unsuitable marker rendered"
            )
        incomplete_enforcing = _run(
            provenance_root,
            "--provenance-file",
            str(incomplete_provenance),
            "--require-provenance",
        )
        if (
            incomplete_enforcing.returncode != 2
            or "missing provenance fields: host_id" not in incomplete_enforcing.stderr
        ):
            failures.append(
                "run-provenance: incomplete enforcing handoff did not fail closed"
            )
        else:
            print(
                "MUTATION PROOF: missing host_id enforcing exit=2; "
                "missing provenance fields: host_id"
            )

        reversed_provenance = Path(td) / "reversed-provenance.txt"
        reversed_provenance.write_text(
            "\n".join(
                "finished_utc=2026-07-29T11:59:59Z"
                if line.startswith("finished_utc=")
                else line
                for line in provenance_lines
            )
            + "\n"
        )
        reversed_run = _run(
            provenance_root,
            "--provenance-file",
            str(reversed_provenance),
            "--require-provenance",
        )
        if reversed_run.returncode != 2 or "precedes started_utc" not in reversed_run.stderr:
            failures.append(
                "run-provenance: reversed wall-clock interval did not fail closed"
            )

        out_of_order_state = dict(machine_states[1])
        out_of_order_state["captured_at_utc"] = "2026-07-29T12:00:05Z"
        out_of_order_provenance = Path(td) / "out-of-order-provenance.txt"
        out_of_order_provenance.write_text(
            "\n".join(
                f"machine_state={json.dumps(out_of_order_state, separators=(',', ':'), sort_keys=True)}"
                if '"label":"between order strata"' in line
                else line
                for line in provenance_lines
            )
            + "\n"
        )
        out_of_order_run = _run(
            provenance_root,
            "--provenance-file",
            str(out_of_order_provenance),
            "--require-provenance",
        )
        if (
            out_of_order_run.returncode != 2
            or "out of phase order" not in out_of_order_run.stderr
        ):
            failures.append(
                "run-provenance: out-of-order phase timestamps did not fail closed"
            )

        unknown_state = dict(machine_states[0])
        unknown_state["power"] = {
            "status": "measured",
            "source": "pmset",
            "state": "wall-outlet",
        }
        invalid_state_provenance = Path(td) / "invalid-state-provenance.txt"
        invalid_state_provenance.write_text(
            "\n".join(
                f"machine_state={json.dumps(unknown_state, separators=(',', ':'), sort_keys=True)}"
                if '"label":"before first arm"' in line
                else line
                for line in provenance_lines
            )
            + "\n"
        )
        invalid_state_run = _run(
            provenance_root,
            "--provenance-file",
            str(invalid_state_provenance),
            "--require-provenance",
        )
        if invalid_state_run.returncode != 2 or "ac, battery" not in invalid_state_run.stderr:
            failures.append(
                "run-provenance: unknown measured power state did not fail closed"
            )
        else:
            print(
                "MUTATION PROOF: malformed measured power capability exit=2; "
                "accepted states remain ac,battery"
            )

        missing_samples = _run(
            ok_root,
            "--provenance-file",
            str(provenance_path),
            "--require-provenance",
        )
        if missing_samples.returncode != 2 or "sample metadata" not in missing_samples.stderr:
            failures.append(
                "run-provenance: missing Criterion sample.json metadata was certified"
            )

        invalid_sample_path = provenance_bench / "new" / "sample.json"
        invalid_sample_path.write_text(
            json.dumps(
                {
                    "sampling_mode": "Flat",
                    "iters": [1.0, 2.0],
                    "times": [1.0, 0.0],
                }
            )
        )
        invalid_sample = _run(
            provenance_root,
            "--provenance-file",
            str(provenance_path),
            "--require-provenance",
        )
        if invalid_sample.returncode != 2 or "sample metadata" not in invalid_sample.stderr:
            failures.append(
                "run-provenance: non-positive Criterion sample was certified"
            )

    for f in failures:
        print(f"FAIL: {f}", file=sys.stderr)
    if failures:
        print(f"SELFTEST: FAIL ({len(failures)} failure(s))")
        return 1
    print("SELFTEST: PASS — base/, compare-base/, orphan-warn, named-wins-over-stale-base, "
          "multi-sibling-refusal, target-qualified manifest handoff (CRLF/whitespace/C-sort, "
          "slash-bearing group, demoted target informational, non-demoted target gated), "
          "require-measurements (empty root, gating pass, explicit informational target, "
          "mismatch refusal, partial-run refusal), ABBA directional-drift correction, "
          "malformed reverse-number refusal, selected-baseline completeness, and "
          "stale-head freshness all correct")
    return 0


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
                         "omit for gating/full-mode runs.")
    ap.add_argument("--resolution", choices=("quick", "full"), default="full",
                    help="Measurement resolution used for verdict classification "
                         "(default: full).")
    ap.add_argument("--require-measurements", action="store_true",
                    help="Fail (exit 2) instead of passing when the run produced no gating "
                         "comparison to judge or a selected-base benchmark has no head "
                         "comparison. Without this, an absent baseline exits 0, which is "
                         "right for a first run but wrong for an automated lane: it cannot "
                          "tell 'nothing regressed' from 'nothing was measured'.")
    ap.add_argument("--ambient-samples", type=Path,
                    help="JSONL before/between/after ambient samples. Requires "
                         "--status-out; an invalid or busy run exits 3.")
    ap.add_argument("--status-out", type=Path,
                    help="Atomically write pass/regression/error/not_measurable JSON.")
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
                         "Markdown report.")
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
        return run_selftest()

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

    ambient: AmbientAssessment | None = None

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

    all_change_files = find_change_files(args.criterion_root)
    change_file_ids = {
        change_file: artifact_bench_id(
            change_file, args.criterion_root, artifact_parts=1
        )
        for change_file in all_change_files
    }
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

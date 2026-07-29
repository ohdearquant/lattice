#!/usr/bin/env python3
"""Parse Criterion change reports and apply the ADR-058 regression gate.

For every Criterion bench under target/criterion/, read change/estimates.json
(produced when running with --baseline <name>). Apply the rule:

  CI-lower of change in (-inf, +3%]    : pass silently
  CI-lower of change in (+3%, +7%]     : warn (PR-comment only, no fail)
  CI-lower of change in (+7%, +inf)    : FAIL
  Point estimate < -3% AND CI-upper<0% : celebrate

Usage:
  perf-bench-gate.py <criterion_root> <arch_label> [--out report.md]
  perf-bench-gate.py <criterion_root> <arch_label> --informational-groups-file <path>
  perf-bench-gate.py <criterion_root> --prepare-baseline-copy
  perf-bench-gate.py <criterion_root> --prepare-head

Exit codes:
  0 — pass (no gated FAILs)
  1 — at least one gated FAIL (regression > 7%, using the LOWER bound of
      Criterion's two-sided 95% CI as a one-sided cutoff — see the
      WARN_PCT/FAIL_PCT note below for the actual one-sided confidence
      level this implies, which is tighter than "95%")
  2 — parse error / bad input, or (with --require-measurements) the gate
      refusing to certify a run it could not judge: no comparison data, or
      no gating comparison among the parsed results, or a benchmark in the
      selected baseline set with no head comparison. An automated lane must not
      read "nothing was measured" as "nothing regressed".

--informational-groups-file (lattice#714): quick-mode Criterion runs on
sub-microsecond micro-benches (lattice-embed's `simd` bench target) are
dominated by scheduler/thermal jitter rather than code changes — confirmed
by two same-toolchain quick-mode A/A runs on identical refs flipping FAIL/
WARN sign across dozens of entries (lattice#714). Groups listed in this file
(one Criterion top-level group name per line, e.g. from `cargo bench ... --
--list`) are still measured and reported, but excluded from the FAIL/WARN
gate and the exit code — they render in a separate "informational" section
labeled below quick-mode resolution. This file should only be passed for
quick-mode runs; full-mode (tight-CI) runs gate every group normally so a
real embed SIMD regression is still caught.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROVENANCE_SCHEMA = "lattice-bench-provenance-v1"
MACHINE_STATE_SCHEMA = "lattice-machine-state-v1"
PHASE_LABELS = ("before base", "between phases", "after head")
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

    @property
    def point_pct(self) -> float: return self.point * 100.0
    @property
    def ci_low_pct(self) -> float: return self.ci_low * 100.0
    @property
    def ci_high_pct(self) -> float: return self.ci_high * 100.0
    @property
    def group(self) -> str:
        """Top-level Criterion group name (name is 'group/function/param' or 'group/param')."""
        return self.name.split("/", 1)[0]

    def is_informational(self, informational_groups: frozenset[str]) -> bool:
        return self.group in informational_groups

    def verdict(self) -> str:
        if self.ci_low_pct > FAIL_PCT:
            return "FAIL"
        if self.ci_low_pct > WARN_PCT:
            return "WARN"
        if self.point_pct < CELEBRATE_PCT and self.ci_high_pct < 0:
            return "WIN"
        return "PASS"


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
    if fields["os"].split(maxsplit=1)[0] == "Darwin":
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


def load_informational_groups(path: Path | None) -> frozenset[str]:
    """Load top-level group names to exclude from gating (lattice#714).

    One group name per line; blank lines and '#'-prefixed comments ignored.
    """
    if path is None:
        return frozenset()
    if not path.exists():
        print(f"warn: --informational-groups-file {path} does not exist — gating "
              f"every group normally", file=sys.stderr)
        return frozenset()
    groups = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        groups.add(line)
    return frozenset(groups)


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


def parse_bench(change_file: Path, root: Path, baseline_name: str) -> BenchResult | None:
    """Parse one change/estimates.json + sibling new/estimates.json + baseline estimates.json.

    Returns None if files are malformed (bench skipped, not failed).
    """
    bench_dir = change_file.parent.parent  # .../<bench>/<test>/
    name = artifact_bench_id(change_file, root, artifact_parts=1)

    try:
        change = json.loads(change_file.read_text())
        mean = change["mean"]
        point = mean["point_estimate"]
        ci_low = mean["confidence_interval"]["lower_bound"]
        ci_high = mean["confidence_interval"]["upper_bound"]

        new_path = bench_dir / "new" / "estimates.json"
        new_ns = json.loads(new_path.read_text())["mean"]["point_estimate"]

        base_path = find_baseline_estimates(bench_dir, baseline_name)
        if base_path is None:
            print(f"warn: {name}: change/estimates.json present but no resolvable "
                  f"baseline dir (tried base/, {baseline_name}/, and other siblings) "
                  f"— skipping", file=sys.stderr)
            return None
        old_ns = json.loads(base_path.read_text())["mean"]["point_estimate"]
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
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


def render_report(results: list[BenchResult], arch: str,
                   informational_groups: frozenset[str] = frozenset(),
                   provenance: RunProvenance | None = None) -> str:
    gated = [r for r in results if not r.is_informational(informational_groups)]
    info = [r for r in results if r.is_informational(informational_groups)]

    fails = [r for r in gated if r.verdict() == "FAIL"]
    warns = [r for r in gated if r.verdict() == "WARN"]
    wins = [r for r in gated if r.verdict() == "WIN"]

    info_fails = [r for r in info if r.verdict() == "FAIL"]
    info_warns = [r for r in info if r.verdict() == "WARN"]
    info_wins = [r for r in info if r.verdict() == "WIN"]

    lines = [f"### `{arch}` — perf regression report\n"]
    if fails:
        lines.append(
            f"**❌ {len(fails)} FAIL** (regression >{FAIL_PCT}% — lower bound of Criterion's "
            "two-sided 95% CI, i.e. about a 97.5% one-sided level, not a calibrated one-sided 95% test)"
        )
    if warns:
        lines.append(
            f"**⚠ {len(warns)} WARN** (regression {WARN_PCT}-{FAIL_PCT}% by the same "
            "two-sided-95%-CI lower bound)"
        )
    if wins:
        lines.append(f"**🚀 {len(wins)} confirmed improvement**")
    if not (fails or warns or wins):
        lines.append(f"✅ All {len(gated)} gated benches within noise band (±{WARN_PCT}%)")
    lines.append("")
    lines.extend(render_run_provenance(provenance, results))

    if fails or warns or wins:
        lines.append(
            "| Bench | Δ point | 95% CI | new ns | base ns | base n/mode | "
            "head n/mode | verdict |"
        )
        lines.append("|---|---:|---|---:|---:|---|---|---|")
        for r in sorted(fails + warns + wins, key=lambda r: -r.ci_low_pct):
            icon = {"FAIL": "❌", "WARN": "⚠", "WIN": "🚀"}[r.verdict()]
            lines.append(
                f"| `{r.name}` | {r.point_pct:+.2f}% | [{r.ci_low_pct:+.2f}%, {r.ci_high_pct:+.2f}%] "
                f"| {r.new_ns:.1f} | {r.old_ns:.1f} | {sample_cell(r, 'base')} "
                f"| {sample_cell(r, 'head')} | {icon} {r.verdict()} |"
            )
        lines.append("")

    if info:
        lines.append(
            f"**ℹ️ {len(info)} informational** (below quick-mode resolution — "
            f"lattice-embed SIMD micro-benches, tracked in #714; not gated here, "
            f"re-run `--full` for a gated verdict)"
        )
        if info_fails or info_warns or info_wins:
            lines.append(
                "| Bench | Δ point | 95% CI | new ns | base ns | base n/mode | "
                "head n/mode | (would-be verdict) |"
            )
            lines.append("|---|---:|---|---:|---:|---|---|---|")
            for r in sorted(info_fails + info_warns + info_wins, key=lambda r: -r.ci_low_pct):
                icon = {"FAIL": "❌", "WARN": "⚠", "WIN": "🚀"}[r.verdict()]
                lines.append(
                    f"| `{r.name}` | {r.point_pct:+.2f}% | [{r.ci_low_pct:+.2f}%, {r.ci_high_pct:+.2f}%] "
                    f"| {r.new_ns:.1f} | {r.old_ns:.1f} | {sample_cell(r, 'base')} "
                    f"| {sample_cell(r, 'head')} | {icon} {r.verdict()} (informational) |"
                )
        lines.append("")

    lines.append(
        f"<details><summary>All {len(results)} measurements</summary>\n\n"
        "| Bench | Δ point | CI-lower | CI-upper | base n/mode | head n/mode |\n"
        "|---|---:|---:|---:|---|---|"
    )
    for r in sorted(results, key=lambda r: r.name):
        lines.append(
            f"| `{r.name}` | {r.point_pct:+.2f}% | {r.ci_low_pct:+.2f}% "
            f"| {r.ci_high_pct:+.2f}% | {sample_cell(r, 'base')} "
            f"| {sample_cell(r, 'head')} |"
        )
    lines.append("\n</details>\n")
    lines.append(
        f"_Rule: CI-lower of change ≤{WARN_PCT}% passes silently; "
        f"({WARN_PCT}%, {FAIL_PCT}%] warns; >{FAIL_PCT}% fails._"
    )
    if informational_groups:
        lines.append(
            f"_{len(informational_groups)} group(s) excluded from gating as quick-mode "
            f"informational-only (lattice#714): {', '.join(sorted(informational_groups))}._\n"
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
    import subprocess
    import tempfile

    failures: list[str] = []

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

        # lattice#714: informational-groups exclusion. Two confirmed FAILs, one in a
        # group named as informational (quick-mode embed-SIMD noise floor), one not —
        # the exit-code fail count and the gated report section must only count the
        # real one; the informational one must still be measured and reported.
        noisy_dir = root / "grp_f" / "noisy_fail"
        _fabricate_bench(noisy_dir, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)
        real_dir = root / "grp_g" / "real_fail"
        _fabricate_bench(real_dir, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)

        for cf in find_change_files(root):
            r = parse_bench(cf, root, baseline_name="compare-base")
            if r is not None:
                results[r.name] = r

        informational = frozenset({"grp_f"})
        all_results = list(results.values())
        gated_fails = [
            r for r in all_results
            if r.verdict() == "FAIL" and not r.is_informational(informational)
        ]
        if "grp_f/noisy_fail" not in results or "grp_g/real_fail" not in results:
            failures.append("informational-groups fixture: benches not parsed")
        elif not results["grp_f/noisy_fail"].is_informational(informational):
            failures.append("informational-groups: grp_f/noisy_fail not classified informational")
        elif results["grp_g/real_fail"].is_informational(informational):
            failures.append("informational-groups: grp_g/real_fail wrongly classified informational")
        elif any(r.name == "grp_f/noisy_fail" for r in gated_fails):
            failures.append("informational-groups: noisy FAIL leaked into gated fail count")
        elif not any(r.name == "grp_g/real_fail" for r in gated_fails):
            failures.append("informational-groups: real FAIL missing from gated fail count")

        report = render_report(all_results, "selftest-arch", informational)
        if "grp_g/real_fail" not in report:
            failures.append("informational-groups: real FAIL missing from rendered report")
        if "ℹ️" not in report or "grp_f/noisy_fail" not in report:
            failures.append("informational-groups: noisy FAIL not shown in informational section")
        if "bench-allow-regression" in report:
            failures.append("rendered report advertises an unsupported label override")

        # lattice#714 / lattice#1060: the shell-side manifest handoff,
        # exercised end-to-end against the real helper and the real
        # manifest (scripts/lib/bench-quick-informational-targets.txt) —
        # the same files bench-compare.sh uses in production. Three
        # probes: (1) --print-targets must equal the reviewed expectation
        # set below, so a manifest-only or expectation-only edit fails
        # the selftest; (2) a demoted target key must emit every group of
        # a controlled listing (target-level semantics — including groups
        # the old per-group allowlist never contained); (3) a non-demoted
        # target key against the same listing must emit nothing — the
        # cross-target guarantee that keeps inference gating intact.
        # Probe 2's output then drives the Python classifier to prove
        # embed FAILs land informational while an inference FAIL gates.
        helper = Path(__file__).resolve().parent / "lib" / "bench-informational-groups.sh"
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
            listing_dir = root / "manifest-listing"
            listing_dir.mkdir(parents=True, exist_ok=True)
            listing_file = listing_dir / "list.txt"
            listing_file.write_text(
                "simd_dot_product/scalar/384: benchmark\n"
                "simd_dot_product/simd/384: benchmark\n"
                "simd_cosine_similarity/scalar/384: benchmark\n"
                "simd_normalize/scalar/384: benchmark\n"
                "simd_dot_product_extra/scalar/384: benchmark\n"
                "int8_raw_dot_product/dot_product_i8_raw/128: benchmark\n"
            )
            expected_groups = frozenset({
                "simd_dot_product", "simd_cosine_similarity", "simd_normalize",
                "simd_dot_product_extra", "int8_raw_dot_product",
            })
            demoted_proc = subprocess.run(
                ["bash", str(helper), "lattice-embed:simd", str(listing_file)],
                capture_output=True, text=True, timeout=30,
            )
            shell_emitted = frozenset(
                ln.strip() for ln in demoted_proc.stdout.splitlines() if ln.strip()
            )
            gated_proc = subprocess.run(
                ["bash", str(helper), "lattice-inference:elementwise_cpu_bench",
                 str(listing_file)],
                capture_output=True, text=True, timeout=30,
            )
            gated_emitted = [ln for ln in gated_proc.stdout.splitlines() if ln.strip()]
            if demoted_proc.returncode != 0:
                failures.append(
                    f"manifest-handoff: demoted-target probe exited "
                    f"{demoted_proc.returncode}: {demoted_proc.stderr}"
                )
            elif shell_emitted != expected_groups:
                failures.append(
                    "manifest-handoff: demoted target emitted "
                    f"{sorted(shell_emitted)}, expected every listing group "
                    f"{sorted(expected_groups)}"
                )
            elif gated_proc.returncode != 0:
                failures.append(
                    f"manifest-handoff: non-demoted-target probe exited "
                    f"{gated_proc.returncode}: {gated_proc.stderr}"
                )
            elif gated_emitted:
                failures.append(
                    "manifest-handoff: non-demoted target emitted "
                    f"{gated_emitted} — cross-target exemption leak"
                )
            else:
                manifest_dir = root / "manifest"
                emb_a = manifest_dir / "simd_dot_product" / "384"
                _fabricate_bench(emb_a, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)
                emb_b = manifest_dir / "simd_normalize" / "384"
                _fabricate_bench(emb_b, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)
                inf_c = manifest_dir / "rms_norm" / "4096"
                _fabricate_bench(inf_c, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)

                manifest_results: dict[str, BenchResult] = {}
                for cf in find_change_files(manifest_dir):
                    r = parse_bench(cf, manifest_dir, baseline_name="compare-base")
                    if r is not None:
                        manifest_results[r.name] = r

                needed = {"simd_dot_product/384", "simd_normalize/384", "rms_norm/4096"}
                if not needed.issubset(manifest_results):
                    failures.append("manifest-handoff fixture: not all benches parsed")
                else:
                    manifest_gated_fails = {
                        r.name for r in manifest_results.values()
                        if r.verdict() == "FAIL" and not r.is_informational(shell_emitted)
                    }
                    if "simd_dot_product/384" in manifest_gated_fails:
                        failures.append(
                            "manifest-handoff: demoted simd_dot_product "
                            "leaked into gated fails"
                        )
                    if "simd_normalize/384" in manifest_gated_fails:
                        failures.append(
                            "manifest-handoff: simd_normalize gated despite "
                            "target-level demotion (listing-derivation broken)"
                        )
                    if "rms_norm/4096" not in manifest_gated_fails:
                        failures.append(
                            "manifest-handoff: inference group rms_norm did not gate"
                        )

        # Composed-path collision guard. The two probes above show the
        # helper is target-aware in isolation, but bench-compare.sh used to
        # concatenate every target's helper output into one flat file with
        # no target attribution — a group name demoted for one target
        # silently exempted an identically-named group produced by a
        # different, gated target. This drives the resolver
        # (scripts/lib/resolve-informational-groups.sh) with a fabricated
        # listing where the demoted target's groups include `rms_norm`,
        # which also appears in the gated target's listing, and asserts
        # the collision gates instead of staying informational.
        resolver = Path(__file__).resolve().parent / "lib" / "resolve-informational-groups.sh"
        if not resolver.exists():
            failures.append(f"collision-guard: resolver missing at {resolver}")
        else:
            collision_dir = root / "collision-listing"
            collision_dir.mkdir(parents=True, exist_ok=True)
            embed_listing = collision_dir / "embed-list.txt"
            embed_listing.write_text(
                "simd_dot_product/scalar/384: benchmark\n"
                "simd_normalize/scalar/384: benchmark\n"
                "rms_norm/scalar/384: benchmark\n"  # fabricated bare-name collision
            )
            inference_listing = collision_dir / "inference-list.txt"
            inference_listing.write_text(
                "rms_norm/4096: benchmark\n"
                "gelu/4096: benchmark\n"
            )

            def list_groups(listing_path: Path) -> str:
                proc = subprocess.run(
                    ["bash", str(helper), "--list-groups", str(listing_path)],
                    capture_output=True, text=True, timeout=30,
                )
                return proc.stdout

            demoted_groups_file = collision_dir / "demoted.txt"
            demoted_groups_file.write_text(list_groups(embed_listing))
            gated_groups_file = collision_dir / "gated.txt"
            gated_groups_file.write_text(list_groups(inference_listing))

            resolve_proc = subprocess.run(
                ["bash", str(resolver),
                 str(demoted_groups_file), "lattice-embed:simd",
                 str(gated_groups_file), "lattice-inference:elementwise_cpu_bench"],
                capture_output=True, text=True, timeout=30,
            )
            resolved = frozenset(
                ln.strip() for ln in resolve_proc.stdout.splitlines() if ln.strip()
            )
            if resolve_proc.returncode != 0:
                failures.append(
                    f"collision-guard: resolver exited {resolve_proc.returncode}: "
                    f"{resolve_proc.stderr}"
                )
            if "rms_norm" in resolved:
                failures.append(
                    "collision-guard: rms_norm (demoted+gated collision) leaked "
                    "into the informational set instead of gating"
                )
            if "rms_norm" not in resolve_proc.stderr or "lattice-embed:simd" not in resolve_proc.stderr:
                failures.append(
                    "collision-guard: no stderr warning naming the colliding "
                    "group and both targets"
                )
            if not {"simd_dot_product", "simd_normalize"}.issubset(resolved):
                failures.append(
                    "collision-guard: non-colliding demoted groups lost "
                    "informational status — resolver over-suppressed"
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

        # All-informational is measured-but-unjudgeable: nothing could FAIL.
        info_file = Path(td) / "informational.txt"
        info_file.write_text("grp_info\n")
        info_root = Path(td) / "info" / "criterion"
        _fabricate_bench(info_root / "grp_info" / "bench_info", "compare-base")
        if _run(info_root, "--require-measurements",
                "--informational-groups-file", str(info_file)).returncode != 2:
            failures.append("require-measurements: an all-informational run was "
                            "certified — no gating comparison was judged")

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

        coverage_info = Path(td) / "coverage-informational.txt"
        coverage_info.write_text("simd_dot_product\n")
        coverage = _run(
            coverage_root,
            "--require-measurements",
            "--informational-groups-file",
            str(coverage_info),
        )
        if coverage.returncode != 2:
            failures.append(
                "baseline-completeness: missing head benchmarks exited "
                f"{coverage.returncode} instead of 2"
            )
        for expected_id in ("rms_norm/4096", "simd_dot_product/scalar/384"):
            if f"  - {expected_id}" not in coverage.stderr:
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
            "--informational-groups-file",
            str(coverage_info),
        ).returncode != 0:
            failures.append(
                "baseline-completeness: reporter mode rejected missing head "
                "benchmarks without --require-measurements"
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
                "label": "before base",
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
                "label": "between phases",
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
                "label": "after head",
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
            "ambient=[quiet] before base: idle 99.0% (floor 70.0%) ok | top: none",
            "ambient=[quiet] between phases: idle 98.0% (floor 70.0%) ok | top: none",
            "ambient=[quiet] after head: idle 97.0% (floor 70.0%) ok | top: none",
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
            "ambient=[quiet] after head",
            "machine_state[between phases]=captured",
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
                if '"label":"after head"' not in line
            )
            + "\n"
        )
        incomplete_run = _run(
            provenance_root,
            "--provenance-file",
            str(incomplete_provenance),
        )
        if (
            incomplete_run.returncode != 2
            or "expected 3 machine-state samples" not in incomplete_run.stderr
        ):
            failures.append(
                "run-provenance: incomplete machine-state handoff did not fail closed"
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
                if '"label":"between phases"' in line
                else line
                for line in provenance_lines
            )
            + "\n"
        )
        out_of_order_run = _run(
            provenance_root,
            "--provenance-file",
            str(out_of_order_provenance),
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
                if '"label":"before base"' in line
                else line
                for line in provenance_lines
            )
            + "\n"
        )
        invalid_state_run = _run(
            provenance_root,
            "--provenance-file",
            str(invalid_state_provenance),
        )
        if invalid_state_run.returncode != 2 or "ac, battery" not in invalid_state_run.stderr:
            failures.append(
                "run-provenance: unknown measured power state did not fail closed"
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
          "multi-sibling-refusal, manifest-handoff (demoted target informational, "
          "non-demoted target gated), composed-path collision-guard, require-measurements "
          "(empty root, gating pass, all-informational refusal, partial-run refusal), "
          "selected-baseline completeness, stale-head freshness, and stored run "
          "provenance/sample shapes all correct")
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
    ap.add_argument("--informational-groups-file", type=Path, default=None,
                    help="Path to a file listing Criterion top-level group names (one per "
                         "line) to measure+report but exclude from gating/exit-code — quick-"
                         "mode noise-floor groups (lattice#714). Omit for full-mode runs.")
    ap.add_argument("--require-measurements", action="store_true",
                    help="Fail (exit 2) instead of passing when the run produced no gating "
                         "comparison to judge or a selected-base benchmark has no head "
                         "comparison. Without this, an absent baseline exits 0, which is "
                         "right for a first run but wrong for an automated lane: it cannot "
                         "tell 'nothing regressed' from 'nothing was measured'.")
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

    if args.require_provenance and args.provenance_file is None:
        print(
            "error: --require-provenance needs --provenance-file; a stored verdict "
            "without machine and phase conditions is not auditable.",
            file=sys.stderr,
        )
        return 2
    provenance = None
    if args.provenance_file is not None:
        try:
            provenance = load_run_provenance(args.provenance_file)
        except ValueError as error:
            print(f"error: invalid run provenance: {error}", file=sys.stderr)
            return 2
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
            return 2

    if not args.criterion_root.exists():
        print(f"error: {args.criterion_root} does not exist", file=sys.stderr)
        return 2

    try:
        baseline_ids = selected_baseline_bench_ids(
            args.criterion_root, args.baseline_name
        )
    except ValueError as error:
        print(f"error: invalid --baseline-name: {error}", file=sys.stderr)
        return 2

    if args.require_measurements and not baseline_ids:
        print(
            f"error: --require-measurements set but selected baseline "
            f"{args.baseline_name!r} contains no benchmark estimates under "
            f"{args.criterion_root}. A run with no selected base set cannot "
            "certify an A/B comparison.",
            file=sys.stderr,
        )
        return 2

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
    if args.require_measurements:
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
            print(f"  - {bench_id}", file=sys.stderr)
        print(
            "A partial A/B is not evidence that nothing regressed.",
            file=sys.stderr,
        )
        return 2

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
        if args.require_measurements:
            if missing_head_ids:
                return refuse_missing_head()
            print(f"error: --require-measurements set but no change/estimates.json under "
                  f"{args.criterion_root}: the run produced no comparison, so it is not "
                  f"evidence that nothing regressed.", file=sys.stderr)
            return 2
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
        return 0

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
        if args.require_measurements and missing_head_ids:
            return refuse_missing_head()
        print("error: change files found but all failed to parse", file=sys.stderr)
        return 2

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
            return 2

    informational_groups = load_informational_groups(args.informational_groups_file)
    report = render_report(results, args.arch, informational_groups, provenance)
    print(report)
    if args.out:
        args.out.write_text(report)

    gating = [r for r in results if not r.is_informational(informational_groups)]

    # Completeness, not existence. The first version of this guard asked whether
    # ANY judgeable comparison existed, which passes a run where one target built
    # a clean comparison and the other produced an unresolvable or malformed one:
    # the failed target is exactly the one nobody measured, so a green exit is a
    # claim about code that was never benched. Reconcile what the run intended
    # (change files on disk) against what was actually judged.
    if args.require_measurements and missing_head_ids:
        return refuse_missing_head()

    if args.require_measurements and unjudged:
        listed = ", ".join(str(cf.parent.parent.relative_to(args.criterion_root))
                           for cf in unjudged[:5])
        more = f" (+{len(unjudged) - 5} more)" if len(unjudged) > 5 else ""
        print(f"error: --require-measurements set but {len(unjudged)} of "
              f"{len(change_files)} comparison(s) could not be judged "
              f"(unresolvable baseline or malformed data): {listed}{more}. A partial "
              f"A/B is not evidence that nothing regressed.", file=sys.stderr)
        return 2

    # Parsed results are not automatically judgeable results. If every parsed
    # result is informational, nothing in this run could have produced a FAIL,
    # so a zero exit says only that no gating comparison existed.
    if args.require_measurements and not gating:
        print(f"error: --require-measurements set but all {len(results)} parsed result(s) are "
              f"informational: no gating comparison was judged.", file=sys.stderr)
        return 2

    fails = sum(1 for r in gating if r.verdict() == "FAIL")
    return 1 if fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

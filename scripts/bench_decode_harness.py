#!/usr/bin/env python3
"""Decode-benchmark harness core (issue #813).

`scripts/` grew seven independent implementations of the same
prompt/decode-benchmark shape (`bench_apples_to_apples.sh`,
`bench_apples_precise.sh`, `bench_q4_apples.sh`, `bench_context_scaling.sh`,
`bench_compare_1k.py`, plus two proven-dead scripts tracked separately), each
owning its own copy of warmup policy, sample-count policy,
aggregation/confidence-interval method, and output rendering. This module is
the single place that owns all of that: configuration validation, warmup
count, requested schedule, run order, monotonic wall timing, the raw
observation schema, aggregation, confidence-interval calculation, and output
rendering.

Engine adapters (Lattice, Ollama, MLX) own only invocation and parsing: they
implement `EngineAdapter.run()`, may surface an engine-native duration as a
diagnostic field, and MUST NOT choose repeat counts, discard samples, or
compute verdicts. No adapters ship in this module yet — this lands the
harness core only ("no public command change", issue #813 step 1). The five
live scripts above are migrated onto profiles defined in
`bench_decode_profiles.toml` one script per PR (issue #813 step 2); until
then `bench_decode_profiles.toml` carries no profile definitions and the
`run` subcommand below has nothing registered in `ADAPTER_REGISTRY`, so an
engine request always resolves as "missing" (see `--allow-missing-engine`).

Raw JSONL observation schema (the contract; see `OBSERVATION_FIELDS` /
`validate_observation`): schema version, git SHA, profile name, engine +
version, model/quantization, prompt hash, requested and actual prompt/
completion token counts, warmup/measured flag, run and order indices, the
harness-measured elapsed nanoseconds, an optional engine-native nanosecond
figure, a hardware identifier, and a timestamp. Reports render from this raw
data; they are not the source of truth.

Run with: python3 -m unittest tests/test_bench_decode_harness.py -v
(the module itself is stdlib-only; no engine binaries or third-party
packages are required to exercise it).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Protocol

SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES_FILE = REPO_ROOT / "scripts" / "bench_decode_profiles.toml"

AGGREGATION_METHODS = ("median", "trimmed_mean")


class ProfileConfigError(ValueError):
    """A profile definition in `bench_decode_profiles.toml` is invalid."""


class ObservationValidationError(ValueError):
    """A raw observation does not conform to the schema contract."""


class MissingEngineError(RuntimeError):
    """A profile requested an engine with no adapter registered for it."""


# --------------------------------------------------------------------------
# Raw observation schema (the contract)
# --------------------------------------------------------------------------

# name -> expected type, or a tuple of types (nullable fields use (T, NoneType))
OBSERVATION_FIELDS: dict[str, object] = {
    "schema_version": int,
    "git_sha": str,
    "profile": str,
    "engine": str,
    "engine_version": str,
    "model": str,
    "quantization": str,
    "prompt_hash": str,
    "requested_prompt_tokens": (int, type(None)),
    "actual_prompt_tokens": (int, type(None)),
    "requested_completion_tokens": int,
    "actual_completion_tokens": int,
    "warmup": bool,
    "run_index": int,
    "order_index": int,
    "elapsed_ns": int,
    "engine_native_ns": (int, type(None)),
    "hardware_id": str,
    "timestamp": str,
}

_NON_EMPTY_STRING_FIELDS = (
    "git_sha",
    "profile",
    "engine",
    "engine_version",
    "model",
    "quantization",
    "prompt_hash",
    "hardware_id",
)


@dataclass(frozen=True)
class Observation:
    schema_version: int
    git_sha: str
    profile: str
    engine: str
    engine_version: str
    model: str
    quantization: str
    prompt_hash: str
    requested_prompt_tokens: int | None
    actual_prompt_tokens: int | None
    requested_completion_tokens: int
    actual_completion_tokens: int
    warmup: bool
    run_index: int
    order_index: int
    elapsed_ns: int
    engine_native_ns: int | None
    hardware_id: str
    timestamp: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def validate_observation(row: dict) -> None:
    """Validate a raw observation dict against the schema contract.

    Fail-closed: unknown fields, missing fields, wrong types, and
    out-of-range values are all rejected. Raises `ObservationValidationError`
    on any violation; returns None on success.
    """
    if not isinstance(row, dict):
        raise ObservationValidationError(f"observation must be an object, got {type(row).__name__}")

    extra = set(row) - set(OBSERVATION_FIELDS)
    if extra:
        raise ObservationValidationError(f"unexpected field(s): {sorted(extra)}")
    missing = set(OBSERVATION_FIELDS) - set(row)
    if missing:
        raise ObservationValidationError(f"missing required field(s): {sorted(missing)}")

    for name, expected in OBSERVATION_FIELDS.items():
        _check_field_type(name, row[name], expected)

    if row["schema_version"] != SCHEMA_VERSION:
        raise ObservationValidationError(
            f"schema_version {row['schema_version']!r} != supported {SCHEMA_VERSION}"
        )
    for name in _NON_EMPTY_STRING_FIELDS:
        if not row[name]:
            raise ObservationValidationError(f"field {name!r} must be non-empty")
    if row["requested_completion_tokens"] < 0:
        raise ObservationValidationError("requested_completion_tokens must be >= 0")
    if row["actual_completion_tokens"] < 0:
        raise ObservationValidationError("actual_completion_tokens must be >= 0")
    for name in ("requested_prompt_tokens", "actual_prompt_tokens"):
        if row[name] is not None and row[name] < 0:
            raise ObservationValidationError(f"field {name!r} must be >= 0 or null")
    if row["run_index"] < 1:
        raise ObservationValidationError("run_index must be >= 1")
    if row["order_index"] < 0:
        raise ObservationValidationError("order_index must be >= 0")
    if row["elapsed_ns"] < 0:
        raise ObservationValidationError("elapsed_ns must be >= 0")
    if row["engine_native_ns"] is not None and row["engine_native_ns"] < 0:
        raise ObservationValidationError("engine_native_ns must be >= 0 or null")
    try:
        datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ObservationValidationError(f"timestamp {row['timestamp']!r} is not ISO 8601: {exc}") from exc


def _check_field_type(name: str, value: object, expected: object) -> None:
    # bool is a subclass of int in Python -- reject it everywhere an int (or
    # int-or-None) is expected, and require it explicitly where `bool` itself
    # is the declared type, so a stray `1`/`0` cannot silently pass as a flag
    # (or vice versa).
    if expected is bool:
        if not isinstance(value, bool):
            raise ObservationValidationError(f"field {name!r} must be bool, got {type(value).__name__}")
        return
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ObservationValidationError(f"field {name!r} must be int, got {type(value).__name__}")
        return
    if isinstance(expected, tuple):
        if isinstance(value, bool) or not isinstance(value, expected):
            names = "/".join(t.__name__ for t in expected)
            raise ObservationValidationError(f"field {name!r} must be one of {names}, got {type(value).__name__}")
        return
    if not isinstance(value, expected):
        raise ObservationValidationError(
            f"field {name!r} must be {expected.__name__}, got {type(value).__name__}"
        )


def validate_jsonl(path: Path) -> list[dict]:
    """Validate every line of a raw-observation JSONL file. Returns the parsed rows."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ObservationValidationError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            try:
                validate_observation(row)
            except ObservationValidationError as exc:
                raise ObservationValidationError(f"{path}:{line_no}: {exc}") from exc
            rows.append(row)
    return rows


def write_jsonl(observations: list[Observation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for obs in observations:
            fh.write(json.dumps(obs.to_dict(), sort_keys=True))
            fh.write("\n")


# --------------------------------------------------------------------------
# Profile configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    description: str
    windows: tuple[int, ...]
    warmup_repeats: int
    measured_repeats: int
    engines: tuple[str, ...]
    prompt: str
    model: str
    quantization: str
    aggregation: str = "median"
    trim: int = 0
    requested_prompt_tokens: int | None = None


def load_profiles_file(path: Path) -> tuple[int, dict[str, ProfileConfig]]:
    try:
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ProfileConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError as exc:
        raise ProfileConfigError(f"{path}: could not read profiles file: {exc}") from exc

    schema_version = doc.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ProfileConfigError(
            f"{path}: schema_version {schema_version!r} != supported {SCHEMA_VERSION}"
        )

    raw_profiles = doc.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise ProfileConfigError(f"{path}: [profiles] must be a table")

    profiles: dict[str, ProfileConfig] = {}
    for name, raw in raw_profiles.items():
        if not isinstance(raw, dict):
            raise ProfileConfigError(f"{path}: profile {name!r} must be a table")
        profiles[name] = _parse_profile(name, raw)
    return schema_version, profiles


def _parse_profile(name: str, raw: dict) -> ProfileConfig:
    def require(key: str, expected_type: type) -> object:
        if key not in raw:
            raise ProfileConfigError(f"profile {name!r}: missing required key {key!r}")
        value = raw[key]
        if expected_type is int and isinstance(value, bool):
            raise ProfileConfigError(f"profile {name!r}: key {key!r} must be int, got bool")
        if not isinstance(value, expected_type):
            raise ProfileConfigError(
                f"profile {name!r}: key {key!r} must be {expected_type.__name__}, got {type(value).__name__}"
            )
        return value

    description = str(raw.get("description", ""))

    windows_raw = require("windows", list)
    if len(windows_raw) < 2:
        raise ProfileConfigError(f"profile {name!r}: windows must contain at least 2 values")
    windows: list[int] = []
    for w in windows_raw:
        if isinstance(w, bool) or not isinstance(w, int) or w <= 0:
            raise ProfileConfigError(f"profile {name!r}: windows entries must be positive integers, got {w!r}")
        windows.append(w)
    if windows != sorted(windows) or len(set(windows)) != len(windows):
        raise ProfileConfigError(f"profile {name!r}: windows must be strictly increasing, got {windows}")

    warmup_repeats = require("warmup_repeats", int)
    if warmup_repeats < 0:
        raise ProfileConfigError(f"profile {name!r}: warmup_repeats must be >= 0")

    measured_repeats = require("measured_repeats", int)
    if measured_repeats < 1:
        raise ProfileConfigError(f"profile {name!r}: measured_repeats must be >= 1")

    engines_raw = require("engines", list)
    if not engines_raw:
        raise ProfileConfigError(f"profile {name!r}: engines must not be empty")
    engines: list[str] = []
    for e in engines_raw:
        if not isinstance(e, str) or not e:
            raise ProfileConfigError(f"profile {name!r}: engines entries must be non-empty strings")
        engines.append(e)
    if len(set(engines)) != len(engines):
        raise ProfileConfigError(f"profile {name!r}: engines must not contain duplicates, got {engines}")

    prompt = require("prompt", str)
    if not prompt:
        raise ProfileConfigError(f"profile {name!r}: prompt must be non-empty")
    model = require("model", str)
    if not model:
        raise ProfileConfigError(f"profile {name!r}: model must be non-empty")
    quantization = require("quantization", str)
    if not quantization:
        raise ProfileConfigError(f"profile {name!r}: quantization must be non-empty")

    aggregation = str(raw.get("aggregation", "median"))
    if aggregation not in AGGREGATION_METHODS:
        raise ProfileConfigError(
            f"profile {name!r}: aggregation must be one of {AGGREGATION_METHODS}, got {aggregation!r}"
        )

    trim = raw.get("trim", 0)
    if isinstance(trim, bool) or not isinstance(trim, int) or trim < 0:
        raise ProfileConfigError(f"profile {name!r}: trim must be a non-negative int")
    if aggregation == "median" and trim != 0:
        raise ProfileConfigError(f"profile {name!r}: trim is only meaningful for aggregation='trimmed_mean'")
    if aggregation == "trimmed_mean" and 2 * trim >= measured_repeats:
        raise ProfileConfigError(
            f"profile {name!r}: trim={trim} leaves no measured samples for measured_repeats={measured_repeats}"
        )

    requested_prompt_tokens = raw.get("requested_prompt_tokens")
    if requested_prompt_tokens is not None:
        if isinstance(requested_prompt_tokens, bool) or not isinstance(requested_prompt_tokens, int):
            raise ProfileConfigError(f"profile {name!r}: requested_prompt_tokens must be an int or omitted")
        if requested_prompt_tokens < 0:
            raise ProfileConfigError(f"profile {name!r}: requested_prompt_tokens must be >= 0")

    return ProfileConfig(
        name=name,
        description=description,
        windows=tuple(windows),
        warmup_repeats=warmup_repeats,
        measured_repeats=measured_repeats,
        engines=tuple(engines),
        prompt=prompt,
        model=model,
        quantization=quantization,
        aggregation=aggregation,
        trim=trim,
        requested_prompt_tokens=requested_prompt_tokens,
    )


# --------------------------------------------------------------------------
# Engine adapters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterRunResult:
    """What an engine adapter reports for one decode call.

    Adapters own invocation and parsing only: they must not choose repeat
    counts, discard samples, or compute verdicts. `native_ns`, when present,
    is surfaced as a diagnostic field alongside the harness-measured timing,
    never as a replacement for it.
    """

    actual_completion_tokens: int
    actual_prompt_tokens: int | None = None
    native_ns: int | None = None
    engine_version: str = "unknown"


class EngineAdapter(Protocol):
    def run(self, *, prompt: str, n_tokens: int, warmup: bool) -> AdapterRunResult: ...


# Populated by engine-adapter modules landed alongside each script migration
# (issue #813 step 2). Empty at harness-core landing time by design: no
# adapter means every profile's engines resolve as "missing" (see
# `run_profile` / `--allow-missing-engine`), so the CLI never presents a
# fabricated result for an engine it cannot actually invoke.
ADAPTER_REGISTRY: dict[str, EngineAdapter] = {}


def register_adapter(name: str, adapter: EngineAdapter) -> None:
    ADAPTER_REGISTRY[name] = adapter


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessRunResult:
    profile: ProfileConfig
    observations: tuple[Observation, ...]
    missing_engines: tuple[str, ...]


def git_sha(repo_root: Path | None = None) -> str:
    root = repo_root or REPO_ROOT
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return "unknown"
    sha = proc.stdout.strip()
    return sha if sha else "unknown"


def hardware_id() -> str:
    override = os.environ.get("LATTICE_BENCH_HARDWARE_ID")
    if override:
        return override
    return f"{platform.system()}-{platform.machine()}-{platform.node()}"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_profile(
    profile: ProfileConfig,
    adapters: Mapping[str, EngineAdapter],
    *,
    allow_missing_engine: bool = False,
    clock: Callable[[], int] = time.perf_counter_ns,
    git_sha_value: str | None = None,
    hardware_id_value: str | None = None,
    timestamp_fn: Callable[[], str] = _utc_timestamp,
    repo_root: Path | None = None,
) -> HarnessRunResult:
    """Execute `profile` against `adapters` and return every raw observation.

    Run order is the requested schedule in list order: for each engine (in
    `profile.engines` order), for each window (in `profile.windows` order —
    `windows[0]` is the shared baseline, executed once and reused for every
    comparison window that follows it), warmup runs first, then measured
    runs. `order_index` is a single counter across the entire call, so a
    later statistical pass (issue #714) can test for order/thermal bias
    without another schema migration.
    """
    resolved_git_sha = git_sha_value if git_sha_value is not None else git_sha(repo_root)
    resolved_hardware_id = hardware_id_value if hardware_id_value is not None else hardware_id()
    p_hash = prompt_hash(profile.prompt)

    active: dict[str, EngineAdapter] = {}
    missing: list[str] = []
    for engine in profile.engines:
        if engine in adapters:
            active[engine] = adapters[engine]
        else:
            missing.append(engine)
    if missing and not allow_missing_engine:
        raise MissingEngineError(
            f"profile {profile.name!r} requires engine adapter(s) {missing} that are not registered; "
            "pass --allow-missing-engine to run without them (stale/prior results are never substituted)"
        )

    observations: list[Observation] = []
    order_index = 0
    for engine_name in profile.engines:
        adapter = active.get(engine_name)
        if adapter is None:
            continue
        for window in profile.windows:
            for is_warmup, count in ((True, profile.warmup_repeats), (False, profile.measured_repeats)):
                for run_index in range(1, count + 1):
                    t0 = clock()
                    result = adapter.run(prompt=profile.prompt, n_tokens=window, warmup=is_warmup)
                    t1 = clock()
                    obs = Observation(
                        schema_version=SCHEMA_VERSION,
                        git_sha=resolved_git_sha,
                        profile=profile.name,
                        engine=engine_name,
                        engine_version=result.engine_version,
                        model=profile.model,
                        quantization=profile.quantization,
                        prompt_hash=p_hash,
                        requested_prompt_tokens=profile.requested_prompt_tokens,
                        actual_prompt_tokens=result.actual_prompt_tokens,
                        requested_completion_tokens=window,
                        actual_completion_tokens=result.actual_completion_tokens,
                        warmup=is_warmup,
                        run_index=run_index,
                        order_index=order_index,
                        elapsed_ns=t1 - t0,
                        engine_native_ns=result.native_ns,
                        hardware_id=resolved_hardware_id,
                        timestamp=timestamp_fn(),
                    )
                    validate_observation(obs.to_dict())
                    observations.append(obs)
                    order_index += 1
    return HarnessRunResult(
        profile=profile,
        observations=tuple(observations),
        missing_engines=tuple(missing),
    )


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SlopeResult:
    engine: str
    baseline_window: int
    window: int
    slope_tok_per_s: float
    slope_ci95: float | None
    baseline_n: int
    window_n: int


def _measured_elapsed_seconds(observations: tuple[Observation, ...], engine: str, window: int) -> list[float]:
    return [
        o.elapsed_ns / 1e9
        for o in observations
        if o.engine == engine and o.requested_completion_tokens == window and not o.warmup
    ]


def _median_stat(values: list[float]) -> tuple[float, None]:
    return statistics.median(values), None


def _trimmed_mean_stat(values: list[float], trim: int) -> tuple[float, float]:
    ordered = sorted(values)
    trimmed = ordered[trim: len(ordered) - trim] if trim > 0 and len(ordered) > 2 * trim else ordered
    mean = statistics.mean(trimmed)
    if len(trimmed) > 1:
        ci95 = 1.96 * (statistics.stdev(trimmed) / math.sqrt(len(trimmed)))
    else:
        ci95 = 0.0
    return mean, ci95


def _window_stat(values: list[float], aggregation: str, trim: int) -> tuple[float, float | None]:
    if aggregation == "trimmed_mean":
        return _trimmed_mean_stat(values, trim)
    return _median_stat(values)


def aggregate(run_result: HarnessRunResult) -> list[SlopeResult]:
    """Compute the marginal decode-throughput slope for every (engine, window) pair.

    `slope_tok_per_s = (window - baseline_window) / (T(window) - T(baseline))`,
    with `T` the profile's configured aggregation of measured (non-warmup)
    elapsed time. `baseline_window` is always `profile.windows[0]`.
    """
    profile = run_result.profile
    baseline_window = profile.windows[0]
    results: list[SlopeResult] = []
    for engine in profile.engines:
        baseline_values = _measured_elapsed_seconds(run_result.observations, engine, baseline_window)
        if not baseline_values:
            continue
        baseline_mean, baseline_ci = _window_stat(baseline_values, profile.aggregation, profile.trim)
        for window in profile.windows[1:]:
            values = _measured_elapsed_seconds(run_result.observations, engine, window)
            if not values:
                continue
            window_mean, window_ci = _window_stat(values, profile.aggregation, profile.trim)
            dt = window_mean - baseline_mean
            slope = (window - baseline_window) / dt if dt > 0 else float("nan")
            slope_ci: float | None = None
            if (
                profile.aggregation == "trimmed_mean"
                and baseline_ci is not None
                and window_ci is not None
                and baseline_mean > 0
                and window_mean > 0
                and slope == slope  # not NaN
            ):
                slope_ci = slope * math.sqrt((baseline_ci / baseline_mean) ** 2 + (window_ci / window_mean) ** 2)
            results.append(
                SlopeResult(
                    engine=engine,
                    baseline_window=baseline_window,
                    window=window,
                    slope_tok_per_s=slope,
                    slope_ci95=slope_ci,
                    baseline_n=len(baseline_values),
                    window_n=len(values),
                )
            )
    return results


def native_throughput(run_result: HarnessRunResult, engine: str) -> float | None:
    """Median engine-native tok/s across every measured observation with a native duration, or None."""
    rates = [
        o.actual_completion_tokens / (o.engine_native_ns / 1e9)
        for o in run_result.observations
        if o.engine == engine and not o.warmup and o.engine_native_ns
    ]
    return statistics.median(rates) if rates else None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_report(run_result: HarnessRunResult, slopes: list[SlopeResult]) -> str:
    profile = run_result.profile
    lines = [f"=== {profile.name} | windows={list(profile.windows)} | aggregation={profile.aggregation} ==="]
    if run_result.missing_engines:
        lines.append(f"  (missing engine adapter(s), skipped: {', '.join(run_result.missing_engines)})")
    if not slopes:
        lines.append("  (no measured data)")
        return "\n".join(lines)
    lines.append(f"  {'engine':<10}{'window':>8}{'slope tok/s':>13}{'±95% CI':>10}{'native tok/s':>14}{'n(base/win)':>13}")
    lines.append("  " + "-" * 68)
    for s in slopes:
        native = native_throughput(run_result, s.engine)
        ci_str = f"{s.slope_ci95:10.1f}" if s.slope_ci95 is not None else f"{'—':>10}"
        native_str = f"{native:14.1f}" if native is not None else f"{'—':>14}"
        lines.append(
            f"  {s.engine:<10}{s.window:>8}{s.slope_tok_per_s:13.1f}{ci_str}{native_str}"
            f"{f'{s.baseline_n}/{s.window_n}':>13}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_decode_harness.py",
        description="Consolidated decode-benchmark harness (issue #813).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a profile and report decode-throughput slopes.")
    run_p.add_argument("--profile", required=True, help="Profile name from bench_decode_profiles.toml")
    run_p.add_argument("--profiles-file", type=Path, default=DEFAULT_PROFILES_FILE)
    run_p.add_argument(
        "--allow-missing-engine",
        action="store_true",
        help="Continue with whatever engine adapters are registered instead of failing on a missing one.",
    )
    run_p.add_argument("--out", type=Path, default=None, help="Write raw observations as JSONL to this path.")

    val_p = sub.add_parser("validate", help="Validate a raw-observation JSONL file against the schema.")
    val_p.add_argument("path", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.command == "validate":
        try:
            rows = validate_jsonl(args.path)
        except ObservationValidationError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"FAIL: could not read {args.path}: {exc}", file=sys.stderr)
            return 1
        print(f"OK: {len(rows)} observation(s) valid against schema v{SCHEMA_VERSION}")
        return 0

    if args.command == "run":
        try:
            _, profiles = load_profiles_file(args.profiles_file)
        except ProfileConfigError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        profile = profiles.get(args.profile)
        if profile is None:
            available = ", ".join(sorted(profiles)) or "(none defined yet)"
            print(f"FAIL: unknown profile {args.profile!r}. Available: {available}", file=sys.stderr)
            return 1
        try:
            result = run_profile(profile, ADAPTER_REGISTRY, allow_missing_engine=args.allow_missing_engine)
        except MissingEngineError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        slopes = aggregate(result)
        print(render_report(result, slopes))
        if args.out is not None:
            write_jsonl(list(result.observations), args.out)
            print(f"\nRaw observations: {args.out}")
        return 0

    return 1  # pragma: no cover -- argparse `required=True` makes this unreachable


if __name__ == "__main__":
    sys.exit(main())

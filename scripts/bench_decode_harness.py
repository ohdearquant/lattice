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

A profile's `engines` list is an ordered set of per-engine run groups, not
a flat list of names: each entry carries its own warmup schedule
(`warmup_repeats`, `warmup_tokens`, optional `warmup_prompt`), its own
MEASURED schedule (`measured_calls`, `measured_order` -- see below), and
`model`/`quantization` (see `EngineRunGroup`), because the legacy scripts
this harness consolidates give each engine its own warmup shape, its own
measured call shape, and, for the Q4/Q8 comparison, a different
quantization per engine within one profile.

A warmup is an explicit pre-window batch: `warmup_repeats` calls at
`warmup_tokens` completion tokens execute ONCE per engine, entirely before
that engine's measured windows -- never once per window -- so
`bench_apples_to_apples.sh`'s single pre-loop 8-token MLX warmup,
`bench_apples_precise.sh`'s twice-at-N2 warmup for every engine, and
`bench_context_scaling.sh`'s single pre-loop 4-token MLX warmup are each
exactly representable, including `bench_compare_1k.py`'s distinct
per-engine warmup prompts via `warmup_prompt` (defaults to `profile.prompt`
when omitted).

An EARLIER version of this module claimed `windows`/`measured_repeats`
alone (profile-wide, one call per window, executed uniformly as
`[w1 x R, w2 x R, ...]` for every engine) were sufficient to represent
every legacy script's MEASURED schedule. That claim was FALSE:
`bench_compare_1k.py` has three different measured shapes in one profile
-- Lattice runs all 1-token calls then all 100-token calls (window-major,
which the uniform default does reproduce); Ollama issues exactly ONE
100-token call per repeat and derives BOTH a window=1 (TTFT) and a
window=100 (total) observation from that single response's two
engine-reported timing components; MLX alternates a 1-token call and a
100-token call inside every repeat (repeat-major, not window-major). No
choice of profile-wide `windows`/`measured_repeats` can produce all three.

Each `EngineRunGroup` therefore carries an optional `measured_calls`: an
ordered tuple of `MeasuredCall(n_tokens, yields_windows)`. When omitted,
the harness derives the default -- one call per `profile.windows` entry,
`yields_windows=(window,)` -- which reproduces every uniform-window
script (`bench_apples_to_apples.sh`, `bench_apples_precise.sh`,
`bench_q4_apples.sh`, `bench_context_scaling.sh`, and Lattice's half of
`bench_compare_1k.py`) unchanged. `measured_order` (`"window_major"`,
the default, or `"repeat_major"`) controls whether the call plan replays
as "all repeats of call[0], then all repeats of call[1], ..." or "one
full pass through the call plan per repeat" -- MLX's alternating
`bench_compare_1k.py` shape needs `"repeat_major"`. A `MeasuredCall` with
`len(yields_windows) > 1` (Ollama's shape: one 100-token call yielding
both window=1 and window=100) requires the adapter to report
`AdapterRunResult.component_ns`, a `{window: elapsed_ns}` map with an
entry for every yielded window -- the harness never re-invokes the engine
to synthesize a second measurement, and never fabricates one from the
single call's own wall-clock time for more than the primary window.
Missing `component_ns` (or a missing window entry within it) raises
`AdapterContractError`, fail-closed. The requested model/quantization is
passed into `EngineAdapter.run()`; an adapter that invokes a different
artifact than requested (fallback, missing checkpoint, ...) reports the
actual identity back via `AdapterRunResult.actual_model`/
`actual_quantization`, and raw observations always record the actual
invoked identity, never a value merely assumed from the profile.

Raw JSONL observation schema (the contract; see `OBSERVATION_FIELDS` /
`validate_observation`): schema version, git SHA, profile name, engine +
version, model/quantization, prompt hash, requested and actual prompt/
completion token counts, warmup/measured flag, run and order indices, the
harness-measured elapsed nanoseconds, an optional engine-native nanosecond
figure, a hardware identifier, and a timestamp. Reports render from this raw
data; they are not the source of truth.

`trimmed_mean` aggregation reports `slope_ci95_legacy`: the pre-existing
`bench_apples_precise.sh` normal-approximation spread heuristic, carried
over unchanged. It is NOT a coverage-valid 95% confidence interval for the
slope -- statistical-methodology correction is explicitly out of scope for
issue #813 and is deferred to a separately reviewed follow-up.

Run with: python3 tests/test_bench_decode_harness.py -v
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


class AdapterContractError(RuntimeError):
    """An adapter's returned result did not satisfy the harness's measured-call
    contract for a multi-window `MeasuredCall` (missing or incomplete
    `AdapterRunResult.component_ns`). Fail-closed: the harness never
    re-invokes the engine to synthesize a missing measurement, and never
    fabricates one from the call's own wall-clock time for more than the
    call's primary window.
    """


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
class MeasuredCall:
    """One call within an engine's measured schedule.

    `n_tokens` is the completion-token budget actually requested from the
    adapter for this call. `yields_windows` names, in order, which
    window(s) (`requested_completion_tokens` values recorded in the raw
    observations) this ONE call produces a measured observation for. The
    common case is `len(yields_windows) == 1` (one call -> one
    observation, `yields_windows == (n_tokens,)`) -- every legacy script
    except `bench_compare_1k.py`'s Ollama adapter fits this shape.
    Ollama's shape is the exception: it issues one `n_tokens=100` call per
    repeat and derives BOTH a window=1 (TTFT) and a window=100 (total)
    observation from that single response's two engine-reported timing
    components, so `yields_windows == (1, 100)` for a call with
    `n_tokens == 100`. When `len(yields_windows) > 1`, every yielded
    window's `elapsed_ns` comes from `AdapterRunResult.component_ns`
    (adapter-reported), never from the harness's own wall-clock
    measurement of the whole call and never fabricated -- see
    `AdapterRunResult.component_ns` and `AdapterContractError`.
    """

    n_tokens: int
    yields_windows: tuple[int, ...]


_MEASURED_ORDER_VALUES = ("window_major", "repeat_major")


@dataclass(frozen=True)
class EngineRunGroup:
    """One engine's schedule within a profile.

    Every legacy script gives each engine its own warmup schedule, its
    own measured call shape, and its own model/quantization identity
    (`bench_apples_to_apples.sh`: only MLX gets a warmup, once, at 8
    tokens, before the N1/N2 loop; `bench_apples_precise.sh`: every
    engine warms twice at N2=512, once before both measured windows;
    `bench_context_scaling.sh`: MLX warms once at 4 tokens before the
    baseline/comparison loop; `bench_compare_1k.py`: Ollama and MLX each
    warm at 4 tokens with their own warmup prompt, distinct from the
    measured prompt, AND the three engines' MEASURED shapes differ --
    Lattice is window-major `[1 x R, 100 x R]`, MLX is repeat-major
    `(1, 100) x R`, and Ollama is one 100-token call per repeat yielding
    both a window=1 and a window=100 observation). A warmup is an
    explicit pre-window batch -- `warmup_repeats` calls at
    `warmup_tokens` completion tokens, using `warmup_prompt` if given or
    `profile.prompt` otherwise -- executed once per engine, entirely
    before that engine's measured calls, never once per window.

    `measured_calls`, when given, overrides the default measured schedule
    (one call per `profile.windows` entry, `yields_windows=(window,)`)
    with an explicit ordered `MeasuredCall` tuple -- required for Ollama's
    single-call-yields-two-windows shape, optional for engines that only
    need a different token budget than `profile.windows` implies.
    `measured_order` (`"window_major"`, the default, or `"repeat_major"`)
    controls whether the (default or explicit) call plan replays as "all
    repeats of call[0], then all repeats of call[1], ..." or "one full
    pass through the call plan per repeat"; `profile.measured_repeats`
    stays profile-wide (every legacy script keeps the repeat COUNT
    uniform across engines; only the call shape/order/token-budget
    differs per engine).
    """

    name: str
    warmup_repeats: int
    warmup_tokens: int | None
    warmup_prompt: str | None
    model: str
    quantization: str
    measured_calls: tuple[MeasuredCall, ...] | None = None
    measured_order: str = "window_major"


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    description: str
    windows: tuple[int, ...]
    measured_repeats: int
    engine_groups: tuple[EngineRunGroup, ...]
    prompt: str
    aggregation: str = "median"
    trim: int = 0
    requested_prompt_tokens: int | None = None

    @property
    def engines(self) -> tuple[str, ...]:
        """Engine names in schedule order (for aggregation/reporting)."""
        return tuple(g.name for g in self.engine_groups)


_TOP_LEVEL_ALLOWED_KEYS = frozenset({"schema_version", "profiles"})
_PROFILE_ALLOWED_KEYS = frozenset(
    {
        "description",
        "windows",
        "measured_repeats",
        "engines",
        "prompt",
        "aggregation",
        "trim",
        "requested_prompt_tokens",
    }
)
_ENGINE_GROUP_ALLOWED_KEYS = frozenset(
    {
        "name",
        "warmup_repeats",
        "warmup_tokens",
        "warmup_prompt",
        "model",
        "quantization",
        "measured_order",
        "measured_calls",
    }
)
_MEASURED_CALL_ALLOWED_KEYS = frozenset({"n_tokens", "yields_windows"})


def load_profiles_file(path: Path) -> tuple[int, dict[str, ProfileConfig]]:
    try:
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ProfileConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError as exc:
        raise ProfileConfigError(f"{path}: could not read profiles file: {exc}") from exc

    extra_top = set(doc) - _TOP_LEVEL_ALLOWED_KEYS
    if extra_top:
        raise ProfileConfigError(f"{path}: unexpected top-level key(s): {sorted(extra_top)}")

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
    extra = set(raw) - _PROFILE_ALLOWED_KEYS
    if extra:
        raise ProfileConfigError(f"profile {name!r}: unexpected key(s): {sorted(extra)}")

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

    if "description" in raw:
        if not isinstance(raw["description"], str):
            raise ProfileConfigError(
                f"profile {name!r}: key 'description' must be str, got {type(raw['description']).__name__}"
            )
        description = raw["description"]
    else:
        description = ""

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

    measured_repeats = require("measured_repeats", int)
    if measured_repeats < 1:
        raise ProfileConfigError(f"profile {name!r}: measured_repeats must be >= 1")

    engines_raw = require("engines", list)
    if not engines_raw:
        raise ProfileConfigError(f"profile {name!r}: engines must not be empty")
    engine_groups: list[EngineRunGroup] = []
    seen_names: set[str] = set()
    for idx, eg_raw in enumerate(engines_raw):
        if not isinstance(eg_raw, dict):
            raise ProfileConfigError(f"profile {name!r}: engines[{idx}] must be a table")
        eg_extra = set(eg_raw) - _ENGINE_GROUP_ALLOWED_KEYS
        if eg_extra:
            raise ProfileConfigError(f"profile {name!r}: engines[{idx}] has unexpected key(s): {sorted(eg_extra)}")

        def eg_require(key: str, expected_type: type, *, _idx: int = idx, _eg_raw: dict = eg_raw) -> object:
            if key not in _eg_raw:
                raise ProfileConfigError(f"profile {name!r}: engines[{_idx}] missing required key {key!r}")
            value = _eg_raw[key]
            if expected_type is int and isinstance(value, bool):
                raise ProfileConfigError(f"profile {name!r}: engines[{_idx}] key {key!r} must be int, got bool")
            if not isinstance(value, expected_type):
                raise ProfileConfigError(
                    f"profile {name!r}: engines[{_idx}] key {key!r} must be "
                    f"{expected_type.__name__}, got {type(value).__name__}"
                )
            return value

        eg_name = eg_require("name", str)
        if not eg_name:
            raise ProfileConfigError(f"profile {name!r}: engines[{idx}] name must be non-empty")
        if eg_name in seen_names:
            raise ProfileConfigError(f"profile {name!r}: engines must not contain duplicate name {eg_name!r}")
        seen_names.add(eg_name)

        eg_warmup = eg_require("warmup_repeats", int)
        if eg_warmup < 0:
            raise ProfileConfigError(f"profile {name!r}: engines[{idx}] warmup_repeats must be >= 0")

        # Warmup is an explicit pre-window batch: `warmup_tokens` is the
        # completion-token budget for those `warmup_repeats` calls, and is
        # only meaningful (and only required) when there is a warmup to
        # schedule -- the same "only meaningful together" pattern as
        # aggregation='trimmed_mean'/trim above.
        eg_warmup_tokens_raw = eg_raw.get("warmup_tokens")
        if eg_warmup_tokens_raw is not None:
            if isinstance(eg_warmup_tokens_raw, bool) or not isinstance(eg_warmup_tokens_raw, int):
                raise ProfileConfigError(f"profile {name!r}: engines[{idx}] warmup_tokens must be an int")
            if eg_warmup_tokens_raw <= 0:
                raise ProfileConfigError(f"profile {name!r}: engines[{idx}] warmup_tokens must be a positive int")
        if eg_warmup > 0 and eg_warmup_tokens_raw is None:
            raise ProfileConfigError(
                f"profile {name!r}: engines[{idx}] warmup_tokens is required when warmup_repeats > 0"
            )
        if eg_warmup == 0 and eg_warmup_tokens_raw is not None:
            raise ProfileConfigError(
                f"profile {name!r}: engines[{idx}] warmup_tokens is only meaningful when warmup_repeats > 0"
            )
        eg_warmup_tokens = eg_warmup_tokens_raw

        # warmup_prompt is optional at any warmup_repeats value (it is
        # simply unused when warmup_repeats == 0); when omitted, the
        # warmup uses `profile.prompt` (see `run_profile`).
        eg_warmup_prompt_raw = eg_raw.get("warmup_prompt")
        if eg_warmup_prompt_raw is not None:
            if not isinstance(eg_warmup_prompt_raw, str):
                raise ProfileConfigError(f"profile {name!r}: engines[{idx}] warmup_prompt must be a string")
            if not eg_warmup_prompt_raw:
                raise ProfileConfigError(
                    f"profile {name!r}: engines[{idx}] warmup_prompt must be non-empty if provided"
                )
        eg_warmup_prompt = eg_warmup_prompt_raw

        eg_model = eg_require("model", str)
        if not eg_model:
            raise ProfileConfigError(f"profile {name!r}: engines[{idx}] model must be non-empty")

        eg_quantization = eg_require("quantization", str)
        if not eg_quantization:
            raise ProfileConfigError(f"profile {name!r}: engines[{idx}] quantization must be non-empty")

        # measured_order: how this engine's measured call plan (default or
        # explicit measured_calls) replays across profile.measured_repeats
        # -- "window_major" (all repeats of call[0], then call[1], ...) or
        # "repeat_major" (one full pass through the plan per repeat, the
        # shape MLX's alternating bench_compare_1k.py calls need).
        eg_measured_order = eg_raw.get("measured_order", "window_major")
        if not isinstance(eg_measured_order, str) or eg_measured_order not in _MEASURED_ORDER_VALUES:
            raise ProfileConfigError(
                f"profile {name!r}: engines[{idx}] measured_order must be one of "
                f"{_MEASURED_ORDER_VALUES}, got {eg_measured_order!r}"
            )

        # measured_calls: an explicit ordered call plan overriding the
        # default (one call per profile.windows entry). Required for any
        # engine whose measured shape the default cannot represent (e.g.
        # Ollama's one-call-yields-two-windows shape).
        eg_measured_calls_raw = eg_raw.get("measured_calls")
        eg_measured_calls: tuple[MeasuredCall, ...] | None = None
        if eg_measured_calls_raw is not None:
            if not isinstance(eg_measured_calls_raw, list) or not eg_measured_calls_raw:
                raise ProfileConfigError(
                    f"profile {name!r}: engines[{idx}] measured_calls must be a non-empty list"
                )
            parsed_calls: list[MeasuredCall] = []
            for call_idx, call_raw in enumerate(eg_measured_calls_raw):
                if not isinstance(call_raw, dict):
                    raise ProfileConfigError(
                        f"profile {name!r}: engines[{idx}] measured_calls[{call_idx}] must be a table"
                    )
                call_extra = set(call_raw) - _MEASURED_CALL_ALLOWED_KEYS
                if call_extra:
                    raise ProfileConfigError(
                        f"profile {name!r}: engines[{idx}] measured_calls[{call_idx}] has "
                        f"unexpected key(s): {sorted(call_extra)}"
                    )
                call_missing = _MEASURED_CALL_ALLOWED_KEYS - set(call_raw)
                if call_missing:
                    raise ProfileConfigError(
                        f"profile {name!r}: engines[{idx}] measured_calls[{call_idx}] missing "
                        f"required key(s): {sorted(call_missing)}"
                    )
                call_n_tokens = call_raw["n_tokens"]
                if isinstance(call_n_tokens, bool) or not isinstance(call_n_tokens, int) or call_n_tokens <= 0:
                    raise ProfileConfigError(
                        f"profile {name!r}: engines[{idx}] measured_calls[{call_idx}] n_tokens "
                        f"must be a positive int, got {call_n_tokens!r}"
                    )
                call_yields_raw = call_raw["yields_windows"]
                if not isinstance(call_yields_raw, list) or not call_yields_raw:
                    raise ProfileConfigError(
                        f"profile {name!r}: engines[{idx}] measured_calls[{call_idx}] yields_windows "
                        "must be a non-empty list"
                    )
                call_yields: list[int] = []
                for y in call_yields_raw:
                    if isinstance(y, bool) or not isinstance(y, int) or y <= 0:
                        raise ProfileConfigError(
                            f"profile {name!r}: engines[{idx}] measured_calls[{call_idx}] yields_windows "
                            f"entries must be positive integers, got {y!r}"
                        )
                    call_yields.append(y)
                if len(set(call_yields)) != len(call_yields):
                    raise ProfileConfigError(
                        f"profile {name!r}: engines[{idx}] measured_calls[{call_idx}] yields_windows "
                        f"must not contain duplicates, got {call_yields}"
                    )
                parsed_calls.append(MeasuredCall(n_tokens=call_n_tokens, yields_windows=tuple(call_yields)))
            eg_measured_calls = tuple(parsed_calls)

        engine_groups.append(
            EngineRunGroup(
                name=eg_name,
                warmup_repeats=eg_warmup,
                warmup_tokens=eg_warmup_tokens,
                warmup_prompt=eg_warmup_prompt,
                model=eg_model,
                quantization=eg_quantization,
                measured_calls=eg_measured_calls,
                measured_order=eg_measured_order,
            )
        )

    prompt = require("prompt", str)
    if not prompt:
        raise ProfileConfigError(f"profile {name!r}: prompt must be non-empty")

    aggregation = raw.get("aggregation", "median")
    if not isinstance(aggregation, str) or aggregation not in AGGREGATION_METHODS:
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
        measured_repeats=measured_repeats,
        engine_groups=tuple(engine_groups),
        prompt=prompt,
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

    `actual_model`/`actual_quantization` report the identity the adapter
    actually invoked. They default to `None`, meaning "the requested
    identity was invoked verbatim" -- the harness then records the
    requested `model`/`quantization` it passed into `run()`. An adapter
    that falls back to a different artifact than requested (a missing
    checkpoint, a resolved default, ...) MUST set these so the raw
    observation records what actually ran, not what was asked for.

    `component_ns` is REQUIRED when the harness's `MeasuredCall` for this
    invocation declared `len(yields_windows) > 1` (one physical call
    producing observations for multiple windows, e.g. Ollama deriving
    both a TTFT-equivalent and a total-equivalent reading from one
    `/api/generate` response): it must map every window in
    `yields_windows` to that window's adapter-reported elapsed
    nanoseconds. It is ignored when the call yields exactly one window
    (that window's `elapsed_ns` always comes from the harness's own
    wall-clock measurement of the call). A missing map, or a map missing
    an entry for one of the declared windows, raises
    `AdapterContractError` -- the harness never fabricates a component it
    was not told.
    """

    actual_completion_tokens: int
    actual_prompt_tokens: int | None = None
    native_ns: int | None = None
    engine_version: str = "unknown"
    actual_model: str | None = None
    actual_quantization: str | None = None
    component_ns: Mapping[int, int] | None = None


class EngineAdapter(Protocol):
    def run(
        self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str
    ) -> AdapterRunResult: ...


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

    Run order is the requested schedule in list order: for each engine run
    group (in `profile.engine_groups` order), that engine's warmup batch —
    `warmup_repeats` calls at `warmup_tokens` completion tokens, using
    `warmup_prompt` if given or `profile.prompt` otherwise — executes
    ONCE, entirely before any of that engine's measured calls (never once
    per window; this is what makes `bench_apples_to_apples.sh`'s single
    pre-loop MLX warmup, `bench_apples_precise.sh`'s twice-at-N2 warmup,
    and `bench_context_scaling.sh`'s single pre-loop warmup all exactly
    representable).

    Measured calls then execute per the group's OWN plan: `group.
    measured_calls` if given, else the default derived from
    `profile.windows` (one call per window, `yields_windows=(window,)`).
    `group.measured_order` selects "window_major" (default -- all repeats
    of call[0], then all repeats of call[1], ...) or "repeat_major" (one
    full pass through the call plan per repeat -- MLX's alternating
    `bench_compare_1k.py` shape). A call with more than one yielded window
    (Ollama's one-call-yields-two-windows shape) produces one Observation
    per yielded window from a SINGLE adapter invocation, using
    `AdapterRunResult.component_ns` for every window's `elapsed_ns` — see
    `MeasuredCall` and `AdapterContractError`. `order_index` is a single
    counter across the entire call, so a later statistical pass (issue
    #714) can test for order/thermal bias without another schema
    migration.
    """
    resolved_git_sha = git_sha_value if git_sha_value is not None else git_sha(repo_root)
    resolved_hardware_id = hardware_id_value if hardware_id_value is not None else hardware_id()
    measured_prompt_hash = prompt_hash(profile.prompt)

    active: dict[str, EngineRunGroup] = {}
    missing: list[str] = []
    for group in profile.engine_groups:
        if group.name in adapters:
            active[group.name] = group
        else:
            missing.append(group.name)
    if missing and not allow_missing_engine:
        raise MissingEngineError(
            f"profile {profile.name!r} requires engine adapter(s) {missing} that are not registered; "
            "pass --allow-missing-engine to run without them (stale/prior results are never substituted)"
        )

    observations: list[Observation] = []
    order_index = 0

    def _append_observation(
        *,
        group: EngineRunGroup,
        result: AdapterRunResult,
        call_prompt_hash: str,
        window: int,
        is_warmup: bool,
        run_index: int,
        elapsed_ns: int,
    ) -> None:
        nonlocal order_index
        actual_model = result.actual_model if result.actual_model is not None else group.model
        actual_quantization = (
            result.actual_quantization if result.actual_quantization is not None else group.quantization
        )
        obs = Observation(
            schema_version=SCHEMA_VERSION,
            git_sha=resolved_git_sha,
            profile=profile.name,
            engine=group.name,
            engine_version=result.engine_version,
            model=actual_model,
            quantization=actual_quantization,
            prompt_hash=call_prompt_hash,
            requested_prompt_tokens=profile.requested_prompt_tokens,
            actual_prompt_tokens=result.actual_prompt_tokens,
            requested_completion_tokens=window,
            actual_completion_tokens=result.actual_completion_tokens,
            warmup=is_warmup,
            run_index=run_index,
            order_index=order_index,
            elapsed_ns=elapsed_ns,
            engine_native_ns=result.native_ns,
            hardware_id=resolved_hardware_id,
            timestamp=timestamp_fn(),
        )
        validate_observation(obs.to_dict())
        observations.append(obs)
        order_index += 1

    def _record(
        *,
        adapter: EngineAdapter,
        group: EngineRunGroup,
        call_prompt: str,
        call_prompt_hash: str,
        n_tokens: int,
        is_warmup: bool,
        run_index: int,
    ) -> None:
        """Single call, single window (warmup calls always take this path)."""
        t0 = clock()
        result = adapter.run(
            prompt=call_prompt,
            n_tokens=n_tokens,
            warmup=is_warmup,
            model=group.model,
            quantization=group.quantization,
        )
        t1 = clock()
        _append_observation(
            group=group,
            result=result,
            call_prompt_hash=call_prompt_hash,
            window=n_tokens,
            is_warmup=is_warmup,
            run_index=run_index,
            elapsed_ns=t1 - t0,
        )

    def _record_measured_call(
        *,
        adapter: EngineAdapter,
        group: EngineRunGroup,
        call: MeasuredCall,
        call_prompt: str,
        call_prompt_hash: str,
        run_index: int,
    ) -> None:
        """One measured call, possibly yielding more than one window's
        observation from a single adapter invocation (see `MeasuredCall`).
        """
        t0 = clock()
        result = adapter.run(
            prompt=call_prompt,
            n_tokens=call.n_tokens,
            warmup=False,
            model=group.model,
            quantization=group.quantization,
        )
        t1 = clock()
        if len(call.yields_windows) == 1:
            _append_observation(
                group=group,
                result=result,
                call_prompt_hash=call_prompt_hash,
                window=call.yields_windows[0],
                is_warmup=False,
                run_index=run_index,
                elapsed_ns=t1 - t0,
            )
            return
        if result.component_ns is None:
            raise AdapterContractError(
                f"engine {group.name!r}: measured call n_tokens={call.n_tokens} declares "
                f"yields_windows={call.yields_windows} but AdapterRunResult.component_ns is None"
            )
        missing_windows = [w for w in call.yields_windows if w not in result.component_ns]
        if missing_windows:
            raise AdapterContractError(
                f"engine {group.name!r}: measured call n_tokens={call.n_tokens} "
                f"component_ns is missing entries for window(s) {missing_windows} "
                f"(declared yields_windows={call.yields_windows})"
            )
        for window in call.yields_windows:
            _append_observation(
                group=group,
                result=result,
                call_prompt_hash=call_prompt_hash,
                window=window,
                is_warmup=False,
                run_index=run_index,
                elapsed_ns=result.component_ns[window],
            )

    def _default_measured_calls() -> tuple[MeasuredCall, ...]:
        return tuple(MeasuredCall(n_tokens=w, yields_windows=(w,)) for w in profile.windows)

    for group in profile.engine_groups:
        if group.name not in active:
            continue
        adapter = adapters[group.name]

        # Warmup batch: once, before any measured call, at the group's
        # own token budget and prompt (default: profile.prompt).
        if group.warmup_repeats > 0:
            warmup_prompt = group.warmup_prompt if group.warmup_prompt is not None else profile.prompt
            warmup_prompt_hash = (
                measured_prompt_hash if group.warmup_prompt is None else prompt_hash(group.warmup_prompt)
            )
            for run_index in range(1, group.warmup_repeats + 1):
                _record(
                    adapter=adapter,
                    group=group,
                    call_prompt=warmup_prompt,
                    call_prompt_hash=warmup_prompt_hash,
                    n_tokens=group.warmup_tokens,
                    is_warmup=True,
                    run_index=run_index,
                )

        # Measured calls: this engine's own plan (default or explicit),
        # replayed window-major or repeat-major per group.measured_order.
        calls = group.measured_calls if group.measured_calls is not None else _default_measured_calls()
        if group.measured_order == "window_major":
            for call in calls:
                for run_index in range(1, profile.measured_repeats + 1):
                    _record_measured_call(
                        adapter=adapter,
                        group=group,
                        call=call,
                        call_prompt=profile.prompt,
                        call_prompt_hash=measured_prompt_hash,
                        run_index=run_index,
                    )
        else:  # "repeat_major"
            for run_index in range(1, profile.measured_repeats + 1):
                for call in calls:
                    _record_measured_call(
                        adapter=adapter,
                        group=group,
                        call=call,
                        call_prompt=profile.prompt,
                        call_prompt_hash=measured_prompt_hash,
                        run_index=run_index,
                    )

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
    # Legacy normal-approximation heuristic carried over verbatim from
    # `bench_apples_precise.sh:118-143` (issue #813 explicitly defers
    # statistical-methodology changes to a separately reviewed follow-up).
    # This is NOT a coverage-valid 95% confidence interval for the slope:
    # it treats the trimmed subset as an ordinary sample, applies a
    # large-sample z=1.96 multiplier regardless of the retained sample
    # size, and combines the two window CIs with relative-error
    # (product/ratio) propagation rather than the sensitivity-coefficient
    # propagation a difference-then-reciprocal statistic requires. Present
    # it to a reader as a legacy spread heuristic only, never as "±95% CI".
    slope_ci95_legacy: float | None
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
    """Trimmed mean + the legacy normal-approximation spread heuristic.

    The second return value is `bench_apples_precise.sh`'s pre-existing
    `1.96 * stdev / sqrt(n)` heuristic over the *retained* (trimmed)
    subset, carried over verbatim rather than corrected -- see
    `SlopeResult.slope_ci95_legacy` for why it is not a coverage-valid CI.
    """
    ordered = sorted(values)
    trimmed = ordered[trim: len(ordered) - trim] if trim > 0 and len(ordered) > 2 * trim else ordered
    mean = statistics.mean(trimmed)
    if len(trimmed) > 1:
        ci95_legacy = 1.96 * (statistics.stdev(trimmed) / math.sqrt(len(trimmed)))
    else:
        ci95_legacy = 0.0
    return mean, ci95_legacy


def _window_stat(values: list[float], aggregation: str, trim: int) -> tuple[float, float | None]:
    if aggregation == "trimmed_mean":
        return _trimmed_mean_stat(values, trim)
    return _median_stat(values)


def aggregate(run_result: HarnessRunResult) -> list[SlopeResult]:
    """Compute the marginal decode-throughput slope for every (engine, window) pair.

    `slope_tok_per_s = (window - baseline_window) / (T(window) - T(baseline))`,
    with `T` the profile's configured aggregation of measured (non-warmup)
    elapsed time. `baseline_window` is always `profile.windows[0]`.

    `SlopeResult.slope_ci95_legacy`, when the profile uses `trimmed_mean`
    aggregation, is the pre-existing `bench_apples_precise.sh` spread
    heuristic propagated through the slope's product/ratio form -- a
    legacy carry-over, not a corrected or coverage-valid CI (issue #813
    defers statistical-methodology changes to a separate review).
    """
    profile = run_result.profile
    baseline_window = profile.windows[0]
    results: list[SlopeResult] = []
    for engine in profile.engines:
        baseline_values = _measured_elapsed_seconds(run_result.observations, engine, baseline_window)
        if not baseline_values:
            continue
        baseline_mean, baseline_ci_legacy = _window_stat(baseline_values, profile.aggregation, profile.trim)
        for window in profile.windows[1:]:
            values = _measured_elapsed_seconds(run_result.observations, engine, window)
            if not values:
                continue
            window_mean, window_ci_legacy = _window_stat(values, profile.aggregation, profile.trim)
            dt = window_mean - baseline_mean
            slope = (window - baseline_window) / dt if dt > 0 else float("nan")
            slope_ci_legacy: float | None = None
            if (
                profile.aggregation == "trimmed_mean"
                and baseline_ci_legacy is not None
                and window_ci_legacy is not None
                and baseline_mean > 0
                and window_mean > 0
                and slope == slope  # not NaN
            ):
                slope_ci_legacy = slope * math.sqrt(
                    (baseline_ci_legacy / baseline_mean) ** 2 + (window_ci_legacy / window_mean) ** 2
                )
            results.append(
                SlopeResult(
                    engine=engine,
                    baseline_window=baseline_window,
                    window=window,
                    slope_tok_per_s=slope,
                    slope_ci95_legacy=slope_ci_legacy,
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
    # "legacy ±CI" -- the normal-approximation heuristic carried over from
    # bench_apples_precise.sh, not a coverage-valid 95% CI. See
    # SlopeResult.slope_ci95_legacy.
    lines.append(
        f"  {'engine':<10}{'window':>8}{'slope tok/s':>13}{'legacy ±CI':>11}{'native tok/s':>14}{'n(base/win)':>13}"
    )
    lines.append("  " + "-" * 69)
    for s in slopes:
        native = native_throughput(run_result, s.engine)
        ci_str = f"{s.slope_ci95_legacy:11.1f}" if s.slope_ci95_legacy is not None else f"{'—':>11}"
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

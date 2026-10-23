#!/usr/bin/env python3
"""Decode-benchmark harness core (issue #813).

`scripts/` grew seven independent implementations of the same
prompt/decode-benchmark shape (`legacy apples-to-apples consumer`,
`legacy precise apples-to-apples consumer`, `legacy Q4 comparison consumer`, `legacy context-scaling consumer`,
`legacy agentic consumer`, plus two proven-dead scripts tracked separately), each
owning its own copy of warmup policy, sample-count policy,
aggregation/confidence-interval method, and output rendering. This module is
the single place that owns all of that: configuration validation, warmup
count, requested schedule, run order, monotonic wall timing, the raw
observation schema, aggregation, confidence-interval calculation, and output
rendering.

Engine adapters (Lattice, Ollama, MLX) own only invocation and parsing: they
implement `EngineAdapter.run()`, may surface an engine-native duration as a
diagnostic field, and MUST NOT choose repeat counts, discard samples, or
compute verdicts. Adapter implementations remain isolated in sibling modules;
the canonical `run --profile` CLI loads the matching module before executing
the shared schedule. All five consumers are defined in
`bench_decode_profiles.toml`.

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
`legacy apples-to-apples consumer`'s single pre-loop 8-token MLX warmup,
`legacy precise apples-to-apples consumer`'s twice-at-N2 warmup for every engine, and
`legacy context-scaling consumer`'s single pre-loop 4-token MLX warmup are each
exactly representable, including `legacy agentic consumer`'s distinct
per-engine warmup prompts via `warmup_prompt` (defaults to `profile.prompt`
when omitted).

An EARLIER version of this module claimed `windows`/`measured_repeats`
alone (profile-wide, one call per window, executed uniformly as
`[w1 x R, w2 x R, ...]` for every engine) were sufficient to represent
every legacy script's MEASURED schedule. That claim was FALSE:
`legacy agentic consumer` has three different measured shapes in one profile
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
script (`legacy apples-to-apples consumer`, `legacy precise apples-to-apples consumer`,
`legacy Q4 comparison consumer`, `legacy context-scaling consumer`, and Lattice's half of
`legacy agentic consumer`) unchanged. `measured_order` (`"window_major"`,
the default, or `"repeat_major"`) controls whether the call plan replays
as "all repeats of call[0], then all repeats of call[1], ..." or "one
full pass through the call plan per repeat" -- MLX's alternating
`legacy agentic consumer` shape needs `"repeat_major"`. A `MeasuredCall` with
`len(yields_windows) > 1` (Ollama's shape: one 100-token call yielding
both window=1 and window=100) requires the adapter to report
`AdapterRunResult.component_ns`, a `{window: elapsed_ns}` map with an
entry for every yielded window -- the harness never re-invokes the engine
to synthesize a second measurement, and never fabricates one from the
single call's own wall-clock time for more than the primary window.
Missing `component_ns` (or a missing window entry within it) raises
`AdapterContractError`, fail-closed. Every explicit `measured_calls` plan
is ALSO validated against `profile.windows`: the flattened
`yields_windows` across the whole plan must cover every declared window
exactly once -- a missing window silently erases a slope, an undeclared
window produces an observation outside the report contract, and a window
yielded twice silently doubles that window's sample count/aggregation
weight. All three are rejected fail-closed at parse time, naming the
engine index and the missing/undeclared/duplicate windows.

A SECOND earlier version of this module additionally assumed the
MEASURED prompt could stay profile-wide (`profile.prompt`, passed
verbatim to every engine's measured calls). That assumption was ALSO
FALSE: `legacy agentic consumer` requires each engine to receive a DIFFERENT
measured prompt for the same nominal context -- Lattice and MLX build a
tokenizer-padded prompt, Ollama builds a distinct character-count-
heuristic prompt. `EngineRunGroup.measured_prompt` (mirroring
`warmup_prompt`) lets an engine override the measured prompt; the harness
passes exactly that resolved string to the adapter and hashes exactly
that string, never a placeholder the adapter is trusted to reinterpret --
prompt construction stays a harness-owned request, not an adapter
implementation detail. The requested model/quantization is passed into
`EngineAdapter.run()`; an adapter that invokes a different artifact than
requested (fallback, missing checkpoint, ...) reports the actual identity
back via `AdapterRunResult.actual_model`/`actual_quantization`, and raw
observations always record the actual invoked identity, never a value
merely assumed from the profile.

Raw JSONL observation schema (the contract; see `OBSERVATION_FIELDS` /
`OPTIONAL_OBSERVATION_FIELDS` / `validate_observation`): schema version, git SHA, profile name, engine +
version, model/quantization, prompt hash, requested and actual prompt/
completion token counts, warmup/measured flag, run and order indices, the
harness-measured elapsed nanoseconds, an optional engine-native nanosecond
figure, a hardware identifier, a timestamp, and an optional scope string. Reports render from this raw
data; they are not the source of truth.

`trimmed_mean` aggregation reports `slope_ci95_legacy`: the pre-existing
`legacy precise apples-to-apples consumer` normal-approximation spread heuristic, carried
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
import importlib
import importlib.util
import inspect
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_RUNNING_PYTHON = ".".join(str(part) for part in sys.version_info[:3])
_PYTHON_REQUIREMENT_ERROR = (
    f"{sys.argv[0]} requires Python 3.11 or newer; "
    f"running Python {_RUNNING_PYTHON} at {sys.executable}"
)
try:
    import tomllib
    from datetime import UTC, datetime
except ImportError:
    raise SystemExit(_PYTHON_REQUIREMENT_ERROR) from None
if sys.version_info[:2] < (3, 11):
    raise SystemExit(_PYTHON_REQUIREMENT_ERROR)

sys.modules.setdefault("bench_decode_harness", sys.modules[__name__])

SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES_FILE = REPO_ROOT / "scripts" / "bench_decode_profiles.toml"

# `bench_gate_math` is a sibling script module, not an installed package;
# inserting this file's own directory onto `sys.path` lets `import
# bench_gate_math` resolve both when this file is run directly
# (`python3 scripts/bench_decode_harness.py ...`) and when it is loaded by
# path via `importlib.util.spec_from_file_location` (the test-suite
# convention -- see tests/test_bench_decode_harness.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_gate_math  # noqa: E402

# Schema v2 (run records, the expected-cell registry and their validators) lives in
# bench_run_record.py; its names stay reachable as attributes of this module.
from bench_run_record import (  # noqa: E402
    RUN_RECORD_SCHEMA_VERSION,
    PHASE_EVENT_NAMES,
    VERDICTS,
    METRIC_FAMILIES,
    CADENCES,
    DEFAULT_EXPECTED_CELLS_FILE,
    RunRecordValidationError,
    ExpectedCellConfigError,
    PhaseEvent,
    ResourceSample,
    LockReceipt,
    ProvenanceRecord,
    CellAggregate,
    CellRecord,
    PromotionRecord,
    _require_keys,
    _PHASE_EVENT_KEYS,
    _PHASE_EVENT_REQUIRED,
    parse_phase_event,
    _RESOURCE_SAMPLE_KEYS,
    _RESOURCE_SAMPLE_REQUIRED,
    parse_resource_sample,
    _LOCK_RECEIPT_KEYS,
    parse_lock_receipt,
    _PROVENANCE_KEYS,
    _PROVENANCE_REQUIRED,
    parse_provenance,
    _CELL_AGGREGATE_KEYS,
    _CELL_AGGREGATE_REQUIRED,
    parse_cell_aggregate,
    _CELL_RECORD_KEYS,
    _CELL_RECORD_REQUIRED,
    parse_cell_record,
    _require_list,
    _PROMOTION_RECORD_KEYS,
    parse_promotion_record,
    ExpectedCellGroup,
    _EXPECTED_GROUP_KEYS,
    load_expected_cells,
    cell_id,
    expand_cell_group,
    expected_cell_ids_for_cadence,
    validate_registry_coverage,
    _REQUIRED_SINGLE_PHASES,
    _PHASE_NAME_RANK,
    _validate_phase_sequence,
    validate_run_record,
    validate_promotion_record,
    rank_cells_by_gap,
)

AGGREGATION_METHODS = ("median", "trimmed_mean")


class ProfileConfigError(ValueError):
    """A profile definition in `bench_decode_profiles.toml` is invalid."""


class ObservationValidationError(ValueError):
    """A raw observation does not conform to the schema contract."""


class MissingEngineError(RuntimeError):
    """A profile requested an engine with no adapter registered for it."""


class NoMeasurementError(MissingEngineError):
    """`allow_missing_engine` was set and the run produced no measured
    observation at all, so there is nothing to report.

    A subclass of `MissingEngineError` so that every caller already treating
    a missing engine as a failed run inherits this without its own check --
    the distinction the message must carry is "an engine was missing"
    (survivable, other engines measured) versus "no measurement was
    produced" (the run reports nothing regardless of how far the reporting
    path got).
    """


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

# Fields added after the first v1 rows were written. Absent is accepted so
# stored rows keep validating; present values are type-checked like any other
# field. `SCHEMA_VERSION` stays 1 because the contract only grows additively.
OPTIONAL_OBSERVATION_FIELDS: dict[str, object] = {
    "scope": (str, type(None)),
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
    scope: str | None = None

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

    extra = set(row) - set(OBSERVATION_FIELDS) - set(OPTIONAL_OBSERVATION_FIELDS)
    if extra:
        raise ObservationValidationError(f"unexpected field(s): {sorted(extra)}")
    missing = set(OBSERVATION_FIELDS) - set(row)
    if missing:
        raise ObservationValidationError(f"missing required field(s): {sorted(missing)}")

    for name, expected in {**OBSERVATION_FIELDS, **OPTIONAL_OBSERVATION_FIELDS}.items():
        if name in row:
            _check_field_type(name, row[name], expected)

    if row["schema_version"] != SCHEMA_VERSION:
        raise ObservationValidationError(
            f"schema_version {row['schema_version']!r} != supported {SCHEMA_VERSION}"
        )
    if row.get("scope") is not None and not row["scope"]:
        raise ObservationValidationError("field 'scope' must be non-empty or null")
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
    if not rows:
        raise ObservationValidationError(
            f"{path}: no observations found (file is empty or contains only blank lines)"
        )
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
    except `legacy agentic consumer`'s Ollama adapter fits this shape.
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
    (`legacy apples-to-apples consumer`: only MLX gets a warmup, once, at 8
    tokens, before the N1/N2 loop; `legacy precise apples-to-apples consumer`: every
    engine warms twice at N2=512, once before both measured windows;
    `legacy context-scaling consumer`: MLX warms once at 4 tokens before the
    baseline/comparison loop; `legacy agentic consumer`: Ollama and MLX each
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

    `measured_prompt`, when given, overrides `profile.prompt` for THIS
    engine's measured calls only -- mirroring `warmup_prompt`'s override
    of the warmup prompt. `legacy agentic consumer` needs this: Lattice and
    MLX build a tokenizer-padded prompt to reach a target context length,
    while Ollama builds a character-count-heuristic prompt for the SAME
    target -- two different strings for the same nominal context. The
    harness passes exactly the resolved prompt (per-engine override or
    `profile.prompt`) to the adapter and hashes exactly that string, never
    a placeholder the adapter is trusted to reinterpret: prompt
    construction/ownership stays in the harness's request, not the
    adapter's implementation.
    """

    name: str
    warmup_repeats: int
    warmup_tokens: int | None
    warmup_prompt: str | None
    model: str
    quantization: str
    measured_calls: tuple[MeasuredCall, ...] | None = None
    measured_order: str = "window_major"
    measured_prompt: str | None = None


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
        "measured_prompt",
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
        # shape MLX's alternating legacy agentic consumer calls need).
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

            # Fail-closed relationship check: the flattened yields_windows
            # across the WHOLE plan (one pass through `calls`) must cover
            # profile.windows exactly once each -- no missing window (an
            # erased slope), no undeclared window (an observation outside
            # the report contract), and no window yielded twice across
            # different calls (a silently doubled sample count/aggregation
            # weight). Order across the flattened plan is NOT required to
            # match `windows`' order: measured_order/call order governs
            # REPLAY order (window-major vs repeat-major), not which
            # windows exist -- only coverage (each exactly once) is
            # enforced here.
            flattened_windows = [w for c in parsed_calls for w in c.yields_windows]
            flattened_counts: dict[int, int] = {}
            for w in flattened_windows:
                flattened_counts[w] = flattened_counts.get(w, 0) + 1
            declared_windows = set(windows)
            flattened_set = set(flattened_windows)
            missing_windows = sorted(declared_windows - flattened_set)
            undeclared_windows = sorted(flattened_set - declared_windows)
            duplicate_windows = sorted(w for w, c in flattened_counts.items() if c > 1 and w in declared_windows)
            if missing_windows or undeclared_windows or duplicate_windows:
                problems = []
                if missing_windows:
                    problems.append(f"missing window(s) {missing_windows}")
                if undeclared_windows:
                    problems.append(f"undeclared window(s) {undeclared_windows}")
                if duplicate_windows:
                    problems.append(f"duplicate window(s) {duplicate_windows}")
                raise ProfileConfigError(
                    f"profile {name!r}: engines[{idx}] measured_calls yields_windows "
                    f"must cover profile windows {windows} exactly once each: {'; '.join(problems)}"
                )

            eg_measured_calls = tuple(parsed_calls)

        # measured_prompt: overrides profile.prompt for THIS engine's
        # measured calls only (mirrors warmup_prompt). Optional at any
        # measured_calls value -- unused (falls back to profile.prompt)
        # when omitted.
        eg_measured_prompt_raw = eg_raw.get("measured_prompt")
        if eg_measured_prompt_raw is not None:
            if not isinstance(eg_measured_prompt_raw, str):
                raise ProfileConfigError(f"profile {name!r}: engines[{idx}] measured_prompt must be a string")
            if not eg_measured_prompt_raw:
                raise ProfileConfigError(
                    f"profile {name!r}: engines[{idx}] measured_prompt must be non-empty if provided"
                )
        eg_measured_prompt = eg_measured_prompt_raw

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
                measured_prompt=eg_measured_prompt,
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
    becomes the primary aggregated duration (`engine_native_ns` on the
    resulting `Observation`, see `_primary_elapsed_seconds`); the
    harness-measured wall-clock timing is the fallback used only when no
    native duration was reported.

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
    scope: str | None = None


class EngineAdapter(Protocol):
    def run(
        self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str
    ) -> AdapterRunResult: ...


# Populated by the adapter module selected by the canonical profile CLI.
# Programmatic callers may instead pass their own adapter mapping directly to
# `run_profile`, as the deterministic tests do.
ADAPTER_REGISTRY: dict[str, EngineAdapter] = {}


def register_adapter(name: str, adapter: EngineAdapter) -> None:
    ADAPTER_REGISTRY[name] = adapter


# Why an engine was left out of ADAPTER_REGISTRY, so the report can say so.
ADAPTER_MISSING_REASONS: dict[str, str] = {}


def register_missing_adapter(name: str, reason: str) -> None:
    ADAPTER_MISSING_REASONS[name] = reason


OLLAMA_SCOPE = "external server; only request interval and unload/check ran under the locks"
WORKER_PROTOCOL_VERSION = 1
LATTICE_HANDOFF_ARGV = [
    "--gpu-handoff", "--durable", "--label", "UNIQUE_LABEL", "--", "cargo", "run", "--locked",
    "--release", "-p", "lattice-inference", "--bin", "bench_decode_ab", "--features", "metal-gpu,f16",
]


def run_lattice_handoff(env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    command = [str((REPO_ROOT / "scripts" / "bench-command.sh").resolve()), *LATTICE_HANDOFF_ARGV]
    command[4] = f"decode-lattice-{uuid.uuid4().hex}"
    return subprocess.run(command, cwd=REPO_ROOT, env=dict(env), capture_output=True, text=True, check=False)


def validate_ollama_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("LATTICE_BENCH_OLLAMA_URL must be an http loopback URL")
    return f"http://{parsed.netloc}"


def ollama_model_available(base_url: str, model_tag: str, *, pull_if_missing: bool = False) -> bool:
    """Check the configured Ollama server for a model, optionally pulling it when absent."""
    base_url = validate_ollama_url(base_url)

    def listed_models() -> set[str] | None:
        try:
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
                data = json.loads(response.read())
        except (OSError, ValueError, urllib.error.URLError):
            return None
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return None
        names: set[str] = set()
        for model in models:
            if isinstance(model, dict):
                names.update(value for value in (model.get("name"), model.get("model")) if isinstance(value, str))
        return names

    names = listed_models()
    if names is None:
        return False
    if model_tag in names:
        return True
    if not pull_if_missing or shutil.which("ollama") is None:
        return False

    env = os.environ.copy()
    env["OLLAMA_HOST"] = base_url
    try:
        listed = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=30, check=False, env=env
        )
        if listed.returncode != 0:
            return False
        if not any(row.split()[:1] == [model_tag] for row in listed.stdout.splitlines()):
            pulled = subprocess.run(
                ["ollama", "pull", model_tag], capture_output=True, text=True, timeout=600, check=False, env=env
            )
            if pulled.returncode != 0:
                return False
    except (OSError, subprocess.TimeoutExpired):
        return False
    names = listed_models()
    return names is not None and model_tag in names


def _wire_group(group: EngineRunGroup) -> dict:
    return {
        "name": group.name,
        "warmup_repeats": group.warmup_repeats,
        "warmup_tokens": group.warmup_tokens,
        "warmup_prompt": group.warmup_prompt,
        "model": group.model,
        "quantization": group.quantization,
        "measured_calls": None if group.measured_calls is None else [dataclasses.asdict(call) for call in group.measured_calls],
        "measured_order": group.measured_order,
        "measured_prompt": group.measured_prompt,
    }


def _wire_profile(profile: ProfileConfig, group: EngineRunGroup) -> dict:
    return {
        "name": profile.name,
        "description": profile.description,
        "windows": list(profile.windows),
        "measured_repeats": profile.measured_repeats,
        "engine_groups": [_wire_group(group)],
        "prompt": profile.prompt,
        "aggregation": profile.aggregation,
        "trim": profile.trim,
        "requested_prompt_tokens": profile.requested_prompt_tokens,
    }


def _adapter_wire(adapter: EngineAdapter) -> dict:
    cls = type(adapter)
    values = vars(adapter)
    kwargs: dict[str, object] = {}
    for name, parameter in inspect.signature(cls.__init__).parameters.items():
        if name == "self":
            continue
        if name not in values:
            if parameter.default is inspect.Parameter.empty:
                raise RuntimeError(f"worker adapter {cls.__name__} has no serializable value for {name}")
            continue
        value = values[name]
        if isinstance(value, Path):
            value = str(value.resolve())
        if value is not None and type(value) not in (str, int, float, bool):
            raise RuntimeError(f"worker adapter {cls.__name__}.{name} is not JSON serializable")
        kwargs[name] = value
    return {"module": cls.__module__, "class": cls.__name__, "kwargs": kwargs}


def prepare_worker_prompt(
    adapter: EngineAdapter, context_tokens: int, missing_reason: list[str] | None = None
) -> tuple[str, int] | None:
    result = _launch_plain_worker(
        {
            "protocol_version": WORKER_PROTOCOL_VERSION,
            "operation": "prepare_prompt",
            "engine": "mlx",
            "profile": None,
            "group": None,
            "adapter": _adapter_wire(adapter),
            "prepared_prompt_tokens": None,
            "prompt_context": context_tokens,
        },
        label=f"decode-mlx-prompt-{uuid.uuid4().hex[:12]}",
    )
    if result["status"] == "missing":
        if missing_reason is not None:
            missing_reason.append(result["reason"])
        return None
    prompt = result["prepared_prompt"]
    count = result["prompt_tokens"]
    if not prompt or type(count) is not int or count < context_tokens:
        raise RuntimeError("MLX prompt worker returned an invalid padded prompt")
    return prompt, count


def _group_schedule(profile: ProfileConfig, group: EngineRunGroup) -> list[tuple[bool, int, int, str]]:
    expected: list[tuple[bool, int, int, str]] = []
    if group.warmup_repeats:
        prompt = group.warmup_prompt if group.warmup_prompt is not None else profile.prompt
        expected.extend((True, n, group.warmup_tokens, prompt) for n in range(1, group.warmup_repeats + 1))
    calls = group.measured_calls or tuple(MeasuredCall(n_tokens=w, yields_windows=(w,)) for w in profile.windows)
    prompt = group.measured_prompt if group.measured_prompt is not None else profile.prompt
    if group.measured_order == "repeat_major":
        sequence = ((call, n) for n in range(1, profile.measured_repeats + 1) for call in calls)
    else:
        sequence = ((call, n) for call in calls for n in range(1, profile.measured_repeats + 1))
    for call, n in sequence:
        expected.extend((False, n, window, prompt) for window in call.yields_windows)
    return expected


def _validate_worker_observations(profile: ProfileConfig, group: EngineRunGroup, rows: object) -> tuple[Observation, ...]:
    if not isinstance(rows, list):
        raise RuntimeError("decode worker observations must be an array")
    schedule = _group_schedule(profile, group)
    if len(rows) != len(schedule):
        raise RuntimeError(f"decode worker returned {len(rows)} observations; expected {len(schedule)}")
    observations = []
    for index, (row, (warmup, run_index, window, prompt)) in enumerate(zip(rows, schedule, strict=True)):
        validate_observation(row)
        if (
            row["profile"] != profile.name
            or row["engine"] != group.name
            or row["warmup"] is not warmup
            or row["run_index"] != run_index
            or row["requested_completion_tokens"] != window
            or row["prompt_hash"] != prompt_hash(prompt)
            or row["order_index"] != index
            or row.get("scope") != (OLLAMA_SCOPE if group.name == "ollama" else None)
        ):
            raise RuntimeError(f"decode worker observation {index} does not match the requested schedule")
        if type(row["elapsed_ns"]) is not int or row["elapsed_ns"] < 0:
            raise RuntimeError(f"decode worker observation {index} has invalid elapsed_ns")
        native = row["engine_native_ns"]
        if native is not None and (type(native) is not int or native < 0):
            raise RuntimeError(f"decode worker observation {index} has invalid engine_native_ns")
        observations.append(Observation(**row))
    return tuple(observations)


def _validate_worker_result(result: object) -> dict:
    keys = {"result_version", "status", "reason", "observations", "metadata", "prepared_prompt", "prompt_tokens"}
    if not isinstance(result, dict) or set(result) != keys:
        raise RuntimeError("decode worker result has invalid fields")
    if type(result["result_version"]) is not int or result["result_version"] != WORKER_PROTOCOL_VERSION:
        raise RuntimeError("decode worker result has an unsupported version")
    if result["status"] not in {"ok", "missing"}:
        raise RuntimeError("decode worker result has an invalid status")
    if result["reason"] is not None and not isinstance(result["reason"], str):
        raise RuntimeError("decode worker reason must be a string or null")
    if not isinstance(result["observations"], list) or not isinstance(result["metadata"], dict):
        raise RuntimeError("decode worker result observations or metadata have an invalid type")
    if set(result["metadata"]) - {"ollama_breakdowns", "scope", "unload_confirmed"}:
        raise RuntimeError("decode worker metadata has unexpected fields")
    if "scope" in result["metadata"] and result["metadata"]["scope"] != OLLAMA_SCOPE:
        raise RuntimeError("decode worker scope is invalid")
    if "unload_confirmed" in result["metadata"] and result["metadata"]["unload_confirmed"] is not True:
        raise RuntimeError("decode worker unload confirmation is invalid")
    if "ollama_breakdowns" in result["metadata"]:
        breakdowns = result["metadata"]["ollama_breakdowns"]
        fields = {"load_ms", "prompt_eval_ms", "eval_ms", "prompt_eval_count"}
        if not isinstance(breakdowns, list):
            raise RuntimeError("decode worker Ollama breakdowns must be an array")
        for breakdown in breakdowns:
            if not isinstance(breakdown, dict) or set(breakdown) != fields:
                raise RuntimeError("decode worker Ollama breakdown has invalid fields")
            for field, value in breakdown.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise RuntimeError(f"decode worker Ollama breakdown {field} is invalid")
    if result["status"] == "missing":
        if not result["reason"] or result["observations"] or result["metadata"]:
            raise RuntimeError("missing decode worker result has inconsistent contents")
    elif result["reason"] is not None:
        raise RuntimeError("successful decode worker result must have a null reason")
    if result["prepared_prompt"] is not None and not isinstance(result["prepared_prompt"], str):
        raise RuntimeError("decode worker prepared_prompt must be a string or null")
    if result["prompt_tokens"] is not None and (type(result["prompt_tokens"]) is not int or result["prompt_tokens"] < 1):
        raise RuntimeError("decode worker prompt_tokens must be a positive int or null")
    return result


def _mlx_missing_reason() -> str:
    return f"mlx_lm is not importable by {sys.executable}"


def _launch_plain_worker(request: dict, *, label: str) -> dict:
    cache = REPO_ROOT / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    request_path = cache / f"decode-worker-{token}.json"
    result_path = cache / f"decode-worker-{token}-result.json"
    request = {**request, "result_path": str(result_path.resolve())}
    request_path.write_text(json.dumps(request, separators=(",", ":")) + "\n")
    command = [
        str((REPO_ROOT / "scripts" / "bench-command.sh").resolve()), "--durable", "--label", label, "--",
        # Not resolved: inside a virtual environment sys.executable is a symlink to the base
        # interpreter, and resolving it would start a worker that cannot see the environment's packages.
        os.path.abspath(sys.executable), str(Path(__file__).resolve()), "worker", "--request", str(request_path.resolve()),
    ]
    try:
        proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"decode worker supervisor exited {proc.returncode}: {proc.stderr[-2000:]}")
        # The after-probe is part of the supervisor result: don't inspect rows before this status is zero.
        return _validate_worker_result(json.loads(result_path.read_text()))
    finally:
        request_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def _require_worker_keys(value: object, keys: set[str], where: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"decode worker {where} has invalid fields")
    return value


def _read_worker_profile(raw: object) -> ProfileConfig:
    keys = {"name", "description", "windows", "measured_repeats", "engine_groups", "prompt", "aggregation", "trim", "requested_prompt_tokens"}
    item = _require_worker_keys(raw, keys, "profile")
    for name in ("name", "description", "prompt", "aggregation"):
        if not isinstance(item[name], str):
            raise ValueError(f"decode worker profile {name} must be a string")
    if not isinstance(item["windows"], list) or any(type(x) is not int for x in item["windows"]):
        raise ValueError("decode worker profile windows must be an integer array")
    if type(item["measured_repeats"]) is not int or type(item["trim"]) is not int:
        raise ValueError("decode worker profile counts must be integers")
    prompt_tokens = item["requested_prompt_tokens"]
    if prompt_tokens is not None and type(prompt_tokens) is not int:
        raise ValueError("decode worker requested_prompt_tokens must be an integer or null")
    if not isinstance(item["engine_groups"], list) or len(item["engine_groups"]) != 1:
        raise ValueError("decode worker profile must contain one engine group")
    group = _read_worker_group(item["engine_groups"][0])
    return ProfileConfig(
        name=item["name"], description=item["description"], windows=tuple(item["windows"]),
        measured_repeats=item["measured_repeats"], engine_groups=(group,), prompt=item["prompt"],
        aggregation=item["aggregation"], trim=item["trim"], requested_prompt_tokens=prompt_tokens,
    )


def _read_worker_group(raw: object) -> EngineRunGroup:
    keys = {"name", "warmup_repeats", "warmup_tokens", "warmup_prompt", "model", "quantization", "measured_calls", "measured_order", "measured_prompt"}
    item = _require_worker_keys(raw, keys, "group")
    for name in ("name", "model", "quantization", "measured_order"):
        if not isinstance(item[name], str):
            raise ValueError(f"decode worker group {name} must be a string")
    for name in ("warmup_prompt", "measured_prompt"):
        if item[name] is not None and not isinstance(item[name], str):
            raise ValueError(f"decode worker group {name} must be a string or null")
    if type(item["warmup_repeats"]) is not int:
        raise ValueError("decode worker warmup_repeats must be an integer")
    if item["warmup_tokens"] is not None and type(item["warmup_tokens"]) is not int:
        raise ValueError("decode worker warmup_tokens must be an integer or null")
    calls = item["measured_calls"]
    if calls is not None:
        if not isinstance(calls, list):
            raise ValueError("decode worker measured_calls must be an array or null")
        parsed = []
        for call in calls:
            call = _require_worker_keys(call, {"n_tokens", "yields_windows"}, "measured call")
            if type(call["n_tokens"]) is not int or not isinstance(call["yields_windows"], list) or any(type(x) is not int for x in call["yields_windows"]):
                raise ValueError("decode worker measured call has invalid token fields")
            parsed.append(MeasuredCall(call["n_tokens"], tuple(call["yields_windows"])))
        calls = tuple(parsed)
    return EngineRunGroup(
        name=item["name"], warmup_repeats=item["warmup_repeats"], warmup_tokens=item["warmup_tokens"],
        warmup_prompt=item["warmup_prompt"], model=item["model"], quantization=item["quantization"],
        measured_calls=calls, measured_order=item["measured_order"], measured_prompt=item["measured_prompt"],
    )


def _atomic_worker_result(path: Path, *, status: str, reason: str | None = None,
                          observations: list[dict] | None = None, metadata: dict | None = None,
                          prepared_prompt: str | None = None, prompt_tokens: int | None = None) -> None:
    payload = {
        "result_version": WORKER_PROTOCOL_VERSION,
        "status": status,
        "reason": reason,
        "observations": observations or [],
        "metadata": metadata or {},
        "prepared_prompt": prepared_prompt,
        "prompt_tokens": prompt_tokens,
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    temporary.replace(path)


class _TrackedAdapter:
    def __init__(self, adapter: EngineAdapter):
        self.adapter = adapter
        self.started = False

    def run(self, **kwargs) -> AdapterRunResult:
        self.started = True
        return self.adapter.run(**kwargs)


def run_decode_worker(request_path: Path) -> int:
    """Child entry for versioned group requests. Called only under a plain durable launch."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
    from bench_supervision import sample_measurement_handoff

    _, pipe_fd, _ = sample_measurement_handoff()
    os.close(pipe_fd)
    keys = {"protocol_version", "operation", "engine", "profile", "group", "adapter", "prepared_prompt_tokens", "prompt_context", "result_path"}
    request = _require_worker_keys(json.loads(request_path.read_text()), keys, "request")
    if type(request["protocol_version"]) is not int or request["protocol_version"] != WORKER_PROTOCOL_VERSION:
        raise ValueError("decode worker request has an unsupported version")
    if request["operation"] not in {"group", "prepare_prompt"}:
        raise ValueError("decode worker operation is invalid")
    engine = request["engine"]
    if engine not in {"mlx", "ollama"} and request["operation"] == "group":
        raise ValueError("decode worker only accepts MLX or Ollama groups")
    result_path = Path(request["result_path"]).resolve()
    try:
        result_path.relative_to((REPO_ROOT / ".cache").resolve())
    except ValueError as exc:
        raise ValueError("decode worker result path must be inside repository .cache") from exc
    adapter_spec = _require_worker_keys(request["adapter"], {"module", "class", "kwargs"}, "adapter")
    if not isinstance(adapter_spec["module"], str) or not adapter_spec["module"].startswith("bench_decode_adapters_"):
        raise ValueError("decode worker adapter module is invalid")
    if not isinstance(adapter_spec["class"], str) or not isinstance(adapter_spec["kwargs"], dict):
        raise ValueError("decode worker adapter class or kwargs are invalid")
    if any(not isinstance(k, str) or (v is not None and type(v) not in (str, int, float, bool)) for k, v in adapter_spec["kwargs"].items()):
        raise ValueError("decode worker adapter kwargs have invalid types")
    if request["operation"] == "prepare_prompt":
        context = request["prompt_context"]
        if type(context) is not int or context < 1 or request["engine"] != "mlx":
            raise ValueError("decode worker prompt preparation request is invalid")
    else:
        if request["prompt_context"] is not None or request["profile"] is None or request["group"] is None:
            raise ValueError("decode worker group request is invalid")
    module = importlib.import_module(adapter_spec["module"])
    adapter_type = getattr(module, adapter_spec["class"], None)
    if not isinstance(adapter_type, type) or adapter_type.__module__ != adapter_spec["module"]:
        raise ValueError("decode worker adapter class does not match its module")
    adapter = adapter_type(**adapter_spec["kwargs"])
    if request["operation"] == "prepare_prompt":
        if importlib.util.find_spec("mlx_lm") is None:
            _atomic_worker_result(result_path, status="missing", reason=_mlx_missing_reason())
            return 0
        prepared = adapter.padded_prompt(request["prompt_context"])
        token_count = adapter._prompt_token_counts[prepared]
        _atomic_worker_result(result_path, status="ok", prepared_prompt=prepared, prompt_tokens=token_count)
        return 0
    profile = _read_worker_profile(request["profile"])
    group = _read_worker_group(request["group"])
    if len(profile.engine_groups) != 1 or profile.engine_groups[0] != group or group.name != engine:
        raise ValueError("decode worker profile and group do not match")
    prepared_tokens = request["prepared_prompt_tokens"]
    if prepared_tokens is not None and (type(prepared_tokens) is not int or prepared_tokens < 1):
        raise ValueError("decode worker prepared_prompt_tokens must be a positive integer or null")
    if engine == "mlx" and importlib.util.find_spec("mlx_lm") is None:
        _atomic_worker_result(result_path, status="missing", reason=_mlx_missing_reason())
        return 0
    if engine == "ollama":
        adapter.base_url = validate_ollama_url(adapter.base_url)
    tracked = _TrackedAdapter(adapter)
    if engine == "mlx" and prepared_tokens is not None:
        prompt = group.measured_prompt if group.measured_prompt is not None else profile.prompt
        if hasattr(adapter, "_prompt_token_counts"):
            adapter._prompt_token_counts[prompt] = prepared_tokens
    run_error: BaseException | None = None
    run_result = None
    try:
        run_result = run_profile(profile, {engine: tracked})
    except BaseException as exc:
        run_error = exc
    unload_error: BaseException | None = None
    if engine == "ollama" and tracked.started:
        try:
            _ollama_unload(adapter.base_url, group.model)
        except BaseException as exc:
            unload_error = exc
    if unload_error is not None:
        raise RuntimeError(f"ollama unload/check failed: {unload_error}") from unload_error
    if run_error is not None:
        raise run_error
    assert run_result is not None
    metadata: dict[str, object] = {}
    breakdowns = getattr(adapter, "breakdowns", None)
    if isinstance(breakdowns, list):
        metadata["ollama_breakdowns"] = breakdowns
    if engine == "ollama":
        metadata["scope"] = OLLAMA_SCOPE
        metadata["unload_confirmed"] = True
    _atomic_worker_result(
        result_path,
        status="ok",
        observations=[observation.to_dict() for observation in run_result.observations],
        metadata=metadata,
    )
    return 0


def _ollama_unload(base_url: str, model: str) -> None:
    base_url = validate_ollama_url(base_url)
    body = json.dumps({"model": model, "prompt": "", "stream": False, "keep_alive": 0}).encode()
    request = urllib.request.Request(f"{base_url}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=600) as response:
        unloaded = json.loads(response.read())
    if not isinstance(unloaded, dict) or unloaded.get("done") is not True:
        raise RuntimeError("ollama unload request did not return done=true")
    with urllib.request.urlopen(f"{base_url}/api/ps", timeout=30) as response:
        loaded = json.loads(response.read())
    if not isinstance(loaded, dict) or not isinstance(loaded.get("models"), list):
        raise RuntimeError("ollama /api/ps response did not contain a models array")
    names = []
    for entry in loaded["models"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"]:
            raise RuntimeError("ollama /api/ps contains an invalid model entry")
        names.append(entry["name"])
    if model in names:
        raise RuntimeError(f"ollama model {model!r} remains loaded after unload")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessRunResult:
    profile: ProfileConfig
    observations: tuple[Observation, ...]
    missing_engines: tuple[str, ...]
    engine_metadata: Mapping[str, object] = dataclasses.field(default_factory=dict)


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


def coordinator_is_supervised() -> bool:
    if any(
        os.environ.get(name)
        for name in (
            "LATTICE_BENCH_LOCK_STATUS",
            "LATTICE_BENCH_LOCK_FDS",
            "LATTICE_BENCH_SUPERVISOR_FD",
            "LATTICE_GPU_HANDOFF_CONTROL",
        )
    ):
        print("FAIL: decode coordinator refuses to run inside existing benchmark supervision", file=sys.stderr)
        return True
    return False


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


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
    supervised_children: bool = False,
    worker_prompt_tokens: int | None = None,
) -> HarnessRunResult:
    """Execute `profile` against `adapters` and return every raw observation.

    Run order is the requested schedule in list order: for each engine run
    group (in `profile.engine_groups` order), that engine's warmup batch —
    `warmup_repeats` calls at `warmup_tokens` completion tokens, using
    `warmup_prompt` if given or `profile.prompt` otherwise — executes
    ONCE, entirely before any of that engine's measured calls (never once
    per window; this is what makes `legacy apples-to-apples consumer`'s single
    pre-loop MLX warmup, `legacy precise apples-to-apples consumer`'s twice-at-N2 warmup,
    and `legacy context-scaling consumer`'s single pre-loop warmup all exactly
    representable).

    Measured calls then execute per the group's OWN plan: `group.
    measured_calls` if given, else the default derived from
    `profile.windows` (one call per window, `yields_windows=(window,)`).
    `group.measured_order` selects "window_major" (default -- all repeats
    of call[0], then all repeats of call[1], ...) or "repeat_major" (one
    full pass through the call plan per repeat -- MLX's alternating
    `legacy agentic consumer` shape). A call with more than one yielded window
    (Ollama's one-call-yields-two-windows shape) produces one Observation
    per yielded window from a SINGLE adapter invocation, using
    `AdapterRunResult.component_ns` for every window's `elapsed_ns` — see
    `MeasuredCall` and `AdapterContractError`. Every measured call for a
    group uses that group's `measured_prompt` (default: `profile.prompt`)
    — the exact string is what is passed to `adapter.run(prompt=...)` and
    what `prompt_hash` records, never a placeholder the adapter
    reinterprets (see `EngineRunGroup.measured_prompt`). `order_index` is
    a single counter across the entire call, so a later statistical pass
    (issue #714) can test for order/thermal bias without another schema
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
    engine_metadata: dict[str, object] = {}
    for name in missing:
        if name in ADAPTER_MISSING_REASONS:
            engine_metadata[name] = {"missing_reason": ADAPTER_MISSING_REASONS[name]}
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
            scope=result.scope if result.scope is not None else (OLLAMA_SCOPE if group.name == "ollama" else None),
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
        if supervised_children and group.name == "lattice":
            result = adapter.run(
                prompt=call_prompt,
                n_tokens=n_tokens,
                warmup=is_warmup,
                model=group.model,
                quantization=group.quantization,
            )
            if result.native_ns is None:
                raise AdapterContractError("lattice handoff must return the binary's per-call duration")
            elapsed_ns = result.native_ns
        else:
            t0 = clock()
            result = adapter.run(
                prompt=call_prompt,
                n_tokens=n_tokens,
                warmup=is_warmup,
                model=group.model,
                quantization=group.quantization,
            )
            elapsed_ns = clock() - t0
        _append_observation(
            group=group,
            result=result,
            call_prompt_hash=call_prompt_hash,
            window=n_tokens,
            is_warmup=is_warmup,
            run_index=run_index,
            elapsed_ns=elapsed_ns,
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
        if supervised_children and group.name == "lattice":
            result = adapter.run(
                prompt=call_prompt,
                n_tokens=call.n_tokens,
                warmup=False,
                model=group.model,
                quantization=group.quantization,
            )
            if result.native_ns is None:
                raise AdapterContractError("lattice handoff must return the binary's per-call duration")
            elapsed_ns = result.native_ns
        else:
            t0 = clock()
            result = adapter.run(
                prompt=call_prompt,
                n_tokens=call.n_tokens,
                warmup=False,
                model=group.model,
                quantization=group.quantization,
            )
            elapsed_ns = clock() - t0
        if len(call.yields_windows) == 1:
            _append_observation(
                group=group,
                result=result,
                call_prompt_hash=call_prompt_hash,
                window=call.yields_windows[0],
                is_warmup=False,
                run_index=run_index,
                elapsed_ns=elapsed_ns,
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

        if (
            supervised_children
            and group.name in {"mlx", "ollama"}
            and type(adapter).__module__.startswith("bench_decode_adapters_")
        ):
            worker = _launch_plain_worker(
                {
                    "protocol_version": WORKER_PROTOCOL_VERSION,
                    "operation": "group",
                    "engine": group.name,
                    "profile": _wire_profile(profile, group),
                    "group": _wire_group(group),
                    "adapter": _adapter_wire(adapter),
                    "prepared_prompt_tokens": worker_prompt_tokens,
                    "prompt_context": None,
                },
                label=f"decode-{group.name}-{uuid.uuid4().hex[:12]}",
            )
            if worker["status"] == "missing":
                missing.append(group.name)
                engine_metadata[group.name] = {"missing_reason": worker["reason"]}
                if not allow_missing_engine:
                    raise MissingEngineError(
                        f"profile {profile.name!r} requires {group.name}, unavailable: {worker['reason']}"
                    )
                continue
            child_rows = _validate_worker_observations(profile, group, worker["observations"])
            for child_row in child_rows:
                observations.append(dataclasses.replace(child_row, order_index=order_index))
                order_index += 1
            engine_metadata[group.name] = worker["metadata"]
            continue

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
        # replayed window-major or repeat-major per group.measured_order,
        # using this engine's own measured prompt (default: profile.prompt)
        # -- e.g. legacy agentic consumer's Lattice/MLX tokenizer-padded prompt
        # vs Ollama's character-count-heuristic prompt for the same
        # nominal context: two different strings, hashed exactly as sent.
        calls = group.measured_calls if group.measured_calls is not None else _default_measured_calls()
        group_measured_prompt = group.measured_prompt if group.measured_prompt is not None else profile.prompt
        group_measured_prompt_hash = (
            measured_prompt_hash if group.measured_prompt is None else prompt_hash(group.measured_prompt)
        )
        if group.measured_order == "window_major":
            for call in calls:
                for run_index in range(1, profile.measured_repeats + 1):
                    _record_measured_call(
                        adapter=adapter,
                        group=group,
                        call=call,
                        call_prompt=group_measured_prompt,
                        call_prompt_hash=group_measured_prompt_hash,
                        run_index=run_index,
                    )
        else:  # "repeat_major"
            for run_index in range(1, profile.measured_repeats + 1):
                for call in calls:
                    _record_measured_call(
                        adapter=adapter,
                        group=group,
                        call=call,
                        call_prompt=group_measured_prompt,
                        call_prompt_hash=group_measured_prompt_hash,
                        run_index=run_index,
                    )

    # A result that a caller can render and exit zero on has to mean something
    # was MEASURED. `allow_missing_engine` lets a run proceed when an engine is
    # absent, which is right for a sweep where one framework is not installed;
    # it must not also let a run where EVERY engine was absent return a result
    # indistinguishable from a full comparison. Enforced here, where the
    # measurement is produced, so every wrapper inherits it: a per-wrapper copy
    # of the check leaves each new wrapper free to omit it.
    if allow_missing_engine and not any(not obs.warmup for obs in observations):
        detail = f"engine(s) {list(missing)} were unavailable" if missing else "no engine ran a measured call"
        raise NoMeasurementError(
            f"profile {profile.name!r} produced no measured observation: {detail}. "
            "--allow-missing-engine tolerates a missing engine, not a missing measurement."
        )

    return HarnessRunResult(
        profile=profile,
        observations=tuple(observations),
        missing_engines=tuple(missing),
        engine_metadata=engine_metadata,
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
    # `legacy precise apples-to-apples consumer:118-143` (issue #813 explicitly defers
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


def _primary_elapsed_seconds(o: Observation) -> float:
    """Duration used for aggregation, slope computation and legacy TSV rendering.
    Use adapter-reported `engine_native_ns` when available; otherwise use the
    harness's outer `elapsed_ns` (for example, MLX's in-process `generate()`,
    where the wall clock measures the work without a separate native reading).

    Outer timing includes the whole adapter invocation. For subprocess-per-call
    Lattice/Ollama adapters, it includes process startup and, for Lattice,
    model loading on every measured or warmup call. Legacy precise, Q4 and
    context-scaling consumers instead loaded once per process and used
    `RESULT total_ms=` or Ollama's `total_duration` for each observation.
    Aggregating outer timing would introduce those costs and break comparison
    with the corresponding legacy metric. Keep `elapsed_ns` for provenance
    and diagnostics, but prefer native timing whenever the adapter provides it.

    Native timing alone does not make different adapter metrics comparable.
    The context-scaling Ollama adapter now reports `eval_duration` instead of
    `total_duration`, excluding model loading and prompt evaluation.
    Its old timings and slopes require a fresh baseline. This correction is
    specific to that adapter; it does not change the other legacy profiles'
    timing definitions or make their observations interchangeable.
    """
    return (o.engine_native_ns if o.engine_native_ns is not None else o.elapsed_ns) / 1e9


def _measured_elapsed_seconds(observations: tuple[Observation, ...], engine: str, window: int) -> list[float]:
    return [
        _primary_elapsed_seconds(o)
        for o in observations
        if o.engine == engine and o.requested_completion_tokens == window and not o.warmup
    ]


def _median_stat(values: list[float]) -> tuple[float, None]:
    return statistics.median(values), None


def _trimmed_mean_stat(values: list[float], trim: int) -> tuple[float, float]:
    """Trimmed mean + the legacy normal-approximation spread heuristic.

    The second return value is `legacy precise apples-to-apples consumer`'s pre-existing
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
    aggregation, is the pre-existing `legacy precise apples-to-apples consumer` spread
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
        for name in run_result.missing_engines:
            metadata = run_result.engine_metadata.get(name)
            reason = metadata.get("missing_reason") if isinstance(metadata, Mapping) else None
            if reason:
                lines.append(f"    {name}: {reason}")
    if "ollama" in profile.engines:
        lines.append(f"  ollama scope: {OLLAMA_SCOPE}")
        ollama_metadata = run_result.engine_metadata.get("ollama", {})
        if isinstance(ollama_metadata, Mapping) and ollama_metadata.get("unload_confirmed") is True:
            lines.append("  ollama unload/check: confirmed")
    if not slopes:
        lines.append("  (no measured data)")
        return "\n".join(lines)
    # "legacy ±CI" -- the normal-approximation heuristic carried over from
    # legacy precise apples-to-apples consumer, not a coverage-valid 95% CI. See
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
    run_p.add_argument("--runs", type=int, default=None, help="Override measured repeats for supported profiles.")
    run_p.add_argument("--ctx", type=int, default=None, help="Agentic profile context depth in tokens.")
    run_p.add_argument("--contexts", default=None, help="Context-scaling comparison windows, comma-separated.")
    run_p.add_argument("--sweep", action="store_true", help="Run the agentic profile at 1000/2000/4000 tokens.")
    run_p.add_argument("--chart-only", action="store_true", help="Regenerate the context-scaling chart only.")

    val_p = sub.add_parser("validate", help="Validate a raw-observation JSONL file against the schema.")
    val_p.add_argument("path", type=Path)

    return parser


_PROFILE_ADAPTER_MODULES = {
    "apples_to_apples_q8": "bench_decode_adapters_apples_to_apples",
    "apples_to_apples_q4": "bench_decode_adapters_apples_to_apples",
    "apples_precise": "bench_decode_adapters_apples_precise",
    "q4_apples": "bench_decode_adapters_q4_apples",
    "context_scaling": "bench_decode_adapters_context_scaling",
    "agentic": "bench_decode_adapters_agentic",
}


def _append_flag(argv: list[str], name: str, value: object | None) -> None:
    if value is not None:
        argv.extend((name, str(value)))


def _dispatch_special_profile(args: argparse.Namespace, module: object) -> int | None:
    if args.profile == "agentic":
        if args.contexts is not None or args.chart_only:
            raise ProfileConfigError("profile 'agentic' does not accept --contexts or --chart-only")
        argv: list[str] = []
        _append_flag(argv, "--ctx", args.ctx)
        _append_flag(argv, "--runs", args.runs)
        _append_flag(argv, "--out", args.out)
        if args.sweep:
            argv.append("--sweep")
        if args.allow_missing_engine:
            argv.append("--allow-missing-engine")
        return module.main(argv)
    if args.profile == "context_scaling":
        if args.ctx is not None or args.sweep:
            raise ProfileConfigError("profile 'context_scaling' does not accept --ctx or --sweep")
        argv = []
        _append_flag(argv, "--contexts", args.contexts)
        _append_flag(argv, "--runs", args.runs)
        _append_flag(argv, "--out", args.out)
        if args.chart_only:
            argv.append("--chart-only")
        if args.allow_missing_engine:
            argv.append("--allow-missing-engine")
        return module.main(argv)
    return None


def _load_profile_adapters(args: argparse.Namespace) -> int | None:
    module_name = _PROFILE_ADAPTER_MODULES.get(args.profile)
    if module_name is None:
        return None
    if args.profile in ("agentic", "context_scaling") and args.profiles_file != DEFAULT_PROFILES_FILE:
        raise ProfileConfigError(f"profile {args.profile!r} does not support --profiles-file")
    module = importlib.import_module(module_name)
    special_result = _dispatch_special_profile(args, module)
    if special_result is not None:
        return special_result
    unsupported = (
        args.ctx is not None
        or args.contexts is not None
        or args.sweep
        or args.chart_only
        or args.runs is not None
    )
    if unsupported:
        raise ProfileConfigError(f"profile {args.profile!r} does not accept the requested runtime override")
    ADAPTER_REGISTRY.clear()
    ADAPTER_MISSING_REASONS.clear()
    if args.profile.startswith("apples_to_apples_"):
        module.register_available_adapters(["run", "--profile", args.profile])
    else:
        module.register_available_adapters()
    return None


def main(argv: list[str] | None = None) -> int:
    command_line = list(sys.argv[1:] if argv is None else argv)
    if command_line[:1] == ["worker"]:
        parser = argparse.ArgumentParser(prog="bench_decode_harness.py worker")
        parser.add_argument("--request", type=Path, required=True)
        try:
            return run_decode_worker(parser.parse_args(command_line[1:]).request)
        except Exception as exc:  # worker failure must fail the supervising command
            print(f"FAIL: decode worker: {exc}", file=sys.stderr)
            return 1
    if command_line[:1] == ["run"] and coordinator_is_supervised():
        return 2
    args = build_arg_parser().parse_args(command_line)

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
            dispatched = _load_profile_adapters(args)
        except ProfileConfigError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        if dispatched is not None:
            return dispatched
        try:
            if args.out is not None:
                args.out = args.out.resolve()
            result = run_profile(
                profile,
                ADAPTER_REGISTRY,
                allow_missing_engine=args.allow_missing_engine,
                supervised_children=True,
            )
        except (MissingEngineError, RuntimeError, OSError, ValueError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        slopes = aggregate(result)
        stage_dir = REPO_ROOT / ".cache" / f"decode-report-{uuid.uuid4().hex}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        report_path = stage_dir / "report.txt"
        report_path.write_text(render_report(result, slopes) + "\n")
        print(report_path.read_text(), end="")
        if args.out is not None:
            staged_out = stage_dir / "observations.jsonl"
            write_jsonl(list(result.observations), staged_out)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged_out, args.out)
            print(f"\nRaw observations: {args.out}")
        shutil.rmtree(stage_dir)
        return 0

    return 1  # pragma: no cover -- argparse `required=True` makes this unreachable


if __name__ == "__main__":
    sys.exit(main())

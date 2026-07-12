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
FALSE: `bench_compare_1k.py` requires each engine to receive a DIFFERENT
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

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

    `measured_prompt`, when given, overrides `profile.prompt` for THIS
    engine's measured calls only -- mirroring `warmup_prompt`'s override
    of the warmup prompt. `bench_compare_1k.py` needs this: Lattice and
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
        # replayed window-major or repeat-major per group.measured_order,
        # using this engine's own measured prompt (default: profile.prompt)
        # -- e.g. bench_compare_1k.py's Lattice/MLX tokenizer-padded prompt
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


# ==========================================================================
# Schema v2: phase events, resource samples, provenance, lock receipts,
# cell records, expected-cell registry, and the fail-closed run-record
# validator (bench-overhaul PR 1, "canonical contract and policy, in
# days" -- DESIGN.md sections 2 and 4). ADDITIVE: the v1 per-call
# `Observation`/JSONL contract above is UNCHANGED and stays fully
# readable -- this section defines separate, new record kinds under their
# own `RUN_RECORD_SCHEMA_VERSION`, not a breaking change to
# `SCHEMA_VERSION`/`OBSERVATION_FIELDS`. Real lattice adapters, the
# `proc_pid_rusage` resource sampler, and CI wiring land in later PRs (see
# DESIGN.md section 5's phased plan); this PR lands the contract, the
# policy, the expected-cell registry, and the validator only -- no
# performance claim yet, and no cell in `bench_expected_cells.toml` is
# executed here.
# ==========================================================================

RUN_RECORD_SCHEMA_VERSION = 2

PHASE_EVENT_NAMES = (
    "load_start",
    "backend_ready",
    "prefill_start",
    "prefill_end",
    "token_available",
)

VERDICTS = ("PASS", "WARN", "FAIL", "INFRA-FAIL", "unsupported")
METRIC_FAMILIES = ("decode", "prefill_ttft", "memory", "embed")
CADENCES = ("hosted_pr_smoke", "protected_metal_pr", "scheduled_trend", "reserved")

DEFAULT_EXPECTED_CELLS_FILE = REPO_ROOT / "scripts" / "bench_expected_cells.toml"


class RunRecordValidationError(ValueError):
    """A schema-v2 run/cell/promotion record does not conform to the
    contract, or a fail-closed gating precondition (registered cell
    present, path proof, lock receipt, finite metric, sufficient n,
    unchanged policy, matching SHA) was not met. Every raise site here is
    one of DESIGN.md's named `INFRA-FAIL` reasons -- see docstrings below
    for which."""


class ExpectedCellConfigError(ValueError):
    """`bench_expected_cells.toml` is malformed."""


# --------------------------------------------------------------------------
# Phase events, resource samples, provenance, lock receipts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseEvent:
    """One monotonic marker in the load -> prefill -> decode boundary
    (DESIGN.md section 2 "Measurement boundary"). `token_index` is
    required (>= 1) iff `name == "token_available"` (per-token
    availability) and must be `None` for every other phase name."""

    name: str
    monotonic_ns: int
    token_index: int | None = None


@dataclass(frozen=True)
class ResourceSample:
    """One `proc_pid_rusage`(`ri_phys_footprint`) sample. `phase`, when
    given, ties the sample to the `PhaseEvent.name` it was taken during
    (for phase-tagged peak memory); `None` means an ambient/interval
    sample not tied to a specific phase. The sampler itself (10ms
    interval macOS supervisor) is PR 2 -- this PR lands only the record
    shape it must emit into."""

    monotonic_ns: int
    phys_footprint_bytes: int
    phase: str | None = None


@dataclass(frozen=True)
class LockReceipt:
    """Proof that a Metal/heavy-lane advisory lock was held for the
    observation it accompanies (DESIGN.md: "no Metal observation exists
    without both lock receipts", heavy-lane then GPU order). `wait_seconds`
    is the acquisition wait, not the hold duration."""

    lock_name: str
    acquired_at: str
    released_at: str | None
    held_continuously: bool
    wait_seconds: float


@dataclass(frozen=True)
class ProvenanceRecord:
    """Per DESIGN.md section 3 "Hosted validation and aggregation" /
    section 4 "Baseline freshness and provenance": the identity binding a
    cell record to the exact code, config, and machine that produced it.
    `policy_sha`/`profile_sha` are `bench_gate_math.policy_sha`-style
    SHA-256 hex digests of the EXACT policy/profile file bytes the gate
    was evaluated against -- `validate_run_record`'s post-run-threshold-
    change check recomputes these and rejects on mismatch."""

    repo_sha: str
    candidate_sha: str
    base_sha: str | None
    dirty: bool
    profile_name: str
    profile_version: int
    profile_sha: str
    policy_version: int
    policy_sha: str
    script_sha: str
    hardware_fingerprint: str
    collected_at: str
    workflow_run_id: str | None = None


# --------------------------------------------------------------------------
# Cell aggregate + cell record (the north-star queryable shape)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellAggregate:
    """The statistical summary DESIGN.md section 4 requires per cell:
    "the point estimate, two-sided 95% CI for diagnosis, one-sided
    corrected gate bound, n_valid, n_invalid, order balance, and raw-
    artifact digest" (order balance / raw-artifact digest live on
    `CellRecord`, not here). `measured_cv` and `required_n` are
    correction 1's fields: a cell with `measured_cv is None` can never be
    promoted (see `validate_run_record`'s low-valid-n / missing-CV
    check) -- `required_n` is what `bench_gate_math.required_n` returned
    for that measured CV at record time, so a later re-derivation can be
    cross-checked without re-running `bench_gate_math`."""

    point_estimate: float
    ci_low: float
    ci_high: float
    corrected_lower_bound: float | None
    n_valid: int
    n_invalid: int
    measured_cv: float | None
    required_n: int | None


@dataclass(frozen=True)
class CellRecord:
    """One row of the north-star queryable shape (program design
    requirement): "largest stable gaps by metric/context/quant"
    must be queryable WITHOUT a later schema change. `cell_id`,
    `metric_family`, `context_point`, `quant_tier`, and `device` are the
    stable grouping/ranking keys; `aggregate` carries the point
    estimate + CI a ranking query sorts on. See `rank_cells_by_gap` for
    the deterministic proof this shape actually supports that query."""

    cell_id: str
    metric_family: str
    metric_name: str
    path: str
    model_tier: str
    quant_tier: str
    device: str
    context_point: int | None
    aggregate: CellAggregate
    verdict: str
    unsupported_reason: str | None
    path_proof: tuple[str, ...]
    lock_receipts: tuple[LockReceipt, ...]
    phase_events: tuple[PhaseEvent, ...]
    resource_samples: tuple[ResourceSample, ...]
    provenance: ProvenanceRecord
    order_balance: tuple[int, int] = (0, 0)  # (n_ab, n_ba)
    raw_artifact_digest: str | None = None


@dataclass(frozen=True)
class PromotionRecord:
    """Correction 3's record shape: reports the ACHIEVED
    Clopper-Pearson bound from (`null_sessions`, `failures`), never an
    asserted bound tighter than that math allows. `mainline_sessions`
    must clear the policy's `min_mainline_sessions` (a shadow cell needs
    live mainline evidence, not just null A/A calibration) and
    `invalid_pair_replacements` must not exceed the policy's registered
    `max_invalid_pair_replacements` cap: an earlier revision accepted a
    0-mainline-session promotion and had no replacement cap at all. See
    `validate_promotion_record`."""

    cell_id: str
    policy_version: int
    null_sessions: int
    failures: int
    cp_bound_95: float
    mainline_sessions: int
    invalid_pair_replacements: int


# --------------------------------------------------------------------------
# Dict <-> dataclass parsing helpers (fail-closed, matching the v1
# OBSERVATION_FIELDS convention: unknown fields, missing fields, wrong
# types are all rejected -- no schema library, stdlib only)
# --------------------------------------------------------------------------


def _require_keys(d: dict, required: frozenset, allowed: frozenset, ctx: str) -> None:
    if not isinstance(d, dict):
        raise RunRecordValidationError(f"{ctx} must be an object, got {type(d).__name__}")
    extra = set(d) - allowed
    if extra:
        raise RunRecordValidationError(f"{ctx}: unexpected field(s): {sorted(extra)}")
    missing = required - set(d)
    if missing:
        raise RunRecordValidationError(f"{ctx}: missing required field(s): {sorted(missing)}")


_PHASE_EVENT_KEYS = frozenset({"name", "monotonic_ns", "token_index"})
_PHASE_EVENT_REQUIRED = frozenset({"name", "monotonic_ns"})


def parse_phase_event(d: dict) -> PhaseEvent:
    _require_keys(d, _PHASE_EVENT_REQUIRED, _PHASE_EVENT_KEYS, "phase_event")
    name = d["name"]
    if name not in PHASE_EVENT_NAMES:
        raise RunRecordValidationError(f"phase_event: name {name!r} not in {PHASE_EVENT_NAMES}")
    monotonic_ns = d["monotonic_ns"]
    if isinstance(monotonic_ns, bool) or not isinstance(monotonic_ns, int) or monotonic_ns < 0:
        raise RunRecordValidationError("phase_event: monotonic_ns must be a non-negative int")
    token_index = d.get("token_index")
    if name == "token_available":
        if isinstance(token_index, bool) or not isinstance(token_index, int) or token_index < 1:
            raise RunRecordValidationError(
                "phase_event: token_index is required (>= 1) when name == 'token_available'"
            )
    elif token_index is not None:
        raise RunRecordValidationError(
            f"phase_event: token_index must be null when name={name!r} (only 'token_available' carries one)"
        )
    return PhaseEvent(name=name, monotonic_ns=monotonic_ns, token_index=token_index)


_RESOURCE_SAMPLE_KEYS = frozenset({"monotonic_ns", "phys_footprint_bytes", "phase"})
_RESOURCE_SAMPLE_REQUIRED = frozenset({"monotonic_ns", "phys_footprint_bytes"})


def parse_resource_sample(d: dict) -> ResourceSample:
    _require_keys(d, _RESOURCE_SAMPLE_REQUIRED, _RESOURCE_SAMPLE_KEYS, "resource_sample")
    monotonic_ns = d["monotonic_ns"]
    if isinstance(monotonic_ns, bool) or not isinstance(monotonic_ns, int) or monotonic_ns < 0:
        raise RunRecordValidationError("resource_sample: monotonic_ns must be a non-negative int")
    footprint = d["phys_footprint_bytes"]
    if isinstance(footprint, bool) or not isinstance(footprint, int) or footprint < 0:
        raise RunRecordValidationError("resource_sample: phys_footprint_bytes must be a non-negative int")
    phase = d.get("phase")
    if phase is not None and phase not in PHASE_EVENT_NAMES:
        raise RunRecordValidationError(f"resource_sample: phase {phase!r} not in {PHASE_EVENT_NAMES}")
    return ResourceSample(monotonic_ns=monotonic_ns, phys_footprint_bytes=footprint, phase=phase)


_LOCK_RECEIPT_KEYS = frozenset({"lock_name", "acquired_at", "released_at", "held_continuously", "wait_seconds"})


def parse_lock_receipt(d: dict) -> LockReceipt:
    _require_keys(d, _LOCK_RECEIPT_KEYS, _LOCK_RECEIPT_KEYS, "lock_receipt")
    lock_name = d["lock_name"]
    if not isinstance(lock_name, str) or not lock_name:
        raise RunRecordValidationError("lock_receipt: lock_name must be a non-empty string")
    acquired_at = d["acquired_at"]
    if not isinstance(acquired_at, str) or not acquired_at:
        raise RunRecordValidationError("lock_receipt: acquired_at must be a non-empty ISO 8601 string")
    released_at = d["released_at"]
    if released_at is not None and (not isinstance(released_at, str) or not released_at):
        raise RunRecordValidationError("lock_receipt: released_at must be a non-empty string or null")
    held = d["held_continuously"]
    if not isinstance(held, bool):
        raise RunRecordValidationError("lock_receipt: held_continuously must be a bool")
    wait_seconds = d["wait_seconds"]
    if isinstance(wait_seconds, bool) or not isinstance(wait_seconds, (int, float)) or wait_seconds < 0:
        raise RunRecordValidationError("lock_receipt: wait_seconds must be a non-negative number")
    return LockReceipt(
        lock_name=lock_name,
        acquired_at=acquired_at,
        released_at=released_at,
        held_continuously=held,
        wait_seconds=float(wait_seconds),
    )


_PROVENANCE_KEYS = frozenset(
    {
        "repo_sha",
        "candidate_sha",
        "base_sha",
        "dirty",
        "profile_name",
        "profile_version",
        "profile_sha",
        "policy_version",
        "policy_sha",
        "script_sha",
        "hardware_fingerprint",
        "collected_at",
        "workflow_run_id",
    }
)
_PROVENANCE_REQUIRED = _PROVENANCE_KEYS - frozenset({"workflow_run_id", "base_sha"})


def parse_provenance(d: dict) -> ProvenanceRecord:
    _require_keys(d, _PROVENANCE_REQUIRED, _PROVENANCE_KEYS, "provenance")
    for key in ("repo_sha", "candidate_sha", "profile_name", "profile_sha", "policy_sha", "script_sha",
                "hardware_fingerprint", "collected_at"):
        val = d[key]
        if not isinstance(val, str) or not val:
            raise RunRecordValidationError(f"provenance: {key} must be a non-empty string")
    base_sha = d.get("base_sha")
    if base_sha is not None and (not isinstance(base_sha, str) or not base_sha):
        raise RunRecordValidationError("provenance: base_sha must be a non-empty string or null")
    dirty = d["dirty"]
    if not isinstance(dirty, bool):
        raise RunRecordValidationError("provenance: dirty must be a bool")
    for key in ("profile_version", "policy_version"):
        val = d[key]
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise RunRecordValidationError(f"provenance: {key} must be a positive int")
    workflow_run_id = d.get("workflow_run_id")
    if workflow_run_id is not None and (not isinstance(workflow_run_id, str) or not workflow_run_id):
        raise RunRecordValidationError("provenance: workflow_run_id must be a non-empty string or null")
    return ProvenanceRecord(
        repo_sha=d["repo_sha"],
        candidate_sha=d["candidate_sha"],
        base_sha=base_sha,
        dirty=dirty,
        profile_name=d["profile_name"],
        profile_version=d["profile_version"],
        profile_sha=d["profile_sha"],
        policy_version=d["policy_version"],
        policy_sha=d["policy_sha"],
        script_sha=d["script_sha"],
        hardware_fingerprint=d["hardware_fingerprint"],
        collected_at=d["collected_at"],
        workflow_run_id=workflow_run_id,
    )


_CELL_AGGREGATE_KEYS = frozenset(
    {"point_estimate", "ci_low", "ci_high", "corrected_lower_bound", "n_valid", "n_invalid",
     "measured_cv", "required_n"}
)
_CELL_AGGREGATE_REQUIRED = _CELL_AGGREGATE_KEYS - frozenset({"corrected_lower_bound", "measured_cv", "required_n"})


def parse_cell_aggregate(d: dict) -> CellAggregate:
    _require_keys(d, _CELL_AGGREGATE_REQUIRED, _CELL_AGGREGATE_KEYS, "cell_aggregate")
    for key in ("point_estimate", "ci_low", "ci_high"):
        val = d[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise RunRecordValidationError(f"cell_aggregate: {key} must be a number")
        if not math.isfinite(float(val)):
            raise RunRecordValidationError(
                f"cell_aggregate: {key}={val!r} is not finite -- non-finite metrics fail closed (INFRA-FAIL)"
            )
    corrected = d.get("corrected_lower_bound")
    if corrected is not None:
        if isinstance(corrected, bool) or not isinstance(corrected, (int, float)):
            raise RunRecordValidationError("cell_aggregate: corrected_lower_bound must be a number or null")
        if not math.isfinite(float(corrected)):
            raise RunRecordValidationError(
                f"cell_aggregate: corrected_lower_bound={corrected!r} is not finite -- fails closed"
            )
    for key in ("n_valid", "n_invalid"):
        val = d[key]
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise RunRecordValidationError(f"cell_aggregate: {key} must be a non-negative int")
    measured_cv = d.get("measured_cv")
    if measured_cv is not None:
        if isinstance(measured_cv, bool) or not isinstance(measured_cv, (int, float)) or measured_cv < 0:
            raise RunRecordValidationError("cell_aggregate: measured_cv must be a non-negative number or null")
    required_n_val = d.get("required_n")
    if required_n_val is not None:
        if isinstance(required_n_val, bool) or not isinstance(required_n_val, int) or required_n_val < 1:
            raise RunRecordValidationError("cell_aggregate: required_n must be a positive int or null")
    return CellAggregate(
        point_estimate=float(d["point_estimate"]),
        ci_low=float(d["ci_low"]),
        ci_high=float(d["ci_high"]),
        corrected_lower_bound=float(corrected) if corrected is not None else None,
        n_valid=d["n_valid"],
        n_invalid=d["n_invalid"],
        measured_cv=float(measured_cv) if measured_cv is not None else None,
        required_n=required_n_val,
    )


_CELL_RECORD_KEYS = frozenset(
    {
        "cell_id", "metric_family", "metric_name", "path", "model_tier", "quant_tier", "device",
        "context_point", "aggregate", "verdict", "unsupported_reason", "path_proof", "lock_receipts",
        "phase_events", "resource_samples", "provenance", "order_balance", "raw_artifact_digest",
    }
)
_CELL_RECORD_REQUIRED = _CELL_RECORD_KEYS - frozenset(
    {"context_point", "unsupported_reason", "order_balance", "raw_artifact_digest"}
)


def parse_cell_record(d: dict) -> CellRecord:
    _require_keys(d, _CELL_RECORD_REQUIRED, _CELL_RECORD_KEYS, "cell_record")
    for key in ("cell_id", "metric_name", "path", "model_tier", "quant_tier", "device"):
        val = d[key]
        if not isinstance(val, str) or not val:
            raise RunRecordValidationError(f"cell_record: {key} must be a non-empty string")
    metric_family = d["metric_family"]
    if metric_family not in METRIC_FAMILIES:
        raise RunRecordValidationError(f"cell_record: metric_family {metric_family!r} not in {METRIC_FAMILIES}")
    context_point = d.get("context_point")
    if context_point is not None:
        if isinstance(context_point, bool) or not isinstance(context_point, int) or context_point < 1:
            raise RunRecordValidationError("cell_record: context_point must be a positive int or null")
    verdict = d["verdict"]
    if verdict not in VERDICTS:
        raise RunRecordValidationError(f"cell_record: verdict {verdict!r} not in {VERDICTS}")
    unsupported_reason = d.get("unsupported_reason")
    if verdict == "unsupported":
        if not isinstance(unsupported_reason, str) or not unsupported_reason:
            raise RunRecordValidationError(
                "cell_record: unsupported_reason is required (non-empty) when verdict == 'unsupported' "
                "-- an unsupported cell must be NAMED, never silently absent"
            )
    elif unsupported_reason is not None:
        raise RunRecordValidationError("cell_record: unsupported_reason must be null unless verdict == 'unsupported'")
    path_proof = d["path_proof"]
    if not isinstance(path_proof, list) or not all(isinstance(p, str) and p for p in path_proof):
        raise RunRecordValidationError("cell_record: path_proof must be a list of non-empty strings")
    lock_receipts = [parse_lock_receipt(r) for r in _require_list(d["lock_receipts"], "cell_record.lock_receipts")]
    phase_events = [parse_phase_event(r) for r in _require_list(d["phase_events"], "cell_record.phase_events")]
    resource_samples = [
        parse_resource_sample(r) for r in _require_list(d["resource_samples"], "cell_record.resource_samples")
    ]
    provenance = parse_provenance(d["provenance"])
    aggregate = parse_cell_aggregate(d["aggregate"])
    order_balance = d.get("order_balance", [0, 0])
    if (
        not isinstance(order_balance, list)
        or len(order_balance) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in order_balance)
    ):
        raise RunRecordValidationError("cell_record: order_balance must be a [n_ab, n_ba] pair of non-negative ints")
    raw_digest = d.get("raw_artifact_digest")
    if raw_digest is not None and (not isinstance(raw_digest, str) or not raw_digest):
        raise RunRecordValidationError("cell_record: raw_artifact_digest must be a non-empty string or null")
    return CellRecord(
        cell_id=d["cell_id"],
        metric_family=metric_family,
        metric_name=d["metric_name"],
        path=d["path"],
        model_tier=d["model_tier"],
        quant_tier=d["quant_tier"],
        device=d["device"],
        context_point=context_point,
        aggregate=aggregate,
        verdict=verdict,
        unsupported_reason=unsupported_reason,
        path_proof=tuple(path_proof),
        lock_receipts=tuple(lock_receipts),
        phase_events=tuple(phase_events),
        resource_samples=tuple(resource_samples),
        provenance=provenance,
        order_balance=(order_balance[0], order_balance[1]),
        raw_artifact_digest=raw_digest,
    )


def _require_list(value: object, ctx: str) -> list:
    if not isinstance(value, list):
        raise RunRecordValidationError(f"{ctx} must be a list")
    return value


_PROMOTION_RECORD_KEYS = frozenset(
    {
        "cell_id", "policy_version", "null_sessions", "failures", "cp_bound_95", "mainline_sessions",
        "invalid_pair_replacements",
    }
)


def parse_promotion_record(d: dict) -> PromotionRecord:
    _require_keys(d, _PROMOTION_RECORD_KEYS, _PROMOTION_RECORD_KEYS, "promotion_record")
    cell_id = d["cell_id"]
    if not isinstance(cell_id, str) or not cell_id:
        raise RunRecordValidationError("promotion_record: cell_id must be a non-empty string")
    for key in ("policy_version", "null_sessions", "failures", "mainline_sessions", "invalid_pair_replacements"):
        val = d[key]
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise RunRecordValidationError(f"promotion_record: {key} must be a non-negative int")
    if d["policy_version"] < 1:
        raise RunRecordValidationError("promotion_record: policy_version must be >= 1")
    if d["failures"] > d["null_sessions"]:
        raise RunRecordValidationError("promotion_record: failures cannot exceed null_sessions")
    cp_bound = d["cp_bound_95"]
    if isinstance(cp_bound, bool) or not isinstance(cp_bound, (int, float)) or not (0.0 <= cp_bound <= 1.0):
        raise RunRecordValidationError("promotion_record: cp_bound_95 must be a number in [0, 1]")
    return PromotionRecord(
        cell_id=cell_id,
        policy_version=d["policy_version"],
        null_sessions=d["null_sessions"],
        failures=d["failures"],
        cp_bound_95=float(cp_bound),
        mainline_sessions=d["mainline_sessions"],
        invalid_pair_replacements=d["invalid_pair_replacements"],
    )


# --------------------------------------------------------------------------
# Expected-cell registry (bench_expected_cells.toml)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedCellGroup:
    name: str
    path: str
    metric_family: str
    device: str
    model_tiers: tuple[str, ...]
    quant_tiers: tuple[str, ...]
    context_points: tuple[int, ...]
    required_in: tuple[str, ...]
    anchor_quant_tiers: tuple[str, ...]
    anchor_context_points: tuple[int, ...]
    anchor_required_in: tuple[str, ...]


_EXPECTED_GROUP_KEYS = frozenset(
    {
        "name", "path", "metric_family", "device", "model_tiers", "quant_tiers", "context_points",
        "required_in", "anchor_quant_tiers", "anchor_context_points", "anchor_required_in",
    }
)


def load_expected_cells(path: Path = DEFAULT_EXPECTED_CELLS_FILE) -> list[ExpectedCellGroup]:
    try:
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ExpectedCellConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError as exc:
        raise ExpectedCellConfigError(f"{path}: could not read expected-cells file: {exc}") from exc

    if doc.get("schema_version") != 1:
        raise ExpectedCellConfigError(f"{path}: schema_version must be 1, got {doc.get('schema_version')!r}")
    raw_groups = doc.get("group", [])
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ExpectedCellConfigError(f"{path}: [[group]] must define at least one group")

    groups: list[ExpectedCellGroup] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_groups):
        if not isinstance(raw, dict):
            raise ExpectedCellConfigError(f"{path}: group[{i}] must be a table")
        extra = set(raw) - _EXPECTED_GROUP_KEYS
        if extra:
            raise ExpectedCellConfigError(f"{path}: group[{i}] unexpected key(s): {sorted(extra)}")
        missing = _EXPECTED_GROUP_KEYS - set(raw)
        if missing:
            raise ExpectedCellConfigError(f"{path}: group[{i}] missing key(s): {sorted(missing)}")
        name = raw["name"]
        if not isinstance(name, str) or not name:
            raise ExpectedCellConfigError(f"{path}: group[{i}] name must be a non-empty string")
        if name in seen:
            raise ExpectedCellConfigError(f"{path}: duplicate group name {name!r}")
        seen.add(name)
        metric_family = raw["metric_family"]
        if metric_family not in METRIC_FAMILIES:
            raise ExpectedCellConfigError(
                f"{path}: group[{i}] metric_family {metric_family!r} not in {METRIC_FAMILIES}"
            )

        def _str_tuple(key: str) -> tuple[str, ...]:
            val = raw[key]
            if not isinstance(val, list) or not all(isinstance(v, str) and v for v in val):
                raise ExpectedCellConfigError(f"{path}: group[{i}] {key} must be a list of non-empty strings")
            return tuple(val)

        def _int_tuple(key: str) -> tuple[int, ...]:
            val = raw[key]
            if not isinstance(val, list) or not all(
                isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in val
            ):
                raise ExpectedCellConfigError(f"{path}: group[{i}] {key} must be a list of positive ints")
            return tuple(val)

        model_tiers = _str_tuple("model_tiers")
        quant_tiers = _str_tuple("quant_tiers")
        context_points = _int_tuple("context_points")
        required_in = _str_tuple("required_in")
        anchor_quant_tiers = _str_tuple("anchor_quant_tiers")
        anchor_context_points = _int_tuple("anchor_context_points")
        anchor_required_in = _str_tuple("anchor_required_in")
        for cadence in (*required_in, *anchor_required_in):
            if cadence not in CADENCES:
                raise ExpectedCellConfigError(f"{path}: group[{i}] cadence {cadence!r} not in {CADENCES}")
        if not model_tiers or not quant_tiers or not context_points:
            raise ExpectedCellConfigError(
                f"{path}: group[{i}] model_tiers/quant_tiers/context_points must each be non-empty"
            )
        stray_anchor_quants = set(anchor_quant_tiers) - set(quant_tiers)
        if stray_anchor_quants:
            raise ExpectedCellConfigError(
                f"{path}: group[{i}] anchor_quant_tiers {sorted(stray_anchor_quants)} not present in quant_tiers"
            )
        stray_anchor_ctx = set(anchor_context_points) - set(context_points)
        if stray_anchor_ctx:
            raise ExpectedCellConfigError(
                f"{path}: group[{i}] anchor_context_points {sorted(stray_anchor_ctx)} not present in context_points"
            )
        groups.append(
            ExpectedCellGroup(
                name=name,
                path=raw["path"],
                metric_family=metric_family,
                device=raw["device"],
                model_tiers=model_tiers,
                quant_tiers=quant_tiers,
                context_points=context_points,
                required_in=required_in,
                anchor_quant_tiers=anchor_quant_tiers,
                anchor_context_points=anchor_context_points,
                anchor_required_in=anchor_required_in,
            )
        )
    return groups


def cell_id(path: str, model_tier: str, quant_tier: str, device: str, context_point: int) -> str:
    return f"{path}:{model_tier}:{quant_tier}:{device}:{context_point}"


def expand_cell_group(group: ExpectedCellGroup) -> dict[str, tuple[str, ...]]:
    """Expand one registry group into `{cell_id: required_in}` over its
    FULL axis product. Anchor cells (the DESIGN.md "small anchor in
    hosted PR A/B" subset) get their `anchor_required_in` cadences
    UNIONED onto the group's own `required_in` -- an anchor cell is
    required at both its own cadence and the group's general cadence."""
    out: dict[str, tuple[str, ...]] = {}
    for model_tier in group.model_tiers:
        for quant_tier in group.quant_tiers:
            for context_point in group.context_points:
                cid = cell_id(group.path, model_tier, quant_tier, group.device, context_point)
                cadences = set(group.required_in)
                if quant_tier in group.anchor_quant_tiers and context_point in group.anchor_context_points:
                    cadences |= set(group.anchor_required_in)
                out[cid] = tuple(sorted(cadences))
    return out


def expected_cell_ids_for_cadence(groups: Sequence[ExpectedCellGroup], cadence: str) -> set[str]:
    """Every `cell_id` across every group whose expansion includes
    `cadence` (e.g. `"hosted_pr_smoke"`) -- the set a run's `CellRecord`s
    must cover (each present, `PASS`/`WARN`/`FAIL`, or explicitly
    `unsupported`) for that cadence to pass coverage."""
    ids: set[str] = set()
    for group in groups:
        for cid, cadences in expand_cell_group(group).items():
            if cadence in cadences:
                ids.add(cid)
    return ids



def validate_registry_coverage(records: Sequence[CellRecord], groups: Sequence[ExpectedCellGroup], cadence: str) -> None:
    """Fail-closed coverage check: every `cell_id` the registry expects at
    `cadence` must appear in `records` (as any verdict, including
    `unsupported` -- DESIGN.md: a matrix cell may be `unsupported` but
    "never silently skipped"). Missing cell(s) -> `RunRecordValidationError`
    (`INFRA-FAIL`: "expected cell absent")."""
    expected = expected_cell_ids_for_cadence(groups, cadence)
    present = {r.cell_id for r in records}
    missing = sorted(expected - present)
    if missing:
        raise RunRecordValidationError(
            f"INFRA-FAIL: expected cell(s) absent for cadence {cadence!r}: {missing} "
            "-- an expected cell that did not execute is infrastructure failure, never a silent skip"
        )


# --------------------------------------------------------------------------
# Fail-closed run-record validator
# --------------------------------------------------------------------------

_REQUIRED_SINGLE_PHASES = ("load_start", "backend_ready", "prefill_start", "prefill_end")
_PHASE_NAME_RANK = {name: i for i, name in enumerate(PHASE_EVENT_NAMES)}


def _validate_phase_sequence(phase_events: Sequence[PhaseEvent], cell_id: str) -> None:
    """Validate the load->prefill->decode measurement-boundary sequence:
    a non-unsupported cell must carry a phase-event sequence that (a) is
    non-empty, (b)
    contains exactly one each of `load_start`, `backend_ready`,
    `prefill_start`, `prefill_end`, (c) contains at least one
    `token_available` event, (d) never runs `monotonic_ns` backwards, and
    (e) never regresses phase order (each single-shot phase's rank in
    `PHASE_EVENT_NAMES` must be non-decreasing across the sequence, and
    `token_available` events must carry strictly increasing
    `token_index`). This is the measurement-boundary proof the contract
    promises -- an empty or misordered sequence proves nothing."""
    if not phase_events:
        raise RunRecordValidationError(
            f"INFRA-FAIL: cell {cell_id!r} has no phase_events -- the load->prefill->decode "
            "measurement boundary must be proven, never assumed"
        )
    seen_single: set[str] = set()
    token_seen = False
    last_rank = -1
    last_ns = -1
    last_token_index = -1
    for ev in phase_events:
        rank = _PHASE_NAME_RANK[ev.name]
        if rank < last_rank:
            raise RunRecordValidationError(
                f"INFRA-FAIL: cell {cell_id!r} phase_events out of order at {ev.name!r} -- the "
                "load->prefill->decode sequence must never regress"
            )
        if ev.monotonic_ns < last_ns:
            raise RunRecordValidationError(
                f"INFRA-FAIL: cell {cell_id!r} phase_events monotonic_ns went backwards at {ev.name!r}"
            )
        if ev.name == "token_available":
            token_index = ev.token_index if ev.token_index is not None else -1
            if token_seen and token_index <= last_token_index:
                raise RunRecordValidationError(
                    f"INFRA-FAIL: cell {cell_id!r} phase_events token_available token_index "
                    "did not strictly increase"
                )
            last_token_index = token_index
            token_seen = True
        else:
            if ev.name in seen_single:
                raise RunRecordValidationError(
                    f"INFRA-FAIL: cell {cell_id!r} phase_events: phase {ev.name!r} appears more than once"
                )
            seen_single.add(ev.name)
        last_rank = rank
        last_ns = ev.monotonic_ns
    missing = [n for n in _REQUIRED_SINGLE_PHASES if n not in seen_single]
    if missing:
        raise RunRecordValidationError(
            f"INFRA-FAIL: cell {cell_id!r} phase_events missing required phase(s) {missing} -- the "
            "measurement boundary must be proven with the full load->prefill->decode sequence"
        )
    if not token_seen:
        raise RunRecordValidationError(
            f"INFRA-FAIL: cell {cell_id!r} phase_events has no token_available marker -- the decode "
            "measurement boundary is unproven without at least one"
        )


def validate_run_record(
    record: CellRecord,
    *,
    expected_repo_sha: str | None = None,
    current_policy_sha: str | None = None,
    policy: dict | None = None,
) -> None:
    """Fail-closed validation of ONE `CellRecord` against DESIGN.md's
    `INFRA-FAIL` taxonomy (section 4 "What blocks a PR"). Raises
    `RunRecordValidationError` naming the specific reason; returns `None`
    on success. Registry coverage (a MISSING cell) is a separate check
    (`validate_registry_coverage`) since it operates over a whole run,
    not one record. `policy` defaults to `bench_gate_math.load_policy()`
    (the shipped `perf-policy.toml`) when not supplied.

    Checks, each one a named `INFRA-FAIL` reason from DESIGN.md:
      - missing path proof: a `device == "metal"` cell must carry >= 1
        non-empty `path_proof` marker.
      - missing lock receipts: a `device == "metal"` cell must carry BOTH
        a `"metal-gpu"` and a `"heavy-lane"` `LockReceipt`, each
        `held_continuously` -- no Metal observation exists without both
        the outer heavy-lane and the GPU receipt.
      - missing/empty resource samples: a `device == "metal"` cell must
        carry >= 1 `ResourceSample` -- the contention/memory-footprint
        proof the Metal lane's receipts promise.
      - missing or misordered phase-event sequence: every non-unsupported
        cell must carry a validated `load_start -> backend_ready ->
        prefill_start -> prefill_end -> token_available(+)` sequence (see
        `_validate_phase_sequence`) -- the measurement boundary must be
        proven, not assumed.
      - non-finite metric: enforced already at `parse_cell_aggregate`
        (kept here as a defense-in-depth re-check for records built by
        hand rather than parsed from JSON).
      - low valid-n / missing measured-CV / submitter-controlled
        required_n (correction 1): a `PASS`/`WARN`/`FAIL` cell (not
        `unsupported`) must carry `measured_cv`; the cell's noise class is
        resolved from the REGISTERED per-metric policy
        (`bench_gate_math.resolve_metric_policy`), never a family-name
        heuristic, and its `required_n` is RE-DERIVED from that class and
        the measured CV via `bench_gate_math.required_n` -- a record whose
        `required_n` field is missing or does not match the re-derived
        value is rejected outright (required_n is policy-derived, never
        submitter-controlled), and so is `n_valid < required_n`. Class "C"
        (informational/trend-only) metrics are exempt from this
        derivation -- they never gate a required cell.
      - post-run threshold change: if `current_policy_sha` is given, it
        must equal `record.provenance.policy_sha` -- the gate must have
        been evaluated against the policy content it claims.
      - wrong SHA: if `expected_repo_sha` is given, it must equal
        `record.provenance.candidate_sha`.
    """
    if record.device == "metal":
        if not record.path_proof:
            raise RunRecordValidationError(
                f"INFRA-FAIL: cell {record.cell_id!r} is device=metal but carries no path_proof marker"
            )
        metal_locks = [
            lr for lr in record.lock_receipts if lr.lock_name == "metal-gpu" and lr.held_continuously
        ]
        if not metal_locks:
            raise RunRecordValidationError(
                f"INFRA-FAIL: cell {record.cell_id!r} is device=metal but carries no continuously-held "
                "'metal-gpu' lock_receipt -- no Metal observation exists without a lock receipt"
            )
        heavy_lane_locks = [
            lr for lr in record.lock_receipts if lr.lock_name == "heavy-lane" and lr.held_continuously
        ]
        if not heavy_lane_locks:
            raise RunRecordValidationError(
                f"INFRA-FAIL: cell {record.cell_id!r} is device=metal but carries no continuously-held "
                "'heavy-lane' lock_receipt -- Metal contention proof requires BOTH the outer "
                "heavy-lane and the metal-gpu receipt, held continuously"
            )
        if not record.resource_samples:
            raise RunRecordValidationError(
                f"INFRA-FAIL: cell {record.cell_id!r} is device=metal but carries no resource_samples "
                "-- the Metal contention/memory-footprint proof requires at least one sample"
            )

    if record.verdict != "unsupported":
        _validate_phase_sequence(record.phase_events, record.cell_id)

    agg = record.aggregate
    for label, val in (("point_estimate", agg.point_estimate), ("ci_low", agg.ci_low), ("ci_high", agg.ci_high)):
        if not math.isfinite(val):
            raise RunRecordValidationError(f"INFRA-FAIL: cell {record.cell_id!r} {label}={val!r} is not finite")

    if record.verdict != "unsupported":
        if agg.measured_cv is None:
            raise RunRecordValidationError(
                f"INFRA-FAIL: cell {record.cell_id!r} has no measured_cv on record -- correction 1 refuses "
                "to promote/gate any cell whose required-n cannot be derived from a measured same-session CV"
            )
        policy_doc = policy if policy is not None else bench_gate_math.load_policy()
        try:
            metric_policy = bench_gate_math.resolve_metric_policy(policy_doc, record.metric_family, record.metric_name)
            noise_class = metric_policy["noise_class"]
        except bench_gate_math.PolicyConfigError as exc:
            raise RunRecordValidationError(
                f"INFRA-FAIL: cell {record.cell_id!r}: {exc}"
            ) from exc

        if noise_class in ("A", "B"):
            cv_bands = bench_gate_math.parse_cv_bands(policy_doc["cv_bands"])
            try:
                derived_required_n, _fail_margin = bench_gate_math.required_n(agg.measured_cv, cv_bands, noise_class)
            except bench_gate_math.GateMathError as exc:
                raise RunRecordValidationError(f"INFRA-FAIL: cell {record.cell_id!r}: {exc}") from exc
            if agg.required_n != derived_required_n:
                raise RunRecordValidationError(
                    f"INFRA-FAIL: cell {record.cell_id!r} required_n={agg.required_n!r} does not match "
                    f"the policy-derived required_n={derived_required_n!r} for measured_cv={agg.measured_cv!r} "
                    f"class {noise_class!r} -- required_n is derived from the registered policy, "
                    "never submitter-controlled metadata"
                )
            if agg.n_valid < derived_required_n:
                raise RunRecordValidationError(
                    f"INFRA-FAIL: cell {record.cell_id!r} n_valid={agg.n_valid} < required_n={derived_required_n} "
                    f"(measured_cv={agg.measured_cv}, class {noise_class}) -- n too small"
                )
        # class "C": informational/trend-only, never gates a required cell
        # (perf-policy.toml) -- no required_n derivation applies.

    if current_policy_sha is not None and record.provenance.policy_sha != current_policy_sha:
        raise RunRecordValidationError(
            f"INFRA-FAIL: cell {record.cell_id!r} was gated against policy_sha="
            f"{record.provenance.policy_sha!r} but the current policy_sha is {current_policy_sha!r} "
            "-- post-run threshold change, this record's verdict is no longer valid"
        )

    if expected_repo_sha is not None and record.provenance.candidate_sha != expected_repo_sha:
        raise RunRecordValidationError(
            f"INFRA-FAIL: cell {record.cell_id!r} candidate_sha={record.provenance.candidate_sha!r} "
            f"!= expected {expected_repo_sha!r} -- base/candidate SHA mismatch"
        )


def validate_promotion_record(
    record: PromotionRecord,
    *,
    policy: dict | None = None,
    min_null_sessions: int | None = None,
    min_mainline_sessions: int | None = None,
) -> None:
    """Correction 3, enforced, consuming the VERSIONED policy: an earlier
    revision hard-coded `min_null_sessions=20` and never examined
    `mainline_sessions` or a replacement cap at all:

      - `null_sessions >= policy["promotion"]["min_null_aa_sessions"]`
        (DESIGN.md's ">= 20 same-SHA null A/A sessions").
      - `mainline_sessions >= policy["promotion"]["min_mainline_sessions"]`
        -- a shadow cell needs live mainline evidence, not just null A/A
        calibration, before promotion.
      - `invalid_pair_replacements <= policy["promotion"]["max_invalid_pair_replacements"]`
        -- an unbounded replacement budget would let a cell keep
        re-rolling null sessions until it got a favorable run.
      - the asserted `cp_bound_95` must be >= the TRUE Clopper-Pearson
        upper bound `bench_gate_math.clopper_pearson_upper` computes from
        `(failures, null_sessions)` -- never a tighter, aspirational
        number.

    `min_null_sessions`/`min_mainline_sessions` override the policy value
    when explicitly supplied (mainly for tests); `policy` defaults to
    `bench_gate_math.load_policy()` when not supplied.
    """
    policy_doc = policy if policy is not None else bench_gate_math.load_policy()
    promotion_policy = policy_doc.get("promotion")
    if not isinstance(promotion_policy, dict):
        raise RunRecordValidationError(
            "promotion policy: perf-policy.toml is missing or has a malformed [promotion] table"
        )


    doc_policy_version = policy_doc.get("policy_version")
    if isinstance(doc_policy_version, bool) or not isinstance(doc_policy_version, int):
        raise RunRecordValidationError(
            "promotion policy: perf-policy.toml policy_version must be a registered int -- "
            "a promotion record cannot be bound to an unversioned policy"
        )
    if record.policy_version != doc_policy_version:
        raise RunRecordValidationError(
            f"promotion of cell {record.cell_id!r} declares policy_version={record.policy_version}, "
            f"but the structurally validated policy in force is policy_version={doc_policy_version} -- "
            "a promotion record must be evaluated under the exact policy version it claims to consume, "
            "never a different version's thresholds"
        )

    resolved_min_null = min_null_sessions if min_null_sessions is not None else promotion_policy.get("min_null_aa_sessions")
    resolved_min_mainline = (
        min_mainline_sessions if min_mainline_sessions is not None else promotion_policy.get("min_mainline_sessions")
    )
    replacement_cap = promotion_policy.get("max_invalid_pair_replacements")
    for label, val in (
        ("min_null_aa_sessions", resolved_min_null),
        ("min_mainline_sessions", resolved_min_mainline),
        ("max_invalid_pair_replacements", replacement_cap),
    ):
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise RunRecordValidationError(
                f"promotion policy: {label} must be a registered non-negative int, got {val!r} -- "
                "an incomplete promotion policy fails closed, never defaults silently"
            )

    if record.null_sessions < resolved_min_null:
        raise RunRecordValidationError(
            f"promotion of cell {record.cell_id!r} requires >= {resolved_min_null} null A/A sessions, "
            f"got {record.null_sessions}"
        )
    if record.mainline_sessions < resolved_min_mainline:
        raise RunRecordValidationError(
            f"promotion of cell {record.cell_id!r} requires >= {resolved_min_mainline} mainline sessions, "
            f"got {record.mainline_sessions} -- null A/A calibration alone is not live mainline evidence"
        )
    if record.invalid_pair_replacements > replacement_cap:
        raise RunRecordValidationError(
            f"promotion of cell {record.cell_id!r} reports invalid_pair_replacements="
            f"{record.invalid_pair_replacements}, exceeding the registered cap of {replacement_cap} -- "
            "an exhausted replacement budget is an INFRA-FAIL, never a silent retry"
        )

    true_bound = bench_gate_math.clopper_pearson_upper(record.failures, record.null_sessions)
    # Tolerance accommodates realistic rounding of a REPORTED bound (e.g.
    # "13.91%" rounded to 4 decimal places is 1e-5-scale below the exact
    # value) without opening the door to a materially tighter, aspirational
    # claim -- 1e-4 is generous rounding slack, not a loophole.
    if record.cp_bound_95 < true_bound - 1e-4:
        raise RunRecordValidationError(
            f"promotion of cell {record.cell_id!r} asserts cp_bound_95={record.cp_bound_95!r}, tighter than "
            f"the true Clopper-Pearson bound {true_bound!r} for failures={record.failures}/{record.null_sessions} "
            "-- report the ACHIEVED bound, never an aspirational one"
        )


# --------------------------------------------------------------------------
# North-star ranking query (program design requirement): "largest
# stable gaps by metric/context/quant" must be queryable without a later
# schema change. See tests/test_bench_run_record.py for the deterministic
# proof.
# --------------------------------------------------------------------------


def rank_cells_by_gap(records: Sequence[CellRecord], *, top_n: int | None = None) -> list[CellRecord]:
    """Rank `CellRecord`s by `abs(aggregate.point_estimate)` descending
    (the "largest stable gap" -- `point_estimate` is already a signed
    log-slowdown/percent-delta in the cell's own orientation, so its
    magnitude IS the gap size). Ties broken by `cell_id` for a
    deterministic order. `unsupported` cells are excluded (no gap to
    rank -- they carry no measurement). This is a pure function over the
    same `CellRecord` fields every run/promotion record already carries:
    no additional schema is needed to answer "largest stable gaps by
    metric/context/quant"."""
    ranked = sorted(
        (r for r in records if r.verdict != "unsupported"),
        key=lambda r: (-abs(r.aggregate.point_estimate), r.cell_id),
    )
    return ranked[:top_n] if top_n is not None else ranked


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

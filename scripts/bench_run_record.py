"""Run-record schema v2 of the decode-benchmark harness.

Phase events, resource samples, provenance, lock receipts, cell records, the
expected-cell registry, and the fail-closed run-record and promotion-record
validators. `scripts/bench_decode_harness.py` imports every name defined here
except `REPO_ROOT`, which it sets itself, so callers that reach them through
the harness module are unchanged. The harness puts `scripts/` on `sys.path`
before importing this module, which is how `import bench_gate_math` resolves.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import bench_gate_math

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==========================================================================
# Schema v2: phase events, resource samples, provenance, lock receipts,
# cell records, expected-cell registry, and the fail-closed run-record
# validator (bench-overhaul PR 1, "canonical contract and policy, in
# days" -- DESIGN.md sections 2 and 4). ADDITIVE: the v1 per-call
# `Observation`/JSONL contract in `bench_decode_harness.py` is UNCHANGED and stays fully
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
    `profile_sha` is the SHA-256 digest of the exact profile bytes.
    `policy_sha` is a tagged `bench_gate_math.policy_sha` identity over
    canonical parsed policy values. `policy_file_sha` retains the exact
    policy bytes for audit without making prose part of the gating identity.
    Legacy records may carry an untagged raw-byte `policy_sha` and omit
    `policy_file_sha`; validation never aliases that legacy scheme to the
    canonical scheme."""

    repo_sha: str
    candidate_sha: str
    base_sha: str | None
    dirty: bool
    profile_name: str
    profile_version: int
    profile_sha: str
    policy_version: int
    policy_sha: str
    policy_file_sha: str | None
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
    correction 1's fields: a recorded CV is either a positive finite
    number or `None`, and a cell with `measured_cv is None` can never be
    promoted (see `validate_run_record`'s low-valid-n / missing-CV check).
    `required_n` is what `bench_gate_math.required_n` returned for that
    measured CV at record time, so a later re-derivation can be
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
        "policy_file_sha",
        "script_sha",
        "hardware_fingerprint",
        "collected_at",
        "workflow_run_id",
    }
)
_PROVENANCE_REQUIRED = _PROVENANCE_KEYS - frozenset(
    {"workflow_run_id", "base_sha", "policy_file_sha"}
)


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
    try:
        policy_scheme = bench_gate_math.policy_sha_scheme(d["policy_sha"])
    except bench_gate_math.PolicyConfigError as exc:
        raise RunRecordValidationError(f"provenance: {exc}") from exc
    policy_file_sha = d.get("policy_file_sha")
    if policy_file_sha is not None and (
        not isinstance(policy_file_sha, str)
        or len(policy_file_sha) != 64
        or any(char not in "0123456789abcdef" for char in policy_file_sha)
    ):
        raise RunRecordValidationError(
            "provenance: policy_file_sha must be a 64-character lowercase SHA-256 or null"
        )
    if policy_scheme == bench_gate_math.POLICY_SHA_SCHEME and policy_file_sha is None:
        raise RunRecordValidationError(
            "provenance: canonical policy_sha requires policy_file_sha so exact policy bytes "
            "remain auditable"
        )
    if (
        policy_scheme == bench_gate_math.LEGACY_POLICY_SHA_SCHEME
        and policy_file_sha is not None
        and policy_file_sha != d["policy_sha"]
    ):
        raise RunRecordValidationError(
            "provenance: legacy policy_sha and policy_file_sha must match"
        )
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
        policy_file_sha=policy_file_sha,
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
        if (
            isinstance(measured_cv, bool)
            or not isinstance(measured_cv, (int, float))
            or not math.isfinite(float(measured_cv))
            or measured_cv <= 0
        ):
            raise RunRecordValidationError(
                "cell_aggregate: measured_cv must be a positive finite number or null"
            )
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
      - non-finite metric or non-positive/non-finite measured CV: enforced
        already at `parse_cell_aggregate` (kept here as a defense-in-depth
        re-check for records built by hand rather than parsed from JSON).
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
      - post-run policy change: if `current_policy_sha` is given, it must
        use the same identity scheme and equal
        `record.provenance.policy_sha`. Legacy byte digests are recognized
        but never implicitly mapped to canonical identities.
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
    if agg.measured_cv is not None and (
        isinstance(agg.measured_cv, bool)
        or not isinstance(agg.measured_cv, (int, float))
        or not math.isfinite(float(agg.measured_cv))
        or agg.measured_cv <= 0
    ):
        raise RunRecordValidationError(
            f"INFRA-FAIL: cell {record.cell_id!r} measured_cv={agg.measured_cv!r} "
            "must be a positive finite number"
        )

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

    if current_policy_sha is not None:
        try:
            record_policy_scheme = bench_gate_math.policy_sha_scheme(record.provenance.policy_sha)
            current_policy_scheme = bench_gate_math.policy_sha_scheme(current_policy_sha)
        except bench_gate_math.PolicyConfigError as exc:
            raise RunRecordValidationError(
                f"INFRA-FAIL: cell {record.cell_id!r} carries an invalid policy identity: {exc}"
            ) from exc
        if record_policy_scheme != current_policy_scheme:
            raise RunRecordValidationError(
                f"INFRA-FAIL: cell {record.cell_id!r} policy_sha schemes differ: record uses "
                f"{record_policy_scheme!r}, current policy uses {current_policy_scheme!r}. "
                "A legacy byte-SHA is not comparable to a canonical policy identity and is never "
                "migrated implicitly; use documented transition evidence or rerun the benchmark"
            )
        if record.provenance.policy_sha != current_policy_sha:
            detail = (
                "legacy byte-SHA mismatch; byte identities cannot distinguish prose from gating "
                "changes and no transition is accepted implicitly"
                if record_policy_scheme == bench_gate_math.LEGACY_POLICY_SHA_SCHEME
                else "canonical gating-policy identity changed"
            )
            raise RunRecordValidationError(
                f"INFRA-FAIL: cell {record.cell_id!r} was gated against policy_sha="
                f"{record.provenance.policy_sha!r} but the current policy_sha is {current_policy_sha!r} "
                f"-- {detail}; this record's verdict is no longer valid"
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

//! Resident-adapter metadata, request parsing, and HTTP responses shared by both servers.

use super::ApiError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

/// Default maximum number of resident adapters per server.
pub const DEFAULT_MAX_RESIDENT_ADAPTERS: usize = 32;
/// Default tensor-payload budget per server: 512 MiB, excluding other memory.
pub const DEFAULT_MAX_RESIDENT_ADAPTER_BYTES: usize = 512 * 1024 * 1024;

/// Admission limits for explicitly loaded adapters. Reaching either rejects a load;
/// existing residents are never evicted. This is not a process-memory budget.
#[derive(Debug, Clone, Copy)]
pub struct ResidencyLimits {
    /// Maximum number of distinct client-supplied `(name, path)` identities.
    pub max_adapters: usize,
    /// Maximum sum of A/B tensor payload bytes, excluding strings and struct overhead.
    pub max_bytes: usize,
}

impl Default for ResidencyLimits {
    fn default() -> Self {
        Self {
            max_adapters: DEFAULT_MAX_RESIDENT_ADAPTERS,
            max_bytes: DEFAULT_MAX_RESIDENT_ADAPTER_BYTES,
        }
    }
}

/// Parse a positive resident limit without silently replacing malformed input.
pub fn parse_resident_limit(raw: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|_| "expected a positive integer".to_string())?;
    if value == 0 {
        return Err("expected a positive integer".into());
    }
    Ok(value)
}

/// Failure of a worker-local adapter load or unload command.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AdapterControlError {
    /// Invalid descriptor, tensor payload, or unavailable control implementation.
    #[error("{0}")]
    InvalidAdapter(String),
    /// The identifier is no longer resident.
    #[error("unknown LoRA adapter id {0}")]
    NotFound(u32),
    /// Admitting a new identity would exceed the count limit.
    #[error("resident LoRA adapter count limit reached ({limit})")]
    CountLimit { limit: usize },
    /// Admitting the tensor payload would exceed the byte limit.
    #[error("resident LoRA adapter tensor payload byte limit exceeded ({limit})")]
    ByteLimit { limit: usize },
    /// All identifiers have been used during this worker's lifetime.
    #[error("LoRA adapter id space exhausted")]
    IdExhausted,
}

impl From<String> for AdapterControlError {
    fn from(message: String) -> Self {
        Self::InvalidAdapter(message)
    }
}

impl From<&str> for AdapterControlError {
    fn from(message: &str) -> Self {
        Self::InvalidAdapter(message.to_string())
    }
}

impl From<AdapterControlError> for ApiError {
    fn from(error: AdapterControlError) -> Self {
        Self::BadRequest {
            code: adapter_failure_code(&error),
            message: error.to_string(),
        }
    }
}

/// Confirmed result, including the resident metadata on an idempotent load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterControlResult {
    /// Newly admitted or already resident adapter.
    Loaded(AdapterMetadata),
    /// Explicitly removed identifier.
    Unloaded(u32),
}

/// Assemble the `GET /v1/lora` body: the residency snapshot with `router`
/// added beside it.
///
/// A separate function because the shape is the thing that breaks and a
/// handler's shape can only be checked by launching a server. It already broke
/// once: adding the router key as `json!({"adapters": index, "router": ...})`
/// reads like adding a field and is not. `AdapterIndex` serializes to
/// `{"adapters": [...], "applied": [...]}`, so wrapping it turned the
/// top-level `adapters` from an array into an object and moved `applied` a
/// level down -- a breaking change to two existing fields, written while
/// intending a purely additive one, and invisible without a test that holds
/// the whole body.
///
/// It lives in the shared module rather than in either binary because the two
/// binaries answer the SAME route: a body assembled in one of them is a body
/// the other can drift from, and it did -- `lattice_serve` was still returning
/// the bare residency snapshot after `lattice` gained the `router` key, so one
/// server reported which gate was serving and the other did not. A route-table
/// mechanism cannot see that: both registered the route correctly.
///
/// `router` is therefore merged in beside the snapshot's own keys rather than
/// containing them.
pub fn lora_list_body(
    index: &AdapterIndex,
    router: Option<crate::router_state::RouterReport<'_>>,
) -> serde_json::Value {
    // ADR-095 decision 3: the response says which gate is serving, because
    // "routing is enabled" and "routing ran with the gate I pinned" are
    // different claims.
    //
    // `pinned` is reported rather than left for the reader to infer, because
    // the version alone cannot carry it. A server reporting version 7 reports
    // the same number whether --router-pin selected it or whether 7 is simply
    // the highest version written so far, and the two only diverge at the next
    // refit and restart -- which is when nobody is looking, and is the entire
    // scenario a pin exists for.
    let router = match router {
        None => serde_json::json!({"enabled": false}),
        Some(report) => {
            // "enabled" and "can route right now" are different claims, and
            // the second one moves: adapters load and unload while the server
            // runs, so a gate that was routable a minute ago is not. Routing
            // refuses a set it cannot match, so without this an operator meets
            // that refusal one 400 at a time with nothing on the surface that
            // predicted it.
            //
            // Computed by the SAME predicate routing refuses on, not by a
            // comparison written here. A second copy would drift, and the
            // drift reads as this endpoint promising a route the next request
            // declines.
            let state = crate::serve::routing::routability(report.artifact, index);
            serde_json::json!({
                "enabled": true,
                "version": report.artifact.version_label(),
                "pinned": report.pinned,
                "embedder": report.embedder,
                "adapter_names": report.artifact.adapter_names,
                "routable": state.routable(),
                "missing": state.missing,
                "unexpected": state.unexpected,
                "duplicate_trained": state.duplicate_trained,
                "duplicate_resident": state.duplicate_resident,
                "blend_refusal": state.blend_refusal,
                // Serving a deliberately untrained baseline and serving a
                // learned gate are different claims, and the responses do not
                // distinguish them: a bootstrap routes, reports a version, and
                // looks exactly like a gate that learned something. ADR-096
                // decision 2 puts the difference on this surface rather than
                // leaving it to be inferred from behaviour.
                //
                // A legacy artifact answers "unrecorded", never "bootstrap":
                // it may well have been trained, and nothing wrote down which.
                "training_state": report.artifact.provenance.training_state(),
                // The policy requests are ACTUALLY served under, read from the
                // live router rather than from the artifact beside it. A server
                // that declined to enable a recorded learned policy would
                // otherwise report the recorded one and serve a different one,
                // and an operator watching a qualification run has no other
                // surface that says which of the two is happening.
                "policy": report.applied_policy,
            })
        }
    };
    let mut body = serde_json::to_value(index).unwrap_or_else(|_| serde_json::json!({}));
    if let Some(map) = body.as_object_mut() {
        map.insert("router".into(), router);
    }
    body
}

/// Every `/v1/lora*` route, as data: path and methods, one copy (ADR-095
/// decision 4).
///
/// Data rather than a shared constructor. The two binaries do not share a
/// state type -- one carries a CPU-or-Metal backend, the other is
/// Metal-worker-only with a metrics registry -- so a shared constructor would
/// be generic over the state and take every handler as a parameter, replacing
/// each `.route()` line with a handler argument. That relocates the
/// duplication into a longer call while the handlers and the state, which are
/// what actually drift, stay exactly where they are. A list has no state type
/// to be generic over.
///
/// This list establishes only that a route is REGISTERED. Whether the two
/// binaries' handlers behave the same is a separate question that no
/// route-table mechanism can answer, and it is audited in the ADR rather than
/// asserted here: the failure that motivated this decision was two binaries
/// that both rejected a non-finite adapter scale, one at the HTTP boundary and
/// one later inside `apply()`. Same route, same shared module, different
/// reachable behaviour, both tables correct.
pub const LORA_ROUTES: &[(&str, &[&str])] = &[
    ("/v1/lora", &["GET"]),
    ("/v1/lora/load", &["POST"]),
    ("/v1/lora/unload", &["POST"]),
];

/// The router feedback routes, as data (ADR-096 decision 3).
///
/// A separate list from [`LORA_ROUTES`] because it has a different scope, and
/// the scope is the point: these are registered by `lattice` ONLY. The other
/// serving binary cannot route -- it takes no `--router-state` -- so it
/// records no episodes, and a feedback route there would accept judgements
/// about completions that no gate produced and answer every one of them with
/// `unknown_completion`. A 404 is the honest answer from a binary that does
/// not route.
///
/// Kept as data anyway, and checked by that binary's own presence test, for
/// the reason `LORA_ROUTES` exists: a route table is not a registration, and
/// the drift between the two is invisible to anything that reads only one.
pub const ROUTER_FEEDBACK_ROUTES: &[(&str, &[&str])] = &[
    ("/v1/router/feedback", &["POST"]),
    ("/v1/router/episodes", &["GET"]),
];

/// One contribution to an ordered request mixture.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct LoraSelection {
    /// Resident adapter identifier.
    pub id: u32,
    /// Multiplier on the adapter's alpha/rank scale.
    pub scale: f32,
}

/// What a request asked for in its `lora` field (ADR-094, amended 2026-09-21).
///
/// The wire field is `Option<Vec<LoraSelection>>` so that an absent field and an
/// explicit `[]` stay distinguishable after deserialization. Under a bare `Vec`
/// with `#[serde(default)]` they are the same value, which leaves no way to spell
/// "choose for me" — the same problem `ChatRequest.model` already solves with
/// `Option<String>` for the same stated reason.
///
/// Both serving binaries resolve the field through [`requested_adapters`] rather
/// than reading it themselves. That is deliberate: the two of them handled the
/// raw field differently, and one shared resolver removes the copy that could
/// drift instead of leaving two to keep in step.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum RequestedAdapters {
    /// The field was absent. Route, when routing is enabled; the base model when
    /// it is not — which is what omitting the field does today.
    Unspecified,
    /// The field was an explicit `[]`: the base model, pinned, never routed.
    ///
    /// This case exists so that enabling routing cannot change what an existing
    /// caller gets. A client sending `"lora": []` today to pin the base model
    /// keeps the base model afterwards, rather than silently starting to receive
    /// adapter output because an operator turned a flag on elsewhere.
    PinnedBase,
    /// An explicit non-empty list: exactly this mixture, never routed.
    Explicit(Vec<LoraSelection>),
}

impl RequestedAdapters {
    /// The selection to apply as things stand, before any routing runs.
    ///
    /// `Unspecified` and `PinnedBase` are both empty here and that is correct
    /// TODAY, when no router exists — it is what makes this change behaviour
    /// preserving. They are not interchangeable: routing replaces the
    /// `Unspecified` arm and must leave `PinnedBase` alone, which is why they are
    /// separate variants rather than one emptiness test.
    pub fn selection(&self) -> &[LoraSelection] {
        match self {
            Self::Unspecified | Self::PinnedBase => &[],
            Self::Explicit(selection) => selection,
        }
    }

    /// Whether routing may choose for this request.
    pub fn is_routable(&self) -> bool {
        matches!(self, Self::Unspecified)
    }
}

/// Classify a request's `lora` field into its three states.
pub fn requested_adapters(field: Option<Vec<LoraSelection>>) -> RequestedAdapters {
    match field {
        None => RequestedAdapters::Unspecified,
        Some(selection) if selection.is_empty() => RequestedAdapters::PinnedBase,
        Some(selection) => RequestedAdapters::Explicit(selection),
    }
}

/// Metadata for one resident adapter; contains no weights.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdapterMetadata {
    /// Identifier, never reused during this worker's lifetime.
    pub id: u32,
    /// Display name supplied at load time.
    pub name: String,
    /// Source file path.
    pub path: String,
    /// Adapter rank.
    pub rank: usize,
    /// Number of projection layer records.
    pub layers: usize,
}

/// Snapshot published by the worker after each successful state change.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AdapterIndex {
    /// Resident adapters in increasing identifier order.
    pub adapters: Vec<AdapterMetadata>,
    /// Ordered mixture currently materialized in the engine, empty for base.
    pub applied: Vec<LoraSelection>,
    /// Why a blend of the FULL resident set (what a routed request actually
    /// blends) would refuse at execution, or `None` when it would not.
    /// Computed once per residency change (issue #1735) -- cached on
    /// `ResidencyRegistry` and recomputed only in its `load` and `unload`,
    /// the only two places a resident is added or removed, from the same
    /// shared plan `blend_lora_layer_data` itself runs, so this can never
    /// disagree with what a routed request meets. `publish` copies the
    /// cached value into this field on every state change; it does not
    /// re-plan.
    ///
    /// Never serialized at this struct's own top level: it belongs beside
    /// `routable` in the `router` object `lora_list_body` builds, which is
    /// where [`crate::serve::routing::routability`] reads it from.
    #[serde(skip)]
    pub blend_refusal: Option<String>,
}

impl AdapterIndex {
    /// Validate without waiting for generation; the worker rechecks at execution.
    pub fn validate(&self, selection: &[LoraSelection]) -> Result<(), ApiError> {
        validate_scales(selection)?;
        validate_unique_ids(selection)?;
        for entry in selection {
            if !self.adapters.iter().any(|adapter| adapter.id == entry.id) {
                return Err(unknown_adapter(entry.id));
            }
        }
        Ok(())
    }
}

pub(super) fn validate_unique_ids(selection: &[LoraSelection]) -> Result<(), ApiError> {
    let mut seen = HashSet::new();
    for entry in selection {
        if !seen.insert(entry.id) {
            return Err(ApiError::BadRequest {
                message: format!("duplicate LoRA adapter id {}", entry.id),
                code: "lora_duplicate_adapter_id",
            });
        }
    }
    Ok(())
}

/// Reject scales that cannot participate in a finite blend.
pub fn validate_scales(selection: &[LoraSelection]) -> Result<(), ApiError> {
    for entry in selection {
        if !entry.scale.is_finite() {
            return Err(ApiError::BadRequest {
                message: format!("LoRA adapter {} scale must be finite", entry.id),
                code: "invalid_request",
            });
        }
    }
    Ok(())
}

/// A missing resident identifier is an input error, including at dequeue time.
pub fn unknown_adapter(id: u32) -> ApiError {
    ApiError::BadRequest {
        message: format!("unknown LoRA adapter id {id}"),
        code: "lora_adapter_not_found",
    }
}

/// Parse a load request, defaulting its display name to its path.
pub fn parse_lora_load(bytes: &[u8]) -> Result<(String, String), ApiError> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Load {
        path: String,
        name: Option<String>,
    }
    let request: Load = serde_json::from_slice(bytes).map_err(|err| ApiError::BadRequest {
        message: format!("invalid LoRA load request: {err}"),
        code: "invalid_request",
    })?;
    let name = request.name.unwrap_or_else(|| request.path.clone());
    if request.path.trim().is_empty() || name.trim().is_empty() {
        return Err(ApiError::BadRequest {
            message: "`path` and `name` must not be empty".into(),
            code: "invalid_request",
        });
    }
    Ok((request.path, name))
}

/// Parse the identifier required by an unload request.
pub fn parse_lora_unload(bytes: &[u8]) -> Result<u32, ApiError> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Unload {
        id: u32,
    }
    serde_json::from_slice::<Unload>(bytes)
        .map(|request| request.id)
        .map_err(|err| ApiError::BadRequest {
            message: format!("invalid LoRA unload request: {err}"),
            code: "invalid_request",
        })
}

/// Parse a load request and return its path.
pub fn parse_lora_load_path(bytes: &[u8]) -> Result<String, ApiError> {
    parse_lora_load(bytes).map(|(path, _)| path)
}

/// Classify a residency command failure without hiding its diagnosis.
pub fn adapter_failure_code(error: &AdapterControlError) -> &'static str {
    match error {
        AdapterControlError::NotFound(_) => "lora_adapter_not_found",
        AdapterControlError::CountLimit { .. } | AdapterControlError::ByteLimit { .. } => {
            "lora_residency_limit_exceeded"
        }
        AdapterControlError::InvalidAdapter(_) | AdapterControlError::IdExhausted => {
            "lora_load_failed"
        }
    }
}

/// Worker-channel failure. A lost control reply leaves its outcome uncertain;
/// callers should inspect the metadata index before retrying.
pub fn worker_unavailable(message: &str) -> ApiError {
    ApiError::ServiceUnavailable {
        message: message.to_string(),
    }
}

/// `POST /v1/lora/load` success body.
pub fn load_success_body(id: u32, name: &str, path: &str, rank: usize, layers: usize) -> Value {
    serde_json::json!({
        "object": "lora.adapter",
        "status": "loaded",
        "id": id,
        "name": name,
        "path": path,
        "rank": rank,
        "layers": layers,
    })
}

/// `POST /v1/lora/unload` success body.
pub fn unload_success_body(id: u32) -> Value {
    serde_json::json!({"object": "lora.adapter", "status": "unloaded", "id": id})
}

/// One confirmed adapter-command result from a backend.
pub type AdapterControlReceiver =
    tokio::sync::oneshot::Receiver<Result<AdapterControlResult, AdapterControlError>>;

async fn adapter_request_bytes(
    headers: &axum::http::HeaderMap,
    body: axum::body::Body,
) -> Result<axum::body::Bytes, ApiError> {
    super::require_json_content_type(headers)?;
    axum::body::to_bytes(body, super::REQUEST_BODY_LIMIT_BYTES)
        .await
        .map_err(|_| ApiError::BadRequest {
            message: "invalid request body".into(),
            code: "invalid_request",
        })
}

/// Validate and admit a load, prepare its command, and report its confirmed result.
///
/// Build refusal precedes reading the request; runtime resolution follows parsing.
/// Backend callbacks provide preparation and submission without owning HTTP policy.
pub async fn load_request<B, P>(
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
    build_admission: Result<(), ApiError>,
    resolve_backend: impl FnOnce() -> Result<B, ApiError>,
    prepare_load: impl FnOnce(&str, &str) -> Result<P, ApiError>,
    submit_load: impl FnOnce(B, P) -> Result<AdapterControlReceiver, ApiError>,
) -> Result<axum::response::Response, ApiError> {
    use axum::response::IntoResponse as _;

    build_admission?;
    let bytes = adapter_request_bytes(&headers, body).await?;
    let (path, name) = parse_lora_load(&bytes)?;
    let backend = resolve_backend()?;
    let prepared = prepare_load(&path, &name)?;
    let receiver = submit_load(backend, prepared)?;
    match receiver.await {
        Ok(Ok(AdapterControlResult::Loaded(adapter))) => Ok(axum::Json(load_success_body(
            adapter.id,
            &adapter.name,
            &adapter.path,
            adapter.rank,
            adapter.layers,
        ))
        .into_response()),
        Ok(Err(error)) => Err(error.into()),
        Ok(Ok(_)) => Err(worker_unavailable("unexpected adapter command reply")),
        Err(_) => Err(worker_unavailable(
            "the inference worker is not running; the adapter was not loaded",
        )),
    }
}

/// Validate and admit an unload, then report its confirmed result.
///
/// Build refusal precedes reading the request; runtime resolution follows parsing.
pub async fn unload_request<B>(
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
    build_admission: Result<(), ApiError>,
    resolve_backend: impl FnOnce() -> Result<B, ApiError>,
    submit_unload: impl FnOnce(B, u32) -> Result<AdapterControlReceiver, ApiError>,
) -> Result<axum::response::Response, ApiError> {
    use axum::response::IntoResponse as _;

    build_admission?;
    let bytes = adapter_request_bytes(&headers, body).await?;
    let id = parse_lora_unload(&bytes)?;
    let backend = resolve_backend()?;
    let receiver = submit_unload(backend, id)?;
    match receiver.await {
        Ok(Ok(AdapterControlResult::Unloaded(id))) => {
            Ok(axum::Json(unload_success_body(id)).into_response())
        }
        Ok(Err(error)) => Err(error.into()),
        Ok(Ok(_)) => Err(worker_unavailable("unexpected adapter command reply")),
        Err(_) => Err(worker_unavailable(
            "the inference worker is not running; no adapter was unloaded",
        )),
    }
}

/// Resolve an adapter backend and report its confirmed snapshot and served router.
pub fn list_request<B>(
    build_admission: Result<(), ApiError>,
    resolve_backend: impl FnOnce() -> Result<B, ApiError>,
    snapshot: impl FnOnce(B) -> AdapterIndex,
    router: Option<&super::routing::ServedRouter>,
) -> Result<axum::response::Response, ApiError> {
    use axum::response::IntoResponse as _;

    build_admission?;
    let backend = resolve_backend()?;
    let index = snapshot(backend);
    let body = match router {
        None => lora_list_body(&index, None),
        Some(served) => served.with_report(|report| lora_list_body(&index, Some(report))),
    };
    Ok(axum::Json(body).into_response())
}

/// A parsed adapter and its source metadata, ready to hand to the worker.
/// The confirmed worker reply supplies response metadata when identity is reused.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub struct PreparedAdapter {
    pub command: super::metal_worker::AdapterCommand,
    pub rank: usize,
    pub layers: usize,
}

/// Read and parse the adapter at `path` into a worker command.
///
/// Runs on the request's own task, off the worker thread, so a malformed file
/// costs the worker nothing and the caller gets the parse error directly. A path
/// that does not exist and a file that fails to parse are different answers and
/// carry different codes: an adapter that was never read is not an adapter that
/// was read and rejected.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub fn prepare_adapter_load(path: &str, name: &str) -> Result<PreparedAdapter, ApiError> {
    let file = std::path::PathBuf::from(path);
    if !file.is_file() {
        return Err(ApiError::BadRequest {
            message: format!("adapter file not found: {path}"),
            code: "lora_adapter_not_found",
        });
    }
    let (layers, descriptor) =
        crate::lora_file::load_lora_safetensors(&file).map_err(|err| ApiError::BadRequest {
            message: err.to_string(),
            code: "lora_load_failed",
        })?;
    let count = layers.len();
    let rank = descriptor.rank;
    Ok(PreparedAdapter {
        command: super::metal_worker::AdapterCommand::Load {
            name: name.to_string(),
            path: path.to_string(),
            layers,
            descriptor: Box::new(descriptor),
        },
        rank,
        layers: count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bootstrap, a learned gate and a legacy one are three different
    /// answers on this surface, and the responses they produce are identical.
    /// The legacy arm is the one that matters: reporting it as `"bootstrap"`
    /// would describe a possibly-trained gate as a deliberate baseline, which
    /// is wrong in the reassuring direction.
    #[test]
    fn the_lora_index_reports_how_the_serving_gate_was_produced() {
        use crate::router_state::{
            RouterArtifact, RouterLineage, RouterPolicy, RouterProvenance, TrainedRepresentation,
        };

        let index = AdapterIndex {
            adapters: Vec::new(),
            applied: Vec::new(),
            blend_refusal: None,
        };
        let representation = TrainedRepresentation {
            embedding_model: "qwen3.5-0.8b".into(),
            pooling: "mean_visual".into(),
            prompt_source: "last_user_message".into(),
            loader_format: "qwen35-f16-decoder".into(),
            input_width: 8,
        };
        let policy = |training_state: &str| RouterPolicy {
            kind: "uniform".into(),
            tau: None,
            epsilon: 0.0,
            tau_floor: 0.0,
            entropy_floor: 0.0,
            max_consecutive_collapsed: 0,
            training_state: training_state.into(),
            objective_version: None,
            exploration_scale: None,
        };
        let lineage = RouterLineage {
            schema_id: "schema-1".into(),
            base_model: "qwen3.5-0.8b".into(),
            parent: None,
            admission_record: None,
        };
        let artifact = |provenance| RouterArtifact {
            version: 1,
            adapter_names: vec!["technical".into()],
            representation: representation.clone(),
            gate_bytes: vec![1, 2, 3],
            provenance,
        };

        for (provenance, expected) in [
            (
                RouterProvenance::recorded(policy("bootstrap"), lineage.clone()),
                "bootstrap",
            ),
            (
                RouterProvenance::recorded(policy("learned"), lineage.clone()),
                "learned",
            ),
            (RouterProvenance::Legacy, "unrecorded"),
        ] {
            let served = artifact(provenance);
            let resolved = crate::router_state::ResolvedRouter {
                artifact: served,
                pinned: false,
            };
            let body = lora_list_body(
                &index,
                Some(resolved.report(
                    "qwen3.5-0.8b",
                    crate::router_state::AppliedPolicy {
                        kind: "uniform".to_string(),
                        tau: None,
                        epsilon: 0.0,
                        exploration_scale: None,
                        learning_enabled: false,
                    },
                )),
            );
            assert_eq!(
                body["router"]["training_state"], expected,
                "the three provenances must not report the same thing"
            );
        }
    }

    /// The body says what requests are served under, not what the file records.
    /// A server that declined to enable a recorded learned policy reports the
    /// uniform one it is running, and this is the only surface an operator has
    /// for telling the two apart during a qualification run.
    #[test]
    fn the_body_reports_the_policy_in_force() {
        let index = AdapterIndex {
            adapters: Vec::new(),
            applied: Vec::new(),
            blend_refusal: None,
        };
        use crate::router_state::{
            RouterArtifact, RouterLineage, RouterPolicy, RouterProvenance, TrainedRepresentation,
        };

        let resolved = crate::router_state::ResolvedRouter {
            artifact: RouterArtifact {
                version: 1,
                adapter_names: vec!["technical".into()],
                representation: TrainedRepresentation {
                    embedding_model: "qwen3.5-0.8b".into(),
                    pooling: "mean_visual".into(),
                    prompt_source: "last_user_message".into(),
                    loader_format: "qwen35-f16-decoder".into(),
                    input_width: 8,
                },
                gate_bytes: vec![1, 2, 3],
                provenance: RouterProvenance::recorded(
                    RouterPolicy {
                        kind: "softmax".into(),
                        tau: Some(0.7),
                        epsilon: 0.05,
                        tau_floor: 0.1,
                        entropy_floor: 0.35,
                        max_consecutive_collapsed: 3,
                        training_state: "learned".into(),
                        objective_version: Some("whole-mixture-v1".into()),
                        exploration_scale: Some(0.5),
                    },
                    RouterLineage {
                        schema_id: "schema-1".into(),
                        base_model: "qwen3.5-0.8b".into(),
                        parent: None,
                        admission_record: None,
                    },
                ),
            },
            pinned: false,
        };
        let body = lora_list_body(
            &index,
            Some(resolved.report(
                "qwen3.5-0.8b",
                crate::router_state::AppliedPolicy {
                    kind: "softmax".to_string(),
                    tau: Some(0.7),
                    epsilon: 0.05,
                    exploration_scale: Some(0.5),
                    learning_enabled: true,
                },
            )),
        );
        assert_eq!(body["router"]["policy"]["kind"], "softmax");
        assert_eq!(body["router"]["policy"]["tau"], 0.7);
        assert_eq!(body["router"]["policy"]["exploration_scale"], 0.5);
        assert_eq!(body["router"]["policy"]["learning_enabled"], true);
        // The control that makes the four above about the policy and not about
        // the fixture: a deterministic server reports its own values under the
        // same keys, and they differ.
        let deterministic = lora_list_body(
            &index,
            Some(resolved.report(
                "qwen3.5-0.8b",
                crate::router_state::AppliedPolicy {
                    kind: "uniform".to_string(),
                    tau: None,
                    epsilon: 0.0,
                    exploration_scale: None,
                    learning_enabled: false,
                },
            )),
        );
        assert_eq!(deterministic["router"]["policy"]["kind"], "uniform");
        assert_eq!(deterministic["router"]["policy"]["learning_enabled"], false);
    }

    #[test]
    fn duplicate_ids_are_rejected_before_admission() {
        let index = AdapterIndex {
            adapters: [7, 11]
                .into_iter()
                .map(|id| AdapterMetadata {
                    id,
                    name: format!("adapter-{id}"),
                    path: format!("{id}.safetensors"),
                    rank: 1,
                    layers: 1,
                })
                .collect(),
            applied: Vec::new(),
            ..Default::default()
        };
        let unique = [
            LoraSelection { id: 7, scale: 0.0 },
            LoraSelection {
                id: 11,
                scale: -0.5,
            },
        ];
        assert!(index.validate(&unique).is_ok());
        assert!(index.validate(&[]).is_ok());
        for scale in [0.0, 1.0, -0.25] {
            let selection = [unique[0], unique[1], LoraSelection { id: 7, scale }];
            let error = index.validate(&selection).unwrap_err();
            assert!(matches!(error, ApiError::BadRequest { .. }));
            assert_eq!(error.code(), "lora_duplicate_adapter_id");
            assert_eq!(error.message(), "duplicate LoRA adapter id 7");
        }
    }

    #[test]
    fn selections_reject_nonfinite_scales_and_unknown_ids() {
        let index = AdapterIndex::default();
        for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = index
                .validate(&[LoraSelection { id: 42, scale }])
                .unwrap_err();
            assert!(error.message().contains("finite"));
        }
        let error = index
            .validate(&[LoraSelection { id: 42, scale: 1.0 }])
            .unwrap_err();
        assert!(matches!(error, ApiError::BadRequest { .. }));
        assert!(error.message().contains("42"));
        assert!(index.validate(&[]).is_ok());
    }

    #[test]
    fn load_name_and_unload_id_are_strictly_parsed() {
        assert_eq!(
            parse_lora_load(br#"{"path":"a","name":"display"}"#).unwrap(),
            ("a".into(), "display".into())
        );
        assert_eq!(
            parse_lora_load(br#"{"path":"a"}"#).unwrap(),
            ("a".into(), "a".into())
        );
        assert!(parse_lora_load(br#"{"path":"a","name":" "}"#).is_err());
        assert_eq!(
            parse_lora_unload(br#"{"id":4294967295}"#).unwrap(),
            u32::MAX
        );
        for invalid in [
            br#"{}"#.as_slice(),
            br#"{"id":-1}"#,
            br#"{"id":4294967296}"#,
            br#"{"id":0,"extra":1}"#,
        ] {
            assert!(parse_lora_unload(invalid).is_err());
        }
    }

    fn code_of(err: ApiError) -> &'static str {
        match err {
            ApiError::BadRequest { code, .. } => code,
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn a_well_formed_body_yields_the_path() {
        assert_eq!(
            parse_lora_load_path(br#"{"path":"/tmp/a.safetensors"}"#).unwrap(),
            "/tmp/a.safetensors"
        );
    }

    #[test]
    fn a_missing_path_is_refused() {
        assert_eq!(
            code_of(parse_lora_load_path(br#"{}"#).unwrap_err()),
            "invalid_request"
        );
    }

    /// The point of the rejection is that the value is not silently dropped: a
    /// caller asking for a scale this route does not apply must be told so, not
    /// served an adapter at the file's own scale as if the field had been honored.
    #[test]
    fn an_unknown_field_is_refused_rather_than_ignored() {
        let err =
            parse_lora_load_path(br#"{"path":"/tmp/a.safetensors","scale":2.0}"#).unwrap_err();
        let ApiError::BadRequest { message, code } = err else {
            panic!("expected BadRequest");
        };
        assert_eq!(code, "invalid_request");
        assert!(
            message.contains("scale"),
            "message must name the field: {message}"
        );
    }

    #[test]
    fn a_whitespace_path_is_refused() {
        assert_eq!(
            code_of(parse_lora_load_path(br#"{"path":"   "}"#).unwrap_err()),
            "invalid_request"
        );
    }

    #[test]
    fn a_non_object_body_is_refused() {
        assert_eq!(
            code_of(parse_lora_load_path(br#"["/tmp/a.safetensors"]"#).unwrap_err()),
            "invalid_request"
        );
    }

    #[test]
    fn a_worker_failure_separates_unknown_id_from_a_bad_adapter() {
        assert_eq!(
            adapter_failure_code(&AdapterControlError::NotFound(7)),
            "lora_adapter_not_found"
        );
        assert_eq!(
            adapter_failure_code(&AdapterControlError::InvalidAdapter(
                "unknown LoRA adapter id 7".into()
            )),
            "lora_load_failed"
        );
    }

    #[test]
    fn the_success_bodies_name_the_object_and_the_state() {
        let loaded = load_success_body(0, "a", "/tmp/a.safetensors", 8, 24);
        assert_eq!(loaded["object"], "lora.adapter");
        assert_eq!(loaded["status"], "loaded");
        assert_eq!(loaded["rank"], 8);
        assert_eq!(loaded["layers"], 24);
        assert_eq!(unload_success_body(0)["status"], "unloaded");
    }
}

#[cfg(test)]
mod orchestration_tests {
    use super::*;
    use axum::body::{Body, Bytes};
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::{IntoResponse as _, Response};
    use std::cell::RefCell;

    fn json_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        headers
    }

    fn refusal(message: &str) -> ApiError {
        ApiError::BadRequest {
            code: "lora_unsupported_backend",
            message: message.into(),
        }
    }

    fn unread_body() -> Body {
        Body::from_stream(futures::stream::poll_fn(
            |_| -> std::task::Poll<Option<Result<Bytes, std::io::Error>>> {
                panic!("body must not be polled")
            },
        ))
    }

    async fn value(response: Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn assert_error(error: ApiError, status: StatusCode, code: &str, message: &str) {
        let response = error.into_response();
        assert_eq!(response.status(), status);
        assert_eq!(
            value(response).await,
            serde_json::json!({"error": {
                "message": message,
                "type": if status == StatusCode::SERVICE_UNAVAILABLE {
                    "server_error"
                } else {
                    "invalid_request_error"
                },
                "code": code,
                "param": null,
            }})
        );
    }

    async fn before_backend(
        load: bool,
        headers: HeaderMap,
        body: Body,
        build: Result<(), ApiError>,
    ) -> ApiError {
        if load {
            load_request::<(), ()>(
                headers,
                body,
                build,
                || panic!("runtime must not be resolved"),
                |_, _| panic!("load must not be prepared"),
                |_, _| panic!("load must not be submitted"),
            )
            .await
            .unwrap_err()
        } else {
            unload_request::<()>(
                headers,
                body,
                build,
                || panic!("runtime must not be resolved"),
                |_, _| panic!("unload must not be submitted"),
            )
            .await
            .unwrap_err()
        }
    }

    fn confirmed() -> AdapterMetadata {
        AdapterMetadata {
            id: 71,
            name: "confirmed name".into(),
            path: "/confirmed/path".into(),
            rank: 16,
            layers: 27,
        }
    }

    fn receiver(
        reply: Option<Result<AdapterControlResult, AdapterControlError>>,
    ) -> AdapterControlReceiver {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        if let Some(reply) = reply {
            sender.send(reply).unwrap();
        }
        receiver
    }

    async fn with_reply(
        load: bool,
        reply: Option<Result<AdapterControlResult, AdapterControlError>>,
    ) -> Result<Response, ApiError> {
        if load {
            load_request(
                json_headers(),
                Body::from(r#"{"path":"requested","name":"request name"}"#),
                Ok(()),
                || Ok(()),
                |_, _| Ok(()),
                |(), ()| Ok(receiver(reply)),
            )
            .await
        } else {
            unload_request(
                json_headers(),
                Body::from(r#"{"id":4}"#),
                Ok(()),
                || Ok(()),
                |(), _| Ok(receiver(reply)),
            )
            .await
        }
    }

    #[tokio::test]
    async fn build_refusal_precedes_headers_body_and_backend() {
        for load in [true, false] {
            let error =
                before_backend(load, HeaderMap::new(), unread_body(), Err(refusal("build"))).await;
            assert_error(
                error,
                StatusCode::BAD_REQUEST,
                "lora_unsupported_backend",
                "build",
            )
            .await;
        }
    }

    #[tokio::test]
    async fn content_type_refusal_precedes_body_and_backend() {
        for load in [true, false] {
            for wrong_type in [false, true] {
                let mut headers = HeaderMap::new();
                if wrong_type {
                    headers.insert("content-type", "text/plain".parse().unwrap());
                }
                let error = before_backend(load, headers, unread_body(), Ok(())).await;
                assert_error(
                    error,
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "unsupported_media_type",
                    "Content-Type must be application/json",
                )
                .await;
            }
        }
    }

    #[tokio::test]
    async fn body_failures_precede_backend_and_keep_the_shared_error() {
        for load in [true, false] {
            let oversized = Body::from(vec![b' '; super::super::REQUEST_BODY_LIMIT_BYTES + 1]);
            let broken = Body::from_stream(futures::stream::once(async {
                Err::<Bytes, _>(std::io::Error::other("broken body"))
            }));
            for body in [oversized, broken] {
                let error = before_backend(load, json_headers(), body, Ok(())).await;
                assert_error(
                    error,
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    "invalid request body",
                )
                .await;
            }
        }
    }

    #[tokio::test]
    async fn strict_parsing_precedes_runtime_admission() {
        for (load, inputs) in [
            (
                true,
                &[
                    "{",
                    "{}",
                    r#"{"path":"x","extra":1}"#,
                    r#"{"path":" "}"#,
                    r#"{"path":"x","name":" "}"#,
                    r#"{"path":3}"#,
                ][..],
            ),
            (
                false,
                &[
                    "{",
                    "{}",
                    r#"{"id":1,"extra":1}"#,
                    r#"{"id":-1}"#,
                    r#"{"id":4294967296}"#,
                    r#"{"id":"1"}"#,
                ][..],
            ),
        ] {
            for input in inputs {
                let expected = if load {
                    parse_lora_load(input.as_bytes()).unwrap_err()
                } else {
                    parse_lora_unload(input.as_bytes()).unwrap_err()
                };
                let error = before_backend(load, json_headers(), Body::from(*input), Ok(())).await;
                assert_error(
                    error,
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    expected.message(),
                )
                .await;
            }
        }
    }

    #[tokio::test]
    async fn runtime_refusal_precedes_preparation_and_submission() {
        let load = load_request::<(), ()>(
            json_headers(),
            Body::from(r#"{"path":"requested"}"#),
            Ok(()),
            || Err(refusal("runtime")),
            |_, _| panic!("runtime refusal must precede preparation"),
            |_, _| panic!("runtime refusal must precede submission"),
        )
        .await
        .unwrap_err();
        let unload = unload_request::<()>(
            json_headers(),
            Body::from(r#"{"id":4}"#),
            Ok(()),
            || Err(refusal("runtime")),
            |_, _| panic!("runtime refusal must precede submission"),
        )
        .await
        .unwrap_err();
        for error in [load, unload] {
            assert_error(
                error,
                StatusCode::BAD_REQUEST,
                "lora_unsupported_backend",
                "runtime",
            )
            .await;
        }
    }

    #[tokio::test]
    async fn preparation_errors_precede_submission() {
        for code in ["lora_adapter_not_found", "lora_load_failed"] {
            let calls = RefCell::new(Vec::new());
            let error = load_request::<(), ()>(
                json_headers(),
                Body::from(r#"{"path":"requested"}"#),
                Ok(()),
                || {
                    calls.borrow_mut().push("resolve");
                    Ok(())
                },
                |path, name| {
                    calls.borrow_mut().push("prepare");
                    assert_eq!((path, name), ("requested", "requested"));
                    Err(ApiError::BadRequest {
                        code,
                        message: "preparation diagnosis".into(),
                    })
                },
                |_, _| panic!("failed preparation must not submit"),
            )
            .await
            .unwrap_err();
            assert_eq!(*calls.borrow(), ["resolve", "prepare"]);
            assert_error(
                error,
                StatusCode::BAD_REQUEST,
                code,
                "preparation diagnosis",
            )
            .await;
        }
    }

    #[tokio::test]
    async fn submit_errors_preserve_status_and_diagnosis() {
        let load = load_request(
            json_headers(),
            Body::from(r#"{"path":"requested"}"#),
            Ok(()),
            || Ok(()),
            |_, _| Ok(()),
            |(), ()| Err(worker_unavailable("control slot busy")),
        )
        .await
        .unwrap_err();
        let unload = unload_request(
            json_headers(),
            Body::from(r#"{"id":4}"#),
            Ok(()),
            || Ok(()),
            |(), id| {
                assert_eq!(id, 4);
                Err(worker_unavailable("control slot busy"))
            },
        )
        .await
        .unwrap_err();
        for error in [load, unload] {
            assert_error(
                error,
                StatusCode::SERVICE_UNAVAILABLE,
                "server_busy",
                "control slot busy",
            )
            .await;
        }
    }

    #[tokio::test]
    async fn load_preserves_literal_input_order_and_confirmed_metadata() {
        let calls = RefCell::new(Vec::new());
        let response = load_request(
            json_headers(),
            Body::from(r#"{"path":" requested ","name":" request name "}"#),
            Ok(()),
            || {
                calls.borrow_mut().push("resolve");
                Ok(17)
            },
            |path, name| {
                calls.borrow_mut().push("prepare");
                assert_eq!((path, name), (" requested ", " request name "));
                Ok((path.to_owned(), name.to_owned(), 2, 3))
            },
            |backend, prepared| {
                calls.borrow_mut().push("submit");
                assert_eq!(backend, 17);
                assert_eq!(
                    prepared,
                    (" requested ".into(), " request name ".into(), 2, 3)
                );
                Ok(receiver(Some(Ok(
                    AdapterControlResult::Loaded(confirmed()),
                ))))
            },
        )
        .await
        .unwrap();
        assert_eq!(*calls.borrow(), ["resolve", "prepare", "submit"]);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            value(response).await,
            serde_json::json!({
                "object":"lora.adapter", "status":"loaded", "id":71,
                "name":"confirmed name", "path":"/confirmed/path", "rank":16, "layers":27,
            })
        );
    }

    #[tokio::test]
    async fn unload_submits_requested_id_and_returns_confirmed_id() {
        let calls = RefCell::new(Vec::new());
        let response = unload_request(
            json_headers(),
            Body::from(r#"{"id":4}"#),
            Ok(()),
            || {
                calls.borrow_mut().push("resolve");
                Ok(17)
            },
            |backend, id| {
                calls.borrow_mut().push("submit");
                assert_eq!((backend, id), (17, 4));
                Ok(receiver(Some(Ok(AdapterControlResult::Unloaded(71)))))
            },
        )
        .await
        .unwrap();
        assert_eq!(*calls.borrow(), ["resolve", "submit"]);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            value(response).await,
            serde_json::json!({"object":"lora.adapter", "status":"unloaded", "id":71})
        );
    }

    #[tokio::test]
    async fn every_worker_error_keeps_its_code_and_message() {
        for load in [true, false] {
            for (error, code, message) in [
                (
                    AdapterControlError::InvalidAdapter("bad tensor".into()),
                    "lora_load_failed",
                    "bad tensor",
                ),
                (
                    AdapterControlError::NotFound(8),
                    "lora_adapter_not_found",
                    "unknown LoRA adapter id 8",
                ),
                (
                    AdapterControlError::CountLimit { limit: 2 },
                    "lora_residency_limit_exceeded",
                    "resident LoRA adapter count limit reached (2)",
                ),
                (
                    AdapterControlError::ByteLimit { limit: 32 },
                    "lora_residency_limit_exceeded",
                    "resident LoRA adapter tensor payload byte limit exceeded (32)",
                ),
                (
                    AdapterControlError::IdExhausted,
                    "lora_load_failed",
                    "LoRA adapter id space exhausted",
                ),
            ] {
                let error = with_reply(load, Some(Err(error))).await.unwrap_err();
                assert_error(error, StatusCode::BAD_REQUEST, code, message).await;
            }
        }
    }

    #[tokio::test]
    async fn wrong_reply_variants_refuse_in_both_operations() {
        for (load, reply) in [
            (true, AdapterControlResult::Unloaded(71)),
            (false, AdapterControlResult::Loaded(confirmed())),
        ] {
            let error = with_reply(load, Some(Ok(reply))).await.unwrap_err();
            assert_error(
                error,
                StatusCode::SERVICE_UNAVAILABLE,
                "server_busy",
                "unexpected adapter command reply",
            )
            .await;
        }
    }

    #[tokio::test]
    async fn dropped_replies_keep_operation_specific_diagnoses() {
        for (load, message) in [
            (
                true,
                "the inference worker is not running; the adapter was not loaded",
            ),
            (
                false,
                "the inference worker is not running; no adapter was unloaded",
            ),
        ] {
            let error = with_reply(load, None).await.unwrap_err();
            assert_error(
                error,
                StatusCode::SERVICE_UNAVAILABLE,
                "server_busy",
                message,
            )
            .await;
        }
    }

    #[tokio::test]
    async fn list_refusals_precede_snapshot_access() {
        let build = list_request::<()>(
            Err(refusal("build")),
            || panic!("build refusal must precede runtime"),
            |()| panic!("build refusal must precede snapshot"),
            None,
        )
        .unwrap_err();
        let runtime = list_request::<()>(
            Ok(()),
            || Err(refusal("runtime")),
            |()| panic!("runtime refusal must precede snapshot"),
            None,
        )
        .unwrap_err();
        assert_error(
            build,
            StatusCode::BAD_REQUEST,
            "lora_unsupported_backend",
            "build",
        )
        .await;
        assert_error(
            runtime,
            StatusCode::BAD_REQUEST,
            "lora_unsupported_backend",
            "runtime",
        )
        .await;
    }

    #[tokio::test]
    async fn list_reads_one_confirmed_snapshot_without_a_router() {
        let calls = RefCell::new(Vec::new());
        let response = list_request(
            Ok(()),
            || {
                calls.borrow_mut().push("resolve");
                Ok(17)
            },
            |backend| {
                calls.borrow_mut().push("snapshot");
                assert_eq!(backend, 17);
                AdapterIndex {
                    adapters: vec![confirmed()],
                    applied: vec![LoraSelection { id: 71, scale: 0.5 }],
                    ..Default::default()
                }
            },
            None,
        )
        .unwrap();
        assert_eq!(*calls.borrow(), ["resolve", "snapshot"]);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            value(response).await,
            serde_json::json!({
                "adapters":[{"id":71,"name":"confirmed name","path":"/confirmed/path","rank":16,"layers":27}],
                "applied":[{"id":71,"scale":0.5}], "router":{"enabled":false},
            })
        );
    }

    #[cfg(feature = "mixture")]
    #[tokio::test]
    async fn list_reports_the_served_router_beside_residency() {
        use crate::router_state::{
            ResolvedRouter, RouterArtifact, RouterProvenance, TrainedRepresentation,
        };
        use crate::serve::routing::{EmbedderIdentity, ServedRouter};
        use lattice_fann::{Activation, NetworkBuilder};

        let artifact = RouterArtifact {
            version: 7,
            adapter_names: vec!["confirmed name".into()],
            representation: TrainedRepresentation {
                embedding_model: "gme-qwen35".into(),
                pooling: "mean_visual".into(),
                prompt_source: "last_user_message".into(),
                loader_format: crate::serve::embeddings::QWEN35_F16_DECODER_LOADER.into(),
                input_width: 4,
            },
            gate_bytes: NetworkBuilder::new()
                .input(4)
                .output(1, Activation::Linear)
                .build()
                .unwrap()
                .to_bytes(),
            provenance: RouterProvenance::Legacy,
        };
        let expected_version = artifact.version_label();
        let identity =
            EmbedderIdentity::resolve(std::path::Path::new("/models"), Some("gme-qwen35"));
        let served = ServedRouter::new(
            ResolvedRouter {
                artifact,
                pinned: true,
            },
            &identity,
            4,
            false,
        )
        .unwrap();
        let response = list_request(
            Ok(()),
            || Ok(()),
            |()| AdapterIndex {
                adapters: vec![confirmed()],
                ..Default::default()
            },
            Some(&served),
        )
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = value(response).await;
        assert_eq!(body["adapters"][0]["id"], 71);
        assert_eq!(body["applied"], serde_json::json!([]));
        assert_eq!(
            body["router"],
            serde_json::json!({
                "enabled":true, "version":expected_version, "pinned":true, "embedder":"gme-qwen35",
                "adapter_names":["confirmed name"], "routable":true, "missing":[], "unexpected":[],
                "duplicate_trained":null, "duplicate_resident":null, "blend_refusal":null,
                "training_state":"unrecorded",
                "policy":{
                    "kind":"uniform", "tau":null, "epsilon":0.0,
                    "exploration_scale":null, "learning_enabled":false,
                },
            })
        );
    }

    #[test]
    fn native_compatible_callbacks_produce_send_futures() {
        fn assert_send<T: Send>(_: T) {}
        assert_send(load_request::<(), ()>(
            json_headers(),
            Body::empty(),
            Ok(()),
            || Ok(()),
            |_, _| Ok(()),
            |(), ()| Ok(receiver(None)),
        ));
        assert_send(unload_request::<()>(
            json_headers(),
            Body::empty(),
            Ok(()),
            || Ok(()),
            |(), _| Ok(receiver(None)),
        ));
    }
}

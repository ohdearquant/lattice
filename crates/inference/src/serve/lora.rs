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
    /// Computed once per residency change (issue #1735), from the same
    /// shared plan `blend_lora_layer_data` itself runs, so this can never
    /// disagree with what a routed request meets.
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

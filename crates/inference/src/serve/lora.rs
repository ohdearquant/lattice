//! Resident-adapter metadata, request parsing, and HTTP responses shared by both servers.

use super::ApiError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

/// One contribution to an ordered request mixture.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoraSelection {
    /// Resident adapter identifier.
    pub id: u32,
    /// Multiplier on the adapter's alpha/rank scale.
    pub scale: f32,
}

/// Metadata for one resident adapter; contains no weights.
#[derive(Debug, Clone, Serialize)]
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
pub fn adapter_failure_code(message: &str) -> &'static str {
    if message.starts_with("unknown LoRA adapter id ") {
        "lora_adapter_not_found"
    } else {
        "lora_load_failed"
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

/// A parsed adapter, ready to hand to the worker, plus the two numbers the
/// success body reports.
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
            adapter_failure_code("unknown LoRA adapter id 7"),
            "lora_adapter_not_found"
        );
        assert_eq!(
            adapter_failure_code("load_lora_adapter: layers must not be empty"),
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

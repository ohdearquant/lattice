//! Runtime LoRA adapter control, shared by both HTTP servers.
//!
//! `POST /v1/lora/load` and `POST /v1/lora/unload` exist on `lattice serve` and on
//! the standalone `lattice_serve` binary, and every part of them that is not axum
//! plumbing lives here: the request parser, the file-to-command translation, the
//! classification of a worker-side failure, and the success bodies. Two copies of a
//! request parser is a fork, and the two binaries already carry a documented
//! parity contract (see [`crate::serve::contract`]).
//!
//! What is deliberately NOT here: the engine's own single-slot adapter API and its
//! validation. The worker owns that, and this module never second-guesses the
//! message it returns.

use super::ApiError;
use serde_json::Value;

/// Parse a `/v1/lora/load` body: a JSON object with exactly one string field,
/// `path`.
///
/// Unknown fields are refused rather than ignored. A caller who sends `scale` is
/// asking for something this route does not do, and silently dropping it would
/// load an adapter at a scale the caller did not ask for; the adapter's own
/// `__metadata__` alpha decides the scale (see
/// [`crate::lora_file::resolve_lora_rank_alpha_scale`]).
pub fn parse_lora_load_path(bytes: &[u8]) -> Result<String, ApiError> {
    let bad = |message: String| ApiError::BadRequest {
        message,
        code: "invalid_request",
    };
    let value: Value =
        serde_json::from_slice(bytes).map_err(|err| bad(format!("invalid JSON body: {err}")))?;
    let Some(object) = value.as_object() else {
        return Err(bad("request body must be a JSON object".to_string()));
    };
    for key in object.keys() {
        if key != "path" {
            return Err(bad(format!("unknown field `{key}`; expected only `path`")));
        }
    }
    let Some(path) = object.get("path").and_then(Value::as_str) else {
        return Err(bad("missing required string field `path`".to_string()));
    };
    if path.trim().is_empty() {
        return Err(bad("`path` must not be empty".to_string()));
    }
    Ok(path.to_string())
}

/// The machine-readable code a worker-side adapter failure carries.
///
/// The worker hands back the loader's own message unchanged, which is the only
/// diagnosis a caller gets, so this classifies rather than rewrites. Both arms are
/// HTTP 400: [`ApiError`] expresses 400, 413, 415, 500 and 503 and has no conflict
/// variant, and widening it changes a contract both binaries and their parity tests
/// share. The code is therefore the discriminator a caller branches on, and
/// `lora_adapter_already_loaded` says the sequencing was wrong rather than the
/// adapter: the worker's command set deliberately has no swap, so unload-then-load
/// is the intended answer.
pub fn adapter_failure_code(message: &str) -> &'static str {
    if message.contains("already loaded") {
        "lora_adapter_already_loaded"
    } else {
        "lora_load_failed"
    }
}

/// A closed reply channel means the command was never applied.
///
/// Reporting that as success is the worst available answer, so both routes turn it
/// into a 503 naming the worker. `ApiError::ServiceUnavailable` carries the fixed
/// code `server_busy`, which is the same code the admission cap uses: in both cases
/// the server is up and the engine is not taking work.
pub fn worker_unavailable(message: &str) -> ApiError {
    ApiError::ServiceUnavailable {
        message: message.to_string(),
    }
}

/// `POST /v1/lora/load` success body.
pub fn load_success_body(path: &str, rank: usize, layers: usize) -> Value {
    serde_json::json!({
        "object": "lora.adapter",
        "status": "loaded",
        "path": path,
        "rank": rank,
        "layers": layers,
    })
}

/// `POST /v1/lora/unload` success body.
pub fn unload_success_body() -> Value {
    serde_json::json!({"object": "lora.adapter", "status": "unloaded"})
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
pub fn prepare_adapter_load(path: &str) -> Result<PreparedAdapter, ApiError> {
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
            layers,
            descriptor: Box::new(descriptor),
            quarot_seed: None,
        },
        rank,
        layers: count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The already-loaded case is a sequencing answer, not a bad-adapter answer.
    /// The fixture string is the engine's own message; if it is ever reworded this
    /// test is what notices the mapping has gone quiet.
    #[test]
    fn a_worker_failure_separates_sequencing_from_a_bad_adapter() {
        assert_eq!(
            adapter_failure_code("LoRA adapter already loaded; call unload_lora_adapter first"),
            "lora_adapter_already_loaded"
        );
        assert_eq!(
            adapter_failure_code("load_lora_adapter: layers must not be empty"),
            "lora_load_failed"
        );
    }

    #[test]
    fn the_success_bodies_name_the_object_and_the_state() {
        let loaded = load_success_body("/tmp/a.safetensors", 8, 24);
        assert_eq!(loaded["object"], "lora.adapter");
        assert_eq!(loaded["status"], "loaded");
        assert_eq!(loaded["rank"], 8);
        assert_eq!(loaded["layers"], 24);
        assert_eq!(unload_success_body()["status"], "unloaded");
    }
}

//! Shared HTTP serving contract for the `lattice` unified server
//! (`crates/inference/src/bin/lattice.rs`) and the `lattice_serve` daemon
//! (`crates/inference/src/bin/lattice_serve.rs`) -- ADR-080 cluster C2 (#782).
//!
//! Both binaries speak a subset of the OpenAI chat-completions wire format,
//! and both previously carried independent copies of: the error envelope
//! shape, the `finish_reason` mapping from the engine's `stopped` flag, the
//! `max_tokens == 0` rejection, and the `/v1/models` response body -- with
//! real drift between the copies (#744, #745, #746: `lattice_serve.rs`
//! discarded the engine's stop cause and hardcoded `finish_reason: "stop"`,
//! silently accepted `max_tokens: 0`, and never installed `/v1/models`'s
//! sibling route on the other binary). This module is the single source of
//! truth for those contracts; each binary still owns its own router wiring
//! and backend-specific generation dispatch (CPU/Metal dispatch for
//! `lattice.rs`, the daemon job queue for `lattice_serve.rs`) -- per the
//! ADR, only the request/response CONTRACT is shared, not the kernels or
//! scheduling behind it.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::Value;
use tokio::sync::watch;

/// Request body size cap shared by both HTTP servers: 1 MiB. Both binaries
/// already enforced this exact limit independently (`lattice.rs` via
/// `DefaultBodyLimit::max`, `lattice_serve.rs` via `to_bytes(body, LIMIT)`);
/// centralizing the constant removes one silent-drift vector even though the
/// two binaries still wire it into axum differently.
pub const REQUEST_BODY_LIMIT_BYTES: usize = 1_048_576;

/// Structured HTTP error shared by both binaries, serializing to the OpenAI
/// error envelope: `{"error": {"message", "type", "code", "param"}}`.
#[derive(Debug)]
pub enum ApiError {
    /// Caller mistake — HTTP 400.
    BadRequest { message: String, code: &'static str },
    /// Request body exceeds size limit — HTTP 413.
    PayloadTooLarge { message: String },
    /// Server-side failure — HTTP 500.
    Internal { message: String },
}

impl ApiError {
    /// The human-readable message, regardless of variant. Used by tests and
    /// by callers that need the text without matching on the variant.
    pub fn message(&self) -> &str {
        match self {
            ApiError::BadRequest { message, .. } => message,
            ApiError::PayloadTooLarge { message } => message,
            ApiError::Internal { message } => message,
        }
    }
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: &'static str,
    code: String,
    param: Option<String>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::BadRequest { message, code } => {
                let body = Json(ErrorBody {
                    error: ErrorDetail {
                        message,
                        r#type: "invalid_request_error",
                        code: code.to_string(),
                        param: None,
                    },
                });
                (StatusCode::BAD_REQUEST, body).into_response()
            }
            ApiError::PayloadTooLarge { message } => {
                let body = Json(ErrorBody {
                    error: ErrorDetail {
                        message,
                        r#type: "invalid_request_error",
                        code: "request_body_too_large".to_string(),
                        param: None,
                    },
                });
                (StatusCode::PAYLOAD_TOO_LARGE, body).into_response()
            }
            ApiError::Internal { message } => {
                let body = Json(ErrorBody {
                    error: ErrorDetail {
                        message,
                        r#type: "server_error",
                        code: "internal_error".to_string(),
                        param: None,
                    },
                });
                (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
            }
        }
    }
}

/// Maps a generation's `stopped` flag to the OpenAI `finish_reason` string:
/// `"stop"` when the engine explicitly ended generation via a stop condition
/// (EOS, stop-token-id, or stop-string match); `"length"` when the token
/// budget was exhausted without one, or generation was interrupted by the
/// caller (a disconnect is not an OpenAI "stop condition" either). The
/// ENGINE-reported `stopped` flag is the single source of truth --
/// `lattice.rs`'s `finish_reason_for` already carried it through correctly;
/// `lattice_serve.rs`'s worker previously discarded it entirely and
/// hardcoded `"stop"` unconditionally in both SSE and JSON responses (#746).
pub fn finish_reason(stopped: bool) -> &'static str {
    if stopped { "stop" } else { "length" }
}

/// Rejects a resolved `effective` `max_tokens` value of zero (#745).
/// `lattice.rs`'s `validate_max_tokens` already enforced this;
/// `lattice_serve.rs`'s `build_cfg` silently let a client-supplied
/// `max_tokens: 0` (or `max_completion_tokens: 0`) clamp through unchanged,
/// producing a zero-budget completion instead of a clear rejection.
///
/// Scoped narrowly to the zero case only, matching #745's triaged scope:
/// the two binaries' cap/alias-conflict policies differ intentionally
/// (`lattice.rs` rejects a request whose resolved `max_tokens` exceeds its
/// server cap; `lattice_serve.rs` clamps it to the model's context window
/// instead) and are deliberately NOT unified by this helper.
pub fn reject_zero_max_tokens(effective: usize) -> Result<(), ApiError> {
    if effective == 0 {
        return Err(ApiError::BadRequest {
            message: "max_tokens must be at least 1".to_string(),
            code: "invalid_max_tokens",
        });
    }
    Ok(())
}

/// `GET /v1/models` response body: advertises the single loaded model.
/// Shared so both binaries expose byte-identical shapes for the same model
/// id and `created` timestamp -- previously only `lattice_serve.rs`
/// installed this route at all; `lattice.rs` had no equivalent endpoint.
pub fn models_list_body(model_id: &str, created: u64) -> Value {
    serde_json::json!({
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": created,
            "owned_by": "lattice",
        }],
    })
}

/// Disconnect-cancellation contract shared by both HTTP servers (#744):
/// flips the paired [`watch::Receiver<bool>`] to `true` the moment this guard
/// is dropped. Held inside the per-request SSE stream state (streaming) or
/// the handler's local scope (non-streaming) so it drops exactly when axum
/// stops caring about the response -- on client disconnect, or harmlessly
/// after the request already finished normally. `lattice_serve.rs` already
/// had this exact type as a private struct; `lattice.rs`'s CPU streaming
/// path had no equivalent at all and kept generating to the token cap after
/// a client left (its own comment documented this as "a future
/// refinement") -- this hoists the ONE contract both binaries now share.
pub struct CancelOnDrop(pub watch::Sender<bool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let _ = self.0.send(true);
    }
}

/// Fresh cancel-on-drop guard/receiver pair for one request. The receiver is
/// threaded into the engine's `should_cancel` predicate (checked before
/// prefill, immediately after prefill, and at the top of every decode
/// iteration); the guard is held for the lifetime of the response so it
/// fires the moment axum drops it.
pub fn cancel_pair() -> (CancelOnDrop, watch::Receiver<bool>) {
    let (tx, rx) = watch::channel(false);
    (CancelOnDrop(tx), rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_reason_stopped_true_is_stop() {
        assert_eq!(finish_reason(true), "stop");
    }

    #[test]
    fn finish_reason_stopped_false_is_length() {
        assert_eq!(finish_reason(false), "length");
    }

    #[test]
    fn reject_zero_max_tokens_rejects_zero() {
        let err = reject_zero_max_tokens(0).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_max_tokens",
                ..
            }
        ));
    }

    #[test]
    fn reject_zero_max_tokens_accepts_positive() {
        assert!(reject_zero_max_tokens(1).is_ok());
        assert!(reject_zero_max_tokens(4096).is_ok());
    }

    #[test]
    fn models_list_body_shape() {
        let body = models_list_body("my-model", 1_700_000_000);
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["id"], "my-model");
        assert_eq!(body["data"][0]["object"], "model");
        assert_eq!(body["data"][0]["created"], 1_700_000_000);
        assert_eq!(body["data"][0]["owned_by"], "lattice");
    }

    #[test]
    fn api_error_bad_request_envelope_shape() {
        let err = ApiError::BadRequest {
            message: "bad".to_string(),
            code: "some_code",
        };
        assert_eq!(err.message(), "bad");
        // IntoResponse is exercised at the HTTP layer in each binary's own
        // tests (axum::response::Response has no public body-introspection
        // API worth duplicating here); this pins the pure data this module
        // owns instead.
    }

    #[test]
    fn api_error_internal_message() {
        let err = ApiError::Internal {
            message: "oops".to_string(),
        };
        assert_eq!(err.message(), "oops");
    }

    #[test]
    fn api_error_payload_too_large_message() {
        let err = ApiError::PayloadTooLarge {
            message: "too big".to_string(),
        };
        assert_eq!(err.message(), "too big");
    }

    #[test]
    fn cancel_pair_receiver_starts_false() {
        let (_guard, rx) = cancel_pair();
        assert!(!*rx.borrow());
    }

    #[test]
    fn cancel_on_drop_flips_receiver_true_on_drop() {
        let (guard, rx) = cancel_pair();
        assert!(!*rx.borrow());
        drop(guard);
        assert!(*rx.borrow());
    }

    #[test]
    fn cancel_on_drop_leaves_receiver_false_while_alive() {
        let (guard, rx) = cancel_pair();
        assert!(!*rx.borrow());
        // Guard still in scope -- receiver must not have flipped yet.
        assert!(!*rx.borrow());
        drop(guard);
    }
}

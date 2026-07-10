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

/// Shared streaming context-overflow parity fixture (ADR-080 C2 round 3,
/// codex round-3 medium finding #2): both binaries' real-router
/// context-overflow tests build their request from these SAME constants and
/// configure their real (tiny, test-only CPU) model's effective context
/// window to this SAME value, so "same input, same effective limit" is
/// enforced by shared constants rather than by two independently-typed
/// literals that could silently drift apart. `lattice.rs`'s tiny test model
/// (`lattice_inference::model::qwen35::test_support::tiny_zero_model`) has a
/// fixed 1024-token context window; `lattice_serve.rs`'s real-worker test
/// configures its `AppState.model_max_context` to the same figure.
pub const OVERFLOW_PARITY_CONTEXT_WINDOW: usize = 1024;
/// `max_tokens` for the shared overflow-parity request: equal to the whole
/// context window, so any non-empty prompt pushes `prompt_len + max_tokens`
/// past it once the worker's full-window check (not just `build_cfg`'s
/// in-isolation clamp) runs.
pub const OVERFLOW_PARITY_MAX_TOKENS: usize = OVERFLOW_PARITY_CONTEXT_WINDOW;
/// Request-level `max_tokens` cap, kept well above
/// [`OVERFLOW_PARITY_MAX_TOKENS`] so a cap-rejection (`max_tokens_exceeds_limit`
/// / equivalent) never fires first and masks the context-window check this
/// fixture exists to isolate.
pub const OVERFLOW_PARITY_MAX_TOKENS_CAP: usize = 4096;
/// The exact request body both binaries' overflow-parity tests send.
pub const OVERFLOW_PARITY_REQUEST_BODY: &str = r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":1024,"stream":true}"#;

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

/// `GET /` response body: a minimal engine-identity/endpoint-discovery
/// document. Shared so both binaries expose a byte-identical root route
/// (ADR-080 C2 round 2, codex finding #1): `lattice_serve.rs` already
/// served this; `lattice.rs` had no `GET /` route at all, an undocumented
/// divergence the review's route-set audit caught. Both binaries expose the
/// same three routes, so the endpoint list is a fixed constant here rather
/// than a per-binary parameter.
pub fn root_body() -> Value {
    serde_json::json!({
        "name": "lattice",
        "object": "engine",
        "endpoints": ["/v1/chat/completions", "/v1/models", "/health"],
    })
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

/// Which binary a [`ParityCase`] expectation applies to. Both binaries build
/// their own `Router` and drive it independently (bins can't cross-import
/// each other's `chat_completions`/router as a normal dependency), so a case
/// carries per-binary expected outcomes rather than one shared HTTP call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binary {
    Lattice,
    LatticeServe,
}

/// A parity case's request body. Most cases pin a small literal fixture;
/// [`CaseBody::Oversized`] generates a body larger than
/// [`REQUEST_BODY_LIMIT_BYTES`] at test time instead of embedding a >1MiB
/// string literal in source (ADR-080 C2 round 2, codex round-2 medium
/// finding #2: neither binary's parity table exercised the oversized-body
/// case at all, so restoring the daemon's old 400/`invalid_request` mapping
/// left its parity test green).
pub enum CaseBody {
    Fixed(&'static str),
    /// A `messages` array whose `content` field alone exceeds
    /// `REQUEST_BODY_LIMIT_BYTES`, forcing both binaries' body-limit
    /// enforcement (`DefaultBodyLimit` on `lattice.rs`, manual
    /// `to_bytes(.., LIMIT)` on `lattice_serve.rs`) to trip.
    Oversized,
}

impl CaseBody {
    pub fn build(&self) -> Vec<u8> {
        match self {
            CaseBody::Fixed(s) => s.as_bytes().to_vec(),
            CaseBody::Oversized => {
                let filler = "x".repeat(REQUEST_BODY_LIMIT_BYTES + 1);
                format!(
                    r#"{{"model":"test-model","messages":[{{"role":"user","content":"{filler}"}}]}}"#
                )
                .into_bytes()
            }
        }
    }
}

/// One row of the cross-binary HTTP parity table (ADR-080 C2 round 2, codex
/// finding #1): a request `method`/`path`/`body`, and the expected
/// `(status, error_code)` for each binary. A case whose `divergence_reason`
/// is `None` means both binaries must produce an identical outcome for this
/// request (the common case, post-alignment); `Some` documents an
/// intentional, reviewed per-binary difference -- an undocumented
/// divergence is exactly the drift this table exists to catch. `method`/
/// `path` were added in round 2 (codex medium finding #2) after an
/// unguarded `GET /` route removal on `lattice.rs` left the round-1 table
/// green: every case before that was implicitly `POST
/// /v1/chat/completions`, so route exposure itself was never actually
/// checked.
pub struct ParityCase {
    pub name: &'static str,
    pub method: &'static str,
    pub path: &'static str,
    pub body: CaseBody,
    lattice: (u16, &'static str),
    lattice_serve: (u16, &'static str),
    /// `Some` only for a documented intentional divergence; explains WHY the
    /// two expected outcomes differ (reviewed alongside the table, not left
    /// to be inferred from the two tuples).
    pub divergence_reason: Option<&'static str>,
}

impl ParityCase {
    pub fn expected(&self, binary: Binary) -> (u16, &'static str) {
        match binary {
            Binary::Lattice => self.lattice,
            Binary::LatticeServe => self.lattice_serve,
        }
    }
}

/// Shared fixture table for both binaries' `/v1/chat/completions` HTTP
/// contract, driven through each binary's real `Router` via
/// `tower::ServiceExt::oneshot` in `lattice.rs`'s and `lattice_serve.rs`'s
/// own test modules. Every case that ISN'T a documented divergence must
/// resolve to the SAME `(status, code)` on both binaries -- codex's review
/// named concrete drift this closes: oversized body (413/`request_body_too_large`
/// vs 400/`invalid_request`), zero `max_tokens` (`invalid_max_tokens` vs
/// erased to `invalid_request`), and unknown role (generic message/no code
/// on one side).
pub const CHAT_COMPLETIONS_PARITY_CASES: &[ParityCase] = &[
    ParityCase {
        name: "unknown_role_not_openai",
        method: "POST",
        path: "/v1/chat/completions",
        // A trailing valid `user` turn keeps this isolated to the role
        // check: `lattice.rs` separately requires the conversation's LAST
        // message to have role `user` (a Qwen ChatML constraint, unrelated
        // to and checked before role-validity), so a single-message
        // `moderator` body would fail on THAT check first with
        // `invalid_messages` instead of exercising role validation at all.
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"moderator","content":"hi"},{"role":"user","content":"hi"}]}"#,
        ),
        lattice: (400, "invalid_role"),
        lattice_serve: (400, "invalid_role"),
        divergence_reason: None,
    },
    ParityCase {
        name: "developer_role_unsupported_feature",
        method: "POST",
        path: "/v1/chat/completions",
        // See `unknown_role_not_openai`'s comment on the trailing `user` turn.
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"developer","content":"hi"},{"role":"user","content":"hi"}]}"#,
        ),
        lattice: (400, "unsupported_feature"),
        lattice_serve: (400, "unsupported_feature"),
        divergence_reason: None,
    },
    ParityCase {
        name: "empty_messages",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(r#"{"model":"test-model","messages":[]}"#),
        lattice: (400, "invalid_messages"),
        lattice_serve: (400, "invalid_messages"),
        divergence_reason: None,
    },
    ParityCase {
        name: "max_tokens_zero",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#,
        ),
        lattice: (400, "invalid_max_tokens"),
        lattice_serve: (400, "invalid_max_tokens"),
        divergence_reason: None,
    },
    ParityCase {
        name: "max_tokens_and_max_completion_tokens_conflict",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"max_completion_tokens":20}"#,
        ),
        lattice: (400, "invalid_request"),
        lattice_serve: (400, "invalid_request"),
        divergence_reason: None,
    },
    ParityCase {
        name: "tools_unsupported",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}]}"#,
        ),
        lattice: (400, "unsupported_feature"),
        lattice_serve: (400, "unsupported_feature"),
        divergence_reason: None,
    },
    ParityCase {
        name: "malformed_json_body",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(r#"{"model":"test-model","messages":"#),
        lattice: (400, "invalid_request_body"),
        lattice_serve: (400, "invalid_request_body"),
        divergence_reason: None,
    },
    ParityCase {
        name: "max_tokens_over_cap_reject_vs_clamp",
        method: "POST",
        path: "/v1/chat/completions",
        // Both servers are configured (in each binary's own oneshot test
        // harness) with a small cap/context window; this body's max_tokens
        // exceeds it. `lattice.rs` rejects before ever touching the
        // model/worker; `lattice_serve.rs` clamps to the model's context
        // window in `build_cfg` and proceeds past validation entirely
        // (#745's triaged scope, kept deliberately unnified by
        // `reject_zero_max_tokens`'s doc comment). `lattice_serve`'s
        // expected (500, "internal_error") here is a harness artifact, not
        // real-server behavior: the router-level test fixture has no live
        // worker behind its job queue (matching this test module's existing
        // `test_app_state()` convention -- HTTP-level 400 tests only, no
        // GPU/model load), so a request that clears validation and reaches
        // `jobs.send(..)` fails there instead. The signal this case actually
        // proves is "not a 400 at the validation cascade" -- clamp-not-reject
        // -- which the diverging (500 vs 400) outcome demonstrates without
        // needing a real model.
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":999999}"#,
        ),
        lattice: (400, "max_tokens_exceeds_limit"),
        lattice_serve: (500, "internal_error"),
        divergence_reason: Some(
            "lattice.rs rejects max_tokens above its server cap at validation time; \
             lattice_serve.rs clamps the resolved value to the model's context \
             window and proceeds past validation instead of rejecting -- an \
             intentional per-binary policy difference, not drift (see \
             reject_zero_max_tokens's doc comment). The lattice_serve 500 here is \
             this router-level fixture's no-live-worker harness artifact once past \
             validation, not the divergence itself.",
        ),
    },
    ParityCase {
        name: "get_root_route_exposed",
        // ADR-080 C2 round 2, codex round-2 medium finding #2, mutation-
        // proven: every case above targets only `POST
        // /v1/chat/completions`, so removing `lattice.rs`'s `.route("/",
        // get(root))` entirely left the parity test green -- route exposure
        // itself was never actually checked. Both binaries must expose
        // `GET /` and return the shared `root_body()` shape (200; no error
        // envelope to check, so no error `code` is meaningful here).
        method: "GET",
        path: "/",
        body: CaseBody::Fixed(""),
        lattice: (200, ""),
        lattice_serve: (200, ""),
        divergence_reason: None,
    },
    ParityCase {
        name: "oversized_body_over_limit",
        // ADR-080 C2 round 2, codex round-2 medium finding #2, mutation-
        // proven: no case above sent a body over
        // `REQUEST_BODY_LIMIT_BYTES`, so restoring `lattice_serve.rs`'s old
        // 400/`invalid_request` oversized-body mapping (instead of the
        // current 413/`request_body_too_large`) also left the parity test
        // green. Both binaries enforce the same 1 MiB cap today (`lattice.rs`
        // via `DefaultBodyLimit`, `lattice_serve.rs` via a manual
        // `to_bytes(.., LIMIT)` check) and must report it identically.
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Oversized,
        lattice: (413, "request_body_too_large"),
        lattice_serve: (413, "request_body_too_large"),
        divergence_reason: None,
    },
];

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
    fn root_body_shape() {
        let body = root_body();
        assert_eq!(body["name"], "lattice");
        assert_eq!(body["object"], "engine");
        assert_eq!(
            body["endpoints"],
            serde_json::json!(["/v1/chat/completions", "/v1/models", "/health"])
        );
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

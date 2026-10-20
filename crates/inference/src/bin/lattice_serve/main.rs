//! lattice_serve - OpenAI-compatible HTTP serving endpoint for Lattice. See [docs/capability-matrix.md](../../../../docs/capability-matrix.md).
//!
//! Exposes the Metal GPU engine over the same `/v1/chat/completions` API that
//! ollama, llama.cpp's server, and most LLM benchmark harnesses already speak,
//! so any OpenAI-compatible client can point at lattice with zero adapter code.
//! A Gemma 4 E2B safetensors text checkpoint is served on the CPU through the
//! same worker queue; the family and backend come from the route table
//! `lattice serve` uses (`lattice_inference::serve::route`).
//!
//! # Usage
//!
//! ```text
//! lattice_serve --model qwen3.5-0.8b               # resolves from ~/.lattice/models
//! lattice_serve --model ~/.lattice/models/qwen3.6-27b-q4 --port 11435
//! lattice_serve --model qwen3.5-0.8b --embedding-model bge-small-en-v1.5
//! ```
//!
//! Then point any OpenAI client at `http://127.0.0.1:11435/v1`:
//!
//! ```text
//! curl http://127.0.0.1:11435/v1/chat/completions -H 'content-type: application/json' \
//!   -d '{"model":"lattice","messages":[{"role":"user","content":"hi"}],"stream":true}'
//! curl http://127.0.0.1:11435/v1/embeddings -H 'content-type: application/json' \
//!   -d '{"input":["hello world","a second passage"]}'
//! ```
//!
//! # Endpoints
//!
//! - `POST /v1/chat/completions` — streaming (SSE) and non-streaming, OpenAI shape
//! - `POST /v1/embeddings`       — OpenAI shape (issue #584); answers 503
//!   `embedding_model_not_loaded` unless `--embedding-model` was passed at
//!   startup (see `EmbeddingState`, `AppState::embedding`)
//! - `GET  /v1/models`           — advertises the single loaded chat model
//! - `GET  /health`              — liveness probe (`ok`)
//! - `GET  /metrics`             — Prometheus text-format metrics (issue #583):
//!   request count + latency histogram by route/status, prompt/completion
//!   token counters, error count by code, queue-depth/in-flight gauge, all
//!   labeled with the loaded model id
//!
//! # Design
//!
//! The selected provider builds its thread-affine model state on one dedicated
//! worker thread for the whole process lifetime, owned by
//! `lattice_inference::serve::metal_worker::MetalWorker`. The async
//! axum handlers never touch Metal directly: each request is submitted via
//! `ServingHandle::submit_standalone`, which ships a job (messages + sampling
//! config + a cancellation watch) to the worker over a tokio mpsc and
//! returns a `WorkerEvent` receiver; the worker drives cache-aware
//! generation, forwarding each token delta back as `WorkerEvent::Delta`.
//! Generation is therefore serialized — correct for a single-GPU local engine
//! (the same default ollama uses). The ChatML template and `<|im_end|>` stop
//! handling are reused verbatim from the engine; this binary only translates the
//! OpenAI wire format on either side.
//!
//! `POST /v1/embeddings` (issue #584) does not go through that worker at
//! all: it is served by an independently loaded, optional `BertModel`
//! (`crate::model::bert`), which is CPU-only and has no `!Send` state of its
//! own, so its `encode_batch` call runs via `tokio::task::spawn_blocking`
//! instead of a dedicated owner thread. This is a deliberate scope choice,
//! not an oversight: `lattice-embed`'s `EmbeddingService` trait is the
//! natural-looking integration point, but `lattice-embed` already depends on
//! `lattice-inference` (for `BertModel`/`QwenModel`), so a dependency edge
//! the other way is a cyclic package dependency Cargo refuses to build.
//! `BertModel` is this crate's own, already-published embedding primitive,
//! so using it directly costs no new dependency edge and no default-build
//! weight. It has no per-model pooling table (BGE-family checkpoints need
//! CLS pooling, not `BertModel`'s mean-pooling default) — pass
//! `--embedding-pooling cls` when serving one.

#![allow(clippy::field_reassign_with_default)]

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("lattice_serve requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        if let Err(e) = imp::run() {
            eprintln!("lattice_serve: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod imp {
    use axum::{
        Json, Router,
        body::{Body, to_bytes},
        extract::State,
        http::{HeaderMap, StatusCode},
        response::{
            IntoResponse, Response,
            sse::{Event, KeepAlive, Sse},
        },
        routing::{get, post},
    };
    use lattice_inference::GenerateOutput;
    use lattice_inference::grammar::{GrammarEngine, GrammarSpec};
    use lattice_inference::serve::contract::{
        ChatRequest as ChatReq, GenerationDefaults, is_message_flood_error, message_flood_text,
    };
    #[cfg(test)]
    use lattice_inference::serve::contract::{
        ContentPart as Part, ImageUrl, Message as InMsg, MessageContent, ServeProfile,
        ValidatedChatRequest, normalize_messages, normalize_request,
    };
    use lattice_inference::serve::embeddings::{
        TextEmbeddingsRequest, build_embeddings_response, check_embeddings_item_fits_window,
        check_embeddings_model, check_embeddings_total_tokens, check_embeddings_unsupported_fields,
        parse_embeddings_input,
    };
    use lattice_inference::serve::into_engine_chat_messages;
    #[cfg(all(test, feature = "test-utils"))]
    use lattice_inference::serve::metal_worker::MetalWorkerClient;
    use lattice_inference::serve::metal_worker::WorkerEvent;
    #[cfg(all(test, feature = "metal-gpu", feature = "test-utils"))]
    use lattice_inference::serve::metal_worker::{ContextWindowPolicy, MetalWorker};
    use lattice_inference::serve::metrics::ServeMetrics;
    use lattice_inference::serving_provider::{
        self, ServingEntry, StandaloneLoadOptions, StandaloneOptionPresence,
    };
    #[cfg(test)]
    use lattice_inference::serving_provider::{
        FALLBACK_MODEL_MAX_CONTEXT, model_context_from_config,
    };
    use lattice_inference::{BertModel, BertPooling};
    use serde_json::{Value, json};
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};
    use tokio::sync::Semaphore;
    #[path = "state.rs"]
    mod state;
    pub(crate) use state::{AppState, IntoServingHost};
    /// Only used by the test module's raw job-channel helpers
    /// (`mpsc::UnboundedReceiver<WorkerMessage>` etc. -- see
    /// `lattice_inference::serve::metal_worker`'s `test-utils`-gated
    /// surface); production state holds a `ServingHandle`.
    #[cfg(all(test, feature = "metal-gpu", feature = "test-utils"))]
    use tokio::sync::mpsc;

    /// Normalizes [`WorkerEvent::Cancelled`] (the job was skipped at dequeue
    /// time: the client's `cancel` watch flag was already `true`, or this
    /// job's own event receiver was already closed) into an ordinary, empty
    /// [`WorkerEvent::Complete`] -- zero tokens, `stopped: false`
    /// ("length"-shaped termination). This is the exact observable shape
    /// `lattice.rs`'s prior `MetalJob` dequeue-cancellation path already
    /// produced (an empty interrupted `GenerateOutput` flowing through the
    /// normal completion path); #832 unifies this binary onto the same
    /// shape instead of its own prior silent "zero events, channel closes"
    /// behavior.
    ///
    /// In practice this is defensive only: the `cancel` guard paired with
    /// any given job is owned by the very same async task that is awaiting
    /// this job's events (moved into the SSE stream state for the streaming
    /// path, held in local scope for non-streaming), so a live HTTP request
    /// can never actually observe `Cancelled` -- if the guard had dropped,
    /// axum would have already dropped the observing task along with it.
    /// Exists so every match below stays exhaustive and well-defined even
    /// if that invariant ever changes, exactly like the pre-existing
    /// `WorkerEvent::Rejected`-inside-`Phase::Body` defensive arm this refactor also
    /// preserves. Leaves every other event unchanged.
    fn normalize_cancelled(ev: WorkerEvent) -> WorkerEvent {
        match ev {
            WorkerEvent::Cancelled => WorkerEvent::Complete(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: 0,
                generated_tokens: 0,
                stopped: false,
                stop_reason: Some(lattice_inference::StopReason::Interrupt),
                token_logprobs: vec![],
            }),
            other => other,
        }
    }

    /// The loaded embedding model and its identity, independent of the
    /// chat model this binary always loads. `BertModel` is CPU-only (no
    /// `metal-gpu` dependency of its own -- see `crate::model::bert`), so it
    /// is loaded directly on the calling thread at startup and its
    /// `encode_batch` calls run via `spawn_blocking` rather than through the
    /// shared Metal worker, which owns only the `!Send` chat model state.
    pub struct EmbeddingState {
        model: BertModel,
        model_id: Arc<str>,
        /// Bounded concurrent-job admission for `encode_batch`'s
        /// `spawn_blocking` calls (issue #584 hardening), mirroring
        /// `MetalWorkerClient`'s admission `Semaphore` (`AppState::max_pending`,
        /// issue #932): each outstanding job can already allocate up to
        /// [`lattice_inference::serve::embeddings::MAX_EMBEDDINGS_TOTAL_TOKENS`]
        /// worth of activation buffers, so total memory pressure across
        /// concurrent requests is bounded by (this cap) x (that per-request
        /// budget) rather than growing with however many requests arrive at
        /// once. Fixed at [`EMBEDDING_MAX_CONCURRENT_JOBS`] rather than a
        /// CLI flag like `--max-pending`: CPU-bound batched matmuls have no
        /// hardware queue depth to model, this cap exists purely to bound
        /// aggregate memory.
        admission: Arc<Semaphore>,
    }

    fn load_embedding_state(
        embedding_model_dir: &std::path::Path,
        pooling_override: Option<&str>,
    ) -> Result<Arc<EmbeddingState>, String> {
        if !embedding_model_dir.exists() {
            return Err(format!(
                "embedding model directory not found: {}",
                embedding_model_dir.display()
            ));
        }
        eprintln!(
            "[lattice_serve] loading embedding model from {} ...",
            embedding_model_dir.display()
        );
        let mut model = BertModel::from_directory(embedding_model_dir).map_err(|e| {
            format!(
                "embedding model load failed ({}): {e}",
                embedding_model_dir.display()
            )
        })?;
        let embedding_model_id: Arc<str> = embedding_model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("embedding")
            .into();
        // Auto-select the same default `--embedding-pooling` would
        // otherwise force the operator to pass explicitly. See
        // `lattice_inference::pool::default_bert_pooling_for_model_name`
        // for the rule and its agreement test against
        // `lattice-embed`'s `EmbeddingModel::bert_pooling` table.
        model.set_pooling(
            lattice_inference::pool::default_bert_pooling_for_model_name(&embedding_model_id),
        );
        match pooling_override {
            None => {}
            Some("mean") => model.set_pooling(BertPooling::Mean),
            Some("cls") => model.set_pooling(BertPooling::CLS),
            Some(other) => {
                return Err(format!(
                    "invalid --embedding-pooling '{other}'; expected 'mean' or 'cls'"
                ));
            }
        }
        eprintln!(
            "[lattice_serve] embedding model '{embedding_model_id}' ready (dimensions={})",
            model.dimensions()
        );
        Ok(Arc::new(EmbeddingState {
            model,
            model_id: embedding_model_id,
            admission: Arc::new(Semaphore::new(EMBEDDING_MAX_CONCURRENT_JOBS)),
        }))
    }

    /// See [`EmbeddingState::admission`].
    const EMBEDDING_MAX_CONCURRENT_JOBS: usize = 4;

    /// Acquires one embedding-job admission slot, mirroring
    /// the serving worker's admission check for
    /// [`EmbeddingState::admission`]. Synchronous and non-blocking: a full
    /// admission cap is rejected immediately, before any tokenization or
    /// `encode_batch` work runs, exactly like the chat path's admission
    /// check.
    ///
    /// # Errors
    ///
    /// Returns [`lattice_inference::serve::ApiError::ServiceUnavailable`]
    /// (`server_busy`) when [`EMBEDDING_MAX_CONCURRENT_JOBS`] jobs are
    /// already outstanding.
    fn try_acquire_embedding_slot(
        admission: &Arc<Semaphore>,
    ) -> Result<tokio::sync::OwnedSemaphorePermit, lattice_inference::serve::ApiError> {
        admission.clone().try_acquire_owned().map_err(|_| {
            lattice_inference::serve::ApiError::ServiceUnavailable {
                message: "too many outstanding /v1/embeddings requests; retry shortly".to_string(),
            }
        })
    }

    /// Runs one `encode_batch` job on the blocking thread pool with `permit`
    /// moved into the closure, so the admission slot releases when the job
    /// finishes rather than when the caller's `.await` is dropped. A
    /// `spawn_blocking` task already under way is not cancelled by dropping
    /// its `JoinHandle` (e.g. a disconnected client dropping the request
    /// future), so a permit held only in the caller's stack frame would
    /// release on disconnect while the job kept running -- letting repeated
    /// disconnects admit unbounded concurrent jobs past the cap.
    async fn run_embedding_encode(
        permit: tokio::sync::OwnedSemaphorePermit,
        work: impl FnOnce() -> Result<Vec<Vec<f32>>, lattice_inference::InferenceError> + Send + 'static,
    ) -> Result<Result<Vec<Vec<f32>>, lattice_inference::InferenceError>, tokio::task::JoinError>
    {
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            work()
        })
        .await
    }

    // ─── OpenAI request shapes ───────────────────────────────────────────────

    /// Request body cap applied before any JSON parsing (serve DoS-hardening
    /// rule: every size field clamps before allocation). ADR-080 C2 (#782):
    /// `lattice_inference::serve::REQUEST_BODY_LIMIT_BYTES` is the single
    /// shared constant now; both binaries previously carried this exact
    /// value independently.
    use lattice_inference::serve::REQUEST_BODY_LIMIT_BYTES;
    /// Maximum number of content parts accepted per message (#649). Enforced
    /// by `validate_content_part_limits` before the typed `ChatReq` is parsed.
    const MAX_CONTENT_PARTS_PER_MESSAGE: usize = 64;
    /// Maximum byte length of a single content string / part payload (#649).
    const MAX_CONTENT_PART_BYTES: usize = 65_536;
    /// Capability-error wording pinned by text-only backend tests. Image
    /// content is admitted only when the loaded worker advertises a complete
    /// vision configuration.
    #[cfg(test)]
    const IMAGE_REQUIRES_VISION_MESSAGE: &str = "image input requires a vision-capable model";

    /// A request-validation failure that must surface as HTTP 400 (#641,
    /// #649). Fail-closed: unknown roles and unsupported content parts are
    /// never coerced or dropped, they always produce one of these.
    ///
    /// `code` carries the OpenAI-style error code (ADR-080 C2): previously
    /// this variant was message-only, so
    /// every validation failure collapsed to `err_response`'s generic
    /// `"invalid_request"` fallback regardless of what specifically went
    /// wrong -- `lattice.rs`'s `ApiError::BadRequest` already differentiated
    /// (`invalid_role`, `unsupported_feature`, `invalid_messages`, ...) for
    /// the equivalent checks. Codes below are chosen to match `lattice.rs`'s
    /// codes for the SAME violation wherever an equivalent check exists on
    /// both binaries; a handful of checks are lattice_serve-only (the #649
    /// content-part size/count DoS hardening has no lattice.rs analog) and
    /// get a new, stable, lattice_serve-only code instead of an invented
    /// false match.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum RequestError {
        BadRequest {
            message: String,
            code: &'static str,
        },
        /// A server-side malfunction discovered during admission (e.g. the
        /// grammar cache's own lock/slot
        /// invariants broke) -- never the caller's fault, so it must not be
        /// reported as the 400 every other `RequestError` is.
        ServerError {
            message: String,
            code: &'static str,
        },
    }

    impl RequestError {
        fn bad_request(message: impl Into<String>, code: &'static str) -> Self {
            Self::BadRequest {
                message: message.into(),
                code,
            }
        }

        fn server_error(message: impl Into<String>, code: &'static str) -> Self {
            Self::ServerError {
                message: message.into(),
                code,
            }
        }

        fn message(&self) -> &str {
            match self {
                Self::BadRequest { message, .. } | Self::ServerError { message, .. } => message,
            }
        }

        fn code(&self) -> &'static str {
            match self {
                Self::BadRequest { code, .. } | Self::ServerError { code, .. } => code,
            }
        }

        fn status(&self) -> StatusCode {
            match self {
                Self::BadRequest { .. } => StatusCode::BAD_REQUEST,
                Self::ServerError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            }
        }
    }

    // ─── structured-output v0 (design note: strict JSON Schema, non-streaming only) ──

    /// Keywords the v0 structured-output schema subset does not admit.
    /// Checked before the narrower per-type validation in
    /// `admit_v0_schema_at` so every rejection names the exact offending
    /// keyword instead of falling through to a generic message.
    /// `enum`/`const` are deferred per the design note's sign-off ruling
    /// (Q2): admitted only once an independent evaluator and fuzz corpus
    /// prove their type/escaping semantics, which this patch does not add.
    const V0_REJECTED_KEYWORDS: &[&str] = &[
        "enum",
        "const",
        "pattern",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "$ref",
        "$defs",
        "definitions",
        "anyOf",
        "oneOf",
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "format",
        "prefixItems",
        "minItems",
        "maxItems",
        "uniqueItems",
        "contains",
    ];

    /// A successful admission: the caller's schema, admitted into the v0
    /// subset (closed objects, uniform arrays, primitive types, arbitrary
    /// nesting of the above -- see the structured-output v0 design note).
    /// Rejection is a `RequestError` with code `unsupported_strict_schema`,
    /// naming the offending keyword/structural problem, and happens before
    /// any worker job is submitted.
    fn admit_v0_schema(schema: &Value) -> Result<(), String> {
        admit_v0_schema_at(schema, "schema")
    }

    fn admit_v0_schema_at(node: &Value, path: &str) -> Result<(), String> {
        let Some(obj) = node.as_object() else {
            return Err(format!("{path}: schema must be a JSON object"));
        };

        for kw in V0_REJECTED_KEYWORDS {
            if obj.contains_key(*kw) {
                return Err(format!(
                    "{path}: keyword '{kw}' is not supported by the v0 structured-output schema subset"
                ));
            }
        }

        let Some(type_value) = obj.get("type") else {
            return Err(format!("{path}: 'type' is required"));
        };
        let Some(ty) = type_value.as_str() else {
            return Err(format!(
                "{path}: 'type' must be a single string, not a union or other value"
            ));
        };

        match ty {
            "object" => {
                const ALLOWED: &[&str] =
                    &["type", "properties", "required", "additionalProperties"];
                if let Some(unknown) = obj.keys().find(|k| !ALLOWED.contains(&k.as_str())) {
                    return Err(format!(
                        "{path}: unknown keyword '{unknown}' is not supported for an object schema"
                    ));
                }
                if !matches!(obj.get("additionalProperties"), Some(Value::Bool(false))) {
                    return Err(format!(
                        "{path}: 'additionalProperties' must be present and set to exactly 'false'"
                    ));
                }
                let Some(props) = obj.get("properties").and_then(Value::as_object) else {
                    return Err(format!(
                        "{path}: 'properties' is required and must be an object"
                    ));
                };
                let Some(required) = obj.get("required").and_then(Value::as_array) else {
                    return Err(format!(
                        "{path}: 'required' is required and must be an array"
                    ));
                };
                let mut required_names = std::collections::BTreeSet::new();
                for entry in required {
                    let Some(name) = entry.as_str() else {
                        return Err(format!("{path}: 'required' entries must be strings"));
                    };
                    required_names.insert(name);
                }
                let prop_names: std::collections::BTreeSet<&str> =
                    props.keys().map(String::as_str).collect();
                if required_names != prop_names {
                    return Err(format!(
                        "{path}: 'required' must list every property exactly once (closed strict object)"
                    ));
                }
                for (name, prop_schema) in props {
                    admit_v0_schema_at(prop_schema, &format!("{path}.{name}"))?;
                }
                Ok(())
            }
            "array" => {
                const ALLOWED: &[&str] = &["type", "items"];
                if let Some(unknown) = obj.keys().find(|k| !ALLOWED.contains(&k.as_str())) {
                    return Err(format!(
                        "{path}: unknown keyword '{unknown}' is not supported for an array schema"
                    ));
                }
                let Some(items) = obj.get("items") else {
                    return Err(format!("{path}: 'items' is required for an array schema"));
                };
                admit_v0_schema_at(items, &format!("{path}[]"))
            }
            "string" | "number" | "integer" | "boolean" | "null" => {
                const ALLOWED: &[&str] = &["type"];
                if let Some(unknown) = obj.keys().find(|k| !ALLOWED.contains(&k.as_str())) {
                    return Err(format!(
                        "{path}: unknown keyword '{unknown}' is not supported for type '{ty}'"
                    ));
                }
                Ok(())
            }
            other => Err(format!("{path}: unsupported type '{other}'")),
        }
    }

    /// Independent post-generation conformance check (structured-output v0
    /// design note's "independent validation" requirement): validates the
    /// raw generated JSON text against an admitted v0 schema by walking
    /// `content`'s bytes directly, in lockstep with the schema, WITHOUT
    /// reusing any of `lattice_inference::grammar::json_schema`'s compiler
    /// code AND without ever going through `serde_json::Value` -- so a bug
    /// shared between admission-time compilation and this check cannot
    /// silently agree on a wrong answer.
    ///
    /// A correctness note on `serde_json::Value`: (without the
    /// `arbitrary_precision` feature, which the workspace does not enable)
    /// parses any JSON number into `u64`/`i64`/`f64` before this function
    /// would ever see it, so a grammar-conforming integer literal outside
    /// `u64`/`i64` range (e.g. `18446744073709551616`) got silently rounded
    /// to the nearest representable `f64`, and `is_i64()`/`is_u64()` then
    /// rejected it -- a false `validation_failed` 500 for output the
    /// compiler itself admitted. Preferred this raw-text walker over
    /// turning on `arbitrary_precision`: that feature changes every
    /// `Value` built anywhere in this binary, not just here, and isn't
    /// already enabled anywhere in the workspace.
    ///
    /// Assumes `schema` already passed `admit_v0_schema` -- only the v0
    /// subset's shapes are handled.
    fn v0_validate_json(content: &str, schema: &Value) -> bool {
        let mut pos = 0usize;
        if !v0_validate_at(content, &mut pos, schema) {
            return false;
        }
        v0_skip_ws(content, &mut pos);
        pos == content.len()
    }

    fn v0_skip_ws(s: &str, pos: &mut usize) {
        let b = s.as_bytes();
        while matches!(b.get(*pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            *pos += 1;
        }
    }

    fn v0_validate_at(s: &str, pos: &mut usize, schema: &Value) -> bool {
        v0_skip_ws(s, pos);
        let Some(obj) = schema.as_object() else {
            return false;
        };
        let Some(ty) = obj.get("type").and_then(Value::as_str) else {
            return false;
        };
        match ty {
            "object" => v0_validate_object(s, pos, obj),
            "array" => v0_validate_array(s, pos, obj),
            "string" => v0_parse_string(s, pos).is_some(),
            "boolean" => v0_parse_literal(s, pos, "true") || v0_parse_literal(s, pos, "false"),
            "null" => v0_parse_literal(s, pos, "null"),
            "number" => v0_parse_number_lexeme(s, pos).is_some(),
            "integer" => {
                v0_parse_number_lexeme(s, pos).is_some_and(|lexeme| v0_is_integer_lexeme(&lexeme))
            }
            _ => false,
        }
    }

    fn v0_parse_literal(s: &str, pos: &mut usize, lit: &str) -> bool {
        if s.as_bytes().get(*pos..*pos + lit.len()) == Some(lit.as_bytes()) {
            *pos += lit.len();
            true
        } else {
            false
        }
    }

    /// Parses a JSON string starting at `s[*pos]` (which must be `"`),
    /// returning its unescaped content and advancing `*pos` past the
    /// closing quote.
    fn v0_parse_string(s: &str, pos: &mut usize) -> Option<String> {
        let b = s.as_bytes();
        if b.get(*pos) != Some(&b'"') {
            return None;
        }
        *pos += 1;
        let mut out = String::new();
        loop {
            match b.get(*pos)? {
                b'"' => {
                    *pos += 1;
                    return Some(out);
                }
                b'\\' => {
                    *pos += 1;
                    match *b.get(*pos)? {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hex = s.get(*pos + 1..*pos + 5)?;
                            let code = u32::from_str_radix(hex, 16).ok()?;
                            *pos += 5; // consume 'u' + 4 hex digits
                            if (0xD800..=0xDBFF).contains(&code) {
                                // High surrogate: the compiled grammar
                                // (json_schema.rs escape_id) admits any
                                // four-hex `\uXXXX` independently of
                                // pairing, so a lone half must still
                                // validate here (validator ⊇ grammar
                                // language). Combine with an immediately
                                // following low-surrogate escape when
                                // present; a lone half is represented as
                                // U+FFFD since `char`/`String` cannot hold
                                // an unpaired surrogate scalar.
                                if s.as_bytes().get(*pos) == Some(&b'\\')
                                    && s.as_bytes().get(*pos + 1) == Some(&b'u')
                                    && let Some(low) = s
                                        .get(*pos + 2..*pos + 6)
                                        .and_then(|h| u32::from_str_radix(h, 16).ok())
                                        .filter(|low| (0xDC00..=0xDFFF).contains(low))
                                {
                                    let combined =
                                        0x10000 + (code - 0xD800) * 0x400 + (low - 0xDC00);
                                    out.push(char::from_u32(combined)?);
                                    *pos += 6;
                                    continue;
                                }
                                out.push('\u{FFFD}');
                                continue;
                            }
                            if (0xDC00..=0xDFFF).contains(&code) {
                                // Lone low surrogate: same grammar-parity
                                // rationale as the high-surrogate case
                                // above.
                                out.push('\u{FFFD}');
                                continue;
                            }
                            out.push(char::from_u32(code)?);
                            continue;
                        }
                        _ => return None,
                    }
                    *pos += 1;
                }
                _ => {
                    // Not a quote or a backslash: copy one UTF-8 scalar
                    // verbatim, O(1) since `s[*pos..]` is a valid-UTF-8
                    // slice already and `chars().next()` decodes lazily.
                    // RFC 8259 §7 requires U+0000-U+001F to be escaped;
                    // the compiled grammar's string-body alternatives
                    // (json_schema.rs, unescaped-ASCII range starts at
                    // 0x20) reject them unescaped too, so an unescaped
                    // raw control scalar here must fail the same way.
                    let ch = s[*pos..].chars().next()?;
                    if (ch as u32) < 0x20 {
                        return None;
                    }
                    out.push(ch);
                    *pos += ch.len_utf8();
                }
            }
        }
    }

    /// Consumes one JSON number token starting at `s[*pos]` per the
    /// standard JSON number grammar (`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`)
    /// and returns its exact source lexeme, unconverted -- the whole point
    /// of this function existing instead of `serde_json::Value::as_f64`.
    fn v0_parse_number_lexeme(s: &str, pos: &mut usize) -> Option<String> {
        let b = s.as_bytes();
        let start = *pos;
        if b.get(*pos) == Some(&b'-') {
            *pos += 1;
        }
        match b.get(*pos) {
            Some(b'0') => *pos += 1,
            Some(c) if c.is_ascii_digit() => {
                while matches!(b.get(*pos), Some(c) if c.is_ascii_digit()) {
                    *pos += 1;
                }
            }
            _ => {
                *pos = start;
                return None;
            }
        }
        if b.get(*pos) == Some(&b'.') {
            let frac_start = *pos + 1;
            let mut probe = frac_start;
            while matches!(b.get(probe), Some(c) if c.is_ascii_digit()) {
                probe += 1;
            }
            if probe > frac_start {
                *pos = probe;
            }
        }
        if matches!(b.get(*pos), Some(b'e' | b'E')) {
            let mut probe = *pos + 1;
            if matches!(b.get(probe), Some(b'+' | b'-')) {
                probe += 1;
            }
            let exp_digits_start = probe;
            while matches!(b.get(probe), Some(c) if c.is_ascii_digit()) {
                probe += 1;
            }
            if probe > exp_digits_start {
                *pos = probe;
            }
        }
        Some(s[start..*pos].to_string())
    }

    /// JSON Schema's `integer` is any number with a zero fractional part
    /// (2020-12 §6.1.1) -- `1.0` is a valid integer, `1.5` is not. The v0
    /// compiler's own `json_integer` grammar production never emits a
    /// fraction or exponent at all, so this is stricter than necessary for
    /// grammar-conformant output and exists to independently reject a
    /// non-conformant generation on its own terms, not the compiler's.
    fn v0_is_integer_lexeme(lexeme: &str) -> bool {
        if lexeme.contains(['e', 'E']) {
            return false;
        }
        match lexeme.split_once('.') {
            None => true,
            Some((_, frac)) => !frac.is_empty() && frac.bytes().all(|c| c == b'0'),
        }
    }

    fn v0_validate_object(s: &str, pos: &mut usize, obj: &serde_json::Map<String, Value>) -> bool {
        let Some(props) = obj.get("properties").and_then(Value::as_object) else {
            return false;
        };
        if s.as_bytes().get(*pos) != Some(&b'{') {
            return false;
        }
        *pos += 1;
        v0_skip_ws(s, pos);
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        if s.as_bytes().get(*pos) == Some(&b'}') {
            *pos += 1;
            return props.is_empty();
        }
        loop {
            v0_skip_ws(s, pos);
            let Some(key) = v0_parse_string(s, pos) else {
                return false;
            };
            v0_skip_ws(s, pos);
            if s.as_bytes().get(*pos) != Some(&b':') {
                return false;
            }
            *pos += 1;
            let Some(prop_schema) = props.get(&key) else {
                return false;
            };
            if !seen.insert(key) {
                return false;
            }
            if !v0_validate_at(s, pos, prop_schema) {
                return false;
            }
            v0_skip_ws(s, pos);
            match s.as_bytes().get(*pos) {
                Some(b',') => *pos += 1,
                Some(b'}') => {
                    *pos += 1;
                    break;
                }
                _ => return false,
            }
        }
        seen.len() == props.len()
    }

    fn v0_validate_array(s: &str, pos: &mut usize, obj: &serde_json::Map<String, Value>) -> bool {
        let Some(items_schema) = obj.get("items") else {
            return false;
        };
        if s.as_bytes().get(*pos) != Some(&b'[') {
            return false;
        }
        *pos += 1;
        v0_skip_ws(s, pos);
        if s.as_bytes().get(*pos) == Some(&b']') {
            *pos += 1;
            return true;
        }
        loop {
            if !v0_validate_at(s, pos, items_schema) {
                return false;
            }
            v0_skip_ws(s, pos);
            match s.as_bytes().get(*pos) {
                Some(b',') => *pos += 1,
                Some(b']') => {
                    *pos += 1;
                    break;
                }
                _ => return false,
            }
        }
        true
    }

    /// Serializes `schema` (already admitted by `admit_v0_schema`) into a
    /// key that is identical for two schemas differing only in object
    /// property order. Deliberately does NOT rely on `serde_json::Value`'s
    /// own `Map` ordering: whether that `Map` is `BTreeMap`-backed (sorted)
    /// or insertion-order-backed depends on whether the `preserve_order`
    /// Cargo feature is unified into the build anywhere in the dependency
    /// graph -- verified empirically for the current lockfile (no crate
    /// enables it, so `serde_json::to_string` already sorts keys), but that
    /// is a whole-workspace build property that a future unrelated
    /// dependency bump could silently flip. This walks the value itself and
    /// sorts object keys explicitly, so the cache key's determinism does not
    /// depend on that build-wide feature-unification detail.
    fn canonical_schema_key(schema: &Value) -> String {
        fn write_canonical(v: &Value, out: &mut String) {
            match v {
                Value::Null => out.push_str("null"),
                Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
                Value::Number(n) => out.push_str(&n.to_string()),
                Value::String(s) => {
                    out.push_str(&serde_json::to_string(s).unwrap_or_default());
                }
                Value::Array(items) => {
                    out.push('[');
                    for (i, item) in items.iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        write_canonical(item, out);
                    }
                    out.push(']');
                }
                Value::Object(map) => {
                    let mut keys: Vec<&String> = map.keys().collect();
                    keys.sort();
                    out.push('{');
                    for (i, k) in keys.iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        out.push_str(&serde_json::to_string(k).unwrap_or_default());
                        out.push(':');
                        write_canonical(&map[*k], out);
                    }
                    out.push('}');
                }
            }
        }
        let mut out = String::new();
        write_canonical(schema, &mut out);
        out
    }

    /// Cache key for [`GrammarCache::get_or_compile`].
    ///
    /// The `GrammarEngine` a lookup resolves to was compiled from *two*
    /// inputs -- the admitted schema (`canonical_schema_key`) and the
    /// vocabulary handed to `GrammarEngine::new` -- but only the schema used
    /// to be reflected in the key. Keyed on the schema alone, a cache cannot
    /// tell two different vocabularies compiled against the same schema
    /// apart: the second vocabulary's lookup would hit the first
    /// vocabulary's entry and silently serve a grammar built for the wrong
    /// tokenizer.
    ///
    /// This key holds a clone of the `vocab_bytes` `Arc` itself, not just its
    /// address: an address alone names an allocation only
    /// for as long as that exact allocation is alive, and the whole point of
    /// a *stale* cache entry is that its vocabulary may no longer be alive
    /// anywhere else -- a later, unrelated `vocab_bytes` allocation (a
    /// reloaded model, a swapped tokenizer) can then legitimately land at the
    /// freed address and collide with the stale entry's key. Storing the
    /// `Arc` inside the key -- used as a `HashMap` key via [`PartialEq`]/
    /// [`Eq`]/[`Hash`] impls below keyed on [`Arc::ptr_eq`] and the pointer
    /// address, never on vocabulary *contents* -- keeps that allocation
    /// alive for as long as the entry (or in-flight compile slot) sits in
    /// the cache, so the address it names cannot be freed and reused while
    /// a stale entry could still alias it. Comparing/hashing by pointer
    /// rather than by content keeps this the same O(1)-per-request cost as
    /// before: no hashing of a (possibly ~250K-entry) vocabulary on every
    /// admitted request.
    #[derive(Clone)]
    struct GrammarCacheKey {
        vocab_bytes: Arc<Vec<Vec<u8>>>,
        schema_key: String,
    }

    impl PartialEq for GrammarCacheKey {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.vocab_bytes, &other.vocab_bytes)
                && self.schema_key == other.schema_key
        }
    }

    impl Eq for GrammarCacheKey {}

    impl std::hash::Hash for GrammarCacheKey {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            (Arc::as_ptr(&self.vocab_bytes) as *const () as usize).hash(state);
            self.schema_key.hash(state);
        }
    }

    fn grammar_cache_key(vocab_bytes: &Arc<Vec<Vec<u8>>>, schema: &Value) -> GrammarCacheKey {
        GrammarCacheKey {
            vocab_bytes: Arc::clone(vocab_bytes),
            schema_key: canonical_schema_key(schema),
        }
    }

    /// Stage 2 design note: shipped default capacity for the LRU compiled-
    /// grammar cache.
    const GRAMMAR_CACHE_CAPACITY: usize = 32;

    /// [`GrammarCache::get_or_compile`]'s error shape distinguishes a legitimate schema-compile
    /// failure -- the caller's fault, an ordinary 400
    /// `unsupported_strict_schema`, exactly the behavior before this change
    /// -- from the cache's own internal machinery breaking (a poisoned
    /// lock, a vanished compiling-slot entry, or the compile closure
    /// panicking instead of returning `Err`), which is never the caller's
    /// fault and must surface as a bounded 500 instead of a panic that
    /// tears down the request or a condvar nobody ever notifies again.
    #[derive(Debug, Clone)]
    enum CacheError {
        Compile(String),
        Internal(String),
    }

    /// The state shared by a single in-flight compile: waiters block on
    /// `cv` until `result` is populated by the one thread that actually
    /// calls the compile closure (structured-output v0 design note,
    /// stage-2 single-flight requirement).
    struct CompileSlot {
        result: Mutex<Option<Result<Arc<GrammarEngine>, CacheError>>>,
        cv: Condvar,
    }

    struct GrammarCacheInner {
        capacity: usize,
        /// Compiled, ready-to-serve entries. Recency order tracked in
        /// `order` (front = least-recently-used, back = most-recently-used);
        /// `ready` itself has no ordering, so eviction always consults
        /// `order`, never `ready`'s iteration order.
        ready: HashMap<GrammarCacheKey, Arc<GrammarEngine>>,
        order: VecDeque<GrammarCacheKey>,
        /// Keys currently being compiled by exactly one thread; every other
        /// concurrent request for the same key waits on the listed slot
        /// instead of starting a second, redundant compile.
        compiling: HashMap<GrammarCacheKey, Arc<CompileSlot>>,
    }

    /// Bounded LRU cache of compiled `GrammarEngine`s keyed by
    /// [`grammar_cache_key`], with single-flight compilation (structured-
    /// output v0 design note §"Schema-to-grammar compilation and cache
    /// placement"). Only successfully admitted-and-compiled schemas are
    /// cached -- a rejected or over-budget schema is never inserted, so a
    /// later, corrected request for the same key compiles fresh rather than
    /// replaying a cached failure.
    struct GrammarCache {
        inner: Mutex<GrammarCacheInner>,
        /// Count of compile closures actually executed (as opposed to
        /// served from cache or a single-flight wait). Test-only signal for
        /// asserting single-flight collapses N concurrent identical
        /// requests into exactly 1 compile; harmless to keep in production
        /// as a cheap counter.
        compile_count: std::sync::atomic::AtomicUsize,
    }

    impl GrammarCache {
        fn new(capacity: usize) -> Self {
            Self {
                inner: Mutex::new(GrammarCacheInner {
                    capacity,
                    ready: HashMap::new(),
                    order: VecDeque::new(),
                    compiling: HashMap::new(),
                }),
                compile_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        #[cfg(test)]
        fn compile_count(&self) -> usize {
            self.compile_count.load(std::sync::atomic::Ordering::SeqCst)
        }

        /// Returns the cached engine for `key`, compiling it via `compile`
        /// if absent. Concurrent callers with the same `key` share one
        /// compile: all but the first block on that first call's result.
        ///
        /// Never panics on a poisoned lock or a broken internal invariant:
        /// every lock is recovered via
        /// `unwrap_or_else(PoisonError::into_inner)` instead of `expect`,
        /// and `compile` runs under `catch_unwind` so a panicking compile
        /// still frees its slot and wakes every waiter with
        /// `CacheError::Internal` instead of leaving them blocked on a
        /// condvar nobody will ever notify again.
        fn get_or_compile(
            &self,
            key: GrammarCacheKey,
            compile: impl FnOnce() -> Result<GrammarEngine, String> + std::panic::UnwindSafe,
        ) -> Result<Arc<GrammarEngine>, CacheError> {
            use std::sync::PoisonError;

            enum Action {
                Hit(Arc<GrammarEngine>),
                Wait(Arc<CompileSlot>),
                Compile,
            }
            let action = {
                let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
                if let Some(engine) = inner.ready.get(&key).cloned() {
                    if let Some(pos) = inner.order.iter().position(|k| k == &key) {
                        inner.order.remove(pos);
                    }
                    inner.order.push_back(key.clone());
                    Action::Hit(engine)
                } else if let Some(slot) = inner.compiling.get(&key).cloned() {
                    Action::Wait(slot)
                } else {
                    let slot = Arc::new(CompileSlot {
                        result: Mutex::new(None),
                        cv: Condvar::new(),
                    });
                    inner.compiling.insert(key.clone(), slot);
                    Action::Compile
                }
            };
            match action {
                Action::Hit(engine) => Ok(engine),
                Action::Wait(slot) => {
                    let guard = slot.result.lock().unwrap_or_else(PoisonError::into_inner);
                    let guard = slot
                        .cv
                        .wait_while(guard, |r| r.is_none())
                        .unwrap_or_else(PoisonError::into_inner);
                    guard.clone().unwrap_or_else(|| {
                        Err(CacheError::Internal(
                            "compile slot signaled with no result".to_string(),
                        ))
                    })
                }
                Action::Compile => {
                    self.compile_count
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let result = match std::panic::catch_unwind(compile) {
                        Ok(Ok(engine)) => Ok(Arc::new(engine)),
                        Ok(Err(message)) => Err(CacheError::Compile(message)),
                        Err(_) => Err(CacheError::Internal(
                            "grammar compilation panicked".to_string(),
                        )),
                    };
                    let slot = {
                        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
                        let slot = inner.compiling.remove(&key);
                        if let Ok(engine) = &result {
                            inner.ready.insert(key.clone(), Arc::clone(engine));
                            inner.order.push_back(key.clone());
                            while inner.ready.len() > inner.capacity {
                                if let Some(lru_key) = inner.order.pop_front() {
                                    inner.ready.remove(&lru_key);
                                } else {
                                    break;
                                }
                            }
                        }
                        slot
                    };
                    // `slot` is only ever absent if this thread's own
                    // `Action::Compile` insert (above) never happened --
                    // structurally unreachable, but not worth an `expect`:
                    // there is simply no waiter to notify in that case.
                    if let Some(slot) = slot {
                        let mut slot_result =
                            slot.result.lock().unwrap_or_else(PoisonError::into_inner);
                        *slot_result = Some(result.clone());
                        slot.cv.notify_all();
                    }
                    result
                }
            }
        }
    }

    /// A structured (`response_format.json_schema`, `strict: true`) request,
    /// admitted and compiled before any worker job is submitted.
    struct StructuredRequest {
        schema: Value,
        engine: Arc<GrammarEngine>,
    }

    /// Parses and admits `req.response_format` (structured-output v0 design
    /// note, §"Accepted request profile"). Returns `Ok(None)` for an
    /// ordinary text request, `Ok(Some(..))` for an admitted+compiled
    /// strict structured request, or `Err` for any request-shape or schema
    /// problem -- always before a worker job is submitted, matching the
    /// design note's fail-closed admission contract.
    fn admit_structured_request(
        req: &ChatReq,
        stream: bool,
        vocab_bytes: &Arc<Vec<Vec<u8>>>,
        grammar_cache: &Arc<GrammarCache>,
    ) -> Result<Option<StructuredRequest>, RequestError> {
        let Some(fmt) = &req.response_format else {
            return Ok(None);
        };
        if fmt.r#type != "json_schema" {
            return Ok(None);
        }
        // Non-streaming only (design note §"Terminal and streaming
        // behavior"): an SSE consumer could observe syntactically
        // incomplete JSON at every intermediate token, so v0 rejects the
        // combination outright rather than hiding partial bytes.
        if stream {
            return Err(RequestError::bad_request(
                "response_format.json_schema requires a non-streaming request \
                 (stream must be false or omitted)",
                "unsupported_feature",
            ));
        }
        let Some(json_schema) = &fmt.json_schema else {
            return Err(RequestError::bad_request(
                "response_format.json_schema is required when response_format.type is 'json_schema'",
                "invalid_request",
            ));
        };
        let name_ok = json_schema
            .name
            .as_deref()
            .is_some_and(|n| !n.trim().is_empty());
        if !name_ok {
            return Err(RequestError::bad_request(
                "response_format.json_schema.name must be a nonempty string",
                "invalid_request",
            ));
        }
        // Missing or false `strict` is rejected rather than silently
        // treated as best-effort JSON (design note §"Accepted request
        // profile", item 3).
        if json_schema.strict != Some(true) {
            return Err(RequestError::bad_request(
                "response_format.json_schema.strict must be true; \
                 best-effort JSON mode is not supported",
                "unsupported_feature",
            ));
        }
        let Some(schema_raw) = &json_schema.schema else {
            return Err(RequestError::bad_request(
                "response_format.json_schema.schema is required",
                "invalid_request",
            ));
        };
        // Materialized here, not during the outer `ChatReq` deserialization:
        // `schema` is a `RawValue` span until this exact point, so a request
        // that fails the cheaper `name`/`strict`/`stream` checks above never
        // pays this parse.
        let schema: Value = serde_json::from_str(schema_raw.get()).map_err(|err| {
            RequestError::bad_request(
                format!("response_format.json_schema.schema is invalid: {err}"),
                "invalid_request",
            )
        })?;
        let schema = &schema;
        if let Err(message) = admit_v0_schema(schema) {
            return Err(RequestError::bad_request(
                message,
                "unsupported_strict_schema",
            ));
        }
        // Compile before enqueueing (design note §"Schema-to-grammar
        // compilation and cache placement"), via the bounded single-flight
        // cache (stage 2): a schema already compiled against this same
        // vocabulary by an earlier request is served without touching the
        // vocabulary clone or the compiler at all; a schema currently being
        // compiled by a concurrent request is waited on instead of
        // independently recompiled. The key names both the schema and the
        // vocabulary (#1557; see `grammar_cache_key`) so a differently
        // loaded vocabulary can never resolve to a grammar compiled for a
        // different one.
        let key = grammar_cache_key(vocab_bytes, schema);
        let schema_for_compile = schema.clone();
        let vocab_bytes = Arc::clone(vocab_bytes);
        let engine = grammar_cache
            .get_or_compile(key, move || {
                let spec = GrammarSpec::JsonSchema(schema_for_compile);
                let engine = GrammarEngine::new(&spec, (*vocab_bytes).clone())
                    .map_err(|e| format!("schema failed to compile: {e}"))?;
                if engine.exceeds_state_budget() {
                    return Err(
                        "schema exceeds the strict-output complexity budget (256 precomputed \
                         grammar states); v0 never falls back to the full-vocabulary simulation \
                         path"
                            .to_string(),
                    );
                }
                Ok(engine)
            })
            .map_err(|err| match err {
                CacheError::Compile(message) => {
                    RequestError::bad_request(message, "unsupported_strict_schema")
                }
                CacheError::Internal(message) => {
                    RequestError::server_error(message, "internal_error")
                }
            })?;
        Ok(Some(StructuredRequest {
            schema: schema.clone(),
            engine,
        }))
    }

    /// Fail-closed chat role (#641): only these three roles are accepted.
    /// Anything else — `tool`, `developer`, a typo, an empty string — is
    /// rejected with HTTP 400 rather than silently coerced to `user`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[cfg(test)]
    enum MessageRole {
        System,
        User,
        Assistant,
    }

    #[cfg(test)]
    impl MessageRole {
        /// ADR-080 C2: mirrors the shared contract's role classification:
        /// `tool`/`developer` are real OpenAI roles this server does not
        /// implement (`unsupported_feature`), while anything else is not an
        /// OpenAI chat role at all (`invalid_role`). Previously both collapsed
        /// to one generic message with no code differentiation.
        fn parse(raw: &str) -> Result<Self, RequestError> {
            match raw {
                "system" => Ok(Self::System),
                "user" => Ok(Self::User),
                "assistant" => Ok(Self::Assistant),
                "tool" | "developer" => Err(RequestError::bad_request(
                    format!("role '{raw}' is not supported by this server"),
                    "unsupported_feature",
                )),
                other => Err(RequestError::bad_request(
                    format!("unsupported role '{other}'; must be 'system', 'user', or 'assistant'"),
                    "invalid_role",
                )),
            }
        }
    }

    fn part_too_large_message(message_index: usize, part_index: usize) -> String {
        format!(
            "messages[{message_index}].content[{part_index}] exceeds {MAX_CONTENT_PART_BYTES} bytes"
        )
    }

    fn too_many_parts_message(message_index: usize) -> String {
        format!(
            "messages[{message_index}].content has too many parts; maximum is {MAX_CONTENT_PARTS_PER_MESSAGE}"
        )
    }

    /// Borrowed serde preflight over the raw request body: walks only
    /// `messages[*].content` and checks part counts / payload byte lengths
    /// *before* the typed `ChatReq` is deserialized, so an oversized or
    /// part-flooded body is rejected before its strings/arrays are ever
    /// allocated (serve DoS-hardening rule: clamp before alloc). Any other
    /// shape mismatch is left for the authoritative typed parse below —
    /// this function only ever raises the two size/count errors above.
    fn validate_content_part_limits(body: &[u8]) -> Result<(), RequestError> {
        use serde::Deserializer as _;
        use serde::de::{
            DeserializeSeed, Error as DeError, IgnoredAny, MapAccess, SeqAccess, Visitor,
        };
        use std::cell::RefCell;
        use std::fmt;

        let violation: RefCell<Option<RequestError>> = RefCell::new(None);

        struct StringLenSeed<'v> {
            violation: &'v RefCell<Option<RequestError>>,
            message_index: usize,
            part_index: usize,
        }
        impl<'de, 'v> DeserializeSeed<'de> for StringLenSeed<'v> {
            type Value = ();
            fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct V<'v> {
                    violation: &'v RefCell<Option<RequestError>>,
                    message_index: usize,
                    part_index: usize,
                }
                impl<'de, 'v> Visitor<'de> for V<'v> {
                    type Value = ();
                    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        f.write_str("a string")
                    }
                    fn visit_str<E: DeError>(self, v: &str) -> Result<(), E> {
                        if v.len() > MAX_CONTENT_PART_BYTES {
                            *self.violation.borrow_mut() = Some(RequestError::bad_request(
                                part_too_large_message(self.message_index, self.part_index),
                                // #649 DoS-hardening surface with no
                                // lattice.rs analog (lattice.rs has no
                                // content-part size/count limit at all) --
                                // a new, stable, lattice_serve-only code
                                // rather than an invented false match with
                                // any lattice.rs code (ADR-080 C2).
                                "content_part_limit_exceeded",
                            ));
                        }
                        Ok(())
                    }
                    fn visit_string<E: DeError>(self, v: String) -> Result<(), E> {
                        self.visit_str(&v)
                    }
                    fn visit_unit<E: DeError>(self) -> Result<(), E> {
                        Ok(())
                    }
                }
                deserializer.deserialize_any(V {
                    violation: self.violation,
                    message_index: self.message_index,
                    part_index: self.part_index,
                })
            }
        }

        struct ImageUrlLenSeed<'v> {
            violation: &'v RefCell<Option<RequestError>>,
            message_index: usize,
            part_index: usize,
        }
        impl<'de, 'v> DeserializeSeed<'de> for ImageUrlLenSeed<'v> {
            type Value = ();
            fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct V<'v> {
                    violation: &'v RefCell<Option<RequestError>>,
                    message_index: usize,
                    part_index: usize,
                }
                impl<'de, 'v> Visitor<'de> for V<'v> {
                    type Value = ();
                    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        f.write_str("an image_url object")
                    }
                    fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
                    where
                        A: MapAccess<'de>,
                    {
                        while let Some(key) = map.next_key::<&str>()? {
                            if key == "url" {
                                map.next_value_seed(StringLenSeed {
                                    violation: self.violation,
                                    message_index: self.message_index,
                                    part_index: self.part_index,
                                })?;
                            } else {
                                map.next_value::<IgnoredAny>()?;
                            }
                        }
                        Ok(())
                    }
                    fn visit_unit<E: DeError>(self) -> Result<(), E> {
                        Ok(())
                    }
                }
                deserializer.deserialize_any(V {
                    violation: self.violation,
                    message_index: self.message_index,
                    part_index: self.part_index,
                })
            }
        }

        struct PartSeed<'v> {
            violation: &'v RefCell<Option<RequestError>>,
            message_index: usize,
            part_index: usize,
        }
        impl<'de, 'v> DeserializeSeed<'de> for PartSeed<'v> {
            type Value = ();
            fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct V<'v> {
                    violation: &'v RefCell<Option<RequestError>>,
                    message_index: usize,
                    part_index: usize,
                }
                impl<'de, 'v> Visitor<'de> for V<'v> {
                    type Value = ();
                    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        f.write_str("a content part object")
                    }
                    fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
                    where
                        A: MapAccess<'de>,
                    {
                        while let Some(key) = map.next_key::<&str>()? {
                            match key {
                                "text" => {
                                    map.next_value_seed(StringLenSeed {
                                        violation: self.violation,
                                        message_index: self.message_index,
                                        part_index: self.part_index,
                                    })?;
                                }
                                "image_url" => {
                                    map.next_value_seed(ImageUrlLenSeed {
                                        violation: self.violation,
                                        message_index: self.message_index,
                                        part_index: self.part_index,
                                    })?;
                                }
                                _ => {
                                    map.next_value::<IgnoredAny>()?;
                                }
                            }
                        }
                        Ok(())
                    }
                }
                deserializer.deserialize_any(V {
                    violation: self.violation,
                    message_index: self.message_index,
                    part_index: self.part_index,
                })
            }
        }

        struct ContentVisitor<'v> {
            violation: &'v RefCell<Option<RequestError>>,
            message_index: usize,
        }
        impl<'de, 'v> Visitor<'de> for ContentVisitor<'v> {
            type Value = ();
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a string or array of content parts")
            }
            fn visit_str<E: DeError>(self, v: &str) -> Result<(), E> {
                if v.len() > MAX_CONTENT_PART_BYTES {
                    *self.violation.borrow_mut() = Some(RequestError::bad_request(
                        part_too_large_message(self.message_index, 0),
                        "content_part_limit_exceeded",
                    ));
                }
                Ok(())
            }
            fn visit_string<E: DeError>(self, v: String) -> Result<(), E> {
                self.visit_str(&v)
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<(), A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut part_index = 0usize;
                loop {
                    let seed = PartSeed {
                        violation: self.violation,
                        message_index: self.message_index,
                        part_index,
                    };
                    match seq.next_element_seed(seed)? {
                        Some(()) => {
                            part_index += 1;
                            // #656: reject only once MORE than the documented
                            // maximum has actually been consumed, so an array
                            // of exactly `MAX_CONTENT_PARTS_PER_MESSAGE` parts
                            // is accepted and only the (MAX+1)-th is rejected.
                            if part_index > MAX_CONTENT_PARTS_PER_MESSAGE {
                                let msg = too_many_parts_message(self.message_index);
                                *self.violation.borrow_mut() = Some(RequestError::bad_request(
                                    msg.clone(),
                                    "content_part_limit_exceeded",
                                ));
                                return Err(A::Error::custom(msg));
                            }
                        }
                        None => break,
                    }
                }
                Ok(())
            }
        }

        struct MessageVisitor<'v> {
            violation: &'v RefCell<Option<RequestError>>,
            message_index: usize,
        }
        impl<'de, 'v> Visitor<'de> for MessageVisitor<'v> {
            type Value = ();
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a message object")
            }
            fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
            where
                A: MapAccess<'de>,
            {
                while let Some(key) = map.next_key::<&str>()? {
                    if key == "content" {
                        map.next_value_seed(ContentSeed {
                            violation: self.violation,
                            message_index: self.message_index,
                        })?;
                    } else {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(())
            }
        }

        struct ContentSeed<'v> {
            violation: &'v RefCell<Option<RequestError>>,
            message_index: usize,
        }
        impl<'de, 'v> DeserializeSeed<'de> for ContentSeed<'v> {
            type Value = ();
            fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                deserializer.deserialize_any(ContentVisitor {
                    violation: self.violation,
                    message_index: self.message_index,
                })
            }
        }

        struct MessageSeed<'v> {
            violation: &'v RefCell<Option<RequestError>>,
            message_index: usize,
        }
        impl<'de, 'v> DeserializeSeed<'de> for MessageSeed<'v> {
            type Value = ();
            fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                deserializer.deserialize_map(MessageVisitor {
                    violation: self.violation,
                    message_index: self.message_index,
                })
            }
        }

        struct MessagesVisitor<'v> {
            violation: &'v RefCell<Option<RequestError>>,
        }
        impl<'de, 'v> Visitor<'de> for MessagesVisitor<'v> {
            type Value = ();
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an array of messages")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<(), A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut message_index = 0usize;
                while seq
                    .next_element_seed(MessageSeed {
                        violation: self.violation,
                        message_index,
                    })?
                    .is_some()
                {
                    message_index += 1;
                }
                Ok(())
            }
        }

        struct TopVisitor<'v> {
            violation: &'v RefCell<Option<RequestError>>,
        }
        impl<'de, 'v> Visitor<'de> for TopVisitor<'v> {
            type Value = ();
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a chat completion request object")
            }
            fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
            where
                A: MapAccess<'de>,
            {
                while let Some(key) = map.next_key::<&str>()? {
                    if key == "messages" {
                        map.next_value_seed(MessagesSeed {
                            violation: self.violation,
                        })?;
                    } else {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(())
            }
        }

        struct MessagesSeed<'v> {
            violation: &'v RefCell<Option<RequestError>>,
        }
        impl<'de, 'v> DeserializeSeed<'de> for MessagesSeed<'v> {
            type Value = ();
            fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                deserializer.deserialize_seq(MessagesVisitor {
                    violation: self.violation,
                })
            }
        }

        let mut de = serde_json::Deserializer::from_slice(body);
        // Any error other than a captured violation is left for the typed
        // `ChatReq` parse below to report authoritatively.
        let _ = de.deserialize_any(TopVisitor {
            violation: &violation,
        });
        match violation.into_inner() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Raw-JSON duplicate-member scan:
    /// rejects a request body containing an object with a repeated member
    /// name, at any nesting depth, ANYWHERE in the body -- not scoped to
    /// just `response_format.json_schema.schema`. Deliberate choice: a
    /// schema-only scope would still let a duplicate elsewhere in the body
    /// (e.g. a duplicated top-level `messages`) pass through the same
    /// receiver-ambiguity RFC 8259 §4 warns about, and scanning the whole
    /// body is no more expensive than the existing raw preflight pass
    /// `validate_content_part_limits` already does below.
    ///
    /// Runs on the raw bytes BEFORE any deserialization into `Value` or
    /// `ChatReq` -- both materialize objects as maps, so by the time
    /// `admit_v0_schema` walks the parsed `Value` a duplicate key has
    /// already silently collapsed to whichever pair `serde_json` kept.
    /// A genuine JSON syntax error is deliberately NOT reported here (this
    /// function returns `Ok(())` for any error other than a captured
    /// duplicate name); the typed `ChatReq` parse below remains the single
    /// authoritative source for "this body is not valid JSON at all",
    /// exactly like `validate_content_part_limits`'s own established
    /// pattern in this file.
    fn reject_duplicate_json_members(body: &[u8]) -> Result<(), RequestError> {
        use serde::Deserializer as _;
        use serde::de::{DeserializeSeed, Error as DeError, MapAccess, SeqAccess, Visitor};
        use std::cell::RefCell;
        use std::fmt;

        let duplicate: RefCell<Option<String>> = RefCell::new(None);

        struct AnySeed<'v> {
            duplicate: &'v RefCell<Option<String>>,
        }
        impl<'de, 'v> DeserializeSeed<'de> for AnySeed<'v> {
            type Value = ();
            fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                deserializer.deserialize_any(AnyVisitor {
                    duplicate: self.duplicate,
                })
            }
        }

        struct AnyVisitor<'v> {
            duplicate: &'v RefCell<Option<String>>,
        }
        impl<'de, 'v> Visitor<'de> for AnyVisitor<'v> {
            type Value = ();
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_bool<E: DeError>(self, _v: bool) -> Result<(), E> {
                Ok(())
            }
            fn visit_i64<E: DeError>(self, _v: i64) -> Result<(), E> {
                Ok(())
            }
            fn visit_u64<E: DeError>(self, _v: u64) -> Result<(), E> {
                Ok(())
            }
            fn visit_f64<E: DeError>(self, _v: f64) -> Result<(), E> {
                Ok(())
            }
            fn visit_str<E: DeError>(self, _v: &str) -> Result<(), E> {
                Ok(())
            }
            fn visit_string<E: DeError>(self, _v: String) -> Result<(), E> {
                Ok(())
            }
            fn visit_unit<E: DeError>(self) -> Result<(), E> {
                Ok(())
            }
            fn visit_none<E: DeError>(self) -> Result<(), E> {
                Ok(())
            }
            fn visit_some<D>(self, deserializer: D) -> Result<(), D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                deserializer.deserialize_any(AnyVisitor {
                    duplicate: self.duplicate,
                })
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<(), A::Error>
            where
                A: SeqAccess<'de>,
            {
                while seq
                    .next_element_seed(AnySeed {
                        duplicate: self.duplicate,
                    })?
                    .is_some()
                {}
                Ok(())
            }
            fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        *self.duplicate.borrow_mut() = Some(key.clone());
                        return Err(A::Error::custom(format!("duplicate object member '{key}'")));
                    }
                    map.next_value_seed(AnySeed {
                        duplicate: self.duplicate,
                    })?;
                }
                Ok(())
            }
        }

        let mut de = serde_json::Deserializer::from_slice(body);
        let _ = de.deserialize_any(AnyVisitor {
            duplicate: &duplicate,
        });
        match duplicate.into_inner() {
            Some(key) => Err(RequestError::bad_request(
                format!("request body contains a duplicate JSON object member name '{key}'"),
                "invalid_request_body",
            )),
            None => Ok(()),
        }
    }

    /// Clamp-then-parse entry point (#649 DoS hardening): validates part
    /// counts/sizes against the raw bytes first, then deserializes the typed
    /// request. Never allocates the request's strings/arrays before the
    /// clamp has run.
    ///
    /// The message-count bound is enforced inline during the typed `ChatReq`
    /// parse below (`serve::contract::deserialize_bounded_messages`): a body
    /// with more than `MAX_MESSAGE_COUNT` tiny messages is rejected without
    /// materializing a `Vec<Message>` entry for each one, in that same parse
    /// -- there is no separate raw-bytes pass over `messages` ahead of it.
    fn parse_chat_req(body: &[u8]) -> Result<ChatReq, RequestError> {
        reject_duplicate_json_members(body)?;
        validate_content_part_limits(body)?;
        serde_json::from_slice::<ChatReq>(body).map_err(|err| {
            if is_message_flood_error(&err) {
                return RequestError::bad_request(message_flood_text(), "invalid_request_body");
            }
            RequestError::bad_request(
                "invalid JSON request body",
                // Matches lattice.rs's JSON-extraction-failure code
                // (ADR-080 C2).
                "invalid_request_body",
            )
        })
    }

    // ─── HTTP handlers ───────────────────────────────────────────────────────

    async fn health<H: Send + Sync + 'static>(State(s): State<AppState<H>>) -> &'static str {
        let t = Instant::now();
        emit_serve_event(
            &s.metrics,
            "GET",
            "/health",
            200,
            None,
            None,
            t.elapsed().as_secs_f64() * 1000.0,
            false,
            None,
        );
        "ok"
    }

    async fn root<H: Send + Sync + 'static>(State(s): State<AppState<H>>) -> Json<Value> {
        let t = Instant::now();
        // ADR-080 C2: shared with lattice.rs's equivalent route so
        // both binaries advertise the same engine-identity document.
        let body = lattice_inference::serve::root_body();
        emit_serve_event(
            &s.metrics,
            "GET",
            "/",
            200,
            None,
            None,
            t.elapsed().as_secs_f64() * 1000.0,
            false,
            None,
        );
        Json(body)
    }

    async fn list_models<H: Send + Sync + 'static>(State(s): State<AppState<H>>) -> Json<Value> {
        let t = Instant::now();
        // ADR-080 C2: shared with `lattice.rs`'s equivalent route so both
        // binaries advertise the single loaded model in byte-identical shape.
        let body = lattice_inference::serve::models_list_body(s.model_id.as_ref(), unix_secs());
        emit_serve_event(
            &s.metrics,
            "GET",
            "/v1/models",
            200,
            None,
            None,
            t.elapsed().as_secs_f64() * 1000.0,
            false,
            None,
        );
        Json(body)
    }

    /// `GET /metrics` (issue #583): Prometheus text-exposition-format
    /// metrics for this server -- request counts/latency by route+status,
    /// prompt/completion token counters, error counts by code (all labeled
    /// with the loaded model id), and a live queue-depth/in-flight gauge
    /// read directly off the shared worker's admission semaphore (issue
    /// #932's cap), not a separately-tracked counter that could drift from
    /// the real admission state.
    ///
    /// **Deployment boundary**: this endpoint has no authentication or
    /// opt-in guard, matching every other route on this server (see
    /// "Auth, rate limiting, and concurrency" in `docs/serve-http-api.md`).
    /// Accepted as-is for the current local-first deployment model (this
    /// server binds `127.0.0.1` by default), but `--host` can bind a
    /// non-loopback address -- doing that exposes `/metrics` (request
    /// volume/latency/error-rate shape) to anyone who can reach the
    /// listening address. Do not bind `--host` to a non-loopback address
    /// without an external auth layer (reverse proxy, firewall) in front.
    async fn metrics_handler<H: IntoServingHost>(
        State(s): State<AppState<H>>,
    ) -> impl IntoResponse {
        let in_flight = s.max_pending.saturating_sub(s.jobs.available_permits());
        let body = s.metrics.render(s.model_id.as_ref(), in_flight);
        (
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            body,
        )
    }

    /// Phase machine for the SSE token stream.
    /// `Body`, `Done`, and `End` carry the (completion_tokens, prompt_tokens)
    /// pair so failure paths preserve the deltas emitted before the worker
    /// failed, and `/metrics` (issue #583) gets prompt-token counts even for
    /// streaming responses. `Done`/`End` also carry an `error outcome`:
    /// `None` for a clean `Complete`, `Some((status, code))` for a mid-stream
    /// `Failed`/`Rejected`/worker-gone event -- the on-the-wire HTTP status
    /// was already committed to 200 before streaming started and can't
    /// change, but `Phase::End`'s `emit_serve_event` call uses this to
    /// record the *true* logical status/error code, mirroring exactly what
    /// the non-streaming branch below records for the same failure classes.
    /// Without this, every failed stream was recorded as a successful 200
    /// with no error code, so `lattice_errors_total` never counted them.
    enum Phase {
        Start,
        Body(usize),
        Done(usize, usize, Option<(u16, &'static str)>), // completion_tokens, prompt_tokens, error outcome
        End(usize, usize, Option<(u16, &'static str)>),  // same; emits telemetry then stream ends
    }

    /// Content-Type guard: unlike `lattice.rs`'s
    /// `chat_completions`, this handler never went through axum's `Json`
    /// extractor at all -- it has always taken the raw `Body` directly, so
    /// it never got `Json`'s free Content-Type enforcement either, before
    /// or after this PR's message-count preflight. Guarded here with the
    /// same `require_json_content_type` check, ahead of `to_bytes`.
    async fn chat_completions<H: IntoServingHost>(
        State(s): State<AppState<H>>,
        headers: HeaderMap,
        body: Body,
    ) -> Response {
        chat_completions_with_serving_host(State(s.into_serving_host()), headers, body).await
    }

    async fn chat_completions_with_serving_host(
        State(s): State<AppState<lattice_inference::serving_host::ServingHandle>>,
        headers: HeaderMap,
        body: Body,
    ) -> Response {
        let timer = Instant::now();
        if let Err(err) = lattice_inference::serve::require_json_content_type(&headers) {
            emit_serve_event(
                &s.metrics,
                "POST",
                "/v1/chat/completions",
                415,
                None,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
                Some(err.code()),
            );
            return err.into_response();
        }
        let body = match to_bytes(body, REQUEST_BODY_LIMIT_BYTES).await {
            Ok(body) => body,
            Err(err) => {
                // Only a length-limit rejection is an oversized-body
                // condition; any other body-buffering failure (e.g. a
                // client disconnecting mid-stream) is not, and must not be
                // reported as 413.
                let is_length_limit = std::error::Error::source(&err)
                    .is_some_and(<dyn std::error::Error>::is::<http_body_util::LengthLimitError>);
                if is_length_limit {
                    emit_serve_event(
                        &s.metrics,
                        "POST",
                        "/v1/chat/completions",
                        413,
                        None,
                        None,
                        timer.elapsed().as_secs_f64() * 1000.0,
                        false,
                        Some("request_body_too_large"),
                    );
                    // ADR-080 C2: previously mapped to
                    // HTTP 400 + generic "invalid_request", diverging from
                    // lattice.rs's 413 + "request_body_too_large" for the identical
                    // oversized-body condition. Aligned to match.
                    return err_response(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        &format!("request body exceeds {REQUEST_BODY_LIMIT_BYTES} bytes"),
                        "request_body_too_large",
                    );
                }
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    "/v1/chat/completions",
                    400,
                    None,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some("invalid_request"),
                );
                return err_response(
                    StatusCode::BAD_REQUEST,
                    "invalid request body",
                    "invalid_request",
                );
            }
        };
        let req = match parse_chat_req(&body) {
            Ok(req) => req,
            Err(err) => {
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    "/v1/chat/completions",
                    400,
                    None,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some(err.code()),
                );
                return err_response(StatusCode::BAD_REQUEST, err.message(), err.code());
            }
        };
        // Features the loaded model cannot serve are refused by name before
        // any other work. A model that serves them all (Qwen3.5) never
        // refuses here.
        if let Err(err) = s.jobs.refuse_standalone_unsupported(&req) {
            emit_serve_event(
                &s.metrics,
                "POST",
                "/v1/chat/completions",
                400,
                None,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
                Some(err.code()),
            );
            return err.into_response();
        }
        // Same resolver as the other binary, for the same reason: these two read
        // the raw field differently today, and this binary had no request-boundary
        // `validate_scales` at all — a non-finite scale reached the worker and was
        // refused inside `apply` instead. Both reject it either way; routing it
        // through the shared resolver makes them reject it at the same place.
        let requested = lattice_inference::serve::lora::requested_adapters(req.lora.clone());
        if let Err(err) = lattice_inference::serve::lora::validate_scales(requested.selection()) {
            emit_serve_event(
                &s.metrics,
                "POST",
                "/v1/chat/completions",
                400,
                None,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
                Some(err.code()),
            );
            return err.into_response();
        }
        if let Err(err) = s.jobs.validate_lora(requested.selection()) {
            emit_serve_event(
                &s.metrics,
                "POST",
                "/v1/chat/completions",
                400,
                None,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
                Some(err.code()),
            );
            return err.into_response();
        }
        let normalized = s.jobs.prepare_standalone_chat_request(
            &req,
            s.defaults,
            s.model_id.as_ref(),
            s.model_max_context,
        );
        let (validated, mut cfg) = match normalized {
            Ok(prepared) => prepared,
            Err(err) => {
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    "/v1/chat/completions",
                    400,
                    None,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some(err.code()),
                );
                return err.into_response();
            }
        };
        // Structured-output v0 (design note): admit + compile
        // `response_format.json_schema` BEFORE any worker job is submitted.
        // `Ok(None)` for an ordinary text request; a schema/streaming
        // problem here always surfaces as a 4xx, never a best-effort
        // fallback to unconstrained generation.
        let structured = match admit_structured_request(
            &req,
            validated.stream,
            &s.vocab_bytes,
            &s.grammar_cache,
        ) {
            Ok(structured) => structured,
            Err(err) => {
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    "/v1/chat/completions",
                    err.status().as_u16(),
                    None,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some(err.code()),
                );
                return err_response(err.status(), err.message(), err.code());
            }
        };
        // Wire the compiled grammar into the worker config (design note
        // §"End-to-end execution", step 4) and force `enable_thinking` off
        // for strict requests regardless of server defaults:
        // the grammar masks from the first decode token, so thinking
        // tokens are unsampleable under it and a think-then-constrain flow
        // is out of scope for v0.
        if let Some(structured) = &structured {
            cfg.grammar = Some(Arc::clone(&structured.engine));
            cfg.enable_thinking = false;
        }
        let streaming = validated.stream;
        let messages = match into_engine_chat_messages(validated.messages) {
            Ok(messages) => messages,
            Err(err) => {
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    "/v1/chat/completions",
                    StatusCode::BAD_REQUEST.as_u16(),
                    None,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some(err.code()),
                );
                return err.into_response();
            }
        };
        let cfg = cfg;
        let model_id = s.model_id.to_string();
        let id = format!("chatcmpl-{}", unix_nanos());
        let created = unix_secs();

        let (cancel_guard, cancel_rx) = lattice_inference::serve::cancel_pair();
        // The serving facade's worker submission only fails outwardly for admission
        // (issue #932, checked synchronously here, before any tokenization/model work): if the
        // worker thread is no longer running, the returned receiver simply closes with zero events,
        // which both branches below detect via their own first-event peek (`rx.recv()` returning
        // `None`) and report identically to this binary's prior up-front `jobs.send(..).is_err()`
        // check.
        let mut rx = match s.jobs.submit_standalone(
            messages,
            cfg,
            cancel_rx,
            requested.selection().to_vec(),
            streaming,
            lattice_inference::serve::prepare::explicitly_enabled_thinking(&req).is_ok_and(|on| on),
        ) {
            Ok(rx) => rx,
            Err(api_err) => {
                let code = api_err.code();
                let response = api_err.into_response();
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    "/v1/chat/completions",
                    response.status().as_u16(),
                    None,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some(code),
                );
                return response;
            }
        };
        // Dropped when nobody cares about the response anymore: at the end of
        // this SSE stream (moved in below) or at the end of this function for
        // the non-streaming branch. Either way that's the client disconnect
        // signal the worker checks in the shared FIFO loop
        // (`lattice_inference::serve::metal_worker::run_worker_loop`).
        if streaming {
            // ADR-080 C2: without this
            // preflight, a prompt-plus-budget overflow was only discoverable
            // AFTER the HTTP response had already committed to 200 SSE (the
            // worker's `WorkerEvent::Rejected` arrived mid-stream, terminating
            // with `finish_reason: "length"` instead of the 400
            // `context_length_exceeded` the non-streaming path and
            // `lattice.rs`'s `check_context_window` preflight both return for
            // the identical request body). Await the worker's FIRST event
            // before deciding the status code at all -- exactly the
            // non-streaming branch's own ordering below, just not looping to
            // drain every event yet. `WorkerEvent::Rejected`/`Failed` on this
            // first event get the same un-committed 400/500 treatment as
            // non-streaming; anything else (`Delta`/`Complete`) means the
            // request passed the worker's checks, so NOW commit to 200 SSE --
            // carrying that already-consumed first event into the stream's
            // state machine (`pending`) so it isn't silently dropped.
            let Some(first_ev) = rx.recv().await.map(normalize_cancelled) else {
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    "/v1/chat/completions",
                    500,
                    None,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    true,
                    Some("internal_error"),
                );
                return err_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "inference worker unavailable",
                    "internal_error",
                );
            };
            let first_ev = match first_ev {
                WorkerEvent::Rejected(api_err) => {
                    emit_serve_event(
                        &s.metrics,
                        "POST",
                        "/v1/chat/completions",
                        400,
                        None,
                        None,
                        timer.elapsed().as_secs_f64() * 1000.0,
                        true,
                        Some(api_err.code()),
                    );
                    return api_err.into_response();
                }
                // `response_format.json_schema` is rejected outright for a
                // streaming request (see `admit_structured_request`), so a
                // `ConstraintBlocked` event can only ever mean the same
                // generic internal failure `Failed` does here -- neither
                // variant gets the `blocked_constraint` machine code on this
                // path.
                WorkerEvent::Failed(message) | WorkerEvent::ConstraintBlocked(message) => {
                    eprintln!("generation error (streaming): {message}");
                    emit_serve_event(
                        &s.metrics,
                        "POST",
                        "/v1/chat/completions",
                        500,
                        None,
                        None,
                        timer.elapsed().as_secs_f64() * 1000.0,
                        true,
                        Some("internal_error"),
                    );
                    return err_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "inference failed",
                        "internal_error",
                    );
                }
                ev @ (WorkerEvent::Delta(_) | WorkerEvent::Complete(_)) => ev,
                WorkerEvent::Cancelled => {
                    unreachable!("normalize_cancelled already rewrote Cancelled into Complete")
                }
                // Unrecognized future event kind as the very FIRST event:
                // nothing has been committed to the client yet, so fail
                // closed the same way the `Failed`/`ConstraintBlocked` arm
                // above does rather than guessing at a status code.
                _ => {
                    emit_serve_event(
                        &s.metrics,
                        "POST",
                        "/v1/chat/completions",
                        500,
                        None,
                        None,
                        timer.elapsed().as_secs_f64() * 1000.0,
                        true,
                        Some("internal_error"),
                    );
                    return err_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "inference failed",
                        "internal_error",
                    );
                }
            };
            let metrics = s.metrics.clone();
            let stream = futures::stream::unfold(
                (rx, Phase::Start, cancel_guard, Some(first_ev), metrics),
                move |(mut rx, phase, cancel_guard, mut pending, metrics)| {
                    let id = id.clone();
                    let model = model_id.clone();
                    async move {
                        match phase {
                            Phase::Start => {
                                let chunk = json!({
                                    "id": id, "object": "chat.completion.chunk",
                                    "created": created, "model": model,
                                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
                                });
                                Some((
                                    Ok::<Event, std::convert::Infallible>(
                                        Event::default().data(chunk.to_string()),
                                    ),
                                    (rx, Phase::Body(0), cancel_guard, pending, metrics),
                                ))
                            }
                            Phase::Body(completion_tokens) => match match pending.take() {
                                Some(ev) => Some(ev),
                                None => rx.recv().await,
                            }
                            .map(normalize_cancelled)
                            {
                                Some(WorkerEvent::Delta(d)) => {
                                    let chunk = json!({
                                        "id": id, "object": "chat.completion.chunk",
                                        "created": created, "model": model,
                                        "choices": [{"index": 0, "delta": {"content": d}, "finish_reason": null}],
                                    });
                                    Some((
                                        Ok(Event::default().data(chunk.to_string())),
                                        (
                                            rx,
                                            Phase::Body(completion_tokens.saturating_add(1)),
                                            cancel_guard,
                                            None,
                                            metrics,
                                        ),
                                    ))
                                }
                                Some(WorkerEvent::Complete(output)) => {
                                    // ADR-080 C2, #746: the engine's actual
                                    // stop cause, not a hardcoded "stop" --
                                    // `finish_reason` is "length" whenever
                                    // `stopped` is false (token cap or
                                    // cancellation), matching `lattice.rs`'s
                                    // contract exactly.
                                    let finish_reason =
                                        lattice_inference::serve::finish_reason(output.stopped);
                                    let chunk = json!({
                                        "id": id, "object": "chat.completion.chunk",
                                        "created": created, "model": model,
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                                    });
                                    Some((
                                        Ok(Event::default().data(chunk.to_string())),
                                        (
                                            rx,
                                            Phase::Done(
                                                output.generated_tokens,
                                                output.prompt_tokens,
                                                None,
                                            ),
                                            cancel_guard,
                                            None,
                                            metrics,
                                        ),
                                    ))
                                }
                                Some(
                                    WorkerEvent::Failed(message)
                                    | WorkerEvent::ConstraintBlocked(message),
                                ) => {
                                    // The HTTP response was already committed as
                                    // 200 + text/event-stream when this SSE stream
                                    // started, so an error mid-stream cannot change
                                    // the status code. Mirror `lattice.rs`'s
                                    // `StreamMsg::Failed` contract: log the real
                                    // cause server-side (#611: e.g. a grammar mask
                                    // that blocks every candidate token) and expose
                                    // only the generic error envelope to the client.
                                    eprintln!("generation error (streaming): {message}");
                                    let error = json!({
                                        "error": {
                                            "message": "inference failed",
                                            "type": "server_error",
                                            "code": "internal_error",
                                            "param": null,
                                        }
                                    });
                                    Some((
                                        Ok(Event::default().data(error.to_string())),
                                        (
                                            rx,
                                            // Mirrors the non-streaming `Failed` arm's
                                            // 500 + "internal_error" recording below.
                                            Phase::Done(
                                                completion_tokens,
                                                0,
                                                Some((500, "internal_error")),
                                            ),
                                            cancel_guard,
                                            None,
                                            metrics,
                                        ),
                                    ))
                                }
                                Some(WorkerEvent::Rejected(api_err)) => {
                                    // Structurally unreachable at this point:
                                    // the shared worker's window check runs
                                    // before any `Delta`/`Complete` is ever
                                    // sent for a job, so `Rejected` can only
                                    // ever be the FIRST event a job produces
                                    // -- and the preflight above (ADR-080 C2)
                                    // already intercepts that one before
                                    // committing to 200 SSE. Kept as a
                                    // defensive fallback
                                    // in case that invariant ever changes
                                    // rather than deleting the arm outright.
                                    eprintln!(
                                        "request rejected (streaming): {}",
                                        api_err.message()
                                    );
                                    let error = json!({
                                        "error": {
                                            "message": "request rejected",
                                            "type": "invalid_request_error",
                                            "code": "invalid_request",
                                            "param": null,
                                        }
                                    });
                                    // Mirrors the non-streaming `Rejected` arm's
                                    // 400 + `api_err.code()` recording below.
                                    let error_outcome = Some((400, api_err.code()));
                                    Some((
                                        Ok(Event::default().data(error.to_string())),
                                        (
                                            rx,
                                            Phase::Done(completion_tokens, 0, error_outcome),
                                            cancel_guard,
                                            None,
                                            metrics,
                                        ),
                                    ))
                                }
                                Some(WorkerEvent::Cancelled) => unreachable!(
                                    "normalize_cancelled already rewrote Cancelled into Complete"
                                ),
                                // Unrecognized future event kind mid-stream: the
                                // wire status was already committed to 200 SSE,
                                // so mirror the `Failed`/`ConstraintBlocked` arm
                                // above and fail the stream closed with a
                                // generic internal error rather than guessing.
                                Some(_) => {
                                    let error = json!({
                                        "error": {
                                            "message": "inference failed",
                                            "type": "server_error",
                                            "code": "internal_error",
                                            "param": null,
                                        }
                                    });
                                    Some((
                                        Ok(Event::default().data(error.to_string())),
                                        (
                                            rx,
                                            Phase::Done(
                                                completion_tokens,
                                                0,
                                                Some((500, "internal_error")),
                                            ),
                                            cancel_guard,
                                            None,
                                            metrics,
                                        ),
                                    ))
                                }
                                None => {
                                    let error = json!({
                                        "error": {
                                            "message": "inference worker unavailable",
                                            "type": "server_error",
                                            "code": "internal_error",
                                            "param": null,
                                        }
                                    });
                                    // Mirrors this same file's non-streaming
                                    // worker-unavailable preflight, which
                                    // records 500 + "internal_error".
                                    Some((
                                        Ok(Event::default().data(error.to_string())),
                                        (
                                            rx,
                                            Phase::Done(
                                                completion_tokens,
                                                0,
                                                Some((500, "internal_error")),
                                            ),
                                            cancel_guard,
                                            None,
                                            metrics,
                                        ),
                                    ))
                                }
                            },
                            Phase::Done(ct, pt, outcome) => Some((
                                Ok(Event::default().data("[DONE]")),
                                (rx, Phase::End(ct, pt, outcome), cancel_guard, None, metrics),
                            )),
                            Phase::End(ct, pt, outcome) => {
                                // `outcome` is the true logical result of the
                                // stream: `None` for a clean `Complete`,
                                // `Some((status, code))` when one of the
                                // failure arms above ran. The wire status was
                                // already committed to 200 SSE and cannot
                                // change, but the RECORDED metric must
                                // reflect what actually happened so
                                // `lattice_errors_total` and the request
                                // counter aren't lying about a failed stream.
                                let (status, error_code) = match outcome {
                                    Some((status, code)) => (status, Some(code)),
                                    None => (200, None),
                                };
                                emit_serve_event(
                                    &metrics,
                                    "POST",
                                    "/v1/chat/completions",
                                    status,
                                    Some(pt),
                                    Some(ct),
                                    timer.elapsed().as_secs_f64() * 1000.0,
                                    true,
                                    error_code,
                                );
                                None
                            }
                        }
                    }
                },
            );
            Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response()
        } else {
            // Mirrors the streaming branch's own first-event peek above.
            // This binary's earlier up-front
            // `jobs.send(..).is_err()` check was the ONLY thing catching a
            // dead/unavailable worker for this path; now that
            // facade submission never fails outwardly, a worker
            // that is gone before this job is even dequeued shows up here as
            // `rx.recv()` returning `None` on the very first poll, same as
            // streaming's preflight. Without this peek the loop below would
            // exit immediately with all-zero/empty fields and silently
            // return a 200 JSON completion with empty content instead of the
            // 500 this binary has always returned for that condition.
            let Some(first_ev) = rx.recv().await.map(normalize_cancelled) else {
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    "/v1/chat/completions",
                    500,
                    None,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some("internal_error"),
                );
                return err_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "inference worker unavailable",
                    "internal_error",
                );
            };
            let mut content = String::new();
            let mut prompt_tokens = 0usize;
            let mut completion_tokens = 0usize;
            let mut stopped = false;
            let mut stop_reason: Option<lattice_inference::StopReason> = None;
            let mut pending = Some(first_ev);
            while let Some(ev) = match pending.take() {
                Some(ev) => Some(ev),
                None => rx.recv().await.map(normalize_cancelled),
            } {
                match ev {
                    WorkerEvent::Delta(d) => content.push_str(&d),
                    WorkerEvent::Complete(output) => {
                        prompt_tokens = output.prompt_tokens;
                        completion_tokens = output.generated_tokens;
                        stopped = output.stopped;
                        stop_reason = output.stop_reason;
                    }
                    WorkerEvent::Failed(message) => {
                        // Unlike streaming, the response has not been committed
                        // yet, so a generation failure (#611: e.g. a grammar mask
                        // that blocks every candidate token) can still surface as
                        // a real HTTP error instead of a disguised 200 -- the
                        // same "generic 500, specific detail logged server-side"
                        // contract the CPU/Metal handlers in `lattice.rs` use.
                        eprintln!("generation error: {message}");
                        emit_serve_event(
                            &s.metrics,
                            "POST",
                            "/v1/chat/completions",
                            500,
                            None,
                            None,
                            timer.elapsed().as_secs_f64() * 1000.0,
                            false,
                            Some("internal_error"),
                        );
                        return lattice_inference::serve::ApiError::ServerError {
                            message: "inference failed".to_string(),
                            code: "internal_error",
                        }
                        .into_response();
                    }
                    WorkerEvent::ConstraintBlocked(message) => {
                        // Structured-output v0 (design note): a
                        // strict request's generation failure is reported
                        // with the `blocked_constraint` machine code instead
                        // of the generic `internal_error`. This used to be
                        // decided by sniffing `message` for the engine's known
                        // grammar-exhausted-mask wording
                        // (`metal_qwen35.rs`'s `has_finite_logit` fail-closed
                        // check); `WorkerEvent::ConstraintBlocked` is now a
                        // distinct type from `WorkerEvent::Failed` all the
                        // way from `InferenceError::GrammarConstraintBlocked`
                        // through `WorkerFailure`, so a backend wording
                        // change can no longer degrade this to
                        // `internal_error`. `message` is logged only, never
                        // inspected for classification. KNOWN GAP
                        // (documented in the stage-1 report): the
                        // prefix-cache generation path does not call
                        // `is_complete_without_continuation` before raising
                        // this error the way the canonical CPU generation
                        // loop does, so a schema that legitimately finished
                        // (no further byte is valid after a closed
                        // top-level value) currently produces the exact same
                        // typed error and also lands here as
                        // `blocked_constraint`, not a 200. Closing that gap
                        // is prefix-cache generation-loop work, out of this
                        // stage's admission+wiring scope.
                        eprintln!("generation error: {message}");
                        emit_serve_event(
                            &s.metrics,
                            "POST",
                            "/v1/chat/completions",
                            500,
                            None,
                            None,
                            timer.elapsed().as_secs_f64() * 1000.0,
                            false,
                            Some("blocked_constraint"),
                        );
                        return lattice_inference::serve::ApiError::ServerError {
                            message: "structured output generation was blocked: no legal \
                                      token continues the schema from this state"
                                .to_string(),
                            code: "blocked_constraint",
                        }
                        .into_response();
                    }
                    WorkerEvent::Rejected(api_err) => {
                        // #656: client-caused request-contract violation (the
                        // prompt plus requested generation does not fit the
                        // model's KV window), caught before any generation
                        // ran -- unlike `Failed`, the response has not been
                        // committed yet, so this surfaces as a real 400 with
                        // the specific reason, not a generic 500.
                        emit_serve_event(
                            &s.metrics,
                            "POST",
                            "/v1/chat/completions",
                            400,
                            None,
                            None,
                            timer.elapsed().as_secs_f64() * 1000.0,
                            false,
                            Some(api_err.code()),
                        );
                        // Matches lattice.rs's `check_context_window` code
                        // for the analogous prompt-plus-budget-exceeds-window
                        // condition (ADR-080 C2).
                        return api_err.into_response();
                    }
                    WorkerEvent::Cancelled => {
                        unreachable!("normalize_cancelled already rewrote Cancelled into Complete")
                    }
                    // Unrecognized future event kind: the response has not
                    // been committed yet, so mirror the `Failed` arm above
                    // and fail closed with a generic internal error rather
                    // than guessing.
                    _ => {
                        emit_serve_event(
                            &s.metrics,
                            "POST",
                            "/v1/chat/completions",
                            500,
                            None,
                            None,
                            timer.elapsed().as_secs_f64() * 1000.0,
                            false,
                            Some("internal_error"),
                        );
                        return lattice_inference::serve::ApiError::ServerError {
                            message: "inference failed".to_string(),
                            code: "internal_error",
                        }
                        .into_response();
                    }
                }
            }
            // Structured-output v0 (design note §"End-to-end execution",
            // steps 5-6): a strict request never returns partial JSON as a
            // 200. `stopped == false` covers length-limit and KV-window
            // exhaustion identically (the engine's `GenerateOutput` does not
            // separate them at this call site any more finely than the
            // `finish_reason` mapping below already does); an interrupted
            // request never reaches this line at all (the client is gone).
            if let Some(structured) = &structured {
                if !stopped {
                    emit_serve_event(
                        &s.metrics,
                        "POST",
                        "/v1/chat/completions",
                        500,
                        Some(prompt_tokens),
                        Some(completion_tokens),
                        timer.elapsed().as_secs_f64() * 1000.0,
                        false,
                        Some("length_limit"),
                    );
                    return lattice_inference::serve::ApiError::ServerError {
                        message: format!(
                            "structured output did not complete within max_tokens \
                             (stop_reason={stop_reason:?}); refusing to return partial JSON"
                        ),
                        code: "length_limit",
                    }
                    .into_response();
                }
                // Independent validation (design note's "independent
                // validation" requirement): a raw-text walk
                // (`v0_validate_json`) that shares no code with the grammar
                // compiler AND never goes through `serde_json::Value`, so a
                // shared compiler bug -- or `Value`'s own numeric-precision
                // limits -- cannot make this check silently agree with a
                // wrong generation.
                let conforms = v0_validate_json(&content, &structured.schema);
                if !conforms {
                    emit_serve_event(
                        &s.metrics,
                        "POST",
                        "/v1/chat/completions",
                        500,
                        Some(prompt_tokens),
                        Some(completion_tokens),
                        timer.elapsed().as_secs_f64() * 1000.0,
                        false,
                        Some("validation_failed"),
                    );
                    return lattice_inference::serve::ApiError::ServerError {
                        message: "generated text failed independent JSON Schema validation"
                            .to_string(),
                        code: "validation_failed",
                    }
                    .into_response();
                }
            }
            // ADR-080 C2, #746: the engine's actual stop cause, not a
            // hardcoded "stop" -- matches the streaming path and
            // `lattice.rs`'s non-streaming contract.
            let finish_reason = lattice_inference::serve::finish_reason(stopped);
            let mut body = json!({
                "id": id, "object": "chat.completion",
                "created": created, "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            });
            // Structured-output v0 route-test marker (design note §"Serve
            // and backend integration", item 1): a route that returns
            // schema-shaped JSON WITHOUT actually having gone through
            // grammar-constrained decoding must fail its route test, so
            // this is set only on the code path above that attached
            // `cfg.grammar` from an admitted `StructuredRequest`.
            if structured.is_some() {
                body["constraint_applied"] = Value::Bool(true);
            }
            emit_serve_event(
                &s.metrics,
                "POST",
                "/v1/chat/completions",
                200,
                Some(prompt_tokens),
                Some(completion_tokens),
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
                None,
            );
            Json(body).into_response()
        }
    }

    /// `POST /v1/embeddings` (issue #584): OpenAI-compatible embeddings over
    /// the independently loaded `--embedding-model` `BertModel`. Follows
    /// `chat_completions`'s content-type/body-size/parse/validate cascade,
    /// but has no worker to submit to: `BertModel::encode_batch` runs
    /// directly, off the async executor via `spawn_blocking` (CPU matmuls,
    /// not `.await`-friendly).
    async fn embeddings<H: Send + Sync + 'static>(
        State(s): State<AppState<H>>,
        headers: HeaderMap,
        body: Body,
    ) -> Response {
        let timer = Instant::now();
        const ROUTE: &str = "/v1/embeddings";
        if let Err(err) = lattice_inference::serve::require_json_content_type(&headers) {
            emit_serve_event(
                &s.metrics,
                "POST",
                ROUTE,
                415,
                None,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
                Some(err.code()),
            );
            return err.into_response();
        }
        let body = match to_bytes(body, REQUEST_BODY_LIMIT_BYTES).await {
            Ok(body) => body,
            Err(err) => {
                let is_length_limit = std::error::Error::source(&err)
                    .is_some_and(<dyn std::error::Error>::is::<http_body_util::LengthLimitError>);
                let (status, code) = if is_length_limit {
                    (StatusCode::PAYLOAD_TOO_LARGE, "request_body_too_large")
                } else {
                    (StatusCode::BAD_REQUEST, "invalid_request")
                };
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    ROUTE,
                    status.as_u16(),
                    None,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some(code),
                );
                return err_response(
                    status,
                    if is_length_limit {
                        "request body too large"
                    } else {
                        "invalid request body"
                    },
                    code,
                );
            }
        };
        let req: TextEmbeddingsRequest = match serde_json::from_slice(&body) {
            Ok(req) => req,
            Err(_) => {
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    ROUTE,
                    400,
                    None,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some("invalid_request_body"),
                );
                return err_response(
                    StatusCode::BAD_REQUEST,
                    "invalid request body",
                    "invalid_request_body",
                );
            }
        };
        if let Err(err) = check_embeddings_unsupported_fields(&req) {
            emit_serve_event(
                &s.metrics,
                "POST",
                ROUTE,
                400,
                None,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
                Some(err.code()),
            );
            return err.into_response();
        }
        let texts = match parse_embeddings_input(&req.input) {
            Ok(texts) => texts,
            Err(err) => {
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    ROUTE,
                    400,
                    None,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some(err.code()),
                );
                return err.into_response();
            }
        };
        let Some(embedding) = s.embedding.clone() else {
            emit_serve_event(
                &s.metrics,
                "POST",
                ROUTE,
                503,
                None,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
                Some("embedding_model_not_loaded"),
            );
            return lattice_inference::serve::ApiError::EmbeddingModelNotLoaded {
                message: "no embedding model is loaded; start lattice_serve with \
                          --embedding-model <dir>"
                    .to_string(),
            }
            .into_response();
        };
        if let Err(err) = check_embeddings_model(req.model.as_deref(), embedding.model_id.as_ref())
        {
            emit_serve_event(
                &s.metrics,
                "POST",
                ROUTE,
                400,
                None,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
                Some(err.code()),
            );
            return err.into_response();
        }

        let embedding_model_id = embedding.model_id.clone();
        let (item_token_counts, item_pre_truncation_counts): (Vec<usize>, Vec<usize>) = {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            embedding
                .model
                .tokenizer()
                .tokenize_batch(&refs)
                .iter()
                .map(|t| (t.real_length, t.pre_truncation_len))
                .unzip()
        };
        let prompt_tokens: usize = item_token_counts.iter().sum();
        // Checked per item, before either `BertModel::encode_batch` path
        // runs: `encode_batch_packed` clamps an over-length item to
        // `max_position_embeddings` and would otherwise answer 200 for a
        // silently truncated embedding, and the single-item `encode` path
        // does not clamp at all. See `check_embeddings_item_fits_window`'s
        // doc comment. The guard tests `pre_truncation_len`, not
        // `real_length`: the tokenizer's own `max_seq_len` truncation runs
        // first, so `real_length` can never exceed it and the guard would be
        // unsatisfiable whenever `max_seq_len <= max_position_embeddings`.
        let max_position_embeddings = embedding.model.config().max_position_embeddings;
        if let Some((index, &token_count)) = item_pre_truncation_counts
            .iter()
            .enumerate()
            .find(|&(_, &count)| count > max_position_embeddings)
        {
            let err =
                check_embeddings_item_fits_window(index, token_count, max_position_embeddings)
                    .unwrap_err();
            emit_serve_event(
                &s.metrics,
                "POST",
                ROUTE,
                400,
                Some(prompt_tokens),
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
                Some(err.code()),
            );
            return err.into_response();
        }
        if let Err(err) = check_embeddings_total_tokens(prompt_tokens) {
            emit_serve_event(
                &s.metrics,
                "POST",
                ROUTE,
                400,
                Some(prompt_tokens),
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
                Some(err.code()),
            );
            return err.into_response();
        }
        let admission_permit = match try_acquire_embedding_slot(&embedding.admission) {
            Ok(permit) => permit,
            Err(err) => {
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    ROUTE,
                    503,
                    Some(prompt_tokens),
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some(err.code()),
                );
                return err.into_response();
            }
        };

        let encode_result = run_embedding_encode(admission_permit, move || {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            embedding.model.encode_batch(&refs)
        })
        .await;
        let vectors = match encode_result {
            Ok(Ok(vectors)) => vectors,
            Ok(Err(err)) => {
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    ROUTE,
                    500,
                    Some(prompt_tokens),
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some("internal_error"),
                );
                return err_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("embedding generation failed: {err}"),
                    "internal_error",
                );
            }
            Err(join_err) => {
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    ROUTE,
                    500,
                    Some(prompt_tokens),
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some("internal_error"),
                );
                return err_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("embedding worker task failed: {join_err}"),
                    "internal_error",
                );
            }
        };

        let body =
            build_embeddings_response(embedding_model_id.as_ref(), &vectors, prompt_tokens as u64);
        emit_serve_event(
            &s.metrics,
            "POST",
            ROUTE,
            200,
            Some(prompt_tokens),
            None,
            timer.elapsed().as_secs_f64() * 1000.0,
            false,
            None,
        );
        Json(body).into_response()
    }

    /// ADR-080 C2 (#782): builds the shared `lattice_inference::serve::
    /// ApiError` envelope instead of this binary's previous ad hoc 2-field
    /// `{"error": {"message", "type"}}` shape (no `code`/`param` at all) --
    /// `lattice.rs` already carried the 4-field OpenAI-style envelope; this
    /// closes the drift so both binaries answer the same bad request with
    /// the same JSON shape. Every existing call site passes only
    /// `StatusCode::BAD_REQUEST`, `PAYLOAD_TOO_LARGE`, or
    /// `INTERNAL_SERVER_ERROR`, so the status code produced here is
    /// unchanged; only the body shape gains `code`/`param`.
    /// `error_code` is the OpenAI-style code for the `BAD_REQUEST` branch
    /// (ADR-080 C2): previously hardcoded to the
    /// generic `"invalid_request"` regardless of what specifically failed,
    /// which is exactly how the `max_tokens: 0` rejection lost its
    /// `"invalid_max_tokens"` code on the way through this function. Ignored
    /// for the `PAYLOAD_TOO_LARGE`/`INTERNAL_SERVER_ERROR` branches, which
    /// carry their own fixed codes in `ApiError`'s `IntoResponse` impl.
    fn err_response(code: StatusCode, msg: &str, error_code: &'static str) -> Response {
        use lattice_inference::serve::ApiError;
        let api_err = if code == StatusCode::PAYLOAD_TOO_LARGE {
            ApiError::PayloadTooLarge {
                message: msg.to_string(),
            }
        } else if code == StatusCode::INTERNAL_SERVER_ERROR {
            ApiError::Internal {
                message: msg.to_string(),
            }
        } else {
            ApiError::BadRequest {
                message: msg.to_string(),
                code: error_code,
            }
        };
        api_err.into_response()
    }

    /// Prints a structured telemetry line to stdout for the app bridge to
    /// parse, AND records the same observation into the process's
    /// `/metrics` registry (issue #583) -- the single call site both
    /// consumers (the stdout bridge, the Prometheus scrape endpoint) share,
    /// so they can never observe a different request count for the same
    /// traffic.
    #[allow(clippy::too_many_arguments)]
    fn emit_serve_event(
        metrics: &ServeMetrics,
        method: &str,
        route: &str,
        status: u16,
        prompt_tokens: Option<usize>,
        completion_tokens: Option<usize>,
        dur_ms: f64,
        stream: bool,
        error_code: Option<&str>,
    ) {
        metrics.record_request(method, route, status, dur_ms / 1000.0);
        metrics.record_tokens(prompt_tokens.unwrap_or(0), completion_tokens.unwrap_or(0));
        if let Some(code) = error_code {
            metrics.record_error(code);
        }
        println!(
            "@@lattice {}",
            json!({
                "ev": "http_request",
                "method": method,
                "route": route,
                "status": status,
                "tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "dur_ms": dur_ms,
                "stream": stream,
            })
        );
    }

    fn unix_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    fn unix_nanos() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    }

    // ─── arg parsing + model resolution ──────────────────────────────────────

    fn parse_arg(args: &[String], flag: &str) -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    }

    /// Presence-only boolean flag (no value): `--preload-vision`, not
    /// `--preload-vision true`.
    fn parse_flag(args: &[String], flag: &str) -> bool {
        args.iter().any(|a| a == flag)
    }

    /// Parses `--max-pending`, rejecting a malformed or negative value
    /// outright instead of silently substituting the default (issue #939):
    /// the prior `.and_then(|s| s.parse().ok()).unwrap_or(DEFAULT)` chain
    /// could not distinguish "flag omitted" from "flag given garbage" --
    /// `--max-pending -1` silently became `32`. Range validation (zero, or
    /// above `Semaphore::MAX_PERMITS`) is deliberately NOT duplicated here:
    /// it happens once, downstream, in `MetalWorker::spawn`
    /// (`StartupError::InvalidMaxPending`), shared with `lattice.rs`.
    fn parse_max_pending(args: &[String]) -> Result<usize, String> {
        match parse_arg(args, "--max-pending") {
            Some(raw) => raw.parse::<usize>().map_err(|_| {
                format!("--max-pending: invalid value {raw:?} (expected a positive integer)")
            }),
            None => Ok(lattice_inference::serve::metal_worker::DEFAULT_MAX_PENDING_JOBS),
        }
    }

    fn parse_residency_limits(
        args: &[String],
    ) -> Result<lattice_inference::serve::lora::ResidencyLimits, String> {
        use lattice_inference::serve::lora::{ResidencyLimits, parse_resident_limit};
        let parse = |flag: &str, default: usize| {
            let Some(index) = args.iter().position(|arg| arg == flag) else {
                return Ok(default);
            };
            let raw = args
                .get(index + 1)
                .ok_or_else(|| format!("{flag}: expected a positive integer"))?;
            parse_resident_limit(raw).map_err(|err| format!("{flag}: {err}"))
        };
        let defaults = ResidencyLimits::default();
        Ok(ResidencyLimits {
            max_adapters: parse("--max-resident-adapters", defaults.max_adapters)?,
            max_bytes: parse("--max-resident-adapter-bytes", defaults.max_bytes)?,
        })
    }

    fn default_model_cache() -> std::path::PathBuf {
        std::env::var("LATTICE_MODEL_CACHE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_default();
                std::path::PathBuf::from(home)
                    .join(".lattice")
                    .join("models")
            })
    }

    fn resolve_model_dir(arg: &str) -> std::path::PathBuf {
        if let Some(rest) = arg.strip_prefix("~/")
            && let Ok(home) = std::env::var("HOME")
        {
            return std::path::PathBuf::from(home).join(rest);
        }
        let p = std::path::PathBuf::from(arg);
        if p.is_absolute() {
            p
        } else if p.components().count() == 1 {
            default_model_cache().join(arg)
        } else {
            p
        }
    }

    /// Builds the daemon's router in isolation from `run()`'s process
    /// startup (arg parsing, model loading, binding a listener) so tests --
    /// including the cross-binary parity table in
    /// `lattice_inference::serve::CHAT_COMPLETIONS_PARITY_CASES` -- can drive
    /// real HTTP requests through it via `tower::ServiceExt::oneshot`
    /// (ADR-080 C2). Deliberately does NOT install
    /// `DefaultBodyLimit` the way `lattice.rs`'s `router()` does: this
    /// binary enforces the same [`lattice_inference::serve::REQUEST_BODY_LIMIT_BYTES`]
    /// cap manually inside `chat_completions` via `to_bytes`, a documented
    /// intentional divergence in ENFORCEMENT MECHANISM only (axum's
    /// `DefaultBodyLimit` layer vs. a direct `to_bytes` cap) -- the
    /// resulting status/code (413 `request_body_too_large`) is identical on
    /// both binaries today (see the
    /// `oversized_body_over_limit` parity case).
    pub fn router<H: IntoServingHost>(state: AppState<H>) -> Router {
        Router::new()
            .route("/", get(root::<H>))
            .route("/health", get(health::<H>))
            .route("/v1/models", get(list_models::<H>))
            .route("/v1/chat/completions", post(chat_completions::<H>))
            .route("/v1/embeddings", post(embeddings::<H>))
            .route("/v1/lora", get(lora_list::<H>))
            .route("/v1/lora/load", post(lora_load::<H>))
            .route("/v1/lora/unload", post(lora_unload::<H>))
            .route("/metrics", get(metrics_handler::<H>))
            .with_state(state)
    }

    /// `POST /v1/lora/load`: load a PEFT or MLX LoRA adapter onto the running model.
    ///
    /// Everything that is not axum plumbing lives in
    /// [`lattice_inference::serve::lora`], shared with `lattice serve`'s route of the
    /// same name: one request parser, one file-to-command translation, one failure
    /// classification. The parse runs here, on the request's own task, so a malformed
    /// file costs the worker nothing; only a parsed `(layers, descriptor)` pair is
    /// ever submitted.
    ///
    /// Deployment boundary: the request names a filesystem path that this process
    /// reads. Every route on this server is unauthenticated, so on a non-loopback
    /// `--host` this route lets anyone who can reach the port make the server open a
    /// file of their choosing. The startup warning covers it; an allow-root is the
    /// obvious next control and is deliberately not invented here.
    async fn lora_list<H: IntoServingHost>(State(state): State<AppState<H>>) -> Response {
        let s = state.into_serving_host();
        lora_list_with_backend(
            || {
                if adapters_supported(&s) {
                    s.jobs.adapter_client_or(
                        lattice_inference::serve::prepare::lora_unsupported_backend(),
                    )
                } else {
                    Err(lattice_inference::serve::prepare::lora_unsupported_backend())
                }
            },
            lattice_inference::serving_worker::WorkerClient::adapter_index,
        )
    }

    /// Whether the loaded model takes runtime LoRA adapters. A worker with no
    /// model-bound preparation is the Qwen3.5 test worker, which does.
    fn adapters_supported<H: IntoServingHost>(s: &AppState<H>) -> bool {
        s.jobs.supports_adapters()
    }

    /// Repeated exact `(name, path)` identities share one resident id and lifetime:
    /// one unload removes it for all callers. Path aliases and changed contents
    /// are not detected. Capacity exhaustion rejects the load without eviction.
    async fn lora_load<H: IntoServingHost>(
        State(state): State<AppState<H>>,
        headers: HeaderMap,
        body: Body,
    ) -> Response {
        lora_load_with_adapter_support(
            State(state.into_serving_host()),
            headers,
            body,
            adapters_supported,
        )
        .await
    }

    async fn lora_load_with_adapter_support<H: IntoServingHost>(
        State(s): State<AppState<H>>,
        headers: HeaderMap,
        body: Body,
        supports_adapters: impl FnOnce(&AppState<H>) -> bool,
    ) -> Response {
        lora_load_with_backend(
            &s.metrics,
            headers,
            body,
            || {
                if supports_adapters(&s) {
                    s.jobs.adapter_client_or(
                        lattice_inference::serve::prepare::lora_unsupported_backend(),
                    )
                } else {
                    Err(lattice_inference::serve::prepare::lora_unsupported_backend())
                }
            },
            lattice_inference::serve::lora::prepare_adapter_load,
            |client, prepared| client.submit_adapter_command(prepared.command),
        )
        .await
    }

    /// Remove one resident identifier, refusing unknown ids.
    async fn lora_unload<H: IntoServingHost>(
        State(state): State<AppState<H>>,
        headers: HeaderMap,
        body: Body,
    ) -> Response {
        lora_unload_with_adapter_support(
            State(state.into_serving_host()),
            headers,
            body,
            adapters_supported,
        )
        .await
    }

    async fn lora_unload_with_adapter_support<H: IntoServingHost>(
        State(s): State<AppState<H>>,
        headers: HeaderMap,
        body: Body,
        supports_adapters: impl FnOnce(&AppState<H>) -> bool,
    ) -> Response {
        lora_unload_with_backend(
            &s.metrics,
            headers,
            body,
            || {
                if supports_adapters(&s) {
                    s.jobs.adapter_client_or(
                        lattice_inference::serve::prepare::lora_unsupported_backend(),
                    )
                } else {
                    Err(lattice_inference::serve::prepare::lora_unsupported_backend())
                }
            },
            |client, id| {
                client.submit_adapter_command(
                    lattice_inference::serve::metal_worker::AdapterCommand::Unload { id },
                )
            },
        )
        .await
    }

    async fn lora_load_with_backend<B, P>(
        metrics: &ServeMetrics,
        headers: HeaderMap,
        body: Body,
        resolve_backend: impl FnOnce() -> Result<B, lattice_inference::serve::ApiError>,
        prepare_load: impl FnOnce(&str, &str) -> Result<P, lattice_inference::serve::ApiError>,
        submit_load: impl FnOnce(
            B,
            P,
        ) -> Result<
            lattice_inference::serve::lora::AdapterControlReceiver,
            lattice_inference::serve::ApiError,
        >,
    ) -> Response {
        let timer = Instant::now();
        let result = lattice_inference::serve::lora::load_request(
            headers,
            body,
            Ok(()),
            resolve_backend,
            prepare_load,
            submit_load,
        )
        .await;
        lora_post_response(metrics, "/v1/lora/load", timer, result)
    }

    async fn lora_unload_with_backend<B>(
        metrics: &ServeMetrics,
        headers: HeaderMap,
        body: Body,
        resolve_backend: impl FnOnce() -> Result<B, lattice_inference::serve::ApiError>,
        submit_unload: impl FnOnce(
            B,
            u32,
        ) -> Result<
            lattice_inference::serve::lora::AdapterControlReceiver,
            lattice_inference::serve::ApiError,
        >,
    ) -> Response {
        let timer = Instant::now();
        let result = lattice_inference::serve::lora::unload_request(
            headers,
            body,
            Ok(()),
            resolve_backend,
            submit_unload,
        )
        .await;
        lora_post_response(metrics, "/v1/lora/unload", timer, result)
    }

    fn lora_post_response(
        metrics: &ServeMetrics,
        route: &str,
        timer: Instant,
        result: Result<Response, lattice_inference::serve::ApiError>,
    ) -> Response {
        let response = match result {
            Ok(response) => response,
            Err(error) => error.into_response(),
        };
        emit_serve_event(
            metrics,
            "POST",
            route,
            response.status().as_u16(),
            None,
            None,
            timer.elapsed().as_secs_f64() * 1000.0,
            false,
            None,
        );
        response
    }

    fn lora_list_with_backend<B>(
        resolve_backend: impl FnOnce() -> Result<B, lattice_inference::serve::ApiError>,
        snapshot: impl FnOnce(B) -> lattice_inference::serve::lora::AdapterIndex,
    ) -> Response {
        // Assembled by the shared helper so this binary and `lattice serve`
        // cannot answer the same route with different shapes (ADR-095
        // decision 4, item 4).
        //
        // `None` because this binary has no `--router-state` yet, so it
        // reports `{"enabled": false}`: the same shape, honestly filled. The
        // flags are the next step, not a silent omission.
        match lattice_inference::serve::lora::list_request(Ok(()), resolve_backend, snapshot, None)
        {
            Ok(response) => response,
            Err(error) => error.into_response(),
        }
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args: Vec<String> = std::env::args().collect();

        let model_arg = parse_arg(&args, "--model")
            .or_else(|| std::env::var("LATTICE_SERVE_MODEL").ok())
            .ok_or("missing --model <name-or-path> (e.g. --model qwen3.5-0.8b)")?;
        let model_dir = resolve_model_dir(&model_arg);
        if !model_dir.exists() {
            return Err(format!("model directory not found: {}", model_dir.display()).into());
        }
        let evidence = serving_provider::inspect_checkpoint(&model_dir);
        // The provider selects the route and preserves fallback loading for
        // checkpoints without a recognized format.
        let selected = match serving_provider::select(evidence, ServingEntry::Standalone) {
            Ok(selected) => selected,
            Err(error) => return Err(error.message().into()),
        };
        let tokenizer_path = parse_arg(&args, "--tokenizer-dir")
            .map(|d| std::path::Path::new(&d).join("tokenizer.json"))
            .unwrap_or_else(|| model_dir.join("tokenizer.json"));

        let host = parse_arg(&args, "--host").unwrap_or_else(|| "127.0.0.1".to_string());
        // `/metrics` (and every other route on this server) has no
        // authentication -- fine for the local-first default (`127.0.0.1`),
        // but a non-loopback `--host` exposes it to the network. Warn once
        // at startup rather than silently doing so; see the deployment-
        // boundary note on `metrics_handler` and `docs/serve-http-api.md`.
        if host != "127.0.0.1" && host != "localhost" && host != "::1" {
            eprintln!(
                "[lattice_serve] WARNING: --host {host} is not loopback; /metrics and every \
                 other route are unauthenticated -- do not expose this to an untrusted network \
                 without an external auth layer (reverse proxy, firewall)."
            );
        }
        let port: u16 = parse_arg(&args, "--port")
            .and_then(|s| s.parse().ok())
            .or_else(|| {
                std::env::var("LATTICE_SERVE_PORT")
                    .ok()
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(11435);

        let standard_defaults = GenerationDefaults::standard(512);
        let defaults = GenerationDefaults {
            max_tokens: parse_arg(&args, "--max-tokens")
                .and_then(|s| s.parse().ok())
                .unwrap_or(standard_defaults.max_tokens),
            temperature: parse_arg(&args, "--temperature")
                .and_then(|s| s.parse().ok())
                .unwrap_or(standard_defaults.temperature),
            top_k: parse_arg(&args, "--top-k")
                .and_then(|s| s.parse().ok())
                .unwrap_or(standard_defaults.top_k),
            top_p: parse_arg(&args, "--top-p")
                .and_then(|s| s.parse().ok())
                .unwrap_or(standard_defaults.top_p),
            repetition_penalty: parse_arg(&args, "--repetition-penalty")
                .and_then(|s| s.parse().ok())
                .unwrap_or(standard_defaults.repetition_penalty),
            reasoning_budget: parse_arg(&args, "--reasoning-budget")
                .and_then(|s| s.parse().ok())
                .filter(|&n| n > 0),
        };

        // Bounded admission cap on outstanding (queued + in-flight) jobs on
        // the shared Metal worker (issue #932) -- see
        // `metal_worker::DEFAULT_MAX_PENDING_JOBS`'s doc comment for why a
        // conservative default is correct even though this binary only ever
        // runs one generation at a time.
        let max_pending: usize = parse_max_pending(&args)?;
        let residency_limits = parse_residency_limits(&args)?;

        // issue #1336: eagerly load vision weights at startup instead of on
        // the first image request. Off by default -- lazy loading keeps
        // text-only startup time and resident memory unchanged; see
        // `lattice/main.rs`'s `--preload-vision` help text for the full
        // startup-time/memory tradeoff this trades away.
        let preload_vision = parse_flag(&args, "--preload-vision");

        selected.validate_standalone_options(
            ServingEntry::Standalone,
            &StandaloneOptionPresence {
                preload_vision,
                tokenizer_dir: args.iter().any(|arg| arg == "--tokenizer-dir"),
                resident_count: args.iter().any(|arg| arg == "--max-resident-adapters"),
                resident_bytes: args.iter().any(|arg| arg == "--max-resident-adapter-bytes"),
                positive_reasoning_budget: defaults.reasoning_budget.is_some(),
            },
        )?;
        let startup = selected.standalone_startup_description();
        if let Some(marker) = startup.route_marker {
            eprintln!("{marker}");
        }
        eprintln!(
            "[lattice_serve] loading model from {} ({}) ...",
            model_dir.display(),
            startup.loading_label
        );
        let plan = selected.prepare_standalone_load(StandaloneLoadOptions {
            tokenizer_path,
            preload_vision,
        })?;
        let started = plan.open(max_pending, residency_limits)?;
        let lattice_inference::serving_provider::StartedStandalone {
            owner,
            jobs,
            metadata,
            preparation: _preparation,
            post_ready,
        } = started;
        let fmt = metadata.format;
        let model_max_context = metadata.model_max_context;
        // Retain the explicit owner through the server lifetime. Every
        // production client also retains an owner clone, so on a normal
        // return whichever reference is dropped last performs the same
        // bounded join after the final job sender closes. That guarantee
        // does not reach the `std::process::exit` call on the server-error
        // path below (nor any other fatal exit in this file):
        // `std::process::exit` never runs destructors, so on those paths
        // this owner -- and any surviving client-held owner clone -- is
        // dropped without ever performing the bounded join.
        let _owner = owner;

        let model_id: Arc<str> = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("lattice")
            .into();
        eprintln!("[lattice_serve] model '{model_id}' ({fmt}) ready (context={model_max_context})");

        // Finish the provider's vocabulary work after the worker is ready.
        let vocab_bytes = post_ready.finish()?;
        let serving = lattice_inference::serving_host::ServingHandle::affine_standalone(jobs);

        // Issue #584: independent, optional second model for
        // `POST /v1/embeddings`. `BertModel` is CPU-only (see
        // `crate::model::bert`'s doc comment) so it loads directly here,
        // synchronously -- no `!Send` worker thread boundary applies to it.
        // `--embedding-model` is deliberately optional: an operator who only
        // wants chat completions should not have to point this flag at
        // anything, and the route itself stays registered either way (see
        // `AppState::embedding`'s doc comment).
        let embedding_state = match parse_arg(&args, "--embedding-model")
            .or_else(|| std::env::var("LATTICE_SERVE_EMBEDDING_MODEL").ok())
        {
            None => None,
            Some(embedding_model_arg) => {
                let embedding_model_dir = resolve_model_dir(&embedding_model_arg);
                Some(load_embedding_state(
                    &embedding_model_dir,
                    parse_arg(&args, "--embedding-pooling").as_deref(),
                )?)
            }
        };

        let state = AppState {
            jobs: serving,
            model_id,
            defaults,
            model_max_context,
            max_pending,
            metrics: Arc::new(ServeMetrics::default()),
            vocab_bytes,
            grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
            embedding: embedding_state,
        };

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        rt.block_on(async move {
            let app = router(state);
            let addr = format!("{host}:{port}");
            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .map_err(|e| format!("bind {addr} failed: {e}"))?;
            eprintln!("[lattice_serve] OpenAI-compatible API on http://{addr}/v1");
            eprintln!(
                "[lattice_serve]   POST /v1/chat/completions   POST /v1/embeddings   \
                 GET /v1/models   GET /health   GET /metrics"
            );
            println!("@@lattice {}", json!({"ev": "ready", "port": port}));
            if let Err(error) = lattice_inference::serve::serve_until_shutdown(listener, app).await
            {
                eprintln!("Server error: {error}");
                std::process::exit(1);
            }
            Ok::<(), String>(())
        })?;

        Ok(())
    }

    // ─── tests ───────────────────────────────────────────────────────────────

    #[cfg(test)]
    mod tests;
}

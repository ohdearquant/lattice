//! lattice_serve - OpenAI-compatible HTTP serving endpoint for Lattice. See [docs/capability-matrix.md](../../../../docs/capability-matrix.md).
//!
//! Exposes the Metal GPU engine over the same `/v1/chat/completions` API that
//! ollama, llama.cpp's server, and most LLM benchmark harnesses already speak,
//! so any OpenAI-compatible client can point at lattice with zero adapter code.
//!
//! # Usage
//!
//! ```text
//! lattice_serve --model qwen3.5-0.8b               # resolves from ~/.lattice/models
//! lattice_serve --model ~/.lattice/models/qwen3.6-27b-q4 --port 11435
//! ```
//!
//! Then point any OpenAI client at `http://127.0.0.1:11435/v1`:
//!
//! ```text
//! curl http://127.0.0.1:11435/v1/chat/completions -H 'content-type: application/json' \
//!   -d '{"model":"lattice","messages":[{"role":"user","content":"hi"}],"stream":true}'
//! ```
//!
//! # Endpoints
//!
//! - `POST /v1/chat/completions` — streaming (SSE) and non-streaming, OpenAI shape
//! - `GET  /v1/models`           — advertises the single loaded model
//! - `GET  /health`              — liveness probe (`ok`)
//! - `GET  /metrics`             — Prometheus text-format metrics (issue #583):
//!   request count + latency histogram by route/status, prompt/completion
//!   token counters, error count by code, queue-depth/in-flight gauge, all
//!   labeled with the loaded model id
//!
//! # Design
//!
//! `MetalQwen35State` owns raw `metal::*` objects and is `!Send`, so it lives on
//! one dedicated worker thread for the whole process lifetime, owned by
//! `lattice_inference::serve::metal_worker::MetalWorker` (issue #832: the
//! same shared owner module `lattice.rs`'s Metal backend uses). The async
//! axum handlers never touch Metal directly: each request is submitted via
//! `MetalWorkerClient::submit`, which ships a job (messages + sampling
//! config + a cancellation watch) to the worker over a tokio mpsc and
//! returns a `WorkerEvent` receiver; the worker drives cache-aware
//! generation, forwarding each token delta back as `WorkerEvent::Delta`.
//! Generation is therefore serialized — correct for a single-GPU local engine
//! (the same default ollama uses). The ChatML template and `<|im_end|>` stop
//! handling are reused verbatim from the engine; this binary only translates the
//! OpenAI wire format on either side.

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
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::grammar::{GrammarEngine, GrammarSpec};
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::{
        GenerateConfig, GenerateOutput, QWEN_CHAT_IM_END_TOKEN_ID, Qwen35Config,
    };
    use lattice_inference::model_format::{self, ModelFormat};
    use lattice_inference::serve::contract::{
        ChatRequest as ChatReq, GenerationDefaults, ServeProfile, ValidatedChatRequest,
        is_message_flood_error, message_flood_text, normalize_request,
    };
    #[cfg(test)]
    use lattice_inference::serve::contract::{
        ContentPart as Part, ImageUrl, Message as InMsg, MessageContent, normalize_messages,
    };
    use lattice_inference::serve::metal_worker::{
        ContextWindowPolicy, MetalWorker, MetalWorkerClient, StartupError, WorkerEvent,
        WorkerMetadata,
    };
    use lattice_inference::serve::metrics::ServeMetrics;
    /// Only used by the test module's `.tokenize(..)` calls on a real (tiny)
    /// tokenizer; production code never tokenizes outside the shared worker
    /// (`lattice_inference::serve::metal_worker`), hence the test feature gate.
    #[cfg(all(test, feature = "metal-gpu", feature = "test-utils"))]
    use lattice_inference::tokenizer::Tokenizer as _;
    use lattice_inference::tokenizer::bpe::BpeTokenizer;
    use serde_json::{Value, json};
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};
    /// Only used by the test module's raw job-channel helpers
    /// (`mpsc::UnboundedReceiver<WorkerJob>` etc. -- see
    /// `lattice_inference::serve::metal_worker`'s `test-utils`-gated
    /// surface); production code only ever holds a `MetalWorkerClient`.
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

    /// Server-side sampling defaults, overridable per-request.
    #[derive(Clone)]
    struct Defaults {
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        reasoning_budget: Option<usize>,
    }

    #[derive(Clone)]
    pub struct AppState {
        jobs: MetalWorkerClient,
        model_id: Arc<str>,
        defaults: Defaults,
        /// Runtime context window derived from the loaded model (#551): the
        /// exact KV cache length `load_model` allocated, never a hard-coded
        /// constant. See `model_context_from_config` and `build_cfg`.
        model_max_context: usize,
        /// The shared worker's outstanding-job admission cap (issue #932),
        /// needed alongside `jobs.available_permits()` to compute the
        /// `/metrics` (#583) queue-depth/in-flight gauge as `max_pending -
        /// available_permits()`.
        max_pending: usize,
        /// Process-wide Prometheus metrics registry (#583), `Arc`-shared so
        /// every clone of `AppState` (one per request) records into the same
        /// counters.
        metrics: Arc<ServeMetrics>,
        /// The loaded model's tokenizer vocabulary, byte-decoded once at
        /// startup (structured-output v0 design note): `GrammarEngine::new`
        /// needs `vocab_bytes[i]` for every token id to compile a schema
        /// into a PDA + vocabulary partition. `Arc`-shared so every request
        /// clones a pointer, not the ~vocab_size-entry `Vec<Vec<u8>>` --
        /// `GrammarEngine::new` itself still takes ownership of a fresh
        /// clone per compile (no cross-request grammar cache in v0; see the
        /// structured-output v0 design note's stage split).
        vocab_bytes: Arc<Vec<Vec<u8>>>,
        /// Bounded (32-entry, LRU) single-flight cache of compiled
        /// `GrammarEngine`s, keyed by canonicalized admitted schema JSON
        /// (structured-output v0 design note, stage 2). `Arc`-shared so
        /// every clone of `AppState` (one per request) hits the same cache.
        grammar_cache: Arc<GrammarCache>,
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
    /// #551 fallback when the loaded model's config has no derivable context.
    const FALLBACK_MODEL_MAX_CONTEXT: usize = 4096;

    /// #649: image input is accepted in the OpenAI wire shape but this server
    /// has no vision tower, so it must fail closed with a clear message
    /// rather than silently dropping the part or coercing it to text.
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
        /// A server-side malfunction discovered during admission (round-1
        /// review medium finding 4a: e.g. the grammar cache's own lock/slot
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
        ready: HashMap<String, Arc<GrammarEngine>>,
        order: VecDeque<String>,
        /// Keys currently being compiled by exactly one thread; every other
        /// concurrent request for the same key waits on the listed slot
        /// instead of starting a second, redundant compile.
        compiling: HashMap<String, Arc<CompileSlot>>,
    }

    /// Bounded LRU cache of compiled `GrammarEngine`s keyed by
    /// `canonical_schema_key`, with single-flight compilation (structured-
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
            key: String,
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
        // cache (stage 2): a schema already compiled by an earlier request
        // is served without touching the vocabulary clone or the compiler
        // at all; a schema currently being compiled by a concurrent request
        // is waited on instead of independently recompiled.
        let key = canonical_schema_key(schema);
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
        /// ADR-080 C2: differentiates the same
        /// two cases `lattice.rs`'s `ValidatedRole::parse` does -- `tool`/
        /// `developer` are real OpenAI roles this server does not implement
        /// (`unsupported_feature`), while anything else is not an OpenAI
        /// chat role at all (`invalid_role`). Previously both collapsed to
        /// one generic message with no code differentiation.
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

    /// Derive the model's usable context window from its config (#551):
    /// `max_position_embeddings` when a real `config.json` was loaded,
    /// otherwise (or when the field is non-positive) the documented
    /// fallback. Never a hard-coded constant divorced from the loaded model.
    fn model_context_from_config(cfg: Option<&Qwen35Config>) -> usize {
        cfg.map(|cfg| cfg.max_position_embeddings)
            .filter(|&n| n > 0)
            .unwrap_or(FALLBACK_MODEL_MAX_CONTEXT)
    }

    fn build_cfg(req: &ValidatedChatRequest) -> GenerateConfig {
        GenerateConfig {
            max_new_tokens: req.max_tokens,
            temperature: req.temperature,
            top_k: req.top_k,
            top_p: req.top_p,
            repetition_penalty: req.repetition_penalty,
            seed: req.seed,
            stop_token_ids: vec![QWEN_CHAT_IM_END_TOKEN_ID],
            enable_thinking: true,
            enable_mtp: None,
            grammar: None,
            stop_strings: req.stop_strings.clone(),
            reasoning_budget: req.reasoning_budget,
            logprobs: req.logprobs,
        }
    }

    // ─── GPU worker thread ───────────────────────────────────────────────────
    //
    // The dedicated thread that owns the `!Send` Metal state, the shared
    // FIFO/cancellation job loop, and the shared KV-window invariant check
    // all now live in `lattice_inference::serve::metal_worker` (issue #832),
    // replacing this binary's previous private `spawn_worker`/
    // `run_worker_loop`/`check_prompt_fits_window`/`enforce_prompt_window`.
    // `load_model`/`LoadedModel` below are unchanged: `run()` wraps
    // `load_model` in a loader closure passed to `MetalWorker::spawn`, which
    // runs it ON the worker thread it creates (the `!Send` `MetalQwen35State`
    // this function returns never crosses a thread boundary).

    /// Everything the worker thread needs after a successful model load,
    /// including the actual KV context (#551) so request clamping never
    /// drifts from what was actually allocated.
    struct LoadedModel {
        metal: MetalQwen35State,
        tokenizer: BpeTokenizer,
        format: String,
        model_max_context: usize,
    }

    fn load_model(
        model_dir: &std::path::Path,
        tokenizer_path: &std::path::Path,
        format: ModelFormat,
    ) -> Result<LoadedModel, String> {
        let tokenizer = BpeTokenizer::from_tokenizer_json(tokenizer_path)
            .map_err(|e| format!("tokenizer load failed ({}): {e}", tokenizer_path.display()))?;

        match format {
            ModelFormat::Q4 => {
                let cfg = Qwen35Config::from_model_dir(model_dir)
                    .map_err(|e| format!("config.json load failed: {e}"))?;
                let requested_context = model_context_from_config(Some(&cfg));
                let metal = MetalQwen35State::from_q4_dir(
                    model_dir,
                    tokenizer_path,
                    &cfg,
                    requested_context,
                )
                .map_err(|e| format!("Q4 model load failed: {e}"))?;
                let model_max_context = metal.max_context();
                Ok(LoadedModel {
                    metal,
                    tokenizer,
                    format: "q4".to_string(),
                    model_max_context,
                })
            }
            ModelFormat::Safetensors => {
                let model = Qwen35Model::from_safetensors(model_dir)
                    .map_err(|e| format!("safetensors load failed: {e}"))?;
                let cfg = model.config().clone();
                let requested_context = model_context_from_config(Some(&cfg));
                let metal = MetalQwen35State::new(model.weights(), &cfg, requested_context)
                    .map_err(|e| format!("Metal init failed: {e}"))?;
                let model_max_context = metal.max_context();
                Ok(LoadedModel {
                    metal,
                    tokenizer,
                    format: "bf16".to_string(),
                    model_max_context,
                })
            }
            ModelFormat::Unknown => Err(model_format::unrecognized_format_message(model_dir)),
        }
    }

    // ─── HTTP handlers ───────────────────────────────────────────────────────

    async fn health(State(s): State<AppState>) -> &'static str {
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

    async fn root(State(s): State<AppState>) -> Json<Value> {
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

    async fn list_models(State(s): State<AppState>) -> Json<Value> {
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
    async fn metrics_handler(State(s): State<AppState>) -> impl IntoResponse {
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
    async fn chat_completions(
        State(s): State<AppState>,
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
        let defaults = GenerationDefaults {
            max_tokens: s.defaults.max_tokens,
            temperature: s.defaults.temperature,
            top_k: s.defaults.top_k,
            top_p: s.defaults.top_p,
            repetition_penalty: s.defaults.repetition_penalty,
            reasoning_budget: s.defaults.reasoning_budget,
        };
        let (validated, ()) = match normalize_request(
            &req,
            defaults,
            ServeProfile::lattice_serve(s.model_id.as_ref(), s.model_max_context),
            |_, _| Ok(()),
        ) {
            Ok(validated) => validated,
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
        let mut cfg = build_cfg(&validated);
        // Wire the compiled grammar into the worker config (design note
        // §"End-to-end execution", step 4) and force `enable_thinking` off
        // for strict requests regardless of server defaults (sign-off Q5):
        // the grammar masks from the first decode token, so thinking
        // tokens are unsampleable under it and a think-then-constrain flow
        // is out of scope for v0.
        if let Some(structured) = &structured {
            cfg.grammar = Some(Arc::clone(&structured.engine));
            cfg.enable_thinking = false;
        }
        let streaming = validated.stream;
        let messages = validated.messages;
        let cfg = cfg;
        let model_id = s.model_id.to_string();
        let id = format!("chatcmpl-{}", unix_nanos());
        let created = unix_secs();

        let (cancel_guard, cancel_rx) = lattice_inference::serve::cancel_pair();
        // `MetalWorkerClient::submit` (issue #832) never fails outwardly
        // EXCEPT for admission (issue #932, checked synchronously here,
        // before any tokenization/model work): if the worker thread is no
        // longer running, the returned receiver simply closes with zero
        // events, which both branches below detect via their own
        // first-event peek (`rx.recv()` returning `None`) and report
        // identically to this binary's prior up-front `jobs.send(..).is_err()`
        // check.
        let mut rx = match s.jobs.submit(messages, cfg, cancel_rx) {
            Ok(rx) => rx,
            Err(api_err) => {
                emit_serve_event(
                    &s.metrics,
                    "POST",
                    "/v1/chat/completions",
                    503,
                    None,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                    Some(api_err.code()),
                );
                return api_err.into_response();
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
            // ADR-080 C2 / #832: mirrors the streaming branch's own
            // first-event peek above -- previously this binary's up-front
            // `jobs.send(..).is_err()` check was the ONLY thing catching a
            // dead/unavailable worker for this path; now that
            // `MetalWorkerClient::submit` never fails outwardly, a worker
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
                        // Structured-output v0 (design note, sign-off Q4): a
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
    pub fn router(state: AppState) -> Router {
        Router::new()
            .route("/", get(root))
            .route("/health", get(health))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/metrics", get(metrics_handler))
            .with_state(state)
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
        let format = model_format::detect_format(&model_dir);
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

        let defaults = Defaults {
            max_tokens: parse_arg(&args, "--max-tokens")
                .and_then(|s| s.parse().ok())
                .unwrap_or(512),
            temperature: parse_arg(&args, "--temperature")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.7),
            top_k: parse_arg(&args, "--top-k")
                .and_then(|s| s.parse().ok())
                .unwrap_or(50),
            top_p: parse_arg(&args, "--top-p")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.9),
            repetition_penalty: parse_arg(&args, "--repetition-penalty")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.1),
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

        eprintln!(
            "[lattice_serve] loading model from {} ({}) ...",
            model_dir.display(),
            match format {
                ModelFormat::Q4 => "q4",
                ModelFormat::Safetensors => "bf16",
                ModelFormat::Unknown => "unknown",
            }
        );
        // #832: `load_model` (unchanged) runs INSIDE this loader closure, on
        // the dedicated worker thread `MetalWorker::spawn` creates -- the
        // `!Send` `MetalQwen35State` it returns never crosses a thread
        // boundary. Blocks here until loading finishes (or fails), exactly
        // like this binary's prior `spawn_worker` + separate `ready` channel
        // did.
        let model_dir_for_loader = model_dir.clone();
        let tokenizer_path_for_vocab = tokenizer_path.clone();
        let (
            owner,
            jobs,
            WorkerMetadata {
                format: fmt,
                model_max_context,
                ..
            },
        ) = match MetalWorker::spawn(
            move || {
                let LoadedModel {
                    metal,
                    tokenizer,
                    format,
                    model_max_context,
                } = load_model(&model_dir_for_loader, &tokenizer_path, format)?;
                Ok((
                    metal,
                    tokenizer,
                    WorkerMetadata {
                        format,
                        model_max_context,
                        context_window_policy: ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                    },
                ))
            },
            max_pending,
        ) {
            Ok(triple) => triple,
            Err(StartupError::Load(e)) => return Err(e.into()),
            // Preserves this binary's exact prior wording (distinct from
            // `MetalWorker::spawn`'s own generic `StartupError::ThreadExited`
            // `Display` text) for the same condition: the worker thread
            // exited/panicked before ever sending a readiness signal.
            Err(StartupError::ThreadExited) => {
                return Err("worker thread exited during model load".into());
            }
            // #939: zero, or above `Semaphore::MAX_PERMITS` -- a
            // configuration error caught before `Semaphore::new` panics.
            Err(err @ StartupError::InvalidMaxPending { .. }) => {
                return Err(err.to_string().into());
            }
        };
        // Retains the worker thread's `JoinHandle` for the life of the
        // server (issue #833's seam: a future graceful-shutdown path has an
        // obvious place to join it). Neither this binary's prior bare
        // `mpsc::UnboundedSender<Job>` nor `lattice.rs`'s prior `MetalHandle`
        // ever joined or explicitly shut down their worker thread either --
        // the process exits and the OS reaps the detached thread -- so
        // today's behavior is unchanged.
        let _owner = owner;

        let model_id: Arc<str> = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("lattice")
            .into();
        eprintln!("[lattice_serve] model '{model_id}' ({fmt}) ready (context={model_max_context})");

        // Structured-output v0 (design note): the tokenizer vocabulary,
        // byte-decoded once here so `admit_structured_request` can compile
        // `GrammarEngine`s without touching the `!Send` worker thread.
        // Reloads the tokenizer and `config.json` a second time -- the
        // worker thread's `load_model` (above) already loaded both once,
        // inside a closure this async-setup code cannot reach across the
        // thread boundary. Stage 2 (the `GrammarCache` below) does NOT
        // remove this duplication: the underlying obstacle is the `!Send`
        // worker closure boundary, not the lack of a compile cache -- the
        // cache only avoids re-cloning `vocab_bytes` into `GrammarEngine`
        // on a cache HIT, it does not change where the first, authoritative
        // tokenizer/config load happens. Removing the second load would
        // require restructuring `load_model`'s ownership across that
        // thread boundary, out of stage 2's cache-placement scope.
        let vocab_bytes: Arc<Vec<Vec<u8>>> = {
            let tokenizer_for_vocab = BpeTokenizer::from_tokenizer_json(&tokenizer_path_for_vocab)
                .map_err(|e| {
                    format!(
                        "tokenizer load failed ({}): {e}",
                        tokenizer_path_for_vocab.display()
                    )
                })?;
            let vocab_size = Qwen35Config::from_model_dir(&model_dir)
                .map(|cfg| cfg.vocab_size)
                .map_err(|e| format!("config.json load failed: {e}"))?;
            Arc::new(
                tokenizer_for_vocab
                    .vocab_bytes(vocab_size)
                    .map_err(|e| format!("tokenizer vocab_bytes failed: {e}"))?,
            )
        };

        let state = AppState {
            jobs,
            model_id,
            defaults,
            model_max_context,
            max_pending,
            metrics: Arc::new(ServeMetrics::default()),
            vocab_bytes,
            grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
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
                "[lattice_serve]   POST /v1/chat/completions   GET /v1/models   GET /health   GET /metrics"
            );
            println!("@@lattice {}", json!({"ev": "ready", "port": port}));
            axum::serve(listener, app)
                .await
                .map_err(|e| format!("serve error: {e}"))?;
            Ok::<(), String>(())
        })?;

        Ok(())
    }

    // ─── tests ───────────────────────────────────────────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        use lattice_inference::serve::ApiError;
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        use lattice_inference::serve::metal_worker::{WorkerJob, spawn_fake, test_client_and_jobs};
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        use std::sync::Arc;

        /// `Content-Type: application/json` `HeaderMap`, for tests that call
        /// `chat_completions` directly as a Rust function (bypassing the
        /// router) and need to satisfy its `require_json_content_type`
        /// guard to exercise the behavior downstream of it.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        fn test_json_headers() -> HeaderMap {
            let mut headers = HeaderMap::new();
            headers.insert(
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("application/json"),
            );
            headers
        }

        fn normalize_for_cfg(
            req: &ChatReq,
            defaults: &Defaults,
            model_max_context: usize,
        ) -> ValidatedChatRequest {
            normalize_request(
                req,
                GenerationDefaults {
                    max_tokens: defaults.max_tokens,
                    temperature: defaults.temperature,
                    top_k: defaults.top_k,
                    top_p: defaults.top_p,
                    repetition_penalty: defaults.repetition_penalty,
                    reasoning_budget: defaults.reasoning_budget,
                },
                ServeProfile::lattice_serve("", model_max_context),
                |_, _| Ok(()),
            )
            .unwrap()
            .0
        }

        // NOTE (issue #832): the FIFO/cancellation/window-check worker-loop
        // test suite that used to live here (`fake_generate`,
        // `fake_generate_with_prefill_gap`, `make_job`,
        // `queued_job_cancelled_before_dequeue_is_skipped_entirely`,
        // `running_job_cancelled_midstream_stops_early_and_worker_survives`,
        // `running_job_cancelled_during_prefill_like_phase_never_calls_on_token`,
        // `fake_generate_fails_once_then_succeeds`,
        // `generation_failure_is_reported_as_ev_failed_not_ev_done`, and the
        // three `check_prompt_fits_window_*` tests further below) exercised
        // this binary's private `Job`/`Ev`/`run_worker_loop`/
        // `check_prompt_fits_window`, all of which moved to the shared
        // `lattice_inference::serve::metal_worker` module and are covered by
        // that module's own `#[cfg(test)] mod tests` instead (ported
        // verbatim, same assertions, same fake generators -- see
        // `crates/inference/src/serve/metal_worker.rs`). The HTTP-level
        // tests below still exercise THIS binary's real `chat_completions`,
        // now routed through `MetalWorkerClient`/`WorkerEvent` via the
        // `metal_worker` `test-utils` seam (`test_client_and_jobs`/
        // `spawn_fake`) instead of this binary's own private job channel.

        /// #832 checklist: "Add a Metal-feature test asserting an explicit
        /// common-worker marker, so a silently-reintroduced per-binary
        /// fallback cannot pass green." `AppState.jobs`'s field type is
        /// private, so this test can only be written from inside `mod imp`
        /// (same module as `AppState`, so private-field access is allowed)
        /// -- it constructs an `AppState` directly from a
        /// `lattice_inference::serve::metal_worker::MetalWorkerClient`,
        /// which only type-checks if `AppState.jobs` still holds that exact
        /// shared type. A private per-binary fallback job channel (any type
        /// other than the shared `MetalWorkerClient`) would fail to compile
        /// here, not just fail some behavioral assertion at runtime. See
        /// `lattice.rs`'s
        /// `metal_handle_is_backed_by_the_shared_metal_worker_client` for
        /// the sibling marker on the other binary.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[test]
        fn app_state_is_backed_by_the_shared_metal_worker_client() {
            fn build_from_shared_client(jobs: MetalWorkerClient) -> AppState {
                AppState {
                    jobs,
                    model_id: Arc::from("marker-test-model"),
                    defaults: Defaults {
                        max_tokens: 100,
                        temperature: 0.7,
                        top_k: 50,
                        top_p: 0.9,
                        repetition_penalty: 1.1,
                        reasoning_budget: None,
                    },
                    model_max_context: 4096,
                    max_pending: 1_000_000,
                    metrics: Arc::new(ServeMetrics::default()),
                    vocab_bytes: Arc::new(vec![]),
                    grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
                }
            }
            let (jobs, _rx) = test_client_and_jobs();
            let _state: AppState = build_from_shared_client(jobs);
        }

        // ── #641 / #649 request parsing and clamp tests ──────────────────

        #[test]
        fn message_content_plain_string_to_chat_message() {
            let msg = InMsg {
                role: "user".to_string(),
                content: MessageContent::Text("hi".to_string()),
            };
            let chat_message = normalize_messages(std::slice::from_ref(&msg))
                .expect("plain string content must parse");
            assert_eq!(
                chat_message[0].role,
                lattice_inference::forward::metal_qwen35::ChatRole::User
            );
            assert_eq!(chat_message[0].content, "hi");
        }

        #[test]
        fn message_content_parts_concatenate_in_order() {
            let msg = InMsg {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![
                    Part::Text {
                        text: "a".to_string(),
                    },
                    Part::Text {
                        text: "b".to_string(),
                    },
                ]),
            };
            let chat_message = normalize_messages(std::slice::from_ref(&msg)).unwrap();
            assert_eq!(chat_message[0].content, "ab");
        }

        #[test]
        fn message_content_image_url_rejected() {
            let msg = InMsg {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![Part::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/cat.png".to_string(),
                        detail: None,
                    },
                }]),
            };
            let err = normalize_messages(std::slice::from_ref(&msg)).unwrap_err();
            assert_eq!(err.message(), IMAGE_REQUIRES_VISION_MESSAGE);
        }

        #[test]
        fn message_content_unknown_part_rejected() {
            // The shared normalizer's wording ("content part type 'file' is
            // not supported; ...") differs from this binary's now-removed
            // local test-only copy ("unsupported content part type 'file';
            // ..."); production always went through the shared normalizer
            // (`normalize_request`), so this pins the wording clients
            // actually receive.
            let msg = InMsg {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![Part::Unsupported {
                    kind: "file".to_string(),
                }]),
            };
            let err = normalize_messages(std::slice::from_ref(&msg)).unwrap_err();
            assert_eq!(
                err.message(),
                "content part type 'file' is not supported; only 'text' parts are accepted"
            );
        }

        #[test]
        fn message_role_unknown_rejected() {
            // Not an OpenAI chat role at all -- `invalid_role`, matching
            // `lattice.rs`'s `ValidatedRole::parse` for the same case.
            let err = MessageRole::parse("moderator").unwrap_err();
            assert_eq!(
                err.message(),
                "unsupported role 'moderator'; must be 'system', 'user', or 'assistant'"
            );
            assert_eq!(err.code(), "invalid_role");
        }

        #[test]
        fn message_role_tool_and_developer_rejected_as_unsupported_feature() {
            // A real OpenAI role this server does not implement --
            // `unsupported_feature`, matching `lattice.rs`'s split between
            // "not a role" and "a role we don't support" (ADR-080 C2).
            for role in ["tool", "developer"] {
                let err = MessageRole::parse(role).unwrap_err();
                assert_eq!(err.code(), "unsupported_feature");
            }
        }

        #[test]
        fn parse_chat_req_rejects_too_many_parts_before_typed_parse() {
            let parts: Vec<String> = (0..65)
                .map(|i| format!(r#"{{"type":"text","text":"p{i}"}}"#))
                .collect();
            let body = format!(
                r#"{{"messages":[{{"role":"user","content":[{}]}}]}}"#,
                parts.join(",")
            );
            let err = parse_chat_req(body.as_bytes()).unwrap_err();
            assert_eq!(
                err.message(),
                "messages[0].content has too many parts; maximum is 64"
            );
        }

        /// #656 boundary fix: exactly `MAX_CONTENT_PARTS_PER_MESSAGE` (64)
        /// parts is the documented maximum and MUST be accepted — only
        /// MAX+1 (covered above) is rejected.
        #[test]
        fn parse_chat_req_accepts_max_parts_boundary() {
            assert_eq!(MAX_CONTENT_PARTS_PER_MESSAGE, 64);
            let parts: Vec<String> = (0..MAX_CONTENT_PARTS_PER_MESSAGE)
                .map(|i| format!(r#"{{"type":"text","text":"p{i}"}}"#))
                .collect();
            let body = format!(
                r#"{{"messages":[{{"role":"user","content":[{}]}}]}}"#,
                parts.join(",")
            );
            let req = parse_chat_req(body.as_bytes()).expect("exactly 64 parts must be accepted");
            assert_eq!(req.messages.len(), 1);
        }

        #[test]
        fn parse_chat_req_rejects_oversized_text_part_before_typed_parse() {
            let big_text = "x".repeat(MAX_CONTENT_PART_BYTES + 1);
            let body = format!(
                r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"{big_text}"}}]}}]}}"#,
            );
            let err = parse_chat_req(body.as_bytes()).unwrap_err();
            assert_eq!(err.message(), "messages[0].content[0] exceeds 65536 bytes");
        }

        #[test]
        fn build_cfg_clamps_to_runtime_context() {
            let defaults = Defaults {
                max_tokens: 100,
                temperature: 0.7,
                top_k: 50,
                top_p: 0.9,
                repetition_penalty: 1.1,
                reasoning_budget: Some(50),
            };
            let req = ChatReq {
                model: None,
                messages: vec![InMsg {
                    role: "user".to_string(),
                    content: MessageContent::Text("hi".to_string()),
                }],
                temperature: None,
                top_p: None,
                top_k: None,
                max_tokens: None,
                seed: None,
                stream: None,
                repetition_penalty: None,
                reasoning_budget: Some(
                    serde_json::value::RawValue::from_string("50".to_string()).unwrap(),
                ),
                max_completion_tokens: None,
                tools: None,
                tool_choice: None,
                response_format: None,
                n: None,
                logprobs: None,
                top_logprobs: None,
                stop: None,
            };
            let normalized = normalize_for_cfg(&req, &defaults, 8);
            let cfg = build_cfg(&normalized);
            assert!(cfg.max_new_tokens <= 8);
            let reasoning_budget = cfg.reasoning_budget.unwrap_or(0);
            assert!(reasoning_budget + cfg.max_new_tokens < 8);
        }

        // NOTE (issue #832): the three `check_prompt_fits_window_*` tests
        // that used to live here (`build_cfg`-clamped `GenerateConfig`
        // values fed into this binary's private `check_prompt_fits_window`)
        // no longer compile: the check moved to
        // `lattice_inference::serve::metal_worker`, which does not expose it
        // (it is a private implementation detail of the shared worker loop,
        // not part of that module's public/test-utils surface). The same
        // math is covered by `check_prompt_fits_window`'s own pure unit
        // tests, ported verbatim to `metal_worker.rs`'s test suite; the same
        // `build_cfg` -> worker -> window-check composition this file's
        // deleted tests exercised is covered end to end, through a REAL
        // worker and a real request, by `real_router_overflow_parity` and
        // `parity_table` below. `build_cfg`'s own clamping arithmetic (the
        // half of the invariant that stays in THIS file) keeps its direct
        // coverage in `build_cfg_clamps_to_runtime_context` above.

        #[test]
        fn model_context_uses_config_max_position_embeddings() {
            let mut cfg = Qwen35Config::qwen35_2b();
            cfg.max_position_embeddings = 12345;
            assert_eq!(model_context_from_config(Some(&cfg)), 12345);
        }

        #[test]
        fn model_context_falls_back_to_4096_when_absent() {
            assert_eq!(model_context_from_config(None), FALLBACK_MODEL_MAX_CONTEXT);
        }

        // ── #939 `--max-pending` daemon-arg boundary tests ────────────────
        //
        // `parse_max_pending` only rejects a MALFORMED string here (this
        // binary's own hand-rolled argv parser, unlike `lattice.rs`'s clap
        // `value_parser`); zero and above-`Semaphore::MAX_PERMITS` are valid
        // `usize`s that parse fine and are instead caught once, downstream,
        // by `MetalWorker::spawn`'s own `StartupError::InvalidMaxPending`
        // (covered by that module's own boundary tests).

        #[test]
        fn max_pending_omitted_defaults_to_32() {
            let args: Vec<String> = vec![];
            assert_eq!(
                parse_max_pending(&args).expect("no --max-pending flag"),
                lattice_inference::serve::metal_worker::DEFAULT_MAX_PENDING_JOBS,
            );
        }

        #[test]
        fn max_pending_valid_override_is_accepted() {
            let args: Vec<String> = vec!["--max-pending".to_string(), "8".to_string()];
            assert_eq!(parse_max_pending(&args).expect("8 is well-formed"), 8);
        }

        #[test]
        fn max_pending_negative_is_rejected_not_silently_defaulted() {
            let args: Vec<String> = vec!["--max-pending".to_string(), "-1".to_string()];
            let err = parse_max_pending(&args)
                .expect_err("-1 must be rejected, not silently replaced by the default");
            assert!(
                err.contains("-1"),
                "error must name the offending value: {err}"
            );
        }

        #[test]
        fn max_pending_malformed_is_rejected_not_silently_defaulted() {
            let args: Vec<String> = vec!["--max-pending".to_string(), "not-a-number".to_string()];
            let err = parse_max_pending(&args)
                .expect_err("a non-numeric value must be rejected, not silently replaced");
            assert!(
                err.contains("not-a-number"),
                "error must name the offending value: {err}"
            );
        }

        #[test]
        fn max_pending_zero_parses_ok_and_is_left_to_metal_worker_spawn_to_reject() {
            // Zero is a syntactically valid `usize` -- `parse_max_pending`
            // itself does not range-check it (that check lives once, in
            // `MetalWorker::spawn`, shared with `lattice.rs`). This test
            // pins that division of responsibility down: it must NOT start
            // rejecting zero itself without `MetalWorker::spawn`'s own
            // boundary tests changing too.
            let args: Vec<String> = vec!["--max-pending".to_string(), "0".to_string()];
            assert_eq!(
                parse_max_pending(&args).expect("0 parses as a valid usize"),
                0
            );
        }

        // ── HTTP-level 400 tests ──────────────────────────────────────────
        //
        // All three failure modes below (`#641` unknown role, `#649` image
        // part, `#649` oversized part) return from `chat_completions` before
        // a job is ever submitted to `s.jobs`, so a `MetalWorkerClient` with
        // no running worker behind it (issue #832's `test_client_and_jobs`
        // seam, receiver half discarded) is a faithful stand-in: no GPU, no
        // model load.

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        fn test_app_state() -> AppState {
            let (jobs, _rx) = test_client_and_jobs();
            AppState {
                jobs,
                model_id: Arc::from("test-model"),
                defaults: Defaults {
                    max_tokens: 100,
                    temperature: 0.7,
                    top_k: 50,
                    top_p: 0.9,
                    repetition_penalty: 1.1,
                    reasoning_budget: None,
                },
                model_max_context: 4096,
                max_pending: 1_000_000,
                metrics: Arc::new(ServeMetrics::default()),
                vocab_bytes: Arc::new(vec![]),
                grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
            }
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        async fn error_message_of(response: Response) -> (StatusCode, String) {
            let status = response.status();
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("response body must be readable");
            let value: serde_json::Value =
                serde_json::from_slice(&body).expect("error response must be valid JSON");
            let message = value["error"]["message"]
                .as_str()
                .expect("error response must carry error.message")
                .to_string();
            (status, message)
        }

        /// Like `error_message_of`, plus the OpenAI-style `error.code` (used
        /// by the structured-output v0 tests to assert the exact machine
        /// code -- `unsupported_strict_schema`, `blocked_constraint`,
        /// `validation_failed`, `length_limit` -- not just the status).
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        async fn error_code_of(response: Response) -> (StatusCode, String) {
            let status = response.status();
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("response body must be readable");
            let value: serde_json::Value =
                serde_json::from_slice(&body).expect("error response must be valid JSON");
            let code = value["error"]["code"]
                .as_str()
                .expect("error response must carry error.code")
                .to_string();
            (status, code)
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_unknown_role_400() {
            // A role string that is not an OpenAI chat role at all (as
            // opposed to "developer"/"tool", which ARE real OpenAI roles
            // this server just doesn't implement -- see
            // `chat_completions_tool_and_developer_role_400_unsupported_feature`
            // below for that split, ADR-080 C2).
            let body =
                Body::from(r#"{"messages":[{"role":"moderator","content":"hi"}]}"#.to_string());
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                message,
                "unsupported role 'moderator'; must be 'system', 'user', or 'assistant'"
            );
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_tool_and_developer_role_400_unsupported_feature() {
            for role in ["tool", "developer"] {
                let body = Body::from(format!(
                    r#"{{"messages":[{{"role":"{role}","content":"hi"}}]}}"#
                ));
                let response =
                    chat_completions(State(test_app_state()), test_json_headers(), body).await;
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .expect("response body must be readable");
                let v: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("response body must be valid JSON");
                assert_eq!(
                    v["error"]["code"], "unsupported_feature",
                    "role '{role}' must be reported as unsupported_feature, not invalid_role"
                );
            }
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_image_url_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":[
                    {"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}
                ]}]}"#
                    .to_string(),
            );
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(message, IMAGE_REQUIRES_VISION_MESSAGE);
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_oversized_part_400() {
            let big_text = "x".repeat(MAX_CONTENT_PART_BYTES + 1);
            let body = Body::from(format!(
                r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"{big_text}"}}]}}]}}"#,
            ));
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(message, "messages[0].content[0] exceeds 65536 bytes");
        }

        // ── #656: known-but-unsupported OpenAI field rejection ────────────
        //
        // Same fail-closed contract as the three tests above: these fields
        // are parsed (not silently dropped by serde's unknown-field
        // default), so a client asking for tool calls / JSON mode / N
        // completions / stop sequences gets an explicit 400 naming the
        // unsupported field instead of a plain-text completion that quietly
        // ignored the request's actual contract.

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_tools_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],
                    "tools":[{"type":"function","function":{"name":"f"}}]}"#
                    .to_string(),
            );
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                message,
                "tools and tool_choice are not supported by this server"
            );
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_tool_choice_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"tool_choice":"auto"}"#.to_string(),
            );
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                message,
                "tools and tool_choice are not supported by this server"
            );
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_json_response_format_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],
                    "response_format":{"type":"json_object"}}"#
                    .to_string(),
            );
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                message,
                "response_format.type 'json_object' is not supported; use 'text'"
            );
        }

        // ── structured-output v0 (design note) ─────────────────────────────

        /// A synthetic vocabulary sized for `V0_ROUTE_SCHEMA` specifically:
        /// literal multi-byte tokens for exactly the bytes a closed
        /// `{"ok": <boolean>}` value needs. Decoupled from whatever
        /// tokenizer a given test's fake worker uses for prompt-side
        /// tokenization -- `AppState.vocab_bytes` and the worker's
        /// `BpeTokenizer` are two independent concerns. Deliberately NOT a
        /// full byte-range vocabulary: `GrammarEngine::new`'s reachable-
        /// state BFS fans out per distinguishable token at every state, and
        /// an open `type: "string"` property (arbitrary content) combined
        /// with a wide single-byte vocabulary blows `MAX_GRAMMAR_STATES`
        /// even for a schema this small -- a real tokenizer's vocabulary is
        /// large but mostly irrelevant to any one schema's few reachable
        /// states, which a literal-token synthetic vocab does not model.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        fn route_test_vocab() -> Arc<Vec<Vec<u8>>> {
            Arc::new(
                ["{", "}", "\"", ":", "ok", "true", "false", ","]
                    .iter()
                    .map(|s| s.as_bytes().to_vec())
                    .collect(),
            )
        }

        const V0_OBJECT_SCHEMA: &str = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}"#;
        /// Compiles within `route_test_vocab`'s small literal-token budget
        /// (see that function's doc comment): used only by tests that
        /// actually run admission through `GrammarEngine::new`, as opposed
        /// to `V0_OBJECT_SCHEMA`, which is used by tests that stop at pure
        /// schema-shape validation (`admit_v0_schema`) or reject before
        /// compilation is ever attempted.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        const V0_ROUTE_SCHEMA: &str = r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}"#;

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        fn structured_body(schema: &str, strict: Option<&str>, stream: Option<bool>) -> String {
            let strict_field = match strict {
                Some(raw) => format!(r#","strict":{raw}"#),
                None => String::new(),
            };
            let stream_field = match stream {
                Some(v) => format!(r#","stream":{v}"#),
                None => String::new(),
            };
            format!(
                r#"{{"messages":[{{"role":"user","content":"hi"}}]{stream_field},
                    "response_format":{{"type":"json_schema","json_schema":{{
                        "name":"result"{strict_field},"schema":{schema}}}}}}}"#
            )
        }

        /// Regression: a duplicate root `"type"`
        /// member inside the submitted schema must be rejected with 400
        /// before any worker job is submitted. `test_app_state()`'s worker
        /// receiver is dropped immediately, so a request that reached
        /// admission/job-submission would simply have its `send` silently
        /// fail rather than this test observing a clean 400 -- reaching the
        /// 400 assertion is itself proof no job was submitted.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_rejects_duplicate_root_type_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],
                    "response_format":{"type":"json_schema","json_schema":{
                        "name":"result","strict":true,
                        "schema":{"type":"boolean","type":"string"}}}}"#
                    .to_string(),
            );
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, code) = error_code_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(code, "invalid_request_body");
        }

        /// Regression: a duplicate nested
        /// `"properties"` member inside an object schema must be rejected
        /// with 400 before any worker job is submitted.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_rejects_duplicate_nested_properties_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],
                    "response_format":{"type":"json_schema","json_schema":{
                        "name":"result","strict":true,
                        "schema":{"type":"object",
                            "properties":{"a":{"type":"boolean"}},
                            "properties":{"a":{"type":"string"}},
                            "required":["a"],"additionalProperties":false}}}}"#
                    .to_string(),
            );
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, code) = error_code_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(code, "invalid_request_body");
        }

        /// Regression: a duplicate key inside a
        /// nested `properties.<name>` schema object must be rejected with
        /// 400 before any worker job is submitted.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_rejects_duplicate_key_inside_property_schema_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],
                    "response_format":{"type":"json_schema","json_schema":{
                        "name":"result","strict":true,
                        "schema":{"type":"object",
                            "properties":{"a":{"type":"boolean","type":"string"}},
                            "required":["a"],"additionalProperties":false}}}}"#
                    .to_string(),
            );
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, code) = error_code_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(code, "invalid_request_body");
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_structured_missing_strict_400() {
            let body = Body::from(structured_body(V0_OBJECT_SCHEMA, None, None));
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(message.contains("strict must be true"), "{message}");
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_structured_strict_false_400() {
            let body = Body::from(structured_body(V0_OBJECT_SCHEMA, Some("false"), None));
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(message.contains("strict must be true"), "{message}");
        }

        /// Spec §"Accepted request profile" item 1: `stream: true` combined
        /// with a structured request is rejected before a worker job is
        /// submitted (`test_app_state()`'s worker has no running loop
        /// behind it at all -- a job send here would simply vanish silently
        /// rather than this test observing a 400, so reaching the 400
        /// assertion below is itself proof admission ran first).
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_structured_streaming_400() {
            let body = Body::from(structured_body(V0_OBJECT_SCHEMA, Some("true"), Some(true)));
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(message.contains("non-streaming"), "{message}");
        }

        /// Rejection coverage per unsupported keyword (design note
        /// §"Correctness tests", item 1), driven through the real HTTP
        /// route (not just the pure `admit_v0_schema` unit tests below) so
        /// the `unsupported_strict_schema` code and pre-worker-submission
        /// contract are both proven at the boundary the design note names.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_structured_pattern_keyword_400() {
            let schema = r#"{"type":"object","properties":{"name":{"type":"string","pattern":"^a"}},"required":["name"],"additionalProperties":false}"#;
            let body = Body::from(structured_body(schema, Some("true"), None));
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, code) = error_code_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(code, "unsupported_strict_schema");
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_structured_enum_keyword_400() {
            let schema = r#"{"type":"string","enum":["a","b"]}"#;
            let body = Body::from(structured_body(schema, Some("true"), None));
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, code) = error_code_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(code, "unsupported_strict_schema");
        }

        /// Route test (design note §"Serve and backend integration", item
        /// 1): an admitted strict request's worker job carries
        /// `grammar.is_some()`, `enable_thinking == false` (sign-off Q5),
        /// and the successful response carries `constraint_applied` --
        /// asserted from a real, compiled `GrammarEngine`, not a stub, so a
        /// route that merely returns schema-shaped JSON without actually
        /// wiring the grammar would fail this test's `grammar.is_some()`
        /// assertion even before the marker check.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_structured_applies_grammar_and_marker() {
            let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                .tokenizer()
                .clone();
            let jobs = spawn_fake(
                ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                4096,
                tokenizer,
                move |_messages, cfg, _prompt_tokens, on_token, _should_cancel| {
                    assert!(
                        cfg.grammar.is_some(),
                        "structured request must attach a compiled grammar to GenerateConfig"
                    );
                    assert!(
                        !cfg.enable_thinking,
                        "strict structured requests must force enable_thinking off (Q5)"
                    );
                    // The route handler builds response `content` from
                    // streamed `Delta` events, not `GenerateOutput::text`
                    // directly -- mirror that by pushing the delta through
                    // `on_token` before returning, exactly like a real
                    // generate closure would.
                    on_token(r#"{"ok":true}"#, 0);
                    Ok(GenerateOutput {
                        text: r#"{"ok":true}"#.to_string(),
                        token_ids: vec![0],
                        prompt_tokens: 1,
                        generated_tokens: 1,
                        stopped: true,
                        stop_reason: Some(lattice_inference::StopReason::Eos),
                        token_logprobs: vec![],
                    })
                },
            );
            let state = AppState {
                jobs,
                model_id: Arc::from("test-model"),
                defaults: Defaults {
                    max_tokens: 100,
                    temperature: 0.7,
                    top_k: 50,
                    top_p: 0.9,
                    repetition_penalty: 1.1,
                    reasoning_budget: None,
                },
                model_max_context: 4096,
                max_pending: 1_000_000,
                metrics: Arc::new(ServeMetrics::default()),
                vocab_bytes: route_test_vocab(),
                grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
            };
            let body = Body::from(structured_body(V0_ROUTE_SCHEMA, Some("true"), None));
            let response = chat_completions(State(state), test_json_headers(), body).await;
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("response body must be readable");
            let value: serde_json::Value =
                serde_json::from_slice(&bytes).expect("response must be valid JSON");
            assert_eq!(
                value["constraint_applied"],
                serde_json::Value::Bool(true),
                "a route returning valid JSON without the constraint_applied marker must fail: {value}"
            );
            assert_eq!(value["choices"][0]["message"]["content"], r#"{"ok":true}"#);
        }

        /// Sign-off Q4: a `WorkerEvent::ConstraintBlocked` event must surface
        /// as HTTP 500 with the `blocked_constraint` machine code for a
        /// structured request, never a 200 with partial/absent JSON. Round-1
        /// review medium finding 2: replies via the raw `WorkerJob::reply`
        /// seam (not `spawn_fake`'s `Result<_, String>` closure, which can
        /// only ever produce a generic `WorkerEvent::Failed`) so this test
        /// exercises the real production classification -- a distinct
        /// `WorkerEvent` variant, not a message-text sniff.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_structured_blocked_constraint_500() {
            let (jobs, mut jobs_rx) = test_client_and_jobs();
            tokio::spawn(async move {
                if let Some(job) = jobs_rx.recv().await {
                    let _ = job.reply(WorkerEvent::ConstraintBlocked(
                        "grammar constraint blocked every token; \
                         no legal continuation exists in the current grammar state"
                            .to_string(),
                    ));
                }
            });
            let state = AppState {
                jobs,
                model_id: Arc::from("test-model"),
                defaults: Defaults {
                    max_tokens: 100,
                    temperature: 0.7,
                    top_k: 50,
                    top_p: 0.9,
                    repetition_penalty: 1.1,
                    reasoning_budget: None,
                },
                model_max_context: 4096,
                max_pending: 1_000_000,
                metrics: Arc::new(ServeMetrics::default()),
                vocab_bytes: route_test_vocab(),
                grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
            };
            let body = Body::from(structured_body(V0_ROUTE_SCHEMA, Some("true"), None));
            let response = chat_completions(State(state), test_json_headers(), body).await;
            let (status, code) = error_code_of(response).await;
            assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(code, "blocked_constraint");
        }

        /// A `WorkerEvent::Failed` (the
        /// generic variant) whose message text happens to contain the
        /// engine's grammar-exhausted-mask wording must NOT be classified as
        /// `blocked_constraint` -- only the distinct `ConstraintBlocked`
        /// variant may. Mutation-sensitive companion to the test above: if
        /// the classification ever regresses back to sniffing `message`,
        /// this test starts failing (this exact message string used to
        /// produce `blocked_constraint` under the old `.contains(..)` check;
        /// it must now produce `internal_error`).
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_structured_failed_with_blocked_wording_stays_internal_error() {
            let (jobs, mut jobs_rx) = test_client_and_jobs();
            tokio::spawn(async move {
                if let Some(job) = jobs_rx.recv().await {
                    let _ = job.reply(WorkerEvent::Failed(
                        "grammar constraint blocked every token; \
                         no legal continuation exists in the current grammar state"
                            .to_string(),
                    ));
                }
            });
            let state = AppState {
                jobs,
                model_id: Arc::from("test-model"),
                defaults: Defaults {
                    max_tokens: 100,
                    temperature: 0.7,
                    top_k: 50,
                    top_p: 0.9,
                    repetition_penalty: 1.1,
                    reasoning_budget: None,
                },
                model_max_context: 4096,
                max_pending: 1_000_000,
                metrics: Arc::new(ServeMetrics::default()),
                vocab_bytes: route_test_vocab(),
                grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
            };
            let body = Body::from(structured_body(V0_ROUTE_SCHEMA, Some("true"), None));
            let response = chat_completions(State(state), test_json_headers(), body).await;
            let (status, code) = error_code_of(response).await;
            assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(code, "internal_error");
        }

        /// Sign-off Q4: `stopped == false` (length/KV-window exhaustion)
        /// on a structured request must surface as HTTP 500
        /// `length_limit`, never HTTP 200 with truncated JSON.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_structured_length_limit_500() {
            let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                .tokenizer()
                .clone();
            let jobs = spawn_fake(
                ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                4096,
                tokenizer,
                move |_messages, _cfg, _prompt_tokens, _on_token, _should_cancel| {
                    Ok(GenerateOutput {
                        text: r#"{"ok":tr"#.to_string(),
                        token_ids: vec![0],
                        prompt_tokens: 1,
                        generated_tokens: 1,
                        stopped: false,
                        stop_reason: Some(lattice_inference::StopReason::Length),
                        token_logprobs: vec![],
                    })
                },
            );
            let state = AppState {
                jobs,
                model_id: Arc::from("test-model"),
                defaults: Defaults {
                    max_tokens: 100,
                    temperature: 0.7,
                    top_k: 50,
                    top_p: 0.9,
                    repetition_penalty: 1.1,
                    reasoning_budget: None,
                },
                model_max_context: 4096,
                max_pending: 1_000_000,
                metrics: Arc::new(ServeMetrics::default()),
                vocab_bytes: route_test_vocab(),
                grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
            };
            let body = Body::from(structured_body(V0_ROUTE_SCHEMA, Some("true"), None));
            let response = chat_completions(State(state), test_json_headers(), body).await;
            let (status, code) = error_code_of(response).await;
            assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(code, "length_limit");
        }

        /// Independent-validation failure path (design note's
        /// "independent validation" requirement): the engine claims a
        /// clean stop, but the produced text does not conform to the
        /// admitted schema -- must still be a 500 `validation_failed`, not
        /// a 200 with nonconforming JSON.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_structured_validation_failed_500() {
            let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                .tokenizer()
                .clone();
            let jobs = spawn_fake(
                ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                4096,
                tokenizer,
                move |_messages, _cfg, _prompt_tokens, on_token, _should_cancel| {
                    on_token(r#"{"unexpected":1}"#, 0);
                    Ok(GenerateOutput {
                        // Conforms to neither `additionalProperties:false`
                        // nor the required `ok` boolean field.
                        text: r#"{"unexpected":1}"#.to_string(),
                        token_ids: vec![0],
                        prompt_tokens: 1,
                        generated_tokens: 1,
                        stopped: true,
                        stop_reason: Some(lattice_inference::StopReason::Eos),
                        token_logprobs: vec![],
                    })
                },
            );
            let state = AppState {
                jobs,
                model_id: Arc::from("test-model"),
                defaults: Defaults {
                    max_tokens: 100,
                    temperature: 0.7,
                    top_k: 50,
                    top_p: 0.9,
                    repetition_penalty: 1.1,
                    reasoning_budget: None,
                },
                model_max_context: 4096,
                max_pending: 1_000_000,
                metrics: Arc::new(ServeMetrics::default()),
                vocab_bytes: route_test_vocab(),
                grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
            };
            let body = Body::from(structured_body(V0_ROUTE_SCHEMA, Some("true"), None));
            let response = chat_completions(State(state), test_json_headers(), body).await;
            let (status, code) = error_code_of(response).await;
            assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(code, "validation_failed");
        }

        // ── GrammarCache (stage 2: LRU + single-flight) ─────────────────────

        const SCHEMA_CACHE_TEST_BOOL: &str = r#"{"type":"boolean"}"#;
        const SCHEMA_CACHE_TEST_NULL: &str = r#"{"type":"null"}"#;
        const SCHEMA_CACHE_TEST_EMPTY_OBJECT: &str =
            r#"{"type":"object","properties":{},"required":[],"additionalProperties":false}"#;

        /// A vocabulary sized only for the tiny literal schemas above --
        /// same rationale as `route_test_vocab`: a wide vocabulary combined
        /// with these schemas' few reachable states is irrelevant work, not
        /// a more faithful test.
        fn cache_test_vocab() -> Arc<Vec<Vec<u8>>> {
            Arc::new(
                ["true", "false", "null", "{", "}"]
                    .iter()
                    .map(|s| s.as_bytes().to_vec())
                    .collect(),
            )
        }

        fn compile_schema_for_cache_test(schema_json: &str) -> Result<GrammarEngine, String> {
            let schema: Value = serde_json::from_str(schema_json).expect("test schema must parse");
            let spec = GrammarSpec::JsonSchema(schema);
            GrammarEngine::new(&spec, (*cache_test_vocab()).clone())
                .map_err(|e| format!("test schema failed to compile: {e}"))
        }

        #[test]
        fn grammar_cache_hit_returns_same_engine_without_recompiling() {
            let cache = GrammarCache::new(GRAMMAR_CACHE_CAPACITY);
            let key = canonical_schema_key(&serde_json::from_str(SCHEMA_CACHE_TEST_BOOL).unwrap());
            let first = cache
                .get_or_compile(key.clone(), || {
                    compile_schema_for_cache_test(SCHEMA_CACHE_TEST_BOOL)
                })
                .expect("first compile must succeed");
            let second = cache
                .get_or_compile(key, || {
                    panic!("cache hit must not invoke the compile closure again")
                })
                .expect("cache hit must succeed");
            assert!(
                Arc::ptr_eq(&first, &second),
                "cache hit must return the exact same Arc"
            );
            assert_eq!(cache.compile_count(), 1);
        }

        #[test]
        fn grammar_cache_evicts_least_recently_used_at_capacity() {
            let cache = GrammarCache::new(2);
            let key_a =
                canonical_schema_key(&serde_json::from_str(SCHEMA_CACHE_TEST_BOOL).unwrap());
            let key_b =
                canonical_schema_key(&serde_json::from_str(SCHEMA_CACHE_TEST_NULL).unwrap());
            let key_c = canonical_schema_key(
                &serde_json::from_str(SCHEMA_CACHE_TEST_EMPTY_OBJECT).unwrap(),
            );

            cache
                .get_or_compile(key_a.clone(), || {
                    compile_schema_for_cache_test(SCHEMA_CACHE_TEST_BOOL)
                })
                .unwrap();
            cache
                .get_or_compile(key_b.clone(), || {
                    compile_schema_for_cache_test(SCHEMA_CACHE_TEST_NULL)
                })
                .unwrap();
            cache
                .get_or_compile(key_c, || {
                    compile_schema_for_cache_test(SCHEMA_CACHE_TEST_EMPTY_OBJECT)
                })
                .unwrap();
            assert_eq!(
                cache.compile_count(),
                3,
                "three distinct schemas must each compile exactly once"
            );

            // `a` was the least-recently-used entry when `c` was inserted
            // into a capacity-2 cache, so `b` must still be cached.
            cache
                .get_or_compile(key_b, || {
                    panic!("b must still be cached: a, not b, was the LRU entry evicted by c")
                })
                .unwrap();
            assert_eq!(cache.compile_count(), 3);

            // `a` was evicted, so requesting it again must recompile.
            cache
                .get_or_compile(key_a, || {
                    compile_schema_for_cache_test(SCHEMA_CACHE_TEST_BOOL)
                })
                .unwrap();
            assert_eq!(
                cache.compile_count(),
                4,
                "evicted key must recompile on next request"
            );
        }

        #[test]
        fn grammar_cache_single_flight_collapses_concurrent_identical_requests() {
            let cache = Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY));
            let key = canonical_schema_key(
                &serde_json::from_str(SCHEMA_CACHE_TEST_EMPTY_OBJECT).unwrap(),
            );
            const N: usize = 8;
            let barrier = Arc::new(std::sync::Barrier::new(N));
            let handles: Vec<_> = (0..N)
                .map(|_| {
                    let cache = Arc::clone(&cache);
                    let key = key.clone();
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        barrier.wait();
                        cache
                            .get_or_compile(key, || {
                                std::thread::sleep(std::time::Duration::from_millis(30));
                                compile_schema_for_cache_test(SCHEMA_CACHE_TEST_EMPTY_OBJECT)
                            })
                            .expect("compile must succeed")
                    })
                })
                .collect();
            let engines: Vec<Arc<GrammarEngine>> =
                handles.into_iter().map(|h| h.join().unwrap()).collect();
            assert_eq!(
                cache.compile_count(),
                1,
                "{N} concurrent identical requests must collapse into exactly 1 compile"
            );
            for engine in &engines[1..] {
                assert!(
                    Arc::ptr_eq(&engines[0], engine),
                    "every waiter must observe the same compiled engine"
                );
            }
        }

        /// Round-2 review medium finding 1: the `catch_unwind` single-flight
        /// recovery path (`get_or_compile`'s `Action::Compile` arm) had no
        /// test exercising an actual owner-closure panic. `compile` is an
        /// injectable `FnOnce` argument to `get_or_compile` already -- no
        /// extra test-only seam is needed, since single-flight semantics
        /// mean only the one thread that wins `Action::Compile` ever
        /// invokes its closure, so every concurrent caller can safely be
        /// handed the same panicking closure. Every caller's result is
        /// collected over a channel with a bounded `recv_timeout` rather
        /// than a plain `.join()`, so a regression that reintroduces a
        /// missed `notify_all` fails this test deterministically instead of
        /// hanging the suite.
        #[test]
        fn grammar_cache_panic_in_compile_notifies_waiters_with_internal_error() {
            let cache = Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY));
            let key = canonical_schema_key(
                &serde_json::from_str(SCHEMA_CACHE_TEST_EMPTY_OBJECT).unwrap(),
            );
            const N: usize = 8;
            let barrier = Arc::new(std::sync::Barrier::new(N));
            let prev_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {})); // silence the expected injected panic
            let (tx, rx) = std::sync::mpsc::channel();
            for _ in 0..N {
                let cache = Arc::clone(&cache);
                let key = key.clone();
                let barrier = Arc::clone(&barrier);
                let tx = tx.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    let result = cache.get_or_compile(key, || {
                        std::thread::sleep(std::time::Duration::from_millis(30));
                        panic!("injected compile panic for single-flight regression test");
                    });
                    let _ = tx.send(result);
                });
            }
            drop(tx);
            for _ in 0..N {
                match rx.recv_timeout(std::time::Duration::from_secs(10)) {
                    Ok(Err(CacheError::Internal(_))) => {}
                    Ok(Err(other)) => panic!("expected CacheError::Internal, got Err({other:?})"),
                    Ok(Ok(_)) => panic!("expected CacheError::Internal, got Ok(engine)"),
                    Err(_) => panic!(
                        "caller did not return within timeout -- single-flight panic recovery hung"
                    ),
                }
            }
            std::panic::set_hook(prev_hook);
            assert_eq!(
                cache.compile_count(),
                1,
                "panic-injected compile must run exactly once across all waiters"
            );

            // Poison-recovery: retrying the same key with a working compile
            // must succeed, proving the slot was removed (not left poisoned
            // or permanently occupied) after the panic.
            cache
                .get_or_compile(key, || {
                    compile_schema_for_cache_test(SCHEMA_CACHE_TEST_EMPTY_OBJECT)
                })
                .expect("retry after a panicked compile must succeed");
            assert_eq!(
                cache.compile_count(),
                2,
                "retry after panic recovery must actually recompile, not replay a poisoned entry"
            );
        }

        #[test]
        fn grammar_cache_key_is_invariant_to_object_property_order() {
            let v1: Value = serde_json::from_str(
                r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"null"}},"required":["a","b"],"additionalProperties":false}"#,
            )
            .unwrap();
            let v2: Value = serde_json::from_str(
                r#"{"additionalProperties":false,"required":["a","b"],"properties":{"b":{"type":"null"},"a":{"type":"boolean"}},"type":"object"}"#,
            )
            .unwrap();
            assert_eq!(
                canonical_schema_key(&v1),
                canonical_schema_key(&v2),
                "reordering object keys (including nested 'properties') must not change the cache key"
            );

            let different: Value = serde_json::from_str(SCHEMA_CACHE_TEST_BOOL).unwrap();
            assert_ne!(
                canonical_schema_key(&v1),
                canonical_schema_key(&different),
                "structurally different schemas must have different cache keys"
            );
        }

        #[test]
        fn grammar_cache_hits_across_reordered_but_equivalent_top_level_keys() {
            let cache = GrammarCache::new(GRAMMAR_CACHE_CAPACITY);
            const EMPTY_OBJ_REORDERED: &str =
                r#"{"additionalProperties":false,"required":[],"properties":{},"type":"object"}"#;
            let v1: Value = serde_json::from_str(SCHEMA_CACHE_TEST_EMPTY_OBJECT).unwrap();
            let v2: Value = serde_json::from_str(EMPTY_OBJ_REORDERED).unwrap();
            assert_eq!(canonical_schema_key(&v1), canonical_schema_key(&v2));

            let first = cache
                .get_or_compile(canonical_schema_key(&v1), || {
                    compile_schema_for_cache_test(SCHEMA_CACHE_TEST_EMPTY_OBJECT)
                })
                .expect("v1 must compile");
            let second = cache
                .get_or_compile(canonical_schema_key(&v2), || {
                    panic!("v2's canonical key must hit the entry compiled for v1")
                })
                .expect("v2 must hit cache");
            assert!(Arc::ptr_eq(&first, &second));
            assert_eq!(cache.compile_count(), 1);
        }

        // ── admit_v0_schema: golden accept per construct ────────────────────

        #[test]
        fn admit_v0_schema_accepts_closed_object() {
            admit_v0_schema(&serde_json::from_str(V0_OBJECT_SCHEMA).unwrap())
                .expect("closed all-required object must be admitted");
        }

        #[test]
        fn admit_v0_schema_accepts_uniform_array() {
            let schema = serde_json::json!({"type": "array", "items": {"type": "integer"}});
            admit_v0_schema(&schema).expect("uniform array must be admitted");
        }

        #[test]
        fn admit_v0_schema_accepts_every_primitive() {
            for ty in ["string", "number", "integer", "boolean", "null"] {
                let schema = serde_json::json!({"type": ty});
                admit_v0_schema(&schema)
                    .unwrap_or_else(|e| panic!("type '{ty}' must be admitted: {e}"));
            }
        }

        #[test]
        fn admit_v0_schema_accepts_nested_object_and_array() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"id": {"type": "integer"}},
                            "required": ["id"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["items"],
                "additionalProperties": false
            });
            admit_v0_schema(&schema).expect("nesting of admitted constructs must be admitted");
        }

        // ── admit_v0_schema: one rejection per unsupported keyword ─────────

        #[test]
        fn admit_v0_schema_rejects_enum() {
            let err =
                admit_v0_schema(&serde_json::json!({"type": "string", "enum": ["a"]})).unwrap_err();
            assert!(err.contains("enum"), "{err}");
        }

        #[test]
        fn admit_v0_schema_rejects_const() {
            let err =
                admit_v0_schema(&serde_json::json!({"type": "string", "const": "a"})).unwrap_err();
            assert!(err.contains("const"), "{err}");
        }

        #[test]
        fn admit_v0_schema_rejects_pattern() {
            let err = admit_v0_schema(&serde_json::json!({"type": "string", "pattern": "^a"}))
                .unwrap_err();
            assert!(err.contains("pattern"), "{err}");
        }

        #[test]
        fn admit_v0_schema_rejects_length_and_numeric_constraints() {
            for (kw, schema) in [
                (
                    "minLength",
                    serde_json::json!({"type": "string", "minLength": 1}),
                ),
                (
                    "maxLength",
                    serde_json::json!({"type": "string", "maxLength": 1}),
                ),
                (
                    "minimum",
                    serde_json::json!({"type": "number", "minimum": 1}),
                ),
                (
                    "maximum",
                    serde_json::json!({"type": "number", "maximum": 1}),
                ),
                (
                    "multipleOf",
                    serde_json::json!({"type": "number", "multipleOf": 2}),
                ),
            ] {
                let err = admit_v0_schema(&schema).unwrap_err();
                assert!(err.contains(kw), "{kw}: {err}");
            }
        }

        #[test]
        fn admit_v0_schema_rejects_ref() {
            let err = admit_v0_schema(&serde_json::json!({"$ref": "#/$defs/foo"})).unwrap_err();
            assert!(err.contains("$ref"), "{err}");
        }

        #[test]
        fn admit_v0_schema_rejects_composition() {
            for kw in ["anyOf", "oneOf", "allOf", "not"] {
                let schema = serde_json::json!({kw: [{"type": "string"}]});
                let err = admit_v0_schema(&schema).unwrap_err();
                assert!(err.contains(kw), "{kw}: {err}");
            }
        }

        #[test]
        fn admit_v0_schema_rejects_type_union() {
            let err =
                admit_v0_schema(&serde_json::json!({"type": ["string", "null"]})).unwrap_err();
            assert!(err.contains("union"), "{err}");
        }

        #[test]
        fn admit_v0_schema_rejects_missing_additional_properties() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"]
            });
            let err = admit_v0_schema(&schema).unwrap_err();
            assert!(err.contains("additionalProperties"), "{err}");
        }

        #[test]
        fn admit_v0_schema_rejects_open_additional_properties() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": true
            });
            let err = admit_v0_schema(&schema).unwrap_err();
            assert!(err.contains("additionalProperties"), "{err}");
        }

        #[test]
        fn admit_v0_schema_rejects_missing_required_entry() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": false
            });
            let err = admit_v0_schema(&schema).unwrap_err();
            assert!(err.contains("required"), "{err}");
        }

        #[test]
        fn admit_v0_schema_rejects_prefix_items() {
            let schema = serde_json::json!({
                "type": "array",
                "items": {"type": "string"},
                "prefixItems": [{"type": "string"}]
            });
            let err = admit_v0_schema(&schema).unwrap_err();
            assert!(err.contains("prefixItems"), "{err}");
        }

        #[test]
        fn admit_v0_schema_rejects_format() {
            let err =
                admit_v0_schema(&serde_json::json!({"type": "string", "format": "date-time"}))
                    .unwrap_err();
            assert!(err.contains("format"), "{err}");
        }

        #[test]
        fn admit_v0_schema_rejects_unknown_keyword() {
            let err =
                admit_v0_schema(&serde_json::json!({"type": "string", "wat": true})).unwrap_err();
            assert!(err.contains("wat"), "{err}");
        }

        /// Mutation-sensitivity check: unlike a plain `.is_err()` check, this asserts the REJECTION
        /// SOURCE is specifically `V0_REJECTED_KEYWORDS` (the denylist
        /// layer walked before the per-type match in `admit_v0_schema_at`),
        /// not the primitive per-type allowlist layer (`ALLOWED` inside the
        /// `"string"` arm) that would also reject an unrecognized `enum`
        /// key on its own. If `enum` is removed from
        /// `V0_REJECTED_KEYWORDS`, this schema still gets rejected -- by the
        /// allowlist layer instead -- but with a DIFFERENT message, so this
        /// test (unlike a bare `.is_err()` check) fails as intended. See
        /// the stage-1 report's mutation transcript for the revert/restore
        /// procedure this test guards, and this session's report for the
        /// re-run transcript.
        #[test]
        fn admit_v0_schema_mutation_sensitive_to_enum_admission() {
            let schema = serde_json::json!({"type": "string", "enum": ["a", "b"]});
            let err = admit_v0_schema(&schema)
                .expect_err("v0 must reject enum (Q2: deferred from the first patch)");
            assert!(
                err.contains("is not supported by the v0 structured-output schema subset"),
                "expected the denylist layer's own rejection wording, got: {err}"
            );
        }

        /// Independent companion to the denylist test above: a keyword that `V0_REJECTED_KEYWORDS` never
        /// mentions at all (so the denylist layer can never be the one that
        /// rejects it) must still be rejected by the primitive per-type
        /// `ALLOWED`-keyword layer. Mutation-sensitive to a DIFFERENT
        /// mutation than the test above: loosening `ALLOWED` for a
        /// primitive type (e.g. adding `"default"`) makes this schema admit
        /// successfully, failing this test -- independently of whatever
        /// `V0_REJECTED_KEYWORDS` contains.
        #[test]
        fn admit_v0_schema_mutation_sensitive_to_primitive_allowlist() {
            let schema = serde_json::json!({"type": "boolean", "default": true});
            let err = admit_v0_schema(&schema)
                .expect_err("v0 must reject an unrecognized primitive keyword");
            assert!(
                err.contains("is not supported for type 'boolean'"),
                "expected the primitive-allowlist layer's own rejection wording, got: {err}"
            );
        }

        // ── v0_validate_json: independent evaluator ─────────────────────────

        #[test]
        fn v0_validate_json_accepts_conforming_object() {
            let schema: Value = serde_json::from_str(V0_OBJECT_SCHEMA).unwrap();
            assert!(v0_validate_json(r#"{"name": "ok"}"#, &schema));
        }

        #[test]
        fn v0_validate_json_rejects_extra_field() {
            let schema: Value = serde_json::from_str(V0_OBJECT_SCHEMA).unwrap();
            assert!(!v0_validate_json(r#"{"name": "ok", "extra": 1}"#, &schema));
        }

        #[test]
        fn v0_validate_json_rejects_missing_field() {
            let schema: Value = serde_json::from_str(V0_OBJECT_SCHEMA).unwrap();
            assert!(!v0_validate_json("{}", &schema));
        }

        #[test]
        fn v0_validate_json_rejects_wrong_type() {
            let schema: Value = serde_json::from_str(V0_OBJECT_SCHEMA).unwrap();
            assert!(!v0_validate_json(r#"{"name": 1}"#, &schema));
        }

        #[test]
        fn v0_validate_json_rejects_malformed_json() {
            let schema: Value = serde_json::from_str(V0_OBJECT_SCHEMA).unwrap();
            assert!(!v0_validate_json(r#"{"name": "ok""#, &schema));
        }

        #[test]
        fn v0_validate_json_rejects_trailing_garbage() {
            let schema: Value = serde_json::from_str(V0_OBJECT_SCHEMA).unwrap();
            assert!(!v0_validate_json(r#"{"name": "ok"} garbage"#, &schema));
        }

        /// A grammar-conforming integer
        /// outside `u64`/`i64` range must validate, not produce a false
        /// `validation_failed`.
        #[test]
        fn v0_validate_json_accepts_integer_outside_u64_range() {
            let schema = serde_json::json!({"type": "integer"});
            assert!(v0_validate_json("18446744073709551616", &schema));
            assert!(v0_validate_json("-18446744073709551616", &schema));
        }

        #[test]
        fn v0_validate_json_accepts_integer_in_u64_range() {
            let schema = serde_json::json!({"type": "integer"});
            assert!(v0_validate_json("42", &schema));
            assert!(v0_validate_json("-1", &schema));
        }

        /// JSON Schema 2020-12 §6.1.1: a zero fractional part is still an
        /// integer.
        #[test]
        fn v0_validate_json_accepts_integer_with_zero_fraction() {
            let schema = serde_json::json!({"type": "integer"});
            assert!(v0_validate_json("1.0", &schema));
        }

        #[test]
        fn v0_validate_json_rejects_integer_with_nonzero_fraction() {
            let schema = serde_json::json!({"type": "integer"});
            assert!(!v0_validate_json("1.5", &schema));
        }

        #[test]
        fn v0_validate_json_rejects_integer_lexeme_for_string_schema() {
            let schema = serde_json::json!({"type": "string"});
            assert!(!v0_validate_json("18446744073709551616", &schema));
        }

        /// End-to-end shape from the review's suggested test: a large
        /// integer nested inside an admitted object schema.
        #[test]
        fn v0_validate_json_accepts_large_integer_nested_in_object() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
                "additionalProperties": false,
            });
            assert!(v0_validate_json(r#"{"n": 18446744073709551616}"#, &schema));
        }

        /// Round-2 review major finding 1: a valid UTF-16 surrogate pair
        /// (the G-clef character, U+1D11E) spelled as its two `\uXXXX`
        /// escape halves must validate under a string schema -- it is a
        /// grammar-conforming strict completion, and rejecting it turned
        /// into a false `validation_failed` 500 before this fix.
        #[test]
        fn v0_validate_json_accepts_surrogate_pair_under_string_schema() {
            let schema = serde_json::json!({"type": "string"});
            assert!(v0_validate_json(r#""\uD834\uDD1E""#, &schema));
        }

        /// Same surrogate pair, but as an object property value -- proves
        /// the fix applies through `v0_validate_object`'s recursive
        /// `v0_validate_at` call, not just the top-level string schema.
        #[test]
        fn v0_validate_json_accepts_surrogate_pair_as_object_property_value() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {"s": {"type": "string"}},
                "required": ["s"],
                "additionalProperties": false,
            });
            assert!(v0_validate_json(r#"{"s": "\uD834\uDD1E"}"#, &schema));
        }

        /// Grammar-parity: the compiled grammar admits any four-hex
        /// `\uXXXX` escape independently of pairing (json_schema.rs
        /// `escape_id` alternatives at ~1564), and its own
        /// `string_accepts_legal_escapes` test (json_schema.rs ~2762)
        /// explicitly accepts lone surrogate halves `\uD800` / `\uDC00`.
        /// The validator must not be narrower than the grammar it guards,
        /// so a lone (unpaired) surrogate escape must also validate here.
        #[test]
        fn v0_validate_json_accepts_lone_surrogate_escape_grammar_parity() {
            let schema = serde_json::json!({"type": "string"});
            assert!(v0_validate_json(r#""\uD834""#, &schema));
            assert!(v0_validate_json(r#""\uDD1E""#, &schema));
        }

        /// Round-2 review major finding 1, second half: RFC 8259 §7
        /// requires U+0000-U+001F to be escaped, and the compiled grammar
        /// rejects them raw too (json_schema.rs
        /// `string_rejects_raw_control_byte`). A literal BEL (0x07) byte
        /// embedded unescaped in a string must fail validation.
        #[test]
        fn v0_validate_json_rejects_raw_control_char() {
            let schema = serde_json::json!({"type": "string"});
            let content = format!("\"{}\"", '\u{7}');
            assert!(!v0_validate_json(&content, &schema));
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_n_greater_than_one_400() {
            let body =
                Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"n":2}"#.to_string());
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(message, "n > 1 is not supported");
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_logprobs_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"logprobs":true}"#.to_string(),
            );
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                message,
                "logprobs/top_logprobs are not supported by this server"
            );
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_stop_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"stop":"\n"}"#.to_string(),
            );
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(message, "stop is not supported by this server");
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_conflicting_max_tokens_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],
                    "max_tokens":10,"max_completion_tokens":20}"#
                    .to_string(),
            );
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                message,
                "max_tokens (10) and max_completion_tokens (20) differ; supply only one"
            );
        }

        // ── ADR-080 C2 (#782): max_tokens=0 rejection + finish_reason round-trip ──

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_max_tokens_zero_400() {
            // #745: previously clamped straight through into a zero-budget
            // completion instead of being rejected.
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#.to_string(),
            );
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(message, "max_tokens must be at least 1");
        }

        /// Builds an `AppState` wired to a fresh raw job channel plus the
        /// receiving half (issue #832's `test_client_and_jobs` seam), so
        /// tests can stand in for the worker: reply with whatever
        /// `WorkerEvent` sequence the test wants (via `WorkerJob::reply`,
        /// also `test-utils`-gated) without a real GPU/model.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        fn test_app_state_with_jobs() -> (AppState, mpsc::UnboundedReceiver<WorkerJob>) {
            let (jobs, jobs_rx) = test_client_and_jobs();
            let state = AppState {
                jobs,
                model_id: Arc::from("test-model"),
                defaults: Defaults {
                    max_tokens: 100,
                    temperature: 0.7,
                    top_k: 50,
                    top_p: 0.9,
                    repetition_penalty: 1.1,
                    reasoning_budget: None,
                },
                model_max_context: 4096,
                max_pending: 1_000_000,
                metrics: Arc::new(ServeMetrics::default()),
                vocab_bytes: Arc::new(vec![]),
                grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
            };
            (state, jobs_rx)
        }

        /// #746 (ADR-080 C2): the non-streaming JSON response's
        /// `finish_reason` must reflect the engine's actual `stopped` flag,
        /// not a hardcoded `"stop"`. A fake worker task stands in for the
        /// GPU, replying with a crafted `WorkerEvent::Complete` (`stopped`)
        /// directly.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        async fn non_streaming_finish_reason_for(stopped: bool) -> String {
            let (state, mut jobs_rx) = test_app_state_with_jobs();
            tokio::spawn(async move {
                if let Some(job) = jobs_rx.recv().await {
                    let _ = job.reply(WorkerEvent::Delta("hi".to_string()));
                    let _ = job.reply(WorkerEvent::Complete(GenerateOutput {
                        text: "hi".to_string(),
                        token_ids: vec![0],
                        prompt_tokens: 1,
                        generated_tokens: 1,
                        stopped,
                        stop_reason: None,
                        token_logprobs: vec![],
                    }));
                }
            });
            let body = Body::from(r#"{"messages":[{"role":"user","content":"hi"}]}"#.to_string());
            let response = chat_completions(State(state), test_json_headers(), body).await;
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("response body must be readable");
            let v: serde_json::Value =
                serde_json::from_slice(&bytes).expect("response body must be valid JSON");
            v["choices"][0]["finish_reason"]
                .as_str()
                .expect("finish_reason must be a string")
                .to_string()
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_non_streaming_finish_reason_stop_when_engine_stopped() {
            assert_eq!(non_streaming_finish_reason_for(true).await, "stop");
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_non_streaming_finish_reason_length_when_not_stopped() {
            // Previously this hardcoded "stop" unconditionally, so a
            // length-capped or cancelled completion was misreported as an
            // explicit stop condition (#746).
            assert_eq!(non_streaming_finish_reason_for(false).await, "length");
        }

        /// #583: `GET /metrics` must reflect REAL traffic recorded through
        /// `emit_serve_event`, not a static template -- mutation check: if
        /// `emit_serve_event` stops forwarding into `ServeMetrics` (or
        /// `metrics_handler` stops rendering it), every assertion below
        /// regresses to "0"/absent while the route itself would still
        /// return 200, so this would NOT be caught by a route-existence
        /// check alone.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn metrics_endpoint_reports_requests_tokens_and_errors() {
            let (state, mut jobs_rx) = test_app_state_with_jobs();
            tokio::spawn(async move {
                if let Some(job) = jobs_rx.recv().await {
                    let _ = job.reply(WorkerEvent::Complete(GenerateOutput {
                        text: "hi".to_string(),
                        token_ids: vec![0],
                        prompt_tokens: 3,
                        generated_tokens: 2,
                        stopped: true,
                        stop_reason: None,
                        token_logprobs: vec![],
                    }));
                }
            });
            let ok_body =
                Body::from(r#"{"messages":[{"role":"user","content":"hi"}]}"#.to_string());
            let ok_response =
                chat_completions(State(state.clone()), test_json_headers(), ok_body).await;
            assert_eq!(ok_response.status(), StatusCode::OK);

            let bad_body = Body::from(r#"{"messages":[]}"#.to_string());
            let bad_response =
                chat_completions(State(state.clone()), test_json_headers(), bad_body).await;
            assert_eq!(bad_response.status(), StatusCode::BAD_REQUEST);

            let _ = health(State(state.clone())).await;

            let metrics_response = metrics_handler(State(state.clone())).await.into_response();
            let bytes = axum::body::to_bytes(metrics_response.into_body(), usize::MAX)
                .await
                .expect("metrics body must be readable");
            let body = String::from_utf8(bytes.to_vec()).expect("metrics body must be UTF-8");

            assert!(
                body.contains(
                    "lattice_http_requests_total{method=\"POST\",route=\"/v1/chat/completions\",status=\"200\",model=\"test-model\"} 1"
                ),
                "missing successful chat_completions counter:\n{body}"
            );
            assert!(
                body.contains(
                    "lattice_http_requests_total{method=\"GET\",route=\"/health\",status=\"200\",model=\"test-model\"} 1"
                ),
                "missing health counter:\n{body}"
            );
            assert!(
                body.contains("lattice_prompt_tokens_total{model=\"test-model\"} 3"),
                "missing prompt token count:\n{body}"
            );
            assert!(
                body.contains("lattice_completion_tokens_total{model=\"test-model\"} 2"),
                "missing completion token count:\n{body}"
            );
            assert!(
                body.contains(
                    "lattice_errors_total{code=\"invalid_messages\",model=\"test-model\"} 1"
                ),
                "missing invalid_messages error count:\n{body}"
            );
        }

        /// Same round-trip as above, but through the SSE streaming path
        /// (`Phase::Body`'s `Ev::Done` arm) instead of the non-streaming
        /// JSON body.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        async fn streaming_finish_reason_for(stopped: bool) -> String {
            let (state, mut jobs_rx) = test_app_state_with_jobs();
            tokio::spawn(async move {
                if let Some(job) = jobs_rx.recv().await {
                    let _ = job.reply(WorkerEvent::Delta("hi".to_string()));
                    let _ = job.reply(WorkerEvent::Complete(GenerateOutput {
                        text: "hi".to_string(),
                        token_ids: vec![0],
                        prompt_tokens: 1,
                        generated_tokens: 1,
                        stopped,
                        stop_reason: None,
                        token_logprobs: vec![],
                    }));
                }
            });
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#.to_string(),
            );
            let response = chat_completions(State(state), test_json_headers(), body).await;
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("SSE response body must be readable");
            String::from_utf8(bytes.to_vec()).expect("SSE body must be valid UTF-8")
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_streaming_finish_reason_stop_when_engine_stopped() {
            let text = streaming_finish_reason_for(true).await;
            assert!(
                text.contains("\"finish_reason\":\"stop\""),
                "SSE body must carry finish_reason: stop; got: {text}"
            );
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_streaming_finish_reason_length_when_not_stopped() {
            let text = streaming_finish_reason_for(false).await;
            assert!(
                text.contains("\"finish_reason\":\"length\""),
                "SSE body must carry finish_reason: length, not a hardcoded \"stop\" \
                 (#746); got: {text}"
            );
        }

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_streaming_failure_emits_error_event() {
            let (state, mut jobs_rx) = test_app_state_with_jobs();
            tokio::spawn(async move {
                if let Some(job) = jobs_rx.recv().await {
                    let _ = job.reply(WorkerEvent::Delta("partial".to_string()));
                    let _ = job.reply(WorkerEvent::Failed("blocked by grammar".to_string()));
                }
            });
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#.to_string(),
            );
            let response = chat_completions(State(state), test_json_headers(), body).await;
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("SSE response body must be readable");
            let text = String::from_utf8(bytes.to_vec()).expect("SSE body must be valid UTF-8");
            assert!(
                text.contains("\"content\":\"partial\""),
                "partial output must precede the error event; got: {text}"
            );
            let error_payload = text
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .find(|payload| payload.contains("\"error\""))
                .expect("a failed generation must emit an SSE error payload");
            let error: serde_json::Value =
                serde_json::from_str(error_payload).expect("SSE error payload must be valid JSON");
            assert_eq!(error["error"]["type"], "server_error");
            assert_eq!(error["error"]["code"], "internal_error");
            assert!(
                !text.contains("\"finish_reason\":\"stop\""),
                "generation failure must not masquerade as a clean stop; got: {text}"
            );
        }

        /// Reviewer finding (b): a mid-stream `WorkerEvent::Failed` (or
        /// `Rejected`/worker-gone) emits a client-facing SSE error event
        /// (asserted above) but, before this fix, the terminal `Phase::End`
        /// still recorded the request in `/metrics` as a plain 200 with no
        /// error code -- `lattice_errors_total` never incremented for a
        /// failed stream. Mutation-sensitive: reverting the `Phase::Done`/
        /// `Phase::End` error-outcome threading (while keeping this test)
        /// makes this fail, because the recorded counter reverts to
        /// `status="200"` and `lattice_errors_total{code="internal_error"}`
        /// stays absent.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_streaming_failure_records_failed_metric() {
            let (state, mut jobs_rx) = test_app_state_with_jobs();
            tokio::spawn(async move {
                if let Some(job) = jobs_rx.recv().await {
                    let _ = job.reply(WorkerEvent::Delta("partial".to_string()));
                    let _ = job.reply(WorkerEvent::Failed("blocked by grammar".to_string()));
                }
            });
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#.to_string(),
            );
            let response = chat_completions(State(state.clone()), test_json_headers(), body).await;
            assert_eq!(response.status(), StatusCode::OK);
            // Drain the full SSE body so the stream runs to `Phase::End`,
            // which is where the metric gets recorded.
            let _ = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("SSE response body must be readable");

            let metrics_response = metrics_handler(State(state.clone())).await.into_response();
            let bytes = axum::body::to_bytes(metrics_response.into_body(), usize::MAX)
                .await
                .expect("metrics body must be readable");
            let text = String::from_utf8(bytes.to_vec()).expect("metrics body must be UTF-8");

            assert!(
                !text.contains(
                    "lattice_http_requests_total{method=\"POST\",route=\"/v1/chat/completions\",status=\"200\""
                ),
                "a failed stream must NOT be recorded as a 200; got:\n{text}"
            );
            assert!(
                text.contains(
                    "lattice_http_requests_total{method=\"POST\",route=\"/v1/chat/completions\",status=\"500\",model=\"test-model\"} 1"
                ),
                "failed stream must be recorded with the same 500 status the non-streaming \
                 `Failed` arm records; got:\n{text}"
            );
            assert!(
                text.contains(
                    "lattice_errors_total{code=\"internal_error\",model=\"test-model\"} 1"
                ),
                "lattice_errors_total must increment for a failed stream; got:\n{text}"
            );
        }

        /// ADR-080 C2: a `stream:
        /// true` request whose prompt overflows the model's context window
        /// must return HTTP 400 `context_length_exceeded` BEFORE any SSE
        /// stream is committed -- not silently commit to a 200 SSE response
        /// that only discovers the overflow later via `WorkerEvent::Rejected`
        /// mid-stream and terminates with `finish_reason: "length"` (the
        /// exact drift this test catches). This fakes the worker side of
        /// the `WorkerEvent::Rejected` contract (production code sends it
        /// via `metal_worker::check_prompt_fits_window`'s typed `ApiError`,
        /// issue #832) so the composition under test is purely
        /// `chat_completions`'s streaming branch: does it await the
        /// worker's first event and inspect it for `WorkerEvent::Rejected`
        /// BEFORE calling `Sse::new(..).into_response()`, or does it commit
        /// unconditionally?
        ///
        /// ADR-080 C2: this test
        /// pins ONLY that response-mapping contract with a request that
        /// would NOT genuinely overflow in production (`max_tokens`
        /// defaults to 100 against a 4096-token `model_max_context`) and a
        /// fake task that manufactures `WorkerEvent::Rejected` directly --
        /// it does NOT exercise the real worker's `check_prompt_fits_window`
        /// call, so it is not, by itself, same-input real-router parity with
        /// `lattice.rs`'s equivalent test (an earlier version of this
        /// comment incorrectly claimed "identical request body"; it is not
        /// -- `real_router_overflow_parity` below is the test that actually
        /// is). Kept because it is still the cheapest, fastest pin of the
        /// response-mapping contract in isolation.
        ///
        /// Mutation-sensitive: reverting the pre-`Sse::new()` `first_ev`
        /// preflight (going back to building `stream::unfold` and
        /// returning `Sse::new(stream).into_response()` unconditionally,
        /// discovering `WorkerEvent::Rejected` only inside `Phase::Body`)
        /// makes this fail -- the response status would be 200 with an SSE body
        /// carrying a `finish_reason: "length"` terminal chunk instead of a
        /// 400 error envelope. Verified by reverting the preflight and
        /// re-running: see the PR body's mutation log.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn chat_completions_streaming_context_overflow_returns_400_before_committing_sse() {
            let (state, mut jobs_rx) = test_app_state_with_jobs();
            tokio::spawn(async move {
                if let Some(job) = jobs_rx.recv().await {
                    let _ = job.reply(WorkerEvent::Rejected(ApiError::BadRequest {
                        message: "prompt has 4090 tokens, leaving 6 of the 4096-token \
                                  context window for generation, but this request needs \
                                  100 generated tokens plus 1 (total 4191); reduce \
                                  max_tokens/reasoning_budget or shorten the prompt"
                            .to_string(),
                        code: "context_length_exceeded",
                    }));
                }
            });
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#.to_string(),
            );
            let response = chat_completions(State(state), test_json_headers(), body).await;
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "an Ev::Rejected worker reply for a stream:true request must surface \
                 as a pre-commit HTTP 400, not a committed 200 SSE stream"
            );
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("error response body must be readable");
            let value: serde_json::Value =
                serde_json::from_slice(&bytes).expect("error response must be JSON");
            assert_eq!(value["error"]["code"], "context_length_exceeded");
            assert!(
                value["error"]["message"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("context window"),
                "error message must carry the worker's actual overflow explanation, \
                 got: {value}"
            );
        }

        /// ADR-080 C2: the
        /// same-input real-router parity the test above does NOT provide.
        /// Drives `lattice_inference::serve::OVERFLOW_PARITY_REQUEST_BODY`
        /// -- the SAME fixture `lattice.rs`'s `streaming_context_overflow`
        /// module drives through its own real `Router` -- through THIS
        /// binary's real `router()`, backed by a REAL worker thread running
        /// the actual `run_worker_loop` + `check_prompt_fits_window`
        /// production code (via `metal_worker::spawn_fake`, issue #832's
        /// cross-binary test seam; a real `BpeTokenizer` from
        /// `test_support::tiny_zero_model`, no Metal engine involved since
        /// the request is expected to be rejected before any generation
        /// call). `AppState.model_max_context` is set to
        /// `OVERFLOW_PARITY_CONTEXT_WINDOW` (1024) -- the same effective
        /// limit `lattice.rs`'s tiny test model's fixed context window uses
        /// -- so both sides genuinely share the same input AND the same
        /// effective context limit, not just the same JSON body.
        ///
        /// Mutation-sensitive: removing `check_prompt_fits_window`'s
        /// overflow check (i.e. having it unconditionally return `Ok(())`)
        /// makes this fail -- the real worker would accept the request,
        /// `chat_completions` would commit a 200 SSE response, and this
        /// test's `StatusCode::BAD_REQUEST` assertion would fail. This is
        /// the exact production worker-side check that removing
        /// `MetalWorker::spawn`'s reliance on it would leave undetected --
        /// `MetalWorker::spawn`'s real Metal closure and this test's
        /// `spawn_fake` seam both route through the SAME `run_worker_loop`
        /// wrapper (`metal_worker::run_worker_loop`, private to that
        /// module) that calls `check_prompt_fits_window`, not two
        /// independent copies of the check, so a mutation to the shared
        /// function is observed here too.
        ///
        /// Gated behind `test-utils` (see
        /// `lattice_inference::model::qwen35::test_support`) for the same
        /// reason as `lattice.rs`'s equivalent test modules: this needs a
        /// real (tiny) `BpeTokenizer`, which is only constructible outside
        /// this crate's own `#[cfg(test)]` build via that feature.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        mod real_router_overflow_parity {
            use super::*;
            use lattice_inference::serve::{
                OVERFLOW_PARITY_CONTEXT_WINDOW, OVERFLOW_PARITY_REQUEST_BODY,
            };
            use tower::ServiceExt as _;

            /// A real worker thread running the actual `run_worker_loop` +
            /// `check_prompt_fits_window` production code (via
            /// `metal_worker::spawn_fake`, issue #832's cross-binary test
            /// seam), with a real (tiny) tokenizer and NO Metal engine --
            /// the generate closure is never reached by an overflowing
            /// request, so it only needs a trivial success stand-in for a
            /// non-overflowing one.
            fn real_worker_state(model_max_context: usize) -> AppState {
                let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                    .tokenizer()
                    .clone();
                let jobs = spawn_fake(
                    ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                    model_max_context,
                    tokenizer,
                    |_messages, _cfg, prompt_tokens, _on_token, _should_cancel| {
                        Ok(GenerateOutput {
                            text: String::new(),
                            token_ids: vec![],
                            prompt_tokens,
                            generated_tokens: 0,
                            stopped: true,
                            stop_reason: None,
                            token_logprobs: vec![],
                        })
                    },
                );
                AppState {
                    jobs,
                    model_id: Arc::from("test-model"),
                    defaults: Defaults {
                        max_tokens: 100,
                        temperature: 0.7,
                        top_k: 50,
                        top_p: 0.9,
                        repetition_penalty: 1.1,
                        reasoning_budget: None,
                    },
                    model_max_context,
                    max_pending: 1_000_000,
                    metrics: Arc::new(ServeMetrics::default()),
                    vocab_bytes: Arc::new(vec![]),
                    grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
                }
            }

            #[tokio::test]
            async fn chat_completions_streaming_context_overflow_matches_lattice_real_router() {
                let body = Body::from(OVERFLOW_PARITY_REQUEST_BODY.to_string());
                let request = axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(body)
                    .expect("fixture request must build");
                let response = router(real_worker_state(OVERFLOW_PARITY_CONTEXT_WINDOW))
                    .oneshot(request)
                    .await
                    .expect("router must produce a response, not a transport error");
                assert_eq!(
                    response.status(),
                    StatusCode::BAD_REQUEST,
                    "the shared overflow-parity request, driven through the real \
                     worker's production enforce_prompt_window check, must be \
                     rejected with HTTP 400 before any SSE stream is committed -- \
                     matching lattice.rs's real-router result for the identical \
                     request body and effective context limit"
                );
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .expect("error response body must be readable");
                let value: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("error response must be JSON");
                assert_eq!(value["error"]["code"], "context_length_exceeded");
            }
        }

        /// HTTP-level admission/backpressure test (issue #932): a real
        /// worker thread running the actual production `MetalWorkerClient`
        /// (via `metal_worker::spawn_fake_with_cap`, the same cross-binary
        /// test seam `real_router_overflow_parity` above uses, with an
        /// explicit small cap instead of the effectively-unbounded
        /// default), driven through this binary's real `router()` end to
        /// end. Proves the 503 JSON envelope shape a real OpenAI-compatible
        /// client would see, and that it is returned fail-fast (before the
        /// worker thread's `generate` closure -- which would tokenize the
        /// prompt -- is ever reached for the rejected request).
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        mod real_router_admission_cap {
            use super::*;
            use lattice_inference::serve::metal_worker::spawn_fake_with_cap;
            use std::sync::mpsc as std_mpsc;
            use tower::ServiceExt as _;

            /// A real worker thread (cap=1) whose `generate` blocks on
            /// `unblock_rx.recv()` until the test releases it, so exactly
            /// one request can be "in flight" for as long as the test needs
            /// -- long enough to prove a second concurrent request is
            /// rejected by admission rather than racing to observe an
            /// already-freed slot.
            fn single_slot_blocking_state() -> (
                AppState,
                std_mpsc::Sender<()>,
                std_mpsc::Receiver<()>, // "generate() started" signal
            ) {
                let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                    .tokenizer()
                    .clone();
                let (unblock_tx, unblock_rx) = std_mpsc::channel::<()>();
                let unblock_rx = std::sync::Mutex::new(unblock_rx);
                let (started_tx, started_rx) = std_mpsc::channel::<()>();
                let jobs = spawn_fake_with_cap(
                    1,
                    ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                    4096,
                    tokenizer,
                    move |_messages, _cfg, prompt_tokens, _on_token, _should_cancel| {
                        let _ = started_tx.send(());
                        // Blocks the dedicated fake-worker OS thread (not
                        // the tokio runtime) until the test explicitly lets
                        // it proceed.
                        let _ = unblock_rx.lock().unwrap().recv();
                        Ok(GenerateOutput {
                            text: String::new(),
                            token_ids: vec![],
                            prompt_tokens,
                            generated_tokens: 0,
                            stopped: true,
                            stop_reason: None,
                            token_logprobs: vec![],
                        })
                    },
                );
                let state = AppState {
                    jobs,
                    model_id: Arc::from("test-model"),
                    defaults: Defaults {
                        max_tokens: 100,
                        temperature: 0.7,
                        top_k: 50,
                        top_p: 0.9,
                        repetition_penalty: 1.1,
                        reasoning_budget: None,
                    },
                    model_max_context: 4096,
                    max_pending: 1_000_000,
                    metrics: Arc::new(ServeMetrics::default()),
                    vocab_bytes: Arc::new(vec![]),
                    grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
                };
                (state, unblock_tx, started_rx)
            }

            fn simple_chat_request() -> axum::http::Request<Body> {
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#
                            .to_string(),
                    ))
                    .expect("fixture request must build")
            }

            #[tokio::test]
            async fn chat_completions_returns_503_json_envelope_when_admission_cap_reached() {
                let (state, unblock_tx, started_rx) = single_slot_blocking_state();
                let app = router(state);

                // Request 1: admitted (cap=1, 0 outstanding); its worker-thread
                // `generate` call blocks until we release it below. Driven in
                // a background task since it will not resolve until then.
                let app1 = app.clone();
                let request1 = simple_chat_request();
                let handle1 = tokio::spawn(async move { app1.oneshot(request1).await });

                // Deterministic readiness: wait for `generate` to actually
                // start (i.e. request 1's job has been dequeued and its
                // admission permit is held) before firing request 2, rather
                // than a fixed sleep.
                tokio::task::spawn_blocking(move || started_rx.recv())
                    .await
                    .expect("blocking wait must not panic")
                    .expect("request 1's worker-thread generate() must signal it started");

                // Request 2: cap is now full (1/1) -- must be rejected with
                // a 503 JSON envelope, fail-fast, before it ever reaches the
                // worker thread's tokenizer/generate call.
                let request2 = simple_chat_request();
                let response2 = app
                    .clone()
                    .oneshot(request2)
                    .await
                    .expect("router must produce a response, not a transport error");
                assert_eq!(
                    response2.status(),
                    StatusCode::SERVICE_UNAVAILABLE,
                    "a request submitted while the admission cap is full must be \
                     rejected with HTTP 503, fail-fast, before any SSE stream is \
                     committed and before any tokenization/generation work runs"
                );
                let bytes2 = axum::body::to_bytes(response2.into_body(), usize::MAX)
                    .await
                    .expect("503 response body must be readable");
                let value2: serde_json::Value =
                    serde_json::from_slice(&bytes2).expect("503 response must be JSON");
                assert_eq!(value2["error"]["code"], "server_busy");
                assert_eq!(value2["error"]["type"], "server_error");
                assert!(
                    !value2["error"]["message"]
                        .as_str()
                        .unwrap_or_default()
                        .is_empty(),
                    "503 envelope must carry a human-readable message: {value2}"
                );

                // Release request 1 so it completes normally and the
                // background task/worker thread wind down cleanly.
                unblock_tx.send(()).expect("unblock send must succeed");
                let response1 = handle1
                    .await
                    .expect("request 1's task must not panic")
                    .expect("router must produce a response, not a transport error");
                assert_eq!(
                    response1.status(),
                    StatusCode::OK,
                    "request 1 itself was never over any cap and must succeed normally"
                );
            }
        }

        #[test]
        fn build_cfg_aliases_max_completion_tokens_when_max_tokens_absent() {
            let defaults = Defaults {
                max_tokens: 100,
                temperature: 0.7,
                top_k: 50,
                top_p: 0.9,
                repetition_penalty: 1.1,
                reasoning_budget: None,
            };
            let req = ChatReq {
                model: None,
                messages: vec![InMsg {
                    role: "user".to_string(),
                    content: MessageContent::Text("hi".to_string()),
                }],
                temperature: None,
                top_p: None,
                top_k: None,
                max_tokens: None,
                seed: None,
                stream: None,
                repetition_penalty: None,
                reasoning_budget: None,
                max_completion_tokens: Some(42),
                tools: None,
                tool_choice: None,
                response_format: None,
                n: None,
                logprobs: None,
                top_logprobs: None,
                stop: None,
            };
            let normalized = normalize_for_cfg(&req, &defaults, 4096);
            let cfg = build_cfg(&normalized);
            assert_eq!(cfg.max_new_tokens, 42);
        }

        // -----------------------------------------------------------------------
        // Capability-matrix fixtures (#654). The `chat_completions_*_400` tests
        // above this block are ALSO capability-matrix fixtures in their own
        // right — their names are the fixture IDs `docs/capability-matrix.md`'s
        // Fixture manifest section cites for the tools/tool_choice/n/
        // response_format/logprobs/stop/role/content-part rows on this surface.
        // `scripts/check-capability-matrix.sh` greps this file for
        // `fn <fixture_id>` and fails the build if a matrix row cites an ID
        // that no longer exists here.
        // -----------------------------------------------------------------------

        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[tokio::test]
        async fn cm_lattice_serve_model_mismatch_rejected() {
            let body = Body::from(
                r#"{"model":"wrong-model","messages":[{"role":"user","content":"hi"}]}"#
                    .to_string(),
            );
            let response =
                chat_completions(State(test_app_state()), test_json_headers(), body).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("response body must be readable");
            let value: serde_json::Value =
                serde_json::from_slice(&bytes).expect("error response must be valid JSON");
            assert_eq!(value["error"]["code"], "model_not_found");
            assert_eq!(
                value["error"]["message"],
                "model 'wrong-model' is not loaded; this server serves 'test-model'"
            );
        }

        // ── cross-binary parity table (ADR-080 C2) ─────────────────────────
        //
        // Drives every fixture body in
        // `lattice_inference::serve::CHAT_COMPLETIONS_PARITY_CASES` through
        // THIS binary's real `Router` (via `router()`, extracted from
        // `run()` specifically so it's testable in isolation from process
        // startup) and compares the resulting status + error code against
        // the case's `lattice_serve`-side expectation. `lattice.rs`'s own
        // test module runs the SAME table against its own router, asserting
        // the `lattice`-side expectation -- together the two prove
        // same-input parity (or a documented, intentional divergence) at
        // the real HTTP layer.
        //
        // Issue #828's `Json`/`Sse` rows need a REAL tiny tokenizer (via
        // `test_support::tiny_zero_model`, gated behind the `test-utils`
        // feature -- see that module's own doc comment) to drive
        // `check_prompt_fits_window` for real, so this whole module now
        // requires `test-utils`, matching `lattice.rs`'s own
        // `parity_table` module gating. CI's "serve-surface capability-
        // matrix fixtures" step adds `test-utils` alongside `f16,metal-gpu`
        // for this binary to keep running it (see `.github/workflows/ci.yml`).
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        mod parity_table {
            use super::*;
            use lattice_inference::serve::{
                BASELINE_CANNED_COMPLETION_TOKENS, BASELINE_CANNED_PROMPT_TOKENS, Binary,
                CHAT_COMPLETIONS_PARITY_CASES, ExpectedResponse, check_sse_events,
            };
            use tower::ServiceExt as _;

            /// Deterministic worker-thread generation seam for every
            /// `Json`/`Sse` row (issue #828): a REAL background thread runs
            /// the actual `run_worker_loop` + `check_prompt_fits_window`
            /// production code (via `metal_worker::spawn_fake`, issue
            /// #832's cross-binary test seam; real tiny tokenizer, no Metal
            /// engine) -- only the terminal generation call itself is
            /// replaced with a canned deterministic completion. Content
            /// deltas are pushed through `on_token` (what both the
            /// streaming and non-streaming arms of `chat_completions` read
            /// content from on this binary -- see its `WorkerEvent::Delta`
            /// accumulation), so one seam serves both response shapes.
            fn baseline_fake_worker_state(model_max_context: usize) -> AppState {
                let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                    .tokenizer()
                    .clone();
                let jobs = spawn_fake(
                    ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                    model_max_context,
                    tokenizer,
                    |_messages, _cfg, _prompt_tokens, on_token, _should_cancel| {
                        // The real `check_prompt_fits_window` (mutation-sensitive
                        // exactly like `real_router_overflow_parity` above) has
                        // already run inside `spawn_fake` by the time this
                        // closure is called; its measured `_prompt_tokens` is
                        // intentionally NOT what gets reported below -- the
                        // returned usage counts are the fixed canned figures
                        // both binaries' field checks assert against
                        // (`BASELINE_CANNED_*`).
                        for (chunk, id) in [("hello", 1u32), (" world", 2u32)] {
                            on_token(chunk, id);
                        }
                        Ok(GenerateOutput {
                            text: "hello world".to_string(),
                            token_ids: vec![1, 2],
                            prompt_tokens: BASELINE_CANNED_PROMPT_TOKENS as usize,
                            generated_tokens: BASELINE_CANNED_COMPLETION_TOKENS as usize,
                            stopped: true,
                            stop_reason: None,
                            token_logprobs: vec![],
                        })
                    },
                );
                AppState {
                    jobs,
                    model_id: Arc::from("test-model"),
                    defaults: Defaults {
                        max_tokens: 100,
                        temperature: 0.7,
                        top_k: 50,
                        top_p: 0.9,
                        repetition_penalty: 1.1,
                        reasoning_budget: None,
                    },
                    model_max_context,
                    max_pending: 1_000_000,
                    metrics: Arc::new(ServeMetrics::default()),
                    vocab_bytes: Arc::new(vec![]),
                    grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
                }
            }

            #[tokio::test]
            async fn chat_completions_matches_shared_parity_table() {
                for case in CHAT_COMPLETIONS_PARITY_CASES {
                    let expected = case.expected(Binary::LatticeServe);
                    // Error-shaped rows never reach the job queue (rejected
                    // at validation) or, for `max_tokens_over_cap_reject_vs_clamp`,
                    // rely on the no-worker harness artifact its own
                    // doc comment describes -- both keep using the plain
                    // no-worker state exactly as before #828.
                    let app = match expected {
                        ExpectedResponse::Error { .. } => router(test_app_state()),
                        ExpectedResponse::Json { .. } | ExpectedResponse::Sse { .. } => {
                            router(baseline_fake_worker_state(4096))
                        }
                    };
                    let request = axum::http::Request::builder()
                        .method(case.method)
                        .uri(case.path)
                        .header("content-type", "application/json")
                        .body(Body::from(case.body.build()))
                        .expect("fixture request must build");
                    let response = app
                        .oneshot(request)
                        .await
                        .expect("router must produce a response, not a transport error");

                    let status = response.status().as_u16();
                    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                        .await
                        .expect("response body reads");
                    let text = String::from_utf8_lossy(&body);

                    assert_eq!(
                        status,
                        expected.status(),
                        "case '{}': expected status {}, got {status} (body: {text})",
                        case.name,
                        expected.status(),
                    );

                    match expected {
                        ExpectedResponse::Error { code, .. } => {
                            let value: serde_json::Value = serde_json::from_slice(&body)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "case '{}': non-2xx response body must be the shared \
                                         error envelope JSON: {e} (body: {text})",
                                        case.name,
                                    )
                                });
                            assert_eq!(
                                value["error"]["code"], code,
                                "case '{}': expected error code '{code}', got {} \
                                 (full body: {value})",
                                case.name, value["error"]["code"]
                            );
                        }
                        ExpectedResponse::Json { fields, .. } => {
                            let value: serde_json::Value = serde_json::from_slice(&body)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "case '{}': 2xx response body must be JSON: {e} \
                                         (body: {text})",
                                        case.name,
                                    )
                                });
                            for field in fields {
                                field.check(&value).unwrap_or_else(|e| {
                                    panic!("case '{}': field check failed: {e}", case.name)
                                });
                            }
                        }
                        ExpectedResponse::Sse { events, .. } => {
                            check_sse_events(&text, events).unwrap_or_else(|e| {
                                panic!("case '{}': SSE check failed: {e}", case.name)
                            });
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Production-adapter observation (issue #828): proves the shared
        // `ProductionAdapterObservation`/`GenerateConfigSnapshot` types
        // capture what THIS binary's real `chat_completions` -> `build_cfg`
        // construction produces, not a value the test independently
        // reconstructs. The injected worker-loop `generate` closure below
        // runs strictly BELOW that real path -- it records the actual
        // `&[ChatMessage]`/`&GenerateConfig` it was called with, then
        // returns a canned result; it never recomputes `build_cfg` itself.
        // -----------------------------------------------------------------------
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        mod production_adapter_observation {
            use super::*;
            use lattice_inference::serve::{
                ExpectedObservation, GenerateConfigSnapshot,
                OBSERVATION_GOLDEN_USER_HI_THERE_CHATML, ProductionAdapterObservation,
                assert_observation_matches,
            };
            use std::sync::Mutex;

            /// Mirrors `lattice.rs`'s equivalent helper (issue #828):
            /// fires the fixed `{"messages":[{"role":"user","content":"hi
            /// there"}],"temperature":1.3,"top_p":0.55,"seed":7,"max_tokens":9}`
            /// request through a REAL background thread running the actual
            /// `run_worker_loop` + `check_prompt_fits_window` production code
            /// (via `metal_worker::spawn_fake`, issue #832's cross-binary test
            /// seam; real tiny tokenizer, no Metal engine) -- only the terminal
            /// generation call is replaced with a canned completion. `stopped`
            /// is threaded through a single local variable into both the
            /// recorded observation and the returned `GenerateOutput`, and the
            /// real `check_prompt_fits_window` measured length -- surfaced by
            /// `spawn_fake` as the closure's `prompt_tokens` argument -- is the
            /// ONLY source for `prompt_tokens` here.
            async fn run_observed(
                model_max_context: usize,
                stopped: bool,
            ) -> ProductionAdapterObservation {
                let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                    .tokenizer()
                    .clone();
                let observed: Arc<Mutex<Option<ProductionAdapterObservation>>> =
                    Arc::new(Mutex::new(None));
                let observed_for_worker = Arc::clone(&observed);
                let jobs = spawn_fake(
                    ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                    model_max_context,
                    tokenizer,
                    move |messages, cfg, prompt_tokens, on_token, _should_cancel| {
                        let normalized: Vec<(String, String)> = messages
                            .iter()
                            .map(|m| {
                                let role = match m.role {
                                    lattice_inference::forward::metal_qwen35::ChatRole::System => {
                                        "system"
                                    }
                                    lattice_inference::forward::metal_qwen35::ChatRole::User => {
                                        "user"
                                    }
                                    lattice_inference::forward::metal_qwen35::ChatRole::Assistant => {
                                        "assistant"
                                    }
                                };
                                (role.to_string(), m.content.clone())
                            })
                            .collect();
                        // Real production check (mutation-sensitive to
                        // `check_prompt_fits_window` exactly like
                        // `real_router_overflow_parity`/`baseline_fake_worker_state`
                        // above) has already run inside `spawn_fake` by the
                        // time this closure is called; its measured
                        // `prompt_tokens` IS what gets reported below.
                        *observed_for_worker
                            .lock()
                            .expect("observation mutex poisoned") =
                            Some(ProductionAdapterObservation {
                                rendered_prompt: None,
                                messages: Some(normalized),
                                gen_cfg: GenerateConfigSnapshot::from(cfg),
                                prompt_tokens,
                                stopped,
                            });
                        on_token("ok", 1);
                        Ok(GenerateOutput {
                            text: "ok".to_string(),
                            token_ids: vec![1],
                            prompt_tokens,
                            generated_tokens: 1,
                            stopped,
                            stop_reason: None,
                            token_logprobs: vec![],
                        })
                    },
                );
                let state = AppState {
                    jobs,
                    model_id: Arc::from("test-model"),
                    defaults: Defaults {
                        max_tokens: 100,
                        temperature: 0.7,
                        top_k: 50,
                        top_p: 0.9,
                        repetition_penalty: 1.1,
                        reasoning_budget: None,
                    },
                    model_max_context,
                    max_pending: 1_000_000,
                    metrics: Arc::new(ServeMetrics::default()),
                    vocab_bytes: Arc::new(vec![]),
                    grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
                };
                let body = Body::from(
                    r#"{"messages":[{"role":"user","content":"hi there"}],"temperature":1.3,"top_p":0.55,"seed":7,"max_tokens":9}"#
                        .to_string(),
                );
                let response = chat_completions(State(state), test_json_headers(), body).await;
                assert_eq!(response.status(), StatusCode::OK);

                observed
                    .lock()
                    .expect("observation mutex poisoned")
                    .clone()
                    .expect("the injected worker-loop generate closure must have recorded an observation")
            }

            /// The `GenerateConfig` `lattice_serve.rs`'s real `build_cfg` must
            /// produce for the fixed request `run_observed` sends, given
            /// `run_observed`'s `Defaults` above: every explicitly-set field
            /// mirrors the request; `build_cfg` always sets the remaining
            /// fields (`stop_token_ids`, `enable_thinking`, `enable_mtp`,
            /// `grammar`, `reasoning_budget`, `logprobs`, `stop_strings`) to
            /// the exact same values `GenerateConfig::default()` carries.
            fn expected_gen_cfg() -> GenerateConfigSnapshot {
                GenerateConfigSnapshot::from(
                    &lattice_inference::model::qwen35_config::GenerateConfig {
                        max_new_tokens: 9,
                        temperature: 1.3,
                        top_p: 0.55,
                        seed: Some(7),
                        ..Default::default()
                    },
                )
            }

            #[tokio::test]
            async fn chat_completions_non_streaming_observation_captures_real_config_and_messages()
            {
                let obs = run_observed(4096, true).await;
                let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                    .tokenizer()
                    .clone();
                let expected_prompt_tokens = tokenizer
                    .tokenize(OBSERVATION_GOLDEN_USER_HI_THERE_CHATML)
                    .real_length;
                assert_observation_matches(
                    &obs,
                    &ExpectedObservation {
                        gen_cfg: expected_gen_cfg(),
                        rendered_prompt: None,
                        messages: Some(&[("user", "hi there")]),
                        prompt_tokens: expected_prompt_tokens,
                        stopped: true,
                    },
                );
            }

            /// Proves `stopped` is genuinely derived from what the worker's
            /// generation closure returned, not an independent hardcoded
            /// literal (this was previously
            /// `stopped: true` regardless of the closure's actual return).
            #[tokio::test]
            async fn chat_completions_non_streaming_observation_captures_real_stopped_false() {
                let obs = run_observed(4096, false).await;
                assert!(
                    !obs.stopped,
                    "observation must report the worker's actual stopped=false, not a hardcoded true"
                );
            }
        }
    }
}

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
//!
//! # Design
//!
//! `MetalQwen35State` owns raw `metal::*` objects and is `!Send`, so it lives on
//! one dedicated worker thread for the whole process lifetime. The async axum
//! handlers never touch Metal directly: each request ships a `Job` (messages +
//! sampling config + a reply channel) to the worker over a tokio mpsc, and the
//! worker drives `chat_completion_streaming`, forwarding each token delta back.
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
        http::StatusCode,
        response::{
            IntoResponse, Response,
            sse::{Event, KeepAlive, Sse},
        },
        routing::{get, post},
    };
    use lattice_inference::forward::metal_qwen35::{
        ChatMessage, MetalQwen35State, format_chat_template,
    };
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::{
        GenerateConfig, QWEN_CHAT_IM_END_TOKEN_ID, Qwen35Config,
    };
    use lattice_inference::model_format::{self, ModelFormat};
    use lattice_inference::tokenizer::Tokenizer as _;
    use lattice_inference::tokenizer::bpe::BpeTokenizer;
    use serde::Deserialize;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};
    use tokio::sync::{mpsc, watch};

    // ─── worker protocol ─────────────────────────────────────────────────────

    /// One token-stream event from the worker back to a request handler.
    enum Ev {
        Delta(String),
        Done {
            prompt_tokens: usize,
            completion_tokens: usize,
            /// The engine's actual stop cause (ADR-080 C2, #746): `true` when
            /// generation ended via an explicit stop condition (EOS,
            /// stop-token-id, or stop-string match), `false` when the token
            /// budget was exhausted or the caller cancelled. Previously
            /// discarded entirely here -- both the SSE and non-streaming
            /// handlers below hardcoded `finish_reason: "stop"`
            /// unconditionally, so a length-capped completion was
            /// misreported as an explicit stop. Fed through
            /// `lattice_inference::serve::finish_reason` at both call sites.
            stopped: bool,
        },
        /// Generation failed closed instead of completing (#611: e.g. a
        /// grammar mask that blocks every candidate token, mirroring the
        /// CPU/`lattice.rs` fail-closed contract). Carries the underlying
        /// error message for server-side logging; the streaming and
        /// non-streaming handlers below decide separately how much (if any)
        /// of it is safe to surface to the HTTP client.
        Failed {
            message: String,
        },
        /// The request itself cannot fit the model's KV window (#656: prompt
        /// length was unknown to `build_cfg`, so this is only checked once
        /// the worker tokenizes the prompt). Client-caused, so it maps to
        /// HTTP 400 rather than the 500 `Failed` uses -- distinct from a
        /// generation-time failure, never a server-side problem.
        Rejected {
            message: String,
        },
    }

    /// A generation request handed to the single GPU worker thread.
    ///
    /// `cancel` reflects whether the client that submitted this job is still
    /// there. It starts `false` and flips to `true` the moment the matching
    /// [`CancelOnDrop`] guard is dropped -- i.e. the instant axum drops the
    /// response future/stream, which is exactly what happens on client
    /// disconnect (browser tab closed, `curl` killed, request future
    /// cancelled). The worker checks it (a) once at dequeue, before doing any
    /// work, and (b) independently of token emission, via
    /// `chat_completion_streaming_with_cancel`'s `should_cancel` predicate --
    /// before prefill, immediately after prefill returns, and at the top of
    /// every decode iteration -- so an abandoned job is skipped entirely,
    /// stopped before paying for prefill, or stopped within one decode step
    /// of the client leaving.
    struct Job {
        messages: Vec<ChatMessage>,
        cfg: GenerateConfig,
        tx: mpsc::UnboundedSender<Ev>,
        cancel: watch::Receiver<bool>,
    }

    /// Flips the paired `cancel` receiver to `true` when dropped. Held inside
    /// the per-request SSE stream state (streaming) or the handler's local
    /// scope (non-streaming) so it drops exactly when axum stops caring about
    /// the response — on client disconnect, or harmlessly after the request
    /// already finished normally (by then the worker has moved on anyway).
    ///
    /// ADR-080 C2 (#782): this used to be a private copy of the exact same
    /// struct; it is now `lattice_inference::serve::CancelOnDrop`, the single
    /// shared definition `lattice.rs`'s CPU streaming path also uses.
    /// Production code below only ever calls `cancel_pair()` (never names
    /// the guard type directly); the `use` is needed for the test helper's
    /// return-type annotation, hence `#[cfg(test)]`.
    #[cfg(test)]
    use lattice_inference::serve::CancelOnDrop;

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
        jobs: mpsc::UnboundedSender<Job>,
        model_id: Arc<str>,
        defaults: Defaults,
        /// Runtime context window derived from the loaded model (#551): the
        /// exact KV cache length `load_model` allocated, never a hard-coded
        /// constant. See `model_context_from_config` and `build_cfg`.
        model_max_context: usize,
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
    const IMAGE_REQUIRES_VISION_MESSAGE: &str = "image input requires a vision-capable model";

    /// A request-validation failure that must surface as HTTP 400 (#641,
    /// #649). Fail-closed: unknown roles and unsupported content parts are
    /// never coerced or dropped, they always produce one of these.
    ///
    /// `code` carries the OpenAI-style error code (ADR-080 C2 round 2,
    /// codex finding #1): previously this variant was message-only, so
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
        BadRequest { message: String, code: &'static str },
    }

    impl RequestError {
        fn bad_request(message: impl Into<String>, code: &'static str) -> Self {
            Self::BadRequest {
                message: message.into(),
                code,
            }
        }

        fn message(&self) -> &str {
            match self {
                Self::BadRequest { message, .. } => message,
            }
        }

        fn code(&self) -> &'static str {
            match self {
                Self::BadRequest { code, .. } => code,
            }
        }
    }

    #[derive(Debug, Deserialize)]
    struct ChatReq {
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        messages: Vec<InMsg>,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        top_p: Option<f32>,
        #[serde(default)]
        top_k: Option<usize>,
        #[serde(default)]
        max_tokens: Option<usize>,
        #[serde(default)]
        seed: Option<u64>,
        #[serde(default)]
        stream: Option<bool>,
        // Lattice extensions (ignored by stock OpenAI clients).
        #[serde(default)]
        repetition_penalty: Option<f32>,
        #[serde(default)]
        reasoning_budget: Option<usize>,
        // #656: known-but-unsupported OpenAI fields, modeled explicitly (not
        // left to serde's default silent-drop of unknown fields) so
        // `reject_unsupported` can name the exact offending field and
        // return HTTP 400 rather than quietly running a tool-calling/
        // JSON-mode/multi-completion request as a plain text completion.
        #[serde(default)]
        max_completion_tokens: Option<usize>,
        #[serde(default)]
        tools: Option<Value>,
        #[serde(default)]
        tool_choice: Option<Value>,
        #[serde(default)]
        response_format: Option<ResponseFormat>,
        #[serde(default)]
        n: Option<usize>,
        #[serde(default)]
        logprobs: Option<bool>,
        #[serde(default)]
        top_logprobs: Option<usize>,
        #[serde(default)]
        stop: Option<Value>,
    }

    #[derive(Debug, Deserialize)]
    struct ResponseFormat {
        r#type: String,
    }

    /// #656: reject known-but-unsupported OpenAI request fields with HTTP
    /// 400 instead of silently ignoring them -- the same fail-closed
    /// philosophy `MessageRole::parse`/`content_text` already apply to
    /// roles and content parts. Mirrors `lattice.rs`'s `reject_unsupported`,
    /// scoped to this minimal server's narrower surface: unlike
    /// `lattice.rs`, this server has no `logprobs`/`stop` implementation at
    /// all, so both are rejected outright rather than conditionally.
    fn reject_unsupported(req: &ChatReq) -> Result<(), RequestError> {
        if req.tools.is_some() || req.tool_choice.is_some() {
            return Err(RequestError::bad_request(
                "tools and tool_choice are not supported by this server",
                "unsupported_feature",
            ));
        }
        if req.n.unwrap_or(1) > 1 {
            return Err(RequestError::bad_request(
                "n > 1 is not supported",
                "unsupported_feature",
            ));
        }
        if let Some(fmt) = &req.response_format
            && fmt.r#type != "text"
        {
            return Err(RequestError::bad_request(
                format!(
                    "response_format.type '{}' is not supported; use 'text'",
                    fmt.r#type
                ),
                "unsupported_feature",
            ));
        }
        if req.logprobs.unwrap_or(false) || req.top_logprobs.is_some() {
            return Err(RequestError::bad_request(
                "logprobs/top_logprobs are not supported by this server",
                "unsupported_feature",
            ));
        }
        if req.stop.is_some() {
            return Err(RequestError::bad_request(
                "stop is not supported by this server",
                "unsupported_feature",
            ));
        }
        if let (Some(a), Some(b)) = (req.max_tokens, req.max_completion_tokens)
            && a != b
        {
            return Err(RequestError::bad_request(
                format!("max_tokens ({a}) and max_completion_tokens ({b}) differ; supply only one"),
                // Matches lattice.rs's `validate_max_tokens`, which uses the
                // generic "invalid_request" code for this exact conflict
                // (not a more specific "invalid_max_tokens" -- that code is
                // reserved for the zero/cap cases).
                "invalid_request",
            ));
        }
        Ok(())
    }

    #[derive(Debug, Deserialize)]
    struct InMsg {
        role: String,
        content: MessageContent,
    }

    /// Fail-closed chat role (#641): only these three roles are accepted.
    /// Anything else — `tool`, `developer`, a typo, an empty string — is
    /// rejected with HTTP 400 rather than silently coerced to `user`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum MessageRole {
        System,
        User,
        Assistant,
    }

    impl MessageRole {
        /// ADR-080 C2 round 2 (codex finding #1): differentiates the same
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

    /// OpenAI message content: either a plain string or an array of typed
    /// parts (`[{"type":"text","text":"..."}]`). Both forms deserialize
    /// successfully here; rejection of unsupported part types happens in
    /// `content_text` so the exact offending part is available for the error
    /// message.
    #[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
    #[serde(untagged)]
    enum MessageContent {
        Text(String),
        Parts(Vec<Part>),
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Part {
        Text {
            text: String,
        },
        ImageUrl {
            image_url: ImageUrl,
        },
        /// Any part `type` other than `text`/`image_url` (#641/#649): kept
        /// instead of dropped so it can be rejected with its real type name.
        Unsupported {
            kind: String,
        },
    }

    #[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
    struct ImageUrl {
        url: String,
        #[serde(default)]
        detail: Option<String>,
    }

    #[derive(Deserialize)]
    struct RawPart {
        #[serde(rename = "type")]
        kind: Option<String>,
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        image_url: Option<ImageUrl>,
    }

    impl<'de> Deserialize<'de> for Part {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let raw = RawPart::deserialize(deserializer)?;
            match raw.kind.as_deref() {
                Some("text") => {
                    let text = raw.text.ok_or_else(|| {
                        serde::de::Error::custom(
                            "text content part must include string field 'text'",
                        )
                    })?;
                    Ok(Self::Text { text })
                }
                Some("image_url") => {
                    let image_url = raw.image_url.ok_or_else(|| {
                        serde::de::Error::custom(
                            "image_url content part must include object field 'image_url'",
                        )
                    })?;
                    Ok(Self::ImageUrl { image_url })
                }
                Some(kind) => Ok(Self::Unsupported {
                    kind: kind.to_string(),
                }),
                None => Ok(Self::Unsupported {
                    kind: "<missing>".to_string(),
                }),
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
                                // any lattice.rs code (ADR-080 C2 round 2).
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

    /// Clamp-then-parse entry point (#649 DoS hardening): validates part
    /// counts/sizes against the raw bytes first, then deserializes the typed
    /// request. Never allocates the request's strings/arrays before the
    /// clamp has run.
    fn parse_chat_req(body: &[u8]) -> Result<ChatReq, RequestError> {
        validate_content_part_limits(body)?;
        serde_json::from_slice::<ChatReq>(body).map_err(|_| {
            RequestError::bad_request(
                "invalid JSON request body",
                // Matches lattice.rs's JSON-extraction-failure code
                // (ADR-080 C2 round 2, codex finding #1).
                "invalid_request_body",
            )
        })
    }

    /// Flatten message content to plain text (#641, #649). Fails closed:
    /// an `image_url` part returns the vision-model message, any other
    /// unsupported part type names itself in the error rather than being
    /// silently dropped.
    fn content_text(content: &MessageContent) -> Result<String, RequestError> {
        match content {
            MessageContent::Text(text) => Ok(text.clone()),
            MessageContent::Parts(parts) => {
                let mut out = String::new();
                for part in parts {
                    match part {
                        Part::Text { text } => out.push_str(text),
                        Part::ImageUrl { .. } => {
                            // Matches lattice.rs's `message_text` image
                            // rejection code (ADR-080 C2 round 2, codex
                            // finding #1).
                            return Err(RequestError::bad_request(
                                IMAGE_REQUIRES_VISION_MESSAGE,
                                "unsupported_feature",
                            ));
                        }
                        Part::Unsupported { kind } => {
                            return Err(RequestError::bad_request(
                                format!(
                                    "unsupported content part type '{kind}'; only 'text' parts are accepted"
                                ),
                                "unsupported_feature",
                            ));
                        }
                    }
                }
                Ok(out)
            }
        }
    }

    fn to_chat_message(m: &InMsg) -> Result<ChatMessage, RequestError> {
        let role = MessageRole::parse(&m.role)?;
        let content = content_text(&m.content)?;
        Ok(match role {
            MessageRole::System => ChatMessage::system(content),
            MessageRole::User => ChatMessage::user(content),
            MessageRole::Assistant => ChatMessage::assistant(content),
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

    fn build_cfg(
        req: &ChatReq,
        d: &Defaults,
        model_max_context: usize,
    ) -> Result<GenerateConfig, lattice_inference::serve::ApiError> {
        // #745 (ADR-080 C2): reject a caller-supplied `max_tokens: 0` (or
        // `max_completion_tokens: 0`) up front, on the alias-resolved but
        // still-unclamped value -- previously this clamped straight through
        // via `.min(..)` below into a zero-budget completion instead of a
        // clear rejection. Checked before the clamp so the rejection reflects
        // the caller's actual request, not a clamp artifact.
        let max_new_tokens_requested = req
            .max_tokens
            // #656: `max_completion_tokens` is the current OpenAI field name;
            // `reject_unsupported` already rejected the two being present
            // and disagreeing, so whichever is set here is the caller's
            // single, unambiguous intent.
            .or(req.max_completion_tokens)
            .unwrap_or(d.max_tokens);
        lattice_inference::serve::reject_zero_max_tokens(max_new_tokens_requested)?;
        // Clamp like `max_tokens`: a budget past the KV window is meaningless and
        // would let a future `with_capacity(decode_cap(..))` abort on overflow.
        // Reserve one slot of headroom so the worst-case decode cap below
        // (`reasoning_budget + max_new_tokens + 1`) never exceeds the KV
        // window even when `reasoning_budget` is absent/zero.
        let max_new_tokens = max_new_tokens_requested.min(model_max_context.saturating_sub(1));
        // The worst-case decode cap is `reasoning_budget + max_new_tokens + 1`
        // (#551): keep it at or below the KV window the worker actually
        // allocated, not just below `model_max_context` in isolation.
        let reasoning_room = model_max_context
            .saturating_sub(max_new_tokens)
            .saturating_sub(1);
        let reasoning_budget = req
            .reasoning_budget
            .filter(|&n| n > 0)
            .or(d.reasoning_budget)
            .map(|n| n.min(reasoning_room))
            .filter(|&n| n > 0);
        Ok(GenerateConfig {
            max_new_tokens,
            temperature: req.temperature.unwrap_or(d.temperature),
            top_k: req.top_k.unwrap_or(d.top_k),
            top_p: req.top_p.unwrap_or(d.top_p),
            repetition_penalty: req.repetition_penalty.unwrap_or(d.repetition_penalty),
            seed: req.seed,
            stop_token_ids: vec![QWEN_CHAT_IM_END_TOKEN_ID],
            enable_thinking: true,
            enable_mtp: None,
            grammar: None,
            stop_strings: vec![],
            reasoning_budget,
            // ChatReq models `logprobs`/`top_logprobs` (#656) only so
            // `reject_unsupported` can fail closed with HTTP 400 -- this
            // minimal server has no logprobs implementation at all, so a
            // `logprobs: true` request never reaches here (see below).
            logprobs: None,
            disable_eos: false,
        })
    }

    /// Prefix marker used only by the KV-window pre-check in `spawn_worker`'s
    /// `generate` closure: distinguishes a client-caused window-overflow
    /// rejection (HTTP 400, `Ev::Rejected`) from a genuine generation
    /// failure (HTTP 500, `Ev::Failed`) without changing `run_worker_loop`'s
    /// generic `Result<(usize, usize), String>` contract, which every other
    /// caller and test fake still uses unchanged.
    const PROMPT_EXCEEDS_WINDOW_PREFIX: &str = "prompt-exceeds-window: ";

    /// #656: `build_cfg` only clamps `max_new_tokens`/`reasoning_budget`
    /// against `model_max_context` *in isolation* -- it has no visibility
    /// into `prompt_len` (unknown until the worker tokenizes the prompt).
    /// The invariant that must actually hold is the full-window one:
    /// `prompt_len + max_new_tokens + reasoning_budget + 1 <= model_max_context`.
    /// Called by the worker right after tokenizing, before generation
    /// starts, so a request that would overrun the KV cache is rejected up
    /// front instead of silently hitting `StopReason::KvFull` mid-generation
    /// while the response still claims `finish_reason: "stop"`.
    fn check_prompt_fits_window(
        model_max_context: usize,
        prompt_len: usize,
        cfg: &GenerateConfig,
    ) -> Result<(), String> {
        let decode_cap = cfg
            .max_new_tokens
            .saturating_add(cfg.reasoning_budget.unwrap_or(0));
        let required = prompt_len.saturating_add(decode_cap).saturating_add(1);
        if required > model_max_context {
            let available = model_max_context.saturating_sub(prompt_len);
            return Err(format!(
                "prompt has {prompt_len} tokens, leaving {available} of the \
                 {model_max_context}-token context window for generation, but this \
                 request needs {decode_cap} generated tokens plus 1 (total {required}); \
                 reduce max_tokens/reasoning_budget or shorten the prompt"
            ));
        }
        Ok(())
    }

    /// Tokenizes `messages` and enforces the full KV-window invariant
    /// (`check_prompt_fits_window`) before any generation work starts.
    ///
    /// ADR-080 C2 round 3 (codex round-3 medium finding #2): `spawn_worker`'s
    /// real Metal closure below and the real-router streaming
    /// context-overflow parity test in this module's test suite both call
    /// this EXACT function -- not two independently-written copies of the
    /// same check -- so a mutation that guts the window enforcement here is
    /// observed by production's own worker closure AND by the test's worker
    /// seam identically. There is no separate test-only reimplementation of
    /// `check_prompt_fits_window` for such a mutation to leave unmutated.
    fn enforce_prompt_window(
        tokenizer: &BpeTokenizer,
        model_max_context: usize,
        messages: &[ChatMessage],
        cfg: &GenerateConfig,
    ) -> Result<usize, String> {
        let prompt_len = tokenizer
            .tokenize(&format_chat_template(messages))
            .real_length;
        if let Err(msg) = check_prompt_fits_window(model_max_context, prompt_len, cfg) {
            return Err(format!("{PROMPT_EXCEEDS_WINDOW_PREFIX}{msg}"));
        }
        Ok(prompt_len)
    }

    // ─── GPU worker thread ───────────────────────────────────────────────────

    /// Spawn the dedicated thread that owns the `!Send` Metal state. Loads the
    /// model, signals readiness (or a load error) over `ready`, then serves jobs
    /// serially until all `Job` senders drop.
    /// Worker readiness payload (#551): carries the actual KV context the
    /// worker allocated so `run()` can store it in `AppState` instead of a
    /// hard-coded constant.
    struct WorkerReady {
        format: String,
        model_max_context: usize,
    }

    fn spawn_worker(
        model_dir: std::path::PathBuf,
        tokenizer_path: std::path::PathBuf,
        format: ModelFormat,
        ready: std::sync::mpsc::Sender<Result<WorkerReady, String>>,
    ) -> mpsc::UnboundedSender<Job> {
        let (job_tx, job_rx) = mpsc::unbounded_channel::<Job>();
        std::thread::spawn(move || {
            let loaded = load_model(&model_dir, &tokenizer_path, format);
            let LoadedModel {
                mut metal,
                tokenizer,
                format,
                model_max_context,
            } = match loaded {
                Ok(t) => t,
                Err(e) => {
                    let _ = ready.send(Err(e));
                    return;
                }
            };
            let _ = ready.send(Ok(WorkerReady {
                format,
                model_max_context,
            }));

            run_worker_loop(job_rx, move |messages, cfg, on_token, should_cancel| {
                // #656: verify the FULL window invariant (prompt included)
                // before doing any GPU work -- `cfg` alone was already
                // clamped by `build_cfg`, but only against the window in
                // isolation, not against this specific prompt's length.
                enforce_prompt_window(&tokenizer, model_max_context, messages, cfg)?;
                // Cache-aware + cancellation-aware call (#462): reuses the
                // previous turn's shared token prefix instead of the old
                // unconditional `reset_state()` + full re-prefill on every
                // request, while still observing client disconnect exactly
                // like the old `chat_completion_streaming_with_cancel` call
                // did (see `Job::cancel`'s doc comment above for what
                // `should_cancel` observes). This is a single-worker,
                // single-model binary sharing one Metal state across every
                // request, so `CrossTurnSlotId::DEFAULT` is the only slot
                // that exists and correctness does not depend on
                // distinguishing clients: the planner token-verifies the
                // retained prefix against this request's messages on every
                // call and falls back to `PrefixReuseMode::FullRefill`
                // whenever they diverge (a new conversation, edited history,
                // or a second client's unrelated prompt interleaved on the
                // same slot). Multi-client interleaving through one slot
                // only forfeits the reuse speedup for whichever turn loses
                // the shared prefix — it can never corrupt output, since the
                // engine never trusts the cache without re-verifying it.
                match metal.chat_completion_streaming_with_prefix_cache_and_cancel(
                    lattice_inference::kv_cache::CrossTurnSlotId::DEFAULT,
                    messages,
                    &tokenizer,
                    cfg,
                    on_token,
                    should_cancel,
                ) {
                    Ok(cached) => {
                        eprintln!(
                            "[lattice_serve] cross-turn cache: mode={:?} reused={} prefetched={} prompt={}",
                            cached.cache.mode,
                            cached.cache.reused_tokens,
                            cached.cache.prefetched_tokens,
                            cached.cache.prompt_tokens,
                        );
                        Ok((
                            cached.output.prompt_tokens,
                            cached.output.completion_tokens,
                            cached.output.stopped,
                        ))
                    }
                    // Fail-closed at the engine level already (live KV/GDN
                    // state and the retained prefix entry are both reset
                    // before this error is returned), so the worker is left
                    // clean for the next job.
                    Err(e) => Err(e.to_string()),
                }
            });
        });
        job_tx
    }

    /// Dequeue -> cancel-check -> generate -> reply, serialized on whatever
    /// thread calls this (the dedicated Metal worker thread in production).
    ///
    /// `generate` is injected so tests can swap in a fake, GPU-free generator
    /// while exercising the exact same queue/cancellation logic production
    /// uses. It must call `on_token` for each generated delta and stop as
    /// soon as `on_token` returns `false`; it must also poll `should_cancel`
    /// independently of `on_token` -- including during any phase that never
    /// calls `on_token` at all (a prefill-like section, or a run of
    /// empty-delta steps) -- and stop as soon as `should_cancel` returns
    /// `true`. Either way, return
    /// `Ok((prompt_tokens, completion_tokens, stopped))` for whatever was
    /// actually produced before stopping (early or at the cap) -- `stopped`
    /// is the engine's actual stop cause (ADR-080 C2, #746: `true` for an
    /// explicit stop condition, `false` for a length cap or cancellation),
    /// fed through `Ev::Done` instead of being discarded -- or
    /// `Err(message)` if generation itself failed closed (#611: e.g. a
    /// grammar mask that blocks every candidate token) rather than
    /// completing or being cancelled.
    fn run_worker_loop(
        mut job_rx: mpsc::UnboundedReceiver<Job>,
        mut generate: impl FnMut(
            &[ChatMessage],
            &GenerateConfig,
            &mut dyn FnMut(&str, u32) -> bool,
            &mut dyn FnMut() -> bool,
        ) -> Result<(usize, usize, bool), String>,
    ) {
        while let Some(job) = job_rx.blocking_recv() {
            if *job.cancel.borrow() {
                // The client was already gone before we ever got to this job:
                // skip it entirely, no prefill, no decode, no reply.
                continue;
            }
            let cb_tx = job.tx.clone();
            let cancel_for_token = job.cancel.clone();
            let mut on_token = move |delta: &str, _id: u32| {
                if *cancel_for_token.borrow() {
                    return false;
                }
                // `send` also fails once the client hangs up; kept as a
                // second, independent check so a job whose cancellation
                // notification is somehow delayed still stops the instant
                // its reply channel is gone.
                cb_tx.send(Ev::Delta(delta.to_string())).is_ok()
            };
            // Separate from `on_token`: this is what reaches the generator's
            // prefill gap and its empty-delta decode iterations, neither of
            // which ever calls `on_token` (see
            // `MetalQwen35State::generate_streaming_with_cancel`).
            let cancel_for_predicate = job.cancel.clone();
            let mut should_cancel = move || *cancel_for_predicate.borrow();
            match generate(&job.messages, &job.cfg, &mut on_token, &mut should_cancel) {
                Ok((prompt_tokens, completion_tokens, stopped)) => {
                    let _ = job.tx.send(Ev::Done {
                        prompt_tokens,
                        completion_tokens,
                        stopped,
                    });
                }
                Err(message) => {
                    if let Some(client_message) = message.strip_prefix(PROMPT_EXCEEDS_WINDOW_PREFIX)
                    {
                        let _ = job.tx.send(Ev::Rejected {
                            message: client_message.to_string(),
                        });
                    } else {
                        eprintln!("generation error: {message}");
                        let _ = job.tx.send(Ev::Failed { message });
                    }
                }
            }
        }
    }

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
                let has_config_json = model_dir.join("config.json").exists();
                let cfg = if has_config_json {
                    Qwen35Config::from_config_json(&model_dir.join("config.json"))
                        .map_err(|e| format!("config.json parse failed: {e}"))?
                } else {
                    Qwen35Config::qwen36_27b()
                };
                let requested_context = if has_config_json {
                    model_context_from_config(Some(&cfg))
                } else {
                    model_context_from_config(None)
                };
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

    async fn health() -> &'static str {
        let t = Instant::now();
        emit_serve_event(
            "GET",
            "/health",
            200,
            None,
            t.elapsed().as_secs_f64() * 1000.0,
            false,
        );
        "ok"
    }

    async fn root() -> Json<Value> {
        let t = Instant::now();
        // ADR-080 C2 round 2: shared with lattice.rs's equivalent route so
        // both binaries advertise the same engine-identity document.
        let body = lattice_inference::serve::root_body();
        emit_serve_event(
            "GET",
            "/",
            200,
            None,
            t.elapsed().as_secs_f64() * 1000.0,
            false,
        );
        Json(body)
    }

    async fn list_models(State(s): State<AppState>) -> Json<Value> {
        let t = Instant::now();
        // ADR-080 C2: shared with `lattice.rs`'s equivalent route so both
        // binaries advertise the single loaded model in byte-identical shape.
        let body = lattice_inference::serve::models_list_body(s.model_id.as_ref(), unix_secs());
        emit_serve_event(
            "GET",
            "/v1/models",
            200,
            None,
            t.elapsed().as_secs_f64() * 1000.0,
            false,
        );
        Json(body)
    }

    /// Phase machine for the SSE token stream.
    /// `Done` and `End` carry the completion token count so the terminal
    /// phase can include it in the telemetry event.
    enum Phase {
        Start,
        Body,
        Done(usize), // holds completion_tokens from worker
        End(usize),  // holds completion_tokens; emits telemetry then stream ends
    }

    async fn chat_completions(State(s): State<AppState>, body: Body) -> Response {
        let timer = Instant::now();
        let Ok(body) = to_bytes(body, REQUEST_BODY_LIMIT_BYTES).await else {
            emit_serve_event(
                "POST",
                "/v1/chat/completions",
                413,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
            );
            // ADR-080 C2 round 2 (codex finding #1): previously mapped to
            // HTTP 400 + generic "invalid_request", diverging from
            // lattice.rs's 413 + "request_body_too_large" for the identical
            // oversized-body condition. Aligned to match.
            return err_response(
                StatusCode::PAYLOAD_TOO_LARGE,
                &format!("request body exceeds {REQUEST_BODY_LIMIT_BYTES} bytes"),
                "request_body_too_large",
            );
        };
        let req = match parse_chat_req(&body) {
            Ok(req) => req,
            Err(err) => {
                emit_serve_event(
                    "POST",
                    "/v1/chat/completions",
                    400,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                );
                return err_response(StatusCode::BAD_REQUEST, err.message(), err.code());
            }
        };
        if let Err(err) = reject_unsupported(&req) {
            emit_serve_event(
                "POST",
                "/v1/chat/completions",
                400,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
            );
            return err_response(StatusCode::BAD_REQUEST, err.message(), err.code());
        }
        if req.messages.is_empty() {
            emit_serve_event(
                "POST",
                "/v1/chat/completions",
                400,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
            );
            // Matches lattice.rs's `validate_chat_request` code for the
            // identical empty-messages condition (ADR-080 C2 round 2).
            return err_response(
                StatusCode::BAD_REQUEST,
                "`messages` must not be empty",
                "invalid_messages",
            );
        }

        let messages: Vec<ChatMessage> = match req.messages.iter().map(to_chat_message).collect() {
            Ok(messages) => messages,
            Err(err) => {
                emit_serve_event(
                    "POST",
                    "/v1/chat/completions",
                    400,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                );
                return err_response(StatusCode::BAD_REQUEST, err.message(), err.code());
            }
        };
        let cfg = match build_cfg(&req, &s.defaults, s.model_max_context) {
            Ok(cfg) => cfg,
            Err(err) => {
                emit_serve_event(
                    "POST",
                    "/v1/chat/completions",
                    400,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    false,
                );
                // ADR-080 C2 round 2 (codex finding #1): `build_cfg` already
                // returns the shared `lattice_inference::serve::ApiError`
                // with the correct status+code (e.g. `invalid_max_tokens`
                // for #745's max_tokens=0 rejection) -- routing it through
                // `err_response`'s (StatusCode, &str) signature discarded
                // that code and re-wrapped it as generic "invalid_request".
                // Propagate the original error directly instead.
                return err.into_response();
            }
        };
        let model_id = req.model.clone().unwrap_or_else(|| s.model_id.to_string());
        let streaming = req.stream.unwrap_or(false);
        let id = format!("chatcmpl-{}", unix_nanos());
        let created = unix_secs();

        let (tx, mut rx) = mpsc::unbounded_channel::<Ev>();
        let (cancel_guard, cancel_rx) = lattice_inference::serve::cancel_pair();
        if s.jobs
            .send(Job {
                messages,
                cfg,
                tx,
                cancel: cancel_rx,
            })
            .is_err()
        {
            emit_serve_event(
                "POST",
                "/v1/chat/completions",
                500,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                streaming,
            );
            return err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference worker unavailable",
                "internal_error",
            );
        }
        // Dropped when nobody cares about the response anymore: at the end of
        // this SSE stream (moved in below) or at the end of this function for
        // the non-streaming branch. Either way that's the client disconnect
        // signal the worker checks in `run_worker_loop`.
        if streaming {
            // ADR-080 C2 round 2, codex round-2 major finding #1: without this
            // preflight, a prompt-plus-budget overflow was only discoverable
            // AFTER the HTTP response had already committed to 200 SSE (the
            // worker's `Ev::Rejected` arrived mid-stream, terminating with
            // `finish_reason: "length"` instead of the 400
            // `context_length_exceeded` the non-streaming path and
            // `lattice.rs`'s `check_context_window` preflight both return for
            // the identical request body). Await the worker's FIRST event
            // before deciding the status code at all -- exactly the
            // non-streaming branch's own ordering below, just not looping to
            // drain every event yet. `Ev::Rejected`/`Ev::Failed` on this first
            // event get the same un-committed 400/500 treatment as
            // non-streaming; anything else (`Ev::Delta`/`Ev::Done`) means the
            // request passed the worker's checks, so NOW commit to 200 SSE --
            // carrying that already-consumed first event into the stream's
            // state machine (`pending`) so it isn't silently dropped.
            let Some(first_ev) = rx.recv().await else {
                emit_serve_event(
                    "POST",
                    "/v1/chat/completions",
                    500,
                    None,
                    timer.elapsed().as_secs_f64() * 1000.0,
                    true,
                );
                return err_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "inference worker unavailable",
                    "internal_error",
                );
            };
            let first_ev = match first_ev {
                Ev::Rejected { message } => {
                    emit_serve_event(
                        "POST",
                        "/v1/chat/completions",
                        400,
                        None,
                        timer.elapsed().as_secs_f64() * 1000.0,
                        true,
                    );
                    return err_response(
                        StatusCode::BAD_REQUEST,
                        &message,
                        "context_length_exceeded",
                    );
                }
                Ev::Failed { message } => {
                    eprintln!("generation error (streaming): {message}");
                    emit_serve_event(
                        "POST",
                        "/v1/chat/completions",
                        500,
                        None,
                        timer.elapsed().as_secs_f64() * 1000.0,
                        true,
                    );
                    return err_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "inference failed",
                        "internal_error",
                    );
                }
                ev @ (Ev::Delta(_) | Ev::Done { .. }) => ev,
            };
            let stream = futures::stream::unfold(
                (rx, Phase::Start, cancel_guard, Some(first_ev)),
                move |(mut rx, phase, cancel_guard, mut pending)| {
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
                                    (rx, Phase::Body, cancel_guard, pending),
                                ))
                            }
                            Phase::Body => match match pending.take() {
                                Some(ev) => Some(ev),
                                None => rx.recv().await,
                            } {
                                Some(Ev::Delta(d)) => {
                                    let chunk = json!({
                                        "id": id, "object": "chat.completion.chunk",
                                        "created": created, "model": model,
                                        "choices": [{"index": 0, "delta": {"content": d}, "finish_reason": null}],
                                    });
                                    Some((
                                        Ok(Event::default().data(chunk.to_string())),
                                        (rx, Phase::Body, cancel_guard, None),
                                    ))
                                }
                                Some(Ev::Done {
                                    completion_tokens: ct,
                                    stopped,
                                    ..
                                }) => {
                                    // ADR-080 C2, #746: the engine's actual
                                    // stop cause, not a hardcoded "stop" --
                                    // `finish_reason` is "length" whenever
                                    // `stopped` is false (token cap or
                                    // cancellation), matching `lattice.rs`'s
                                    // contract exactly.
                                    let finish_reason =
                                        lattice_inference::serve::finish_reason(stopped);
                                    let chunk = json!({
                                        "id": id, "object": "chat.completion.chunk",
                                        "created": created, "model": model,
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                                    });
                                    Some((
                                        Ok(Event::default().data(chunk.to_string())),
                                        (rx, Phase::Done(ct), cancel_guard, None),
                                    ))
                                }
                                Some(Ev::Failed { message }) => {
                                    // The HTTP response was already committed as
                                    // 200 + text/event-stream when this SSE stream
                                    // started, so an error mid-stream cannot change
                                    // the status code. Mirror `lattice.rs`'s
                                    // `StreamMsg::Failed` contract: log the real
                                    // cause server-side (#611: e.g. a grammar mask
                                    // that blocks every candidate token) and give
                                    // the client a well-formed termination rather
                                    // than hanging the stream open.
                                    eprintln!("generation error (streaming): {message}");
                                    let chunk = json!({
                                        "id": id, "object": "chat.completion.chunk",
                                        "created": created, "model": model,
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                    });
                                    Some((
                                        Ok(Event::default().data(chunk.to_string())),
                                        (rx, Phase::Done(0), cancel_guard, None),
                                    ))
                                }
                                Some(Ev::Rejected { message }) => {
                                    // Structurally unreachable at this point:
                                    // `run_worker_loop`'s `check_prompt_fits_window`
                                    // call happens before any `Ev::Delta`/`Ev::Done`
                                    // is ever sent for a job, so `Ev::Rejected` can
                                    // only ever be the FIRST event a job produces --
                                    // and the preflight above (ADR-080 C2 round 2,
                                    // codex round-2 major finding #1) already
                                    // intercepts that one before committing to 200
                                    // SSE. Kept as a defensive fallback (same
                                    // graceful-termination shape as before) in case
                                    // that invariant ever changes rather than
                                    // deleting the arm outright.
                                    eprintln!("request rejected (streaming): {message}");
                                    let chunk = json!({
                                        "id": id, "object": "chat.completion.chunk",
                                        "created": created, "model": model,
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                                    });
                                    Some((
                                        Ok(Event::default().data(chunk.to_string())),
                                        (rx, Phase::Done(0), cancel_guard, None),
                                    ))
                                }
                                None => {
                                    let chunk = json!({
                                        "id": id, "object": "chat.completion.chunk",
                                        "created": created, "model": model,
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                    });
                                    Some((
                                        Ok(Event::default().data(chunk.to_string())),
                                        (rx, Phase::Done(0), cancel_guard, None),
                                    ))
                                }
                            },
                            Phase::Done(ct) => Some((
                                Ok(Event::default().data("[DONE]")),
                                (rx, Phase::End(ct), cancel_guard, None),
                            )),
                            Phase::End(ct) => {
                                emit_serve_event(
                                    "POST",
                                    "/v1/chat/completions",
                                    200,
                                    Some(ct),
                                    timer.elapsed().as_secs_f64() * 1000.0,
                                    true,
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
            let mut content = String::new();
            let mut prompt_tokens = 0usize;
            let mut completion_tokens = 0usize;
            let mut stopped = false;
            while let Some(ev) = rx.recv().await {
                match ev {
                    Ev::Delta(d) => content.push_str(&d),
                    Ev::Done {
                        prompt_tokens: pt,
                        completion_tokens: ct,
                        stopped: s,
                    } => {
                        prompt_tokens = pt;
                        completion_tokens = ct;
                        stopped = s;
                    }
                    Ev::Failed { message } => {
                        // Unlike streaming, the response has not been committed
                        // yet, so a generation failure (#611: e.g. a grammar mask
                        // that blocks every candidate token) can still surface as
                        // a real HTTP error instead of a disguised 200 -- the
                        // same "generic 500, specific detail logged server-side"
                        // contract the CPU/Metal handlers in `lattice.rs` use.
                        eprintln!("generation error: {message}");
                        emit_serve_event(
                            "POST",
                            "/v1/chat/completions",
                            500,
                            None,
                            timer.elapsed().as_secs_f64() * 1000.0,
                            false,
                        );
                        return err_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "inference failed",
                            "internal_error",
                        );
                    }
                    Ev::Rejected { message } => {
                        // #656: client-caused request-contract violation (the
                        // prompt plus requested generation does not fit the
                        // model's KV window), caught before any generation
                        // ran -- unlike `Ev::Failed`, the response has not
                        // been committed yet, so this surfaces as a real 400
                        // with the specific reason, not a generic 500.
                        emit_serve_event(
                            "POST",
                            "/v1/chat/completions",
                            400,
                            None,
                            timer.elapsed().as_secs_f64() * 1000.0,
                            false,
                        );
                        // Matches lattice.rs's `check_context_window` code
                        // for the analogous prompt-plus-budget-exceeds-window
                        // condition (ADR-080 C2 round 2, codex finding #1).
                        return err_response(
                            StatusCode::BAD_REQUEST,
                            &message,
                            "context_length_exceeded",
                        );
                    }
                }
            }
            // ADR-080 C2, #746: the engine's actual stop cause, not a
            // hardcoded "stop" -- matches the streaming path and
            // `lattice.rs`'s non-streaming contract.
            let finish_reason = lattice_inference::serve::finish_reason(stopped);
            let body = json!({
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
            emit_serve_event(
                "POST",
                "/v1/chat/completions",
                200,
                Some(completion_tokens),
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
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
    /// (ADR-080 C2 round 2, codex finding #1): previously hardcoded to the
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

    /// Print a structured telemetry line to stdout for the app bridge to parse.
    fn emit_serve_event(
        method: &str,
        route: &str,
        status: u16,
        tokens: Option<usize>,
        dur_ms: f64,
        stream: bool,
    ) {
        println!(
            "@@lattice {}",
            json!({
                "ev": "http_request",
                "method": method,
                "route": route,
                "status": status,
                "tokens": tokens,
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
    /// (ADR-080 C2 round 2, codex finding #1). Deliberately does NOT install
    /// `DefaultBodyLimit` the way `lattice.rs`'s `router()` does: this
    /// binary enforces the same [`lattice_inference::serve::REQUEST_BODY_LIMIT_BYTES`]
    /// cap manually inside `chat_completions` via `to_bytes`, a documented
    /// intentional divergence in ENFORCEMENT MECHANISM only (axum's
    /// `DefaultBodyLimit` layer vs. a direct `to_bytes` cap) -- the
    /// resulting status/code (413 `request_body_too_large`) is identical on
    /// both binaries today (round 2, codex finding #1's fix; see the
    /// `oversized_body_over_limit` parity case).
    pub fn router(state: AppState) -> Router {
        Router::new()
            .route("/", get(root))
            .route("/health", get(health))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
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

        eprintln!(
            "[lattice_serve] loading model from {} ({}) ...",
            model_dir.display(),
            match format {
                ModelFormat::Q4 => "q4",
                ModelFormat::Safetensors => "bf16",
                ModelFormat::Unknown => "unknown",
            }
        );
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let jobs = spawn_worker(model_dir.clone(), tokenizer_path, format, ready_tx);
        let WorkerReady {
            format: fmt,
            model_max_context,
        } = match ready_rx.recv() {
            Ok(Ok(ready)) => ready,
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => return Err("worker thread exited during model load".into()),
        };

        let model_id: Arc<str> = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("lattice")
            .into();
        eprintln!("[lattice_serve] model '{model_id}' ({fmt}) ready (context={model_max_context})");

        let state = AppState {
            jobs,
            model_id,
            defaults,
            model_max_context,
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
            eprintln!("[lattice_serve]   POST /v1/chat/completions   GET /v1/models   GET /health");
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
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::time::Duration;

        /// A GPU-free stand-in for `MetalQwen35State::chat_completion_streaming_with_cancel`:
        /// "generates" up to `cap` fake tokens, sleeping briefly between each so
        /// a cancelled job has many opportunities to be observed running past
        /// where it should have stopped. Counts how many times it was entered
        /// (`started`) and how many fake tokens actually ran (`ran_tokens`), so
        /// tests can assert a cancelled queued job's generator was never called
        /// at all. Checks `should_cancel` at the top of each iteration in
        /// addition to `on_token`'s own check, mirroring the production
        /// contract; existing tests here only rely on the `on_token` path, so
        /// this addition does not change their outcomes.
        #[allow(clippy::type_complexity)]
        fn fake_generate(
            cap: usize,
            started: Arc<AtomicUsize>,
            ran_tokens: Arc<AtomicUsize>,
        ) -> impl FnMut(
            &[ChatMessage],
            &GenerateConfig,
            &mut dyn FnMut(&str, u32) -> bool,
            &mut dyn FnMut() -> bool,
        ) -> Result<(usize, usize, bool), String> {
            move |_messages, _cfg, on_token, should_cancel| {
                started.fetch_add(1, Ordering::SeqCst);
                let mut n = 0usize;
                for i in 0..cap {
                    std::thread::sleep(Duration::from_millis(5));
                    if should_cancel() {
                        break;
                    }
                    if !on_token("x", i as u32) {
                        break;
                    }
                    n += 1;
                    ran_tokens.fetch_add(1, Ordering::SeqCst);
                }
                Ok((1, n, false))
            }
        }

        /// A GPU-free fake with an explicit prefill-like phase *before* any
        /// `on_token` call -- mirroring the real gap this fix closes:
        /// production prefill has no callback point at all, so only
        /// `should_cancel` (never `on_token`) can observe a disconnect that
        /// happens during it. `entered_decode` flips only if the prefill-like
        /// phase runs to completion uncancelled, so tests can assert it never
        /// does.
        #[allow(clippy::type_complexity)]
        fn fake_generate_with_prefill_gap(
            prefill_steps: usize,
            decode_cap: usize,
            entered_decode: Arc<AtomicBool>,
        ) -> impl FnMut(
            &[ChatMessage],
            &GenerateConfig,
            &mut dyn FnMut(&str, u32) -> bool,
            &mut dyn FnMut() -> bool,
        ) -> Result<(usize, usize, bool), String> {
            move |_messages, _cfg, on_token, should_cancel| {
                for _ in 0..prefill_steps {
                    std::thread::sleep(Duration::from_millis(5));
                    if should_cancel() {
                        return Ok((1, 0, false));
                    }
                }
                entered_decode.store(true, Ordering::SeqCst);
                let mut n = 0usize;
                for i in 0..decode_cap {
                    std::thread::sleep(Duration::from_millis(5));
                    if should_cancel() {
                        break;
                    }
                    if !on_token("x", i as u32) {
                        break;
                    }
                    n += 1;
                }
                Ok((1, n, false))
            }
        }

        /// Builds a `Job` plus the receiver its worker replies on and the guard
        /// that cancels it when dropped (the same guard `chat_completions`
        /// moves into the SSE stream / keeps local for non-streaming, standing
        /// in here for "the client is still connected").
        fn make_job() -> (Job, mpsc::UnboundedReceiver<Ev>, CancelOnDrop) {
            let (tx, rx) = mpsc::unbounded_channel::<Ev>();
            let (cancel_guard, cancel_rx) = lattice_inference::serve::cancel_pair();
            let job = Job {
                messages: vec![ChatMessage::user("hi")],
                cfg: GenerateConfig::default(),
                tx,
                cancel: cancel_rx,
            };
            (job, rx, cancel_guard)
        }

        #[test]
        fn queued_job_cancelled_before_dequeue_is_skipped_entirely() {
            let (job_tx, job_rx) = mpsc::unbounded_channel::<Job>();
            let started = Arc::new(AtomicUsize::new(0));
            let ran_tokens = Arc::new(AtomicUsize::new(0));

            // Job 1 occupies the worker (50 fake tokens, 5ms apart = ~250ms)
            // long enough that job 2 is still sitting in the queue, untouched,
            // when we cancel it a few lines down.
            let (job1, rx1, _guard1) = make_job();
            job_tx.send(job1).unwrap();

            // Job 2: cancelled client-side (guard dropped) immediately, while
            // it is still queued behind job 1.
            let (job2, mut rx2, guard2) = make_job();
            job_tx.send(job2).unwrap();
            drop(guard2);

            // Job 3: submitted after the cancelled one, to prove the worker
            // moves on and keeps serving correctly afterward.
            let (job3, rx3, _guard3) = make_job();
            job_tx.send(job3).unwrap();
            drop(job_tx);

            let started2 = started.clone();
            let ran2 = ran_tokens.clone();
            let handle = std::thread::spawn(move || {
                run_worker_loop(job_rx, fake_generate(50, started2, ran2))
            });

            let completion_tokens_of = |mut rx: mpsc::UnboundedReceiver<Ev>| -> Option<usize> {
                let mut ct = None;
                while let Some(ev) = rx.blocking_recv() {
                    if let Ev::Done {
                        completion_tokens, ..
                    } = ev
                    {
                        ct = Some(completion_tokens);
                    }
                }
                ct
            };

            assert_eq!(
                completion_tokens_of(rx1),
                Some(50),
                "job 1 should run to completion undisturbed"
            );

            // Job 2 must produce NOTHING: no Delta, no Done -- the worker
            // `continue`d past it without ever touching `generate`, so its
            // `tx` is simply dropped with the rest of the `Job`.
            assert!(
                rx2.blocking_recv().is_none(),
                "cancelled queued job must be skipped entirely: no events at all"
            );

            assert_eq!(
                completion_tokens_of(rx3),
                Some(50),
                "worker must survive cancelling job 2 and serve job 3 normally afterward"
            );

            handle.join().expect("worker thread must not panic");

            assert_eq!(
                started.load(Ordering::SeqCst),
                2,
                "generate() must run exactly twice (job 1, job 3) -- never for cancelled job 2"
            );
            assert_eq!(
                ran_tokens.load(Ordering::SeqCst),
                100,
                "50 real fake-tokens each for job 1 and job 3, zero for cancelled job 2"
            );
        }

        #[test]
        fn running_job_cancelled_midstream_stops_early_and_worker_survives() {
            let (job_tx, job_rx) = mpsc::unbounded_channel::<Job>();
            let started = Arc::new(AtomicUsize::new(0));
            let ran_tokens = Arc::new(AtomicUsize::new(0));

            // Job 1: a long fake generation (2000 tokens, 5ms apart) that we
            // cancel partway through -- it must stop well short of the cap.
            let (job1, mut rx1, guard1) = make_job();
            job_tx.send(job1).unwrap();
            // `Option` so it can be moved-out-and-dropped at most once from
            // inside the loop below; the borrow checker cannot see that the
            // `seen == 5` runtime condition only ever holds on one iteration,
            // so a bare `drop(guard1)` there is rejected as a repeated move.
            let mut guard1 = Some(guard1);

            let (job2, mut rx2, _guard2) = make_job();
            job_tx.send(job2).unwrap();
            drop(job_tx);

            let started2 = started.clone();
            let ran2 = ran_tokens.clone();
            let handle = std::thread::spawn(move || {
                run_worker_loop(job_rx, fake_generate(2000, started2, ran2))
            });

            let mut seen = 0;
            loop {
                match rx1.blocking_recv() {
                    Some(Ev::Delta(_)) => {
                        seen += 1;
                        if seen == 5 {
                            // "Client disconnects" mid-stream.
                            guard1.take();
                        }
                    }
                    Some(Ev::Done {
                        completion_tokens, ..
                    }) => {
                        assert!(
                            completion_tokens < 2000,
                            "job 1 must stop well short of its 2000-token cap after \
                             cancellation, got {completion_tokens}"
                        );
                        assert!(
                            completion_tokens < 100,
                            "job 1 must stop within a handful of tokens of the client \
                             disconnecting, not run on regardless; got {completion_tokens}"
                        );
                        break;
                    }
                    Some(Ev::Failed { message }) => {
                        panic!("fake_generate never fails; unexpected Ev::Failed: {message}")
                    }
                    Some(Ev::Rejected { message }) => {
                        panic!("fake_generate never rejects; unexpected Ev::Rejected: {message}")
                    }
                    None => panic!("job 1's reply channel closed before a Done event"),
                }
            }

            // Job 2 must still complete in full: the worker thread did not
            // panic or wedge when job 1 was cancelled mid-generation.
            let mut n2 = None;
            while let Some(ev) = rx2.blocking_recv() {
                if let Ev::Done {
                    completion_tokens, ..
                } = ev
                {
                    n2 = Some(completion_tokens);
                }
            }
            assert_eq!(
                n2,
                Some(2000),
                "worker must survive mid-stream cancellation and serve the next job to completion"
            );

            handle.join().expect("worker thread must not panic");
        }

        /// PR #606: cancellation was only observed through the
        /// `on_token` callback, so a generator phase that never calls it -- the
        /// real prefill pass has no callback point at all -- could run
        /// unbounded after the client already disconnected. This proves
        /// `run_worker_loop` threads an independent `should_cancel` signal
        /// through to `generate` and that a fake generator honoring only that
        /// signal (never `on_token`) still gets stopped promptly, well short
        /// of its prefill-like phase's natural end.
        #[test]
        fn running_job_cancelled_during_prefill_like_phase_never_calls_on_token() {
            let (job_tx, job_rx) = mpsc::unbounded_channel::<Job>();
            let entered_decode = Arc::new(AtomicBool::new(false));

            let (job1, mut rx1, guard1) = make_job();
            job_tx.send(job1).unwrap();
            drop(job_tx);

            // 400 * 5ms = up to 2s of "prefill" if never cancelled -- the test
            // cancels at 20ms in, ~100x margin, so reaching Done quickly is
            // only possible if should_cancel actually stopped it early.
            let entered2 = entered_decode.clone();
            let handle = std::thread::spawn(move || {
                run_worker_loop(job_rx, fake_generate_with_prefill_gap(400, 50, entered2))
            });

            std::thread::sleep(Duration::from_millis(20));
            drop(guard1);

            match rx1.blocking_recv() {
                Some(Ev::Delta(_)) => panic!(
                    "on_token must never be called: cancellation happened while the \
                     fake generator was still in its prefill-like phase, which does \
                     not call on_token at all"
                ),
                Some(Ev::Done {
                    completion_tokens, ..
                }) => {
                    assert_eq!(
                        completion_tokens, 0,
                        "job cancelled during the prefill-like phase must produce \
                         zero tokens, got {completion_tokens}"
                    );
                }
                Some(Ev::Failed { message }) => panic!(
                    "fake_generate_with_prefill_gap never fails; unexpected Ev::Failed: {message}"
                ),
                Some(Ev::Rejected { message }) => panic!(
                    "fake_generate_with_prefill_gap never rejects; unexpected Ev::Rejected: {message}"
                ),
                None => panic!("job 1's reply channel closed before a Done event"),
            }

            handle.join().expect("worker thread must not panic");

            assert!(
                !entered_decode.load(Ordering::SeqCst),
                "should_cancel alone (on_token is never called during this phase) \
                 must stop the job before the decode phase is ever reached -- this \
                 is the exact blind spot from the PR #606 review, where production \
                 prefill has no on_token callback point and so could run to \
                 completion after the client already disconnected"
            );
        }

        /// A GPU-free fake that fails closed on its first call (mirroring the
        /// #611 contract: `chat_completion_streaming_with_cancel` now returns
        /// `Err` instead of silently sampling when a grammar mask blocks every
        /// candidate token) and succeeds normally on every call after that, so
        /// a single test can prove both halves of the contract: the failure is
        /// reported honestly, and the worker thread survives it.
        #[allow(clippy::type_complexity)]
        fn fake_generate_fails_once_then_succeeds(
            message: &'static str,
            call_count: Arc<AtomicUsize>,
        ) -> impl FnMut(
            &[ChatMessage],
            &GenerateConfig,
            &mut dyn FnMut(&str, u32) -> bool,
            &mut dyn FnMut() -> bool,
        ) -> Result<(usize, usize, bool), String> {
            move |_messages, _cfg, on_token, _should_cancel| {
                if call_count.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Err(message.to_string());
                }
                let _ = on_token("x", 0);
                Ok((1, 1, true))
            }
        }

        /// #611: a generation failure must reach the HTTP layer as
        /// `Ev::Failed` carrying the real error, never as `Ev::Done` with a
        /// fabricated token count -- the latter is exactly the fail-open shape
        /// this issue closes (a blocked grammar silently reported as a normal
        /// zero/short completion instead of an error). The worker thread must
        /// also survive the failure and keep serving subsequent jobs, exactly
        /// as it survives a cancelled job in the tests above.
        #[test]
        fn generation_failure_is_reported_as_ev_failed_not_ev_done() {
            let (job_tx, job_rx) = mpsc::unbounded_channel::<Job>();

            let (job1, mut rx1, _guard1) = make_job();
            job_tx.send(job1).unwrap();
            let (job2, mut rx2, _guard2) = make_job();
            job_tx.send(job2).unwrap();
            drop(job_tx);

            let call_count = Arc::new(AtomicUsize::new(0));
            let handle = std::thread::spawn({
                let call_count = call_count.clone();
                move || {
                    run_worker_loop(
                        job_rx,
                        fake_generate_fails_once_then_succeeds(
                            "grammar constraint blocked every token; no legal \
                             continuation exists in the current grammar state",
                            call_count,
                        ),
                    )
                }
            });

            match rx1.blocking_recv() {
                Some(Ev::Failed { message }) => {
                    assert!(
                        message.contains("grammar constraint blocked every token"),
                        "Ev::Failed must carry the underlying error message, got: {message}"
                    );
                }
                Some(Ev::Done { .. }) => panic!(
                    "a failed generation must never be reported as Ev::Done -- that \
                     would silently hand the HTTP layer a fabricated token count for \
                     a request that produced no legal output, which is the #611 \
                     fail-open failure mode this test guards against"
                ),
                Some(Ev::Delta(_)) => {
                    panic!("a generator that fails on its first call must never emit a Delta first")
                }
                Some(Ev::Rejected { message }) => {
                    panic!("this fake generator never rejects; unexpected Ev::Rejected: {message}")
                }
                None => panic!("job 1's reply channel closed with no event at all"),
            }

            let mut done = None;
            while let Some(ev) = rx2.blocking_recv() {
                if let Ev::Done {
                    completion_tokens, ..
                } = ev
                {
                    done = Some(completion_tokens);
                }
            }
            assert_eq!(
                done,
                Some(1),
                "worker thread must survive a failed generation and serve the next \
                 job normally afterward"
            );

            handle
                .join()
                .expect("worker thread must not panic on a generation error");
        }

        // ── #641 / #649 request parsing and clamp tests ──────────────────

        #[test]
        fn message_content_plain_string_to_chat_message() {
            let msg = InMsg {
                role: "user".to_string(),
                content: MessageContent::Text("hi".to_string()),
            };
            let chat_message = to_chat_message(&msg).expect("plain string content must parse");
            assert_eq!(
                chat_message.role,
                lattice_inference::forward::metal_qwen35::ChatRole::User
            );
            assert_eq!(chat_message.content, "hi");
        }

        #[test]
        fn message_content_parts_concatenate_in_order() {
            let content = MessageContent::Parts(vec![
                Part::Text {
                    text: "a".to_string(),
                },
                Part::Text {
                    text: "b".to_string(),
                },
            ]);
            assert_eq!(content_text(&content).unwrap(), "ab");
        }

        #[test]
        fn message_content_image_url_rejected() {
            let content = MessageContent::Parts(vec![Part::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/cat.png".to_string(),
                    detail: None,
                },
            }]);
            let err = content_text(&content).unwrap_err();
            assert_eq!(err.message(), IMAGE_REQUIRES_VISION_MESSAGE);
        }

        #[test]
        fn message_content_unknown_part_rejected() {
            let content = MessageContent::Parts(vec![Part::Unsupported {
                kind: "file".to_string(),
            }]);
            let err = content_text(&content).unwrap_err();
            assert_eq!(
                err.message(),
                "unsupported content part type 'file'; only 'text' parts are accepted"
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
            // "not a role" and "a role we don't support" (ADR-080 C2
            // round 2, codex finding #1).
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
                messages: vec![],
                temperature: None,
                top_p: None,
                top_k: None,
                max_tokens: None,
                seed: None,
                stream: None,
                repetition_penalty: None,
                reasoning_budget: Some(50),
                max_completion_tokens: None,
                tools: None,
                tool_choice: None,
                response_format: None,
                n: None,
                logprobs: None,
                top_logprobs: None,
                stop: None,
            };
            let cfg = build_cfg(&req, &defaults, 8).unwrap();
            assert!(cfg.max_new_tokens <= 8);
            let reasoning_budget = cfg.reasoning_budget.unwrap_or(0);
            assert!(reasoning_budget + cfg.max_new_tokens < 8);
        }

        // ── #656: prompt-aware KV-window invariant ────────────────────────
        //
        // `build_cfg` alone only clamps `max_new_tokens`/`reasoning_budget`
        // against the window in isolation; `check_prompt_fits_window` is the
        // second half that accounts for `prompt_len`, which is only known
        // once the worker tokenizes the prompt (see `spawn_worker`).

        #[test]
        fn check_prompt_fits_window_rejects_when_prompt_plus_decode_overflows() {
            // model_max_context=8, prompt_len=2, max_tokens=7, reasoning_budget=None:
            // build_cfg clamps max_new_tokens to min(7, 8-1)=7 in isolation, but
            // 2 (prompt) + 7 (decode) + 1 (delimiter) = 10 > 8 -- must reject.
            let defaults = Defaults {
                max_tokens: 7,
                temperature: 0.7,
                top_k: 50,
                top_p: 0.9,
                repetition_penalty: 1.1,
                reasoning_budget: None,
            };
            let req = ChatReq {
                model: None,
                messages: vec![],
                temperature: None,
                top_p: None,
                top_k: None,
                max_tokens: Some(7),
                seed: None,
                stream: None,
                repetition_penalty: None,
                reasoning_budget: None,
                max_completion_tokens: None,
                tools: None,
                tool_choice: None,
                response_format: None,
                n: None,
                logprobs: None,
                top_logprobs: None,
                stop: None,
            };
            let cfg = build_cfg(&req, &defaults, 8).unwrap();
            let err = check_prompt_fits_window(8, 2, &cfg).unwrap_err();
            assert!(
                err.contains("2 tokens") && err.contains("8-token"),
                "error must name the actual prompt length and window: {err}"
            );
        }

        #[test]
        fn check_prompt_fits_window_accepts_exact_boundary_no_needless_truncation() {
            // model_max_context=8, prompt_len=1, max_tokens=6, reasoning_budget=1:
            // build_cfg clamps max_new_tokens to min(6, 7)=6, reasoning_room=8-6-1=1,
            // reasoning_budget=min(1,1)=1. 1 (prompt) + 6 (max_new_tokens) +
            // 1 (reasoning_budget) + 1 (delimiter) = 9 > 8 -- the "+1 delimiter"
            // edge case from the review: still overflows by exactly one slot,
            // so it must ALSO reject, proving the check catches the boundary
            // exactly rather than off-by-one under-rejecting.
            let defaults = Defaults {
                max_tokens: 6,
                temperature: 0.7,
                top_k: 50,
                top_p: 0.9,
                repetition_penalty: 1.1,
                reasoning_budget: None,
            };
            let req = ChatReq {
                model: None,
                messages: vec![],
                temperature: None,
                top_p: None,
                top_k: None,
                max_tokens: Some(6),
                seed: None,
                stream: None,
                repetition_penalty: None,
                reasoning_budget: Some(1),
                max_completion_tokens: None,
                tools: None,
                tool_choice: None,
                response_format: None,
                n: None,
                logprobs: None,
                top_logprobs: None,
                stop: None,
            };
            let cfg = build_cfg(&req, &defaults, 8).unwrap();
            assert!(check_prompt_fits_window(8, 1, &cfg).is_err());

            // Same request, but the window has one more slot of room (9):
            // 1 + 6 + 1 + 1 = 9 <= 9 -- must be ACCEPTED, and `max_new_tokens`
            // must be the full requested 6, not needlessly truncated.
            let cfg2 = build_cfg(&req, &defaults, 9).unwrap();
            assert_eq!(cfg2.max_new_tokens, 6, "must not needlessly truncate");
            assert!(check_prompt_fits_window(9, 1, &cfg2).is_ok());
        }

        #[test]
        fn check_prompt_fits_window_accepts_ordinary_prompt_unclamped() {
            // A near-window but comfortably-fitting prompt must generate its
            // full requested budget -- no needless truncation in the common case.
            let defaults = Defaults {
                max_tokens: 50,
                temperature: 0.7,
                top_k: 50,
                top_p: 0.9,
                repetition_penalty: 1.1,
                reasoning_budget: None,
            };
            let req = ChatReq {
                model: None,
                messages: vec![],
                temperature: None,
                top_p: None,
                top_k: None,
                max_tokens: Some(50),
                seed: None,
                stream: None,
                repetition_penalty: None,
                reasoning_budget: None,
                max_completion_tokens: None,
                tools: None,
                tool_choice: None,
                response_format: None,
                n: None,
                logprobs: None,
                top_logprobs: None,
                stop: None,
            };
            let cfg = build_cfg(&req, &defaults, 4096).unwrap();
            assert_eq!(cfg.max_new_tokens, 50);
            // prompt_len=100: 100 + 50 + 0 + 1 = 151 <= 4096.
            assert!(check_prompt_fits_window(4096, 100, &cfg).is_ok());
        }

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

        // ── HTTP-level 400 tests ──────────────────────────────────────────
        //
        // All three failure modes below (`#641` unknown role, `#649` image
        // part, `#649` oversized part) return from `chat_completions` before
        // a `Job` is ever sent to `s.jobs`, so a fake unbounded sender with no
        // running worker is a faithful stand-in: no GPU, no model load.

        fn test_app_state() -> AppState {
            let (jobs, _rx) = mpsc::unbounded_channel::<Job>();
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
            }
        }

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

        #[tokio::test]
        async fn chat_completions_unknown_role_400() {
            // A role string that is not an OpenAI chat role at all (as
            // opposed to "developer"/"tool", which ARE real OpenAI roles
            // this server just doesn't implement -- see
            // `chat_completions_tool_and_developer_role_400_unsupported_feature`
            // below for that split, ADR-080 C2 round 2 codex finding #1).
            let body =
                Body::from(r#"{"messages":[{"role":"moderator","content":"hi"}]}"#.to_string());
            let response = chat_completions(State(test_app_state()), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                message,
                "unsupported role 'moderator'; must be 'system', 'user', or 'assistant'"
            );
        }

        #[tokio::test]
        async fn chat_completions_tool_and_developer_role_400_unsupported_feature() {
            for role in ["tool", "developer"] {
                let body = Body::from(format!(
                    r#"{{"messages":[{{"role":"{role}","content":"hi"}}]}}"#
                ));
                let response = chat_completions(State(test_app_state()), body).await;
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

        #[tokio::test]
        async fn chat_completions_image_url_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":[
                    {"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}
                ]}]}"#
                    .to_string(),
            );
            let response = chat_completions(State(test_app_state()), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(message, IMAGE_REQUIRES_VISION_MESSAGE);
        }

        #[tokio::test]
        async fn chat_completions_oversized_part_400() {
            let big_text = "x".repeat(MAX_CONTENT_PART_BYTES + 1);
            let body = Body::from(format!(
                r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"{big_text}"}}]}}]}}"#,
            ));
            let response = chat_completions(State(test_app_state()), body).await;
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

        #[tokio::test]
        async fn chat_completions_tools_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],
                    "tools":[{"type":"function","function":{"name":"f"}}]}"#
                    .to_string(),
            );
            let response = chat_completions(State(test_app_state()), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                message,
                "tools and tool_choice are not supported by this server"
            );
        }

        #[tokio::test]
        async fn chat_completions_tool_choice_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"tool_choice":"auto"}"#.to_string(),
            );
            let response = chat_completions(State(test_app_state()), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                message,
                "tools and tool_choice are not supported by this server"
            );
        }

        #[tokio::test]
        async fn chat_completions_json_response_format_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],
                    "response_format":{"type":"json_object"}}"#
                    .to_string(),
            );
            let response = chat_completions(State(test_app_state()), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                message,
                "response_format.type 'json_object' is not supported; use 'text'"
            );
        }

        #[tokio::test]
        async fn chat_completions_n_greater_than_one_400() {
            let body =
                Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"n":2}"#.to_string());
            let response = chat_completions(State(test_app_state()), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(message, "n > 1 is not supported");
        }

        #[tokio::test]
        async fn chat_completions_logprobs_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"logprobs":true}"#.to_string(),
            );
            let response = chat_completions(State(test_app_state()), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                message,
                "logprobs/top_logprobs are not supported by this server"
            );
        }

        #[tokio::test]
        async fn chat_completions_stop_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"stop":"\n"}"#.to_string(),
            );
            let response = chat_completions(State(test_app_state()), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(message, "stop is not supported by this server");
        }

        #[tokio::test]
        async fn chat_completions_conflicting_max_tokens_400() {
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],
                    "max_tokens":10,"max_completion_tokens":20}"#
                    .to_string(),
            );
            let response = chat_completions(State(test_app_state()), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                message,
                "max_tokens (10) and max_completion_tokens (20) differ; supply only one"
            );
        }

        // ── ADR-080 C2 (#782): max_tokens=0 rejection + finish_reason round-trip ──

        #[tokio::test]
        async fn chat_completions_max_tokens_zero_400() {
            // #745: previously clamped straight through into a zero-budget
            // completion instead of being rejected.
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#.to_string(),
            );
            let response = chat_completions(State(test_app_state()), body).await;
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(message, "max_tokens must be at least 1");
        }

        /// Builds an `AppState` wired to a fresh `jobs` channel plus the
        /// receiving half, so tests can stand in for the worker: reply with
        /// whatever `Ev` sequence the test wants without a real GPU/model.
        fn test_app_state_with_jobs() -> (AppState, mpsc::UnboundedReceiver<Job>) {
            let (jobs, jobs_rx) = mpsc::unbounded_channel::<Job>();
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
            };
            (state, jobs_rx)
        }

        /// #746 (ADR-080 C2): the non-streaming JSON response's
        /// `finish_reason` must reflect the engine's actual `stopped` flag,
        /// not a hardcoded `"stop"`. A fake worker task stands in for the
        /// GPU, replying with a crafted `Ev::Done { stopped }` directly.
        async fn non_streaming_finish_reason_for(stopped: bool) -> String {
            let (state, mut jobs_rx) = test_app_state_with_jobs();
            tokio::spawn(async move {
                if let Some(job) = jobs_rx.recv().await {
                    let _ = job.tx.send(Ev::Delta("hi".to_string()));
                    let _ = job.tx.send(Ev::Done {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        stopped,
                    });
                }
            });
            let body = Body::from(r#"{"messages":[{"role":"user","content":"hi"}]}"#.to_string());
            let response = chat_completions(State(state), body).await;
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

        #[tokio::test]
        async fn chat_completions_non_streaming_finish_reason_stop_when_engine_stopped() {
            assert_eq!(non_streaming_finish_reason_for(true).await, "stop");
        }

        #[tokio::test]
        async fn chat_completions_non_streaming_finish_reason_length_when_not_stopped() {
            // Previously this hardcoded "stop" unconditionally, so a
            // length-capped or cancelled completion was misreported as an
            // explicit stop condition (#746).
            assert_eq!(non_streaming_finish_reason_for(false).await, "length");
        }

        /// Same round-trip as above, but through the SSE streaming path
        /// (`Phase::Body`'s `Ev::Done` arm) instead of the non-streaming
        /// JSON body.
        async fn streaming_finish_reason_for(stopped: bool) -> String {
            let (state, mut jobs_rx) = test_app_state_with_jobs();
            tokio::spawn(async move {
                if let Some(job) = jobs_rx.recv().await {
                    let _ = job.tx.send(Ev::Delta("hi".to_string()));
                    let _ = job.tx.send(Ev::Done {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        stopped,
                    });
                }
            });
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#.to_string(),
            );
            let response = chat_completions(State(state), body).await;
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("SSE response body must be readable");
            String::from_utf8(bytes.to_vec()).expect("SSE body must be valid UTF-8")
        }

        #[tokio::test]
        async fn chat_completions_streaming_finish_reason_stop_when_engine_stopped() {
            let text = streaming_finish_reason_for(true).await;
            assert!(
                text.contains("\"finish_reason\":\"stop\""),
                "SSE body must carry finish_reason: stop; got: {text}"
            );
        }

        #[tokio::test]
        async fn chat_completions_streaming_finish_reason_length_when_not_stopped() {
            let text = streaming_finish_reason_for(false).await;
            assert!(
                text.contains("\"finish_reason\":\"length\""),
                "SSE body must carry finish_reason: length, not a hardcoded \"stop\" \
                 (#746); got: {text}"
            );
        }

        /// ADR-080 C2 round 2, codex round-2 major finding #1: a `stream:
        /// true` request whose prompt overflows the model's context window
        /// must return HTTP 400 `context_length_exceeded` BEFORE any SSE
        /// stream is committed -- not silently commit to a 200 SSE response
        /// that only discovers the overflow later via `Ev::Rejected`
        /// mid-stream and terminates with `finish_reason: "length"` (the
        /// exact drift this finding named). This fakes the worker side of
        /// the `Ev::Rejected` contract (production code sends it from
        /// `enforce_prompt_window` inside `run_worker_loop`, prefixed with
        /// `PROMPT_EXCEEDS_WINDOW_PREFIX`) so the composition under test is
        /// purely `chat_completions`'s streaming branch: does it await the
        /// worker's first event and inspect it for `Ev::Rejected` BEFORE
        /// calling `Sse::new(..).into_response()`, or does it commit
        /// unconditionally?
        ///
        /// ADR-080 C2 round 3, codex round-3 medium finding #2: this test
        /// pins ONLY that response-mapping contract with a request that
        /// would NOT genuinely overflow in production (`max_tokens`
        /// defaults to 100 against a 4096-token `model_max_context`) and a
        /// fake task that manufactures `Ev::Rejected` directly -- it does
        /// NOT exercise the real worker's `enforce_prompt_window` call, so
        /// it is not, by itself, same-input real-router parity with
        /// `lattice.rs`'s equivalent test (an earlier version of this
        /// comment incorrectly claimed "identical request body"; it is not
        /// -- `real_router_overflow_parity` below is the test that actually
        /// is). Kept because it is still the cheapest, fastest pin of the
        /// response-mapping contract in isolation.
        ///
        /// Mutation-sensitive: reverting the pre-`Sse::new()` `first_ev`
        /// preflight (going back to building `stream::unfold` and
        /// returning `Sse::new(stream).into_response()` unconditionally,
        /// discovering `Ev::Rejected` only inside `Phase::Body`) makes this
        /// fail -- the response status would be 200 with an SSE body
        /// carrying a `finish_reason: "length"` terminal chunk instead of a
        /// 400 error envelope. Verified by reverting the preflight and
        /// re-running: see the PR body's mutation log.
        #[tokio::test]
        async fn chat_completions_streaming_context_overflow_returns_400_before_committing_sse() {
            let (state, mut jobs_rx) = test_app_state_with_jobs();
            tokio::spawn(async move {
                if let Some(job) = jobs_rx.recv().await {
                    let _ = job.tx.send(Ev::Rejected {
                        message: "prompt has 4090 tokens, leaving 6 of the 4096-token \
                                  context window for generation, but this request needs \
                                  100 generated tokens plus 1 (total 4191); reduce \
                                  max_tokens/reasoning_budget or shorten the prompt"
                            .to_string(),
                    });
                }
            });
            let body = Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#.to_string(),
            );
            let response = chat_completions(State(state), body).await;
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

        /// ADR-080 C2 round 3 (codex round-3 medium finding #2): the
        /// same-input real-router parity the test above does NOT provide.
        /// Drives `lattice_inference::serve::OVERFLOW_PARITY_REQUEST_BODY`
        /// -- the SAME fixture `lattice.rs`'s `streaming_context_overflow`
        /// module drives through its own real `Router` -- through THIS
        /// binary's real `router()`, backed by a REAL worker thread running
        /// the actual `run_worker_loop` + `enforce_prompt_window` production
        /// code (a real `BpeTokenizer` from
        /// `test_support::tiny_zero_model`, no Metal engine involved since
        /// the request is expected to be rejected before any generation
        /// call). `AppState.model_max_context` is set to
        /// `OVERFLOW_PARITY_CONTEXT_WINDOW` (1024) -- the same effective
        /// limit `lattice.rs`'s tiny test model's fixed context window uses
        /// -- so both sides genuinely share the same input AND the same
        /// effective context limit, not just the same JSON body.
        ///
        /// Mutation-sensitive: removing `enforce_prompt_window`'s call to
        /// `check_prompt_fits_window` (i.e. having it unconditionally
        /// return `Ok(prompt_len)`) makes this fail -- the real worker
        /// would accept the request, `chat_completions` would commit a 200
        /// SSE response, and this test's `StatusCode::BAD_REQUEST`
        /// assertion would fail. This is the exact production worker-side
        /// check codex named ("removing the production worker closure's
        /// check_prompt_fits_window call ... would leave both the direct
        /// handler test and the pure helper test green") -- `spawn_worker`'s
        /// real Metal closure and this test's worker seam both call
        /// `enforce_prompt_window`, not two independent copies of the
        /// check, so a mutation to the shared function is observed here too.
        ///
        /// Gated behind `test-utils` (see
        /// `lattice_inference::model::qwen35::test_support`) for the same
        /// reason as `lattice.rs`'s equivalent test modules: this needs a
        /// real (tiny) `BpeTokenizer`, which is only constructible outside
        /// this crate's own `#[cfg(test)]` build via that feature.
        #[cfg(feature = "test-utils")]
        mod real_router_overflow_parity {
            use super::*;
            use lattice_inference::serve::{
                OVERFLOW_PARITY_CONTEXT_WINDOW, OVERFLOW_PARITY_REQUEST_BODY,
            };
            use tower::ServiceExt as _;

            /// A real worker thread running the actual `run_worker_loop` +
            /// `enforce_prompt_window` production code, with a real (tiny)
            /// tokenizer and NO Metal engine -- the generate closure is
            /// never reached by an overflowing request, so it only needs a
            /// trivial success stand-in for a non-overflowing one.
            fn real_worker_state(model_max_context: usize) -> AppState {
                let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                    .tokenizer()
                    .clone();
                let (jobs, jobs_rx) = mpsc::unbounded_channel::<Job>();
                std::thread::spawn(move || {
                    run_worker_loop(jobs_rx, move |messages, cfg, _on_token, _should_cancel| {
                        let prompt_len =
                            enforce_prompt_window(&tokenizer, model_max_context, messages, cfg)?;
                        Ok((prompt_len, 0, true))
                    });
                });
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
                messages: vec![],
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
            let cfg = build_cfg(&req, &defaults, 4096).unwrap();
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

        #[tokio::test]
        async fn cm_lattice_serve_model_mismatch_accepted_expected_deviation() {
            // Expected-deviation fixture, not a bug fix: `lattice serve` (the
            // CLI's HTTP subcommand) rejects a request whose `model` field
            // doesn't match the served model id with HTTP 400
            // (`lattice.rs`'s `model_not_found` check, exercised by
            // `cm_serve_model_mismatch_rejected` in that binary's own test
            // module). This surface has no equivalent check at all: `req.model`
            // is accepted as-is, and the request is queued for generation like
            // any other. This fixture pins that CURRENT, documented divergence
            // -- tracked in #663 -- by proving a mismatched `model` reaches the
            // job queue rather than being rejected up front. If #663 ships a
            // `model_not_found` check here, this fixture starts failing
            // (`jobs_rx` never receives a `Job`), which is the intended signal
            // to update it rather than a silent pass either way.
            let (jobs, mut jobs_rx) = mpsc::unbounded_channel::<Job>();
            let state = AppState {
                jobs,
                model_id: Arc::from("served-model"),
                defaults: Defaults {
                    max_tokens: 100,
                    temperature: 0.7,
                    top_k: 50,
                    top_p: 0.9,
                    repetition_penalty: 1.1,
                    reasoning_budget: None,
                },
                model_max_context: 4096,
            };
            let body = Body::from(
                r#"{"model":"some-other-model","messages":[{"role":"user","content":"hi"}]}"#
                    .to_string(),
            );
            // The handler awaits a reply from a worker that doesn't exist in
            // this test, so it never resolves; run it in the background and
            // only wait on the job queue it should have fed synchronously
            // before that await point.
            tokio::spawn(async move {
                let _ = chat_completions(State(state), body).await;
            });
            let queued = tokio::time::timeout(Duration::from_millis(500), jobs_rx.recv()).await;
            assert!(
                matches!(queued, Ok(Some(_))),
                "a mismatched `model` must reach the job queue on this surface today \
                 (expected deviation, tracked in #663)"
            );
        }

        // ── cross-binary parity table (ADR-080 C2 round 2, codex round-1
        // finding #1) ────────────────────────────────────────────────────
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
        // `enforce_prompt_window` for real, so this whole module now
        // requires `test-utils`, matching `lattice.rs`'s own
        // `parity_table` module gating. CI's "serve-surface capability-
        // matrix fixtures" step adds `test-utils` alongside `f16,metal-gpu`
        // for this binary to keep running it (see `.github/workflows/ci.yml`).
        #[cfg(feature = "test-utils")]
        mod parity_table {
            use super::*;
            use lattice_inference::serve::{
                BASELINE_CANNED_COMPLETION_TOKENS, BASELINE_CANNED_PROMPT_TOKENS, Binary,
                CHAT_COMPLETIONS_PARITY_CASES, ExpectedResponse, check_sse_events,
            };
            use tower::ServiceExt as _;

            /// Deterministic worker-thread generation seam for every
            /// `Json`/`Sse` row (issue #828): a REAL background thread runs
            /// the actual `run_worker_loop` + `enforce_prompt_window`
            /// production code (real tiny tokenizer, no Metal engine) --
            /// only the terminal generation call itself is replaced with a
            /// canned deterministic completion. Content deltas are pushed
            /// through `on_token` (what both the streaming and
            /// non-streaming arms of `chat_completions` read content from
            /// on this binary -- see its `Ev::Delta` accumulation), so one
            /// seam serves both response shapes.
            fn baseline_fake_worker_state(model_max_context: usize) -> AppState {
                let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                    .tokenizer()
                    .clone();
                let (jobs, jobs_rx) = mpsc::unbounded_channel::<Job>();
                std::thread::spawn(move || {
                    run_worker_loop(jobs_rx, move |messages, cfg, on_token, _should_cancel| {
                        // Real production check (mutation-sensitive to
                        // `enforce_prompt_window`/`check_prompt_fits_window`
                        // exactly like `real_router_overflow_parity` above);
                        // its measured length is intentionally NOT what
                        // gets reported below -- the returned usage counts
                        // are the fixed canned figures both binaries' field
                        // checks assert against (`BASELINE_CANNED_*`).
                        enforce_prompt_window(&tokenizer, model_max_context, messages, cfg)?;
                        for (chunk, id) in [("hello", 1u32), (" world", 2u32)] {
                            on_token(chunk, id);
                        }
                        Ok((
                            BASELINE_CANNED_PROMPT_TOKENS as usize,
                            BASELINE_CANNED_COMPLETION_TOKENS as usize,
                            true,
                        ))
                    });
                });
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
        #[cfg(feature = "test-utils")]
        mod production_adapter_observation {
            use super::*;
            use lattice_inference::serve::{
                ExpectedObservation, GenerateConfigSnapshot,
                OBSERVATION_GOLDEN_USER_HI_THERE_CHATML, ProductionAdapterObservation,
                assert_observation_matches,
            };
            use std::sync::Mutex;

            /// Mirrors `lattice.rs`'s equivalent helper (issue #828 round 2):
            /// fires the fixed `{"messages":[{"role":"user","content":"hi
            /// there"}],"temperature":1.3,"top_p":0.55,"seed":7,"max_tokens":9}`
            /// request through a REAL background thread running the actual
            /// `run_worker_loop` + `enforce_prompt_window` production code (real
            /// tiny tokenizer, no Metal engine) -- only the terminal generation
            /// call is replaced with a canned completion. `stopped` is threaded
            /// through a single local variable into both the recorded
            /// observation and the returned tuple's third element, and the
            /// real `enforce_prompt_window` return value is the ONLY source for
            /// `prompt_tokens` -- round 2 major finding fixed the prior
            /// `prompt_tokens: 3` / `stopped: true` independent literals.
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
                let (jobs, jobs_rx) = mpsc::unbounded_channel::<Job>();
                std::thread::spawn(move || {
                    run_worker_loop(jobs_rx, move |messages, cfg, on_token, _should_cancel| {
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
                        // `enforce_prompt_window`/`check_prompt_fits_window`
                        // exactly like `real_router_overflow_parity`/
                        // `baseline_fake_worker_state` above); its measured
                        // length IS what gets reported below.
                        let prompt_tokens =
                            enforce_prompt_window(&tokenizer, model_max_context, messages, cfg)?;
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
                        Ok((prompt_tokens, 1, stopped))
                    });
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
                    model_max_context,
                };
                let body = Body::from(
                    r#"{"messages":[{"role":"user","content":"hi there"}],"temperature":1.3,"top_p":0.55,"seed":7,"max_tokens":9}"#
                        .to_string(),
                );
                let response = chat_completions(State(state), body).await;
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
            /// literal (round 2 major finding: this was previously
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

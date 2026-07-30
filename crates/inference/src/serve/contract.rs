//! Shared OpenAI chat-completions request parsing and normalization.

use serde::Deserialize;
use serde_json::Value;
use serde_json::value::RawValue;

use crate::forward::metal_qwen35::ChatMessage;

use super::{ApiError, REQUEST_BODY_LIMIT_BYTES};

/// Maximum number of messages accepted in a single chat request, enforced
/// before per-message engine allocation (`normalize_messages`), chat-template
/// formatting, or tokenization runs. [`REQUEST_BODY_LIMIT_BYTES`] bounds the
/// wire size of a request but not the CPU/memory cost of processing it:
/// a request built from many tiny (even empty) messages pays the per-message
/// cost — a `ChatMessage` allocation, a chat-template formatting pass, and a
/// share of tokenization — `messages.len()` times regardless of how little
/// content each message carries. 4096 is generous headroom over any real
/// chat history (a long-running multi-turn agent conversation runs to low
/// hundreds of turns) while closing that amplification.
pub const MAX_MESSAGE_COUNT: usize = 4096;

/// Maximum summed byte length of all message content in a single chat
/// request, enforced at the same point as [`MAX_MESSAGE_COUNT`]. Content is
/// a strict subset of the request body, so this can never exceed
/// [`REQUEST_BODY_LIMIT_BYTES`] in practice — the shared contract still
/// states it as its own explicit invariant so the bound holds independent of
/// the current body-size cap and of which content encoding (a plain string
/// vs. typed content parts) a client uses.
pub const MAX_CUMULATIVE_CONTENT_BYTES: usize = REQUEST_BODY_LIMIT_BYTES;

/// Maximum byte length of a single stop string, enforced in
/// [`parse_stop_strings`] before the string is stored in a
/// `StopStringMatcher`. `StopStringMatcher` rescans a `max_stop`-byte window
/// of the accumulated decode text on every generated token
/// (`crates/inference/src/model/qwen35/stop_strings.rs`), where `max_stop` is
/// the longest configured stop string — so an unbounded stop string turns
/// every decode step for the rest of generation into a scan proportional to
/// that string's length, and the matcher also retains that many bytes of
/// held-back output. 4096 bytes is generous headroom over any real stop
/// delimiter (a handful of characters up to a full chat-template turn
/// marker) while closing that amplification.
pub const MAX_STOP_STRING_BYTES: usize = 4096;

/// Maximum summed byte length across all stop strings in one request,
/// enforced at the same point as [`MAX_STOP_STRING_BYTES`]. Deliberately set
/// below `4 * MAX_STOP_STRING_BYTES` (the max the 4-element array-length cap
/// combined with the per-string cap could otherwise reach,
/// [`parse_stop_strings`]) so this is a genuine, independently reachable
/// bound rather than one implied away by the other two caps.
pub const MAX_CUMULATIVE_STOP_BYTES: usize = 2 * MAX_STOP_STRING_BYTES;

/// Superset of the chat-completions request fields accepted by either HTTP server.
///
/// Top-level fields not modeled here (e.g. `presence_penalty`, `frequency_penalty`,
/// `logit_bias`, `user`) are ignored rather than rejected, matching standard
/// OpenAI-compatible client behavior on both serving endpoints.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// Requested model identifier. `None` when the field is omitted; `Some("")`
    /// when the client sends an explicit empty string — these are validated
    /// differently, so the distinction must survive deserialization.
    #[serde(default)]
    pub model: Option<String>,
    /// Conversation messages. Deserialization itself enforces
    /// [`MAX_MESSAGE_COUNT`] (see [`deserialize_bounded_messages`]), so the
    /// single authoritative `ChatRequest` parse both counts and materializes
    /// the array in one traversal.
    #[serde(default, deserialize_with = "deserialize_bounded_messages")]
    pub messages: Vec<Message>,
    /// Legacy generation-token budget.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Current OpenAI alias for `max_tokens`.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability mass.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k sampling limit. Kept as a raw, unparsed [`RawValue`] rather
    /// than `Option<usize>` or `Option<Value>`: on profiles that don't
    /// honor this field (it is accepted-and-ignored) a type-mismatched
    /// value must not hard-fail deserialization of the whole request, and
    /// `RawValue` -- unlike `Value` -- does not materialize a parsed tree
    /// at all, so an ignored field submitted as a huge nested payload costs
    /// nothing but the byte span it already occupies in `body`.
    /// [`normalize_request`] parses it strictly only on profiles that
    /// actually use it.
    #[serde(default)]
    pub top_k: Option<Box<RawValue>>,
    /// Repetition penalty. See `top_k` for why this is a raw [`RawValue`].
    #[serde(default)]
    pub repetition_penalty: Option<Box<RawValue>>,
    /// Deterministic sampling seed.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Whether to stream the response as SSE.
    #[serde(default)]
    pub stream: Option<bool>,
    /// String-level stop condition in OpenAI string-or-array form.
    #[serde(default)]
    pub stop: Option<Value>,
    /// Daemon reasoning-token budget extension. See `top_k` for why this is
    /// a raw [`RawValue`].
    #[serde(default)]
    pub reasoning_budget: Option<Box<RawValue>>,
    /// Response-format constraint.
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// Tool definitions.
    #[serde(default)]
    pub tools: Option<Value>,
    /// Tool-selection policy.
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// Whether to return sampled-token log probabilities.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of alternative-token log probabilities to return.
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    /// Number of completions to generate.
    #[serde(default)]
    pub n: Option<usize>,
}

/// One input chat message.
#[derive(Debug, Deserialize)]
pub struct Message {
    /// OpenAI message role.
    pub role: String,
    /// Plain text or typed content parts.
    pub content: MessageContent,
}

/// Message content represented as a string or a list of typed parts.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// OpenAI typed content parts.
    Parts(Vec<ContentPart>),
}

/// One typed OpenAI message-content part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentPart {
    /// A text content part.
    Text { text: String },
    /// An image URL content part.
    ImageUrl { image_url: ImageUrl },
    /// A recognized wire part whose type is not supported by the engine.
    Unsupported { kind: String },
}

/// Image URL payload carried by an image content part.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ImageUrl {
    /// Image URL or data URL.
    pub url: String,
    /// Optional OpenAI image-detail hint.
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Deserialize)]
struct RawContentPart {
    #[serde(rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    image_url: Option<ImageUrl>,
}

impl<'de> Deserialize<'de> for ContentPart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawContentPart::deserialize(deserializer)?;
        match raw.kind.as_deref() {
            Some("text") => raw.text.map(|text| Self::Text { text }).ok_or_else(|| {
                serde::de::Error::custom("text content part must include string field 'text'")
            }),
            Some("image_url") => raw
                .image_url
                .map(|image_url| Self::ImageUrl { image_url })
                .ok_or_else(|| {
                    serde::de::Error::custom(
                        "image_url content part must include object field 'image_url'",
                    )
                }),
            Some(kind) => Ok(Self::Unsupported {
                kind: kind.to_string(),
            }),
            None => Ok(Self::Unsupported {
                kind: "<missing>".to_string(),
            }),
        }
    }
}

/// OpenAI response-format request.
#[derive(Debug, Deserialize)]
pub struct ResponseFormat {
    /// Requested response format name.
    pub r#type: String,
    /// Strict JSON Schema definition used by the daemon profile.
    #[serde(default)]
    pub json_schema: Option<JsonSchemaFormat>,
}

/// Strict JSON Schema response-format payload.
#[derive(Debug, Deserialize)]
pub struct JsonSchemaFormat {
    /// Client-visible schema name.
    #[serde(default)]
    pub name: Option<String>,
    /// Whether strict schema enforcement is requested.
    #[serde(default)]
    pub strict: Option<bool>,
    /// JSON Schema document, kept as a raw, unparsed [`RawValue`] rather
    /// than `Value`: the `lattice` profile rejects `json_schema` outright
    /// (`structured_output_supported` is `false`), so materializing a full
    /// `Value` tree for this field during the outer `ChatRequest`
    /// deserialization would pay parse/allocation cost for every request on
    /// that profile before `reject_unsupported` ever runs -- a client can
    /// submit an arbitrarily large schema to force that cost repeatedly.
    /// The daemon profile, which does support `json_schema`, parses this
    /// span into a `Value` itself once admission has cheaply passed the
    /// type/name/strict checks (`admit_structured_request`).
    #[serde(default)]
    pub schema: Option<Box<RawValue>>,
}

/// Server sampling defaults used when a request omits a generation field.
#[derive(Debug, Clone, Copy)]
pub struct GenerationDefaults {
    /// Default generation-token budget.
    pub max_tokens: usize,
    /// Default temperature.
    pub temperature: f32,
    /// Default top-k limit.
    pub top_k: usize,
    /// Default nucleus-sampling mass.
    pub top_p: f32,
    /// Default repetition penalty.
    pub repetition_penalty: f32,
    /// Default reasoning budget.
    pub reasoning_budget: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
enum ModelNamePolicy<'a> {
    RequiredExact(&'a str),
    OptionalExact(&'a str),
}

#[derive(Debug, Clone, Copy)]
enum MaxTokensPolicy {
    RejectAbove { limit: usize },
    ClampToContext { context: usize },
}

/// Request-policy differences between the two server binaries.
#[derive(Debug, Clone, Copy)]
pub struct ServeProfile<'a> {
    model_name: ModelNamePolicy<'a>,
    max_tokens: MaxTokensPolicy,
    require_last_user: bool,
    stop_supported: bool,
    sampling_extensions_supported: bool,
    reasoning_budget_supported: bool,
    structured_output_supported: bool,
    logprobs_supported: bool,
    /// Whether a conflicting `max_tokens`/`max_completion_tokens` pair is
    /// rejected ahead of the served-model check (matching each binary's
    /// pre-shared-contract precedence for this one check): `true` on the
    /// daemon profile, `false` on the `lattice` profile, where it is
    /// checked after the model check instead (see [`normalize_max_tokens`]).
    max_tokens_conflict_checked_early: bool,
}

impl<'a> ServeProfile<'a> {
    /// Profile for the unified `lattice serve` route.
    pub fn lattice(model_id: &'a str, max_tokens_cap: usize) -> Self {
        Self {
            model_name: ModelNamePolicy::RequiredExact(model_id),
            max_tokens: MaxTokensPolicy::RejectAbove {
                limit: max_tokens_cap,
            },
            require_last_user: true,
            stop_supported: true,
            sampling_extensions_supported: false,
            reasoning_budget_supported: false,
            structured_output_supported: false,
            logprobs_supported: true,
            max_tokens_conflict_checked_early: false,
        }
    }

    /// Profile for the standalone `lattice_serve` daemon route.
    pub fn lattice_serve(model_id: &'a str, model_max_context: usize) -> Self {
        Self {
            model_name: ModelNamePolicy::OptionalExact(model_id),
            max_tokens: MaxTokensPolicy::ClampToContext {
                context: model_max_context,
            },
            require_last_user: false,
            stop_supported: false,
            sampling_extensions_supported: true,
            reasoning_budget_supported: true,
            structured_output_supported: true,
            logprobs_supported: false,
            max_tokens_conflict_checked_early: true,
        }
    }
}

/// Fully normalized request fields consumed by generation adapters.
#[derive(Debug)]
pub struct ValidatedChatRequest {
    /// Validated engine chat messages.
    pub messages: Vec<ChatMessage>,
    /// Effective generation-token budget after profile policy.
    pub max_tokens: usize,
    /// Validated temperature in `[0.0, 2.0]`.
    pub temperature: f32,
    /// Effective top-k sampling limit.
    pub top_k: usize,
    /// Validated top-p mass in `(0.0, 1.0]`.
    pub top_p: f32,
    /// Effective repetition penalty.
    pub repetition_penalty: f32,
    /// Deterministic sampling seed.
    pub seed: Option<u64>,
    /// Whether to stream the response.
    pub stream: bool,
    /// Parsed string-level stop conditions.
    pub stop_strings: Vec<String>,
    /// Effective reasoning-token budget.
    pub reasoning_budget: Option<usize>,
    /// Number of alternative log probabilities to capture when enabled.
    pub logprobs: Option<usize>,
}

/// Normalize one shared wire request according to a named server profile.
///
/// `check_context` runs after messages and sampling scalars are validated but
/// before stop strings are parsed. The unified server uses it for its prompt
/// plus decode-window preflight; the daemon supplies a no-op because its
/// worker performs the prompt-aware check after tokenization.
pub fn normalize_request<C>(
    req: &ChatRequest,
    defaults: GenerationDefaults,
    profile: ServeProfile<'_>,
    check_context: impl FnOnce(&[ChatMessage], usize) -> Result<C, ApiError>,
) -> Result<(ValidatedChatRequest, C), ApiError> {
    reject_unsupported(req, profile)?;
    validate_model_name(req.model.as_deref(), profile.model_name)?;

    if req.messages.is_empty() {
        return Err(ApiError::BadRequest {
            message: "messages must not be empty".to_string(),
            code: "invalid_messages",
        });
    }
    check_message_bounds(&req.messages)?;
    if profile.require_last_user
        && req.messages.last().map(|message| message.role.as_str()) != Some("user")
    {
        return Err(ApiError::BadRequest {
            message: "the last message must have role 'user'".to_string(),
            code: "invalid_messages",
        });
    }

    let max_tokens = normalize_max_tokens(
        req,
        defaults.max_tokens,
        profile.max_tokens,
        profile.max_tokens_conflict_checked_early,
    )?;
    let temperature = validate_temperature(req.temperature.unwrap_or(defaults.temperature))?;
    let top_p = validate_top_p(req.top_p.unwrap_or(defaults.top_p))?;
    let logprobs = normalize_logprobs(req)?;
    let messages = normalize_messages(&req.messages)?;
    let context = check_context(&messages, max_tokens)?;
    let stop_strings = if profile.stop_supported {
        parse_stop_strings(&req.stop)?
    } else {
        Vec::new()
    };

    let top_k = if profile.sampling_extensions_supported {
        parse_ignorable_field::<usize>(&req.top_k, "top_k")?.unwrap_or(defaults.top_k)
    } else {
        defaults.top_k
    };
    let repetition_penalty = if profile.sampling_extensions_supported {
        parse_ignorable_field::<f32>(&req.repetition_penalty, "repetition_penalty")?
            .unwrap_or(defaults.repetition_penalty)
    } else {
        defaults.repetition_penalty
    };
    let mut reasoning_budget = if profile.reasoning_budget_supported {
        parse_ignorable_field::<usize>(&req.reasoning_budget, "reasoning_budget")?
            .filter(|&value| value > 0)
            .or(defaults.reasoning_budget)
    } else {
        None
    };
    if let MaxTokensPolicy::ClampToContext { context } = profile.max_tokens {
        let reasoning_room = context.saturating_sub(max_tokens).saturating_sub(1);
        reasoning_budget = reasoning_budget
            .map(|value| value.min(reasoning_room))
            .filter(|&value| value > 0);
    }

    Ok((
        ValidatedChatRequest {
            messages,
            max_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed: req.seed,
            stream: req.stream.unwrap_or(false),
            stop_strings,
            reasoning_budget,
            logprobs,
        },
        context,
    ))
}

/// Hard capability/shape checks that both pre-shared-contract binaries ran
/// *before* the served-model check, so a request combining a wrong model
/// with one of these violations gets the more specific rejection rather
/// than `model_not_found`. This grouping restores that precedence (B3):
/// pre-refactor, `lattice.rs`'s own `reject_unsupported` rejected
/// `stream: true` + `logprobs: true` here, and the standalone daemon's own
/// `reject_unsupported` rejected `logprobs`/`top_logprobs` outright (it
/// never supported them), ahead of the model-id match. The shared
/// `normalize_request` cascade had moved those two checks into
/// `normalize_logprobs`, which runs after `validate_model_name` -- silently
/// reordering client-observable first-error precedence. `top_logprobs`-
/// without-`logprobs` and `top_logprobs > 20` stay in `normalize_logprobs`,
/// matching both binaries' original post-model-check position for those
/// two. The conflicting `max_tokens`/`max_completion_tokens` check is
/// profile-gated (`ServeProfile::max_tokens_conflict_checked_early`)
/// instead of unconditionally grouped here: the daemon profile checked it
/// ahead of the model-id match pre-refactor, but the `lattice` profile
/// checked it after (inside its `validate_max_tokens`), so it runs from
/// [`normalize_max_tokens`] on that profile instead.
fn reject_unsupported(req: &ChatRequest, profile: ServeProfile<'_>) -> Result<(), ApiError> {
    if req.tools.is_some() || req.tool_choice.is_some() {
        return unsupported("tools and tool_choice are not supported by this server");
    }
    if req.n.unwrap_or(1) > 1 {
        return unsupported("n > 1 is not supported");
    }
    if req.stop.is_some() && !profile.stop_supported {
        return unsupported("stop is not supported by this server");
    }
    if !profile.logprobs_supported && (req.logprobs.unwrap_or(false) || req.top_logprobs.is_some())
    {
        return unsupported("logprobs/top_logprobs are not supported by this server");
    }
    if req.stream == Some(true) && req.logprobs.unwrap_or(false) {
        return unsupported("logprobs is not supported together with stream: true");
    }
    if profile.max_tokens_conflict_checked_early {
        reject_conflicting_max_tokens(req)?;
    }
    // top_k, repetition_penalty, and reasoning_budget are accepted-and-ignored
    // (not rejected) on profiles that don't support them, matching each
    // server's pre-shared-contract tolerance for these fields.
    if let Some(format) = &req.response_format
        && format.r#type != "text"
        && !(profile.structured_output_supported && format.r#type == "json_schema")
    {
        return unsupported(format!(
            "response_format.type '{}' is not supported; use 'text'",
            format.r#type
        ));
    }
    Ok(())
}

/// Rejects a present-but-disagreeing `max_tokens`/`max_completion_tokens`
/// pair. Called from [`reject_unsupported`] (ahead of the model check) or
/// [`normalize_max_tokens`] (after it), depending on
/// `ServeProfile::max_tokens_conflict_checked_early`.
fn reject_conflicting_max_tokens(req: &ChatRequest) -> Result<(), ApiError> {
    if let (Some(max_tokens), Some(max_completion_tokens)) =
        (req.max_tokens, req.max_completion_tokens)
        && max_tokens != max_completion_tokens
    {
        return Err(ApiError::BadRequest {
            message: format!(
                "max_tokens ({max_tokens}) and max_completion_tokens ({max_completion_tokens}) differ; supply only one"
            ),
            code: "invalid_request",
        });
    }
    Ok(())
}

/// Sentinel embedded in [`deserialize_bounded_messages`]'s error so callers
/// can distinguish "the array exceeded [`MAX_MESSAGE_COUNT`]" from any
/// other deserialization failure and surface the specific over-limit
/// message instead of a generic invalid-body error.
const MESSAGE_FLOOD_SENTINEL: &str = "lattice_message_flood_exceeded";

/// Deserializes `messages` while bailing the instant its element count
/// exceeds [`MAX_MESSAGE_COUNT`], instead of materializing an unbounded
/// `Vec<Message>` and checking its length afterward. The bail is
/// structural: `visit_seq` returns `Err` from inside its `next_element`
/// loop on the (`MAX_MESSAGE_COUNT` + 1)-th element, so an over-limit array
/// of any size N costs exactly `MAX_MESSAGE_COUNT` + 1 element-visits,
/// never N. This runs inline as part of the single authoritative
/// `ChatRequest` deserialization -- there is no separate raw-bytes preflight
/// pass over the same array.
fn deserialize_bounded_messages<'de, D>(deserializer: D) -> Result<Vec<Message>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Visitor;
    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = Vec<Message>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("an array of messages")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut messages = Vec::new();
            while let Some(message) = seq.next_element::<Message>()? {
                messages.push(message);
                if messages.len() > MAX_MESSAGE_COUNT {
                    return Err(serde::de::Error::custom(MESSAGE_FLOOD_SENTINEL));
                }
            }
            Ok(messages)
        }
    }
    deserializer.deserialize_seq(Visitor)
}

/// True when a `ChatRequest` deserialization failure was raised by
/// [`deserialize_bounded_messages`]'s count bound, so callers can surface
/// [`message_flood_text`] instead of a generic invalid-JSON-body error.
pub fn is_message_flood_error(err: &serde_json::Error) -> bool {
    err.to_string().contains(MESSAGE_FLOOD_SENTINEL)
}

/// Client-facing text for a message-flood rejection, shared so both server
/// binaries surface identical wording.
pub fn message_flood_text() -> String {
    format!("messages has more than {MAX_MESSAGE_COUNT} entries; maximum is {MAX_MESSAGE_COUNT}")
}

/// Reject requests whose message count or cumulative content size exceeds
/// [`MAX_MESSAGE_COUNT`] / [`MAX_CUMULATIVE_CONTENT_BYTES`], before any
/// per-message engine allocation, chat-template formatting, or tokenization
/// runs: a sub-body-cap request built from tens of thousands of tiny
/// messages must not reach those expensive stages before being rejected.
/// The message-count bound is already enforced during `ChatRequest`
/// deserialization ([`deserialize_bounded_messages`]) for both server
/// binaries, which build every `ChatRequest` that way; the check here
/// stays as defense-in-depth for any caller that constructs `messages`
/// directly. The cumulative-content-bytes bound has no earlier check.
fn check_message_bounds(messages: &[Message]) -> Result<(), ApiError> {
    if messages.len() > MAX_MESSAGE_COUNT {
        return Err(ApiError::BadRequest {
            message: format!(
                "messages has {} entries; maximum is {MAX_MESSAGE_COUNT}",
                messages.len()
            ),
            code: "invalid_request_body",
        });
    }
    let cumulative: usize = messages
        .iter()
        .map(|message| content_byte_len(&message.content))
        .sum();
    if cumulative > MAX_CUMULATIVE_CONTENT_BYTES {
        return Err(ApiError::BadRequest {
            message: format!(
                "messages content totals {cumulative} bytes; maximum is {MAX_CUMULATIVE_CONTENT_BYTES}"
            ),
            code: "invalid_request_body",
        });
    }
    Ok(())
}

fn content_byte_len(content: &MessageContent) -> usize {
    match content {
        MessageContent::Text(text) => text.len(),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => text.len(),
                ContentPart::ImageUrl { image_url } => image_url.url.len(),
                ContentPart::Unsupported { kind } => kind.len(),
            })
            .sum(),
    }
}

/// Parse a raw, unparsed [`RawValue`]-typed field that is only honored on
/// profiles that support it, and accepted-and-ignored otherwise. Called
/// only when the caller's profile honors the field, so a type-mismatched
/// value here (e.g. a string where a number is expected) is a genuine
/// client error, not a value this server would ignore anyway.
///
/// Never materializes a `Value` tree: `value` is the exact, unparsed byte
/// span serde captured for the field during the outer `ChatRequest`
/// deserialization (`Deserialize` for `RawValue` never recurses into the
/// span -- it just records where it starts and ends), so a malformed
/// extension field submitted as a near-body-sized nested array/object costs
/// nothing beyond that capture. Non-scalar values are rejected by
/// inspecting only the first non-whitespace byte of the span (`{` or `[`)
/// before any parse is attempted; scalars (string/number/bool/null) are
/// cheap by construction, so only they reach `from_str`.
///
/// An explicit JSON `null` is treated as an absent field (`None`), matching
/// what an omitted field would do -- serde's `Option<T>` handling already
/// maps a `null` field value to `None` at the outer `ChatRequest` level
/// before this function ever runs, so `value` is `None` here for a
/// `"field": null` request in practice; the `"null"` span check below is a
/// second line of defense in case `value` is ever produced some other way.
fn parse_ignorable_field<T: serde::de::DeserializeOwned>(
    value: &Option<Box<RawValue>>,
    field_name: &str,
) -> Result<Option<T>, ApiError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let text = value.get().trim();
    if text == "null" {
        return Ok(None);
    }
    if text.starts_with('{') || text.starts_with('[') {
        return Err(ApiError::BadRequest {
            message: format!("{field_name} must be a scalar value"),
            code: "invalid_request_body",
        });
    }
    serde_json::from_str(text)
        .map(Some)
        .map_err(|err| ApiError::BadRequest {
            message: format!("{field_name} is invalid: {err}"),
            code: "invalid_request_body",
        })
}

fn unsupported(message: impl Into<String>) -> Result<(), ApiError> {
    Err(ApiError::BadRequest {
        message: message.into(),
        code: "unsupported_feature",
    })
}

fn validate_model_name(
    requested: Option<&str>,
    policy: ModelNamePolicy<'_>,
) -> Result<(), ApiError> {
    let expected = match policy {
        ModelNamePolicy::RequiredExact(expected) => match requested {
            None | Some("") => {
                return Err(ApiError::BadRequest {
                    message: "model is required".to_string(),
                    code: "invalid_request",
                });
            }
            Some(requested) if requested == expected => return Ok(()),
            Some(_) => expected,
        },
        ModelNamePolicy::OptionalExact(_) if requested.is_none() => return Ok(()),
        ModelNamePolicy::OptionalExact(expected) if requested == Some(expected) => return Ok(()),
        ModelNamePolicy::OptionalExact(expected) => expected,
    };
    let requested = requested.unwrap_or_default();
    Err(ApiError::BadRequest {
        message: format!("model '{requested}' is not loaded; this server serves '{expected}'"),
        code: "model_not_found",
    })
}

/// Resolves the effective `max_tokens` request. When
/// `conflict_checked_early` is `true`, `reject_unsupported` has already
/// rejected a conflicting `max_tokens`/`max_completion_tokens` pair (B3:
/// restores the daemon profile's original first-error precedence over the
/// served-model check), so a `Some, Some` pair reaching here is guaranteed
/// equal. When it is `false` (the `lattice` profile), that check has not
/// run yet -- it runs here instead, after the model check, matching that
/// profile's original `validate_max_tokens` position.
fn normalize_max_tokens(
    req: &ChatRequest,
    default_max_tokens: usize,
    policy: MaxTokensPolicy,
    conflict_checked_early: bool,
) -> Result<usize, ApiError> {
    if !conflict_checked_early {
        reject_conflicting_max_tokens(req)?;
    }
    let requested = match (req.max_tokens, req.max_completion_tokens) {
        (None, None) => default_max_tokens,
        (Some(value), None) | (None, Some(value)) => value,
        (Some(left), Some(_right)) => left,
    };
    super::reject_zero_max_tokens(requested)?;
    match policy {
        MaxTokensPolicy::RejectAbove { limit } if requested > limit => Err(ApiError::BadRequest {
            message: format!("max_tokens {requested} exceeds server limit {limit}"),
            code: "max_tokens_exceeds_limit",
        }),
        MaxTokensPolicy::RejectAbove { .. } => Ok(requested),
        MaxTokensPolicy::ClampToContext { context } => Ok(requested.min(context.saturating_sub(1))),
    }
}

/// Validate a temperature value against the shared `[0.0, 2.0]` contract.
pub fn validate_temperature(temperature: f32) -> Result<f32, ApiError> {
    if !(0.0..=2.0).contains(&temperature) {
        return Err(ApiError::BadRequest {
            message: "temperature must be between 0 and 2".to_string(),
            code: "invalid_temperature",
        });
    }
    Ok(temperature)
}

/// Validate a top-p value against the shared `(0.0, 1.0]` contract.
pub fn validate_top_p(top_p: f32) -> Result<f32, ApiError> {
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(ApiError::BadRequest {
            message: "top_p must be greater than 0 and at most 1".to_string(),
            code: "invalid_top_p",
        });
    }
    Ok(top_p)
}

/// Validates `top_logprobs` given an already-admitted `logprobs` request.
/// Whether `logprobs` is supported at all, and the `stream` + `logprobs`
/// conflict, are rejected earlier by `reject_unsupported` (B3: restores
/// original first-error precedence over the served-model check).
fn normalize_logprobs(req: &ChatRequest) -> Result<Option<usize>, ApiError> {
    if !req.logprobs.unwrap_or(false) {
        if req.top_logprobs.is_some() {
            return Err(ApiError::BadRequest {
                message: "top_logprobs requires logprobs: true".to_string(),
                code: "invalid_request",
            });
        }
        return Ok(None);
    }
    let top_logprobs = req.top_logprobs.unwrap_or(0);
    if top_logprobs > 20 {
        return Err(ApiError::BadRequest {
            message: format!("top_logprobs {top_logprobs} exceeds the maximum of 20"),
            code: "invalid_top_logprobs",
        });
    }
    Ok(Some(top_logprobs))
}

/// Validate and convert wire messages into engine chat messages.
pub fn normalize_messages(messages: &[Message]) -> Result<Vec<ChatMessage>, ApiError> {
    messages
        .iter()
        .map(|message| {
            let content = message_text(&message.content)?;
            match message.role.as_str() {
                "system" => Ok(ChatMessage::system(content)),
                "user" => Ok(ChatMessage::user(content)),
                "assistant" => Ok(ChatMessage::assistant(content)),
                "tool" | "developer" => Err(ApiError::BadRequest {
                    message: format!("role '{}' is not supported by this server", message.role),
                    code: "unsupported_feature",
                }),
                role => Err(ApiError::BadRequest {
                    message: format!(
                        "unsupported role '{role}'; must be 'system', 'user', or 'assistant'"
                    ),
                    code: "invalid_role",
                }),
            }
        })
        .collect()
}

fn message_text(content: &MessageContent) -> Result<String, ApiError> {
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::Parts(parts) => {
            let mut output = String::new();
            for part in parts {
                match part {
                    ContentPart::Text { text } => output.push_str(text),
                    ContentPart::ImageUrl { .. } => {
                        return Err(ApiError::BadRequest {
                            message: "image input requires a vision-capable model".to_string(),
                            code: "unsupported_feature",
                        });
                    }
                    ContentPart::Unsupported { kind } => {
                        return Err(ApiError::BadRequest {
                            message: format!(
                                "content part type '{kind}' is not supported; only 'text' parts are accepted"
                            ),
                            code: "unsupported_feature",
                        });
                    }
                }
            }
            Ok(output)
        }
    }
}

/// Reject a stop string over [`MAX_STOP_STRING_BYTES`], before it reaches
/// `StopStringMatcher` storage/use. See [`MAX_STOP_STRING_BYTES`] for why.
fn check_stop_string_bytes(value: &str) -> Result<(), ApiError> {
    if value.len() > MAX_STOP_STRING_BYTES {
        return Err(ApiError::BadRequest {
            message: format!(
                "stop string has {} bytes; maximum is {MAX_STOP_STRING_BYTES}",
                value.len()
            ),
            code: "invalid_request_body",
        });
    }
    Ok(())
}

/// Parse an OpenAI stop field into at most four non-empty strings, each
/// bounded by [`MAX_STOP_STRING_BYTES`] and summing to at most
/// [`MAX_CUMULATIVE_STOP_BYTES`].
pub fn parse_stop_strings(stop: &Option<Value>) -> Result<Vec<String>, ApiError> {
    match stop {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(value)) if value.is_empty() => Err(ApiError::BadRequest {
            message: "stop string must not be empty".to_string(),
            code: "invalid_stop",
        }),
        Some(Value::String(value)) => {
            check_stop_string_bytes(value)?;
            Ok(vec![value.clone()])
        }
        Some(Value::Array(values)) if values.is_empty() => Err(ApiError::BadRequest {
            message: "stop array must not be empty".to_string(),
            code: "invalid_stop",
        }),
        Some(Value::Array(values)) if values.len() > 4 => Err(ApiError::BadRequest {
            message: format!("stop array has {} elements; maximum is 4", values.len()),
            code: "invalid_stop",
        }),
        Some(Value::Array(values)) => {
            let stops: Vec<String> = values
                .iter()
                .map(|value| match value {
                    Value::String(value) if value.is_empty() => Err(ApiError::BadRequest {
                        message: "stop string must not be empty".to_string(),
                        code: "invalid_stop",
                    }),
                    Value::String(value) => {
                        check_stop_string_bytes(value)?;
                        Ok(value.clone())
                    }
                    _ => Err(ApiError::BadRequest {
                        message: "each element of stop must be a string".to_string(),
                        code: "invalid_stop",
                    }),
                })
                .collect::<Result<_, _>>()?;
            let cumulative: usize = stops.iter().map(String::len).sum();
            if cumulative > MAX_CUMULATIVE_STOP_BYTES {
                return Err(ApiError::BadRequest {
                    message: format!(
                        "stop strings total {cumulative} bytes; maximum is {MAX_CUMULATIVE_STOP_BYTES}"
                    ),
                    code: "invalid_request_body",
                });
            }
            Ok(stops)
        }
        Some(_) => Err(ApiError::BadRequest {
            message: "stop must be a string or array of strings".to_string(),
            code: "invalid_stop",
        }),
    }
}

/// Reject prompt plus decode budgets that exceed a model context window.
pub fn validate_context_window(
    prompt_tokens: usize,
    max_tokens: usize,
    max_context: usize,
) -> Result<(), ApiError> {
    if prompt_tokens == 0 || prompt_tokens.saturating_add(max_tokens) > max_context {
        return Err(ApiError::BadRequest {
            message: format!(
                "prompt ({prompt_tokens} tokens) plus max_tokens ({max_tokens}) exceeds model context window ({max_context})"
            ),
            code: "context_length_exceeded",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::metal_qwen35::ChatRole;

    fn request(body: &str) -> ChatRequest {
        serde_json::from_str(body).unwrap()
    }

    fn defaults() -> GenerationDefaults {
        GenerationDefaults {
            max_tokens: 16,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            reasoning_budget: None,
        }
    }

    #[test]
    fn shared_sampling_bounds_reject_invalid_values() {
        assert_eq!(
            validate_temperature(-0.1).unwrap_err().code(),
            "invalid_temperature"
        );
        assert_eq!(
            validate_temperature(2.1).unwrap_err().code(),
            "invalid_temperature"
        );
        assert_eq!(validate_top_p(0.0).unwrap_err().code(), "invalid_top_p");
        assert_eq!(validate_top_p(1.1).unwrap_err().code(), "invalid_top_p");
    }

    #[test]
    fn both_profiles_use_shared_sampling_bounds() {
        let req = request(
            r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"temperature":2.5}"#,
        );
        for profile in [
            ServeProfile::lattice("model", 32),
            ServeProfile::lattice_serve("model", 32),
        ] {
            assert_eq!(
                normalize_request(&req, defaults(), profile, |_, _| Ok(()))
                    .unwrap_err()
                    .code(),
                "invalid_temperature"
            );
        }
    }

    #[test]
    fn profiles_preserve_stop_and_max_token_policy() {
        let req = request(
            r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"max_tokens":31,"stop":"done"}"#,
        );
        let (lattice, ()) = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice("model", 32),
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(lattice.max_tokens, 31);
        assert_eq!(lattice.stop_strings, ["done"]);
        assert_eq!(
            normalize_request(
                &req,
                defaults(),
                ServeProfile::lattice_serve("model", 16),
                |_, _| Ok(())
            )
            .unwrap_err()
            .code(),
            "unsupported_feature"
        );

        let req = request(
            r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"max_tokens":31}"#,
        );
        let (daemon, ()) = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice_serve("model", 16),
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(daemon.max_tokens, 15);
    }

    #[test]
    fn profiles_preserve_sampling_extension_policy() {
        // lattice serve never carried top_k/repetition_penalty in its
        // pre-shared-contract DTO and silently ignored them; the shared
        // contract must keep accepting-and-ignoring these fields for that
        // profile rather than rejecting the request (regression guard).
        let req = request(
            r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"top_k":20,"repetition_penalty":1.2}"#,
        );
        let (lattice, ()) = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice("model", 32),
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(lattice.top_k, defaults().top_k);
        assert_eq!(lattice.repetition_penalty, defaults().repetition_penalty);

        let (daemon, ()) = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice_serve("model", 32),
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(daemon.top_k, 20);
        assert_eq!(daemon.repetition_penalty, 1.2);
    }

    #[test]
    fn lattice_profile_accepts_and_ignores_reasoning_budget() {
        // Mirrors profiles_preserve_sampling_extension_policy for
        // reasoning_budget: lattice serve must accept the field without a
        // 400 and must not let it influence the effective budget.
        let req = request(
            r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"reasoning_budget":40}"#,
        );
        let (lattice, ()) = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice("model", 32),
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(lattice.reasoning_budget, None);

        // Context must leave enough decode room (>= reasoning_budget) so the
        // context-window clamp in normalize_request doesn't also shrink the
        // value under test; that clamp is exercised separately.
        let (daemon, ()) = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice_serve("model", 100),
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(daemon.reasoning_budget, Some(40));
    }

    #[test]
    fn lattice_profile_tolerates_malformed_ignored_sampling_fields() {
        // Regression: pre-refactor, lattice serve's DTO didn't model
        // top_k/repetition_penalty/reasoning_budget at all, so a
        // type-mismatched value (e.g. a string where a number is expected)
        // was silently ignored along with everything else serde didn't
        // recognize. The shared typed DTO must preserve that tolerance
        // rather than hard-failing the whole request with
        // invalid_request_body before profile normalization runs.
        let req = request(
            r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"top_k":"ignored","repetition_penalty":"x","reasoning_budget":"y"}"#,
        );
        let (lattice, ()) = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice("model", 32),
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(lattice.top_k, defaults().top_k);
        assert_eq!(lattice.repetition_penalty, defaults().repetition_penalty);
        assert_eq!(lattice.reasoning_budget, None);
    }

    #[test]
    fn honoring_profile_still_rejects_malformed_sampling_fields() {
        // Control: a profile that actually USES top_k /
        // repetition_penalty / reasoning_budget must keep strict
        // validation — tolerance is scoped to profiles that ignore the
        // field, not a blanket type-laxness relaxation.
        for (field, bad_value) in [
            ("top_k", r#""ignored""#),
            ("repetition_penalty", r#""x""#),
            ("reasoning_budget", r#""y""#),
        ] {
            let body = format!(
                r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"{field}":{bad_value}}}"#
            );
            let req = request(&body);
            assert_eq!(
                normalize_request(
                    &req,
                    defaults(),
                    ServeProfile::lattice_serve("model", 32),
                    |_, _| Ok(())
                )
                .unwrap_err()
                .code(),
                "invalid_request_body",
                "field {field} should have been rejected on the honoring profile"
            );
        }
    }

    #[test]
    fn message_count_over_limit_is_rejected_during_deserialization() {
        // Regression: the message-count bound is enforced inline while
        // `messages` deserializes ([`deserialize_bounded_messages`]), so an
        // over-limit request never reaches a constructed `ChatRequest`,
        // `normalize_request`, or `check_context` at all -- there is no
        // separate preflight pass and no post-deserialization gap to close.
        let messages: Vec<String> = (0..MAX_MESSAGE_COUNT + 1)
            .map(|_| r#"{"role":"user","content":""}"#.to_string())
            .collect();
        let body = format!(r#"{{"model":"model","messages":[{}]}}"#, messages.join(","));
        let err = serde_json::from_str::<ChatRequest>(&body).unwrap_err();
        assert!(is_message_flood_error(&err));
    }

    #[test]
    fn message_flood_error_short_circuits_at_max_plus_one() {
        // Single-pass, short-circuiting: an array far larger than
        // `MAX_MESSAGE_COUNT` still rejects promptly, because
        // `deserialize_bounded_messages`'s visitor bails the instant its
        // running count exceeds the limit -- it never visits the remaining
        // elements. A large-but-finite array here proves the bail actually
        // fires past the boundary, not just that a walk-to-completion would
        // eventually reject it too.
        let messages: Vec<String> = (0..MAX_MESSAGE_COUNT * 4)
            .map(|_| r#"{"role":"user","content":""}"#.to_string())
            .collect();
        let body = format!(r#"{{"model":"model","messages":[{}]}}"#, messages.join(","));
        let err = serde_json::from_str::<ChatRequest>(&body).unwrap_err();
        assert!(is_message_flood_error(&err));
    }

    #[test]
    fn non_flood_deserialize_errors_are_not_misclassified_as_message_flood() {
        assert!(!is_message_flood_error(
            &serde_json::from_str::<ChatRequest>("not json").unwrap_err()
        ));
        assert!(!is_message_flood_error(
            &serde_json::from_str::<ChatRequest>(
                r#"{"model":"model","messages":[{"role":123,"content":"hi"}]}"#
            )
            .unwrap_err()
        ));
    }

    #[test]
    fn cumulative_content_bytes_over_limit_is_rejected() {
        let big_content = "x".repeat(MAX_CUMULATIVE_CONTENT_BYTES + 1);
        let body = format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"{big_content}"}}]}}"#
        );
        let req = request(&body);
        assert_eq!(
            normalize_request(
                &req,
                defaults(),
                ServeProfile::lattice("model", 32),
                |_, _| Ok(())
            )
            .unwrap_err()
            .code(),
            "invalid_request_body"
        );
    }

    #[test]
    fn message_count_and_content_bytes_at_the_limit_are_accepted() {
        let messages: Vec<String> = (0..MAX_MESSAGE_COUNT)
            .map(|i| {
                if i + 1 == MAX_MESSAGE_COUNT {
                    r#"{"role":"user","content":"hi"}"#.to_string()
                } else {
                    r#"{"role":"user","content":""}"#.to_string()
                }
            })
            .collect();
        let body = format!(r#"{{"model":"model","messages":[{}]}}"#, messages.join(","));
        let req = request(&body);
        normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice("model", 32),
            |_, _| Ok(()),
        )
        .unwrap();
    }

    #[test]
    fn stop_string_over_byte_limit_is_rejected_before_matcher_construction() {
        // Regression (B1): a near-1MiB stop string must be rejected as
        // invalid_request_body before StopStringMatcher ever sees it -- not
        // parsed into `stop_strings` and handed to the matcher/decode loop.
        let big_stop = "x".repeat(MAX_STOP_STRING_BYTES + 1);
        let body = format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"stop":"{big_stop}"}}"#
        );
        let req = request(&body);
        let err = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice("model", 32),
            |_, _| Ok(()),
        )
        .unwrap_err();
        assert_eq!(err.code(), "invalid_request_body");

        // Same bound applies to each element of a stop array.
        let body = format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"stop":["{big_stop}"]}}"#
        );
        let req = request(&body);
        assert_eq!(
            normalize_request(
                &req,
                defaults(),
                ServeProfile::lattice("model", 32),
                |_, _| Ok(())
            )
            .unwrap_err()
            .code(),
            "invalid_request_body"
        );
    }

    #[test]
    fn stop_strings_over_cumulative_byte_limit_are_rejected() {
        // Regression (B1): four individually-under-limit stop strings that
        // together exceed MAX_CUMULATIVE_STOP_BYTES must still be rejected
        // -- the per-string cap alone is not the only bound enforced. Each
        // string is well under MAX_STOP_STRING_BYTES; only the sum crosses
        // MAX_CUMULATIVE_STOP_BYTES.
        let each = "x".repeat(MAX_CUMULATIVE_STOP_BYTES / 4 + 1);
        let stops = format!(r#""{each}","{each}","{each}","{each}""#);
        let body = format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"stop":[{stops}]}}"#
        );
        let req = request(&body);
        assert_eq!(
            normalize_request(
                &req,
                defaults(),
                ServeProfile::lattice("model", 32),
                |_, _| Ok(())
            )
            .unwrap_err()
            .code(),
            "invalid_request_body"
        );
    }

    #[test]
    fn stop_string_at_the_byte_limit_is_accepted() {
        let ok_stop = "x".repeat(MAX_STOP_STRING_BYTES);
        let body = format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"stop":"{ok_stop}"}}"#
        );
        let req = request(&body);
        let (validated, ()) = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice("model", 32),
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(validated.stop_strings, vec![ok_stop]);
    }

    #[test]
    fn non_scalar_extension_field_rejected_without_cloning_large_payload() {
        // Regression (B2): a large non-scalar (array) value for a
        // sampling-extension field must be rejected on the honoring profile
        // via the cheap `is_array`/`is_object` shape check in
        // `parse_ignorable_field`, before any `Value::clone()` or
        // `from_value` deserialization attempt is made on the whole payload.
        // Asserting the exact "must be a scalar value" message (not just the
        // shared `invalid_request_body` code, which `from_value`'s own
        // type-mismatch error also produces) pins that the shape-check path
        // fired, not the deserialize-then-fail path the pre-B2 code took.
        let big_array = format!("[{}]", vec!["1"; 100_000].join(","));
        let body = format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"top_k":{big_array}}}"#
        );
        let req = request(&body);
        let err = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice_serve("model", 32),
            |_, _| Ok(()),
        )
        .unwrap_err();
        assert_eq!(err.code(), "invalid_request_body");
        assert_eq!(err.message(), "top_k must be a scalar value");
    }

    #[test]
    fn wrong_model_with_conflicting_stream_and_logprobs_rejects_as_unsupported_feature() {
        // Regression (B3): restores the pre-shared-contract first-error
        // precedence -- `stream: true` + `logprobs: true` must be rejected
        // as unsupported_feature even when the request ALSO targets the
        // wrong model, matching both binaries' original `reject_unsupported`
        // ordering (ran before the model-id check).
        let req = request(
            r#"{"model":"wrong-model","messages":[{"role":"user","content":"hi"}],"stream":true,"logprobs":true}"#,
        );
        assert_eq!(
            normalize_request(
                &req,
                defaults(),
                ServeProfile::lattice("served-model", 32),
                |_, _| Ok(())
            )
            .unwrap_err()
            .code(),
            "unsupported_feature"
        );
    }

    #[test]
    fn wrong_model_with_unsupported_logprobs_on_daemon_profile_rejects_as_unsupported_feature() {
        // Regression (B3): the standalone daemon profile never supports
        // logprobs at all; that rejection must still win over model_not_found.
        let req = request(
            r#"{"model":"wrong-model","messages":[{"role":"user","content":"hi"}],"logprobs":true}"#,
        );
        assert_eq!(
            normalize_request(
                &req,
                defaults(),
                ServeProfile::lattice_serve("served-model", 32),
                |_, _| Ok(())
            )
            .unwrap_err()
            .code(),
            "unsupported_feature"
        );
    }

    #[test]
    fn wrong_model_with_conflicting_max_tokens_alias_rejects_as_invalid_request() {
        // Regression (B3): a conflicting max_tokens/max_completion_tokens
        // pair must be rejected as invalid_request even when the request
        // ALSO targets the wrong model, matching the standalone daemon's
        // original `reject_unsupported` ordering (ran before the model-id
        // check).
        let req = request(
            r#"{"model":"wrong-model","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"max_completion_tokens":20}"#,
        );
        assert_eq!(
            normalize_request(
                &req,
                defaults(),
                ServeProfile::lattice_serve("served-model", 32),
                |_, _| Ok(())
            )
            .unwrap_err()
            .code(),
            "invalid_request"
        );
    }

    #[test]
    fn wrong_model_with_conflicting_max_tokens_alias_rejects_as_model_not_found_on_lattice_profile()
    {
        // Regression: the `lattice` profile's original (pre-shared-contract)
        // precedence checked the max_tokens/max_completion_tokens conflict
        // inside `validate_max_tokens`, which ran AFTER the model-id check
        // -- unlike the daemon profile above. A wrong model combined with a
        // conflicting alias pair must still surface `model_not_found` on
        // this profile.
        let req = request(
            r#"{"model":"wrong-model","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"max_completion_tokens":20}"#,
        );
        assert_eq!(
            normalize_request(
                &req,
                defaults(),
                ServeProfile::lattice("served-model", 32),
                |_, _| Ok(())
            )
            .unwrap_err()
            .code(),
            "model_not_found"
        );
    }

    #[test]
    fn lattice_profile_still_rejects_genuinely_unsupported_fields() {
        // Control: tools/tool_choice remain a hard 400 on both profiles —
        // the restored tolerance is scoped to exactly the three named
        // fields, not a blanket "accept everything" relaxation.
        let req = request(
            r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}]}"#,
        );
        assert_eq!(
            normalize_request(
                &req,
                defaults(),
                ServeProfile::lattice("model", 32),
                |_, _| Ok(())
            )
            .unwrap_err()
            .code(),
            "unsupported_feature"
        );
    }

    #[test]
    fn unmodeled_openai_top_level_fields_are_ignored_not_rejected() {
        let req = serde_json::from_str::<ChatRequest>(
            r#"{
                "model": "model",
                "messages": [{"role": "user", "content": "hi"}],
                "presence_penalty": 1.0,
                "frequency_penalty": 0.5,
                "logit_bias": {"123": -100},
                "user": "end-user-id"
            }"#,
        )
        .expect("unmodeled top-level fields must be ignored, not rejected");
        assert_eq!(req.model.as_deref(), Some("model"));
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn context_window_accepts_boundary_and_rejects_overflow() {
        validate_context_window(8, 8, 16).unwrap();
        assert_eq!(
            validate_context_window(8, 9, 16).unwrap_err().code(),
            "context_length_exceeded"
        );
    }

    #[test]
    fn optional_exact_rejects_explicit_empty_model() {
        let err =
            validate_model_name(Some(""), ModelNamePolicy::OptionalExact("served")).unwrap_err();
        assert_eq!(err.code(), "model_not_found");
    }

    #[test]
    fn optional_exact_accepts_omitted_model() {
        validate_model_name(None, ModelNamePolicy::OptionalExact("served")).unwrap();
    }

    #[test]
    fn optional_exact_matches_and_rejects_by_name() {
        validate_model_name(Some("served"), ModelNamePolicy::OptionalExact("served")).unwrap();
        assert_eq!(
            validate_model_name(Some("other"), ModelNamePolicy::OptionalExact("served"))
                .unwrap_err()
                .code(),
            "model_not_found"
        );
    }

    #[test]
    fn required_exact_rejects_missing_or_empty_and_accepts_match() {
        assert_eq!(
            validate_model_name(None, ModelNamePolicy::RequiredExact("served"))
                .unwrap_err()
                .code(),
            "invalid_request"
        );
        assert_eq!(
            validate_model_name(Some(""), ModelNamePolicy::RequiredExact("served"))
                .unwrap_err()
                .code(),
            "invalid_request"
        );
        validate_model_name(Some("served"), ModelNamePolicy::RequiredExact("served")).unwrap();
    }

    fn message(body: &str) -> Message {
        serde_json::from_str(body).unwrap()
    }

    // Both binaries keep `#[cfg(test)]`-local role/content copies for their
    // own request-parsing tests, but production traffic on both servers
    // normalizes messages through this shared `normalize_messages` — so the
    // validator itself needs direct coverage or production behavior can
    // drift while the binary-local test copies stay green.
    #[test]
    fn normalize_messages_accepts_system_user_assistant_roles() {
        let messages = [
            message(r#"{"role":"system","content":"sys"}"#),
            message(r#"{"role":"user","content":"usr"}"#),
            message(r#"{"role":"assistant","content":"asst"}"#),
        ];
        let normalized = normalize_messages(&messages).unwrap();
        assert_eq!(normalized[0].role, ChatRole::System);
        assert_eq!(normalized[0].content, "sys");
        assert_eq!(normalized[1].role, ChatRole::User);
        assert_eq!(normalized[1].content, "usr");
        assert_eq!(normalized[2].role, ChatRole::Assistant);
        assert_eq!(normalized[2].content, "asst");
    }

    #[test]
    fn normalize_messages_rejects_unrecognized_role() {
        let messages = [message(r#"{"role":"moderator","content":"hi"}"#)];
        assert_eq!(
            normalize_messages(&messages).unwrap_err().code(),
            "invalid_role"
        );
    }

    #[test]
    fn normalize_messages_rejects_known_but_unsupported_roles() {
        for role in ["tool", "developer"] {
            let messages = [message(&format!(r#"{{"role":"{role}","content":"hi"}}"#))];
            assert_eq!(
                normalize_messages(&messages).unwrap_err().code(),
                "unsupported_feature"
            );
        }
    }

    #[test]
    fn normalize_messages_checks_content_before_role() {
        // Contract choice, deliberately kept rather than restored:
        // `normalize_messages` validates content before role, matching
        // `lattice.rs`'s original pre-shared-contract order.
        // `lattice_serve.rs`'s original order was the reverse (role before
        // content); the shared normalizer now applies content-before-role
        // uniformly on both server profiles. A message with both an invalid
        // role and unsupported (image) content surfaces the content error.
        let messages = [message(
            r#"{"role":"moderator","content":[{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}]}"#,
        )];
        let err = normalize_messages(&messages).unwrap_err();
        assert_eq!(err.message(), "image input requires a vision-capable model");
    }

    #[test]
    fn honoring_profile_treats_explicit_null_sampling_fields_as_absent() {
        // Regression (null-handling): an explicit JSON `null` for an honored
        // extension field must default, same as an omitted field, not
        // reject. Also pins that the `Option<Value>` -> `Option<Box<RawValue>>`
        // field-type change preserves serde's existing
        // null-maps-to-`None` `Option<T>` handling -- `parse_ignorable_field`
        // never even runs for these three fields on a `"field": null`
        // request, since the value is `None` by the time it reaches there.
        let req = request(
            r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"top_k":null,"repetition_penalty":null,"reasoning_budget":null}"#,
        );
        let (daemon, ()) = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice_serve("model", 100),
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(daemon.top_k, defaults().top_k);
        assert_eq!(daemon.repetition_penalty, defaults().repetition_penalty);
        assert_eq!(daemon.reasoning_budget, None);
    }

    #[test]
    fn honoring_profile_rejects_near_body_cap_nested_extension_field() {
        // Regression: a near-1MiB nested array submitted for an
        // honored extension field must be rejected via the cheap first-byte
        // span check in `parse_ignorable_field`, never via `from_str` on the
        // whole span and never via building a `Value` tree -- `RawValue`'s
        // `Deserialize` impl only records the span during the outer
        // `ChatRequest` parse, it does not recurse into it.
        let big_array = format!("[{}]", vec!["1"; REQUEST_BODY_LIMIT_BYTES / 2].join(","));
        let body = format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"top_k":{big_array}}}"#
        );
        let req = request(&body);
        let err = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice_serve("model", 32),
            |_, _| Ok(()),
        )
        .unwrap_err();
        assert_eq!(err.code(), "invalid_request_body");
        assert_eq!(err.message(), "top_k must be a scalar value");
    }

    #[test]
    fn non_honoring_profile_ignores_near_body_cap_nested_extension_field() {
        // Regression: the same near-1MiB nested array on a
        // profile that does not honor the field must be silently ignored,
        // not rejected -- `parse_ignorable_field` is never even called for
        // it (see the `else { defaults.top_k }` branch in
        // `normalize_request`), so the span is never inspected at all.
        let big_array = format!("[{}]", vec!["1"; REQUEST_BODY_LIMIT_BYTES / 2].join(","));
        let body = format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"top_k":{big_array}}}"#
        );
        let req = request(&body);
        let (lattice, ()) = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice("model", 32),
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(lattice.top_k, defaults().top_k);
    }

    #[test]
    fn lattice_profile_rejects_json_schema_without_materializing_the_schema() {
        // Regression: `JsonSchemaFormat::schema` must stay a `RawValue` span
        // rather than an eagerly-parsed `Value` tree, because the `lattice`
        // profile (`structured_output_supported: false`) always rejects
        // `response_format.type: "json_schema"` via `reject_unsupported`
        // without ever reading `schema`. A schema whose nesting exceeds
        // `serde_json`'s ~128-level recursion limit proves the point
        // structurally: parsing it into `Value` during the outer
        // `ChatRequest` deserialization would itself fail with a recursion
        // error before `normalize_request` ever runs, but `RawValue`
        // (which only records the byte span, never recurses into it) lets
        // deserialization succeed, so the request reaches
        // `reject_unsupported` and is rejected for the right reason:
        // unsupported response-format type, not an accidental parse error
        // one level of the stack never asked to pay for.
        let mut nested = String::new();
        for _ in 0..200 {
            nested.push('[');
        }
        nested.push('1');
        for _ in 0..200 {
            nested.push(']');
        }
        assert!(
            serde_json::from_str::<Value>(&nested).is_err(),
            "fixture must exceed serde_json's Value recursion limit"
        );
        let body = format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],
            "response_format":{{"type":"json_schema","json_schema":{{"name":"n","strict":true,"schema":{nested}}}}}}}"#
        );
        let req = request(&body);
        let err = normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice("model", 32),
            |_, _| Ok(()),
        )
        .unwrap_err();
        assert_eq!(err.code(), "unsupported_feature");
        assert_eq!(
            err.message(),
            "response_format.type 'json_schema' is not supported; use 'text'"
        );
    }

    #[test]
    fn valid_request_body_is_parsed_by_a_single_deserialize_call() {
        // Regression: there is no separate raw-bytes preflight pass over
        // `messages` anymore -- `serde_json::from_str::<ChatRequest>` is the
        // only parse a caller needs to run, and it both counts and
        // materializes the array in that one call.
        let messages: Vec<String> = (0..MAX_MESSAGE_COUNT)
            .map(|_| r#"{"role":"user","content":""}"#.to_string())
            .collect();
        let body = format!(r#"{{"model":"model","messages":[{}]}}"#, messages.join(","));
        let req = serde_json::from_str::<ChatRequest>(&body).unwrap();
        assert_eq!(req.messages.len(), MAX_MESSAGE_COUNT);
    }

    #[test]
    fn normalize_messages_passes_through_empty_and_whitespace_content() {
        // Documents the current production contract: normalize_messages
        // does not trim or reject empty/whitespace content — it is passed
        // through unchanged, same as any other text content.
        let messages = [
            message(r#"{"role":"user","content":""}"#),
            message(r#"{"role":"user","content":"   "}"#),
        ];
        let normalized = normalize_messages(&messages).unwrap();
        assert_eq!(normalized[0].content, "");
        assert_eq!(normalized[1].content, "   ");
    }
}

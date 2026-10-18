//! `POST /v1/embeddings`: pooled text and image embeddings from a loaded
//! Qwen3.5 vision-language checkpoint (used by the `lattice` binary's
//! `--embeddings-model`), plus a separate, GPU-independent OpenAI-compatible
//! `/v1/embeddings` wire contract for the standalone `lattice_serve` daemon's
//! own `--embedding-model` (issue #584), which serves through
//! `crate::model::bert::BertModel` rather than the vision-language loader
//! below.
//!
//! The two are intentionally independent code paths sharing only this file
//! and the `ApiError` envelope: `lattice_serve`'s text-only route predates
//! and is unrelated in scope to the vision-language loader, and reuses
//! neither its model type nor its wire DTOs (`TextEmbeddingsRequest`/
//! `TextEmbeddingsInput` below, vs. `EmbeddingsRequest`/`EmbeddingsInput` for
//! the vision-language route).
//!
//! The model type here ([`EmbeddingModel`](crate::serve::embeddings::EmbeddingModel)) loads the same way
//! `lattice-embed`'s `VisionEmbeddingModel` does -- an f16-packed decoder plus
//! vision-encoder weights from one safetensors checkpoint -- but is defined
//! in this crate rather than reused from `lattice-embed`, because
//! `lattice-embed` depends on `lattice-inference` (not the other way around):
//! a serving binary in this crate cannot name `lattice-embed`'s type without
//! a dependency cycle. This loader calls the same public
//! [`crate::vision::checkpoint`] / [`crate::weights::f16_weights`] functions
//! `VisionEmbeddingModel::from_directory` wraps; no pooling or checkpoint
//! math is duplicated, only the small amount of loading glue.
//!
//! `TextEmbeddingsRequest`/`TextEmbeddingsInput`/`parse_embeddings_input`/
//! `check_embeddings_model`/`build_embeddings_response` below are
//! deliberately independent of `lattice-embed`'s `EmbeddingService` trait for
//! the same dependency-cycle reason (`lattice-embed` already depends on
//! `lattice-inference` for `BertModel`/`QwenModel`); they normalize the wire
//! request and build the wire response, while the actual embedding compute
//! is `crate::model::bert::BertModel`, called directly by
//! `lattice_serve.rs`'s own handler.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::InferenceError;
use crate::forward::cpu_f16::{PoolingStrategy, embed_text_vlm_f16};
use crate::model::qwen35_config::Qwen35Config;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer as _;
use crate::vision::checkpoint::{
    Qwen35VisionWeights, load_qwen35_vision_weights_from_safetensors,
    open_qwen35_single_decoder_safetensors,
};
use crate::vision::qwen35_vit::preprocess_qwen35_image;
use crate::vision::{embed_image_from_bytes_f16, embed_image_from_bytes_f16_metal};
use crate::weights::f16_weights::{F16ModelWeights, load_f16_weights};

use super::ApiError;
use super::contract::decode_inline_image;
use serde_json::{Value, json};

/// Maximum number of items accepted in one `/v1/embeddings` request's
/// `input` array, mirroring [`super::contract::MAX_MESSAGE_COUNT`]'s role for
/// chat requests: caps per-item allocation and pooled-decode cost before any
/// item is processed, independent of [`super::REQUEST_BODY_LIMIT_BYTES`].
pub const MAX_EMBEDDING_INPUT_COUNT: usize = 4096;

// ---------------------------------------------------------------------------
// Model loading and execution
// ---------------------------------------------------------------------------

/// Raises `tokenizer`'s own truncation cap to `max_context` when it sits
/// below it, otherwise returns `tokenizer` unchanged.
///
/// This is the shared enforcement point for the invariant issue #1408
/// requires: no constructed [`EmbeddingModel`] may have a tokenizer cap
/// below its own [`EmbeddingModel::max_context`]. If it did,
/// `check_item_fits_window`'s pre-truncation-count admission check (see
/// [`embed_items`]) would still admit an over-window item -- its
/// pre-truncation count is compared against `max_context`, which the
/// tokenizer cap sits below -- but the tokenizer would then silently
/// truncate it before the forward pass, embedding a truncated prefix
/// instead of rejecting the input. Only ever raises the cap, never lowers
/// it: a caller-supplied tokenizer already capped at or above
/// `max_context` (e.g. a fixture deliberately testing a *narrower* cap
/// than its own window) is left untouched.
fn capped_tokenizer(tokenizer: BpeTokenizer, max_context: usize) -> BpeTokenizer {
    if tokenizer.max_seq_len() < max_context {
        tokenizer.with_max_seq_len(max_context)
    } else {
        tokenizer
    }
}

/// How a loader reads a checkpoint, recorded in a router artifact's
/// representation so a gate cannot be served by a loader it was not trained
/// through. Two names exist because this workspace ships two serving binaries
/// whose embedders really do read the bytes differently: one shared constant
/// would make every cross-binary comparison pass by construction, which is
/// the one way this check can be wrong without ever failing.
///
/// `EmbeddingModel::from_directory` in this module -- a Qwen3.5 f16 decoder.
/// The `lattice` binary reads this one; it is the only binary that checks a
/// router gate.
pub const QWEN35_F16_DECODER_LOADER: &str = "qwen35-f16-decoder";

/// `BertModel::from_directory`, which is what `lattice_serve` embeds through.
/// That binary has no routing path today, so nothing compares against this
/// value at startup. It is declared anyway for two reasons: it gives the
/// comparison in `check_representation` a second value that is REAL rather
/// than invented, and it means a later wiring of routing into that binary
/// names its own loader instead of silently inheriting this module's.
pub const BERT_ENCODER_LOADER: &str = "bert-encoder";

/// A loaded Qwen3.5 vision-language checkpoint used to serve pooled text and
/// image embeddings. See the module doc comment for why this loader lives
/// here instead of reusing `lattice-embed::vision::VisionEmbeddingModel`.
///
/// The two constants above used to sit between this comment and the struct,
/// which silently reattached these three lines to a string constant and left
/// the struct undocumented.
pub struct EmbeddingModel {
    weights: F16ModelWeights,
    config: Qwen35Config,
    vision_weights: Qwen35VisionWeights,
    tokenizer: BpeTokenizer,
}

impl EmbeddingModel {
    /// Compose a model from already-loaded components (no I/O). Used by
    /// tests and by callers that share components with another in-process
    /// model instance.
    ///
    /// Raises `tokenizer`'s own truncation cap to this checkpoint's
    /// [`Self::max_context`] window when it sits below it (see
    /// `capped_tokenizer`), so this constructor gives the same guarantee
    /// [`Self::from_directory`] does: no `EmbeddingModel`, however
    /// constructed, can have a tokenizer cap below its own `max_context()`.
    /// Without this, a caller passing a default `BpeTokenizer` (whose
    /// constructors cap at 4096) alongside a checkpoint config whose
    /// `max_position_embeddings.min(8192)` exceeds 4096 would reproduce
    /// exactly the #1408 bug through this constructor instead of the loader.
    pub fn new(
        weights: F16ModelWeights,
        config: Qwen35Config,
        vision_weights: Qwen35VisionWeights,
        tokenizer: BpeTokenizer,
    ) -> Self {
        let max_context = config.max_position_embeddings.min(8192);
        let tokenizer = capped_tokenizer(tokenizer, max_context);
        Self {
            weights,
            config,
            vision_weights,
            tokenizer,
        }
    }

    /// Load a Qwen3.5 vision-language checkpoint directory for embedding
    /// service. Same layout and error conditions as `lattice-embed`'s
    /// `VisionEmbeddingModel::from_directory`: `config.json` must declare a
    /// `vision_config`, the directory must resolve to exactly one decoder
    /// safetensors shard, and a `quantize_index.json` (native Q4) checkpoint
    /// is rejected -- this loader only understands the f16 decoder format.
    pub fn from_directory(dir: &Path) -> Result<Self, String> {
        let quantized_index = dir.join("quantize_index.json");
        if quantized_index.exists() {
            return Err(format!(
                "{} is present, but quantized checkpoints are not supported by the embeddings \
                 f16 decoder loader",
                quantized_index.display()
            ));
        }

        let config = Qwen35Config::from_model_dir(dir).map_err(|e| format!("config.json: {e}"))?;
        let vision_cfg = config.vision_config.clone().ok_or_else(|| {
            format!(
                "{} has no vision_config; not a vision-language checkpoint",
                dir.display()
            )
        })?;

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path)
            .map_err(|e| format!("{}: {e}", tokenizer_path.display()))?;

        let (mut sf, shard_path) = open_qwen35_single_decoder_safetensors(dir)
            .map_err(|e| format!("decoder checkpoint: {e}"))?;
        let vision_weights =
            load_qwen35_vision_weights_from_safetensors(&mut sf, &shard_path, &vision_cfg)
                .map_err(|e| format!("vision weights: {e}"))?;
        let weights =
            load_f16_weights(&sf, &config).map_err(|e| format!("decoder weights: {e}"))?;

        // Routed through `Self::new` (not a bare struct literal) so the
        // tokenizer's truncation cap is raised to this checkpoint's context
        // window exactly as any other construction path would be -- see
        // `Self::new`'s and `capped_tokenizer`'s docs for why: otherwise,
        // whenever `max_position_embeddings.min(8192)` exceeds
        // `BpeTokenizer`'s `DEFAULT_BPE_MAX_SEQ_LEN` (4096), the tokenizer
        // can silently truncate an input to 4096 tokens while
        // `check_item_fits_window` admits it (its pre-truncation count is
        // still under the wider window), embedding a truncated prefix
        // instead of rejecting it.
        Ok(Self::new(weights, config, vision_weights, tokenizer))
    }

    /// Output embedding dimension (the checkpoint's decoder hidden size).
    pub fn dimensions(&self) -> usize {
        self.config.hidden_size
    }

    /// Real (possibly truncated) tokenized length of `text`, used for
    /// `usage.prompt_tokens` -- this must report what the pooled forward
    /// pass actually consumes, not what the caller sent. Never use this for
    /// a context-window admission check: see `Self::tokenize_lengths`.
    pub fn tokenize_len(&self, text: &str) -> usize {
        self.tokenizer.tokenize(text).real_length
    }

    /// Tokenizes `text` once, returning `(pre_truncation_len, real_length)`
    /// -- the count the context-window admission guard
    /// ([`check_item_fits_window`]) must compare against, and the count the
    /// forward pass actually consumes for `usage.prompt_tokens`,
    /// respectively. Exists so a caller that needs both counts (the
    /// [`embed_items`] text branch) tokenizes once instead of twice;
    /// [`Self::tokenize_len`] above remains the single-value accessor for a
    /// caller (e.g. [`Self::image_scaffold_token_count`]) that only needs
    /// the real, post-truncation count.
    ///
    /// `pre_truncation_len` must be used for admission, never `real_length`:
    /// `real_length` can never exceed the tokenizer's `max_seq_len`, so
    /// comparing it against `max_context()` can never observe an
    /// over-window input once `max_seq_len <= max_context()` -- the
    /// rejection becomes unreachable and an over-long input is silently
    /// embedded from a truncated prefix instead. See
    /// `TokenizedInput::pre_truncation_len`'s doc for the same reasoning
    /// applied to the sibling `lattice_serve` text-embeddings route.
    fn tokenize_lengths(&self, text: &str) -> (usize, usize) {
        let tokenized = self.tokenizer.tokenize(text);
        (tokenized.pre_truncation_len, tokenized.real_length)
    }

    /// Maximum decoder scaffold length this checkpoint can process in one
    /// pooled prefill, mirroring the exact cap the chat serving path derives
    /// from the same field for its RoPE table (`Qwen35Model::from_safetensors`:
    /// `config.max_position_embeddings.min(8192)`). Used as the source of
    /// truth for `check_item_fits_window`.
    pub fn max_context(&self) -> usize {
        self.config.max_position_embeddings.min(8192)
    }

    /// Real scaffold token count an image item consumes: one
    /// `vision_start_token_id`, one `vision_end_token_id`, the checkpoint's
    /// per-image pad-token count derived from the patch grid (mirroring
    /// `pooled_embed.rs`'s own `image_pad_positions`/`num_pads` arithmetic,
    /// via a separate, cheap preprocessing call rather than by reaching into
    /// that module's internals), and `prompt`'s tokenized length. Used both
    /// for `usage.prompt_tokens` accounting and (via the returned count) can
    /// be compared against [`Self::max_context`] the same way a text item is.
    ///
    /// Uses [`Self::tokenize_len`] (the real, possibly-truncated count), not
    /// the pre-truncation count `Self::tokenize_lengths` also exposes, for
    /// the `prompt` component -- unlike the text item guard, this is not a
    /// live gap today: this
    /// method's one production call site ([`embed_items`]'s image branch)
    /// always passes `prompt = ""`, whose tokenized length is `0` either
    /// way, so no input can currently exercise the truncation-vs-window
    /// mismatch through this path. A future caller that passes a real
    /// `prompt` here would need the same admission/usage split the text
    /// branch got; left as real-length-only for now rather than adding an
    /// unexercised second accessor for a code path with no non-empty-prompt
    /// caller.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidInput`] if the checkpoint has no
    /// `vision_config`, or if the image cannot be decoded/preprocessed (see
    /// [`preprocess_qwen35_image`]'s error conditions).
    pub fn image_scaffold_token_count(
        &self,
        image_bytes: &[u8],
        prompt: &str,
    ) -> Result<usize, InferenceError> {
        let vision_cfg = self.config.vision_config.as_ref().ok_or_else(|| {
            InferenceError::InvalidInput(
                "checkpoint has no vision_config; cannot count image scaffold tokens".to_string(),
            )
        })?;
        let (_pixel_values, grid) =
            preprocess_qwen35_image(image_bytes, vision_cfg).map_err(|e| {
                InferenceError::InvalidInput(format!("image preprocessing failed: {e}"))
            })?;
        let merge_sq = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size;
        if merge_sq == 0 || !grid.num_patches().is_multiple_of(merge_sq) {
            return Err(InferenceError::InvalidInput(format!(
                "image grid {grid:?} patch count is not a multiple of spatial_merge_size^2"
            )));
        }
        let pads = grid.num_patches() / merge_sq;
        // vision_start + pads + vision_end + prompt tokens.
        Ok(2 + pads + self.tokenize_len(prompt))
    }

    /// Pool a text-only prompt through the same decoder + pooling path used
    /// for images, landing in the same vector space (see
    /// [`embed_text_vlm_f16`]).
    pub fn embed_text(
        &self,
        prompt: &str,
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>, InferenceError> {
        embed_text_vlm_f16(
            &self.weights,
            &self.config,
            &self.tokenizer,
            prompt,
            pooling,
        )
    }

    /// Pool an image into a single L2-normalized embedding vector, running
    /// the ViT forward pass on the CPU.
    pub fn embed_image(
        &self,
        image_bytes: &[u8],
        prompt: &str,
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>, InferenceError> {
        embed_image_from_bytes_f16(
            &self.weights,
            &self.config,
            &self.vision_weights,
            &self.tokenizer,
            image_bytes,
            prompt,
            pooling,
        )
    }

    /// Metal-dispatching sibling of [`Self::embed_image`]: runs the ViT
    /// forward pass on the Metal GPU. Returns
    /// [`InferenceError::UnsupportedModel`] when no Metal device is
    /// available (build-level or runtime) rather than silently falling back
    /// -- see [`Self::embed_image_best_effort`] for the fallback policy this
    /// module actually serves requests through.
    pub fn embed_image_metal(
        &self,
        image_bytes: &[u8],
        prompt: &str,
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>, InferenceError> {
        embed_image_from_bytes_f16_metal(
            &self.weights,
            &self.config,
            &self.vision_weights,
            &self.tokenizer,
            image_bytes,
            prompt,
            pooling,
        )
    }

    /// Serving policy for image items: try the Metal ViT dispatch first,
    /// falling back to the CPU ViT forward only when Metal is unavailable
    /// (build-level or runtime) rather than for every error -- a real
    /// inference failure (bad image, invalid assembled request) must still
    /// surface as such, not retry silently on the CPU path and mask it.
    pub fn embed_image_best_effort(
        &self,
        image_bytes: &[u8],
        prompt: &str,
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>, InferenceError> {
        match self.embed_image_metal(image_bytes, prompt, pooling) {
            Err(InferenceError::UnsupportedModel(_)) => {
                self.embed_image(image_bytes, prompt, pooling)
            }
            other => other,
        }
    }
}

/// Maps an [`InferenceError`] from the pooling machinery to the shared HTTP
/// error envelope: caller-supplied-input problems (bad image geometry, empty
/// prompt) become `HTTP 400`; everything else (including `UnsupportedModel`,
/// which can only reach here if a caller bypasses
/// [`EmbeddingModel::embed_image_best_effort`]) is a `HTTP 500`.
pub fn map_embedding_error(e: InferenceError) -> ApiError {
    match e {
        InferenceError::InvalidInput(message) => ApiError::BadRequest {
            message,
            code: "invalid_input",
        },
        other => {
            eprintln!("embedding error: {other:?}");
            ApiError::Internal {
                message: "inference failed".to_string(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Wire contract
// ---------------------------------------------------------------------------

/// `POST /v1/embeddings` request body.
#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    /// Requested model identifier; accepted and echoed back, not validated
    /// against the loaded checkpoint (this server serves exactly one model).
    #[serde(default)]
    pub model: Option<String>,
    /// One item, or an array of items.
    pub input: EmbeddingsInput,
    /// `"mean_visual"` (default) or `"last_token"`. See [`parse_pooling`].
    #[serde(default)]
    pub pooling: Option<String>,
}

/// `input`: either a single item or an array of items, exactly like OpenAI's
/// `embeddings` endpoint accepts a single string or an array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    One(EmbeddingInputItem),
    Many(Vec<EmbeddingInputItem>),
}

impl EmbeddingsInput {
    pub fn into_items(self) -> Vec<EmbeddingInputItem> {
        match self {
            EmbeddingsInput::One(item) => vec![item],
            EmbeddingsInput::Many(items) => items,
        }
    }
}

/// One `input` array element: a plain string (text embedding) or an
/// `{"type": "image_url", "image_url": {"url": "..."}}` object -- the same
/// inline data-URI shape the chat contract's `ContentPart::ImageUrl` accepts
/// (see [`super::contract`]).
#[derive(Debug)]
pub enum EmbeddingInputItem {
    Text(String),
    Image { url: String },
}

impl<'de> Deserialize<'de> for EmbeddingInputItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawImageUrl {
            url: String,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Text(String),
            Object {
                #[serde(rename = "type")]
                kind: String,
                #[serde(default)]
                image_url: Option<RawImageUrl>,
            },
        }

        match Raw::deserialize(deserializer)? {
            Raw::Text(text) => Ok(EmbeddingInputItem::Text(text)),
            Raw::Object { kind, image_url } => {
                if kind != "image_url" {
                    return Err(serde::de::Error::custom(format!(
                        "input item type '{kind}' is not supported; only plain strings and \
                         'image_url' objects are accepted"
                    )));
                }
                let image_url = image_url.ok_or_else(|| {
                    serde::de::Error::custom(
                        "image_url input item must include object field 'image_url'",
                    )
                })?;
                Ok(EmbeddingInputItem::Image { url: image_url.url })
            }
        }
    }
}

/// One input item after decode/validation: text kept as-is, an image
/// data-URI already base64-decoded to raw file bytes via
/// [`decode_inline_image`] (the exact same parser and caps the chat contract
/// uses -- remote `http(s)` URLs and malformed data URIs are rejected there).
#[derive(Debug)]
pub enum NormalizedEmbeddingItem {
    Text(String),
    Image(Vec<u8>),
}

/// Parses the wire `pooling` field, defaulting to `mean_visual`
/// ([`PoolingStrategy::MeanVisualTokens`], which degrades to a plain mean
/// over decoder positions for a text-only item -- see [`PoolingStrategy`]'s
/// own doc comment).
///
/// # Errors
///
/// Returns [`ApiError::BadRequest`] (`invalid_pooling`) for any value other
/// than `"mean_visual"` or `"last_token"`.
pub fn parse_pooling(value: Option<&str>) -> Result<PoolingStrategy, ApiError> {
    match value {
        None | Some("mean_visual") => Ok(PoolingStrategy::MeanVisualTokens),
        Some("last_token") => Ok(PoolingStrategy::LastToken),
        Some(other) => Err(ApiError::BadRequest {
            message: format!("pooling must be 'mean_visual' or 'last_token', got '{other}'"),
            code: "invalid_pooling",
        }),
    }
}
/// RFC 2397 shape: `data:[<mediatype>][;base64],<data>`. The header before
/// the first comma never contains whitespace, which is what separates a data
/// URI from prose that happens to open with the word "data:" (a log line, a
/// label), so only the former is refused as a text item.
fn is_data_uri(text: &str) -> bool {
    text.strip_prefix("data:")
        .and_then(|rest| rest.split_once(','))
        .is_some_and(|(header, _)| !header.chars().any(char::is_whitespace))
}

/// Validates and decodes every `input` item, in order.
///
/// # Errors
///
/// Returns [`ApiError::BadRequest`] (`invalid_input`) if `items` is empty or
/// exceeds [`MAX_EMBEDDING_INPUT_COUNT`]; [`ApiError::BadRequest`]
/// (`data_uri_as_text_item`) for a plain-string item that is shaped like a
/// data URI (see `is_data_uri`) -- a data URI is only ever meaningful as an
/// `image_url` object (see [`EmbeddingInputItem`]'s own doc comment), so one
/// arriving as a text item is a caller passing the wrong shape, not text to
/// embed; propagates [`decode_inline_image`]'s error for a malformed or
/// rejected image item.
pub fn normalize_embedding_items(
    items: Vec<EmbeddingInputItem>,
) -> Result<Vec<NormalizedEmbeddingItem>, ApiError> {
    if items.is_empty() {
        return Err(ApiError::BadRequest {
            message: "input must not be empty".to_string(),
            code: "invalid_input",
        });
    }
    if items.len() > MAX_EMBEDDING_INPUT_COUNT {
        return Err(ApiError::BadRequest {
            message: format!(
                "input has {} items; maximum is {MAX_EMBEDDING_INPUT_COUNT}",
                items.len()
            ),
            code: "invalid_input",
        });
    }
    items
        .into_iter()
        .map(|item| match item {
            EmbeddingInputItem::Text(text) if is_data_uri(&text) => Err(ApiError::BadRequest {
                message: "input item is a plain string beginning with \"data:\"; to embed \
                              an image, pass {\"type\": \"image_url\", \"image_url\": \
                              {\"url\": \"...\"}}"
                    .to_string(),
                code: "data_uri_as_text_item",
            }),
            EmbeddingInputItem::Text(text) => Ok(NormalizedEmbeddingItem::Text(text)),
            EmbeddingInputItem::Image { url } => {
                decode_inline_image(&url).map(NormalizedEmbeddingItem::Image)
            }
        })
        .collect()
}

/// `POST /v1/embeddings` response body: OpenAI `embeddings` shape.
#[derive(Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingDatum>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingDatum {
    pub object: &'static str,
    pub index: usize,
    pub embedding: Vec<f32>,
}

/// Token accounting for the request. `prompt_tokens` counts the real
/// decoder-scaffold token count each item processes: a text item's real
/// tokenized length, or an image item's `vision_start` + pad tokens +
/// `vision_end` + prompt tokens (see
/// [`EmbeddingModel::image_scaffold_token_count`]) -- image cost is billed
/// in tokens of the decoder scaffold the pooled path actually runs through,
/// not a fabricated zero.
#[derive(Debug, Serialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

/// Rejects an item whose scaffold token count exceeds `max_context`, before
/// it reaches full-attention pooled prefill -- mirrors the chat serving
/// path's own context-window preflight
/// (`contract::validate_context_window_with_budget`), which runs before any
/// generation call for the same reason: an unbounded prompt is O(n^2)
/// attention work with no cap otherwise, since the 1 MiB request-body limit
/// alone permits roughly 250K characters of text.
///
/// # Errors
///
/// Returns [`ApiError::BadRequest`] (`context_length_exceeded`) naming the
/// item's index, its token count, and the limit.
fn check_item_fits_window(
    index: usize,
    token_count: usize,
    max_context: usize,
) -> Result<(), ApiError> {
    if token_count > max_context {
        return Err(ApiError::BadRequest {
            message: format!(
                "input item {index} has {token_count} scaffold tokens, exceeding the model's \
                 context window of {max_context} tokens"
            ),
            code: "context_length_exceeded",
        });
    }
    Ok(())
}

/// Runs every normalized item through `embedder`, in order, building the
/// response's `data` array with `index` matching each item's position in the
/// original `input` array (mixed text/image batches preserve input order).
///
/// # Errors
///
/// Returns [`ApiError::BadRequest`] (`context_length_exceeded`, via
/// `check_item_fits_window`) for the first item whose scaffold token count
/// exceeds the loaded model's context window. Otherwise returns the first
/// item's mapped [`ApiError`] (via [`map_embedding_error`]) on failure;
/// earlier items' work is discarded rather than partially returned; there is
/// no partial-batch response shape in this contract.
pub fn embed_items(
    embedder: &EmbeddingModel,
    items: Vec<NormalizedEmbeddingItem>,
    pooling: PoolingStrategy,
) -> Result<(Vec<EmbeddingDatum>, EmbeddingsUsage), ApiError> {
    let max_context = embedder.max_context();
    let mut data = Vec::with_capacity(items.len());
    let mut prompt_tokens = 0usize;
    for (index, item) in items.into_iter().enumerate() {
        let embedding = match &item {
            NormalizedEmbeddingItem::Text(text) => {
                // Single tokenize() call: admission uses the pre-truncation
                // count so an input the tokenizer would truncate is rejected
                // instead of silently embedded from a truncated prefix;
                // `usage.prompt_tokens` still reports the real (possibly
                // truncated) count the forward pass actually consumes --
                // see `tokenize_lengths`'s doc comment.
                let (admission_count, token_count) = embedder.tokenize_lengths(text);
                check_item_fits_window(index, admission_count, max_context)?;
                prompt_tokens += token_count;
                embedder.embed_text(text, pooling)
            }
            NormalizedEmbeddingItem::Image(bytes) => {
                let token_count = embedder
                    .image_scaffold_token_count(bytes, "")
                    .map_err(map_embedding_error)?;
                check_item_fits_window(index, token_count, max_context)?;
                prompt_tokens += token_count;
                embedder.embed_image_best_effort(bytes, "", pooling)
            }
        }
        .map_err(map_embedding_error)?;
        debug_assert!(
            {
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                (norm - 1.0).abs() < 1e-3 || embedding.iter().all(|x| *x == 0.0)
            },
            "pooled embedding must already be L2-normalized"
        );
        data.push(EmbeddingDatum {
            object: "embedding",
            index,
            embedding,
        });
    }
    Ok((
        data,
        EmbeddingsUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    ))
}

/// Maximum aggregate scaffold token count across every item in a single
/// pooled `/v1/embeddings` request (the `lattice` binary's vision-language
/// route), independent of each item's own `check_item_fits_window` cap.
/// Without this, [`MAX_EMBEDDING_INPUT_COUNT`] alone does not bound
/// [`embed_items`]'s aggregate work: a request at that item-count cap, with
/// every item near the checkpoint's own hard per-item ceiling
/// (`max_position_embeddings.min(8192)`, see [`EmbeddingModel::max_context`]),
/// would still run thousands of individual pooled-decoder forward passes
/// back to back inside one unbounded `spawn_blocking` job, with no
/// concurrency cap on top either (issue #1397).
///
/// Chosen as a bound on the worst case the route's own admission cap can
/// ever admit at once, rather than derived from a measured per-token memory
/// footprint the way the sibling `lattice_serve` route's
/// `MAX_EMBEDDINGS_TOTAL_TOKENS` is for `BertModel`'s packed-matmul buffers:
/// 4 concurrent jobs (`lattice`'s own `EMBEDDING_MAX_CONCURRENT_JOBS`) times
/// 8,192 (the hard per-item scaffold-token ceiling every item is already
/// capped to) bounds the case where the admission cap is entirely full of
/// maximal-length items. Numerically the same total as
/// `MAX_EMBEDDINGS_TOTAL_TOKENS`, by coincidence of both being 4 x 8,192 --
/// the two constants bound different compute shapes (packed batched-matmul
/// buffer size there, pooled per-item attention cost here) and are not
/// meant to stay equal if either changes.
pub const MAX_EMBEDDING_TOTAL_TOKENS: usize = 4 * 8_192;

/// Rejects a pooled `/v1/embeddings` request whose aggregate scaffold token
/// count (summed across every `input` item, after each item's own
/// `check_item_fits_window` admission) exceeds
/// [`MAX_EMBEDDING_TOTAL_TOKENS`] -- the pooled-route sibling of
/// [`check_embeddings_total_tokens`]. Computes the same per-item counts
/// [`embed_items`] does (tokenizing text, preprocessing images for their
/// scaffold length); `embed_items` recomputes these counts itself when it
/// actually runs, the same tokenize-twice trade `BertModel::encode_batch`
/// already makes for the sibling route.
///
/// The `lattice` binary's caller (`embeddings()` in `serve.rs`) deliberately
/// does NOT run this before its admission semaphore the way the sibling
/// route runs `check_embeddings_total_tokens` before ITS admission check:
/// this function's image branch calls [`EmbeddingModel::image_scaffold_token_count`],
/// which fully decodes and resizes the image, so running it pre-admission
/// would (a) let a request the admission cap should have refused still pay
/// for that decode, defeating the cap's purpose of bounding aggregate work,
/// and (b) put CPU-heavy work back on the async executor instead of inside
/// `spawn_blocking`. The caller therefore acquires its admission permit
/// FIRST and calls this function from inside the admitted `spawn_blocking`
/// job, before [`embed_items`]. The resulting caller-visible priority is
/// **busy (503) before budget (400)** -- the reverse of the sibling route's
/// own within-request order (which checks budget with no admission cap to
/// order it against in the first place) -- and this is NOT sibling-order
/// parity for that reason; it is a different, deliberate tradeoff for a
/// route whose per-item counting is expensive enough to need admission
/// first.
///
/// Runs each item's window check in `input` order, mirroring
/// [`embed_items`]'s own order, so a single oversized item is reported as
/// `context_length_exceeded` naming that item rather than the coarser
/// aggregate `batch_token_budget_exceeded`.
///
/// # Errors
///
/// Returns [`ApiError::BadRequest`] (`context_length_exceeded`), via
/// `check_item_fits_window`, for the first item whose scaffold token count
/// exceeds the loaded model's context window. Otherwise returns
/// [`ApiError::BadRequest`] (`batch_token_budget_exceeded`) naming the
/// request's total token count and the limit; propagates
/// [`map_embedding_error`]'s mapped error if an image item's scaffold token
/// count cannot be computed.
pub fn check_pooled_embedding_total_tokens(
    embedder: &EmbeddingModel,
    items: &[NormalizedEmbeddingItem],
) -> Result<(), ApiError> {
    let max_context = embedder.max_context();
    let mut total_tokens = 0usize;
    for (index, item) in items.iter().enumerate() {
        let (admission_count, real_count) = match item {
            NormalizedEmbeddingItem::Text(text) => embedder.tokenize_lengths(text),
            NormalizedEmbeddingItem::Image(bytes) => {
                let count = embedder
                    .image_scaffold_token_count(bytes, "")
                    .map_err(map_embedding_error)?;
                (count, count)
            }
        };
        check_item_fits_window(index, admission_count, max_context)?;
        total_tokens += real_count;
    }
    if total_tokens > MAX_EMBEDDING_TOTAL_TOKENS {
        return Err(ApiError::BadRequest {
            message: format!(
                "input has {total_tokens} total scaffold tokens across all items; maximum is \
                 {MAX_EMBEDDING_TOTAL_TOKENS}"
            ),
            code: "batch_token_budget_exceeded",
        });
    }
    Ok(())
}

/// Tiny end-to-end fixture: same shape as `pooled_embed`'s own unit tests
/// (one full-attention layer, 8-dim hidden, 8x8 synthetic PNG) so the whole
/// decode + pooling path is exercised without a real checkpoint. Duplicated
/// here rather than exposed from `pooled_embed` because that module's test
/// fixtures are private -- this is loader glue and wire-contract testing,
/// not a change to its pooling math. `pub` (behind `test-utils`) so the
/// `lattice`/`lattice_serve` binaries' own router-level tests can build a
/// real, working `EmbeddingModel` too -- bin targets cannot see this crate's
/// private `#[cfg(test)]` items across the bin/lib compilation boundary.
#[cfg(any(test, feature = "test-utils"))]
pub mod test_support {
    use super::EmbeddingModel;
    use crate::model::qwen35_config::{LayerType, Qwen35Config, RopeParams, VisionModelConfig};
    use crate::tokenizer::bpe::BpeTokenizer;
    use crate::vision::checkpoint::{Qwen35VisionWeights, VisualBlockWeights, VisualMergerWeights};
    use crate::weights::f16_weights::{
        F16AttentionWeights, F16CommonLayerWeights, F16FeedForwardWeights,
        F16FullAttentionLayerWeights, F16ModelWeights,
    };

    fn pseudo_random_fill(seed: u32, n: usize) -> Vec<f32> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 0.2 - 0.1
        };
        (0..n).map(|_| next()).collect()
    }

    fn tiny_vision_cfg() -> VisionModelConfig {
        VisionModelConfig {
            depth: 1,
            hidden_size: 8,
            num_heads: 2,
            patch_size: 2,
            spatial_merge_size: 2,
            out_hidden_size: 8,
            temporal_patch_size: 1,
            num_position_embeddings: 16,
            in_channels: 3,
            deepstack_visual_indexes: vec![],
            intermediate_size: None,
        }
    }

    fn tiny_vision_weights(vision_cfg: &VisionModelConfig, seed: u32) -> Qwen35VisionWeights {
        let hidden = vision_cfg.hidden_size;
        let patch_len = vision_cfg.in_channels
            * vision_cfg.temporal_patch_size
            * vision_cfg.patch_size
            * vision_cfg.patch_size;
        let mlp_dim = 2 * hidden;
        let merge_in = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size * hidden;

        let block = VisualBlockWeights {
            qkv_weight: pseudo_random_fill(seed, 3 * hidden * hidden),
            qkv_bias: pseudo_random_fill(seed.wrapping_add(1), 3 * hidden),
            proj_weight: pseudo_random_fill(seed.wrapping_add(2), hidden * hidden),
            proj_bias: pseudo_random_fill(seed.wrapping_add(3), hidden),
            fc1_weight: pseudo_random_fill(seed.wrapping_add(4), mlp_dim * hidden),
            fc1_bias: pseudo_random_fill(seed.wrapping_add(5), mlp_dim),
            fc2_weight: pseudo_random_fill(seed.wrapping_add(6), hidden * mlp_dim),
            fc2_bias: pseudo_random_fill(seed.wrapping_add(7), hidden),
            norm1_weight: vec![1.0; hidden],
            norm1_bias: vec![0.0; hidden],
            norm2_weight: vec![1.0; hidden],
            norm2_bias: vec![0.0; hidden],
        };

        Qwen35VisionWeights {
            patch_embed_weight: pseudo_random_fill(seed.wrapping_add(8), hidden * patch_len),
            patch_embed_weight_shape: vec![
                hidden,
                vision_cfg.in_channels,
                vision_cfg.temporal_patch_size,
                vision_cfg.patch_size,
                vision_cfg.patch_size,
            ],
            patch_embed_bias: pseudo_random_fill(seed.wrapping_add(9), hidden),
            pos_embed: pseudo_random_fill(
                seed.wrapping_add(10),
                vision_cfg.num_position_embeddings * hidden,
            ),
            blocks: vec![block],
            merger: VisualMergerWeights {
                fc1_weight: pseudo_random_fill(seed.wrapping_add(11), merge_in * merge_in),
                fc1_bias: pseudo_random_fill(seed.wrapping_add(12), merge_in),
                fc2_weight: pseudo_random_fill(
                    seed.wrapping_add(13),
                    vision_cfg.out_hidden_size * merge_in,
                ),
                fc2_bias: pseudo_random_fill(seed.wrapping_add(14), vision_cfg.out_hidden_size),
                norm_weight: vec![1.0; hidden],
                norm_bias: vec![0.0; hidden],
            },
        }
    }

    /// A working, deterministic `EmbeddingModel` built from in-memory
    /// components -- no checkpoint directory needed. Text prompts must use
    /// single-character words from `{"a", "b", "c"}` (see the comment at
    /// this function's tokenizer construction below for why).
    pub fn tiny_embedding_model() -> EmbeddingModel {
        let hidden = 8usize;
        let vocab = 16usize;
        let vision_cfg = tiny_vision_cfg();

        let config = Qwen35Config {
            hidden_size: hidden,
            num_hidden_layers: 1,
            vocab_size: vocab,
            intermediate_size: 4,
            rms_norm_eps: 1e-6,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: hidden,
            rope_theta: 1.0e7,
            partial_rotary_factor: 1.0,
            rope_parameters: Some(RopeParams {
                rope_theta: 1.0e7,
                partial_rotary_factor: Some(1.0),
                mrope_section: Some(vec![2, 1, 1]),
                mrope_interleaved: Some(true),
            }),
            linear_num_key_heads: 2,
            linear_num_value_heads: Some(2),
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            linear_conv_kernel_dim: 4,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            full_attention_interval: 1,
            layer_types: vec![LayerType::FullAttention],
            layer_mask: vec![true],
            eos_token_id: 999,
            max_position_embeddings: 512,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
            vision_config: Some(vision_cfg.clone()),
            image_token_id: Some(9),
            video_token_id: None,
            vision_start_token_id: Some(10),
            vision_end_token_id: Some(11),
        };

        let to_f16 = |src: &[f32]| -> Vec<u16> {
            let mut dst = vec![0u16; src.len()];
            crate::weights::f16_weights::f32_to_f16_slice(src, &mut dst);
            dst
        };
        let q_dim = config.full_q_dim();
        let kv_dim = config.full_kv_dim();
        let full_weights = F16FullAttentionLayerWeights {
            q_proj: to_f16(&pseudo_random_fill(101, 2 * q_dim * hidden)),
            k_proj: to_f16(&pseudo_random_fill(102, kv_dim * hidden)),
            v_proj: to_f16(&pseudo_random_fill(103, kv_dim * hidden)),
            o_proj: to_f16(&pseudo_random_fill(104, hidden * q_dim)),
            q_norm: vec![0.0f32; hidden],
            k_norm: vec![0.0f32; hidden],
        };
        let common = F16CommonLayerWeights {
            input_layernorm: vec![0.0f32; hidden],
            post_attention_layernorm: vec![0.0f32; hidden],
            ffn: F16FeedForwardWeights::Dense {
                gate_proj: to_f16(&vec![0.0f32; 4 * hidden]),
                up_proj: to_f16(&vec![0.0f32; 4 * hidden]),
                down_proj: to_f16(&vec![0.0f32; hidden * 4]),
            },
        };
        let weights = F16ModelWeights {
            embed_tokens: to_f16(&pseudo_random_fill(777, vocab * hidden)),
            final_norm: vec![0.0f32; hidden],
            layers: vec![(F16AttentionWeights::Full(full_weights), common)],
        };

        let vision_weights = tiny_vision_weights(&vision_cfg, 555);
        // Single-character vocab entries, no merges: matches the pattern
        // `cpu_f16.rs`'s own `embed_text_vlm_f16` unit tests use (a
        // whole-word vocab entry does not tokenize through the general
        // `Tokenizer::tokenize` byte-level pretokenizer without merge rules
        // to rebuild it from bytes; a single already-atomic character does).
        let mut vocab_map = std::collections::HashMap::new();
        for (i, c) in ["a", "b", "c"].iter().enumerate() {
            vocab_map.insert((*c).to_string(), i as u32);
        }
        let tokenizer =
            BpeTokenizer::from_vocab_and_merges(vocab_map, vec![]).expect("tokenizer constructs");

        EmbeddingModel::new(weights, config, vision_weights, tokenizer)
    }

    /// An 8x8 synthetic PNG data URI, varied by `seed` so two calls produce
    /// distinguishable images.
    pub fn tiny_png_data_uri(seed: u8) -> String {
        use base64::Engine as _;
        use image::RgbImage;
        let mut img = RgbImage::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                let v = ((x + y + seed as u32) % 256) as u8;
                img.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&buf)
        )
    }
}

// ---------------------------------------------------------------------------
// `lattice_serve` text-only embeddings (issue #584)
// ---------------------------------------------------------------------------

/// Maximum number of input texts accepted in a single `/v1/embeddings`
/// request. Mirrors OpenAI's own documented cap ("any array must be 2048
/// dimensions or less") so a client that already respects the real API's
/// limit never trips this one; it also bounds the CPU time and memory a
/// single request can force onto this server's `encode_batch` call before
/// any inference happens.
pub const MAX_EMBEDDINGS_BATCH_SIZE: usize = 2048;

/// Maximum sum of real token counts across every item in a single
/// `/v1/embeddings` request, independent of [`MAX_EMBEDDINGS_BATCH_SIZE`]'s
/// per-request item-count cap. `BertModel`'s packed batched-encode pipeline
/// (`encode_batch_packed` / `forward_batch` in
/// `crates/inference/src/model/bert.rs`) allocates its per-call activation
/// buffers (`hidden`, `q`, `k`, `v`, `qkv`, `concat`, `attn_out`, `temp`,
/// each sized `total_tokens * hidden_size`, plus `ffn_intermediate` sized
/// `total_tokens * intermediate_size`) proportional to the *sum* of every
/// item's real token count, not the item count -- so `MAX_EMBEDDINGS_BATCH_SIZE`
/// alone does not bound this allocation: a request at the 2048-item cap
/// with long inputs can reach roughly a million aggregate tokens (2048
/// items times a representative 512-token context window), which at a
/// representative embedding-BERT scale (`hidden_size=768`,
/// `intermediate_size=3072`: `9 * hidden_size + intermediate_size = 9,984`
/// f32s of buffer per aggregate token) is tens of gigabytes of activation
/// memory for one request. `32,768` keeps that same calculation to roughly
/// 1.3 GB (`32,768 * 9,984 * 4` bytes) at that representative scale, while
/// still admitting large batches of short texts.
pub const MAX_EMBEDDINGS_TOTAL_TOKENS: usize = 32_768;

/// Rejects a request whose aggregate token count (summed across every
/// `input` item) exceeds [`MAX_EMBEDDINGS_TOTAL_TOKENS`]. See that
/// constant's doc comment for the memory-allocation rationale.
///
/// # Errors
///
/// Returns [`ApiError::BadRequest`] (`batch_token_budget_exceeded`) naming
/// the request's total token count and the limit.
pub fn check_embeddings_total_tokens(total_tokens: usize) -> Result<(), ApiError> {
    if total_tokens > MAX_EMBEDDINGS_TOTAL_TOKENS {
        return Err(ApiError::BadRequest {
            message: format!(
                "input has {total_tokens} total tokens across all items; maximum is \
                 {MAX_EMBEDDINGS_TOTAL_TOKENS}"
            ),
            code: "batch_token_budget_exceeded",
        });
    }
    Ok(())
}

/// Rejects a single `input` item whose pre-truncation (original) token count
/// exceeds the loaded BERT model's `max_position_embeddings`, before the
/// request reaches `BertModel::encode`/`encode_batch`.
///
/// Both `BertModel` encode paths silently diverge on an over-length item
/// instead of raising a caller-visible error: the batched path
/// (`encode_batch_packed`, `crates/inference/src/model/bert.rs`) clamps each
/// item's length to `max_position_embeddings`, returning HTTP 200 for a
/// vector computed from silently truncated text; the single-item path
/// (`encode`) does not clamp before slicing its tokenizer output, and can
/// panic in the `spawn_blocking` task instead. Calling this for every item
/// before either path runs makes both cases the same caller-visible error
/// instead of two different failure modes for the same oversized input.
///
/// # Errors
///
/// Returns [`ApiError::BadRequest`] (`context_length_exceeded`) naming the
/// item's index, its token count, and the limit.
pub fn check_embeddings_item_fits_window(
    index: usize,
    token_count: usize,
    max_position_embeddings: usize,
) -> Result<(), ApiError> {
    if token_count > max_position_embeddings {
        return Err(ApiError::BadRequest {
            message: format!(
                "input item {index} has {token_count} tokens, exceeding the loaded model's \
                 context window of {max_position_embeddings} tokens"
            ),
            code: "context_length_exceeded",
        });
    }
    Ok(())
}

/// OpenAI `input` field: a single string, or an array of strings.
///
/// `#[serde(untagged)]` tries each variant in order, matching how OpenAI
/// clients actually serialize this field (a bare JSON string or a bare JSON
/// array, never a wrapper object).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TextEmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

/// Wire request body for `POST /v1/embeddings`.
///
/// `deny_unknown_fields`: a field this route does not implement must be
/// rejected, not silently dropped -- see [`check_embeddings_unsupported_fields`]'s
/// doc comment for the caller-visible-contract reasoning. `dimensions`,
/// `encoding_format`, and `user` are declared explicitly (rather than left to
/// fall through to the unknown-field rejection) so each gets its own
/// specific error message instead of a generic "unknown field" one; `user`
/// is caller-side metadata this route accepts and ignores, matching OpenAI's
/// own semantics for it.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextEmbeddingsRequest {
    /// Text(s) to embed. `None` when the field is omitted or explicitly
    /// `null` — validated by [`parse_embeddings_input`].
    #[serde(default)]
    pub input: Option<TextEmbeddingsInput>,
    /// Requested model identifier. `None` when omitted; `Some("")` when the
    /// client sends an explicit empty string — both are validated by
    /// [`check_embeddings_model`], matching `contract::validate_model_name`'s
    /// `OptionalExact` policy (an explicit empty string is NOT treated as
    /// absent).
    #[serde(default)]
    pub model: Option<String>,
    /// OpenAI's Matryoshka output-dimension override. Not implemented by
    /// `BertModel` (fixed-width output) -- captured as a raw
    /// [`serde_json::Value`] (rather than e.g. `Option<u32>`) purely to
    /// detect presence; any non-`null` value is rejected by
    /// [`check_embeddings_unsupported_fields`] regardless of its shape.
    #[serde(default)]
    pub dimensions: Option<Value>,
    /// OpenAI's output encoding selector. This route always returns JSON
    /// floats, so only an absent field or the literal `"float"` is a
    /// genuine no-op; anything else (starting with `"base64"`) is rejected
    /// by [`check_embeddings_unsupported_fields`].
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// OpenAI's caller-side tracing identifier. Never read by this route;
    /// declared so it deserializes instead of tripping `deny_unknown_fields`.
    #[serde(default)]
    pub user: Option<Value>,
}

/// Rejects a request that sets a field this route does not implement, so
/// the caller learns its requested output shape was not honored instead of
/// silently receiving a response that does not match what it asked for
/// (Matryoshka `dimensions`, or a `base64` `encoding_format`). Genuinely
/// unrecognized fields are already rejected by
/// [`TextEmbeddingsRequest`]'s `deny_unknown_fields`, at parse time, before
/// this runs; this covers the two fields the wire type accepts syntactically
/// but this route cannot actually honor.
///
/// # Errors
///
/// Returns [`ApiError::BadRequest`] (`unsupported_parameter`) naming the
/// offending field.
pub fn check_embeddings_unsupported_fields(req: &TextEmbeddingsRequest) -> Result<(), ApiError> {
    if req.dimensions.as_ref().is_some_and(|v| !v.is_null()) {
        return Err(ApiError::BadRequest {
            message: "dimensions is not supported by this server; the loaded model's full \
                      output width is always returned"
                .to_string(),
            code: "unsupported_parameter",
        });
    }
    if let Some(format) = req.encoding_format.as_deref()
        && format != "float"
    {
        return Err(ApiError::BadRequest {
            message: format!(
                "encoding_format '{format}' is not supported by this server; only 'float' is \
                 implemented"
            ),
            code: "unsupported_parameter",
        });
    }
    Ok(())
}

/// Extracts and validates the input texts from a parsed [`TextEmbeddingsRequest`].
///
/// Order is preserved exactly: index `i` of the returned `Vec` is index `i`
/// of the caller's array (or the sole element for a single string).
///
/// # Errors
///
/// - `input` missing or `null` — [`ApiError::BadRequest`] `invalid_request`.
/// - `input` is an empty string, or an empty array — `invalid_input`.
/// - any element is an empty string (including the single-string case) —
///   `invalid_input`, naming the offending index.
/// - the array has more than [`MAX_EMBEDDINGS_BATCH_SIZE`] elements —
///   `batch_size_exceeds_limit`.
pub fn parse_embeddings_input(
    input: &Option<TextEmbeddingsInput>,
) -> Result<Vec<String>, ApiError> {
    let texts = match input {
        None => {
            return Err(ApiError::BadRequest {
                message: "input is required".to_string(),
                code: "invalid_request",
            });
        }
        Some(TextEmbeddingsInput::One(text)) => vec![text.clone()],
        Some(TextEmbeddingsInput::Many(texts)) => texts.clone(),
    };
    if texts.is_empty() {
        return Err(ApiError::BadRequest {
            message: "input must not be empty".to_string(),
            code: "invalid_input",
        });
    }
    if texts.len() > MAX_EMBEDDINGS_BATCH_SIZE {
        return Err(ApiError::BadRequest {
            message: format!(
                "input has {} elements; maximum is {MAX_EMBEDDINGS_BATCH_SIZE}",
                texts.len()
            ),
            code: "batch_size_exceeds_limit",
        });
    }
    if let Some(index) = texts.iter().position(String::is_empty) {
        return Err(ApiError::BadRequest {
            message: format!("input[{index}] must not be empty"),
            code: "invalid_input",
        });
    }
    Ok(texts)
}

/// Validates a requested `model` against the single model this server has
/// loaded, using the same `OptionalExact` policy
/// `contract::ServeProfile::lattice_serve` applies to chat completions: an
/// omitted `model` is accepted; a present `model` (including an explicit
/// empty string) must equal `served`.
///
/// # Errors
///
/// Returns [`ApiError::BadRequest`] `model_not_found` on a mismatch.
pub fn check_embeddings_model(requested: Option<&str>, served: &str) -> Result<(), ApiError> {
    match requested {
        None => Ok(()),
        Some(requested) if requested == served => Ok(()),
        Some(requested) => Err(ApiError::BadRequest {
            message: format!("model '{requested}' is not loaded; this server serves '{served}'"),
            code: "model_not_found",
        }),
    }
}

/// Builds the `POST /v1/embeddings` success response body, OpenAI's
/// `CreateEmbeddingResponse` shape (verified against the `openai-python` SDK's
/// `src/openai/types/create_embedding_response.py` and
/// `src/openai/types/embedding.py`, since `platform.openai.com`'s own API
/// reference is not fetchable from this environment):
///
/// ```text
/// { object: "list", data: [{ object: "embedding", embedding: [f32], index }],
///   model, usage: { prompt_tokens, total_tokens } }
/// ```
///
/// `index` is assigned from each embedding's position in `embeddings`, which
/// must already be in the caller's original input order — this function
/// does not reorder anything.
///
/// `prompt_tokens` and `total_tokens` are equal: an embeddings request has no
/// completion tokens, matching OpenAI's own `Usage` shape for this endpoint.
pub fn build_embeddings_response(
    model: &str,
    embeddings: &[Vec<f32>],
    prompt_tokens: u64,
) -> Value {
    let data: Vec<Value> = embeddings
        .iter()
        .enumerate()
        .map(|(index, embedding)| {
            json!({
                "object": "embedding",
                "embedding": embedding,
                "index": index,
            })
        })
        .collect();
    json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    })
}

#[cfg(test)]
mod tests;

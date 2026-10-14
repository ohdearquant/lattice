//! EmbeddingGemma 2 text encoder, CPU f32 path.
//!
//! A 24-layer bidirectional encoder in the Gemma family. Compared with the
//! Gemma 4 text decoder it reuses the per-layer input gate, unscaled value
//! norm, `layer_scalar`, attention scaling of 1.0 and mixed head widths. What
//! differs is the attention mask (bidirectional on every layer, with a
//! two-sided window on sliding layers), a projection-only per-layer signal,
//! a per-token `hidden -> embedding_dim` projection, mean pooling over every
//! token (the task prefix included) and L2 normalization.
//!
//! Everything runs in f32 from bf16 checkpoint weights. f16 activations are
//! not supported: the model card reports that they overflow.
//!
//! Text calls load only the text tower. Vision weights have a separate deferred
//! loader; audio tensors are never read.

use super::embeddinggemma2_config::{
    EmbeddingGemma2Config, EmbeddingGemma2LayerKind, ImagePartsConfig, image_parts_config,
};
use super::embeddinggemma2_image::{EmbeddingGemma2Rgb8Image, PreparedImage, prepare_image};
use super::embeddinggemma2_vision::{LazyVision, VisionStages, VisionWeights};
use super::gemma4_ops::{
    gemma4_apply_rope, gemma4_geglu_mlp, gemma4_gelu_tanh, gemma4_qk_norm_v_unscaled,
    gemma4_rms_norm, gemma4_rope_cos_sin, gemma4_rope_inv_freq,
};
use crate::error::InferenceError;
use crate::forward::cpu::{elementwise_mul, matmul_bt, matmul_into};
use crate::forward::metal_embeddinggemma2::{MetalEmbeddingGemma2State, VisionTimings};
use crate::model::config_file::read_config_json_bounded;
use crate::tokenizer::common::{EmbeddingTokenCounts, Tokenizer};
use crate::tokenizer::gemma_bpe::GemmaBpeTokenizer;
use crate::weights::{SafetensorsFile, TensorSource};
use std::path::Path;
use std::sync::Arc;

/// Default token limit applied by [`EmbeddingGemma2Model::tokenize`]. Longer text is truncated,
/// never rejected. The model itself has no length limit; see
/// [`EmbeddingGemma2Model::with_max_tokens`].
pub const EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS: usize = 8192;

/// Query rows processed per attention tile.
const ATTN_QUERY_TILE: usize = 128;

/// Key prefixes under which the text tower's tensors may sit, tried in order.
const TENSOR_PREFIXES: [&str; 4] = ["language_model.", "model.language_model.", "model.", ""];

// ---------------------------------------------------------------------------
// Task prompts
// ---------------------------------------------------------------------------

/// The instruction prefixes the model was trained with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingGemma2Task {
    /// A search query.
    Query,
    /// A document with no title. Use [`format_titled_document`] when a title is known.
    Document,
    /// A natural-language query against code.
    CodeRetrieval,
    /// Sentence similarity and paraphrase detection.
    SentenceSimilarity,
    /// Classification features.
    Classification,
    /// Clustering features.
    Clustering,
    /// A question to be answered from documents.
    QuestionAnswering,
    /// A claim to be checked against documents.
    FactChecking,
}

impl EmbeddingGemma2Task {
    /// The prefix this task prepends to the text.
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Query => "task: search result | query: ",
            Self::Document => "title: none | text: ",
            Self::CodeRetrieval => "task: code retrieval | query: ",
            Self::SentenceSimilarity => "task: sentence similarity | query: ",
            Self::Classification => "task: classification | query: ",
            Self::Clustering => "task: clustering | query: ",
            Self::QuestionAnswering => "task: question answering | query: ",
            Self::FactChecking => "task: fact checking | query: ",
        }
    }

    /// `text` with this task's prefix prepended.
    pub fn format(self, text: &str) -> String {
        format!("{}{text}", self.prefix())
    }
}

/// A document with a title, in the form the model was trained on.
pub fn format_titled_document(title: &str, content: &str) -> String {
    format!("title: {title} | text: {content}")
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

struct TextTokenizer {
    inner: GemmaBpeTokenizer,
    bos: u32,
    eos: u32,
}

impl TextTokenizer {
    fn from_tokenizer_json_str(text: &str, bos: u32, eos: u32) -> Result<Self, InferenceError> {
        // The sequence limit is applied in `tokenize`, where the two wrapping tokens are counted.
        let inner = GemmaBpeTokenizer::from_tokenizer_json_str(text)?.with_max_seq_len(usize::MAX);
        Ok(Self { inner, bos, eos })
    }

    /// `[bos] text [eos]`. With `max_tokens = Some(n)` the text is cut so that the whole sequence
    /// is at most `n` tokens, keeping both wrapping tokens.
    fn tokenize(&self, text: &str, max_tokens: Option<usize>) -> Vec<u32> {
        self.tokenize_counted(text, max_tokens).0
    }

    /// As [`TextTokenizer::tokenize`], also returning the length the sequence had before
    /// `max_tokens` was applied. This is the one tokenization body.
    fn tokenize_counted(&self, text: &str, max_tokens: Option<usize>) -> (Vec<u32>, usize) {
        let encoded = self.inner.tokenize_batch(&[text]).pop();
        let (ids, len) = match &encoded {
            Some(e) => (e.input_ids.as_slice(), e.real_length),
            None => (&[][..], 0),
        };
        let keep = max_tokens.map_or(len, |n| len.min(n.saturating_sub(2)));
        (self.wrap(&ids[..keep]), len + 2)
    }

    fn wrap(&self, body: &[u32]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(body.len() + 2);
        ids.push(self.bos);
        ids.extend_from_slice(body);
        ids.push(self.eos);
        ids
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

pub(crate) struct LayerWeights {
    pub(crate) input_layernorm: Vec<f32>,
    pub(crate) post_attention_layernorm: Vec<f32>,
    pub(crate) pre_feedforward_layernorm: Vec<f32>,
    pub(crate) post_feedforward_layernorm: Vec<f32>,
    pub(crate) q_proj: Vec<f32>,
    pub(crate) k_proj: Vec<f32>,
    pub(crate) v_proj: Vec<f32>,
    pub(crate) o_proj: Vec<f32>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) gate_proj: Vec<f32>,
    pub(crate) up_proj: Vec<f32>,
    pub(crate) down_proj: Vec<f32>,
    pub(crate) per_layer_input_gate: Vec<f32>,
    pub(crate) per_layer_projection: Vec<f32>,
    pub(crate) post_per_layer_input_norm: Vec<f32>,
    pub(crate) layer_scalar: f32,
}

pub(crate) struct Weights {
    pub(crate) embed_tokens: Vec<f32>,
    pub(crate) per_layer_model_projection: Vec<f32>,
    pub(crate) per_layer_projection_norm: Vec<f32>,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) norm: Vec<f32>,
    pub(crate) embedding_projection: Vec<f32>,
}

fn load_tensor<T: TensorSource + ?Sized>(
    source: &mut T,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, InferenceError> {
    if let Some(dtype) = source.tensor_dtype(name)?
        && dtype != "BF16"
    {
        return Err(InferenceError::Inference(format!(
            "embeddinggemma2 loading: tensor {name} has dtype {dtype:?}, expected \"BF16\""
        )));
    }
    if let Some(declared) = source.tensor_shape(name)?
        && declared != expected
    {
        return Err(InferenceError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: declared,
        });
    }
    let (data, shape) = source.get_f32_tensor_owned(name)?;
    if shape != expected {
        return Err(InferenceError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: shape,
        });
    }
    Ok(data)
}

/// The tensor-name prefix under which `embed_tokens.weight` has the text tower's shape.
fn detect_prefix<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &EmbeddingGemma2Config,
) -> Result<&'static str, InferenceError> {
    let expected = [cfg.vocab_size, cfg.hidden_size];
    for prefix in TENSOR_PREFIXES {
        let name = format!("{prefix}embed_tokens.weight");
        if source.tensor_shape(&name)?.as_deref() == Some(&expected[..]) {
            return Ok(prefix);
        }
    }
    Err(InferenceError::MissingTensor(format!(
        "embed_tokens.weight {expected:?} under any of the prefixes {TENSOR_PREFIXES:?}"
    )))
}

fn load_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &EmbeddingGemma2Config,
) -> Result<Weights, InferenceError> {
    let p = detect_prefix(source, cfg)?;
    let hidden = cfg.hidden_size;
    let per_layer = cfg.hidden_size_per_layer_input;
    let heads = cfg.num_attention_heads;

    let embed_tokens = load_tensor(
        source,
        &format!("{p}embed_tokens.weight"),
        &[cfg.vocab_size, hidden],
    )?;
    let per_layer_model_projection = load_tensor(
        source,
        &format!("{p}ple.per_layer_model_projection.weight"),
        &[cfg.num_hidden_layers * per_layer, hidden],
    )?;
    let per_layer_projection_norm = load_tensor(
        source,
        &format!("{p}ple.per_layer_projection_norm.weight"),
        &[per_layer],
    )?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for l in 0..cfg.num_hidden_layers {
        let shape = cfg.layer_shapes[l];
        let q_dim = heads * shape.head_dim;
        let kv_dim = shape.num_key_value_heads * shape.head_dim;
        let ff = cfg.intermediate_size;
        let lp = format!("{p}layers.{l}.");
        let mut get = |suffix: &str, expected: &[usize]| {
            load_tensor(source, &format!("{lp}{suffix}"), expected)
        };
        let layer_scalar = get("layer_scalar", &[1])?[0];
        layers.push(LayerWeights {
            input_layernorm: get("input_layernorm.weight", &[hidden])?,
            post_attention_layernorm: get("post_attention_layernorm.weight", &[hidden])?,
            pre_feedforward_layernorm: get("pre_feedforward_layernorm.weight", &[hidden])?,
            post_feedforward_layernorm: get("post_feedforward_layernorm.weight", &[hidden])?,
            q_proj: get("self_attn.q_proj.weight", &[q_dim, hidden])?,
            k_proj: get("self_attn.k_proj.weight", &[kv_dim, hidden])?,
            v_proj: get("self_attn.v_proj.weight", &[kv_dim, hidden])?,
            o_proj: get("self_attn.o_proj.weight", &[hidden, q_dim])?,
            q_norm: get("self_attn.q_norm.weight", &[shape.head_dim])?,
            k_norm: get("self_attn.k_norm.weight", &[shape.head_dim])?,
            gate_proj: get("mlp.gate_proj.weight", &[ff, hidden])?,
            up_proj: get("mlp.up_proj.weight", &[ff, hidden])?,
            down_proj: get("mlp.down_proj.weight", &[hidden, ff])?,
            per_layer_input_gate: get(
                "ple_block.per_layer_input_gate.weight",
                &[per_layer, hidden],
            )?,
            per_layer_projection: get(
                "ple_block.per_layer_projection.weight",
                &[hidden, per_layer],
            )?,
            post_per_layer_input_norm: get(
                "ple_block.post_per_layer_input_norm.weight",
                &[hidden],
            )?,
            layer_scalar,
        });
    }

    let norm = load_tensor(source, &format!("{p}norm.weight"), &[hidden])?;
    let embedding_projection = load_tensor(
        source,
        &format!("{p}embedding_projection.weight"),
        &[cfg.embedding_dim, hidden],
    )?;
    Ok(Weights {
        embed_tokens,
        per_layer_model_projection,
        per_layer_projection_norm,
        layers,
        norm,
        embedding_projection,
    })
}

// ---------------------------------------------------------------------------
// Image parts
// ---------------------------------------------------------------------------

fn image_error(message: impl Into<String>) -> InferenceError {
    InferenceError::InvalidInput(format!("EmbeddingGemma 2 image: {}", message.into()))
}

fn no_image_layout() -> InferenceError {
    InferenceError::UnsupportedModel(
        "this EmbeddingGemma 2 checkpoint declares no image token layout".to_string(),
    )
}

/// One ordered entry of a multimodal [`EmbeddingGemma2Model::encode_parts_at_widths`] input:
/// either text or a decoded RGB8 image.
#[derive(Clone, Debug)]
pub enum EmbeddingGemma2Part {
    /// Literal text, tokenized with the model's beginning and end tokens.
    Text(String),
    /// A decoded RGB8 image.
    Image(EmbeddingGemma2Rgb8Image),
}

/// Counts reported alongside a multimodal embedding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddingGemma2PartCounts {
    /// Length of the combined token sequence.
    pub sequence_length: usize,
    /// Number of valid image patches across every image part.
    pub valid_patches: usize,
    /// Number of image soft rows across every image part.
    pub image_rows: usize,
}

/// Result of [`EmbeddingGemma2Model::encode_parts_at_widths`].
#[derive(Clone, Debug)]
pub struct EmbeddingGemma2PartsEncoding {
    /// One embedding per requested width, L2-normalized, in the requested order.
    pub embeddings: Vec<Vec<f32>>,
    /// The combined token ids actually run through the text encoder.
    pub input_ids: Vec<u32>,
    /// Layout counts for the combined sequence.
    pub counts: EmbeddingGemma2PartCounts,
    /// Image soft rows after the bridge, row-major `[image_rows, text_hidden]`.
    pub image_rows: Vec<f32>,
    /// Per-image stage tensors, in part order.
    pub image_stages: Vec<EmbeddingGemma2ImageStages>,
    /// Tokens by the native embedding width after the final norm and projection, before
    /// pooling, row-major `[sequence_length, embedding_dim]`.
    pub token_states: Vec<f32>,
    /// Mean over every valid row, before the per-width normalization.
    pub pre_normalization: Vec<f32>,
}

/// Stage tensors of one image part, at the boundaries a reference capture uses.
///
/// Widths come from the checkpoint's vision config; the values are the valid rows only,
/// so a comparison against a padded reference capture takes its valid prefix.
#[derive(Clone, Debug)]
pub struct EmbeddingGemma2ImageStages {
    /// Patch embedding: the centered, projected patches plus learned positions.
    pub patch: Vec<f32>,
    /// Tower state after encoder layers 0, 8 and 15.
    pub layer_0: Vec<f32>,
    /// See [`EmbeddingGemma2ImageStages::layer_0`].
    pub layer_8: Vec<f32>,
    /// See [`EmbeddingGemma2ImageStages::layer_0`].
    pub layer_15: Vec<f32>,
    /// Tower state after the last encoder layer.
    pub encoder: Vec<f32>,
    /// Coordinate pooling values, scaled by `sqrt(vision hidden)`.
    pub pooler_values: Vec<f32>,
    /// One flag per pooled cell; every entry is `true` because only valid patches exist.
    pub pooler_occupancy: Vec<bool>,
    /// The tower output after stripping, equal to [`EmbeddingGemma2ImageStages::pooler_values`].
    pub valid: Vec<f32>,
    /// Bridge output, row-major `[soft_rows, text_hidden]`.
    pub projected: Vec<f32>,
    /// Layer-0 queries after the Q norm, before RoPE, row-major `[valid_patches, heads, head]`.
    pub layer_0_q_normed: Vec<f32>,
    /// Layer-0 keys after the K norm, before RoPE.
    pub layer_0_k_normed: Vec<f32>,
    /// Layer-0 queries after RoPE.
    pub layer_0_q_rotated: Vec<f32>,
    /// Layer-0 keys after RoPE.
    pub layer_0_k_rotated: Vec<f32>,
    /// Valid patch count for this image.
    pub valid_patches: usize,
    /// Soft rows for this image.
    pub soft_rows: usize,
    /// Tower hidden width.
    pub hidden_size: usize,
    /// Bridge output width (the text tower's hidden width).
    pub text_hidden: usize,
    /// Vision attention head count.
    pub heads: usize,
    /// Vision attention head width.
    pub head_dim: usize,
}

impl EmbeddingGemma2ImageStages {
    fn from_parts(valid_patches: usize, soft_rows: usize, stages: VisionStages) -> Self {
        Self {
            patch: stages.patch,
            layer_0: stages.layer_0,
            layer_8: stages.layer_8,
            layer_15: stages.layer_15,
            encoder: stages.encoder,
            pooler_values: stages.pooler_values,
            pooler_occupancy: stages.pooler_occupancy,
            valid: stages.valid,
            projected: stages.projected,
            layer_0_q_normed: stages.layer_0_q_normed,
            layer_0_k_normed: stages.layer_0_k_normed,
            layer_0_q_rotated: stages.layer_0_q_rotated,
            layer_0_k_rotated: stages.layer_0_k_rotated,
            valid_patches,
            soft_rows,
            hidden_size: stages.hidden_size,
            text_hidden: stages.text_hidden,
            heads: stages.heads,
            head_dim: stages.head_dim,
        }
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// `[seq, heads, head_dim]` to `[heads, seq, head_dim]`.
fn to_head_major(x: &[f32], seq: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for t in 0..seq {
        for h in 0..heads {
            let src = (t * heads + h) * head_dim;
            let dst = (h * seq + t) * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&x[src..src + head_dim]);
        }
    }
    out
}

fn softmax_in_place(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// Bidirectional attention with scaling 1.0 over `[seq, heads, head_dim]` queries and
/// `[seq, kv_heads, head_dim]` keys and values, returning `[seq, heads, head_dim]`.
///
/// With `window = Some(w)` a query at position `i` attends to the keys `j` with
/// `|i - j| <= w`; with `None` it attends to every key. Query heads share key/value heads
/// in contiguous groups (`kv = head / (heads / kv_heads)`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn bidirectional_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: Option<usize>,
) -> Vec<f32> {
    let n_rep = heads / kv_heads;
    let qh = to_head_major(q, seq, heads, head_dim);
    let kh = to_head_major(k, seq, kv_heads, head_dim);
    let vh = to_head_major(v, seq, kv_heads, head_dim);
    let mut out = vec![0f32; seq * heads * head_dim];

    let tile = ATTN_QUERY_TILE.min(seq);
    let max_keys = match window {
        None => seq,
        Some(w) => (tile + 2 * w).min(seq),
    };
    let mut scores = vec![0f32; tile * max_keys];
    let mut ctx = vec![0f32; tile * head_dim];

    for h in 0..heads {
        let kv = h / n_rep;
        let qs = &qh[h * seq * head_dim..(h + 1) * seq * head_dim];
        let ks = &kh[kv * seq * head_dim..(kv + 1) * seq * head_dim];
        let vs = &vh[kv * seq * head_dim..(kv + 1) * seq * head_dim];
        let mut q0 = 0;
        while q0 < seq {
            let q1 = (q0 + tile).min(seq);
            let rows = q1 - q0;
            let (lo, hi) = match window {
                None => (0, seq),
                Some(w) => (q0.saturating_sub(w), (q1 + w).min(seq)),
            };
            let keys = hi - lo;
            matmul_bt(
                &qs[q0 * head_dim..q1 * head_dim],
                &ks[lo * head_dim..hi * head_dim],
                &mut scores[..rows * keys],
                rows,
                head_dim,
                keys,
            );
            for r in 0..rows {
                let qi = q0 + r;
                let (a, b) = match window {
                    None => (0, keys),
                    Some(w) => (qi.saturating_sub(w) - lo, (qi + w + 1).min(seq) - lo),
                };
                let row = &mut scores[r * keys..(r + 1) * keys];
                softmax_in_place(&mut row[a..b]);
                row[..a].fill(0.0);
                row[b..].fill(0.0);
            }
            matmul_into(
                &scores[..rows * keys],
                &vs[lo * head_dim..hi * head_dim],
                &mut ctx[..rows * head_dim],
                rows,
                keys,
                head_dim,
            );
            for r in 0..rows {
                let dst = ((q0 + r) * heads + h) * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&ctx[r * head_dim..(r + 1) * head_dim]);
            }
            q0 = q1;
        }
    }
    out
}

fn add_in_place(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

/// Tower stages for one image part, on the backend `state` names.
///
/// With `state` present the host hands the patch rows to the Metal tower and folds the pass's
/// per-segment times into `timings`; with `state` absent the whole encoding stays on the host.
/// Host runs are never mixed into a Metal pass and a device failure is never replayed on the
/// host.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn vision_part_stages(
    state: Option<&mut MetalEmbeddingGemma2State>,
    weights: &VisionWeights,
    prepared: &PreparedImage,
    should_cancel: &mut dyn FnMut() -> bool,
    timings: &mut VisionTimings,
) -> Result<VisionStages, InferenceError> {
    match state {
        Some(metal) => {
            let (stages, part_timings) = metal.vision_forward(weights, prepared, should_cancel)?;
            timings.add(&part_timings);
            Ok(stages)
        }
        None => Ok(weights.encode(prepared, should_cancel)?.stages),
    }
}

/// Host-only counterpart of [`vision_part_stages`].
///
/// A request that names a Metal state on a build without the `metal-gpu` feature is refused
/// rather than silently answered from the host.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
fn vision_part_stages(
    state: Option<&mut MetalEmbeddingGemma2State>,
    weights: &VisionWeights,
    prepared: &PreparedImage,
    should_cancel: &mut dyn FnMut() -> bool,
    _timings: &mut VisionTimings,
) -> Result<VisionStages, InferenceError> {
    if state.is_some() {
        return Err(InferenceError::Inference(
            "EmbeddingGemma 2 Metal forward requires macOS and the metal-gpu feature".to_string(),
        ));
    }
    Ok(weights.encode(prepared, should_cancel)?.stages)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// The EmbeddingGemma 2 text encoder, CPU f32 path.
pub struct EmbeddingGemma2Model {
    cfg: EmbeddingGemma2Config,
    weights: Weights,
    tokenizer: Option<TextTokenizer>,
    max_tokens: Option<usize>,
    vision: LazyVision,
    image_parts: Option<ImagePartsConfig>,
}

impl EmbeddingGemma2Model {
    /// Loads `config.json`, `model.safetensors` and `tokenizer.json` from `dir`.
    ///
    /// Only text-tower tensors are read; vision and audio tensors in the same file are ignored.
    pub fn from_model_dir(dir: &Path) -> Result<Self, InferenceError> {
        let config_path = dir.join("config.json");
        if !config_path.exists() {
            return Err(InferenceError::ModelNotFound(format!(
                "missing config.json in {}",
                dir.display()
            )));
        }
        let config_json = read_config_json_bounded(&config_path, "config.json")?;
        let cfg = EmbeddingGemma2Config::from_config_json_str(&config_json)?;
        let weights_path = dir.join("model.safetensors");
        if !weights_path.exists() {
            return Err(InferenceError::ModelNotFound(format!(
                "missing model.safetensors in {}",
                dir.display()
            )));
        }
        let tokenizer_path = dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(InferenceError::ModelNotFound(format!(
                "missing tokenizer.json in {}",
                dir.display()
            )));
        }
        let checkpoint = crate::weights::mmap_trust::open_regular_file_no_hang(&weights_path)
            .map_err(InferenceError::InvalidSafetensors)?;
        let vision_file = checkpoint.try_clone().map_err(InferenceError::Io)?;
        let weights = {
            let mut file = SafetensorsFile::from_open_file(checkpoint, &weights_path)?;
            load_weights(&mut file, &cfg)?
        };
        let text = std::fs::read_to_string(&tokenizer_path).map_err(InferenceError::Io)?;
        let tokenizer =
            TextTokenizer::from_tokenizer_json_str(&text, cfg.bos_token_id, cfg.eos_token_id)?;
        let image_parts = image_parts_config(&config_json, cfg.vocab_size);
        let vision =
            LazyVision::from_checkpoint(config_json, vision_file, weights_path, cfg.hidden_size);
        Ok(Self {
            cfg,
            weights,
            tokenizer: Some(tokenizer),
            max_tokens: Some(EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS),
            vision,
            image_parts,
        })
    }

    #[allow(
        dead_code,
        reason = "the image forward path consumes this internal cache separately"
    )]
    pub(crate) fn vision_weights(&self) -> Result<Arc<VisionWeights>, InferenceError> {
        self.vision.get()
    }

    /// Sets the longest token sequence [`EmbeddingGemma2Model::tokenize`] produces, or removes the
    /// limit with `None`. The default is [`EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS`].
    ///
    /// The count includes the beginning and end tokens that wrap every input, so the limit must be
    /// at least 2. Text beyond it is dropped from the end; the end token is kept.
    pub fn with_max_tokens(mut self, max_tokens: Option<usize>) -> Result<Self, InferenceError> {
        if let Some(n) = max_tokens
            && n < 2
        {
            return Err(InferenceError::InvalidInput(format!(
                "max_tokens {n} leaves no room for the beginning and end tokens"
            )));
        }
        self.max_tokens = max_tokens;
        Ok(self)
    }

    /// The parsed configuration.
    pub fn config(&self) -> &EmbeddingGemma2Config {
        &self.cfg
    }

    /// Native embedding width (768 for the released checkpoint).
    pub fn dimensions(&self) -> usize {
        self.cfg.embedding_dim
    }

    /// Token ids for `text`: the beginning token, the text, then the end token.
    ///
    /// With the default limit of [`EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS`] (see
    /// [`EmbeddingGemma2Model::with_max_tokens`]) the result is at most that long: longer text is
    /// truncated from the end, never rejected, and both wrapping tokens are kept.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            InferenceError::Tokenizer("this model was built without a tokenizer".to_string())
        })?;
        Ok(tokenizer.tokenize(text, self.max_tokens))
    }

    /// As [`EmbeddingGemma2Model::tokenize`], also reporting the sequence length before and
    /// after the limit. Both counts include the beginning and end tokens, and one tokenization
    /// yields the ids and both figures.
    pub fn tokenize_with_counts(
        &self,
        text: &str,
    ) -> Result<(Vec<u32>, EmbeddingTokenCounts), InferenceError> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            InferenceError::Tokenizer("this model was built without a tokenizer".to_string())
        })?;
        let (ids, before_truncation) = tokenizer.tokenize_counted(text, self.max_tokens);
        let embedded = ids.len();
        Ok((
            ids,
            EmbeddingTokenCounts {
                before_truncation,
                embedded,
            },
        ))
    }

    /// Embedding of `text`, L2-normalized, truncated to `output_dim` leading dimensions when set
    /// and re-normalized.
    ///
    /// The text is used as given: apply [`EmbeddingGemma2Task::format`] or
    /// [`format_titled_document`] first when a prompt is wanted. Input beyond the token limit of
    /// [`EmbeddingGemma2Model::tokenize`] is truncated.
    pub fn encode(
        &self,
        text: &str,
        output_dim: Option<usize>,
    ) -> Result<Vec<f32>, InferenceError> {
        let ids = self.tokenize(text)?;
        self.encode_ids(&ids, output_dim)
    }

    /// Embedding of a token sequence, as [`EmbeddingGemma2Model::encode`].
    ///
    /// Mean pooling covers every token. The sequence is used as given and is not truncated; the
    /// token limit applies in [`EmbeddingGemma2Model::tokenize`].
    pub fn encode_ids(
        &self,
        ids: &[u32],
        output_dim: Option<usize>,
    ) -> Result<Vec<f32>, InferenceError> {
        let dim = output_dim.unwrap_or(self.cfg.embedding_dim);
        let mut out = self.encode_ids_at_widths(ids, &[dim])?;
        out.pop()
            .ok_or_else(|| InferenceError::Inference("encode produced no embedding".to_string()))
    }

    /// One embedding per entry of `widths`, from a single forward pass. Each is the leading
    /// `width` dimensions of the pooled vector, L2-normalized.
    pub fn encode_ids_at_widths(
        &self,
        ids: &[u32],
        widths: &[usize],
    ) -> Result<Vec<Vec<f32>>, InferenceError> {
        self.check_widths(widths)?;
        let states = self.token_states(ids)?;
        self.pool_widths(&states, ids.len(), widths)
    }

    /// Builds the combined token layout from the ordered parts.
    ///
    /// The tokenizer wraps each text part on its own (`beginning, text, end`), so the
    /// interior tokens are the body between the two wrapping tokens; the outer wrap comes
    /// from the sequence itself. An image part expands to begin-image, its soft-row count
    /// of image markers and end-image. Returns the ids and the marker-slot indices where
    /// the image rows land, in part order.
    fn assemble_ids(
        &self,
        parts: &[EmbeddingGemma2Part],
        soft_rows_per_part: &[usize],
    ) -> Result<(Vec<u32>, Vec<usize>), InferenceError> {
        if soft_rows_per_part.len() != parts.len() {
            return Err(image_error("soft-row counts must align with the parts"));
        }
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            InferenceError::Tokenizer("this model was built without a tokenizer".to_string())
        })?;
        let has_image = parts
            .iter()
            .any(|part| matches!(part, EmbeddingGemma2Part::Image(_)));
        let images = match (has_image, self.image_parts) {
            (false, _) => None,
            (true, Some(layout)) => Some(layout),
            (true, None) => return Err(no_image_layout()),
        };
        let mut ids = Vec::new();
        ids.push(tokenizer.bos);
        let mut slots = Vec::new();
        let mut marker_total = 0usize;
        for (index, part) in parts.iter().enumerate() {
            match part {
                EmbeddingGemma2Part::Text(text) => {
                    let part_ids = tokenizer.tokenize(text, self.max_tokens);
                    if part_ids.len() < 2 {
                        return Err(image_error("a text part tokenized to fewer than two ids"));
                    }
                    ids.extend_from_slice(&part_ids[1..part_ids.len() - 1]);
                }
                EmbeddingGemma2Part::Image(_) => {
                    let Some(images) = images else {
                        return Err(no_image_layout());
                    };
                    let soft = soft_rows_per_part[index];
                    ids.push(images.boi_token_id);
                    marker_total = marker_total
                        .checked_add(soft)
                        .ok_or_else(|| image_error("image marker count overflow"))?;
                    for _ in 0..soft {
                        slots.push(ids.len());
                        ids.push(images.image_token_id);
                    }
                    ids.push(images.eoi_token_id);
                }
            }
        }
        ids.push(tokenizer.eos);
        if slots.len() != marker_total {
            return Err(image_error(
                "the image marker slots do not match the soft-row counts",
            ));
        }
        Ok((ids, slots))
    }

    /// Materializes the combined rows: hard-token embeddings with the image markers read
    /// as padding, then the bridged image rows scattered into the marker slots with the
    /// text embedding scale restored to one.
    fn combine_rows(
        &self,
        input_ids: &[u32],
        image_slots: &[usize],
        image_rows: &[f32],
    ) -> Result<Vec<f32>, InferenceError> {
        let hidden = self.cfg.hidden_size;
        let mut gather: Vec<u32> = input_ids.to_vec();
        if let Some(images) = self.image_parts {
            for id in &mut gather {
                if *id == images.image_token_id {
                    *id = 0;
                }
            }
        }
        let mut combined = self.scaled_embeddings(&gather)?;
        if !image_slots.is_empty() {
            if image_rows.len() != image_slots.len() * hidden {
                return Err(image_error(format!(
                    "image rows must hold {} values, got {}",
                    image_slots.len() * hidden,
                    image_rows.len()
                )));
            }
            for (row, &slot) in image_slots.iter().enumerate() {
                if slot >= combined.len() / hidden {
                    return Err(image_error("an image marker slot is outside the sequence"));
                }
                let source = &image_rows[row * hidden..(row + 1) * hidden];
                let target = &mut combined[slot * hidden..(slot + 1) * hidden];
                target.copy_from_slice(source);
            }
        }
        Ok(combined)
    }

    fn check_widths(&self, widths: &[usize]) -> Result<(), InferenceError> {
        for &width in widths {
            if width == 0 || width > self.cfg.embedding_dim {
                return Err(InferenceError::InvalidInput(format!(
                    "output dimension {width} is outside 1..={}",
                    self.cfg.embedding_dim
                )));
            }
        }
        Ok(())
    }

    /// Mean-pools `[tokens, embedding_dim]` states and returns the L2-normalized leading
    /// `width` dimensions for each entry of `widths`.
    fn pool_widths(
        &self,
        states: &[f32],
        tokens: usize,
        widths: &[usize],
    ) -> Result<Vec<Vec<f32>>, InferenceError> {
        let pooled = mean_pool(states, tokens, self.cfg.embedding_dim);
        widths
            .iter()
            .map(|&width| {
                let mut v = pooled[..width].to_vec();
                l2_normalize(&mut v)?;
                Ok(v)
            })
            .collect()
    }

    /// Embeds ordered text and image parts at each requested width, from one forward pass.
    ///
    /// The parts are laid out as the model's processor does: the beginning token,
    /// then each part in order, then the end token. A text part contributes its
    /// tokens verbatim; an image part contributes a begin-image marker, its soft-row
    /// count of image marker tokens and an end-image marker. The image soft rows carry
    /// no text embedding scale. Positions are one-dimensional over the whole sequence,
    /// and both the initial hidden state and the per-layer signal are derived from the
    /// same combined rows.
    ///
    /// Text-only calls reproduce [`EmbeddingGemma2Model::encode_ids_at_widths`] on the
    /// identical token sequence and never load the vision weights. An image part returns
    /// a typed error when the checkpoint has no image token layout. `should_cancel` is
    /// polled between preparation, the patch embed, each vision layer and the text
    /// encoder; a predicate that reports cancellation stops the call before the text
    /// encoder runs.
    pub fn encode_parts_at_widths(
        &self,
        parts: &[EmbeddingGemma2Part],
        widths: &[usize],
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<EmbeddingGemma2PartsEncoding, InferenceError> {
        let (encoding, _) = self.encode_parts_impl(parts, widths, should_cancel, None)?;
        Ok(encoding)
    }

    /// As [`EmbeddingGemma2Model::encode_parts_at_widths`], with the vision tower running on the
    /// GPU in f32 through `state`, which must have been built from this model with
    /// [`MetalEmbeddingGemma2State::new`]. The patch embed, pooling, the bridge and the text
    /// encoder stay on the host; the per-segment timings of the whole pass are returned with the
    /// encoding. A device failure returns an error and the call never falls back to the host.
    ///
    /// The caller holds the machine GPU lock for the whole call.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub fn encode_parts_at_widths_metal(
        &self,
        state: &mut MetalEmbeddingGemma2State,
        parts: &[EmbeddingGemma2Part],
        widths: &[usize],
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<(EmbeddingGemma2PartsEncoding, VisionTimings), InferenceError> {
        state.check_model(self)?;
        self.encode_parts_impl(parts, widths, should_cancel, Some(state))
    }

    /// The shared body of the host and Metal entry points. `state` is `Some` only on the Metal
    /// path, and `None` from the host entry point, which never touches Metal.
    fn encode_parts_impl(
        &self,
        parts: &[EmbeddingGemma2Part],
        widths: &[usize],
        should_cancel: &mut dyn FnMut() -> bool,
        mut state: Option<&mut MetalEmbeddingGemma2State>,
    ) -> Result<(EmbeddingGemma2PartsEncoding, VisionTimings), InferenceError> {
        self.check_widths(widths)?;
        if parts.is_empty() {
            return Err(InferenceError::InvalidInput(
                "cannot embed an empty part list".to_string(),
            ));
        }
        if parts
            .iter()
            .any(|part| matches!(part, EmbeddingGemma2Part::Image(_)))
            && self.image_parts.is_none()
        {
            return Err(no_image_layout());
        }
        if should_cancel() {
            return Err(InferenceError::Inference(
                "cancelled before embedding preparations".to_string(),
            ));
        }
        #[cfg_attr(
            not(all(target_os = "macos", feature = "metal-gpu")),
            allow(unused_mut, reason = "the host-only dispatch leaves the total at zero")
        )]
        let mut timings = VisionTimings::default();
        let mut image_rows: Vec<f32> = Vec::new();
        let mut image_stages: Vec<EmbeddingGemma2ImageStages> = Vec::new();
        let mut soft_rows_per_part = vec![0usize; parts.len()];
        let mut image_row_count = 0usize;
        let mut valid_patches = 0usize;
        for (index, part) in parts.iter().enumerate() {
            let EmbeddingGemma2Part::Image(image) = part else {
                continue;
            };
            let prepared = prepare_image(image)?;
            soft_rows_per_part[index] = prepared.soft_rows();
            valid_patches = valid_patches
                .checked_add(prepared.valid_count())
                .ok_or_else(|| image_error("valid patch count overflow"))?;
            image_row_count = image_row_count
                .checked_add(prepared.soft_rows())
                .ok_or_else(|| image_error("image row count overflow"))?;
            let weights = self.vision.get()?;
            let stages = vision_part_stages(
                state.as_deref_mut(),
                &weights,
                &prepared,
                should_cancel,
                &mut timings,
            )?;
            image_rows.extend_from_slice(&stages.projected);
            image_stages.push(EmbeddingGemma2ImageStages::from_parts(
                prepared.valid_count(),
                prepared.soft_rows(),
                stages,
            ));
            if should_cancel() {
                return Err(InferenceError::Inference(
                    "cancelled before assembling the combined sequence".to_string(),
                ));
            }
        }
        let (input_ids, image_slots) = self.assemble_ids(parts, &soft_rows_per_part)?;
        let combined = self.combine_rows(&input_ids, &image_slots, &image_rows)?;
        let tokens = input_ids.len();
        let mut hidden = combined.clone();
        self.run_text_encoder(&mut hidden, &combined, tokens, should_cancel)?;
        let states = self.project_states(&hidden, tokens)?;
        let embeddings = self.pool_widths(&states, tokens, widths)?;
        let pre_normalization = mean_pool(&states, tokens, self.cfg.embedding_dim);
        Ok((
            EmbeddingGemma2PartsEncoding {
                embeddings,
                input_ids,
                counts: EmbeddingGemma2PartCounts {
                    sequence_length: tokens,
                    valid_patches,
                    image_rows: image_row_count,
                },
                image_rows,
                image_stages,
                token_states: states,
                pre_normalization,
            },
            timings,
        ))
    }

    /// Per-token states after the final norm and the `hidden -> embedding_dim` projection,
    /// row-major `[ids.len(), embedding_dim]`.
    pub fn token_states(&self, ids: &[u32]) -> Result<Vec<f32>, InferenceError> {
        let mut h = self.scaled_embeddings(ids)?;
        let rows = h.clone();
        self.run_text_encoder(&mut h, &rows, ids.len(), &mut || false)?;
        self.project_states(&h, ids.len())
    }

    /// Runs the encoder layers over `h` in place, then the final norm. `rows` is the matrix
    /// the per-layer signal is derived from: for a text call the scaled token embeddings, and
    /// for a mixed text-and-image call the combined rows. `should_cancel` is polled once
    /// before the loop and between layers.
    fn run_text_encoder(
        &self,
        h: &mut [f32],
        rows: &[f32],
        tokens: usize,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<(), InferenceError> {
        let cfg = &self.cfg;
        if should_cancel() {
            return Err(InferenceError::Inference(
                "cancelled before the text encoder".to_string(),
            ));
        }
        let mut rope: [Option<(Vec<f32>, Vec<f32>)>; 2] = [None, None];
        for l in 0..cfg.num_hidden_layers {
            if should_cancel() {
                return Err(InferenceError::Inference(format!(
                    "cancelled before text layer {l}"
                )));
            }
            let slot = usize::from(cfg.layer_types[l] == EmbeddingGemma2LayerKind::Full);
            if rope[slot].is_none() {
                rope[slot] = Some(self.rope_table(l, tokens));
            }
            let Some((cos, sin)) = rope[slot].as_ref() else {
                return Err(InferenceError::Inference(
                    "rope table missing after construction".to_string(),
                ));
            };
            self.run_layer(l, h, rows, cos, sin);
        }
        gemma4_rms_norm(h, &self.weights.norm, cfg.hidden_size, cfg.rms_norm_eps);
        Ok(())
    }

    /// Per-token `hidden -> embedding_dim` projection over states that already carry the
    /// final norm, row-major `[tokens, embedding_dim]`.
    fn project_states(&self, h: &[f32], tokens: usize) -> Result<Vec<f32>, InferenceError> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden_size;
        if h.len() != tokens * hidden {
            return Err(InferenceError::Inference(format!(
                "text states must hold {} values, got {}",
                tokens * hidden,
                h.len()
            )));
        }
        let mut states = vec![0f32; tokens * cfg.embedding_dim];
        matmul_bt(
            h,
            &self.weights.embedding_projection,
            &mut states,
            tokens,
            hidden,
            cfg.embedding_dim,
        );
        Ok(states)
    }

    /// Per-token states as [`EmbeddingGemma2Model::token_states`], computed on the GPU in f32
    /// through `state`, which must have been built from this model with
    /// [`MetalEmbeddingGemma2State::new`]. Requires macOS and the `metal-gpu` feature; the CPU
    /// path stays the default.
    ///
    /// The caller holds the machine GPU lock for the whole call when measuring or testing.
    pub fn token_states_metal(
        &self,
        state: &mut MetalEmbeddingGemma2State,
        ids: &[u32],
    ) -> Result<Vec<f32>, InferenceError> {
        state.check_model(self)?;
        let embeddings = self.scaled_embeddings(ids)?;
        state.forward(&embeddings, ids.len())
    }

    /// As [`EmbeddingGemma2Model::encode_ids_at_widths`], with the forward pass on the GPU.
    /// Mean pooling and normalization run on the host over the `[tokens, embedding_dim]` states
    /// read back from the GPU.
    pub fn encode_ids_at_widths_metal(
        &self,
        state: &mut MetalEmbeddingGemma2State,
        ids: &[u32],
        widths: &[usize],
    ) -> Result<Vec<Vec<f32>>, InferenceError> {
        self.check_widths(widths)?;
        let states = self.token_states_metal(state, ids)?;
        self.pool_widths(&states, ids.len(), widths)
    }

    /// Token embeddings times `embed_scale`, row-major `[ids.len(), hidden_size]`. Rejects an
    /// empty sequence and ids outside the vocabulary.
    pub(crate) fn scaled_embeddings(&self, ids: &[u32]) -> Result<Vec<f32>, InferenceError> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden_size;
        if ids.is_empty() {
            return Err(InferenceError::InvalidInput(
                "cannot embed an empty token sequence".to_string(),
            ));
        }
        let mut h = vec![0f32; ids.len() * hidden];
        for (row, &id) in h.chunks_exact_mut(hidden).zip(ids) {
            let id = id as usize;
            if id >= cfg.vocab_size {
                return Err(InferenceError::InvalidInput(format!(
                    "token id {id} is outside the {} entry vocabulary",
                    cfg.vocab_size
                )));
            }
            for (o, &e) in row
                .iter_mut()
                .zip(&self.weights.embed_tokens[id * hidden..(id + 1) * hidden])
            {
                *o = e * cfg.embed_scale;
            }
        }
        Ok(h)
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub(crate) fn weights(&self) -> &Weights {
        &self.weights
    }

    /// RoPE cosine and sine tables for the first `t` positions of layer `l`'s type.
    fn rope_table(&self, l: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
        let theta = match self.cfg.layer_types[l] {
            EmbeddingGemma2LayerKind::Sliding => self.cfg.rope_theta_sliding,
            EmbeddingGemma2LayerKind::Full => self.cfg.rope_theta_full,
        };
        let inv_freq = gemma4_rope_inv_freq(self.cfg.layer_shapes[l].head_dim, theta, None);
        let positions: Vec<u32> = (0..t as u32).collect();
        gemma4_rope_cos_sin(&inv_freq, &positions)
    }

    /// One encoder layer applied in place to `h` (`[t, hidden]`). `embeddings` is the row
    /// matrix the per-layer signal is derived from: the scaled token embeddings for a text
    /// call, and the combined rows for a mixed image call.
    fn run_layer(&self, l: usize, h: &mut [f32], embeddings: &[f32], cos: &[f32], sin: &[f32]) {
        let cfg = &self.cfg;
        let w = &self.weights;
        let lw = &w.layers[l];
        let hidden = cfg.hidden_size;
        let per_layer = cfg.hidden_size_per_layer_input;
        let heads = cfg.num_attention_heads;
        let eps = cfg.rms_norm_eps;
        let t = h.len() / hidden;
        let shape = cfg.layer_shapes[l];
        let (hd, kv_heads) = (shape.head_dim, shape.num_key_value_heads);

        // Attention block.
        let mut x = h.to_vec();
        gemma4_rms_norm(&mut x, &lw.input_layernorm, hidden, eps);
        let mut q = vec![0f32; t * heads * hd];
        let mut k = vec![0f32; t * kv_heads * hd];
        let mut v = vec![0f32; t * kv_heads * hd];
        matmul_bt(&x, &lw.q_proj, &mut q, t, hidden, heads * hd);
        matmul_bt(&x, &lw.k_proj, &mut k, t, hidden, kv_heads * hd);
        matmul_bt(&x, &lw.v_proj, &mut v, t, hidden, kv_heads * hd);
        gemma4_qk_norm_v_unscaled(&mut q, &mut k, &mut v, &lw.q_norm, &lw.k_norm, hd, eps);
        gemma4_apply_rope(&mut q, cos, sin, t, heads, hd);
        gemma4_apply_rope(&mut k, cos, sin, t, kv_heads, hd);
        let window = match cfg.layer_types[l] {
            EmbeddingGemma2LayerKind::Sliding => Some(cfg.sliding_window),
            EmbeddingGemma2LayerKind::Full => None,
        };
        let ctx = bidirectional_attention(&q, &k, &v, t, heads, kv_heads, hd, window);
        let mut attn_out = vec![0f32; t * hidden];
        matmul_bt(&ctx, &lw.o_proj, &mut attn_out, t, heads * hd, hidden);
        gemma4_rms_norm(&mut attn_out, &lw.post_attention_layernorm, hidden, eps);
        add_in_place(h, &attn_out);

        // Feed-forward block.
        let mut x = h.to_vec();
        gemma4_rms_norm(&mut x, &lw.pre_feedforward_layernorm, hidden, eps);
        let mut gate_scratch = vec![0f32; t * cfg.intermediate_size];
        let mut up_scratch = vec![0f32; t * cfg.intermediate_size];
        let mut mlp_out = vec![0f32; t * hidden];
        gemma4_geglu_mlp(
            &x,
            &lw.gate_proj,
            &lw.up_proj,
            &lw.down_proj,
            t,
            hidden,
            cfg.intermediate_size,
            &mut gate_scratch,
            &mut up_scratch,
            &mut mlp_out,
        );
        gemma4_rms_norm(&mut mlp_out, &lw.post_feedforward_layernorm, hidden, eps);
        add_in_place(h, &mlp_out);

        // Per-layer input block. This layer's slice of the projection-only per-layer signal is
        // computed here from the scaled embeddings instead of materializing every layer's slice.
        let ple_scale = (hidden as f64).powf(-0.5) as f32;
        let slice =
            &w.per_layer_model_projection[l * per_layer * hidden..(l + 1) * per_layer * hidden];
        let mut ple_input = vec![0f32; t * per_layer];
        matmul_bt(embeddings, slice, &mut ple_input, t, hidden, per_layer);
        for v in ple_input.iter_mut() {
            *v *= ple_scale;
        }
        gemma4_rms_norm(&mut ple_input, &w.per_layer_projection_norm, per_layer, eps);
        let mut gate = vec![0f32; t * per_layer];
        matmul_bt(h, &lw.per_layer_input_gate, &mut gate, t, hidden, per_layer);
        gemma4_gelu_tanh(&mut gate);
        elementwise_mul(&mut gate, &ple_input);
        let mut ple_out = vec![0f32; t * hidden];
        matmul_bt(
            &gate,
            &lw.per_layer_projection,
            &mut ple_out,
            t,
            per_layer,
            hidden,
        );
        gemma4_rms_norm(&mut ple_out, &lw.post_per_layer_input_norm, hidden, eps);
        add_in_place(h, &ple_out);

        for v in h.iter_mut() {
            *v *= lw.layer_scalar;
        }
    }
}

/// Mean over `tokens` rows of `[tokens, dim]`, accumulated in f64.
fn mean_pool(states: &[f32], tokens: usize, dim: usize) -> Vec<f32> {
    let mut acc = vec![0f64; dim];
    for row in states.chunks_exact(dim) {
        for (a, &v) in acc.iter_mut().zip(row) {
            *a += f64::from(v);
        }
    }
    acc.into_iter()
        .map(|a| (a / tokens as f64) as f32)
        .collect()
}

fn l2_normalize(v: &mut [f32]) -> Result<(), InferenceError> {
    let norm = v
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() {
        return Err(InferenceError::Inference(
            "embedding contains a non-finite value".to_string(),
        ));
    }
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x = (f64::from(*x) / norm) as f32;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;

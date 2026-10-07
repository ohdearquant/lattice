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
//! Only the text tower is loaded. Vision and audio tensors in the checkpoint
//! are never read.

use super::embeddinggemma2_config::{EmbeddingGemma2Config, EmbeddingGemma2LayerKind};
use super::gemma4_ops::{
    gemma4_apply_rope, gemma4_geglu_mlp, gemma4_gelu_tanh, gemma4_qk_norm_v_unscaled,
    gemma4_rms_norm, gemma4_rope_cos_sin, gemma4_rope_inv_freq,
};
use crate::error::InferenceError;
use crate::forward::cpu::{elementwise_mul, matmul_bt, matmul_into};
use crate::forward::metal_embeddinggemma2::MetalEmbeddingGemma2State;
use crate::tokenizer::common::Tokenizer;
use crate::tokenizer::gemma_bpe::GemmaBpeTokenizer;
use crate::weights::{SafetensorsFile, TensorSource};
use std::path::Path;

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
        let encoded = self.inner.tokenize_batch(&[text]).pop();
        let (ids, len) = match &encoded {
            Some(e) => (e.input_ids.as_slice(), e.real_length),
            None => (&[][..], 0),
        };
        let keep = max_tokens.map_or(len, |n| len.min(n.saturating_sub(2)));
        self.wrap(&ids[..keep])
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

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// The EmbeddingGemma 2 text encoder, CPU f32 path.
pub struct EmbeddingGemma2Model {
    cfg: EmbeddingGemma2Config,
    weights: Weights,
    tokenizer: Option<TextTokenizer>,
    max_tokens: Option<usize>,
}

impl EmbeddingGemma2Model {
    /// Loads `config.json`, `model.safetensors` and `tokenizer.json` from `dir`.
    ///
    /// Only text-tower tensors are read; vision and audio tensors in the same file are ignored.
    pub fn from_model_dir(dir: &Path) -> Result<Self, InferenceError> {
        let cfg = EmbeddingGemma2Config::from_model_dir(dir)?;
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
        let mut file = SafetensorsFile::open(&weights_path)?;
        let weights = load_weights(&mut file, &cfg)?;
        let text = std::fs::read_to_string(&tokenizer_path).map_err(InferenceError::Io)?;
        let tokenizer =
            TextTokenizer::from_tokenizer_json_str(&text, cfg.bos_token_id, cfg.eos_token_id)?;
        Ok(Self {
            cfg,
            weights,
            tokenizer: Some(tokenizer),
            max_tokens: Some(EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS),
        })
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

    /// Per-token states after the final norm and the `hidden -> embedding_dim` projection,
    /// row-major `[ids.len(), embedding_dim]`.
    pub fn token_states(&self, ids: &[u32]) -> Result<Vec<f32>, InferenceError> {
        let cfg = &self.cfg;
        let w = &self.weights;
        let t = ids.len();
        let hidden = cfg.hidden_size;

        let mut h = self.scaled_embeddings(ids)?;
        let embeddings = h.clone();

        let mut rope: [Option<(Vec<f32>, Vec<f32>)>; 2] = [None, None];
        for l in 0..cfg.num_hidden_layers {
            let slot = usize::from(cfg.layer_types[l] == EmbeddingGemma2LayerKind::Full);
            if rope[slot].is_none() {
                rope[slot] = Some(self.rope_table(l, t));
            }
            let Some((cos, sin)) = rope[slot].as_ref() else {
                return Err(InferenceError::Inference(
                    "rope table missing after construction".to_string(),
                ));
            };
            self.run_layer(l, &mut h, &embeddings, cos, sin);
        }

        gemma4_rms_norm(&mut h, &w.norm, hidden, cfg.rms_norm_eps);
        let mut states = vec![0f32; t * cfg.embedding_dim];
        matmul_bt(
            &h,
            &w.embedding_projection,
            &mut states,
            t,
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

    /// One encoder layer applied in place to `h` (`[t, hidden]`). `embeddings` is the scaled
    /// token-embedding matrix the per-layer signal is derived from.
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

//! ERNIE-4.5 dense text decoder (PaddleOCR-VL family): CPU f32 reference
//! forward.
//!
//! The supported layout is PaddleOCR-VL's ERNIE-4.5 text decoder at revision
//! `c5630abae1d940eafe0697512a0325494b02ab42`; the loader is fail-closed on
//! any other tensor layout.
//!
//! Scope is the language-model slice of `PaddleOCRVLForConditionalGeneration`
//! (`Ernie4_5ForCausalLM` in the checkpoint's own modeling source): 18-layer
//! pre-norm dense decoder, GQA with expanded heads (`num_heads * head_dim !=
//! hidden_size`: q 1024->2048, k/v 1024->256, o 2048->1024), SiLU gate/up/down
//! MLP, RMSNorm (`w * x * rsqrt(mean(x^2) + eps)`), untied `lm_head`. No KV
//! cache and no sampling here — this module exists to be held against the HF
//! reference activations captured in
//! `tests/fixtures/paddleocr_vl/decoder/decoder_goldens.json`, and later
//! slices build generation on top of a verified forward.
//!
//! RoPE: the reference applies Qwen2-VL-style multimodal-section RoPE
//! (`apply_multimodal_rotary_pos_emb`, `mrope_section` doubled and chunks
//! gathered per position row `i % 3`) with **stride-half** `rotate_half`
//! pairing. For text-only input the reference model derives all three
//! position rows from the same `arange`, which makes the section gather a
//! no-op: every chunk reads identical angles, so the result equals plain 1-D
//! neox RoPE at `rope_theta`. This slice implements exactly that reduction
//! (via the shared stride-half rope kernels) and asserts the config declares
//! the multimodal sections it reduces away; the sectioned form becomes
//! load-bearing only when vision positions enter in a later slice.

use std::path::Path;

use serde::Deserialize;

use crate::error::InferenceError;
use crate::forward::cpu::{matmul_bt, rms_norm, silu_inplace};
use crate::model::gemma4_ops::{gemma4_apply_rope, gemma4_rope_cos_sin, gemma4_rope_inv_freq};
use crate::weights::TensorSource;

/// Maximum number of decoder layers accepted by the loader, eight times the
/// shipped checkpoint's 18 layers to leave room for larger compatible models.
pub const MAX_LAYERS: usize = 144;
/// Maximum hidden width accepted by the loader, eight times the shipped
/// checkpoint's 1024-wide hidden state.
pub const MAX_HIDDEN_SIZE: usize = 8192;
/// Maximum MLP width accepted by the loader, four times the shipped
/// checkpoint's 3072-wide intermediate state.
pub const MAX_INTERMEDIATE_SIZE: usize = 12_288;
/// Maximum query-head count accepted by the loader, eight times the shipped
/// checkpoint's 16 query heads.
pub const MAX_HEADS: usize = 128;
/// Maximum KV-head count accepted by the loader, eight times the shipped
/// checkpoint's 2 KV heads.
pub const MAX_KV_HEADS: usize = 16;
/// Maximum head dimension accepted by the loader, eight times the shipped
/// checkpoint's 128-wide heads.
pub const MAX_HEAD_DIM: usize = 1024;
/// Maximum vocabulary size accepted by the loader, eight times the shipped
/// checkpoint's 103424-token vocabulary.
pub const MAX_VOCAB_SIZE: usize = 827_392;
/// Maximum sequence length accepted by [`Ernie45Model::forward_trace`]. At
/// this cap, its logits buffer occupies `4096 * vocab * 4` bytes.
pub const MAX_SEQ_LEN: usize = 4096;

fn checked_product(a: usize, b: usize, what: &str) -> Result<usize, InferenceError> {
    a.checked_mul(b)
        .ok_or_else(|| InferenceError::Inference(format!("ernie45 {what} overflow: {a} * {b}")))
}

fn checked_element_count(shape: &[usize], name: &str) -> Result<usize, InferenceError> {
    shape.iter().try_fold(1usize, |count, &dim| {
        checked_product(count, dim, &format!("tensor {name} element count"))
    })
}

fn validate_dimension(name: &str, value: usize, max: usize) -> Result<(), InferenceError> {
    if value == 0 || value > max {
        return Err(InferenceError::Inference(format!(
            "ernie45 config: {name} must be in 1..={max}, got {value}"
        )));
    }
    Ok(())
}

fn reserve_error(what: &str, error: std::collections::TryReserveError) -> InferenceError {
    InferenceError::Inference(format!("ernie45 {what} allocation failed: {error}"))
}

/// Text-decoder subset of the checkpoint's `config.json`, deserialized
/// fail-closed: every field this forward pass depends on is required, so a
/// checkpoint missing one refuses to load instead of running on a default.
#[derive(Debug, Clone, Deserialize)]
pub struct Ernie45Config {
    /// Hidden state width used by embeddings, norms, and residual streams.
    pub hidden_size: usize,
    /// MLP width used by the gate and up projections.
    pub intermediate_size: usize,
    /// Number of transformer layers applied to the residual stream.
    pub num_hidden_layers: usize,
    /// Number of query heads used to form attention queries and outputs.
    pub num_attention_heads: usize,
    /// Number of KV heads whose keys and values are shared by query groups.
    pub num_key_value_heads: usize,
    /// Width of each attention head used by projections and RoPE.
    pub head_dim: usize,
    /// Vocabulary size used by token lookup and the untied language head.
    pub vocab_size: usize,
    /// Epsilon added to the RMSNorm variance before the reciprocal square root.
    pub rms_norm_eps: f32,
    /// Base period used to build the text-only RoPE frequencies.
    pub rope_theta: f64,
    /// Multimodal RoPE section metadata reduced to text-only 1-D RoPE.
    pub rope_scaling: Ernie45RopeScaling,
    /// Whether embeddings and the language head share weights; this layout
    /// requires it to be false.
    pub tie_word_embeddings: bool,
    /// Whether attention and MLP projections contain biases; this layout
    /// requires it to be false.
    pub use_bias: bool,
}

#[derive(Debug, Clone, Deserialize)]
/// Multimodal RoPE metadata used by the text-only decoder reduction.
pub struct Ernie45RopeScaling {
    /// Section widths whose sum must equal half of the attention head width.
    pub mrope_section: Vec<usize>,
}

impl Ernie45Config {
    /// Parse the decoder fields out of a full PaddleOCR-VL `config.json`
    /// string and validate the invariants this forward pass hardcodes.
    pub fn from_config_json_str(text: &str) -> Result<Self, InferenceError> {
        let cfg: Self = serde_json::from_str(text).map_err(|e| {
            InferenceError::Inference(format!("ernie45 config: failed to parse config.json: {e}"))
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse and validate the decoder configuration from a JSON file.
    pub fn from_config_json(path: &Path) -> Result<Self, InferenceError> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            InferenceError::Inference(format!(
                "ernie45 config: failed to read {}: {e}",
                path.display()
            ))
        })?;
        Self::from_config_json_str(&text)
    }

    fn validate(&self) -> Result<(), InferenceError> {
        validate_dimension("hidden_size", self.hidden_size, MAX_HIDDEN_SIZE)?;
        validate_dimension(
            "intermediate_size",
            self.intermediate_size,
            MAX_INTERMEDIATE_SIZE,
        )?;
        validate_dimension("num_hidden_layers", self.num_hidden_layers, MAX_LAYERS)?;
        validate_dimension("num_attention_heads", self.num_attention_heads, MAX_HEADS)?;
        validate_dimension(
            "num_key_value_heads",
            self.num_key_value_heads,
            MAX_KV_HEADS,
        )?;
        validate_dimension("head_dim", self.head_dim, MAX_HEAD_DIM)?;
        validate_dimension("vocab_size", self.vocab_size, MAX_VOCAB_SIZE)?;
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(InferenceError::Inference(format!(
                "ernie45 config: attention heads {} must be a positive multiple of kv heads {}",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return Err(InferenceError::Inference(format!(
                "ernie45 config: head_dim {} must be positive and even (stride-half RoPE)",
                self.head_dim
            )));
        }
        // The text-only forward reduces sectioned mrope to 1-D RoPE, which is
        // only the reference's own behavior when the sections tile the half
        // dimension exactly (`sum(mrope_section) == head_dim / 2`).
        let section_sum = self
            .rope_scaling
            .mrope_section
            .iter()
            .try_fold(0usize, |sum, &section| sum.checked_add(section))
            .ok_or_else(|| {
                InferenceError::Inference(
                    "ernie45 config: mrope_section sum overflows usize".into(),
                )
            })?;
        if section_sum != self.head_dim / 2 {
            return Err(InferenceError::Inference(format!(
                "ernie45 config: mrope_section {:?} sums to {section_sum}, expected head_dim/2 = \
                 {}; refusing to reduce an unrecognized section layout to 1-D RoPE",
                self.rope_scaling.mrope_section,
                self.head_dim / 2
            )));
        }
        if self.tie_word_embeddings {
            return Err(InferenceError::Inference(
                "ernie45 config: tie_word_embeddings=true is not this loader's checkpoint shape \
                 (PaddleOCR-VL ships an untied lm_head)"
                    .into(),
            ));
        }
        if self.use_bias {
            return Err(InferenceError::Inference(
                "ernie45 config: use_bias=true is not this loader's checkpoint shape (no \
                 projection in the pinned checkpoint carries a bias)"
                    .into(),
            ));
        }
        Ok(())
    }

    fn q_dim(&self) -> Result<usize, InferenceError> {
        checked_product(
            self.num_attention_heads,
            self.head_dim,
            "query projection dimension",
        )
    }

    fn kv_dim(&self) -> Result<usize, InferenceError> {
        checked_product(
            self.num_key_value_heads,
            self.head_dim,
            "key/value projection dimension",
        )
    }
}

/// One decoder layer's weights, all row-major `[out, in]` as stored in the
/// checkpoint (so `matmul_bt` computes `x @ W^T` directly).
pub struct Ernie45LayerWeights {
    pub(crate) q_proj: Vec<f32>,
    pub(crate) k_proj: Vec<f32>,
    pub(crate) v_proj: Vec<f32>,
    pub(crate) o_proj: Vec<f32>,
    pub(crate) gate_proj: Vec<f32>,
    pub(crate) up_proj: Vec<f32>,
    pub(crate) down_proj: Vec<f32>,
    pub(crate) input_layernorm: Vec<f32>,
    pub(crate) post_attention_layernorm: Vec<f32>,
}

/// Owned checkpoint tensors for the supported ERNIE-4.5 decoder layout.
pub struct Ernie45Weights {
    pub(crate) embed_tokens: Vec<f32>,
    pub(crate) layers: Vec<Ernie45LayerWeights>,
    pub(crate) final_norm: Vec<f32>,
    pub(crate) lm_head: Vec<f32>,
}

fn load_tensor<T: TensorSource + ?Sized>(
    source: &mut T,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, InferenceError> {
    let expected_elements = checked_element_count(expected, name)?;
    let declared_shape = source
        .tensor_shape(name)?
        .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))?;
    if declared_shape != expected {
        return Err(InferenceError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: declared_shape,
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
    if data.len() != expected_elements {
        return Err(InferenceError::Inference(format!(
            "ernie45 tensor {name} contains {} elements, expected {expected_elements}",
            data.len()
        )));
    }
    Ok(data)
}

fn validate_weight_len(name: &str, actual: usize, expected: usize) -> Result<(), InferenceError> {
    if actual != expected {
        return Err(InferenceError::Inference(format!(
            "ernie45 weight {name} contains {actual} elements, expected {expected}"
        )));
    }
    Ok(())
}

impl Ernie45Weights {
    /// Load the text-decoder tensors (`model.*` + `lm_head.*`; the
    /// checkpoint's `visual.*` and `mlp_AR.*` trees are outside this slice)
    /// with fail-closed shape checks against `cfg`.
    pub fn load<T: TensorSource + ?Sized>(
        source: &mut T,
        cfg: &Ernie45Config,
    ) -> Result<Self, InferenceError> {
        cfg.validate()?;
        let h = cfg.hidden_size;
        let q_dim = cfg.q_dim()?;
        let kv_dim = cfg.kv_dim()?;
        let embed_tokens = load_tensor(source, "model.embed_tokens.weight", &[cfg.vocab_size, h])?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(cfg.num_hidden_layers)
            .map_err(|error| reserve_error("decoder layers", error))?;
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}.");
            layers.push(Ernie45LayerWeights {
                q_proj: load_tensor(source, &format!("{p}self_attn.q_proj.weight"), &[q_dim, h])?,
                k_proj: load_tensor(source, &format!("{p}self_attn.k_proj.weight"), &[kv_dim, h])?,
                v_proj: load_tensor(source, &format!("{p}self_attn.v_proj.weight"), &[kv_dim, h])?,
                o_proj: load_tensor(source, &format!("{p}self_attn.o_proj.weight"), &[h, q_dim])?,
                gate_proj: load_tensor(
                    source,
                    &format!("{p}mlp.gate_proj.weight"),
                    &[cfg.intermediate_size, h],
                )?,
                up_proj: load_tensor(
                    source,
                    &format!("{p}mlp.up_proj.weight"),
                    &[cfg.intermediate_size, h],
                )?,
                down_proj: load_tensor(
                    source,
                    &format!("{p}mlp.down_proj.weight"),
                    &[h, cfg.intermediate_size],
                )?,
                input_layernorm: load_tensor(source, &format!("{p}input_layernorm.weight"), &[h])?,
                post_attention_layernorm: load_tensor(
                    source,
                    &format!("{p}post_attention_layernorm.weight"),
                    &[h],
                )?,
            });
        }
        let final_norm = load_tensor(source, "model.norm.weight", &[h])?;
        let lm_head = load_tensor(source, "lm_head.weight", &[cfg.vocab_size, h])?;
        Ok(Self {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        })
    }
}

/// Per-checkpoint activations from one full-sequence forward, in the same
/// order the HF-side golden generator captures them (embedding output, each
/// decoder layer's output, the final norm output, then logits). All buffers
/// are `[seq_len, hidden]` row-major except `logits` (`[seq_len, vocab]`).
pub struct Ernie45Trace {
    /// Embedding output with shape `[seq_len, hidden_size]`.
    pub embed: Vec<f32>,
    /// Residual output after each decoder layer, each with shape
    /// `[seq_len, hidden_size]`.
    pub layer_outputs: Vec<Vec<f32>>,
    /// Final RMSNorm output with shape `[seq_len, hidden_size]`.
    pub final_norm: Vec<f32>,
    /// Language-model logits with shape `[seq_len, vocab_size]`.
    pub logits: Vec<f32>,
}

/// Row-wise softmax with exact `f32::exp`, fail-closed on non-finite scores
/// (same contract as `gemma4_model.rs`'s private helper: a NaN or +/-inf
/// score zeroes the whole row rather than propagating).
fn softmax_row_fail_closed(row: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in row.iter() {
        if !v.is_finite() {
            row.fill(0.0);
            return;
        }
        max = max.max(v);
    }
    let mut sum = 0f32;
    for v in row.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        sum += e;
    }
    if sum > 0.0 && sum.is_finite() {
        for v in row.iter_mut() {
            *v /= sum;
        }
    } else {
        row.fill(0.0);
    }
}

/// Validated ERNIE-4.5 decoder configuration and checkpoint weights.
pub struct Ernie45Model {
    cfg: Ernie45Config,
    weights: Ernie45Weights,
}

impl Ernie45Model {
    /// Validate a decoder configuration and its owned weights before creating
    /// a model ready for CPU forward passes.
    pub fn new(cfg: Ernie45Config, weights: Ernie45Weights) -> Result<Self, InferenceError> {
        cfg.validate()?;
        let q_dim = cfg.q_dim()?;
        let kv_dim = cfg.kv_dim()?;
        let h = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let vocab = cfg.vocab_size;

        validate_weight_len(
            "model.embed_tokens.weight",
            weights.embed_tokens.len(),
            checked_product(vocab, h, "embedding weight size")?,
        )?;
        validate_weight_len("model.norm.weight", weights.final_norm.len(), h)?;
        validate_weight_len(
            "lm_head.weight",
            weights.lm_head.len(),
            checked_product(vocab, h, "language-head weight size")?,
        )?;
        if weights.layers.len() != cfg.num_hidden_layers {
            return Err(InferenceError::Inference(format!(
                "ernie45 decoder has {} layers, expected {}",
                weights.layers.len(),
                cfg.num_hidden_layers
            )));
        }
        let q_proj_len = checked_product(q_dim, h, "query weight size")?;
        let kv_proj_len = checked_product(kv_dim, h, "key/value weight size")?;
        let o_proj_len = checked_product(h, q_dim, "output projection weight size")?;
        let gate_up_len = checked_product(intermediate, h, "MLP weight size")?;
        let down_proj_len = checked_product(h, intermediate, "down projection weight size")?;
        for (index, layer) in weights.layers.iter().enumerate() {
            validate_weight_len(
                &format!("model.layers.{index}.self_attn.q_proj.weight"),
                layer.q_proj.len(),
                q_proj_len,
            )?;
            validate_weight_len(
                &format!("model.layers.{index}.self_attn.k_proj.weight"),
                layer.k_proj.len(),
                kv_proj_len,
            )?;
            validate_weight_len(
                &format!("model.layers.{index}.self_attn.v_proj.weight"),
                layer.v_proj.len(),
                kv_proj_len,
            )?;
            validate_weight_len(
                &format!("model.layers.{index}.self_attn.o_proj.weight"),
                layer.o_proj.len(),
                o_proj_len,
            )?;
            validate_weight_len(
                &format!("model.layers.{index}.mlp.gate_proj.weight"),
                layer.gate_proj.len(),
                gate_up_len,
            )?;
            validate_weight_len(
                &format!("model.layers.{index}.mlp.up_proj.weight"),
                layer.up_proj.len(),
                gate_up_len,
            )?;
            validate_weight_len(
                &format!("model.layers.{index}.mlp.down_proj.weight"),
                layer.down_proj.len(),
                down_proj_len,
            )?;
            validate_weight_len(
                &format!("model.layers.{index}.input_layernorm.weight"),
                layer.input_layernorm.len(),
                h,
            )?;
            validate_weight_len(
                &format!("model.layers.{index}.post_attention_layernorm.weight"),
                layer.post_attention_layernorm.len(),
                h,
            )?;
        }
        Ok(Self { cfg, weights })
    }

    /// Return the validated decoder configuration.
    pub fn config(&self) -> &Ernie45Config {
        &self.cfg
    }

    /// Full-sequence causal forward over `ids` (batch 1, positions
    /// `0..seq_len`, no cache), capturing every checkpoint the golden
    /// comparison reads.
    pub fn forward_trace(&self, ids: &[u32]) -> Result<Ernie45Trace, InferenceError> {
        let cfg = &self.cfg;
        let w = &self.weights;
        let s = ids.len();
        if s > MAX_SEQ_LEN {
            return Err(InferenceError::InvalidInput(format!(
                "ernie45 forward: sequence length {s} exceeds MAX_SEQ_LEN {MAX_SEQ_LEN}"
            )));
        }
        let h = cfg.hidden_size;
        if s == 0 {
            return Err(InferenceError::Inference(
                "ernie45 forward: empty token sequence".into(),
            ));
        }
        for &id in ids {
            if id as usize >= cfg.vocab_size {
                return Err(InferenceError::Inference(format!(
                    "ernie45 forward: token id {id} out of vocab range {}",
                    cfg.vocab_size
                )));
            }
        }

        let q_dim = cfg.q_dim()?;
        let kv_dim = cfg.kv_dim()?;
        let s_h = checked_product(s, h, "forward hidden buffer size")?;
        let s_q_dim = checked_product(s, q_dim, "forward query buffer size")?;
        let s_kv_dim = checked_product(s, kv_dim, "forward key/value buffer size")?;
        let s_intermediate =
            checked_product(s, cfg.intermediate_size, "forward intermediate buffer size")?;
        let s_vocab = checked_product(s, cfg.vocab_size, "forward logits buffer size")?;
        let s_s = checked_product(s, s, "forward attention score buffer size")?;
        let s_hd = checked_product(s, cfg.head_dim, "forward head buffer size")?;
        let hd_s = checked_product(cfg.head_dim, s, "forward transposed value buffer size")?;

        let mut x = vec![0f32; s_h];
        for (t, &id) in ids.iter().enumerate() {
            x[t * h..(t + 1) * h].copy_from_slice(&w.embed_tokens[id as usize * h..][..h]);
        }
        let embed = x.clone();

        let inv_freq = gemma4_rope_inv_freq(cfg.head_dim, cfg.rope_theta, None);
        let mut positions = Vec::new();
        positions
            .try_reserve_exact(s)
            .map_err(|error| reserve_error("position buffer", error))?;
        positions.extend(0..s as u32);
        let (cos, sin) = gemma4_rope_cos_sin(&inv_freq, &positions);

        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let groups = heads / kv_heads;
        let hd = cfg.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut layer_outputs = Vec::new();
        layer_outputs
            .try_reserve_exact(cfg.num_hidden_layers)
            .map_err(|error| reserve_error("trace layer outputs", error))?;
        let mut normed = vec![0f32; s_h];
        let mut q = vec![0f32; s_q_dim];
        let mut k = vec![0f32; s_kv_dim];
        let mut v = vec![0f32; s_kv_dim];
        let mut attn_out = vec![0f32; s_q_dim];
        let mut proj = vec![0f32; s_h];
        let mut gate = vec![0f32; s_intermediate];
        let mut up = vec![0f32; s_intermediate];
        let mut scores = vec![0f32; s_s];
        let mut q_h = vec![0f32; s_hd];
        let mut k_h = vec![0f32; s_hd];
        let mut v_t = vec![0f32; hd_s];
        let mut out_h = vec![0f32; s_hd];

        for layer in &w.layers {
            // Attention block.
            normed.copy_from_slice(&x);
            rms_norm(&mut normed, &layer.input_layernorm, h, cfg.rms_norm_eps);
            matmul_bt(&normed, &layer.q_proj, &mut q, s, h, q_dim);
            matmul_bt(&normed, &layer.k_proj, &mut k, s, h, kv_dim);
            matmul_bt(&normed, &layer.v_proj, &mut v, s, h, kv_dim);
            gemma4_apply_rope(&mut q, &cos, &sin, s, heads, hd);
            gemma4_apply_rope(&mut k, &cos, &sin, s, kv_heads, hd);

            // Causal GQA attention, batched through GEMM one KV head at a time. The softmax is
            // exact `f32::exp` deliberately (mirroring `gemma4_model.rs`'s
            // `softmax_row_fail_closed` and `gemma4_ops.rs`'s exact-tanh
            // rationale): the shared `softmax_attention` kernel's Schraudolph
            // `fast_exp` carries ~1% relative error, which this differential
            // gate measured as O(1e-2) activation divergence against the
            // exact-exp HF reference by layer 1.
            for kvh in 0..kv_heads {
                for tk in 0..s {
                    let k_row = &k[(tk * kv_heads + kvh) * hd..][..hd];
                    k_h[tk * hd..(tk + 1) * hd].copy_from_slice(k_row);
                    for d in 0..hd {
                        v_t[d * s + tk] = v[(tk * kv_heads + kvh) * hd + d];
                    }
                }
                for qh in kvh * groups..(kvh + 1) * groups {
                    for tq in 0..s {
                        let q_row = &q[(tq * heads + qh) * hd..][..hd];
                        q_h[tq * hd..(tq + 1) * hd].copy_from_slice(q_row);
                    }
                    matmul_bt(&q_h, &k_h, &mut scores, s, hd, s);
                    for tq in 0..s {
                        let row = &mut scores[tq * s..(tq + 1) * s];
                        for score in &mut row[..=tq] {
                            *score *= scale;
                        }
                        softmax_row_fail_closed(&mut row[..=tq]);
                        row[tq + 1..].fill(0.0);
                    }
                    matmul_bt(&scores, &v_t, &mut out_h, s, s, hd);
                    for tq in 0..s {
                        let out_row = &out_h[tq * hd..(tq + 1) * hd];
                        let dst = &mut attn_out[(tq * heads + qh) * hd..][..hd];
                        dst.copy_from_slice(out_row);
                    }
                }
            }
            matmul_bt(&attn_out, &layer.o_proj, &mut proj, s, q_dim, h);
            for (xi, pi) in x.iter_mut().zip(proj.iter()) {
                *xi += pi;
            }

            // MLP block.
            normed.copy_from_slice(&x);
            rms_norm(
                &mut normed,
                &layer.post_attention_layernorm,
                h,
                cfg.rms_norm_eps,
            );
            matmul_bt(
                &normed,
                &layer.gate_proj,
                &mut gate,
                s,
                h,
                cfg.intermediate_size,
            );
            matmul_bt(
                &normed,
                &layer.up_proj,
                &mut up,
                s,
                h,
                cfg.intermediate_size,
            );
            silu_inplace(&mut gate);
            for (g, &u) in gate.iter_mut().zip(up.iter()) {
                *g *= u;
            }
            matmul_bt(
                &gate,
                &layer.down_proj,
                &mut proj,
                s,
                cfg.intermediate_size,
                h,
            );
            for (xi, pi) in x.iter_mut().zip(proj.iter()) {
                *xi += pi;
            }

            layer_outputs.push(x.clone());
        }

        let mut final_normed = x;
        rms_norm(&mut final_normed, &w.final_norm, h, cfg.rms_norm_eps);
        let mut logits = vec![0f32; s_vocab];
        matmul_bt(&final_normed, &w.lm_head, &mut logits, s, h, cfg.vocab_size);

        Ok(Ernie45Trace {
            embed,
            layer_outputs,
            final_norm: final_normed,
            logits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct StubSource {
        tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
        materializations: usize,
    }

    impl StubSource {
        fn insert(&mut self, name: impl Into<String>, shape: Vec<usize>) {
            let elements = shape.iter().product();
            self.tensors
                .insert(name.into(), (shape, vec![0.0; elements]));
        }
    }

    impl TensorSource for StubSource {
        fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError> {
            Ok(self.tensors.contains_key(name))
        }

        fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
            Ok(self.tensors.get(name).map(|(shape, _)| shape.clone()))
        }

        fn get_f32_tensor_owned(
            &mut self,
            name: &str,
        ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
            self.materializations += 1;
            self.tensors
                .get(name)
                .map(|(shape, data)| (data.clone(), shape.clone()))
                .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
        }
    }

    fn tiny_config() -> Ernie45Config {
        Ernie45Config {
            hidden_size: 2,
            intermediate_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            vocab_size: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            rope_scaling: Ernie45RopeScaling {
                mrope_section: vec![1],
            },
            tie_word_embeddings: false,
            use_bias: false,
        }
    }

    fn full_source(cfg: &Ernie45Config) -> StubSource {
        let h = cfg.hidden_size;
        let q_dim = cfg.q_dim().unwrap();
        let kv_dim = cfg.kv_dim().unwrap();
        let mut source = StubSource::default();
        source.insert("model.embed_tokens.weight", vec![cfg.vocab_size, h]);
        source.insert("model.norm.weight", vec![h]);
        source.insert("lm_head.weight", vec![cfg.vocab_size, h]);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}.");
            source.insert(format!("{p}self_attn.q_proj.weight"), vec![q_dim, h]);
            source.insert(format!("{p}self_attn.k_proj.weight"), vec![kv_dim, h]);
            source.insert(format!("{p}self_attn.v_proj.weight"), vec![kv_dim, h]);
            source.insert(format!("{p}self_attn.o_proj.weight"), vec![h, q_dim]);
            source.insert(
                format!("{p}mlp.gate_proj.weight"),
                vec![cfg.intermediate_size, h],
            );
            source.insert(
                format!("{p}mlp.up_proj.weight"),
                vec![cfg.intermediate_size, h],
            );
            source.insert(
                format!("{p}mlp.down_proj.weight"),
                vec![h, cfg.intermediate_size],
            );
            source.insert(format!("{p}input_layernorm.weight"), vec![h]);
            source.insert(format!("{p}post_attention_layernorm.weight"), vec![h]);
        }
        source
    }

    #[test]
    fn validate_accepts_shipped_values_and_each_documented_cap() {
        let base = Ernie45Config {
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 18,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 128,
            vocab_size: 103_424,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            rope_scaling: Ernie45RopeScaling {
                mrope_section: vec![16, 24, 24],
            },
            tie_word_embeddings: false,
            use_bias: false,
        };
        assert!(base.validate().is_ok());

        let mut at_cap = base.clone();
        at_cap.hidden_size = MAX_HIDDEN_SIZE;
        assert!(at_cap.validate().is_ok());
        at_cap = base.clone();
        at_cap.intermediate_size = MAX_INTERMEDIATE_SIZE;
        assert!(at_cap.validate().is_ok());
        at_cap = base.clone();
        at_cap.num_hidden_layers = MAX_LAYERS;
        assert!(at_cap.validate().is_ok());
        at_cap = base.clone();
        at_cap.num_attention_heads = MAX_HEADS;
        assert!(at_cap.validate().is_ok());
        at_cap = base.clone();
        at_cap.num_key_value_heads = MAX_KV_HEADS;
        at_cap.num_attention_heads = MAX_KV_HEADS;
        assert!(at_cap.validate().is_ok());
        at_cap = base.clone();
        at_cap.head_dim = MAX_HEAD_DIM;
        at_cap.rope_scaling.mrope_section = vec![MAX_HEAD_DIM / 2];
        assert!(at_cap.validate().is_ok());
        at_cap = base.clone();
        at_cap.vocab_size = MAX_VOCAB_SIZE;
        assert!(at_cap.validate().is_ok());

        let mut over_cap = base.clone();
        over_cap.hidden_size = MAX_HIDDEN_SIZE + 1;
        assert!(over_cap.validate().is_err());
        over_cap = base.clone();
        over_cap.hidden_size = usize::MAX;
        assert!(over_cap.validate().is_err());
        over_cap = base.clone();
        over_cap.intermediate_size = MAX_INTERMEDIATE_SIZE + 1;
        assert!(over_cap.validate().is_err());
        over_cap.intermediate_size = usize::MAX;
        assert!(over_cap.validate().is_err());
        over_cap = base.clone();
        over_cap.num_hidden_layers = MAX_LAYERS + 1;
        assert!(over_cap.validate().is_err());
        over_cap.num_hidden_layers = usize::MAX;
        assert!(over_cap.validate().is_err());
        over_cap = base.clone();
        over_cap.num_attention_heads = MAX_HEADS + 1;
        assert!(over_cap.validate().is_err());
        over_cap.num_attention_heads = usize::MAX;
        assert!(over_cap.validate().is_err());
        over_cap = base.clone();
        over_cap.num_key_value_heads = MAX_KV_HEADS + 1;
        assert!(over_cap.validate().is_err());
        over_cap.num_key_value_heads = usize::MAX;
        assert!(over_cap.validate().is_err());
        over_cap = base.clone();
        over_cap.head_dim = MAX_HEAD_DIM + 1;
        assert!(over_cap.validate().is_err());
        over_cap.head_dim = usize::MAX;
        assert!(over_cap.validate().is_err());
        over_cap = base;
        over_cap.vocab_size = MAX_VOCAB_SIZE + 1;
        assert!(over_cap.validate().is_err());
        over_cap.vocab_size = usize::MAX;
        assert!(over_cap.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_dimensions() {
        let base = tiny_config();
        for field in [
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
        ] {
            let mut cfg = base.clone();
            match field {
                "hidden_size" => cfg.hidden_size = 0,
                "intermediate_size" => cfg.intermediate_size = 0,
                "num_hidden_layers" => cfg.num_hidden_layers = 0,
                "num_attention_heads" => cfg.num_attention_heads = 0,
                "num_key_value_heads" => cfg.num_key_value_heads = 0,
                "head_dim" => cfg.head_dim = 0,
                "vocab_size" => cfg.vocab_size = 0,
                _ => unreachable!(),
            }
            assert!(cfg.validate().is_err(), "{field} zero was accepted");
        }
    }

    #[test]
    fn load_tensor_preflights_wrong_shape_without_materialization() {
        let mut source = StubSource::default();
        source.insert("tensor", vec![3, 3]);
        let result = load_tensor(&mut source, "tensor", &[2, 2]);
        assert!(matches!(result, Err(InferenceError::ShapeMismatch { .. })));
        assert_eq!(source.materializations, 0);
    }

    #[test]
    fn load_tensor_preflights_missing_tensor_without_materialization() {
        let mut source = StubSource::default();
        let result = load_tensor(&mut source, "missing", &[2, 2]);
        assert!(matches!(result, Err(InferenceError::MissingTensor(name)) if name == "missing"));
        assert_eq!(source.materializations, 0);
    }

    #[test]
    fn model_new_rejects_mismatched_weight_lengths() {
        let cfg = tiny_config();
        let mut source = full_source(&cfg);
        let mut weights = Ernie45Weights::load(&mut source, &cfg).unwrap();
        weights.embed_tokens.pop();
        let result = Ernie45Model::new(cfg, weights);
        assert!(
            matches!(result, Err(InferenceError::Inference(message)) if message.contains("model.embed_tokens.weight"))
        );
    }

    #[test]
    fn forward_rejects_sequence_over_max_before_allocating_buffers() {
        let cfg = tiny_config();
        let mut source = full_source(&cfg);
        let weights = Ernie45Weights::load(&mut source, &cfg).unwrap();
        let model = Ernie45Model::new(cfg, weights).unwrap();
        let ids = vec![0u32; MAX_SEQ_LEN + 1];
        let result = model.forward_trace(&ids);
        assert!(
            matches!(result, Err(InferenceError::InvalidInput(message)) if message.contains("MAX_SEQ_LEN"))
        );
    }
}

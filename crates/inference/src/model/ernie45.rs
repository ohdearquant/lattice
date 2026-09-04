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
//! MLP, RMSNorm (`w * x * rsqrt(mean(x^2) + eps)`), untied `lm_head`. No
//! sampling here — this module exists to be held against the HF reference
//! activations captured in
//! `tests/fixtures/paddleocr_vl/decoder/decoder_goldens.json`, with greedy
//! generation built on top of the verified forward: the uncached entry
//! points re-run the whole sequence per step (the differential reference),
//! and the caller-owned [`Ernie45KvCache`] prefills once, then steps one
//! token at a time.
//!
//! RoPE: the reference applies Qwen2-VL-style multimodal-section RoPE
//! (`apply_multimodal_rotary_pos_emb`, `mrope_section` doubled and chunks
//! gathered per position row `i % 3`) with **stride-half** `rotate_half`
//! pairing. Every rotary lane `i` in the half dimension reads the cos/sin of
//! its token's position on row `i % 3` (`t`, `h`, `w`); the sections are
//! doubled to cover both halves of the stride-half layout. For text-only
//! input the reference model derives all three position rows from the same
//! `arange`, which makes the section gather a no-op: every lane reads
//! identical angles, so the result is bit-identical to plain 1-D neox RoPE
//! at `rope_theta` (unit-tested). The sectioned form becomes load-bearing
//! for the vision slice: `forward_embeds_trace` accepts pre-built
//! embeddings (with the vision projector rows spliced in) and per-token
//! 3-row positions from `paddleocr_vl::rope_index`.

use std::path::Path;

use serde::Deserialize;

use crate::error::InferenceError;
use crate::forward::cpu::{matmul_bt, rms_norm, silu_inplace};
use crate::model::gemma4_ops::{gemma4_apply_rope, gemma4_rope_inv_freq};
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

    /// `num_key_value_heads * head_dim`, checked. Crate-visible because it
    /// is also the width an [`Ernie45KvCache`] must be built for, and a
    /// caller recomputing that product by hand could disagree with the
    /// value `kv_prefill` validates against.
    pub(crate) fn kv_dim(&self) -> Result<usize, InferenceError> {
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

/// Build the `[seq_len * head_dim]` cos/sin tables for the reference's
/// sectioned multimodal RoPE.
///
/// Reference layout (`apply_multimodal_rotary_pos_emb`): `mrope_section`
/// is repeated twice to cover the stride-half pairing (lane `j` pairs with
/// `j + head_dim/2`, so both lanes of a pair must rotate by the same
/// angle), and lane `i` of the half dimension gathers the cos/sin of
/// position row `i % 3` over the tripled pattern `[t, h, w]`. With the
/// pinned section `[16, 24, 24]` doubled, lanes 0..15 read `t`, 16..39
/// read `h`, 40..63 read `w`, and the pattern repeats for lanes 64..127 —
/// identical angles at `j` and `j + 64`, so `gemma4_apply_rope`'s
/// stride-half rotation is unaffected. When a token's three rows agree,
/// the tables are bit-identical to `gemma4_rope_cos_sin` on that row
/// (verified by a unit test), which is why the text-only forward reduces
/// to plain 1-D RoPE.
fn mrope_cos_sin_table(cfg: &Ernie45Config, positions: &[[u32; 3]]) -> (Vec<f32>, Vec<f32>) {
    let inv_freq = gemma4_rope_inv_freq(cfg.head_dim, cfg.rope_theta, None);
    let s = positions.len();
    let head_dim = cfg.head_dim;
    let half = head_dim / 2;
    // Lane -> position-row map: the half dimension's `mrope_section`
    // (`[16, 24, 24]` pinned) is repeated twice to cover the stride-half
    // pairing, so lane `j` reads position row `j`'s section index —
    // `t, h, w, t, h, w` over the 128 lanes — and always uses frequency
    // `j % half`. The repetition is what keeps paired lanes `j` and
    // `j + half` on the same row.
    let mut lane_row = vec![0usize; head_dim];
    let mut j = 0;
    for _rep in 0..2 {
        for (row, &size) in cfg.rope_scaling.mrope_section.iter().enumerate() {
            for _ in 0..size {
                lane_row[j] = row;
                j += 1;
            }
        }
    }
    // Config validation guarantees `sum(mrope_section) == head_dim / 2`,
    // so the doubled sections tile the lanes exactly.
    assert_eq!(j, head_dim);
    let mut cos = vec![0f32; s * head_dim];
    let mut sin = vec![0f32; s * head_dim];
    for (t, pos) in positions.iter().enumerate() {
        for j in 0..head_dim {
            let angle = pos[lane_row[j]] as f32 * inv_freq[j % half];
            let (sn, cs) = angle.sin_cos();
            cos[t * head_dim + j] = cs;
            sin[t * head_dim + j] = sn;
        }
    }
    (cos, sin)
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

    /// The decoder's embedding table (`[vocab_size, hidden_size]`):
    /// the multimodal prefill splices vision rows over copies of these.
    pub fn embed_tokens(&self) -> &[f32] {
        &self.weights.embed_tokens
    }

    /// Full-sequence causal forward over `ids` (batch 1, positions
    /// `0..seq_len`, no cache), capturing every checkpoint the golden
    /// comparison reads. Thin wrapper over [`Self::forward_embeds_trace`]:
    /// the embedding rows come from `embed_tokens` and all three mrope
    /// position rows are the plain `arange`, so the sectioned gather
    /// reduces to 1-D RoPE — bit-identical to the pre-vision path.
    pub fn forward_trace(&self, ids: &[u32]) -> Result<Ernie45Trace, InferenceError> {
        let s = ids.len();
        if s > MAX_SEQ_LEN {
            return Err(InferenceError::InvalidInput(format!(
                "ernie45 forward: sequence length {s} exceeds MAX_SEQ_LEN {MAX_SEQ_LEN}"
            )));
        }
        let h = self.cfg.hidden_size;
        if s == 0 {
            return Err(InferenceError::Inference(
                "ernie45 forward: empty token sequence".into(),
            ));
        }
        for &id in ids {
            if id as usize >= self.cfg.vocab_size {
                return Err(InferenceError::Inference(format!(
                    "ernie45 forward: token id {id} out of vocab range {}",
                    self.cfg.vocab_size
                )));
            }
        }
        let embed_len = checked_product(s, h, "forward hidden buffer size")?;
        let mut embeds = vec![0f32; embed_len];
        for (t, &id) in ids.iter().enumerate() {
            embeds[t * h..(t + 1) * h]
                .copy_from_slice(&self.weights.embed_tokens[id as usize * h..][..h]);
        }
        let mut positions = Vec::new();
        positions
            .try_reserve_exact(s)
            .map_err(|error| reserve_error("position buffer", error))?;
        positions.extend((0..s as u32).map(|i| [i, i, i]));
        self.forward_embeds_trace(&embeds, &positions)
    }

    /// Full-sequence causal forward over pre-built token embeddings and
    /// per-token 3-row mrope positions (batch 1, no cache), capturing every
    /// checkpoint the golden comparison reads.
    ///
    /// `embeds` is `[seq_len, hidden_size]` row-major — the vision slice
    /// splices the projector rows into the image-placeholder positions
    /// before calling. `positions` is one `[t, h, w]` triple per token, in
    /// the reference's `get_rope_index` layout. The head's lanes are split
    /// into `mrope_section` chunks (doubled for the stride-half pairing);
    /// each chunk takes the cos/sin of the position row it names —
    /// `t, h, w, t, h, w` — at that lane's own frequency. When all three rows
    /// of every token agree (text only), the gather is a no-op and this is
    /// bit-identical to the 1-D path — the invariant that keeps the text
    /// decoder gate valid.
    ///
    /// # Errors
    ///
    /// [`InferenceError::Inference`] on empty input or a buffer/position
    /// length that disagrees with the config.
    pub fn forward_embeds_trace(
        &self,
        embeds: &[f32],
        positions: &[[u32; 3]],
    ) -> Result<Ernie45Trace, InferenceError> {
        let core = self.forward_embeds_core(embeds, positions, true, None)?;
        Ok(Ernie45Trace {
            embed: core.embed,
            layer_outputs: core.layer_outputs,
            final_norm: core.final_norm,
            logits: core.logits,
        })
    }

    /// Same forward as [`Self::forward_embeds_trace`], but returns only the
    /// last row of logits (`vocab_size` values) instead of materializing
    /// `[seq_len, vocab_size]`, and skips every intermediate-activation
    /// clone the trace keeps for the golden comparison. Every validation
    /// [`Self::forward_embeds_trace`] performs — `MAX_SEQ_LEN`, empty input,
    /// `embeds` length, non-finite `embeds` — runs identically here: both
    /// entry points share [`Self::forward_embeds_core`]'s front half.
    ///
    /// Intended for a greedy decode loop that re-forwards the whole
    /// sequence per step and only ever reads the last row's logits.
    ///
    /// # Errors
    ///
    /// Same as [`Self::forward_embeds_trace`].
    pub fn forward_embeds_last_logits(
        &self,
        embeds: &[f32],
        positions: &[[u32; 3]],
    ) -> Result<Vec<f32>, InferenceError> {
        let core = self.forward_embeds_core(embeds, positions, false, None)?;
        Ok(core.logits)
    }

    /// Fill a fresh [`Ernie45KvCache`] with the prompt's post-RoPE keys and
    /// values and return the last row's logits (`vocab_size` values).
    ///
    /// This is the full-sequence causal forward over `embeds` and
    /// `positions` — the same arithmetic as
    /// [`Self::forward_embeds_last_logits`], which stays the uncached
    /// differential reference — with one addition: each layer's post-RoPE
    /// key rows and value rows are copied into the cache at rows
    /// `0..seq_len` as they are produced, so every subsequent
    /// [`Self::kv_decode_step`] attends over them without recomputing the
    /// prompt. The returned logits are therefore bit-for-bit what
    /// [`Self::forward_embeds_last_logits`] returns for the same input:
    /// the sink only copies rows the kernel already computed.
    ///
    /// `cache` must be empty ([`Ernie45KvCache::is_empty`]) and large
    /// enough for `positions.len()` tokens; the cache is caller-owned and
    /// the model keeps no state of its own.
    ///
    /// Build a cache laid out for this model, sized for `capacity` tokens.
    ///
    /// [`Ernie45KvCache::new`] takes the layer count and the per-token KV width
    /// as raw numbers, and the checked helper that derives the second one from a
    /// config is crate-private. An out-of-crate caller therefore had to re-derive
    /// `num_key_value_heads * head_dim` by hand, unchecked, and a layout that
    /// disagrees with the model's is not rejected at construction: it surfaces
    /// later and indirectly, as a shape mismatch inside a prefill. Going through
    /// the model removes both the duplicated arithmetic and that failure mode.
    ///
    /// # Errors
    ///
    /// [`InferenceError::Inference`] on a zero dimension or a checked-arithmetic
    /// overflow in the layout, propagated from [`Ernie45KvCache::new`].
    pub fn new_kv_cache(&self, capacity: usize) -> Result<Ernie45KvCache, InferenceError> {
        Ernie45KvCache::new(self.cfg.num_hidden_layers, self.cfg.kv_dim()?, capacity)
    }

    /// # Errors
    ///
    /// [`InferenceError::Inference`] on an empty sequence, a sequence that
    /// exceeds the cache capacity, a non-empty or shape-mismatched cache,
    /// or a checked-arithmetic overflow; [`InferenceError::InvalidInput`]
    /// on an `embeds` length that disagrees with `positions` and the
    /// config.
    pub fn kv_prefill(
        &self,
        embeds: &[f32],
        positions: &[[u32; 3]],
        cache: &mut Ernie45KvCache,
    ) -> Result<Vec<f32>, InferenceError> {
        let s = positions.len();
        if s == 0 {
            return Err(InferenceError::Inference(
                "ernie45 kv prefill: empty token sequence".into(),
            ));
        }
        let kv_dim = self.cfg.kv_dim()?;
        self.check_cache_shape(cache, kv_dim)?;
        if s > cache.capacity {
            return Err(InferenceError::Inference(format!(
                "ernie45 kv prefill: {} prompt tokens exceed the cache capacity {capacity}",
                s,
                capacity = cache.capacity
            )));
        }
        if !cache.is_empty() {
            return Err(InferenceError::Inference(format!(
                "ernie45 kv prefill: cache already holds {} tokens; prefill requires an empty cache",
                cache.len()
            )));
        }
        let h = self.cfg.hidden_size;
        let embed_len = checked_product(s, h, "kv prefill hidden buffer size")?;
        if embeds.len() != embed_len {
            return Err(InferenceError::InvalidInput(format!(
                "ernie45 kv prefill: embeds has {} values; expected {} tokens x {} hidden",
                embeds.len(),
                s,
                h
            )));
        }
        if !embeds.iter().all(|v| v.is_finite()) {
            return Err(InferenceError::Inference(
                "ernie45 kv prefill: embeds carry a non-finite value".into(),
            ));
        }
        // Per-layer views of the cache's two row stores; the field borrows
        // end before `cache.len` is written below. `cap_kv` is one layer's
        // footprint in elements.
        let cap_kv = checked_product(cache.capacity, kv_dim, "kv cache per-layer capacity")?;
        let k_sink = &mut cache.layer_k;
        let v_sink = &mut cache.layer_v;
        let core =
            self.forward_embeds_core(embeds, positions, false, Some((k_sink, v_sink, cap_kv)))?;
        cache.len = s;
        Ok(core.logits)
    }

    /// Decode one token against a prefilled [`Ernie45KvCache`]: append the
    /// token's post-RoPE key and value rows to the cache, run the decoder
    /// over that single token with its attention attending over cache rows
    /// `0..=cache.len()`, and return its logits (`vocab_size` values).
    ///
    /// `embeds` is exactly one row of `hidden_size` values (the embedding,
    /// or spliced projector row, of the new token); `position` is its
    /// 3-row mrope position, the same triple the uncached path would have
    /// used for that token. Per layer the arithmetic mirrors
    /// [`Self::forward_embeds_core`] for a one-token sequence — same
    /// norm/projection/MLP calls, same exact-exp softmax — except the
    /// attention reads its keys and values from the cache instead of from
    /// this call's own projections, which is precisely the KV-cache
    /// invariant: the cached rows are bit-for-bit the rows the full
    /// forward would have computed.
    ///
    /// # Errors
    ///
    /// [`InferenceError::Inference`] on an empty cache (no prefill ran), a
    /// cache at capacity, a shape-mismatched cache, a non-finite `embeds`,
    /// or a checked-arithmetic overflow; [`InferenceError::InvalidInput`]
    /// on an `embeds` length that disagrees with the config (the same
    /// split [`Self::kv_prefill`] uses).
    pub fn kv_decode_step(
        &self,
        embeds: &[f32],
        position: [u32; 3],
        cache: &mut Ernie45KvCache,
    ) -> Result<Vec<f32>, InferenceError> {
        let h = self.cfg.hidden_size;
        let len = cache.len();
        if len == 0 {
            return Err(InferenceError::Inference(
                "ernie45 kv step: cache is empty; run kv_prefill first".into(),
            ));
        }
        if len == cache.capacity {
            return Err(InferenceError::Inference(format!(
                "ernie45 kv step: cache is full ({} tokens at capacity {capacity}); the sequence would overflow",
                len,
                capacity = cache.capacity
            )));
        }
        let kv_dim = self.cfg.kv_dim()?;
        self.check_cache_shape(cache, kv_dim)?;
        if embeds.len() != h {
            return Err(InferenceError::InvalidInput(format!(
                "ernie45 kv step: embeds has {} values; expected 1 token x {} hidden",
                embeds.len(),
                h
            )));
        }
        if !embeds.iter().all(|v| v.is_finite()) {
            return Err(InferenceError::Inference(
                "ernie45 kv step: embeds carry a non-finite value".into(),
            ));
        }
        self.kv_decode_core(embeds, position, cache, kv_dim)
    }

    /// The cached decode kernel: one token through every layer, attention
    /// reading keys/values from `cache` rows `0..=cache.len()`.
    fn kv_decode_core(
        &self,
        embeds: &[f32],
        position: [u32; 3],
        cache: &mut Ernie45KvCache,
        kv_dim: usize,
    ) -> Result<Vec<f32>, InferenceError> {
        let cfg = &self.cfg;
        let w = &self.weights;
        let h = cfg.hidden_size;
        let len = cache.len();
        // The caller checked len < capacity, so len + 1 cannot wrap.
        let keys = len + 1;

        let q_dim = cfg.q_dim()?;
        let s_q_dim = checked_product(1, q_dim, "kv step query buffer size")?;
        let s_kv_dim = checked_product(1, kv_dim, "kv step key/value buffer size")?;
        let s_h = checked_product(1, h, "kv step hidden buffer size")?;
        let s_intermediate =
            checked_product(1, cfg.intermediate_size, "kv step intermediate buffer size")?;
        let s_keys = checked_product(1, keys, "kv step attention score buffer size")?;
        let keys_hd = checked_product(keys, cfg.head_dim, "kv step key head buffer size")?;
        let hd_keys = checked_product(cfg.head_dim, keys, "kv step transposed value buffer size")?;
        let hd = cfg.head_dim;

        let mut x = embeds.to_vec();
        let (cos, sin) = mrope_cos_sin_table(cfg, &[position]);

        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let groups = heads / kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let cap_kv = checked_product(cache.capacity, kv_dim, "kv cache per-layer capacity")?;

        let mut normed = vec![0f32; s_h];
        let mut q = vec![0f32; s_q_dim];
        let mut k = vec![0f32; s_kv_dim];
        let mut v = vec![0f32; s_kv_dim];
        let mut attn_out = vec![0f32; s_q_dim];
        let mut proj = vec![0f32; s_h];
        let mut gate = vec![0f32; s_intermediate];
        let mut up = vec![0f32; s_intermediate];
        let mut scores = vec![0f32; s_keys];
        let mut q_h = vec![0f32; hd];
        let mut k_h = vec![0f32; keys_hd];
        let mut v_t = vec![0f32; hd_keys];
        let mut out_h = vec![0f32; hd];

        for (layer_idx, layer) in w.layers.iter().enumerate() {
            // Attention block: same norm/projection/RoPE calls as
            // `forward_embeds_core` at s == 1.
            normed.copy_from_slice(&x);
            rms_norm(&mut normed, &layer.input_layernorm, h, cfg.rms_norm_eps);
            matmul_bt(&normed, &layer.q_proj, &mut q, 1, h, q_dim);
            matmul_bt(&normed, &layer.k_proj, &mut k, 1, h, kv_dim);
            matmul_bt(&normed, &layer.v_proj, &mut v, 1, h, kv_dim);
            gemma4_apply_rope(&mut q, &cos, &sin, 1, heads, hd);
            gemma4_apply_rope(&mut k, &cos, &sin, 1, kv_heads, hd);

            // Append this token's rows to the cache at row `len`, in the
            // same `[token, kv_head, head_dim]` layout the kernel's own
            // `k` / `v` buffers use.
            let row_base = checked_product(
                checked_product(layer_idx, cache.capacity, "kv cache layer offset")?,
                kv_dim,
                "kv cache row offset",
            )? + checked_product(len, kv_dim, "kv cache row offset")?;
            cache.layer_k[row_base..row_base + kv_dim].copy_from_slice(&k);
            cache.layer_v[row_base..row_base + kv_dim].copy_from_slice(&v);

            // Causal GQA attention over the cached keys. Every cached key
            // is at or before this token's position, so there is no
            // masking row to zero (the uncached path's `row[tq + 1..]`
            // tail). The exact `f32::exp` softmax and the one-KV-head-at-a-
            // time `k_h` / `v_t` scratch copies are the prefill kernel's.
            let layer_k = &cache.layer_k[layer_idx * cap_kv..][..keys * kv_dim];
            let layer_v = &cache.layer_v[layer_idx * cap_kv..][..keys * kv_dim];
            for kvh in 0..kv_heads {
                for tk in 0..keys {
                    let k_row = &layer_k[tk * kv_dim + kvh * hd..][..hd];
                    k_h[tk * hd..(tk + 1) * hd].copy_from_slice(k_row);
                    for d in 0..hd {
                        v_t[d * keys + tk] = layer_v[tk * kv_dim + kvh * hd + d];
                    }
                }
                for qh in kvh * groups..(kvh + 1) * groups {
                    let q_row = &q[qh * hd..(qh + 1) * hd];
                    q_h.copy_from_slice(q_row);
                    matmul_bt(&q_h, &k_h, &mut scores, 1, hd, keys);
                    for score in &mut scores[..keys] {
                        *score *= scale;
                    }
                    softmax_row_fail_closed(&mut scores[..keys]);
                    matmul_bt(&scores, &v_t, &mut out_h, 1, keys, hd);
                    attn_out[qh * hd..(qh + 1) * hd].copy_from_slice(&out_h);
                }
            }
            matmul_bt(&attn_out, &layer.o_proj, &mut proj, 1, q_dim, h);
            for (xi, pi) in x.iter_mut().zip(proj.iter()) {
                *xi += pi;
            }

            // MLP block: identical calls to `forward_embeds_core` at s == 1.
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
                1,
                h,
                cfg.intermediate_size,
            );
            matmul_bt(
                &normed,
                &layer.up_proj,
                &mut up,
                1,
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
                1,
                cfg.intermediate_size,
                h,
            );
            for (xi, pi) in x.iter_mut().zip(proj.iter()) {
                *xi += pi;
            }
        }

        let mut final_normed = x;
        rms_norm(&mut final_normed, &w.final_norm, h, cfg.rms_norm_eps);
        let mut logits = vec![0f32; cfg.vocab_size];
        matmul_bt(&final_normed, &w.lm_head, &mut logits, 1, h, cfg.vocab_size);
        cache.len = len + 1;
        Ok(logits)
    }

    /// Fail-closed cache/model shape check, run before any allocation: a
    /// cache built from a different checkpoint layout would index the
    /// layer buffers at wrong offsets.
    fn check_cache_shape(
        &self,
        cache: &Ernie45KvCache,
        kv_dim: usize,
    ) -> Result<(), InferenceError> {
        if cache.layers != self.cfg.num_hidden_layers || cache.kv_dim != kv_dim {
            return Err(InferenceError::Inference(format!(
                "ernie45 kv: cache was built for {} layers x kv_dim {}, this model has {} layers x kv_dim {}",
                cache.layers, cache.kv_dim, self.cfg.num_hidden_layers, kv_dim
            )));
        }
        Ok(())
    }

    /// Shared body of [`Self::forward_embeds_trace`] and
    /// [`Self::forward_embeds_last_logits`]. `keep_trace == true` reproduces
    /// today's trace exactly (every clone kept, `logits` is `[seq_len,
    /// vocab]`); `keep_trace == false` skips the `embed` clone and every
    /// per-layer `layer_outputs` clone, and applies `lm_head` to the last
    /// row only, allocating a `vocab`-length `logits` buffer instead of
    /// `seq_len * vocab`.
    ///
    /// `kv_sink`, when `Some`, is `(&mut layer_k, &mut layer_v, cap_kv)`:
    /// the caller cache's per-layer key and value row stores, each
    /// `[layers * cap_kv]` with `cap_kv = capacity * kv_dim`, and `cap_kv`
    /// the element stride between layers. After each layer's post-RoPE
    /// `k` / `v` buffers are produced, every token's row is copied into
    /// both stores at row `token` — the pure-memcpy side effect of
    /// [`Self::kv_prefill`]. `None` (the uncached paths) skips it, so the
    /// no-cache forward is bit-for-bit the pre-sink kernel.
    fn forward_embeds_core(
        &self,
        embeds: &[f32],
        positions: &[[u32; 3]],
        keep_trace: bool,
        mut kv_sink: Option<(&mut Vec<f32>, &mut Vec<f32>, usize)>,
    ) -> Result<Ernie45CoreOut, InferenceError> {
        let cfg = &self.cfg;
        let w = &self.weights;
        let s = positions.len();
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
        let embed_len = checked_product(s, h, "forward hidden buffer size")?;
        if embeds.len() != embed_len {
            return Err(InferenceError::InvalidInput(format!(
                "ernie45 forward: embeds has {} values; expected {} tokens x {} hidden",
                embeds.len(),
                s,
                h
            )));
        }
        if !embeds.iter().all(|v| v.is_finite()) {
            return Err(InferenceError::Inference(
                "ernie45 forward: embeds carry a non-finite value".into(),
            ));
        }

        let q_dim = cfg.q_dim()?;
        let kv_dim = cfg.kv_dim()?;
        let s_h = checked_product(s, h, "forward hidden buffer size")?;
        let s_q_dim = checked_product(s, q_dim, "forward query buffer size")?;
        let s_kv_dim = checked_product(s, kv_dim, "forward key/value buffer size")?;
        let s_intermediate =
            checked_product(s, cfg.intermediate_size, "forward intermediate buffer size")?;
        let s_vocab = if keep_trace {
            checked_product(s, cfg.vocab_size, "forward logits buffer size")?
        } else {
            cfg.vocab_size
        };
        let s_s = checked_product(s, s, "forward attention score buffer size")?;
        let s_hd = checked_product(s, cfg.head_dim, "forward head buffer size")?;
        let hd_s = checked_product(cfg.head_dim, s, "forward transposed value buffer size")?;

        let mut x = embeds.to_vec();
        let embed = if keep_trace { x.clone() } else { Vec::new() };

        let (cos, sin) = mrope_cos_sin_table(cfg, positions);

        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let groups = heads / kv_heads;
        let hd = cfg.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut layer_outputs = Vec::new();
        if keep_trace {
            layer_outputs
                .try_reserve_exact(cfg.num_hidden_layers)
                .map_err(|error| reserve_error("trace layer outputs", error))?;
        }
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

        for (layer_idx, layer) in w.layers.iter().enumerate() {
            // Attention block.
            normed.copy_from_slice(&x);
            rms_norm(&mut normed, &layer.input_layernorm, h, cfg.rms_norm_eps);
            matmul_bt(&normed, &layer.q_proj, &mut q, s, h, q_dim);
            matmul_bt(&normed, &layer.k_proj, &mut k, s, h, kv_dim);
            matmul_bt(&normed, &layer.v_proj, &mut v, s, h, kv_dim);
            gemma4_apply_rope(&mut q, &cos, &sin, s, heads, hd);
            gemma4_apply_rope(&mut k, &cos, &sin, s, kv_heads, hd);

            if let Some((k_sink, v_sink, cap_kv)) = kv_sink.as_mut() {
                let layer_base = checked_product(layer_idx, *cap_kv, "kv cache layer offset")?;
                let (k_chunks, v_chunks) = (k.chunks_exact(kv_dim), v.chunks_exact(kv_dim));
                for (tk, (k_row, v_row)) in k_chunks.zip(v_chunks).enumerate() {
                    let dst = layer_base + tk * kv_dim;
                    k_sink[dst..dst + kv_dim].copy_from_slice(k_row);
                    v_sink[dst..dst + kv_dim].copy_from_slice(v_row);
                }
            }

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

            if keep_trace {
                layer_outputs.push(x.clone());
            }
        }

        let mut final_normed = x;
        rms_norm(&mut final_normed, &w.final_norm, h, cfg.rms_norm_eps);
        let mut logits = vec![0f32; s_vocab];
        if keep_trace {
            matmul_bt(&final_normed, &w.lm_head, &mut logits, s, h, cfg.vocab_size);
        } else {
            matmul_bt(
                &final_normed[(s - 1) * h..],
                &w.lm_head,
                &mut logits,
                1,
                h,
                cfg.vocab_size,
            );
        }

        Ok(Ernie45CoreOut {
            embed,
            layer_outputs,
            final_norm: if keep_trace { final_normed } else { Vec::new() },
            logits,
        })
    }
}

/// Internal output of [`Ernie45Model::forward_embeds_core`], shared by the
/// full-trace and last-row-logits entry points. `embed`, `layer_outputs`,
/// and `final_norm` are only populated when `keep_trace == true`; `logits`
/// is `[seq_len, vocab_size]` when `keep_trace == true` and `vocab_size`
/// (the last row only) otherwise.
struct Ernie45CoreOut {
    embed: Vec<f32>,
    layer_outputs: Vec<Vec<f32>>,
    final_norm: Vec<f32>,
    logits: Vec<f32>,
}

/// Caller-owned KV cache for the ERNIE-4.5 decoder's cached decode:
/// `kv_prefill` fills it with the prompt's post-RoPE keys and values, and
/// `kv_decode_step` appends one row per step.
///
/// Layout: per layer, the post-RoPE key rows and value rows for tokens
/// `0..len`, each row `[kv_head, head_dim]` — the exact layout the
/// uncached kernel's per-layer `k` / `v` buffers use, so the attention
/// code reads cached rows with the same indexing and no second layout.
/// The two stores are sized for `capacity` tokens in [`Self::new`] and
/// are only ever written into afterwards, so the cache itself never
/// reallocates or grows as the sequence advances. That is a statement
/// about the cache, not about the step: [`Ernie45Model::kv_decode_step`]
/// still allocates its own per-call scratch buffers, exactly as the
/// uncached kernel does, and reusing those across steps is a separate
/// change.
///
/// The cache is opaque and explicit — [`Ernie45Model`] keeps no hidden
/// state, and every entry point re-validates the cache's shape before
/// touching it (a cache built for a different checkpoint layout is
/// rejected rather than indexed at wrong offsets).
///
/// Why a type of its own rather than
/// [`FlatKVCache`][crate::kv_cache::flat::FlatKVCache]: that cache stores
/// f16 and dequantises on read, which is the right trade where memory is
/// the constraint, but it cannot carry this decoder's contract — the
/// cached rows must be *bit-for-bit* the rows the uncached forward
/// computed, which is what
/// [`Ernie45Model::kv_prefill`]'s equality with
/// [`Ernie45Model::forward_embeds_last_logits`] rests on. Storing f32
/// keeps that equality exact. `model/gemma4_cache.rs` holds its own f32
/// cache for the same reason.
pub struct Ernie45KvCache {
    layers: usize,
    capacity: usize,
    kv_dim: usize,
    /// Number of tokens currently stored (`0..=capacity`).
    len: usize,
    /// Post-RoPE key rows, `[layers * capacity * kv_dim]`.
    layer_k: Vec<f32>,
    /// Value rows, `[layers * capacity * kv_dim]`.
    layer_v: Vec<f32>,
}

impl Ernie45KvCache {
    /// Allocate an empty cache for `num_hidden_layers` layers, each row of
    /// `kv_dim` values (the model's `num_key_value_heads * head_dim`),
    /// holding up to `capacity` tokens.
    ///
    /// # Errors
    ///
    /// [`InferenceError::Inference`] on zero dimensions or a
    /// checked-arithmetic overflow; the OS returning no memory for the
    /// allocation.
    pub fn new(
        num_hidden_layers: usize,
        kv_dim: usize,
        capacity: usize,
    ) -> Result<Self, InferenceError> {
        for (name, value) in [
            ("num_hidden_layers", num_hidden_layers),
            ("kv_dim", kv_dim),
            ("capacity", capacity),
        ] {
            if value == 0 {
                return Err(InferenceError::Inference(format!(
                    "ernie45 kv cache: {name} must be positive, got 0"
                )));
            }
        }
        let per_layer = checked_product(
            checked_product(num_hidden_layers, capacity, "kv cache token capacity")?,
            kv_dim,
            "kv cache row store",
        )?;
        let mut layer_k = Vec::new();
        let mut layer_v = Vec::new();
        layer_k
            .try_reserve_exact(per_layer)
            .map_err(|error| reserve_error("kv cache key rows", error))?;
        layer_k.resize(per_layer, 0.0);
        layer_v
            .try_reserve_exact(per_layer)
            .map_err(|error| reserve_error("kv cache value rows", error))?;
        layer_v.resize(per_layer, 0.0);
        Ok(Self {
            layers: num_hidden_layers,
            capacity,
            kv_dim,
            len: 0,
            layer_k,
            layer_v,
        })
    }

    /// Tokens currently stored, `0..=capacity`.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no token has been stored yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The token count this cache was allocated for.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The decoder layer count this cache was built for.
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Per-token key/value projection width this cache was built for.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Drop the stored rows without releasing the allocation, so the cache can
    /// serve another prompt.
    ///
    /// Without this a caller decoding a second prompt has to drop the cache and
    /// build a new one, because [`Ernie45Model::kv_prefill`] refuses a non-empty
    /// cache and the stores are private. That throws away buffers which were
    /// sized correctly and are about to be refilled with the same shapes, which
    /// is the ordinary serving pattern rather than an edge case.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Roll the stored row count back to `len`, keeping the allocation.
    ///
    /// Rows above `len` are left in place rather than zeroed, and that is sound
    /// rather than a shortcut: every read of the stores is bounded by the live
    /// row count (the decode step attends over `len + 1` keys and slices
    /// `..keys * kv_dim`), and each appended row is written at row `len` before
    /// any read of it. So a stale row above `len` is unreachable until it is
    /// overwritten by the append that makes it live again.
    ///
    /// Growing is not truncation: `len` above the current length is refused
    /// rather than clamped, because the rows it would expose were never written
    /// by this sequence and clamping would silently hand back a shorter cache
    /// than the caller asked for.
    pub fn truncate(&mut self, len: usize) -> Result<(), InferenceError> {
        if len > self.len {
            return Err(InferenceError::Inference(format!(
                "ernie45 kv cache truncate: requested length {len} exceeds the {} rows stored; \
                 truncation cannot grow a cache",
                self.len
            )));
        }
        self.len = len;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::gemma4_ops::gemma4_rope_cos_sin;
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

    fn test_config(head_dim: usize, mrope: &[usize]) -> Ernie45Config {
        Ernie45Config {
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim,
            vocab_size: 300,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_scaling: Ernie45RopeScaling {
                mrope_section: mrope.to_vec(),
            },
            tie_word_embeddings: false,
            use_bias: false,
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
    fn config_rejects_section_layout_that_does_not_tile_half_dim() {
        let bad = test_config(128, &[16, 24]);
        assert!(bad.validate().is_err(), "sum 40 != 64 must be refused");
        let good = test_config(128, &[16, 24, 24]);
        assert!(good.validate().is_ok());
    }

    /// The sectioned gather must be a no-op when all three position rows
    /// agree: bit-identical to the plain 1-D cos/sin tables. This is the
    /// invariant that keeps the text-only decoder gate on the same
    /// arithmetic as the pre-vision forward.
    #[test]
    fn mrope_table_equals_1d_rope_when_rows_agree() {
        let cfg = test_config(128, &[16, 24, 24]);
        let positions: Vec<[u32; 3]> = (0..9u32).map(|i| [i, i, i]).collect();
        let (mcos, msin) = mrope_cos_sin_table(&cfg, &positions);
        let inv = gemma4_rope_inv_freq(128, 1_000_000.0, None);
        let (cos1, sin1) =
            gemma4_rope_cos_sin(&inv, &positions.iter().map(|p| p[0]).collect::<Vec<_>>());
        assert_eq!(mcos, cos1, "cos tables diverge on equal rows");
        assert_eq!(msin, sin1, "sin tables diverge on equal rows");
    }

    /// The reference's pinned `[16, 24, 24]` section: lane 0..15 -> t,
    /// 16..39 -> h, 40..63 -> w, and the pattern repeats at 64..127 so the
    /// stride-half pair (j, j+64) always rotates by the same angle.
    #[test]
    fn mrope_table_section_rows_follow_pinned_layout() {
        let cfg = test_config(128, &[16, 24, 24]);
        // One token at position [3, 5, 7]: rows disagree, so a wrong row
        // assignment changes the table.
        let positions = [[3u32, 5, 7]];
        let (cos, _sin) = mrope_cos_sin_table(&cfg, &positions);
        let inv = gemma4_rope_inv_freq(128, 1_000_000.0, None);
        let expect = |lane: usize| {
            let row = match lane % 64 {
                0..16 => 3f32,
                16..40 => 5.0,
                _ => 7.0,
            };
            (row * inv[lane % 64]).cos()
        };
        for lane in [0usize, 15, 16, 39, 40, 63, 64, 79, 103, 127] {
            assert!(
                (cos[lane] - expect(lane)).abs() < 1e-6,
                "lane {lane}: {} vs {} — row assignment",
                cos[lane],
                expect(lane)
            );
        }
        // A lane reading the wrong position row would shift its angle by
        // at least 2 * inv_freq[j] * |pos diff| — far above 1e-6 for the
        // frequencies this test samples.
        assert!(
            (cos[0] - cos[16]).abs() > 1e-4,
            "t and h rows must produce different angles"
        );
    }

    #[test]
    fn forward_trace_matches_embedding_forward_on_arange_positions() {
        let cfg = tiny_config();
        let mut source = full_source(&cfg);
        let weights = Ernie45Weights::load(&mut source, &cfg).unwrap();
        let model = Ernie45Model::new(cfg, weights).unwrap();
        let ids = [0u32, 1];
        let positions = [[0u32, 0, 0], [1, 1, 1]];
        let embeds = ids
            .iter()
            .flat_map(|&id| model.embed_tokens()[id as usize * 2..][..2].iter().copied())
            .collect::<Vec<_>>();

        let id_trace = model.forward_trace(&ids).unwrap();
        let embed_trace = model.forward_embeds_trace(&embeds, &positions).unwrap();
        assert_eq!(id_trace.embed, embed_trace.embed);
        assert_eq!(id_trace.layer_outputs, embed_trace.layer_outputs);
        assert_eq!(id_trace.final_norm, embed_trace.final_norm);
        assert_eq!(id_trace.logits, embed_trace.logits);
    }

    #[test]
    fn forward_embeds_trace_rejects_wrong_lengths() {
        let cfg = test_config(64, &[16]);
        // sum(mrope) must tile head_dim/2 = 32: fix config for this test.
        let cfg = Ernie45Config {
            rope_scaling: Ernie45RopeScaling {
                mrope_section: vec![8, 8, 8, 8],
            },
            ..cfg
        };
        assert!(cfg.validate().is_ok());
        let weights = Ernie45Weights {
            embed_tokens: vec![0f32; 300 * 128],
            layers: vec![
                layer_weights(128, 256, 8 * 64, 2 * 64),
                layer_weights(128, 256, 8 * 64, 2 * 64),
            ],
            final_norm: vec![1f32; 128],
            lm_head: vec![0f32; 300 * 128],
        };
        let model = Ernie45Model::new(cfg, weights).unwrap();
        assert!(model.forward_embeds_trace(&[], &[]).is_err());
        assert!(
            model
                .forward_embeds_trace(&[0.0f32; 128], &[[0, 0, 0], [1, 1, 1]])
                .is_err()
        );
    }

    fn layer_weights(h: usize, inter: usize, qd: usize, kvd: usize) -> Ernie45LayerWeights {
        Ernie45LayerWeights {
            q_proj: vec![0f32; qd * h],
            k_proj: vec![0f32; kvd * h],
            v_proj: vec![0f32; kvd * h],
            o_proj: vec![0f32; h * qd],
            gate_proj: vec![0f32; inter * h],
            up_proj: vec![0f32; inter * h],
            down_proj: vec![0f32; h * inter],
            input_layernorm: vec![1f32; h],
            post_attention_layernorm: vec![1f32; h],
        }
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

    // -- forward_embeds_last_logits: equivalence with forward_embeds_trace --
    //
    // All-zero weights (as `full_source`/`layer_weights` build) make every
    // row's hidden state identical regardless of position, which cannot
    // distinguish "last row" from "any other row". These tests use a small
    // deterministic pseudo-random model instead, so the bit-for-bit
    // comparison is actually load-bearing.

    /// Deterministic (seeded) pseudo-random f32 values in `[-1, 1)`, used to
    /// build non-trivial weights/embeds so equivalence tests can't pass by
    /// every row collapsing to the same value.
    fn pseudo_random_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let bits = (state >> 33) as u32;
                (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn equiv_test_config() -> Ernie45Config {
        Ernie45Config {
            hidden_size: 8,
            intermediate_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            vocab_size: 6,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling: Ernie45RopeScaling {
                mrope_section: vec![2],
            },
            tie_word_embeddings: false,
            use_bias: false,
        }
    }

    fn equiv_weights(cfg: &Ernie45Config) -> Ernie45Weights {
        let h = cfg.hidden_size;
        let q_dim = cfg.q_dim().unwrap();
        let kv_dim = cfg.kv_dim().unwrap();
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = |len: usize| -> Vec<f32> {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            pseudo_random_vec(len, seed)
        };
        // Norm gammas near 1.0 so RMSNorm doesn't zero out the signal.
        let next_gamma = |len: usize, next: &mut dyn FnMut(usize) -> Vec<f32>| -> Vec<f32> {
            next(len).into_iter().map(|v| v + 1.5).collect()
        };
        let embed_tokens = next(cfg.vocab_size * h);
        let mut layers = Vec::new();
        for _ in 0..cfg.num_hidden_layers {
            layers.push(Ernie45LayerWeights {
                q_proj: next(q_dim * h),
                k_proj: next(kv_dim * h),
                v_proj: next(kv_dim * h),
                o_proj: next(h * q_dim),
                gate_proj: next(cfg.intermediate_size * h),
                up_proj: next(cfg.intermediate_size * h),
                down_proj: next(h * cfg.intermediate_size),
                input_layernorm: next_gamma(h, &mut next),
                post_attention_layernorm: next_gamma(h, &mut next),
            });
        }
        let final_norm = next_gamma(h, &mut next);
        let lm_head = next(cfg.vocab_size * h);
        Ernie45Weights {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        }
    }

    fn equiv_model() -> (Ernie45Config, Ernie45Model) {
        let cfg = equiv_test_config();
        let weights = equiv_weights(&cfg);
        let model = Ernie45Model::new(cfg.clone(), weights).unwrap();
        (cfg, model)
    }

    /// The load-bearing equivalence check: `forward_embeds_last_logits`
    /// must equal the last `vocab_size` values of `forward_embeds_trace`'s
    /// logits bit-for-bit (same arithmetic, same order), at `s == 1`.
    #[test]
    fn forward_embeds_last_logits_matches_trace_last_row_s1() {
        let (cfg, model) = equiv_model();
        let s = 1usize;
        let embeds = pseudo_random_vec(s * cfg.hidden_size, 111);
        let positions: Vec<[u32; 3]> = (0..s as u32).map(|i| [i, i, i]).collect();
        let trace = model.forward_embeds_trace(&embeds, &positions).unwrap();
        let last = model
            .forward_embeds_last_logits(&embeds, &positions)
            .unwrap();
        let vocab = cfg.vocab_size;
        assert_eq!(&last[..], &trace.logits[(s - 1) * vocab..][..vocab]);
    }

    /// Same equivalence check at a multi-row sequence length, so a mutation
    /// that reads the wrong row (e.g. row 0 instead of row `s - 1`) is
    /// actually distinguishable — at `s == 1` row 0 and row `s - 1` are the
    /// same row.
    #[test]
    fn forward_embeds_last_logits_matches_trace_last_row_s5() {
        let (cfg, model) = equiv_model();
        let s = 5usize;
        let embeds = pseudo_random_vec(s * cfg.hidden_size, 222);
        let positions: Vec<[u32; 3]> = (0..s as u32).map(|i| [i, i, i]).collect();
        let trace = model.forward_embeds_trace(&embeds, &positions).unwrap();
        let last = model
            .forward_embeds_last_logits(&embeds, &positions)
            .unwrap();
        let vocab = cfg.vocab_size;
        assert_eq!(&last[..], &trace.logits[(s - 1) * vocab..][..vocab]);
    }

    #[test]
    fn forward_embeds_last_logits_rejects_empty_input() {
        let (_cfg, model) = equiv_model();
        let result = model.forward_embeds_last_logits(&[], &[]);
        assert!(
            matches!(result, Err(InferenceError::Inference(message)) if message.contains("empty token sequence"))
        );
    }

    #[test]
    fn forward_embeds_last_logits_rejects_wrong_embeds_length() {
        let (cfg, model) = equiv_model();
        let positions = [[0u32, 0, 0], [1, 1, 1]];
        // 2 positions but only 1 row of embeds.
        let embeds = vec![0f32; cfg.hidden_size];
        let result = model.forward_embeds_last_logits(&embeds, &positions);
        assert!(matches!(result, Err(InferenceError::InvalidInput(_))));
    }

    #[test]
    fn forward_embeds_last_logits_rejects_non_finite_embeds() {
        let (cfg, model) = equiv_model();
        let positions = [[0u32, 0, 0]];
        let mut embeds = vec![0.1f32; cfg.hidden_size];
        embeds[0] = f32::NAN;
        let result = model.forward_embeds_last_logits(&embeds, &positions);
        assert!(
            matches!(result, Err(InferenceError::Inference(message)) if message.contains("non-finite"))
        );
    }

    // -- kv_prefill / kv_decode_step: the cached decode path --
    //
    // The sink copies post-RoPE k/v rows the kernel already computed, and
    // the step kernel mirrors the full forward at s == 1, so the cached
    // path must be bit-for-bit the uncached one — assert_eq, no tolerance.

    fn kv_test_cache(model: &Ernie45Model, capacity: usize) -> Ernie45KvCache {
        let cfg = model.config();
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        Ernie45KvCache::new(cfg.num_hidden_layers, kv_dim, capacity).unwrap()
    }

    /// The point of `clear`: a reused cache must be indistinguishable from a
    /// freshly allocated one, not merely "close". Both arms run the same prompt
    /// through prefill plus one decode step; the reused arm has already served a
    /// different, longer prompt first, so if `clear` left any observable trace --
    /// a stale row inside the attention window, a length that did not reset --
    /// the two logit vectors would part. They are compared bitwise because both
    /// arms perform identical arithmetic in identical order, so anything less
    /// than equality is a defect rather than drift.
    #[test]
    fn kv_cleared_cache_is_indistinguishable_from_a_fresh_one() {
        let (cfg, model) = equiv_model();
        let h = cfg.hidden_size;
        let embeds_a: Vec<f32> = pseudo_random_vec(6 * h, 0x51ed_c0de_c1ea_0001);
        let positions_a: Vec<[u32; 3]> = (0..6u32).map(|i| [i, i, i]).collect();
        let embeds_b: Vec<f32> = pseudo_random_vec(3 * h, 0x51ed_c0de_c1ea_0002);
        let positions_b: Vec<[u32; 3]> = (0..3u32).map(|i| [i, i, i]).collect();
        let step_row: Vec<f32> = pseudo_random_vec(h, 0x51ed_c0de_c1ea_0003);

        // Reused arm: serve the LONGER prompt first, so any row left behind sits
        // inside the window the second prompt will attend over.
        let mut reused = kv_test_cache(&model, 8);
        model
            .kv_prefill(&embeds_a, &positions_a, &mut reused)
            .unwrap();
        model
            .kv_decode_step(&step_row, [6, 6, 6], &mut reused)
            .unwrap();
        assert_eq!(
            reused.len(),
            7,
            "setup: the first prompt must actually fill rows"
        );
        reused.clear();
        assert!(reused.is_empty(), "clear must reset the live row count");
        assert_eq!(
            reused.capacity(),
            8,
            "clear must keep the allocation -- releasing it defeats the purpose"
        );
        let reused_prefill = model
            .kv_prefill(&embeds_b, &positions_b, &mut reused)
            .unwrap();
        let reused_step = model
            .kv_decode_step(&step_row, [3, 3, 3], &mut reused)
            .unwrap();

        // Fresh arm: same second prompt, never used before.
        let mut fresh = kv_test_cache(&model, 8);
        let fresh_prefill = model
            .kv_prefill(&embeds_b, &positions_b, &mut fresh)
            .unwrap();
        let fresh_step = model
            .kv_decode_step(&step_row, [3, 3, 3], &mut fresh)
            .unwrap();

        assert_eq!(
            reused_prefill, fresh_prefill,
            "prefill into a cleared cache diverged from a fresh one"
        );
        assert_eq!(
            reused_step, fresh_step,
            "decode after a cleared cache diverged from a fresh one"
        );
        assert_eq!(reused.len(), fresh.len());
    }

    /// `truncate` rolls back and refuses to grow. The growth arm matters more
    /// than it looks: clamping instead of refusing would hand back a cache
    /// shorter than asked for while reporting success.
    #[test]
    fn kv_truncate_rolls_back_and_refuses_to_grow() {
        let (cfg, model) = equiv_model();
        let h = cfg.hidden_size;
        let embeds: Vec<f32> = pseudo_random_vec(5 * h, 0x51ed_c0de_c1ea_0004);
        let positions: Vec<[u32; 3]> = (0..5u32).map(|i| [i, i, i]).collect();
        let mut cache = kv_test_cache(&model, 8);
        model.kv_prefill(&embeds, &positions, &mut cache).unwrap();
        assert_eq!(cache.len(), 5);

        cache.truncate(2).unwrap();
        assert_eq!(cache.len(), 2, "truncate must roll the live row count back");
        assert_eq!(
            cache.capacity(),
            8,
            "truncate must not release the allocation"
        );

        // Growing is refused, and the length is left untouched by the refusal.
        let grown = cache.truncate(4);
        assert!(
            matches!(&grown, Err(InferenceError::Inference(m)) if m.contains("cannot grow")),
            "truncate must refuse to grow, got {grown:?}"
        );
        assert_eq!(
            cache.len(),
            2,
            "a refused truncate must not move the length"
        );

        // Truncating to the current length is a no-op, not an error.
        cache.truncate(2).unwrap();
        assert_eq!(cache.len(), 2);
        cache.truncate(0).unwrap();
        assert!(cache.is_empty());
    }

    /// The model-bound constructor must produce exactly the layout the hand-built
    /// one does -- otherwise it trades a duplicated calculation for a divergent
    /// one, which is worse than the problem it solves.
    #[test]
    fn kv_model_bound_cache_matches_the_hand_built_layout() {
        let (cfg, model) = equiv_model();
        let built = model.new_kv_cache(8).unwrap();
        let hand = kv_test_cache(&model, 8);
        assert_eq!(built.layers(), hand.layers());
        assert_eq!(built.kv_dim(), hand.kv_dim());
        assert_eq!(built.capacity(), hand.capacity());
        assert!(built.is_empty());
        // And it agrees with the config's own checked derivation.
        assert_eq!(built.kv_dim(), cfg.kv_dim().unwrap());
        assert_eq!(built.layers(), cfg.num_hidden_layers);
        // A cache from this path is usable by prefill -- the layout check inside
        // kv_prefill is what would reject a wrong one.
        let h = cfg.hidden_size;
        let embeds: Vec<f32> = pseudo_random_vec(2 * h, 0x51ed_c0de_c1ea_0005);
        let positions: Vec<[u32; 3]> = (0..2u32).map(|i| [i, i, i]).collect();
        let mut built = built;
        model.kv_prefill(&embeds, &positions, &mut built).unwrap();
        assert_eq!(built.len(), 2);
    }

    /// The prefill's returned logits are the full forward's last row —
    /// the sink is a pure copy side effect, so the two must agree exactly,
    /// at a multi-row sequence length.
    #[test]
    fn kv_prefill_last_logits_bit_exact_vs_uncached() {
        let (cfg, model) = equiv_model();
        let s = 5usize;
        let embeds = pseudo_random_vec(s * cfg.hidden_size, 333);
        let positions: Vec<[u32; 3]> = (0..s as u32).map(|i| [i, i, i]).collect();
        let uncached = model
            .forward_embeds_last_logits(&embeds, &positions)
            .unwrap();
        let mut cache = kv_test_cache(&model, s);
        let cached = model.kv_prefill(&embeds, &positions, &mut cache).unwrap();
        assert_eq!(cache.len(), s);
        assert_eq!(cached, uncached);
    }

    /// The cached decode loop (prefill + one step per token) must match
    /// the uncached re-forward loop: the prefill's logits bit-for-bit
    /// (identical code path), every step's logits within a quoted
    /// measured bound — see the comparison block below for why exactness
    /// is unavailable for the steps — and the prompt's cached rows
    /// untouched by the stepping.
    #[test]
    fn kv_cached_decode_sequence_matches_uncached() {
        let (cfg, model) = equiv_model();
        let s = 3usize;
        let steps = 2usize;
        let embeds = pseudo_random_vec((s + steps) * cfg.hidden_size, 444);
        let positions: Vec<[u32; 3]> = (0..(s + steps) as u32).map(|i| [i, i, i]).collect();

        // Uncached reference: prefill slice + one re-forward per step.
        let ref_prefill = model
            .forward_embeds_last_logits(&embeds[..s * cfg.hidden_size], &positions[..s])
            .unwrap();
        let mut ref_step_logits: Vec<Vec<f32>> = Vec::new();
        for t in 0..steps {
            let n = s + t + 1;
            ref_step_logits.push(
                model
                    .forward_embeds_last_logits(&embeds[..n * cfg.hidden_size], &positions[..n])
                    .unwrap(),
            );
        }

        // Cached path: one prefill, then one token at a time.
        let mut cache = kv_test_cache(&model, s + steps);
        let cached_prefill = model
            .kv_prefill(&embeds[..s * cfg.hidden_size], &positions[..s], &mut cache)
            .unwrap();
        // Snapshot the prompt rows before any step mutates the cache.
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        // The layout is `[layer][capacity][kv_dim]` — the layer stride is
        // `capacity * kv_dim`, not `kv_dim`.
        let layer_stride = cache.capacity() * kv_dim;
        let snapshot_rows = |cache: &Ernie45KvCache, rows: usize| -> (Vec<f32>, Vec<f32>) {
            let mut k = Vec::new();
            let mut v = Vec::new();
            for layer in 0..cfg.num_hidden_layers {
                let base = layer * layer_stride;
                k.extend_from_slice(&cache.layer_k[base..base + rows * kv_dim]);
                v.extend_from_slice(&cache.layer_v[base..base + rows * kv_dim]);
            }
            (k, v)
        };
        let (prompt_k, prompt_v) = snapshot_rows(&cache, s);
        let mut cached_step_logits: Vec<Vec<f32>> = Vec::new();
        for t in 0..steps {
            let n = s + t;
            let row = &embeds[n * cfg.hidden_size..(n + 1) * cfg.hidden_size];
            // The step attends over `cache.len() + 1` keys; printed so the
            // comment below about the two paths' GEMM shapes is read off a
            // run rather than taken on trust.
            println!("kv step {t}: keys = {}", cache.len() + 1);
            cached_step_logits.push(model.kv_decode_step(row, positions[n], &mut cache).unwrap());
        }

        assert_eq!(cached_prefill, ref_prefill);
        assert_eq!(cache.len(), s + steps);
        // Every step gets the tolerance, step 0 included. The two paths'
        // GEMMs differ only in `m`: for the last row the cached path calls
        // `matmul_bt(q, K, scores, 1, hd, keys)` and `matmul_bt(scores,
        // V^T, out, 1, keys, hd)`, while the uncached path calls the same
        // two with `m = seq_len` and reads the last output row. `k` and
        // `n` agree — `keys == seq_len` at every step — so the difference
        // is purely which BLAS kernel Accelerate picks for a 1-row versus
        // an m-row call, and how it orders the accumulation. Nothing about
        // that varies between step 0 and step 1: an earlier version of
        // this test asserted step 0 bit-for-bit on the stated grounds that
        // its attention had `keys == 1`, which is false — after a
        // 3-token prefill step 0 runs with `keys == 4` (printed below).
        // It passed, but on coincidence, so it was a flake waiting for a
        // different machine or a different Accelerate build.
        //
        // The bound is what the equality is actually worth. Measured on
        // this seeded model (dev profile, 2026-09-03, this machine):
        // worst |diff| 9.54e-7. The bound allows ~1000x headroom over
        // that, and is still orders of magnitude below the smallest
        // greedy top-1/top-2 margin this codebase gates on (3.78, e2e
        // fixture), so it cannot hide a decision change. It is not a
        // weak check for the failures that matter: a structural error
        // (wrong cache offset, wrong head mapping, wrong position row)
        // moves logits by O(1), which this catches by three orders of
        // magnitude, and the prefill assert above is still exact.
        const STEP_LOGIT_MAX_DIFF: f32 = 1e-3;
        for (t, (cached, ref_row)) in cached_step_logits
            .iter()
            .zip(ref_step_logits.iter())
            .enumerate()
        {
            let worst = cached
                .iter()
                .zip(ref_row.iter())
                .map(|(a, b)| (*a - *b).abs())
                .fold(0f32, f32::max);
            println!("kv step {t} worst |diff| vs uncached: {worst:e}");
            assert!(
                worst <= STEP_LOGIT_MAX_DIFF,
                "step {t} worst |diff| {worst:e} exceeds the measured bound {STEP_LOGIT_MAX_DIFF:e}"
            );
        }

        // Cache row invariant: the steps only append — every prompt row
        // written by the prefill is untouched afterward (bit-for-bit).
        let (cached_k, cached_v) = snapshot_rows(&cache, s);
        assert_eq!(
            cached_k, prompt_k,
            "prompt key rows disturbed after stepping"
        );
        assert_eq!(
            cached_v, prompt_v,
            "prompt value rows disturbed after stepping"
        );
    }

    /// Disagreed mrope rows (the vision-block layout, where a token's
    /// three position rows differ from each other) must round-trip
    /// through the cache: a step's RoPE tables come from the step's own
    /// triple, not the prompt's.
    ///
    /// Two arms, because the matching arm alone proves nothing about what
    /// this test is named for — it would still pass if the step ignored
    /// its position argument and the reference happened to agree. The
    /// second arm feeds the step a *wrong* triple (the prompt's last
    /// token) and requires the result to move far more than the
    /// reduction-order bound, so a step that dropped its own position
    /// would redden this test rather than sail through it.
    #[test]
    fn kv_decode_step_uses_its_own_mrope_position_rows() {
        let (cfg, model) = equiv_model();
        let s = 2usize;
        let prompt_embeds = pseudo_random_vec(s * cfg.hidden_size, 555);
        let prompt_positions = [[0u32, 2, 5], [1, 4, 7]];
        let step_embeds = pseudo_random_vec(cfg.hidden_size, 666);
        let step_position = [9u32, 11, 13];

        let mut all_embeds = prompt_embeds.clone();
        all_embeds.extend_from_slice(&step_embeds);
        let ref_logit = model
            .forward_embeds_last_logits(
                &all_embeds,
                &[prompt_positions[0], prompt_positions[1], step_position],
            )
            .unwrap();

        // Both arms are measured against one bound, `MRope_TOL`, so they
        // bracket: arm 1 must land inside it and arm 2 outside, which is
        // only possible if the step actually reads its position argument.
        // The bound is 1e-4 — 100x above the reduction-order noise the
        // sequence test measured for a step (9.5e-7, and 0 for this one),
        // and 20x below the arm-2 gap measured here (2.06e-3, 2026-09-03,
        // this machine). A step that ignored its own triple and reused
        // the prompt's would put arm 2 at the noise floor, three orders
        // of magnitude the wrong side of the bound.
        const MROPE_TOL: f32 = 1e-4;

        // Arm 1: the step's own triple reproduces the uncached forward.
        // Not asserted bit-for-bit: the GEMMs differ in `m`; see
        // `kv_cached_decode_sequence_matches_uncached`.
        let mut cache = kv_test_cache(&model, s + 1);
        model
            .kv_prefill(&prompt_embeds, &prompt_positions, &mut cache)
            .unwrap();
        let cached_logit = model
            .kv_decode_step(&step_embeds, step_position, &mut cache)
            .unwrap();
        let worst = cached_logit
            .iter()
            .zip(ref_logit.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst <= MROPE_TOL,
            "step with its own mrope triple diverged by {worst:e} from the uncached forward"
        );

        // Arm 2: the same step with the wrong triple must land outside
        // the bound arm 1 passed under. The comparison is against arm 1's
        // own result, not against `ref_logit`: a step that ignored its
        // `position` argument entirely would return the *same* value for
        // both triples, which is a gap of exactly zero here, but would
        // still sit far from `ref_logit` and so would sail past a
        // gap-vs-reference form of this arm.
        let mut wrong_cache = kv_test_cache(&model, s + 1);
        model
            .kv_prefill(&prompt_embeds, &prompt_positions, &mut wrong_cache)
            .unwrap();
        let wrong_logit = model
            .kv_decode_step(&step_embeds, prompt_positions[s - 1], &mut wrong_cache)
            .unwrap();
        let gap = wrong_logit
            .iter()
            .zip(cached_logit.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0f32, f32::max);
        println!("mrope arm 1 worst |diff| {worst:e}, arm 2 (wrong triple) gap {gap:e}");
        assert!(
            gap > MROPE_TOL,
            "stepping with the prompt's triple instead of the step's own moved the logits by \
             only {gap:e}: this test cannot see the defect it is named for"
        );
        let _ = cfg;
    }

    #[test]
    fn kv_prefill_rejects_empty_sequence() {
        let (_cfg, model) = equiv_model();
        let mut cache = kv_test_cache(&model, 4);
        let result = model.kv_prefill(&[], &[], &mut cache);
        assert!(matches!(
            result,
            Err(InferenceError::Inference(message))
                if message.contains("empty token sequence")
        ));
        assert!(cache.is_empty());
    }

    #[test]
    fn kv_prefill_rejects_over_capacity_before_filling() {
        let (cfg, model) = equiv_model();
        let mut cache = kv_test_cache(&model, 3);
        let s = 4usize;
        let embeds = pseudo_random_vec(s * cfg.hidden_size, 777);
        let positions: Vec<[u32; 3]> = (0..s as u32).map(|i| [i, i, i]).collect();
        let result = model.kv_prefill(&embeds, &positions, &mut cache);
        assert!(matches!(
            result,
            Err(InferenceError::Inference(message))
                if message.contains("exceed the cache capacity")
        ));
        assert!(cache.is_empty());
    }

    #[test]
    fn kv_prefill_rejects_non_empty_cache() {
        let (cfg, model) = equiv_model();
        let s = 2usize;
        let mut cache = kv_test_cache(&model, 4);
        let fill = pseudo_random_vec(s * cfg.hidden_size, 888);
        let fill_pos: Vec<[u32; 3]> = (0..s as u32).map(|i| [i, i, i]).collect();
        model.kv_prefill(&fill, &fill_pos, &mut cache).unwrap();
        let result = model.kv_prefill(&fill, &fill_pos, &mut cache);
        assert!(matches!(
            result,
            Err(InferenceError::Inference(message))
                if message.contains("already holds")
        ));
        assert_eq!(cache.len(), s);
    }

    #[test]
    fn kv_prefill_rejects_wrong_embeds_length() {
        let (cfg, model) = equiv_model();
        let mut cache = kv_test_cache(&model, 4);
        let positions: Vec<[u32; 3]> = (0..3u32).map(|i| [i, i, i]).collect();
        // 3 positions but only 2 rows of embeds.
        let embeds = pseudo_random_vec(2 * cfg.hidden_size, 999);
        let result = model.kv_prefill(&embeds, &positions, &mut cache);
        assert!(matches!(result, Err(InferenceError::InvalidInput(_))));
        assert!(cache.is_empty());
    }

    #[test]
    fn kv_prefill_rejects_non_finite_embeds() {
        let (cfg, model) = equiv_model();
        let mut cache = kv_test_cache(&model, 4);
        let mut embeds = pseudo_random_vec(2 * cfg.hidden_size, 1010);
        embeds[0] = f32::NAN;
        let positions = [[0u32, 0, 0], [1, 1, 1]];
        let result = model.kv_prefill(&embeds, &positions, &mut cache);
        assert!(matches!(
            result,
            Err(InferenceError::Inference(message))
                if message.contains("non-finite")
        ));
        assert!(cache.is_empty());
    }

    #[test]
    fn kv_decode_step_rejects_empty_cache() {
        let (cfg, model) = equiv_model();
        let mut cache = kv_test_cache(&model, 4);
        let embed = pseudo_random_vec(cfg.hidden_size, 1111);
        let result = model.kv_decode_step(&embed, [0, 0, 0], &mut cache);
        assert!(matches!(
            result,
            Err(InferenceError::Inference(message))
                if message.contains("cache is empty")
        ));
    }

    #[test]
    fn kv_decode_step_rejects_full_cache() {
        let (cfg, model) = equiv_model();
        let s = 2usize;
        let mut cache = kv_test_cache(&model, s);
        let embeds = pseudo_random_vec(s * cfg.hidden_size, 1212);
        let positions: Vec<[u32; 3]> = (0..s as u32).map(|i| [i, i, i]).collect();
        model.kv_prefill(&embeds, &positions, &mut cache).unwrap();
        let embed = pseudo_random_vec(cfg.hidden_size, 1313);
        let result = model.kv_decode_step(&embed, [s as u32, s as u32, s as u32], &mut cache);
        assert!(matches!(
            result,
            Err(InferenceError::Inference(message))
                if message.contains("cache is full")
        ));
        assert_eq!(cache.len(), s);
    }

    #[test]
    fn kv_entries_reject_mismatched_cache_shape() {
        let (cfg, model) = equiv_model();
        // A cache for a different layer count.
        let mut foreign_layers = Ernie45KvCache::new(
            cfg.num_hidden_layers + 1,
            cfg.num_key_value_heads * cfg.head_dim,
            4,
        )
        .unwrap();
        let s = 2usize;
        let embeds = pseudo_random_vec(s * cfg.hidden_size, 1414);
        let positions: Vec<[u32; 3]> = (0..s as u32).map(|i| [i, i, i]).collect();
        let result = model.kv_prefill(&embeds, &positions, &mut foreign_layers);
        assert!(matches!(
            result,
            Err(InferenceError::Inference(message))
                if message.contains("built for")
        ));
        // Same layer count, wrong kv_dim.
        let mut narrow = Ernie45KvCache::new(cfg.num_hidden_layers, cfg.head_dim, 4).unwrap();
        let result = model.kv_prefill(&embeds, &positions, &mut narrow);
        assert!(matches!(
            result,
            Err(InferenceError::Inference(message))
                if message.contains("built for")
        ));
        // The step path checks the shape too: a non-empty cache with the
        // right layer count but the wrong kv_dim must be rejected, not
        // indexed at wrong offsets.
        let mut nonempty_wrong_kv =
            Ernie45KvCache::new(cfg.num_hidden_layers, cfg.head_dim, 4).unwrap();
        nonempty_wrong_kv.len = 1;
        let embed = pseudo_random_vec(cfg.hidden_size, 1416);
        let result = model.kv_decode_step(&embed, [0, 0, 0], &mut nonempty_wrong_kv);
        assert!(matches!(
            result,
            Err(InferenceError::Inference(message))
                if message.contains("built for")
        ));
    }

    #[test]
    fn kv_decode_step_rejects_wrong_embeds_length() {
        let (cfg, model) = equiv_model();
        let s = 2usize;
        let mut cache = kv_test_cache(&model, 4);
        let embeds = pseudo_random_vec(s * cfg.hidden_size, 1515);
        let positions: Vec<[u32; 3]> = (0..s as u32).map(|i| [i, i, i]).collect();
        model.kv_prefill(&embeds, &positions, &mut cache).unwrap();
        let short = pseudo_random_vec(cfg.hidden_size - 1, 1616);
        let result = model.kv_decode_step(&short, [2, 2, 2], &mut cache);
        assert!(matches!(result, Err(InferenceError::InvalidInput(_))));
        assert_eq!(cache.len(), s);
    }

    #[test]
    fn kv_decode_step_rejects_non_finite_embeds() {
        let (cfg, model) = equiv_model();
        let s = 2usize;
        let mut cache = kv_test_cache(&model, 4);
        let embeds = pseudo_random_vec(s * cfg.hidden_size, 1717);
        let positions: Vec<[u32; 3]> = (0..s as u32).map(|i| [i, i, i]).collect();
        model.kv_prefill(&embeds, &positions, &mut cache).unwrap();
        let mut bad = pseudo_random_vec(cfg.hidden_size, 1818);
        bad[0] = f32::INFINITY;
        let result = model.kv_decode_step(&bad, [2, 2, 2], &mut cache);
        assert!(matches!(
            result,
            Err(InferenceError::Inference(message))
                if message.contains("non-finite")
        ));
        assert_eq!(cache.len(), s);
    }

    /// The issue's acceptance test at the decoder level, on the committed
    /// fixture's token sequences: the greedy token sequence produced with
    /// the KV cache (prefill once, then one step per token) must equal
    /// the sequence produced by the uncached re-forward loop. Embedding
    /// rows come from the model's own `embed_tokens`, positions continue
    /// the plain `arange` — the text-only reduction the whole module is
    /// built on.
    ///
    /// The layer count is the one shape here that is NOT production-sized, and
    /// it is the only one that can be cut without weakening the test: the cache
    /// is indexed per layer, so four layers exercise the same offset arithmetic
    /// eighteen do, while every shape the kernels actually compute against --
    /// `hidden_size`, `intermediate_size`, `head_dim`, and the 2/16 KV-to-query
    /// head ratio -- stays exactly as the checkpoint ships it. `vocab_size` is
    /// pinned by the committed fixture, whose token ids are real tokenizer
    /// output and index straight into `embed_tokens`. Measured on this test:
    /// eighteen layers cost 1.45 GiB peak RSS and 12.1s in an ordinary
    /// `cargo test` run, of which the extra fourteen layers are 0.79 GiB.
    #[test]
    fn kv_greedy_sequence_matches_uncached_on_committed_fixture() {
        let cfg = Ernie45Config::from_config_json_str(
            r#"{"hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 4,
                "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 128,
                "vocab_size": 103424, "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
                "rope_scaling": {"mrope_section": [16, 24, 24]},
                "tie_word_embeddings": false, "use_bias": false}"#,
        )
        .unwrap();
        let mut seed = 0x51ed_c0de_a11c_e5b0_u64;
        let mut next = |len: usize| -> Vec<f32> {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            pseudo_random_vec(len, seed)
        };
        let h = cfg.hidden_size;
        let weights = Ernie45Weights {
            embed_tokens: next(cfg.vocab_size * h),
            layers: (0..cfg.num_hidden_layers)
                .map(|_| Ernie45LayerWeights {
                    q_proj: next(cfg.num_attention_heads * cfg.head_dim * h),
                    k_proj: next(cfg.num_key_value_heads * cfg.head_dim * h),
                    v_proj: next(cfg.num_key_value_heads * cfg.head_dim * h),
                    o_proj: next(h * cfg.num_attention_heads * cfg.head_dim),
                    gate_proj: next(cfg.intermediate_size * h),
                    up_proj: next(cfg.intermediate_size * h),
                    down_proj: next(h * cfg.intermediate_size),
                    input_layernorm: next(h).into_iter().map(|v| v + 1.5).collect(),
                    post_attention_layernorm: next(h).into_iter().map(|v| v + 1.5).collect(),
                })
                .collect(),
            final_norm: next(h).into_iter().map(|v| v + 1.5).collect(),
            lm_head: next(cfg.vocab_size * h),
        };
        let model = Ernie45Model::new(cfg, weights).unwrap();
        let embeds = model.embed_tokens();

        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/paddleocr_vl/decoder/decoder_goldens.json");
        let golden: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&fixture).unwrap()).unwrap();
        let cases: Vec<Vec<u32>> = golden
            .get("cases")
            .and_then(|c| c.as_array())
            .unwrap()
            .iter()
            .map(|c| {
                c["ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|i| i.as_u64().unwrap() as u32)
                    .collect()
            })
            .collect();
        assert!(cases.len() >= 4, "fixture shrank to {} cases", cases.len());

        for (case_idx, ids) in cases.iter().enumerate() {
            let prompt_len = ids.len();
            let max_new_tokens = 4usize;
            let prompt_embeds: Vec<f32> = ids
                .iter()
                .flat_map(|&id| embeds[id as usize * h..][..h].iter().copied())
                .collect();
            let prompt_positions: Vec<[u32; 3]> =
                (0..prompt_len as u32).map(|i| [i, i, i]).collect();

            // Uncached reference loop: re-forward the whole sequence.
            let mut ref_embeds = prompt_embeds.clone();
            let mut ref_positions = prompt_positions.clone();
            let mut ref_logits = model
                .forward_embeds_last_logits(&ref_embeds, &ref_positions)
                .unwrap();
            let mut ref_greedy: Vec<u32> = Vec::new();
            for _ in 0..max_new_tokens {
                let next = ref_logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i as u32)
                    .unwrap();
                ref_greedy.push(next);
                if next == 2 {
                    break;
                }
                let p = ref_positions
                    .iter()
                    .fold(0u32, |m, r| m.max(r[0]).max(r[1]).max(r[2]))
                    + 1;
                ref_embeds.extend_from_slice(&embeds[next as usize * h..][..h]);
                ref_positions.push([p, p, p]);
                ref_logits = model
                    .forward_embeds_last_logits(&ref_embeds, &ref_positions)
                    .unwrap();
            }

            // Cached loop: one prefill, then one step per token.
            let kv_dim = model.config().num_key_value_heads * model.config().head_dim;
            let mut cache = Ernie45KvCache::new(
                model.config().num_hidden_layers,
                kv_dim,
                prompt_len + max_new_tokens,
            )
            .unwrap();
            let mut max_pos = prompt_positions
                .iter()
                .fold(0u32, |m, r| m.max(r[0]).max(r[1]).max(r[2]));
            let mut cached_logits = model
                .kv_prefill(&prompt_embeds, &prompt_positions, &mut cache)
                .unwrap();
            let mut cached_greedy: Vec<u32> = Vec::new();
            for _ in 0..max_new_tokens {
                let next = cached_logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i as u32)
                    .unwrap();
                cached_greedy.push(next);
                if next == 2 {
                    break;
                }
                max_pos += 1;
                let position = [max_pos, max_pos, max_pos];
                let row = &embeds[next as usize * h..][..h];
                cached_logits = model.kv_decode_step(row, position, &mut cache).unwrap();
            }

            // Compared BEFORE the equality assert, because two loops that both
            // stopped after one token agree trivially and would report a pass
            // that covered a single decode step. The bound is what the loop is
            // allowed to produce, so a fixture or a break condition that
            // silently shortens the sequence fails here rather than passing.
            assert_eq!(
                ref_greedy.len(),
                max_new_tokens,
                "case {case_idx}: reference loop produced {} of {max_new_tokens} tokens, so the \
                 equality below would compare a shorter sequence than intended",
                ref_greedy.len()
            );
            assert_eq!(
                cached_greedy,
                ref_greedy,
                "case {case_idx}: greedy token sequence diverges ({} tokens)",
                ref_greedy.len()
            );
        }
    }

    #[test]
    fn kv_cache_new_rejects_zero_dimensions_and_overflow() {
        assert!(Ernie45KvCache::new(0, 8, 4).is_err());
        assert!(Ernie45KvCache::new(2, 0, 4).is_err());
        assert!(Ernie45KvCache::new(2, 8, 0).is_err());
        let result = Ernie45KvCache::new(4, 8, usize::MAX);
        assert!(matches!(
            result,
            Err(InferenceError::Inference(message))
                if message.contains("overflow")
        ));
        let cache = Ernie45KvCache::new(2, 8, 4).unwrap();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 4);
        assert_eq!(cache.layers(), 2);
        assert_eq!(cache.kv_dim(), 8);
    }
}

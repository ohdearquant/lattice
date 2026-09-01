//! ERNIE-4.5 dense text decoder (PaddleOCR-VL family): CPU f32 reference
//! forward.
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

/// Text-decoder subset of the checkpoint's `config.json`, deserialized
/// fail-closed: every field this forward pass depends on is required, so a
/// checkpoint missing one refuses to load instead of running on a default.
#[derive(Debug, Clone, Deserialize)]
pub struct Ernie45Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub rope_scaling: Ernie45RopeScaling,
    pub tie_word_embeddings: bool,
    pub use_bias: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Ernie45RopeScaling {
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
        let section_sum: usize = self.rope_scaling.mrope_section.iter().sum();
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

    fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

/// One decoder layer's weights, all row-major `[out, in]` as stored in the
/// checkpoint (so `matmul_bt` computes `x @ W^T` directly).
pub struct Ernie45LayerWeights {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
}

pub struct Ernie45Weights {
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<Ernie45LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>,
}

fn load_tensor<T: TensorSource + ?Sized>(
    source: &mut T,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, InferenceError> {
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

impl Ernie45Weights {
    /// Load the text-decoder tensors (`model.*` + `lm_head.*`; the
    /// checkpoint's `visual.*` and `mlp_AR.*` trees are outside this slice)
    /// with fail-closed shape checks against `cfg`.
    pub fn load<T: TensorSource + ?Sized>(
        source: &mut T,
        cfg: &Ernie45Config,
    ) -> Result<Self, InferenceError> {
        let h = cfg.hidden_size;
        let embed_tokens = load_tensor(source, "model.embed_tokens.weight", &[cfg.vocab_size, h])?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}.");
            layers.push(Ernie45LayerWeights {
                q_proj: load_tensor(
                    source,
                    &format!("{p}self_attn.q_proj.weight"),
                    &[cfg.q_dim(), h],
                )?,
                k_proj: load_tensor(
                    source,
                    &format!("{p}self_attn.k_proj.weight"),
                    &[cfg.kv_dim(), h],
                )?,
                v_proj: load_tensor(
                    source,
                    &format!("{p}self_attn.v_proj.weight"),
                    &[cfg.kv_dim(), h],
                )?,
                o_proj: load_tensor(
                    source,
                    &format!("{p}self_attn.o_proj.weight"),
                    &[h, cfg.q_dim()],
                )?,
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
    pub embed: Vec<f32>,
    pub layer_outputs: Vec<Vec<f32>>,
    pub final_norm: Vec<f32>,
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

pub struct Ernie45Model {
    cfg: Ernie45Config,
    weights: Ernie45Weights,
}

impl Ernie45Model {
    pub fn new(cfg: Ernie45Config, weights: Ernie45Weights) -> Self {
        Self { cfg, weights }
    }

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

        let mut x = vec![0f32; s * h];
        for (t, &id) in ids.iter().enumerate() {
            x[t * h..(t + 1) * h].copy_from_slice(&w.embed_tokens[id as usize * h..][..h]);
        }
        let embed = x.clone();

        let inv_freq = gemma4_rope_inv_freq(cfg.head_dim, cfg.rope_theta, None);
        let positions: Vec<u32> = (0..s as u32).collect();
        let (cos, sin) = gemma4_rope_cos_sin(&inv_freq, &positions);

        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let groups = heads / kv_heads;
        let hd = cfg.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut layer_outputs = Vec::with_capacity(cfg.num_hidden_layers);
        let mut normed = vec![0f32; s * h];
        let mut q = vec![0f32; s * cfg.q_dim()];
        let mut k = vec![0f32; s * cfg.kv_dim()];
        let mut v = vec![0f32; s * cfg.kv_dim()];
        let mut attn_out = vec![0f32; s * cfg.q_dim()];
        let mut proj = vec![0f32; s * h];
        let mut gate = vec![0f32; s * cfg.intermediate_size];
        let mut up = vec![0f32; s * cfg.intermediate_size];

        for layer in &w.layers {
            // Attention block.
            normed.copy_from_slice(&x);
            rms_norm(&mut normed, &layer.input_layernorm, h, cfg.rms_norm_eps);
            matmul_bt(&normed, &layer.q_proj, &mut q, s, h, cfg.q_dim());
            matmul_bt(&normed, &layer.k_proj, &mut k, s, h, cfg.kv_dim());
            matmul_bt(&normed, &layer.v_proj, &mut v, s, h, cfg.kv_dim());
            gemma4_apply_rope(&mut q, &cos, &sin, s, heads, hd);
            gemma4_apply_rope(&mut k, &cos, &sin, s, kv_heads, hd);

            // Causal GQA attention, one query head at a time. The softmax is
            // exact `f32::exp` deliberately (mirroring `gemma4_model.rs`'s
            // `softmax_row_fail_closed` and `gemma4_ops.rs`'s exact-tanh
            // rationale): the shared `softmax_attention` kernel's Schraudolph
            // `fast_exp` carries ~1% relative error, which this differential
            // gate measured as O(1e-2) activation divergence against the
            // exact-exp HF reference by layer 1.
            let mut scores = vec![0f32; s];
            for qh in 0..heads {
                let kvh = qh / groups;
                for tq in 0..s {
                    let q_row = &q[(tq * heads + qh) * hd..][..hd];
                    let row = &mut scores[..=tq];
                    for (tk, slot) in row.iter_mut().enumerate() {
                        let k_row = &k[(tk * kv_heads + kvh) * hd..][..hd];
                        *slot = q_row
                            .iter()
                            .zip(k_row.iter())
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            * scale;
                    }
                    softmax_row_fail_closed(row);
                    let out_row = &mut attn_out[(tq * heads + qh) * hd..][..hd];
                    out_row.fill(0.0);
                    for (tk, &weight) in row.iter().enumerate() {
                        let v_row = &v[(tk * kv_heads + kvh) * hd..][..hd];
                        for (o, &vv) in out_row.iter_mut().zip(v_row.iter()) {
                            *o += weight * vv;
                        }
                    }
                }
            }
            matmul_bt(&attn_out, &layer.o_proj, &mut proj, s, cfg.q_dim(), h);
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
        let mut logits = vec![0f32; s * cfg.vocab_size];
        matmul_bt(&final_normed, &w.lm_head, &mut logits, s, h, cfg.vocab_size);

        Ok(Ernie45Trace {
            embed,
            layer_outputs,
            final_norm: final_normed,
            logits,
        })
    }
}

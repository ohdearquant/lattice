//! Fuse Qwen3.5's **shifted** RMSNorm `(1 + gamma)` scale into the
//! immediately-following linear layer's weight matrix as a column multiply,
//! then neutralize the norm weight to zero so the runtime `(1 + gamma)`
//! evaluates to identity.
//!
//! Background: Qwen3.5 applies RMSNorm with shifted scale `(1 + gamma)` in
//! `crate::model::qwen35::norm::qwen35_rms_norm` (see `quant/quarot/plan.rs`
//! §Known gaps). A diagonal scale does NOT commute with Hadamard rotation,
//! so the rotation pipeline must fold the scale into the next layer BEFORE
//! absorbing rotations.
//!
//! ## Identity
//!
//! Original forward: `n = (1 + gamma) ⊙ normalize(h)`, then the next linear
//! computes `y = W · n = W · diag(1 + gamma) · normalize(h)`.
//!
//! After fusion: `W_fused := W · diag(1 + gamma)`, and the runtime norm
//! weight is set to `gamma = 0` so `(1 + 0) = 1` evaluates to identity. The
//! forward then computes `y = W_fused · normalize(h)`, mathematically equal
//! to the original up to floating-point rounding.
//!
//! `W` is row-major `[rows × cols]` where `cols == hidden_size == gamma.len()`.
//! The column-multiply storage form is `W[i, j] *= (1 + gamma[j])`.
//!
//! ## Scope — v0 (step 3c-2)
//!
//! This module fuses `input_layernorm` and `post_attention_layernorm` only.
//! `final_norm` (`model.language_model.norm.weight`) feeds `lm_head`, which
//! requires materialization when `tie_word_embeddings=true` (Qwen3.5 default).
//! That materialization + final-norm fusion lives in step 3c-3.
//!
//! GDN's internal `linear_attn.norm.weight` is a **plain** `gamma` (not
//! `(1 + gamma)`) and runs inside the GDN block in a basis unaffected by
//! residual rotation — see `plan.rs` §Known gaps. This module does NOT
//! attempt to fuse it.
//!
//! ## Order in the conversion pipeline
//!
//! ```text
//!   1. read f64 tensors
//!   2. fuse_per_layer_shifted_rmsnorm     ← THIS MODULE
//!   3. absorb rotation plan               ← quant/quarot/rotation.rs
//!   4. quantize                           ← weights/q4_weights.rs (step 3c-1)
//! ```
//!
//! Fusion MUST happen before rotation absorption: input-side absorption
//! right-multiplies the weight by `R^T`, and `diag(1 + gamma) · R^T` is
//! not equal to `R^T · diag(1 + gamma)` in general (the two only commute
//! when `R` is itself diagonal, which Hadamard rotations are not).

use crate::error::InferenceError;
use crate::model::qwen35_config::Qwen35Config;

/// Description of one fusion site: a `*_layernorm.weight` tensor whose
/// `(1 + gamma)` must be folded into one or more downstream linear weight
/// tensors as a column multiply.
///
/// `downstream_weights` must be non-empty — a single norm feeds multiple
/// parallel projections in Qwen3.5 (e.g., `input_layernorm` feeds `q_proj`,
/// `k_proj`, `v_proj` for full-attention layers). The pipeline entry point
/// [`crate::quant::quarot::pipeline::fuse_rmsnorms`] enforces this
/// invariant and rejects empty lists before any mutation, because
/// neutralizing the norm without folding the scale anywhere would silently
/// change the model's output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmsNormFusionTarget {
    pub norm_tensor: String,
    pub downstream_weights: Vec<String>,
}

/// Build the per-layer RMSNorm fusion plan for a Qwen3.5 config.
///
/// Covers `input_layernorm` (feeds attention input projections) and
/// `post_attention_layernorm` (feeds MLP input projections) for every
/// layer. Does NOT cover `final_norm`/`lm_head` (step 3c-3) or MoE expert
/// fusion (v1; see `plan.rs` §Deferred).
///
/// Returns one [`RmsNormFusionTarget`] per layer per norm site
/// (`2 * num_hidden_layers` entries for non-MoE configs).
///
/// # Errors
///
/// Returns `InferenceError::Inference` if the config selects MoE for any
/// layer — MoE expert fusion is deferred to v1 because the expert weight
/// layout (`[num_experts, ...]`) requires per-expert slicing that the
/// row-major column-multiply primitive in this module does not handle.
pub fn qwen35_per_layer_fusion_plan(
    cfg: &Qwen35Config,
) -> Result<Vec<RmsNormFusionTarget>, InferenceError> {
    if cfg.is_moe() {
        return Err(InferenceError::Inference(
            "qwen35_per_layer_fusion_plan: MoE configs are deferred to v1 \
             (per-expert fusion is not implemented). See \
             `quant/quarot/plan.rs` §Deferred."
                .to_string(),
        ));
    }

    let mut out = Vec::with_capacity(2 * cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{i}");

        let attn_downstream = if cfg.is_full_attention(i) {
            vec![
                format!("{prefix}.self_attn.q_proj.weight"),
                format!("{prefix}.self_attn.k_proj.weight"),
                format!("{prefix}.self_attn.v_proj.weight"),
            ]
        } else {
            vec![
                format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                format!("{prefix}.linear_attn.in_proj_z.weight"),
                format!("{prefix}.linear_attn.in_proj_b.weight"),
                format!("{prefix}.linear_attn.in_proj_a.weight"),
            ]
        };
        out.push(RmsNormFusionTarget {
            norm_tensor: format!("{prefix}.input_layernorm.weight"),
            downstream_weights: attn_downstream,
        });

        out.push(RmsNormFusionTarget {
            norm_tensor: format!("{prefix}.post_attention_layernorm.weight"),
            downstream_weights: vec![
                format!("{prefix}.mlp.gate_proj.weight"),
                format!("{prefix}.mlp.up_proj.weight"),
            ],
        });
    }
    Ok(out)
}

/// Fuse a shifted RMSNorm `(1 + gamma)` into a row-major `[rows × cols]`
/// weight matrix in place: `W[i, j] *= (1.0 + gamma[j])`.
///
/// `cols == gamma.len()` is required (the norm operates on the input
/// dimension of the downstream linear layer, which is the matrix's column
/// count).
///
/// # Errors
///
/// - `cols != gamma.len()` → norm/weight dimension mismatch.
/// - `rows * cols` overflows `usize` → caller passed a malformed shape.
/// - `weight.len() != rows * cols` → caller passed mismatched buffer.
pub fn fuse_shifted_rmsnorm_into_next_layer_f64(
    weight: &mut [f64],
    rows: usize,
    cols: usize,
    gamma: &[f64],
) -> Result<(), InferenceError> {
    if gamma.len() != cols {
        return Err(InferenceError::Inference(format!(
            "fuse_shifted_rmsnorm_into_next_layer_f64: gamma.len()={} != cols={cols}",
            gamma.len()
        )));
    }
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        InferenceError::Inference(format!(
            "fuse_shifted_rmsnorm_into_next_layer_f64: rows*cols overflow (rows={rows}, cols={cols})"
        ))
    })?;
    if weight.len() != expected {
        return Err(InferenceError::Inference(format!(
            "fuse_shifted_rmsnorm_into_next_layer_f64: weight.len()={} != rows*cols {expected}",
            weight.len()
        )));
    }
    for r in 0..rows {
        let row = &mut weight[r * cols..(r + 1) * cols];
        for (w, &g) in row.iter_mut().zip(gamma.iter()) {
            *w *= 1.0 + g;
        }
    }
    Ok(())
}

/// Neutralize a `*_layernorm.weight` tensor in place by zeroing every
/// element. After fusion has folded `(1 + gamma)` into the downstream
/// linear, the runtime `(1 + gamma[j])` formula must evaluate to `1` so
/// the norm acts as the identity in scale — i.e., `gamma[j] = 0`.
pub fn neutralize_rmsnorm_gamma_f64(gamma: &mut [f64]) {
    for g in gamma.iter_mut() {
        *g = 0.0;
    }
}

#[cfg(test)]
mod tests;

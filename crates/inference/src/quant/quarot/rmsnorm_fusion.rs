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
mod tests {
    use super::*;

    fn synthetic_f64(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 11) as u32;
                (bits as f64 / u32::MAX as f64) - 0.5
            })
            .collect()
    }

    /// matvec: y[i] = sum_j W[i,j] * x[j].
    fn matvec_f64(w: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), cols);
        let mut y = vec![0.0_f64; rows];
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            y[r] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        }
        y
    }

    fn max_abs_diff_f64(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn fusion_preserves_forward_pass() {
        // Layer pre-fusion: y = W · ((1 + g) ⊙ x)
        // After fusion:     y = W_fused · x where W_fused = W · diag(1 + g)
        // The two must produce identical (up to rounding) output.
        let rows = 8;
        let cols = 16;
        let w = synthetic_f64(rows * cols, 1);
        let g = synthetic_f64(cols, 2);
        let x = synthetic_f64(cols, 3);

        let pre_fusion_input: Vec<f64> = x
            .iter()
            .zip(g.iter())
            .map(|(xi, gi)| xi * (1.0 + gi))
            .collect();
        let y_before = matvec_f64(&w, rows, cols, &pre_fusion_input);

        let mut w_fused = w.clone();
        fuse_shifted_rmsnorm_into_next_layer_f64(&mut w_fused, rows, cols, &g).unwrap();
        let y_after = matvec_f64(&w_fused, rows, cols, &x);

        let delta = max_abs_diff_f64(&y_before, &y_after);
        assert!(delta < 1e-12, "fusion diverged: max delta {delta}");
    }

    #[test]
    fn fusion_acts_as_column_multiply() {
        // Explicit per-element check: W[i, j] *= (1 + gamma[j])
        let rows = 3;
        let cols = 4;
        let mut w: Vec<f64> = (0..(rows * cols)).map(|k| k as f64 + 1.0).collect();
        let g = vec![0.0, 1.0, -0.5, 2.0]; // (1 + g) = [1, 2, 0.5, 3]

        fuse_shifted_rmsnorm_into_next_layer_f64(&mut w, rows, cols, &g).unwrap();

        // Row 0 originally [1, 2, 3, 4] → [1*1, 2*2, 3*0.5, 4*3] = [1, 4, 1.5, 12]
        assert_eq!(w[0..4], [1.0, 4.0, 1.5, 12.0]);
        // Row 1 originally [5, 6, 7, 8] → [5*1, 6*2, 7*0.5, 8*3] = [5, 12, 3.5, 24]
        assert_eq!(w[4..8], [5.0, 12.0, 3.5, 24.0]);
        // Row 2 originally [9, 10, 11, 12] → [9, 20, 5.5, 36]
        assert_eq!(w[8..12], [9.0, 20.0, 5.5, 36.0]);
    }

    #[test]
    fn fusion_rejects_gamma_cols_mismatch() {
        let mut w = vec![0.0_f64; 12];
        let g = vec![0.0; 3]; // expected 4 (cols)
        let err = fuse_shifted_rmsnorm_into_next_layer_f64(&mut w, 3, 4, &g).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("gamma.len()=3"), "unexpected error: {msg}");
        assert!(msg.contains("cols=4"), "unexpected error: {msg}");
    }

    #[test]
    fn fusion_rejects_weight_len_mismatch() {
        let mut w = vec![0.0_f64; 100];
        let g = vec![0.0; 4];
        let err = fuse_shifted_rmsnorm_into_next_layer_f64(&mut w, 3, 4, &g).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("weight.len()=100"), "unexpected error: {msg}");
        assert!(msg.contains("rows*cols 12"), "unexpected error: {msg}");
    }

    #[test]
    fn fusion_rejects_rows_cols_overflow() {
        let mut w = vec![0.0_f64; 4];
        let g = vec![0.0; 4];
        let err = fuse_shifted_rmsnorm_into_next_layer_f64(&mut w, usize::MAX, 4, &g).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("overflow"), "unexpected error: {msg}");
    }

    #[test]
    fn neutralize_zeros_all_elements() {
        let mut g = vec![1.0_f64, -2.5, 0.3, 7.0];
        neutralize_rmsnorm_gamma_f64(&mut g);
        assert_eq!(g, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn neutralize_handles_empty_slice() {
        let mut g: Vec<f64> = Vec::new();
        neutralize_rmsnorm_gamma_f64(&mut g);
        assert!(g.is_empty());
    }

    /// Combined: fusion + neutralization gives a pipeline where the
    /// runtime `(1 + gamma_new[j])` evaluates to 1 and the downstream
    /// weight holds the original fused scale.
    #[test]
    fn fusion_plus_neutralization_round_trip() {
        let rows = 4;
        let cols = 8;
        let w_orig = synthetic_f64(rows * cols, 10);
        let g_orig = synthetic_f64(cols, 11);
        let x = synthetic_f64(cols, 12);

        // Original forward via shifted norm:
        let pre_norm: Vec<f64> = x
            .iter()
            .zip(g_orig.iter())
            .map(|(xi, gi)| xi * (1.0 + gi))
            .collect();
        let y_original = matvec_f64(&w_orig, rows, cols, &pre_norm);

        // Fused pipeline forward:
        let mut w_fused = w_orig.clone();
        let mut g_new = g_orig.clone();
        fuse_shifted_rmsnorm_into_next_layer_f64(&mut w_fused, rows, cols, &g_new).unwrap();
        neutralize_rmsnorm_gamma_f64(&mut g_new);
        // After neutralization, runtime computes `(1 + 0) · x = x` for the norm output.
        let pre_norm_after_neutralize: Vec<f64> = x
            .iter()
            .zip(g_new.iter())
            .map(|(xi, gi)| xi * (1.0 + gi))
            .collect();
        let y_pipeline = matvec_f64(&w_fused, rows, cols, &pre_norm_after_neutralize);

        let delta = max_abs_diff_f64(&y_original, &y_pipeline);
        assert!(
            delta < 1e-12,
            "pipeline diverged from original: max delta {delta}"
        );
    }

    #[test]
    fn qwen35_fusion_plan_two_entries_per_layer() {
        let cfg = Qwen35Config::qwen35_0_8b();
        let plan = qwen35_per_layer_fusion_plan(&cfg).unwrap();
        assert_eq!(plan.len(), 2 * cfg.num_hidden_layers);
        for tgt in &plan {
            assert!(!tgt.downstream_weights.is_empty(), "{tgt:?}");
        }
    }

    #[test]
    fn qwen35_fusion_plan_norm_names_match_loader() {
        // The norm tensor names must match what `qwen_required_tensor_names`
        // emits — otherwise the converter cannot find the tensors to fuse.
        let cfg = Qwen35Config::qwen35_0_8b();
        let required = crate::model::qwen35::qwen_required_tensor_names(&cfg);
        let plan = qwen35_per_layer_fusion_plan(&cfg).unwrap();
        for tgt in &plan {
            assert!(
                required.contains(&tgt.norm_tensor),
                "norm tensor `{}` not in qwen_required_tensor_names",
                tgt.norm_tensor
            );
            for d in &tgt.downstream_weights {
                assert!(
                    required.contains(d),
                    "downstream weight `{d}` not in qwen_required_tensor_names"
                );
            }
        }
    }

    #[test]
    fn qwen35_fusion_plan_covers_full_attention_and_gdn() {
        let cfg = Qwen35Config::qwen35_0_8b();
        let plan = qwen35_per_layer_fusion_plan(&cfg).unwrap();

        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{i}");
            let in_target = plan
                .iter()
                .find(|t| t.norm_tensor == format!("{prefix}.input_layernorm.weight"))
                .expect("input_layernorm target present");
            let post_target = plan
                .iter()
                .find(|t| t.norm_tensor == format!("{prefix}.post_attention_layernorm.weight"))
                .expect("post_attention_layernorm target present");

            if cfg.is_full_attention(i) {
                assert!(
                    in_target
                        .downstream_weights
                        .iter()
                        .any(|n| n.ends_with(".self_attn.q_proj.weight"))
                );
            } else {
                assert!(
                    in_target
                        .downstream_weights
                        .iter()
                        .any(|n| n.ends_with(".linear_attn.in_proj_qkv.weight"))
                );
            }
            assert!(
                post_target
                    .downstream_weights
                    .iter()
                    .any(|n| n.ends_with(".mlp.gate_proj.weight"))
            );
            assert!(
                post_target
                    .downstream_weights
                    .iter()
                    .any(|n| n.ends_with(".mlp.up_proj.weight"))
            );
        }
    }

    #[test]
    fn qwen35_fusion_plan_rejects_moe_config() {
        let cfg = Qwen35Config::qwen36_35b_a3b();
        assert!(cfg.is_moe());
        let err = qwen35_per_layer_fusion_plan(&cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("MoE"), "unexpected error: {msg}");
    }
}

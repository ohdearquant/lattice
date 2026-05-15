//! Forward-equivalence refuse-on-fail harness for QuaRot Qwen3.5 conversion
//! (step 3c-4 of ADR-044).
//!
//! ## What this module owns
//!
//! [`assert_forward_equivalence_qwen35`] — the converter binary (step 3c-5)
//! must call this gate AFTER running the full pipeline
//! (materialize_lm_head → fuse_rmsnorms → absorb_rotations) and BEFORE
//! quantizing or writing artifacts to disk. The gate runs a deterministic
//! probe forward on both the pre-pipeline `original` working set and the
//! post-pipeline `rotated` working set, then refuses the conversion when
//! the per-element max-abs error exceeds `tolerance`.
//!
//! The refuse semantic is **`Err(InferenceError::Inference)`**. The caller
//! must not write any conversion artifacts when this returns `Err` — the
//! pipeline produced logits that disagree with the original model, and
//! shipping that quantized output would silently degrade accuracy.
//!
//! ## What the probe does (and does NOT do)
//!
//! The probe is **NOT a faithful Qwen3.5 forward.** It is a deterministic
//! *rotation-chain probe* that exercises every linear tensor with a
//! [`crate::quant::quarot::plan::RotationPlan`] rule plus every
//! `*_layernorm` tensor with a
//! [`crate::quant::quarot::rmsnorm_fusion::RmsNormFusionTarget`] entry:
//!
//! ```text
//!   h ← embed_tokens[token]
//!   for each layer L:
//!       h_pre ← (1 + γ_in) ⊙ rms_normalize(h)               [input_layernorm]
//!       if full_attention(L):
//!           q_full ← q_proj · h_pre                          [2*q_dim long]
//!           attn_out ← o_proj · q_full[0..q_dim]             [hidden]
//!       else:
//!           z ← in_proj_z · h_pre                            [output_dim]
//!           attn_out ← out_proj · z                          [hidden]
//!       h ← h + attn_out
//!       h_pre ← (1 + γ_post) ⊙ rms_normalize(h)              [post_attention_layernorm]
//!       gate ← gate_proj · h_pre                             [intermediate]
//!       up   ← up_proj · h_pre                               [intermediate]
//!       mid  ← gate + up        ← LINEARIZED (NOT silu(gate)⊙up)
//!       mlp_out ← down_proj · mid
//!       h ← h + mlp_out
//!   h_final ← (1 + γ_final) ⊙ rms_normalize(h)               [final_norm]
//!   logits ← lm_head · h_final                               [or embed_tokens fallback]
//! ```
//!
//! `silu(gate) ⊙ up` is replaced with `gate + up` so the probe is purely
//! linear in `h`. The QuaRot rotation absorption identities then hold
//! exactly across the chain — any divergence between `original` and
//! `rotated` logits is a pipeline bug, not a non-linearity artifact.
//!
//! For the GQA attention block the probe shortcuts through `q_proj`'s Q
//! half only (the first `full_q_dim` elements of the `[2 * full_q_dim,
//! hidden]` Q + gate_z fused row), feeds that to `o_proj` directly, and
//! skips the actual attention compute (softmax / RoPE / K, V). The
//! QuaRot v0 residual rotation does not depend on K, V being part of
//! the chain — they are input-side rotated for activation quantization
//! deferred to v1 and are still exercised by the per-layer RMSNorm
//! fusion targets (the fusion plan column-multiplies `(1 + γ_in)` into
//! all of `q_proj`, `k_proj`, `v_proj`).
//!
//! ### What the probe IS sufficient to catch
//!
//! - Missing rotation absorption on any tensor in the chain: `embed_tokens`,
//!   `q_proj`, `o_proj`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`,
//!   `down_proj`, `lm_head`.
//! - RMSNorm fusion missing: `(1 + γ)` left online for `input_layernorm`,
//!   `post_attention_layernorm`, or `final_norm`.
//! - `lm_head` not materialized when the input config is tied — the
//!   rotated probe needs a present `lm_head.weight` to consume the
//!   `(1 + γ_final)` column scale and the input-side rotation.
//! - `final_norm` fusion target not appended — γ_final left non-zero in
//!   the rotated set means the runtime double-applies the scale once
//!   it's also baked into lm_head.
//! - Wrong rotation (different `R` between two pipelines, or wrong
//!   absorption side).
//!
//! ### What the probe is NOT sufficient to catch
//!
//! - Bugs that only manifest in `silu(gate) ⊙ up` — a real Qwen forward
//!   would exercise this; the probe linearizes it. Step 4 of ADR-044
//!   (perplexity bench on real calibration data) is the full check.
//! - Bugs in the full attention compute (softmax / RoPE / K, V) — the
//!   probe shortcuts through projections only.
//! - Quantization error — the probe runs in f64 on the pre-quantization
//!   working set. Q4 round-trip error is bounded by the Q4 bridge's own
//!   tests (`weights::q4_weights`).
//!
//! ## Tied vs untied originals
//!
//! For the typical converter flow, `materialize_lm_head_for_qwen35` runs
//! BEFORE this gate, so both `original` (snapshot post-materialize, pre-fuse)
//! and `rotated` have `lm_head.weight` populated. The probe uses it in
//! both cases.
//!
//! For callers that take the `original` snapshot BEFORE materialization
//! (or for direct unit-level tests of a tied input) the probe falls back
//! to `embed_tokens.weight` when `lm_head.weight` is absent and the
//! config says `tie_word_embeddings=true`. This matches the runtime
//! `logits_weight()` semantics at `model/qwen35/weights.rs` and produces
//! the same logits as the rotated probe by construction (`lm_head_rot =
//! embed_tokens · diag(1 + γ_final) · R^T` cancels against `R · rmsnorm(h_orig)`).
//!
//! ## Scope
//!
//! Qwen3.5 dense (non-MoE) configs only. MoE is rejected explicitly here
//! — the rotation pipeline already refuses MoE (`fuse_rmsnorms` errors
//! via [`crate::quant::quarot::rmsnorm_fusion::qwen35_per_layer_fusion_plan`]),
//! and the probe has no expert-mixing path.

use std::collections::HashMap;

use crate::error::InferenceError;
use crate::model::qwen35_config::Qwen35Config;
use crate::quant::quarot::lm_head::{
    QWEN35_EMBED_TOKENS_NAME, QWEN35_FINAL_NORM_NAME, QWEN35_LM_HEAD_NAME,
};
use crate::quant::quarot::pipeline::TensorEntry;

/// Probe sample size + tolerance settings for
/// [`assert_forward_equivalence_qwen35`].
///
/// The default is `num_probe_tokens = 4`, `tolerance = 1e-5`, matching
/// the ADR-044 §"Step 3c contract" requirement
/// (`‖rotated_forward − original_forward‖ < 1e-5` on a small batch).
#[derive(Debug, Clone)]
pub struct ForwardEquivalenceConfig {
    /// Number of token IDs to probe. Must be > 0.
    pub num_probe_tokens: usize,
    /// Max-abs-error threshold (per-logit). Must be > 0.
    pub tolerance: f64,
    /// Seed for the deterministic token sampler.
    pub seed: u64,
}

impl Default for ForwardEquivalenceConfig {
    fn default() -> Self {
        Self {
            num_probe_tokens: 4,
            tolerance: 1e-5,
            seed: 0xCAFE_BABE_DEAD_BEEF,
        }
    }
}

/// Result of a successful [`assert_forward_equivalence_qwen35`] call.
///
/// The numerical metrics are diagnostic — `max_abs_error <= tolerance` is
/// the assertion contract that gated the `Ok` return.
#[derive(Debug, Clone)]
pub struct ForwardEquivalenceReport {
    /// Largest per-element absolute error across all probe tokens.
    pub max_abs_error: f64,
    /// Mean absolute error across all `num_probe_tokens * vocab_size` logits.
    pub mean_abs_error: f64,
    /// Tokens actually probed (deterministic from `seed` and `vocab_size`).
    pub probe_tokens: Vec<u32>,
    /// Tolerance the gate evaluated against (echoed for downstream logging).
    pub tolerance: f64,
}

/// **Refuse-on-fail forward-equivalence gate** for QuaRot Qwen3.5 conversion.
///
/// Runs the rotation-chain probe (see module doc) on `original` and
/// `rotated`, returns `Ok(report)` when the per-element max-abs error
/// stays within `forward_cfg.tolerance`, and **returns
/// `Err(InferenceError::Inference)`** otherwise. The error message
/// includes the observed max-abs and mean-abs deltas and the configured
/// tolerance so callers can surface a useful diagnostic without
/// re-running the probe.
///
/// The converter binary (step 3c-5) MUST treat the `Err` return as a
/// hard refuse: do NOT proceed to quantization, do NOT write the output
/// `.q4` file, do NOT mutate the output `config.json`. The whole point
/// of this gate is to keep wrong-output artifacts off disk.
///
/// # Errors
///
/// - `cfg.is_moe()` — MoE conversion is deferred to v1.
/// - `forward_cfg.num_probe_tokens == 0` or `tolerance <= 0`.
/// - Any required tensor is missing from either working set, or has the
///   wrong shape or data length.
/// - Forward delta exceeds tolerance — the refuse-on-fail case.
pub fn assert_forward_equivalence_qwen35(
    original: &HashMap<String, TensorEntry>,
    rotated: &HashMap<String, TensorEntry>,
    cfg: &Qwen35Config,
    forward_cfg: &ForwardEquivalenceConfig,
) -> Result<ForwardEquivalenceReport, InferenceError> {
    if cfg.is_moe() {
        return Err(InferenceError::Inference(
            "assert_forward_equivalence_qwen35: MoE configs are deferred to v1 \
             (the rotation/fusion pipeline rejects MoE upstream; this probe \
             has no expert-mixing path)"
                .to_string(),
        ));
    }
    if forward_cfg.num_probe_tokens == 0 {
        return Err(InferenceError::Inference(
            "assert_forward_equivalence_qwen35: num_probe_tokens must be > 0".to_string(),
        ));
    }
    if !forward_cfg.tolerance.is_finite() || forward_cfg.tolerance <= 0.0 {
        return Err(InferenceError::Inference(format!(
            "assert_forward_equivalence_qwen35: tolerance must be a positive finite value, got {}",
            forward_cfg.tolerance
        )));
    }
    if cfg.vocab_size == 0 {
        return Err(InferenceError::Inference(
            "assert_forward_equivalence_qwen35: cfg.vocab_size must be > 0".to_string(),
        ));
    }

    let probe_tokens = deterministic_probe_tokens(
        forward_cfg.seed,
        forward_cfg.num_probe_tokens,
        cfg.vocab_size,
    );

    let mut max_abs = 0.0_f64;
    let mut total_abs = 0.0_f64;
    let mut count: usize = 0;

    for &token in &probe_tokens {
        let logits_orig = rotation_chain_probe_qwen35(original, cfg, token)?;
        let logits_rot = rotation_chain_probe_qwen35(rotated, cfg, token)?;
        if logits_orig.len() != logits_rot.len() {
            return Err(InferenceError::Inference(format!(
                "assert_forward_equivalence_qwen35: probe logits length mismatch \
                 (original={}, rotated={}) on token {token}",
                logits_orig.len(),
                logits_rot.len()
            )));
        }
        for (a, b) in logits_orig.iter().zip(logits_rot.iter()) {
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
            }
            total_abs += d;
            count += 1;
        }
    }

    let mean_abs = if count > 0 {
        total_abs / count as f64
    } else {
        0.0
    };

    if max_abs > forward_cfg.tolerance {
        return Err(InferenceError::Inference(format!(
            "forward-equivalence refused: max_abs_error={max_abs} exceeds tolerance={} \
             (mean_abs_error={mean_abs}, probe_tokens={:?}). \
             Do NOT write conversion artifacts — the pipeline produced logits \
             that disagree with the original model.",
            forward_cfg.tolerance, probe_tokens
        )));
    }

    Ok(ForwardEquivalenceReport {
        max_abs_error: max_abs,
        mean_abs_error: mean_abs,
        probe_tokens,
        tolerance: forward_cfg.tolerance,
    })
}

fn deterministic_probe_tokens(seed: u64, n: usize, vocab_size: usize) -> Vec<u32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 32) % vocab_size as u64) as u32
        })
        .collect()
}

fn rotation_chain_probe_qwen35(
    tensors: &HashMap<String, TensorEntry>,
    cfg: &Qwen35Config,
    token: u32,
) -> Result<Vec<f64>, InferenceError> {
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let intermediate = cfg.intermediate_size;
    let full_q_dim = cfg.full_q_dim();
    let linear_output_dim = cfg.linear_output_dim();
    let eps = f64::from(cfg.rms_norm_eps);

    if (token as usize) >= vocab {
        return Err(InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: token id {token} out of range (vocab_size={vocab})"
        )));
    }

    // 1. Embed lookup.
    let embed = get_tensor_2d(tensors, QWEN35_EMBED_TOKENS_NAME, vocab, hidden)?;
    let row = (token as usize) * hidden;
    let mut h: Vec<f64> = embed.data[row..row + hidden].to_vec();

    // 2. Per-layer residual chain.
    for layer in 0..cfg.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{layer}");

        // Pre-attention norm (shifted).
        let gamma_in = get_tensor_1d(tensors, &format!("{prefix}.input_layernorm.weight"), hidden)?;
        let h_pre = rms_normalize_shifted(&h, &gamma_in.data, eps);

        // Linearized attention block.
        let attn_out = if cfg.is_full_attention(layer) {
            let q_proj = get_tensor_2d(
                tensors,
                &format!("{prefix}.self_attn.q_proj.weight"),
                2 * full_q_dim,
                hidden,
            )?;
            let o_proj = get_tensor_2d(
                tensors,
                &format!("{prefix}.self_attn.o_proj.weight"),
                hidden,
                full_q_dim,
            )?;
            let q_full = matvec_f64(&q_proj.data, 2 * full_q_dim, hidden, &h_pre);
            matvec_f64(&o_proj.data, hidden, full_q_dim, &q_full[..full_q_dim])
        } else {
            let in_proj_z = get_tensor_2d(
                tensors,
                &format!("{prefix}.linear_attn.in_proj_z.weight"),
                linear_output_dim,
                hidden,
            )?;
            let out_proj = get_tensor_2d(
                tensors,
                &format!("{prefix}.linear_attn.out_proj.weight"),
                hidden,
                linear_output_dim,
            )?;
            let z = matvec_f64(&in_proj_z.data, linear_output_dim, hidden, &h_pre);
            matvec_f64(&out_proj.data, hidden, linear_output_dim, &z)
        };
        add_in_place(&mut h, &attn_out);

        // Post-attention norm (shifted).
        let gamma_post = get_tensor_1d(
            tensors,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
        )?;
        let h_pre_mlp = rms_normalize_shifted(&h, &gamma_post.data, eps);

        // Linearized MLP block: `gate + up` replaces `silu(gate) ⊙ up`.
        let gate_proj = get_tensor_2d(
            tensors,
            &format!("{prefix}.mlp.gate_proj.weight"),
            intermediate,
            hidden,
        )?;
        let up_proj = get_tensor_2d(
            tensors,
            &format!("{prefix}.mlp.up_proj.weight"),
            intermediate,
            hidden,
        )?;
        let down_proj = get_tensor_2d(
            tensors,
            &format!("{prefix}.mlp.down_proj.weight"),
            hidden,
            intermediate,
        )?;
        let gate = matvec_f64(&gate_proj.data, intermediate, hidden, &h_pre_mlp);
        let up = matvec_f64(&up_proj.data, intermediate, hidden, &h_pre_mlp);
        let mid: Vec<f64> = gate.iter().zip(up.iter()).map(|(a, b)| a + b).collect();
        let mlp_out = matvec_f64(&down_proj.data, hidden, intermediate, &mid);
        add_in_place(&mut h, &mlp_out);
    }

    // 3. Final norm (shifted).
    let gamma_final = get_tensor_1d(tensors, QWEN35_FINAL_NORM_NAME, hidden)?;
    let h_final = rms_normalize_shifted(&h, &gamma_final.data, eps);

    // 4. lm_head matvec — fall back to embed_tokens for tied originals.
    let lm_tensor = if tensors.contains_key(QWEN35_LM_HEAD_NAME) {
        get_tensor_2d(tensors, QWEN35_LM_HEAD_NAME, vocab, hidden)?
    } else if cfg.tie_word_embeddings {
        embed
    } else {
        return Err(InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: untied config requires `{QWEN35_LM_HEAD_NAME}` \
             in the working set (config says `tie_word_embeddings=false` and no fallback \
             is valid in that case)"
        )));
    };
    Ok(matvec_f64(&lm_tensor.data, vocab, hidden, &h_final))
}

fn get_tensor_2d<'a>(
    tensors: &'a HashMap<String, TensorEntry>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<&'a TensorEntry, InferenceError> {
    let t = tensors.get(name).ok_or_else(|| {
        InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: tensor `{name}` not in working set"
        ))
    })?;
    if t.shape.len() != 2 || t.shape[0] != rows || t.shape[1] != cols {
        return Err(InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: tensor `{name}` shape {:?} != expected [{rows}, {cols}]",
            t.shape
        )));
    }
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: rows*cols overflow on `{name}` ({rows}*{cols})"
        ))
    })?;
    if t.data.len() != expected {
        return Err(InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: tensor `{name}` data.len()={} != rows*cols {expected}",
            t.data.len()
        )));
    }
    Ok(t)
}

fn get_tensor_1d<'a>(
    tensors: &'a HashMap<String, TensorEntry>,
    name: &str,
    len: usize,
) -> Result<&'a TensorEntry, InferenceError> {
    let t = tensors.get(name).ok_or_else(|| {
        InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: tensor `{name}` not in working set"
        ))
    })?;
    if t.shape.len() != 1 || t.shape[0] != len {
        return Err(InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: tensor `{name}` shape {:?} != expected [{len}]",
            t.shape
        )));
    }
    if t.data.len() != len {
        return Err(InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: tensor `{name}` data.len()={} != expected {len}",
            t.data.len()
        )));
    }
    Ok(t)
}

fn rms_normalize_shifted(h: &[f64], gamma: &[f64], eps: f64) -> Vec<f64> {
    debug_assert_eq!(h.len(), gamma.len());
    let n = h.len();
    let sum_sq: f64 = h.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f64 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    h.iter()
        .zip(gamma.iter())
        .map(|(v, g)| v * inv_rms * (1.0 + g))
        .collect()
}

fn matvec_f64(w: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(w.len(), rows * cols);
    let mut y = vec![0.0_f64; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        y[r] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }
    y
}

fn add_in_place(h: &mut [f64], addend: &[f64]) {
    debug_assert_eq!(h.len(), addend.len());
    for (a, b) in h.iter_mut().zip(addend.iter()) {
        *a += b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen35_config::{LayerType, Qwen35Config, compute_layer_types};
    use crate::quant::quarot::hadamard::RandomizedHadamard;
    use crate::quant::quarot::lm_head::{
        materialize_lm_head_for_qwen35, qwen35_final_norm_fusion_target,
    };
    use crate::quant::quarot::pipeline::{absorb_rotations, fuse_rmsnorms};
    use crate::quant::quarot::plan::RotationPlan;
    use crate::quant::quarot::rmsnorm_fusion::qwen35_per_layer_fusion_plan;

    /// Tiny Qwen3.5 config tuned for tractable probe tests:
    /// - `hidden_size = 8` (power of 2 for Hadamard)
    /// - 2 layers, one GDN + one GQA so the probe exercises both branches
    /// - small intermediate/vocab so matvecs stay sub-millisecond
    fn tiny_test_cfg() -> Qwen35Config {
        let mut cfg = Qwen35Config::qwen35_0_8b();
        cfg.hidden_size = 8;
        cfg.num_hidden_layers = 2;
        cfg.vocab_size = 4;
        cfg.intermediate_size = 16;
        cfg.num_attention_heads = 2;
        cfg.num_key_value_heads = 1;
        cfg.head_dim = 4;
        cfg.linear_num_key_heads = 1;
        cfg.linear_key_head_dim = 2;
        cfg.linear_value_head_dim = 2;
        cfg.linear_num_value_heads = Some(1);
        cfg.full_attention_interval = 2;
        cfg.layer_types = compute_layer_types(cfg.num_hidden_layers, cfg.full_attention_interval);
        cfg.layer_mask = vec![true; cfg.num_hidden_layers];
        cfg.tie_word_embeddings = false;
        cfg.rms_norm_eps = 1e-6;
        cfg
    }

    fn tied_tiny_test_cfg() -> Qwen35Config {
        let mut cfg = tiny_test_cfg();
        cfg.tie_word_embeddings = true;
        cfg
    }

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

    fn insert_tensor(
        tensors: &mut HashMap<String, TensorEntry>,
        name: &str,
        shape: Vec<usize>,
        data: Vec<f64>,
    ) {
        tensors.insert(
            name.to_string(),
            TensorEntry {
                name: name.to_string(),
                shape,
                data,
            },
        );
    }

    /// Build a working set with shapes consistent with `cfg`'s loader
    /// expectations. Includes every tensor the fusion plan and rotation
    /// plan reference, plus `embed_tokens` and `final_norm`. Does NOT
    /// include `lm_head.weight` — the caller adds it separately (or via
    /// `materialize_lm_head_for_qwen35` for tied configs).
    fn build_working_set(cfg: &Qwen35Config, seed: u64) -> HashMap<String, TensorEntry> {
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let intermediate = cfg.intermediate_size;
        let full_q_dim = cfg.full_q_dim();
        let full_kv_dim = cfg.full_kv_dim();
        let linear_qkv_dim = cfg.linear_qkv_dim();
        let linear_output_dim = cfg.linear_output_dim();
        let linear_num_heads = cfg.linear_num_key_heads;

        let mut tensors = HashMap::new();
        insert_tensor(
            &mut tensors,
            QWEN35_EMBED_TOKENS_NAME,
            vec![vocab, hidden],
            synthetic_f64(vocab * hidden, seed.wrapping_add(1)),
        );
        insert_tensor(
            &mut tensors,
            QWEN35_FINAL_NORM_NAME,
            vec![hidden],
            synthetic_f64(hidden, seed.wrapping_add(2)),
        );

        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{i}");
            let layer_seed = seed.wrapping_add(100 + i as u64);

            insert_tensor(
                &mut tensors,
                &format!("{prefix}.input_layernorm.weight"),
                vec![hidden],
                synthetic_f64(hidden, layer_seed.wrapping_add(1)),
            );
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
                vec![hidden],
                synthetic_f64(hidden, layer_seed.wrapping_add(2)),
            );

            if cfg.is_full_attention(i) {
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    vec![2 * full_q_dim, hidden],
                    synthetic_f64(2 * full_q_dim * hidden, layer_seed.wrapping_add(10)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    vec![full_kv_dim, hidden],
                    synthetic_f64(full_kv_dim * hidden, layer_seed.wrapping_add(11)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    vec![full_kv_dim, hidden],
                    synthetic_f64(full_kv_dim * hidden, layer_seed.wrapping_add(12)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    vec![hidden, full_q_dim],
                    synthetic_f64(hidden * full_q_dim, layer_seed.wrapping_add(13)),
                );
            } else {
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    vec![linear_qkv_dim, hidden],
                    synthetic_f64(linear_qkv_dim * hidden, layer_seed.wrapping_add(20)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.linear_attn.in_proj_z.weight"),
                    vec![linear_output_dim, hidden],
                    synthetic_f64(linear_output_dim * hidden, layer_seed.wrapping_add(21)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.linear_attn.in_proj_a.weight"),
                    vec![linear_num_heads, hidden],
                    synthetic_f64(linear_num_heads * hidden, layer_seed.wrapping_add(22)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.linear_attn.in_proj_b.weight"),
                    vec![linear_num_heads, hidden],
                    synthetic_f64(linear_num_heads * hidden, layer_seed.wrapping_add(23)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.linear_attn.out_proj.weight"),
                    vec![hidden, linear_output_dim],
                    synthetic_f64(hidden * linear_output_dim, layer_seed.wrapping_add(24)),
                );
            }

            insert_tensor(
                &mut tensors,
                &format!("{prefix}.mlp.gate_proj.weight"),
                vec![intermediate, hidden],
                synthetic_f64(intermediate * hidden, layer_seed.wrapping_add(30)),
            );
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.mlp.up_proj.weight"),
                vec![intermediate, hidden],
                synthetic_f64(intermediate * hidden, layer_seed.wrapping_add(31)),
            );
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.mlp.down_proj.weight"),
                vec![hidden, intermediate],
                synthetic_f64(hidden * intermediate, layer_seed.wrapping_add(32)),
            );
        }

        tensors
    }

    fn full_pipeline_plans(
        cfg: &Qwen35Config,
    ) -> (
        Vec<crate::quant::quarot::rmsnorm_fusion::RmsNormFusionTarget>,
        RotationPlan,
    ) {
        let mut fusion = qwen35_per_layer_fusion_plan(cfg).unwrap();
        fusion.push(qwen35_final_norm_fusion_target());
        (fusion, RotationPlan::qwen35_residual_stream_linear_layers())
    }

    /// Sanity: the tiny test config really has one GDN + one GQA layer.
    #[test]
    fn tiny_cfg_has_one_full_and_one_linear_layer() {
        let cfg = tiny_test_cfg();
        assert_eq!(cfg.num_hidden_layers, 2);
        assert_eq!(cfg.layer_types[0], LayerType::LinearAttention);
        assert_eq!(cfg.layer_types[1], LayerType::FullAttention);
    }

    /// Happy path (untied): build → materialize is a no-op (already untied
    /// with lm_head present) → fuse → rotate → harness must pass.
    #[test]
    fn forward_equivalence_passes_on_full_untied_pipeline() {
        let cfg = tiny_test_cfg();
        let mut original = build_working_set(&cfg, 1);
        // Untied: lm_head must be present pre-pipeline.
        insert_tensor(
            &mut original,
            QWEN35_LM_HEAD_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            synthetic_f64(cfg.vocab_size * cfg.hidden_size, 999),
        );
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        let (fusion, rot_plan) = full_pipeline_plans(&cfg);
        fuse_rmsnorms(&mut rotated, &fusion).unwrap();
        let rotation = RandomizedHadamard::new(0xC0FFEE, cfg.hidden_size).unwrap();
        absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

        let report = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap();
        assert!(report.max_abs_error < 1e-5, "unexpected delta: {report:?}");
        assert_eq!(report.probe_tokens.len(), 4);
        for &t in &report.probe_tokens {
            assert!((t as usize) < cfg.vocab_size);
        }
    }

    /// Happy path (tied): pre-pipeline lm_head is absent (probe falls
    /// back to embed_tokens); rotated set materializes + fuses + rotates.
    #[test]
    fn forward_equivalence_passes_on_full_tied_pipeline() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 2);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        let (fusion, rot_plan) = full_pipeline_plans(&cfg);
        fuse_rmsnorms(&mut rotated, &fusion).unwrap();
        let rotation = RandomizedHadamard::new(0xFEED_FACE, cfg.hidden_size).unwrap();
        absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

        let report = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap();
        assert!(report.max_abs_error < 1e-5, "unexpected delta: {report:?}");
    }

    /// Sanity: identical working sets give zero delta. Catches probe
    /// non-determinism that would silently mask real bugs.
    #[test]
    fn forward_equivalence_is_zero_on_identical_sets() {
        let cfg = tiny_test_cfg();
        let mut original = build_working_set(&cfg, 3);
        insert_tensor(
            &mut original,
            QWEN35_LM_HEAD_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            synthetic_f64(cfg.vocab_size * cfg.hidden_size, 333),
        );
        let rotated = original.clone();

        let report = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap();
        assert_eq!(report.max_abs_error, 0.0);
        assert_eq!(report.mean_abs_error, 0.0);
    }

    /// Refuse: rotate without fusing the final_norm target. The mismatch
    /// `lm_head_rot · rmsnorm(R·h)` vs `embed_tokens · diag(1+γ_final) · rmsnorm(h)`
    /// produces a non-trivial delta and the gate must refuse.
    #[test]
    fn forward_equivalence_refuses_when_final_norm_fusion_skipped() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 4);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        // BUG: drop the final_norm fusion target — fuse only per-layer norms.
        let per_layer = qwen35_per_layer_fusion_plan(&cfg).unwrap();
        fuse_rmsnorms(&mut rotated, &per_layer).unwrap();
        let rotation = RandomizedHadamard::new(0xBADC0DE, cfg.hidden_size).unwrap();
        absorb_rotations(
            &mut rotated,
            &RotationPlan::qwen35_residual_stream_linear_layers(),
            &rotation,
        )
        .unwrap();

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("forward-equivalence refused"),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("exceeds tolerance"), "unexpected error: {msg}");
    }

    /// Refuse: a single tensor in the rotated set is silently corrupted
    /// after a correct pipeline run. Real-world analogue: a converter bug
    /// that re-quantizes a tensor a second time, or scales the wrong
    /// matrix. The harness must catch this even though the rotation
    /// algebra itself was applied correctly to every other tensor.
    #[test]
    fn forward_equivalence_refuses_on_corrupted_rotated_tensor() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 5);
        let mut rotated = original.clone();

        // Correct pipeline.
        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        let (fusion, rot_plan) = full_pipeline_plans(&cfg);
        fuse_rmsnorms(&mut rotated, &fusion).unwrap();
        let rotation = RandomizedHadamard::new(0x55AA_55AA, cfg.hidden_size).unwrap();
        absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

        // Corrupt one matrix after the pipeline.
        let lm = rotated
            .get_mut(QWEN35_LM_HEAD_NAME)
            .expect("lm_head should exist after materialize");
        for v in lm.data.iter_mut() {
            *v *= 1.25;
        }

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("forward-equivalence refused"),
            "unexpected error: {msg}"
        );
    }

    /// Refuse: rotate without fusing per-layer norms first. Rotation
    /// absorbed AROUND a non-zero γ_in breaks the residual stream
    /// because `diag(1+γ) · R^T ≠ R^T · diag(1+γ)`.
    #[test]
    fn forward_equivalence_refuses_when_per_layer_fusion_skipped() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 6);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        // BUG: only fuse final_norm, skip per-layer.
        let final_only = vec![qwen35_final_norm_fusion_target()];
        fuse_rmsnorms(&mut rotated, &final_only).unwrap();
        let rotation = RandomizedHadamard::new(0xABCDEF, cfg.hidden_size).unwrap();
        absorb_rotations(
            &mut rotated,
            &RotationPlan::qwen35_residual_stream_linear_layers(),
            &rotation,
        )
        .unwrap();

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("forward-equivalence refused"),
            "unexpected error: {msg}"
        );
    }

    /// Untied original missing lm_head → harness errors with a useful
    /// message (vs. the silent fall-through that would happen for tied).
    #[test]
    fn forward_equivalence_errors_on_untied_original_missing_lm_head() {
        let cfg = tiny_test_cfg(); // untied
        let original = build_working_set(&cfg, 7); // no lm_head added
        let mut rotated = original.clone();
        insert_tensor(
            &mut rotated,
            QWEN35_LM_HEAD_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            synthetic_f64(cfg.vocab_size * cfg.hidden_size, 7777),
        );

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(QWEN35_LM_HEAD_NAME), "unexpected error: {msg}");
        assert!(msg.contains("untied"), "unexpected error: {msg}");
    }

    #[test]
    fn forward_equivalence_errors_on_missing_required_tensor() {
        let cfg = tiny_test_cfg();
        let mut original = build_working_set(&cfg, 8);
        insert_tensor(
            &mut original,
            QWEN35_LM_HEAD_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            synthetic_f64(cfg.vocab_size * cfg.hidden_size, 888),
        );
        let rotated = original.clone();

        // Remove a required tensor from `original` to simulate a partial load.
        original.remove(QWEN35_FINAL_NORM_NAME);

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(QWEN35_FINAL_NORM_NAME),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("not in working set"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn forward_equivalence_errors_on_shape_mismatch() {
        let cfg = tiny_test_cfg();
        let mut original = build_working_set(&cfg, 9);
        // Inject a wrong-shape embed_tokens.
        original.insert(
            QWEN35_EMBED_TOKENS_NAME.to_string(),
            TensorEntry {
                name: QWEN35_EMBED_TOKENS_NAME.to_string(),
                shape: vec![cfg.vocab_size, cfg.hidden_size + 1],
                data: vec![0.0; cfg.vocab_size * (cfg.hidden_size + 1)],
            },
        );
        insert_tensor(
            &mut original,
            QWEN35_LM_HEAD_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            synthetic_f64(cfg.vocab_size * cfg.hidden_size, 999),
        );
        let rotated = original.clone();

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(QWEN35_EMBED_TOKENS_NAME),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("shape"), "unexpected error: {msg}");
    }

    #[test]
    fn forward_equivalence_rejects_moe_config() {
        let cfg = Qwen35Config::qwen36_35b_a3b();
        assert!(cfg.is_moe());
        let original = HashMap::new();
        let rotated = HashMap::new();
        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("MoE"), "unexpected error: {msg}");
    }

    #[test]
    fn forward_equivalence_rejects_zero_probe_tokens() {
        let cfg = tiny_test_cfg();
        let original = HashMap::new();
        let rotated = HashMap::new();
        let fc = ForwardEquivalenceConfig {
            num_probe_tokens: 0,
            ..Default::default()
        };
        let err = assert_forward_equivalence_qwen35(&original, &rotated, &cfg, &fc).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("num_probe_tokens must be > 0"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn forward_equivalence_rejects_non_positive_tolerance() {
        let cfg = tiny_test_cfg();
        let original = HashMap::new();
        let rotated = HashMap::new();
        for bad in [0.0_f64, -1e-5, f64::NAN] {
            let fc = ForwardEquivalenceConfig {
                tolerance: bad,
                ..Default::default()
            };
            let err =
                assert_forward_equivalence_qwen35(&original, &rotated, &cfg, &fc).unwrap_err();
            let msg = format!("{err}");
            assert!(
                msg.contains("tolerance must be a positive finite value"),
                "unexpected error for tolerance={bad}: {msg}"
            );
        }
    }

    /// Determinism: same seed → same probe tokens. Catches a clock-based or
    /// uninitialized RNG regression.
    #[test]
    fn probe_tokens_are_deterministic_in_seed() {
        let a = deterministic_probe_tokens(0xDEAD_BEEF, 4, 100);
        let b = deterministic_probe_tokens(0xDEAD_BEEF, 4, 100);
        assert_eq!(a, b);
        let c = deterministic_probe_tokens(0xDEAD_BEEF_u64.wrapping_add(1), 4, 100);
        assert_ne!(
            a, c,
            "different seeds should produce different probe tokens"
        );
    }

    /// Refuse error message includes the observed delta and tolerance —
    /// catches the regression where the error message would lose the
    /// diagnostic numbers operators need to triage in the binary.
    #[test]
    fn refuse_error_message_includes_max_abs_and_tolerance() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 10);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        // Real divergence: fuse only per-layer norms (skip final_norm) then
        // rotate. `diag(1 + γ_final) · R^T ≠ R^T · diag(1 + γ_final)` makes
        // the rotated lm_head produce different logits than the original.
        let per_layer = qwen35_per_layer_fusion_plan(&cfg).unwrap();
        fuse_rmsnorms(&mut rotated, &per_layer).unwrap();
        let rotation = RandomizedHadamard::new(0x1234_5678, cfg.hidden_size).unwrap();
        absorb_rotations(
            &mut rotated,
            &RotationPlan::qwen35_residual_stream_linear_layers(),
            &rotation,
        )
        .unwrap();

        let fc = ForwardEquivalenceConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        let err = assert_forward_equivalence_qwen35(&original, &rotated, &cfg, &fc).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("max_abs_error="), "unexpected error: {msg}");
        assert!(
            msg.contains("exceeds tolerance="),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("mean_abs_error="), "unexpected error: {msg}");
        assert!(msg.contains("probe_tokens="), "unexpected error: {msg}");
    }
}

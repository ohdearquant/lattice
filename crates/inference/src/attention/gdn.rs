//! GatedDeltaNet: linear attention with gated recurrent state updates.
//!
//! This implements the GatedDeltaNet attention mechanism used in Qwen3.5-2B's
//! 18 linear-attention layers. Each step maintains a recurrent state matrix S
//! per head (key_dim x value_dim) and a causal conv1d buffer.
//!
//! Algorithm per step:
//! 1. Project input → Q, K, V, output gate z, update rate beta, decay alpha
//! 2. Causal depthwise conv1d on QKV
//! 3. L2-normalize Q and K
//! 4. Compute decay gate: g = exp(-exp(A_log) * softplus(alpha + dt_bias))
//! 5. Update recurrent state: S = g*S + outer(k, (v - S@k) * beta)
//! 6. Retrieve output: o = S @ q / sqrt(key_dim)
//! 7. Gated RMSNorm + output projection

use crate::forward::cpu::matmul_bt;
use crate::model::qwen35_config::Qwen35Config;

/// **Unstable**: weight tensors for a GatedDeltaNet layer; field layout follows Qwen3.5 checkpoint format.
#[derive(Debug, Clone)]
pub struct GatedDeltaNetWeights {
    /// QKV projection: [qkv_dim, hidden_size] where qkv_dim = Q_dim + K_dim + V_dim
    pub in_proj_qkv: Vec<f32>,
    pub in_proj_qkv_rows: usize,
    pub in_proj_qkv_cols: usize,

    /// Output gate projection: [output_dim, hidden_size]
    pub in_proj_z: Vec<f32>,
    pub in_proj_z_rows: usize,
    pub in_proj_z_cols: usize,

    /// Update rate projection: [num_heads, hidden_size]
    pub in_proj_b: Vec<f32>,
    pub in_proj_b_rows: usize,
    pub in_proj_b_cols: usize,

    /// Decay input projection: [num_heads, hidden_size]
    pub in_proj_a: Vec<f32>,
    pub in_proj_a_rows: usize,
    pub in_proj_a_cols: usize,

    /// Learnable log-decay: `[num_heads]`
    pub a_log: Vec<f32>,

    /// Learnable time-step bias: `[num_heads]`
    pub dt_bias: Vec<f32>,

    /// Depthwise conv1d weights: [qkv_dim, 1, kernel_size] stored as [qkv_dim, kernel_size]
    pub conv1d_weight: Vec<f32>,
    pub conv_dim: usize,
    pub kernel_size: usize,

    /// Output gated RMSNorm gamma: `[output_dim]`
    pub norm_weight: Vec<f32>,

    /// Output projection: [hidden_size, output_dim]
    pub out_proj: Vec<f32>,
    pub out_proj_rows: usize,
    pub out_proj_cols: usize,
}

/// **Unstable**: recurrent KV state for GatedDeltaNet; layout tied to Qwen3.5 architecture.
#[derive(Debug, Clone)]
pub struct GatedDeltaNetState {
    /// Recurrent state matrices: [num_heads, key_dim, value_dim] in f32.
    /// Stored flat as num_heads consecutive key_dim*value_dim blocks.
    pub s_matrices: Vec<f32>,

    /// Conv1d rolling buffer: [conv_dim, kernel_size - 1].
    /// Each channel maintains the last (kernel_size - 1) inputs.
    pub conv_buffer: Vec<f32>,

    key_dim: usize,
    value_dim: usize,
}

impl GatedDeltaNetState {
    /// **Unstable**: allocate zeroed recurrent state for one layer from config.
    pub fn new(cfg: &Qwen35Config) -> Self {
        let value_heads = cfg.linear_num_value_heads();
        let key_dim = cfg.linear_key_head_dim;
        let value_dim = cfg.linear_value_head_dim;
        let conv_dim = cfg.linear_qkv_dim();
        let buf_len = cfg.linear_conv_kernel_dim - 1;

        Self {
            s_matrices: vec![0.0; value_heads * key_dim * value_dim],
            conv_buffer: vec![0.0; conv_dim * buf_len],
            key_dim,
            value_dim,
        }
    }

    /// **Unstable**: zero all recurrent state matrices and conv buffers.
    pub fn reset(&mut self) {
        self.s_matrices.fill(0.0);
        self.conv_buffer.fill(0.0);
    }

    /// Snapshot the live recurrent state for speculative rollback.
    /// See ADR-052: snapshot-before-draft protocol.
    pub fn snapshot(&self) -> GdnLayerSnapshot {
        (self.s_matrices.clone(), self.conv_buffer.clone())
    }

    /// Restore recurrent state from a prior `snapshot`. The snapshot must have been taken
    /// from a state with the same key/value dims; in debug builds this is enforced via
    /// `debug_assert_eq!` on buffer lengths.
    pub fn restore_from(&mut self, snap: &GdnLayerSnapshot) {
        debug_assert_eq!(self.s_matrices.len(), snap.0.len());
        debug_assert_eq!(self.conv_buffer.len(), snap.1.len());
        self.s_matrices.copy_from_slice(&snap.0);
        self.conv_buffer.copy_from_slice(&snap.1);
    }

    /// Access S matrix for head h as a mutable slice [key_dim, value_dim].
    #[inline]
    fn s_matrix_mut(&mut self, h: usize) -> &mut [f32] {
        let size = self.key_dim * self.value_dim;
        &mut self.s_matrices[h * size..(h + 1) * size]
    }
}

/// Opaque snapshot of one GDN layer's recurrent state.
///
/// Layout: `(s_matrices clone, conv_buffer clone)`. See ADR-052.
pub type GdnLayerSnapshot = (Vec<f32>, Vec<f32>);

/// Snapshot of all GDN layers for a single model instance.
pub type GdnSnapshot = Vec<GdnLayerSnapshot>;

/// **Unstable**: scratch buffers for the scalar GatedDeltaNet step; buffer set may change.
#[derive(Debug, Default)]
pub struct GatedDeltaNetScratch {
    qkv_proj: Vec<f32>,
    z_proj: Vec<f32>,
    beta_proj: Vec<f32>,
    alpha_proj: Vec<f32>,
    conv_input: Vec<f32>,
    conv_output: Vec<f32>,
    output_heads: Vec<f32>,
    gated_norm_buf: Vec<f32>,
    final_out: Vec<f32>,
}

impl GatedDeltaNetScratch {
    /// **Unstable**: grow scratch buffers to fit the given projections.
    pub fn ensure_capacity(
        &mut self,
        qkv_dim: usize,
        output_dim: usize,
        num_heads: usize,
        hidden_size: usize,
    ) {
        resize_if_needed(&mut self.qkv_proj, qkv_dim);
        resize_if_needed(&mut self.z_proj, output_dim);
        resize_if_needed(&mut self.beta_proj, num_heads);
        resize_if_needed(&mut self.alpha_proj, num_heads);
        resize_if_needed(&mut self.conv_input, qkv_dim);
        resize_if_needed(&mut self.conv_output, qkv_dim);
        resize_if_needed(&mut self.output_heads, output_dim);
        resize_if_needed(&mut self.gated_norm_buf, output_dim);
        resize_if_needed(&mut self.final_out, hidden_size);
    }
}

#[inline]
fn resize_if_needed(buf: &mut Vec<f32>, needed: usize) {
    if buf.len() < needed {
        buf.resize(needed, 0.0);
    }
}

/// **Unstable**: scalar GatedDeltaNet single-token step; being replaced by fused SIMD path.
///
/// Process a single token through the GatedDeltaNet layer.
///
/// `input`: hidden state `[hidden_size]`
/// `state`: mutable recurrent state for this layer
/// `weights`: layer weights
/// `cfg`: model config
/// `scratch`: reusable scratch buffers
/// `output`: output buffer `[hidden_size]`, written in-place
pub fn gated_delta_net_step(
    input: &[f32],
    state: &mut GatedDeltaNetState,
    weights: &GatedDeltaNetWeights,
    cfg: &Qwen35Config,
    scratch: &mut GatedDeltaNetScratch,
    output: &mut [f32],
) {
    let hidden = cfg.hidden_size;
    let num_heads = cfg.linear_num_key_heads;
    let value_heads = cfg.linear_num_value_heads();
    let ratio = value_heads / num_heads;
    let key_dim = cfg.linear_key_head_dim;
    let value_dim = cfg.linear_value_head_dim;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let kernel_size = cfg.linear_conv_kernel_dim;

    scratch.ensure_capacity(qkv_dim, output_dim, value_heads, hidden);

    // 1. Projections (BLAS: input [1, hidden] @ weight^T [qkv_dim, hidden])
    matmul_bt(
        input,
        &weights.in_proj_qkv,
        &mut scratch.qkv_proj[..qkv_dim],
        1,
        hidden,
        qkv_dim,
    );

    matmul_bt(
        input,
        &weights.in_proj_z,
        &mut scratch.z_proj[..output_dim],
        1,
        hidden,
        output_dim,
    );

    matmul_bt(
        input,
        &weights.in_proj_b,
        &mut scratch.beta_proj[..value_heads],
        1,
        hidden,
        value_heads,
    );

    matmul_bt(
        input,
        &weights.in_proj_a,
        &mut scratch.alpha_proj[..value_heads],
        1,
        hidden,
        value_heads,
    );

    // Apply sigmoid to beta
    for b in &mut scratch.beta_proj[..value_heads] {
        *b = sigmoid(*b);
    }

    // 2. Conv1d (causal, depthwise)
    // Shift conv buffer left and append new QKV
    apply_causal_conv1d(
        &scratch.qkv_proj[..qkv_dim],
        &mut state.conv_buffer,
        &weights.conv1d_weight,
        &mut scratch.conv_output[..qkv_dim],
        qkv_dim,
        kernel_size,
    );

    // 2b. Apply SiLU activation to conv output (critical — HuggingFace applies this before split)
    for v in &mut scratch.conv_output[..qkv_dim] {
        *v = *v / (1.0 + (-*v).exp()); // silu(x) = x * sigmoid(x)
    }

    // 3. Split conv output into Q, K, V per head
    let q_total = num_heads * key_dim;
    let k_total = num_heads * key_dim;
    // V starts after Q and K
    let v_offset = q_total + k_total;

    // 4-7. Process each head (value-head loop; Q/K use k_head = h/ratio)
    for h in 0..value_heads {
        let k_head = h / ratio;
        let q_start = k_head * key_dim;
        let k_start = q_total + k_head * key_dim;
        let v_start = v_offset + h * value_dim;

        // Extract per-head Q, K, V
        let mut q = scratch.conv_output[q_start..q_start + key_dim].to_vec();
        let mut k = scratch.conv_output[k_start..k_start + key_dim].to_vec();
        let v: Vec<f32> = scratch.conv_output[v_start..v_start + value_dim].to_vec();

        // 4. L2-normalize Q and K
        l2_normalize_vec(&mut q);
        l2_normalize_vec(&mut k);

        // 5. Compute decay gate
        let alpha_h = scratch.alpha_proj[h];
        let g = compute_decay_gate(weights.a_log[h], alpha_h, weights.dt_bias[h]);

        // 6. Recurrent state update
        let s = state.s_matrix_mut(h);

        // Decay: S *= g
        for val in s.iter_mut() {
            *val *= g;
        }

        // Retrieve: kv_mem = S^T @ k → [value_dim]
        // S is [key_dim, value_dim], k is [key_dim]
        // kv_mem[j] = sum_i S[i,j] * k[i]
        let mut kv_mem = vec![0.0f32; value_dim];
        for i in 0..key_dim {
            for j in 0..value_dim {
                kv_mem[j] += s[i * value_dim + j] * k[i];
            }
        }

        // Delta: delta = (v - kv_mem) * beta
        let beta_h = scratch.beta_proj[h];
        let mut delta = vec![0.0f32; value_dim];
        for j in 0..value_dim {
            delta[j] = (v[j] - kv_mem[j]) * beta_h;
        }

        // Update: S += outer(k, delta)
        for i in 0..key_dim {
            for j in 0..value_dim {
                s[i * value_dim + j] += k[i] * delta[j];
            }
        }

        // 7. Output: o = S^T @ q / sqrt(key_dim)
        let scale = 1.0 / (key_dim as f32).sqrt();
        let out_start = h * value_dim;
        for j in 0..value_dim {
            let mut sum = 0.0f32;
            for i in 0..key_dim {
                sum += s[i * value_dim + j] * q[i];
            }
            scratch.output_heads[out_start + j] = sum * scale;
        }
    }

    // 8. Gated RMSNorm + output projection
    // norm_weight is [value_dim] (per-head), applied to each head independently
    // Then gated with z: out = z * (x / rms(x)) * gamma (per head)
    let value_dim = cfg.linear_value_head_dim;
    for h in 0..value_heads {
        let start = h * value_dim;
        let end = start + value_dim;
        gated_rms_norm(
            &scratch.output_heads[start..end],
            &scratch.z_proj[start..end],
            &weights.norm_weight[..value_dim],
            &mut scratch.gated_norm_buf[start..end],
            cfg.rms_norm_eps,
        );
    }

    // Output projection: [1, output_dim] @ out_proj^T [hidden, output_dim]
    matmul_bt(
        &scratch.gated_norm_buf[..output_dim],
        &weights.out_proj,
        &mut output[..hidden],
        1,
        output_dim,
        hidden,
    );
}

/// Apply causal depthwise conv1d.
///
/// Maintains a rolling buffer of the last (kernel_size - 1) inputs.
/// For each channel independently: output[ch] = sum(buffer[ch, :] * weight[ch, :])
fn apply_causal_conv1d(
    new_input: &[f32],
    conv_buffer: &mut [f32],
    conv_weight: &[f32],
    output: &mut [f32],
    conv_dim: usize,
    kernel_size: usize,
) {
    let buf_len = kernel_size - 1;

    for ch in 0..conv_dim {
        // Compute convolution: last (kernel_size-1) values from buffer + new value
        let mut sum = 0.0f32;

        // Weights are stored as [conv_dim, kernel_size] — one row per channel
        let w_offset = ch * kernel_size;

        // Buffer values (oldest to newest)
        for t in 0..buf_len {
            sum += conv_buffer[ch * buf_len + t] * conv_weight[w_offset + t];
        }

        // Current input value (last weight)
        sum += new_input[ch] * conv_weight[w_offset + buf_len];

        output[ch] = sum;

        // Shift buffer left and append new value
        for t in 0..buf_len.saturating_sub(1) {
            conv_buffer[ch * buf_len + t] = conv_buffer[ch * buf_len + t + 1];
        }
        if buf_len > 0 {
            conv_buffer[ch * buf_len + buf_len - 1] = new_input[ch];
        }
    }
}

/// **Unstable**: scalar L2 normalisation; may be replaced by SIMD path from gdn_fused.
///
/// L2-normalize a vector in-place. Matches FLA library: x / sqrt(sum(x^2) + eps).
///
/// ADR-080 C1 fail-closed (#850): a non-finite input (NaN/Inf lane) makes `norm_sq`
/// non-finite too (squaring any non-finite value yields +inf or NaN, and summing any
/// non-finite term keeps the sum non-finite), so checking `norm_sq.is_finite()` alone
/// detects the whole-vector invalid case. GDN state persists across tokens, so letting
/// a poisoned vector through here would contaminate every subsequent token; the whole
/// vector is assigned the literal `0.0` directly rather than multiplied through a
/// zeroed reciprocal (`NaN * 0.0 == NaN` under IEEE-754, so a plain guarded-reciprocal
/// multiply cannot recover a poisoned lane). A zero-input vector stays the CPU's
/// original epsilon-regularized graceful case (`norm_sq == 0.0` is finite, so it flows
/// through the normal path and yields an all-zero output via the numerator).
#[inline]
pub fn l2_normalize_vec(x: &mut [f32]) {
    let eps = 1e-6f32;
    let norm_sq: f32 = x.iter().map(|v| v * v).sum();
    if !norm_sq.is_finite() {
        for v in x.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let inv_norm = 1.0 / (norm_sq + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv_norm;
    }
}

/// Compute decay gate: g = exp(-exp(a_log) * softplus(alpha + dt_bias))
#[inline]
fn compute_decay_gate(a_log: f32, alpha: f32, dt_bias: f32) -> f32 {
    // `a_log.exp()` overflows f32 to +inf for a_log > ~88. Paired with a very negative
    // `alpha + dt_bias` (softplus hard-returns 0.0), the product is `inf * 0.0` = NaN and
    // `exp(NaN)` = NaN — which then propagates through the recurrent state S and poisons
    // every subsequent token. Clamp the decay rate to finite so the product stays finite:
    // `huge * 0 = 0` → g = 1 (the dt≈0 "no decay" limit), `huge * positive` → -inf → g = 0
    // (full decay). No effect for valid checkpoints, where a_log stays O(1).
    let a = a_log.exp().min(f32::MAX);
    let sp = softplus(alpha + dt_bias);
    (-a * sp).exp()
}

/// **Unstable**: numerically-stable softplus, shared with gdn_fused.
///
/// softplus(x) = ln(1 + exp(x)), numerically stable.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // For large x, ln(1+exp(x)) ≈ x
    } else if x < -20.0 {
        0.0 // For very negative x, ln(1+exp(x)) ≈ 0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// **Unstable**: sigmoid activation, shared with gdn_fused.
///
/// sigmoid(x) = 1 / (1 + exp(-x))
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// **Unstable**: gated RMSNorm scalar reference path; used in tests as parity baseline.
///
/// Gated RMSNorm: out = z * (x / rms(x)) * gamma
///
/// `x`: input `[dim]`
/// `z`: output gate `[dim]` (NOT sigmoided — raw projection output used as gate)
/// `gamma`: learnable scale `[dim]`
/// `out`: output buffer `[dim]`
/// `eps`: epsilon for numerical stability
pub fn gated_rms_norm(x: &[f32], z: &[f32], gamma: &[f32], out: &mut [f32], eps: f32) {
    let dim = x.len();
    debug_assert_eq!(z.len(), dim);
    debug_assert_eq!(gamma.len(), dim);
    debug_assert!(out.len() >= dim);

    // RMS of x
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / dim as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Reference: hidden = weight * rms_norm(x) * silu(gate)
    // silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
    for i in 0..dim {
        let silu_z = z[i] / (1.0 + (-z[i]).exp());
        out[i] = (x[i] * inv_rms) * gamma[i] * silu_z;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalization_correctness() {
        let mut v = vec![3.0, 4.0];
        l2_normalize_vec(&mut v);

        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "L2-normalized vector should have unit norm, got {norm}"
        );

        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize_vec(&mut v);
        // Should remain zero, not NaN
        for &x in &v {
            assert!(x == 0.0, "zero vector should remain zero after L2 norm");
        }
    }

    /// ADR-080 C1 fail-closed (#850) table test: every non-finite input class must
    /// zero the WHOLE vector by direct assignment, never leave a poisoned lane by
    /// multiplying it through a guarded-but-zeroed reciprocal (`NaN * 0.0 == NaN` /
    /// `inf * 0.0 == NaN` under IEEE-754). Finite classes (all-zero, very-small,
    /// ordinary) must be numerically unchanged from the pre-#850 behavior.
    ///
    /// Mutation-sensitive: reverting the `!norm_sq.is_finite()` early-return in
    /// `l2_normalize_vec` makes the NaN/+inf/-inf cases below fail (the corrupted
    /// lane stays non-finite in the output instead of being zeroed).
    #[test]
    fn test_l2_normalize_vec_fail_closed_table() {
        // (label, input, expect_all_zero)
        let cases: &[(&str, &[f32], bool)] = &[
            ("nan_lane", &[f32::NAN, 1.0, 2.0, 3.0], true),
            ("pos_inf_lane", &[f32::INFINITY, 1.0, 2.0, 3.0], true),
            ("neg_inf_lane", &[f32::NEG_INFINITY, 1.0, 2.0, 3.0], true),
            ("all_zero", &[0.0, 0.0, 0.0, 0.0], true),
        ];
        for (label, input, expect_all_zero) in cases {
            let mut v = input.to_vec();
            l2_normalize_vec(&mut v);
            let all_finite = v.iter().all(|x| x.is_finite());
            assert!(
                all_finite,
                "case {label}: l2_normalize_vec output must be fully finite, got {v:?}"
            );
            if *expect_all_zero {
                assert!(
                    v.iter().all(|x| *x == 0.0),
                    "case {label}: expected the whole vector zeroed, got {v:?}"
                );
            }
        }

        // very-small finite: norm_sq is finite and > 0, must NOT hit the fail-closed
        // branch -- output should be the ordinary (non-zero) epsilon-regularized result.
        let mut small = vec![1e-20f32, 0.0, 0.0];
        l2_normalize_vec(&mut small);
        assert!(
            small[0] > 0.0 && small[0].is_finite(),
            "very-small finite input must take the ordinary normalize path, got {small:?}"
        );

        // ordinary finite: numerically unchanged from the pre-#850 formula.
        let mut ordinary = vec![3.0f32, 4.0];
        l2_normalize_vec(&mut ordinary);
        assert!((ordinary[0] - 0.6).abs() < 1e-6);
        assert!((ordinary[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_gated_rms_norm_correctness() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let z = vec![1.0, 1.0, 1.0, 1.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];
        let eps = 1e-6;

        gated_rms_norm(&x, &z, &gamma, &mut out, eps);

        let rms = (30.0_f32 / 4.0 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        let silu_1 = 1.0 / (1.0 + (-1.0_f32).exp()); // silu(1) = sigmoid(1)

        for i in 0..4 {
            let expected = x[i] * inv_rms * gamma[i] * silu_1;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "gated_rms_norm[{i}]: expected {expected}, got {}",
                out[i]
            );
        }
    }

    #[test]
    fn test_gated_rms_norm_with_gate() {
        let x = vec![1.0, 2.0];
        let z = vec![0.5, -1.0];
        let gamma = vec![2.0, 3.0];
        let mut out = vec![0.0; 2];

        gated_rms_norm(&x, &z, &gamma, &mut out, 1e-6);

        let rms = ((1.0 + 4.0) / 2.0 + 1e-6_f32).sqrt();
        let inv_rms = 1.0 / rms;

        // out[i] = (x[i] * inv_rms) * gamma[i] * silu(z[i])
        let silu_z0 = 0.5 / (1.0 + (-0.5_f32).exp());
        let silu_z1 = -1.0 / (1.0 + 1.0_f32.exp());
        let expected_0 = (1.0 * inv_rms) * 2.0 * silu_z0;
        let expected_1 = (2.0 * inv_rms) * 3.0 * silu_z1;

        assert!(
            (out[0] - expected_0).abs() < 1e-5,
            "expected {expected_0}, got {}",
            out[0]
        );
        assert!(
            (out[1] - expected_1).abs() < 1e-5,
            "expected {expected_1}, got {}",
            out[1]
        );
    }

    #[test]
    fn test_softplus() {
        // softplus(0) = ln(2) ≈ 0.6931
        assert!((softplus(0.0) - 2.0_f32.ln()).abs() < 1e-6);

        // For large x, softplus(x) ≈ x
        assert!((softplus(30.0) - 30.0).abs() < 1e-3);

        // For very negative x, softplus(x) ≈ 0
        assert!(softplus(-30.0).abs() < 1e-6);

        // Monotonically increasing
        assert!(softplus(1.0) > softplus(0.0));
        assert!(softplus(0.0) > softplus(-1.0));
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_decay_gate_range() {
        // The decay gate should be in (0, 1) for any finite inputs
        let test_cases = [
            (0.0_f32, 0.0_f32, 0.0_f32),
            (1.0, 0.5, 0.1),
            (-1.0, -0.5, -0.1),
            (2.0, 3.0, 1.0),
        ];

        for (a_log, alpha, dt_bias) in test_cases {
            let g = compute_decay_gate(a_log, alpha, dt_bias);
            assert!(
                g > 0.0 && g <= 1.0,
                "decay gate should be in (0, 1], got {g} for a_log={a_log}, alpha={alpha}, dt_bias={dt_bias}"
            );
        }
    }

    #[test]
    fn test_decay_gate_finite_on_exp_overflow() {
        // a_log large enough that exp(a_log) overflows f32 to +inf, paired with a very
        // negative (alpha + dt_bias) where softplus hard-returns 0.0. Pre-guard this was
        // `inf * 0.0` = NaN → `exp(NaN)` = NaN, which then poisons the entire recurrent
        // state S and every subsequent token.
        let g = compute_decay_gate(100.0, -200.0, 0.0);
        assert!(
            g.is_finite() && (0.0..=1.0).contains(&g),
            "decay gate must stay finite in [0,1] under exp overflow, got {g}"
        );
        // softplus(very_negative) ≈ 0 ⇒ dt ≈ 0 ⇒ no time step ⇒ no decay ⇒ g = 1.
        assert!((g - 1.0).abs() < 1e-6, "expected g≈1 for dt≈0, got {g}");

        // The full-decay limit also stays finite: overflowing a_log with a positive dt
        // drives the exponent to -inf, so g → 0 (not NaN).
        let g0 = compute_decay_gate(100.0, 5.0, 0.0);
        assert!(
            g0.is_finite() && g0 >= 0.0,
            "decay gate must stay finite, got {g0}"
        );
        assert!(g0 < 1e-6, "expected g≈0 for huge decay rate, got {g0}");
    }

    #[test]
    fn test_state_decay_reduces_magnitude() {
        let cfg = Qwen35Config::qwen35_2b();
        let mut state = GatedDeltaNetState::new(&cfg);

        // Set state to non-zero values
        for v in state.s_matrices.iter_mut() {
            *v = 1.0;
        }

        let initial_norm: f32 = state.s_matrices.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Apply decay with g < 1
        let g = 0.5_f32;
        for v in state.s_matrices.iter_mut() {
            *v *= g;
        }

        let decayed_norm: f32 = state.s_matrices.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            decayed_norm < initial_norm,
            "decayed norm ({decayed_norm}) should be less than initial ({initial_norm})"
        );

        // Check exact ratio
        let expected_ratio = g;
        let actual_ratio = decayed_norm / initial_norm;
        assert!(
            (actual_ratio - expected_ratio).abs() < 1e-5,
            "norm ratio should be {expected_ratio}, got {actual_ratio}"
        );
    }

    #[test]
    fn test_conv_state_rolling() {
        let conv_dim = 4;
        let kernel_size = 4;
        let buf_len = kernel_size - 1; // 3

        let mut conv_buffer = vec![0.0f32; conv_dim * buf_len];
        // All weights = 1.0 for simplicity
        let conv_weight = vec![1.0f32; conv_dim * kernel_size];
        let mut output = vec![0.0f32; conv_dim];

        // Step 1: input = [1, 1, 1, 1]
        let input1 = vec![1.0f32; conv_dim];
        apply_causal_conv1d(
            &input1,
            &mut conv_buffer,
            &conv_weight,
            &mut output,
            conv_dim,
            kernel_size,
        );
        // Buffer was [0,0,0], now [0,0,1]. Conv = 0*1 + 0*1 + 0*1 + 1*1 = 1
        for ch in 0..conv_dim {
            assert!(
                (output[ch] - 1.0).abs() < 1e-6,
                "step 1 ch {ch}: expected 1.0, got {}",
                output[ch]
            );
        }

        // Step 2: input = [2, 2, 2, 2]
        let input2 = vec![2.0f32; conv_dim];
        apply_causal_conv1d(
            &input2,
            &mut conv_buffer,
            &conv_weight,
            &mut output,
            conv_dim,
            kernel_size,
        );
        // Buffer was [0,0,1], now [0,1,2]. Conv = 0*1 + 0*1 + 1*1 + 2*1 = 3
        for ch in 0..conv_dim {
            assert!(
                (output[ch] - 3.0).abs() < 1e-6,
                "step 2 ch {ch}: expected 3.0, got {}",
                output[ch]
            );
        }

        // Step 3: input = [3, 3, 3, 3]
        let input3 = vec![3.0f32; conv_dim];
        apply_causal_conv1d(
            &input3,
            &mut conv_buffer,
            &conv_weight,
            &mut output,
            conv_dim,
            kernel_size,
        );
        // Buffer was [0,1,2], now [1,2,3]. Conv = 0*1 + 1*1 + 2*1 + 3*1 = 6
        for ch in 0..conv_dim {
            assert!(
                (output[ch] - 6.0).abs() < 1e-6,
                "step 3 ch {ch}: expected 6.0, got {}",
                output[ch]
            );
        }

        // Step 4: input = [4, 4, 4, 4]
        let input4 = vec![4.0f32; conv_dim];
        apply_causal_conv1d(
            &input4,
            &mut conv_buffer,
            &conv_weight,
            &mut output,
            conv_dim,
            kernel_size,
        );
        // Buffer was [1,2,3], now [2,3,4]. Conv = 1*1 + 2*1 + 3*1 + 4*1 = 10
        for ch in 0..conv_dim {
            assert!(
                (output[ch] - 10.0).abs() < 1e-6,
                "step 4 ch {ch}: expected 10.0, got {}",
                output[ch]
            );
        }
    }

    #[test]
    fn test_gated_delta_net_single_step_known_output() {
        // Create a minimal config for testing
        let cfg = Qwen35Config::qwen35_2b();
        let num_heads = cfg.linear_num_key_heads;
        let _key_dim = cfg.linear_key_head_dim;
        let _value_dim = cfg.linear_value_head_dim;
        let hidden = cfg.hidden_size;
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let kernel_size = cfg.linear_conv_kernel_dim;

        // Create zero weights (output should be zero for zero weights + zero input)
        let weights = GatedDeltaNetWeights {
            in_proj_qkv: vec![0.0; qkv_dim * hidden],
            in_proj_qkv_rows: qkv_dim,
            in_proj_qkv_cols: hidden,
            in_proj_z: vec![0.0; output_dim * hidden],
            in_proj_z_rows: output_dim,
            in_proj_z_cols: hidden,
            in_proj_b: vec![0.0; num_heads * hidden],
            in_proj_b_rows: num_heads,
            in_proj_b_cols: hidden,
            in_proj_a: vec![0.0; num_heads * hidden],
            in_proj_a_rows: num_heads,
            in_proj_a_cols: hidden,
            a_log: vec![0.0; num_heads],
            dt_bias: vec![0.0; num_heads],
            conv1d_weight: vec![0.0; qkv_dim * kernel_size],
            conv_dim: qkv_dim,
            kernel_size,
            norm_weight: vec![1.0; cfg.linear_value_head_dim],
            out_proj: vec![0.0; hidden * output_dim],
            out_proj_rows: hidden,
            out_proj_cols: output_dim,
        };

        let mut state = GatedDeltaNetState::new(&cfg);
        let mut scratch = GatedDeltaNetScratch::default();
        let input = vec![0.0f32; hidden];
        let mut output = vec![0.0f32; hidden];

        gated_delta_net_step(
            &input,
            &mut state,
            &weights,
            &cfg,
            &mut scratch,
            &mut output,
        );

        // With all-zero weights and input, output should be all zeros
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "output[{i}] should be ~0 with zero weights, got {v}"
            );
        }
    }

    /// ADR-080 C1 fail-closed (#850): GDN state persists across tokens, so a NaN Q
    /// lane must not leak past the token it appears in. Unlike an earlier version of
    /// this test (see #862 review round 1, major finding 3), the poison here is
    /// applied via a weight object used ONLY for step 0 -- steps 1 and 2 run with
    /// ordinary, never-zeroed, never-poisoned weights in BOTH the poisoned and
    /// reference runs, so this actually proves "one poisoned token, then two clean
    /// tokens," not "every token stays poisoned via a permanently-zeroed weight."
    ///
    /// Step-0 poisoned build: head 0's ENTIRE Q input-projection weight rows are
    /// zeroed, then ONE weight entry inside those rows is additionally set to NaN.
    /// Step-0 reference build: the same rows, zeroed only (no NaN). Both therefore
    /// compute head 0's Q vector as mathematically all-zero -- the reference reaches
    /// it by ordinary finite arithmetic (every weight is exactly `0.0`), the poisoned
    /// build reaches it only via the `l2_normalize_vec` fail-closed guard (one NaN
    /// lane makes `norm_sq` non-finite, so the whole vector is assigned `0.0`
    /// directly, taking the same `isfinite(norm_sq) && norm_sq > eps`-rejects branch
    /// as an exact-zero norm). If the guard's zeroing is bit-exact with true-zero
    /// weights, all three steps must be BIT-IDENTICAL between the two runs -- any
    /// deviation means either NaN leaked past step 0 (not fail-closed) or the guard
    /// takes a numerically different path than a genuinely-never-poisoned run.
    ///
    /// Uses a degenerate `linear_conv_kernel_dim = 1` config (pointwise conv1d, no
    /// rolling history window), the same disclosed test-design choice as
    /// `gdn_fused::tests::gated_delta_net_step_fused_k_poison_step1_state_isolation_through_shipping_path`:
    /// with a real multi-tap kernel, step 0's NaN raw projection also gets pushed
    /// into `state.conv_buffer`'s per-channel history and re-enters the conv1d
    /// window at step 1/2 regardless of the L2-norm guard -- a real but separate
    /// causal-conv history-carryover propagation vector that #850 does not cover.
    /// Collapsing the kernel to one tap isolates exactly the contract this PR fixes.
    ///
    /// A companion test, `gdn_fused::tests::gated_delta_net_step_fused_k_poison_step1_state_isolation_through_shipping_path`,
    /// runs the same shape of proof (poison-only-step-0, two clean steps, explicit-zero
    /// reference) for K instead of Q, through the SHIPPING `gated_delta_net_step_fused`
    /// SIMD path instead of this scalar reference -- that is the production-shaped gap
    /// #862 round 1 found (blocker 2: `simd_l2_normalize`'s scalar/AVX2/NEON backends
    /// were fail-open even though this scalar reference and `l2_normalize_vec` were
    /// already fixed). This test remains as the Q-side / scalar-reference-path proof;
    /// it does not by itself cover K or the fused/SIMD path.
    ///
    /// Mutation-sensitive: reverting the `!norm_sq.is_finite()` guard in
    /// `l2_normalize_vec` back to the plain `1.0 / (norm_sq + eps).sqrt()` computation
    /// makes the poisoned run's step-0 output/state NaN, corrupting every subsequent
    /// step once it enters `state.s_matrices` -- so the bit-identical assertions below
    /// fail immediately.
    #[test]
    fn test_l2_normalize_vec_state_isolation_poisoned_token_matches_never_poisoned_reference() {
        let mut cfg = Qwen35Config::qwen35_2b();
        cfg.linear_conv_kernel_dim = 1;
        let num_heads = cfg.linear_num_key_heads;
        let value_heads = cfg.linear_num_value_heads();
        assert_eq!(
            num_heads, value_heads,
            "this test assumes qwen35_2b's ratio=1 (num_key_heads == num_value_heads) \
             so in_proj_b/in_proj_a can be sized by num_heads alone, matching the \
             existing test_gated_delta_net_single_step_known_output fixture pattern"
        );
        let key_dim = cfg.linear_key_head_dim;
        let value_dim = cfg.linear_value_head_dim;
        let hidden = cfg.hidden_size;
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let kernel_size = cfg.linear_conv_kernel_dim;

        // Deterministic pseudo-random fill (no external `rand` dependency), matching
        // the small-magnitude range other GDN tests in this file use.
        fn make_lcg(seed: u64) -> impl FnMut() -> f32 {
            let mut state = seed;
            move || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (((state >> 32) as u32) as f32 / u32::MAX as f32 - 0.5) * 0.2
            }
        }

        let make_ordinary_weights = |seed: u64| -> GatedDeltaNetWeights {
            let mut rng = make_lcg(seed);
            GatedDeltaNetWeights {
                in_proj_qkv: (0..qkv_dim * hidden).map(|_| rng()).collect(),
                in_proj_qkv_rows: qkv_dim,
                in_proj_qkv_cols: hidden,
                in_proj_z: (0..output_dim * hidden).map(|_| rng()).collect(),
                in_proj_z_rows: output_dim,
                in_proj_z_cols: hidden,
                in_proj_b: (0..num_heads * hidden).map(|_| rng()).collect(),
                in_proj_b_rows: num_heads,
                in_proj_b_cols: hidden,
                in_proj_a: (0..num_heads * hidden).map(|_| rng()).collect(),
                in_proj_a_rows: num_heads,
                in_proj_a_cols: hidden,
                a_log: vec![-1.0; num_heads],
                dt_bias: vec![0.0; num_heads],
                conv1d_weight: vec![1.0 / kernel_size as f32; qkv_dim * kernel_size],
                conv_dim: qkv_dim,
                kernel_size,
                norm_weight: vec![1.0; value_dim],
                out_proj: (0..hidden * output_dim).map(|_| rng()).collect(),
                out_proj_rows: hidden,
                out_proj_cols: output_dim,
            }
        };

        // Step-0-only weights: same seed for poisoned/reference so every
        // non-corrupted weight is bit-identical between the two builds.
        let build_step0_weights = |poison: bool, seed: u64| -> GatedDeltaNetWeights {
            let mut w = make_ordinary_weights(seed);
            // Zero ALL of head 0's Q weight rows (rows [0, key_dim) of the Q block,
            // which occupies rows [0, num_heads*key_dim); head 0 maps to k_head 0).
            for row in w.in_proj_qkv.iter_mut().take(key_dim * hidden) {
                *row = 0.0;
            }
            if poison {
                // One NaN lane inside head 0's (all-zero) Q rows -- this specific Q
                // lane, and via the fail-closed guard the WHOLE head-0 Q vector, goes
                // non-finite pre-guard, exactly like the reference's true-zero-weight
                // Q vector post-guard.
                w.in_proj_qkv[0] = f32::NAN;
            }
            w
        };
        let step0_poisoned = build_step0_weights(true, 0xA5A5_1234_F00D_BEEF);
        let step0_reference = build_step0_weights(false, 0xA5A5_1234_F00D_BEEF);
        // Steps 1-2: identical ordinary weights (no zeroing, no NaN) in BOTH runs.
        let clean_weights = make_ordinary_weights(0xC0FF_EE00_1357_9BDF);

        let mut token_rng = make_lcg(0x0102_0304_0506_0708);
        let tokens: Vec<Vec<f32>> = (0..3)
            .map(|_| (0..hidden).map(|_| token_rng()).collect())
            .collect();

        let run = |step0_weights: &GatedDeltaNetWeights| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
            let mut state = GatedDeltaNetState::new(&cfg);
            let mut scratch = GatedDeltaNetScratch::default();
            let mut outputs = Vec::with_capacity(3);
            let mut state_snapshots = Vec::with_capacity(3);
            let step_weights = [step0_weights, &clean_weights, &clean_weights];
            for (tok, weights) in tokens.iter().zip(step_weights.iter()) {
                let mut output = vec![0.0f32; hidden];
                gated_delta_net_step(tok, &mut state, weights, &cfg, &mut scratch, &mut output);
                outputs.push(output);
                state_snapshots.push(state.s_matrices.clone());
            }
            (outputs, state_snapshots)
        };

        let (poisoned_outputs, poisoned_states) = run(&step0_poisoned);
        let (reference_outputs, reference_states) = run(&step0_reference);

        for step in 0..3 {
            let nan_count = poisoned_outputs[step]
                .iter()
                .filter(|v| !v.is_finite())
                .count();
            assert_eq!(
                nan_count, 0,
                "step {step}: poisoned run must fail closed (finite output), found \
                 {nan_count} non-finite element(s) -- a NaN Q lane leaked past the \
                 l2_normalize_vec guard"
            );
            let state_nan_count = poisoned_states[step]
                .iter()
                .filter(|v| !v.is_finite())
                .count();
            assert_eq!(
                state_nan_count, 0,
                "step {step}: poisoned run's recurrent state must stay finite \
                 (GDN state persists across tokens; a non-finite S contaminates \
                 every subsequent token), found {state_nan_count} non-finite element(s)"
            );

            assert_eq!(
                poisoned_outputs[step], reference_outputs[step],
                "step {step}: poisoned-then-fail-closed output must be BIT-IDENTICAL \
                 to the never-poisoned (true-zero-weight) reference output"
            );
            assert_eq!(
                poisoned_states[step], reference_states[step],
                "step {step}: poisoned-then-fail-closed recurrent state must be \
                 BIT-IDENTICAL to the never-poisoned (true-zero-weight) reference state"
            );
        }
    }

    #[test]
    #[allow(clippy::erasing_op)]
    fn test_k_head_mapping_ratio_3() {
        let num_key_heads: usize = 16;
        let num_value_heads: usize = 48;
        let ratio = num_value_heads / num_key_heads;
        assert_eq!(ratio, 3);

        assert_eq!(0 / ratio, 0);
        assert_eq!(1 / ratio, 0);
        assert_eq!(2 / ratio, 0);

        assert_eq!(3 / ratio, 1);
        assert_eq!(4 / ratio, 1);
        assert_eq!(5 / ratio, 1);

        assert_eq!(24 / ratio, 8);
        assert_eq!(25 / ratio, 8);
        assert_eq!(26 / ratio, 8);

        assert_eq!(45 / ratio, 15);
        assert_eq!(46 / ratio, 15);
        assert_eq!(47 / ratio, 15);

        for v_head in 0..num_value_heads {
            let k_head = v_head / ratio;
            assert!(
                k_head < num_key_heads,
                "v_head {v_head} mapped to out-of-range k_head {k_head}"
            );
        }
    }

    #[test]
    fn test_k_head_mapping_identity() {
        let ratio: usize = 16 / 16;
        assert_eq!(ratio, 1);
        for h in 0..16_usize {
            assert_eq!(h / ratio, h, "identity mapping broken at h={h}");
        }
    }

    #[test]
    fn test_s_matrix_allocation_asymmetric() {
        let mut cfg = Qwen35Config::qwen35_2b();
        cfg.linear_num_value_heads = Some(48);
        let state = GatedDeltaNetState::new(&cfg);
        let expected = 48 * 128 * 128;
        assert_eq!(
            state.s_matrices.len(),
            expected,
            "asymmetric (16k/48v): expected {expected}, got {}",
            state.s_matrices.len()
        );
    }

    #[test]
    fn test_s_matrix_allocation_symmetric() {
        let cfg = Qwen35Config::qwen35_2b();
        let state = GatedDeltaNetState::new(&cfg);
        let expected = 16 * 128 * 128;
        assert_eq!(
            state.s_matrices.len(),
            expected,
            "symmetric (16k/16v): expected {expected}, got {}",
            state.s_matrices.len()
        );
    }

    #[test]
    fn test_output_dim_asymmetric() {
        let mut cfg = Qwen35Config::qwen35_2b();
        cfg.linear_num_value_heads = Some(48);
        assert_eq!(cfg.linear_output_dim(), 48 * 128);
    }

    #[test]
    fn test_output_dim_symmetric() {
        let cfg = Qwen35Config::qwen35_2b();
        assert_eq!(cfg.linear_output_dim(), 16 * 128);
    }

    #[test]
    fn snapshot_roundtrip_preserves_state() {
        let cfg = Qwen35Config::qwen35_2b();
        let mut state = GatedDeltaNetState::new(&cfg);
        for (i, v) in state.s_matrices.iter_mut().enumerate() {
            *v = (i as f32) * 0.001;
        }
        for (i, v) in state.conv_buffer.iter_mut().enumerate() {
            *v = (i as f32) * 0.01;
        }
        let snap = state.snapshot();
        state.reset();
        assert!(state.s_matrices.iter().all(|v| *v == 0.0));
        state.restore_from(&snap);
        assert_eq!(state.s_matrices[7], 7.0 * 0.001);
        assert_eq!(state.conv_buffer[3], 3.0 * 0.01);
    }

    #[test]
    fn snapshot_independent_after_state_mutation() {
        let cfg = Qwen35Config::qwen35_2b();
        let mut state = GatedDeltaNetState::new(&cfg);
        state.s_matrices[0] = 1.0;
        let snap = state.snapshot();
        state.s_matrices[0] = 999.0;
        assert_eq!(snap.0[0], 1.0, "snapshot must own its buffers");
    }
}

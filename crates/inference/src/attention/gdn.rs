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
#[derive(Debug)]
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

    /// Learnable log-decay: [num_heads]
    pub a_log: Vec<f32>,

    /// Learnable time-step bias: [num_heads]
    pub dt_bias: Vec<f32>,

    /// Depthwise conv1d weights: [qkv_dim, 1, kernel_size] stored as [qkv_dim, kernel_size]
    pub conv1d_weight: Vec<f32>,
    pub conv_dim: usize,
    pub kernel_size: usize,

    /// Output gated RMSNorm gamma: [output_dim]
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

    /// Access S matrix for head h as a mutable slice [key_dim, value_dim].
    #[inline]
    fn s_matrix_mut(&mut self, h: usize) -> &mut [f32] {
        let size = self.key_dim * self.value_dim;
        &mut self.s_matrices[h * size..(h + 1) * size]
    }
}

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
/// `input`: hidden state [hidden_size]
/// `state`: mutable recurrent state for this layer
/// `weights`: layer weights
/// `cfg`: model config
/// `scratch`: reusable scratch buffers
/// `output`: output buffer [hidden_size], written in-place
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
    let key_dim = cfg.linear_key_head_dim;
    let value_dim = cfg.linear_value_head_dim;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let kernel_size = cfg.linear_conv_kernel_dim;

    scratch.ensure_capacity(qkv_dim, output_dim, num_heads, hidden);

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
        &mut scratch.beta_proj[..num_heads],
        1,
        hidden,
        num_heads,
    );

    matmul_bt(
        input,
        &weights.in_proj_a,
        &mut scratch.alpha_proj[..num_heads],
        1,
        hidden,
        num_heads,
    );

    // Apply sigmoid to beta
    for b in &mut scratch.beta_proj[..num_heads] {
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

    // 4-7. Process each head
    for h in 0..num_heads {
        let q_start = h * key_dim;
        let k_start = q_total + h * key_dim;
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
    for h in 0..num_heads {
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
#[inline]
pub fn l2_normalize_vec(x: &mut [f32]) {
    let eps = 1e-6f32;
    let norm_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_norm = 1.0 / (norm_sq + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv_norm;
    }
}

/// Compute decay gate: g = exp(-exp(a_log) * softplus(alpha + dt_bias))
#[inline]
fn compute_decay_gate(a_log: f32, alpha: f32, dt_bias: f32) -> f32 {
    let a = a_log.exp();
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
/// `x`: input [dim]
/// `z`: output gate [dim] (NOT sigmoided — raw projection output used as gate)
/// `gamma`: learnable scale [dim]
/// `out`: output buffer [dim]
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

    #[test]
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
}

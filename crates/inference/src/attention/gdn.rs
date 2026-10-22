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
    // `alpha + dt_bias` (softplus underflows to 0.0), the product is `inf * 0.0` = NaN and
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
    } else {
        // `(1.0 + x.exp()).ln()` rounds to 0 below x ≈ -16.6 in f32, where the true value is
        // ≈ exp(x); `ln_1p` keeps that tail, so the decay gate still decays there.
        x.exp().ln_1p()
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
mod tests;

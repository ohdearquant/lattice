//! Fused GatedDeltaNet: SIMD-accelerated alternative to `gated_delta_net`.
//!
//! This module provides `gated_delta_net_step_fused`, an optimized drop-in
//! alternative to `gated_delta_net_step` that fuses conv1d + SiLU activation,
//! uses SIMD (AVX2/NEON) for L2 normalization, matrix-vector products, decay +
//! rank-1 updates, and gated RMS norm. The two implementations are numerically
//! equivalent within floating-point tolerance; tests verify parity.
//!
//! Both modules coexist — callers choose which to use.

use crate::attention::gdn::{GatedDeltaNetState, GatedDeltaNetWeights, sigmoid, softplus};
use crate::forward::cpu::{matmul_bt, validate_gemm_nn};
use crate::model::qwen35_config::Qwen35Config;

/// **Unstable**: scratch buffers for the SIMD-fused GatedDeltaNet step; buffer layout evolving.
///
/// Scratch buffers for the fused GatedDeltaNet kernel.
///
/// Compared to `GatedDeltaNetScratch`, this adds per-head temporaries
/// (`q_head`, `k_head`, `kv_mem`, `delta`) to avoid per-step allocations
/// in the inner loop, and removes `conv_input` / `final_out` (the fused
/// conv1d+SiLU path does not need a separate conv_input copy).
#[derive(Debug, Default)]
pub struct GatedDeltaNetFusedScratch {
    pub qkv_proj: Vec<f32>,
    pub z_proj: Vec<f32>,
    pub beta_proj: Vec<f32>,
    pub alpha_proj: Vec<f32>,
    pub conv_output: Vec<f32>,
    pub output_heads: Vec<f32>,
    pub gated_norm_buf: Vec<f32>,
    pub q_head: Vec<f32>,
    pub k_head: Vec<f32>,
    pub kv_mem: Vec<f32>,
    pub delta: Vec<f32>,
    /// Packed [k; q] for two-pass BLAS recurrence, shape [2, key_dim].
    pub kq_pack: Vec<f32>,
    /// Result of [k; q] @ S, shape [2, value_dim]. Row 0 = S^T k, row 1 = S^T q.
    pub proj2: Vec<f32>,
}

impl GatedDeltaNetFusedScratch {
    /// **Unstable**: grow fused scratch buffers to the required dimensions.
    #[inline]
    pub fn ensure_capacity(
        &mut self,
        qkv_dim: usize,
        output_dim: usize,
        num_heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) {
        resize_if_needed(&mut self.qkv_proj, qkv_dim);
        resize_if_needed(&mut self.z_proj, output_dim);
        resize_if_needed(&mut self.beta_proj, num_heads);
        resize_if_needed(&mut self.alpha_proj, num_heads);
        resize_if_needed(&mut self.conv_output, qkv_dim);
        resize_if_needed(&mut self.output_heads, output_dim);
        resize_if_needed(&mut self.gated_norm_buf, output_dim);
        resize_if_needed(&mut self.q_head, key_dim);
        resize_if_needed(&mut self.k_head, key_dim);
        resize_if_needed(&mut self.kv_mem, value_dim);
        resize_if_needed(&mut self.delta, value_dim);
        resize_if_needed(&mut self.kq_pack, 2 * key_dim);
        resize_if_needed(&mut self.proj2, 2 * value_dim);
    }
}

#[inline]
fn resize_if_needed(buf: &mut Vec<f32>, needed: usize) {
    if buf.len() < needed {
        buf.resize(needed, 0.0);
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// ADR-080 C1 fail-closed (#850): a non-finite input lane makes `norm_sq` non-finite too
/// (squaring any non-finite value yields +inf/NaN, and summing any non-finite term keeps
/// the sum non-finite), so checking `norm_sq.is_finite()` alone detects the whole-vector
/// invalid case. The whole vector is assigned the literal `0.0` directly rather than
/// multiplied through a zeroed reciprocal (`NaN * 0.0 == NaN` under IEEE-754). Mirrors
/// `attention::gdn::l2_normalize_vec` (the scalar reference helper).
#[inline]
fn scalar_l2_normalize(x: &mut [f32]) {
    let norm_sq: f32 = x.iter().map(|v| v * v).sum();
    if !norm_sq.is_finite() {
        for v in x.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let inv_norm = 1.0 / (norm_sq + 1e-6).sqrt();
    for v in x.iter_mut() {
        *v *= inv_norm;
    }
}

#[inline]
fn scalar_matvec_transpose(
    s: &[f32],
    k: &[f32],
    kv_mem: &mut [f32],
    key_dim: usize,
    value_dim: usize,
) {
    kv_mem[..value_dim].fill(0.0);
    for i in 0..key_dim {
        let ki = k[i];
        let row = &s[i * value_dim..(i + 1) * value_dim];
        for j in 0..value_dim {
            kv_mem[j] += row[j] * ki;
        }
    }
}

#[inline]
fn scalar_decay_and_rank1_update(
    s: &mut [f32],
    k: &[f32],
    delta: &[f32],
    g: f32,
    key_dim: usize,
    value_dim: usize,
) {
    for i in 0..key_dim {
        let ki = k[i];
        let row = &mut s[i * value_dim..(i + 1) * value_dim];
        for j in 0..value_dim {
            row[j] = row[j] * g + ki * delta[j];
        }
    }
}

#[inline]
fn scalar_gated_rms_norm(x: &[f32], z: &[f32], gamma: &[f32], out: &mut [f32], eps: f32) {
    let dim = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        out[i] = (x[i] * inv_rms) * gamma[i] * silu(z[i]);
    }
}

#[inline]
fn compute_decay_gate(a_log: f32, alpha: f32, dt_bias: f32) -> f32 {
    // Clamp the decay rate to finite: `a_log.exp()` overflows to +inf for a_log > ~88, and
    // `inf * softplus(very_negative)=0.0` is NaN that poisons the recurrent state. Mirrors
    // the scalar `gdn::compute_decay_gate` guard (kept in lockstep for scalar/fused parity).
    let a = a_log.exp().min(f32::MAX);
    let sp = softplus(alpha + dt_bias);
    (-a * sp).exp()
}

// ---------------------------------------------------------------------------
// Fused conv1d + SiLU
// ---------------------------------------------------------------------------

/// **Unstable**: fused conv1d + SiLU kernel; SIMD path and API may evolve.
#[inline]
pub fn conv1d_silu_fused(
    new_input: &[f32],
    conv_buffer: &mut [f32],
    conv_weight: &[f32],
    output: &mut [f32],
    conv_dim: usize,
    kernel_size: usize,
) {
    let buf_len = kernel_size.saturating_sub(1);
    debug_assert!(new_input.len() >= conv_dim);
    debug_assert!(output.len() >= conv_dim);
    debug_assert!(conv_buffer.len() >= conv_dim * buf_len);
    debug_assert!(conv_weight.len() >= conv_dim * kernel_size);

    for ch in 0..conv_dim {
        let w_offset = ch * kernel_size;
        let buf_offset = ch * buf_len;
        let row = &mut conv_buffer[buf_offset..buf_offset + buf_len];

        let mut sum = 0.0f32;
        for t in 0..buf_len {
            sum += row[t] * conv_weight[w_offset + t];
        }

        let x = new_input[ch];
        sum += x * conv_weight[w_offset + buf_len];
        output[ch] = silu(sum);

        if buf_len > 1 {
            row.copy_within(1..buf_len, 0);
        }
        if buf_len > 0 {
            row[buf_len - 1] = x;
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD dispatch wrappers
// ---------------------------------------------------------------------------

/// **Unstable**: SIMD-dispatched L2 normalisation (AVX2/NEON/scalar fallback).
#[inline]
pub fn simd_l2_normalize(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: The call is guarded by runtime feature detection for AVX2 + FMA.
            unsafe {
                simd_l2_normalize_avx2(x);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: The call is guarded by runtime feature detection for NEON.
            unsafe {
                simd_l2_normalize_neon(x);
            }
            return;
        }
    }

    scalar_l2_normalize(x);
}

/// **Unstable**: SIMD-dispatched transposed matrix-vector product (AVX2/NEON/scalar fallback).
#[inline]
pub fn simd_matvec_transpose(
    s: &[f32],
    k: &[f32],
    kv_mem: &mut [f32],
    key_dim: usize,
    value_dim: usize,
) {
    // Release-active, overflow-first: `kv_mem = k @ s` (m=1, k=key_dim, n=value_dim) —
    // these are the soundness preconditions for the unsafe SIMD kernels below (which load
    // `key_dim`/`value_dim` lanes via raw pointers). A plain `key_dim * value_dim` multiply
    // (as this used before ADR-080 C4) can wrap in release and make the length check pass
    // spuriously on a malformed shape; `validate_gemm_nn` checks overflow first.
    validate_gemm_nn(
        k.len(),
        s.len(),
        kv_mem.len(),
        1,
        key_dim,
        value_dim,
        "gdn_matvec_transpose",
    );
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: The call is guarded by runtime feature detection for AVX2 + FMA.
            unsafe {
                simd_matvec_transpose_avx2(s, k, kv_mem, key_dim, value_dim);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: The call is guarded by runtime feature detection for NEON.
            unsafe {
                simd_matvec_transpose_neon(s, k, kv_mem, key_dim, value_dim);
            }
            return;
        }
    }

    scalar_matvec_transpose(s, k, kv_mem, key_dim, value_dim);
}

/// **Unstable**: SIMD-dispatched decay + rank-1 state update (AVX2/NEON/scalar fallback).
#[inline]
pub fn simd_decay_and_rank1_update(
    s: &mut [f32],
    k: &[f32],
    delta: &[f32],
    g: f32,
    key_dim: usize,
    value_dim: usize,
) {
    // Release-active, overflow-first: `s` is mutated in place at the same [key_dim,
    // value_dim] footprint as an `A(1,key_dim) @ B(key_dim,value_dim)` operand pair (`k`
    // plays the role of A, `delta` the per-column update); reuse `validate_gemm_nn`'s
    // overflow-first `key_dim*value_dim` check rather than the raw multiply this used
    // before ADR-080 C4 (see `simd_matvec_transpose` for the overflow rationale).
    validate_gemm_nn(
        k.len(),
        s.len(),
        s.len(),
        1,
        key_dim,
        value_dim,
        "gdn_decay_and_rank1_update",
    );
    assert!(
        delta.len() >= value_dim,
        "gdn_decay_and_rank1_update: delta too short for value_dim"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: The call is guarded by runtime feature detection for AVX2 + FMA.
            unsafe {
                simd_decay_and_rank1_update_avx2(s, k, delta, g, key_dim, value_dim);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: The call is guarded by runtime feature detection for NEON.
            unsafe {
                simd_decay_and_rank1_update_neon(s, k, delta, g, key_dim, value_dim);
            }
            return;
        }
    }

    scalar_decay_and_rank1_update(s, k, delta, g, key_dim, value_dim);
}

/// **Unstable**: SIMD-dispatched gated RMSNorm (AVX2/NEON/scalar fallback).
#[inline]
pub fn simd_gated_rms_norm(x: &[f32], z: &[f32], gamma: &[f32], out: &mut [f32], eps: f32) {
    // Release-active: soundness preconditions for the unsafe SIMD kernels below.
    // See `simd_matvec_transpose` for the rationale (debug_assert is removed in
    // release, leaving the safe wrapper able to drive an OOB SIMD load).
    assert_eq!(z.len(), x.len());
    assert_eq!(gamma.len(), x.len());
    assert!(out.len() >= x.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: The call is guarded by runtime feature detection for AVX2 + FMA.
            unsafe {
                simd_gated_rms_norm_avx2(x, z, gamma, out, eps);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: The call is guarded by runtime feature detection for NEON.
            unsafe {
                simd_gated_rms_norm_neon(x, z, gamma, out, eps);
            }
            return;
        }
    }

    scalar_gated_rms_norm(x, z, gamma, out, eps);
}

// ---------------------------------------------------------------------------
// Main fused entry point
// ---------------------------------------------------------------------------

/// **Unstable**: fused + SIMD GatedDeltaNet step; primary hot path, under active optimization.
///
/// Process a single token through the GatedDeltaNet layer (fused + SIMD path).
///
/// Numerically equivalent to `gated_delta_net_step` within f32 tolerance.
/// Fusions: conv1d+SiLU, decay+rank1 update. SIMD: L2 norm, matvec, RMS norm.
///
/// `input`: hidden state `[hidden_size]`
/// `state`: mutable recurrent state for this layer
/// `weights`: layer weights
/// `cfg`: model config
/// `scratch`: reusable fused scratch buffers
/// `output`: output buffer `[hidden_size]`, written in-place
#[inline]
pub fn gated_delta_net_step_fused(
    input: &[f32],
    state: &mut GatedDeltaNetState,
    weights: &GatedDeltaNetWeights,
    cfg: &Qwen35Config,
    scratch: &mut GatedDeltaNetFusedScratch,
    output: &mut [f32],
    lora: &dyn crate::lora_hook::LoraHook,
    layer_idx: usize,
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

    debug_assert_eq!(
        value_heads % num_heads,
        0,
        "value_heads must be divisible by key_heads"
    );
    debug_assert!(input.len() >= hidden);
    debug_assert!(output.len() >= hidden);

    scratch.ensure_capacity(qkv_dim, output_dim, value_heads, key_dim, value_dim);

    // 1. Projections (with LoRA hooks for fine-tuned adapters)
    matmul_bt(
        input,
        &weights.in_proj_qkv,
        &mut scratch.qkv_proj[..qkv_dim],
        1,
        hidden,
        qkv_dim,
    );
    lora.apply(
        layer_idx,
        "in_proj_qkv",
        input,
        &mut scratch.qkv_proj[..qkv_dim],
    );

    matmul_bt(
        input,
        &weights.in_proj_z,
        &mut scratch.z_proj[..output_dim],
        1,
        hidden,
        output_dim,
    );
    lora.apply(
        layer_idx,
        "in_proj_z",
        input,
        &mut scratch.z_proj[..output_dim],
    );

    matmul_bt(
        input,
        &weights.in_proj_b,
        &mut scratch.beta_proj[..value_heads],
        1,
        hidden,
        value_heads,
    );
    lora.apply(
        layer_idx,
        "in_proj_b",
        input,
        &mut scratch.beta_proj[..value_heads],
    );

    matmul_bt(
        input,
        &weights.in_proj_a,
        &mut scratch.alpha_proj[..value_heads],
        1,
        hidden,
        value_heads,
    );
    lora.apply(
        layer_idx,
        "in_proj_a",
        input,
        &mut scratch.alpha_proj[..value_heads],
    );

    // sigmoid(beta)
    for b in &mut scratch.beta_proj[..value_heads] {
        *b = sigmoid(*b);
    }

    // 2. Fused conv1d + SiLU
    conv1d_silu_fused(
        &scratch.qkv_proj[..qkv_dim],
        &mut state.conv_buffer,
        &weights.conv1d_weight,
        &mut scratch.conv_output[..qkv_dim],
        qkv_dim,
        kernel_size,
    );

    // 3-7. Per-head processing
    let q_total = num_heads * key_dim;
    let k_total = num_heads * key_dim;
    let v_offset = q_total + k_total;
    let scale = 1.0 / (key_dim as f32).sqrt();

    for h in 0..value_heads {
        let k_head = h / ratio;
        let q_start = k_head * key_dim;
        let k_start = q_total + k_head * key_dim;
        let v_start = v_offset + h * value_dim;

        scratch.q_head[..key_dim].copy_from_slice(&scratch.conv_output[q_start..q_start + key_dim]);
        scratch.k_head[..key_dim].copy_from_slice(&scratch.conv_output[k_start..k_start + key_dim]);
        let v = &scratch.conv_output[v_start..v_start + value_dim];

        // L2-normalize Q and K (SIMD-accelerated)
        simd_l2_normalize(&mut scratch.q_head[..key_dim]);
        simd_l2_normalize(&mut scratch.k_head[..key_dim]);

        // Decay gate (indexed per value head)
        let g = compute_decay_gate(weights.a_log[h], scratch.alpha_proj[h], weights.dt_bias[h]);
        let s_offset = h * key_dim * value_dim;
        let s = &mut state.s_matrices[s_offset..s_offset + key_dim * value_dim];

        // Retrieve: kv_mem = S^T @ k (SIMD-accelerated)
        simd_matvec_transpose(
            s,
            &scratch.k_head[..key_dim],
            &mut scratch.kv_mem[..value_dim],
            key_dim,
            value_dim,
        );

        // Delta: (v - g * kv_mem) * beta
        // Using (v - g * kv_mem) preserves the reference semantics where
        // retrieval happens from the decayed state g * S_old.
        let beta_h = scratch.beta_proj[h];
        for j in 0..value_dim {
            scratch.delta[j] = (v[j] - scratch.kv_mem[j] * g) * beta_h;
        }

        // Fused decay + rank-1 update: S = g*S + outer(k, delta) (SIMD-accelerated)
        simd_decay_and_rank1_update(
            s,
            &scratch.k_head[..key_dim],
            &scratch.delta[..value_dim],
            g,
            key_dim,
            value_dim,
        );

        // Output: o = S^T @ q / sqrt(key_dim) (SIMD-accelerated)
        let out_start = h * value_dim;
        let out_head = &mut scratch.output_heads[out_start..out_start + value_dim];
        simd_matvec_transpose(s, &scratch.q_head[..key_dim], out_head, key_dim, value_dim);
        for val in out_head.iter_mut() {
            *val *= scale;
        }
    }

    // 8. Gated RMSNorm + output projection
    // norm_weight is [value_dim] per-head, applied to each head independently.
    // Fail fast if weight dimension doesn't match: no silent fallback.
    let gamma = &weights.norm_weight[..value_dim];
    debug_assert_eq!(gamma.len(), value_dim);

    for h in 0..value_heads {
        let start = h * value_dim;
        let end = start + value_dim;
        simd_gated_rms_norm(
            &scratch.output_heads[start..end],
            &scratch.z_proj[start..end],
            gamma,
            &mut scratch.gated_norm_buf[start..end],
            cfg.rms_norm_eps,
        );
    }

    // Output projection
    matmul_bt(
        &scratch.gated_norm_buf[..output_dim],
        &weights.out_proj,
        &mut output[..hidden],
        1,
        output_dim,
        hidden,
    );
    lora.apply(
        layer_idx,
        "out_proj",
        &scratch.gated_norm_buf[..output_dim],
        &mut output[..hidden],
    );
}

// ---------------------------------------------------------------------------
// AVX2 SIMD kernels
// ---------------------------------------------------------------------------

/// ADR-080 C1 fail-closed (#850): same whole-vector-zero-by-assignment contract as
/// `scalar_l2_normalize` (see its doc comment for the IEEE-754 rationale).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn simd_l2_normalize_avx2(x: &mut [f32]) {
    use core::arch::x86_64::*;

    let len = x.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
        // SAFETY: i..i+8 is in-bounds and AVX2 is enabled for this function.
        let v = unsafe { _mm256_loadu_ps(x.as_ptr().add(i)) };
        acc = _mm256_fmadd_ps(v, v, acc);
        i += 8;
    }

    let mut lanes = [0.0f32; 8];
    // SAFETY: lanes has space for 8 f32 values.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut norm_sq: f32 = lanes.iter().sum();
    while i < len {
        norm_sq += x[i] * x[i];
        i += 1;
    }

    if !norm_sq.is_finite() {
        for v in x.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let inv_norm = 1.0 / (norm_sq + 1e-6).sqrt();
    let inv = _mm256_set1_ps(inv_norm);
    let mut i = 0usize;
    while i + 8 <= len {
        // SAFETY: i..i+8 is in-bounds and AVX2 is enabled for this function.
        let v = unsafe { _mm256_loadu_ps(x.as_ptr().add(i)) };
        let y = _mm256_mul_ps(v, inv);
        // SAFETY: i..i+8 is in-bounds and AVX2 is enabled for this function.
        unsafe { _mm256_storeu_ps(x.as_mut_ptr().add(i), y) };
        i += 8;
    }
    while i < len {
        x[i] *= inv_norm;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn simd_matvec_transpose_avx2(
    s: &[f32],
    k: &[f32],
    kv_mem: &mut [f32],
    key_dim: usize,
    value_dim: usize,
) {
    use core::arch::x86_64::*;

    kv_mem[..value_dim].fill(0.0);
    for i in 0..key_dim {
        let row_ptr = s.as_ptr().wrapping_add(i * value_dim);
        let ki = _mm256_set1_ps(k[i]);
        let mut j = 0usize;
        while j + 8 <= value_dim {
            // SAFETY: j..j+8 is in-bounds for both row and kv_mem, and AVX2 is enabled.
            let acc = unsafe { _mm256_loadu_ps(kv_mem.as_ptr().add(j)) };
            let row = unsafe { _mm256_loadu_ps(row_ptr.add(j)) };
            let updated = _mm256_fmadd_ps(row, ki, acc);
            // SAFETY: j..j+8 is in-bounds for kv_mem and AVX2 is enabled.
            unsafe { _mm256_storeu_ps(kv_mem.as_mut_ptr().add(j), updated) };
            j += 8;
        }
        while j < value_dim {
            kv_mem[j] += s[i * value_dim + j] * k[i];
            j += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn simd_decay_and_rank1_update_avx2(
    s: &mut [f32],
    k: &[f32],
    delta: &[f32],
    g: f32,
    key_dim: usize,
    value_dim: usize,
) {
    use core::arch::x86_64::*;

    let g_vec = _mm256_set1_ps(g);
    for i in 0..key_dim {
        let row_ptr = s.as_mut_ptr().wrapping_add(i * value_dim);
        let ki = _mm256_set1_ps(k[i]);
        let mut j = 0usize;
        while j + 8 <= value_dim {
            // SAFETY: j..j+8 is in-bounds for row and delta, and AVX2 is enabled.
            let row = unsafe { _mm256_loadu_ps(row_ptr.add(j)) };
            let delta_vec = unsafe { _mm256_loadu_ps(delta.as_ptr().add(j)) };
            let updated = _mm256_fmadd_ps(ki, delta_vec, _mm256_mul_ps(g_vec, row));
            // SAFETY: j..j+8 is in-bounds for row and AVX2 is enabled.
            unsafe { _mm256_storeu_ps(row_ptr.add(j), updated) };
            j += 8;
        }
        while j < value_dim {
            let idx = i * value_dim + j;
            s[idx] = s[idx] * g + k[i] * delta[j];
            j += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn simd_gated_rms_norm_avx2(x: &[f32], z: &[f32], gamma: &[f32], out: &mut [f32], eps: f32) {
    use core::arch::x86_64::*;

    let len = x.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
        // SAFETY: i..i+8 is in-bounds and AVX2 is enabled.
        let xv = unsafe { _mm256_loadu_ps(x.as_ptr().add(i)) };
        acc = _mm256_fmadd_ps(xv, xv, acc);
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    // SAFETY: lanes has space for 8 f32 values.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum_sq: f32 = lanes.iter().sum();
    while i < len {
        sum_sq += x[i] * x[i];
        i += 1;
    }

    let inv_rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let inv = _mm256_set1_ps(inv_rms);
    let mut gate_tmp = [0.0f32; 8];
    let mut i = 0usize;
    while i + 8 <= len {
        for lane in 0..8 {
            gate_tmp[lane] = silu(z[i + lane]);
        }
        // SAFETY: i..i+8 is in-bounds and AVX2 is enabled.
        let xv = unsafe { _mm256_loadu_ps(x.as_ptr().add(i)) };
        let gv = unsafe { _mm256_loadu_ps(gamma.as_ptr().add(i)) };
        let zv = unsafe { _mm256_loadu_ps(gate_tmp.as_ptr()) };
        let y = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(xv, inv), gv), zv);
        // SAFETY: i..i+8 is in-bounds and AVX2 is enabled.
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), y) };
        i += 8;
    }
    while i < len {
        out[i] = (x[i] * inv_rms) * gamma[i] * silu(z[i]);
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// NEON SIMD kernels
// ---------------------------------------------------------------------------

/// ADR-080 C1 fail-closed (#850): same whole-vector-zero-by-assignment contract as
/// `scalar_l2_normalize` (see its doc comment for the IEEE-754 rationale).
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn simd_l2_normalize_neon(x: &mut [f32]) {
    use core::arch::aarch64::*;

    let len = x.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= len {
        // SAFETY: i..i+4 is in-bounds and NEON is enabled for this function.
        let v = unsafe { vld1q_f32(x.as_ptr().add(i)) };
        acc = vfmaq_f32(acc, v, v);
        i += 4;
    }

    let mut norm_sq = vaddvq_f32(acc);
    while i < len {
        norm_sq += x[i] * x[i];
        i += 1;
    }

    if !norm_sq.is_finite() {
        for v in x.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let inv_norm = 1.0 / (norm_sq + 1e-6).sqrt();
    let inv = vdupq_n_f32(inv_norm);
    let mut i = 0usize;
    while i + 4 <= len {
        // SAFETY: i..i+4 is in-bounds and NEON is enabled for this function.
        let v = unsafe { vld1q_f32(x.as_ptr().add(i)) };
        let y = vmulq_f32(v, inv);
        // SAFETY: i..i+4 is in-bounds and NEON is enabled for this function.
        unsafe { vst1q_f32(x.as_mut_ptr().add(i), y) };
        i += 4;
    }
    while i < len {
        x[i] *= inv_norm;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn simd_matvec_transpose_neon(
    s: &[f32],
    k: &[f32],
    kv_mem: &mut [f32],
    key_dim: usize,
    value_dim: usize,
) {
    use core::arch::aarch64::*;

    kv_mem[..value_dim].fill(0.0);
    for i in 0..key_dim {
        let row_ptr = s.as_ptr().wrapping_add(i * value_dim);
        let ki = vdupq_n_f32(k[i]);
        let mut j = 0usize;
        while j + 4 <= value_dim {
            // SAFETY: j..j+4 is in-bounds for both row and kv_mem, and NEON is enabled.
            let acc = unsafe { vld1q_f32(kv_mem.as_ptr().add(j)) };
            let row = unsafe { vld1q_f32(row_ptr.add(j)) };
            let updated = vfmaq_f32(acc, row, ki);
            // SAFETY: j..j+4 is in-bounds for kv_mem and NEON is enabled.
            unsafe { vst1q_f32(kv_mem.as_mut_ptr().add(j), updated) };
            j += 4;
        }
        while j < value_dim {
            kv_mem[j] += s[i * value_dim + j] * k[i];
            j += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn simd_decay_and_rank1_update_neon(
    s: &mut [f32],
    k: &[f32],
    delta: &[f32],
    g: f32,
    key_dim: usize,
    value_dim: usize,
) {
    use core::arch::aarch64::*;

    let g_vec = vdupq_n_f32(g);
    for i in 0..key_dim {
        let row_ptr = s.as_mut_ptr().wrapping_add(i * value_dim);
        let ki = vdupq_n_f32(k[i]);
        let mut j = 0usize;
        while j + 4 <= value_dim {
            // SAFETY: j..j+4 is in-bounds for row and delta, and NEON is enabled.
            let row = unsafe { vld1q_f32(row_ptr.add(j)) };
            let delta_vec = unsafe { vld1q_f32(delta.as_ptr().add(j)) };
            let updated = vfmaq_f32(vmulq_f32(g_vec, row), ki, delta_vec);
            // SAFETY: j..j+4 is in-bounds for row and NEON is enabled.
            unsafe { vst1q_f32(row_ptr.add(j), updated) };
            j += 4;
        }
        while j < value_dim {
            let idx = i * value_dim + j;
            s[idx] = s[idx] * g + k[i] * delta[j];
            j += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn simd_gated_rms_norm_neon(x: &[f32], z: &[f32], gamma: &[f32], out: &mut [f32], eps: f32) {
    use core::arch::aarch64::*;

    let len = x.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= len {
        // SAFETY: i..i+4 is in-bounds and NEON is enabled.
        let xv = unsafe { vld1q_f32(x.as_ptr().add(i)) };
        acc = vfmaq_f32(acc, xv, xv);
        i += 4;
    }
    let mut sum_sq = vaddvq_f32(acc);
    while i < len {
        sum_sq += x[i] * x[i];
        i += 1;
    }

    let inv_rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let inv = vdupq_n_f32(inv_rms);
    let mut gate_tmp = [0.0f32; 4];
    let mut i = 0usize;
    while i + 4 <= len {
        for lane in 0..4 {
            gate_tmp[lane] = silu(z[i + lane]);
        }
        // SAFETY: i..i+4 is in-bounds and NEON is enabled.
        let xv = unsafe { vld1q_f32(x.as_ptr().add(i)) };
        let gv = unsafe { vld1q_f32(gamma.as_ptr().add(i)) };
        let zv = unsafe { vld1q_f32(gate_tmp.as_ptr()) };
        let y = vmulq_f32(vmulq_f32(vmulq_f32(xv, inv), gv), zv);
        // SAFETY: i..i+4 is in-bounds and NEON is enabled.
        unsafe { vst1q_f32(out.as_mut_ptr().add(i), y) };
        i += 4;
    }
    while i < len {
        out[i] = (x[i] * inv_rms) * gamma[i] * silu(z[i]);
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;

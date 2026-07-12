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
    // Fail fast if weight dimension doesn't match (reviewer fix: no silent fallback).
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
mod tests {
    use super::*;
    use crate::attention::gdn::{
        GatedDeltaNetScratch, gated_delta_net_step, gated_rms_norm, l2_normalize_vec,
    };
    use proptest::prelude::*;
    use proptest::test_runner::Config as ProptestConfig;

    #[derive(Clone, Debug)]
    struct XorShift64 {
        state: u64,
    }

    impl XorShift64 {
        fn new(seed: u64) -> Self {
            let state = if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            };
            Self { state }
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        fn next_f32(&mut self) -> f32 {
            let bits = (self.next_u64() >> 40) as u32;
            (bits as f32 + 1.0) / ((1u32 << 24) as f32 + 2.0)
        }

        fn next_gaussian(&mut self, stddev: f32) -> f32 {
            let u1 = self.next_f32().max(1e-7);
            let u2 = self.next_f32();
            let mag = (-2.0 * u1.ln()).sqrt();
            let phase = 2.0 * core::f32::consts::PI * u2;
            mag * phase.cos() * stddev
        }

        fn fill_gaussian(&mut self, out: &mut [f32], stddev: f32) {
            for v in out {
                *v = self.next_gaussian(stddev);
            }
        }
    }

    fn make_test_weights(seed: u64) -> (GatedDeltaNetWeights, Qwen35Config) {
        make_test_weights_for_cfg(Qwen35Config::qwen35_2b(), seed)
    }

    fn make_test_weights_for_cfg(
        cfg: Qwen35Config,
        seed: u64,
    ) -> (GatedDeltaNetWeights, Qwen35Config) {
        let hidden = cfg.hidden_size;
        let value_heads = cfg.linear_num_value_heads();
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let value_dim = cfg.linear_value_head_dim;
        let kernel_size = cfg.linear_conv_kernel_dim;

        let mut rng = XorShift64::new(seed ^ 0xA5A5_5A5A_DEAD_BEEF);

        let mut in_proj_qkv = vec![0.0; qkv_dim * hidden];
        let mut in_proj_z = vec![0.0; output_dim * hidden];
        let mut in_proj_b = vec![0.0; value_heads * hidden];
        let mut in_proj_a = vec![0.0; value_heads * hidden];
        let mut conv1d_weight = vec![0.0; qkv_dim * kernel_size];
        let mut out_proj = vec![0.0; hidden * output_dim];
        let mut norm_weight = vec![0.0; value_dim];
        let mut a_log = vec![0.0; value_heads];
        let mut dt_bias = vec![0.0; value_heads];

        rng.fill_gaussian(&mut in_proj_qkv, 0.02);
        rng.fill_gaussian(&mut in_proj_z, 0.02);
        rng.fill_gaussian(&mut in_proj_b, 0.02);
        rng.fill_gaussian(&mut in_proj_a, 0.02);
        rng.fill_gaussian(&mut conv1d_weight, 0.02);
        rng.fill_gaussian(&mut out_proj, 0.02);
        for g in &mut norm_weight {
            *g = 1.0 + rng.next_gaussian(0.01);
        }
        for a in &mut a_log {
            *a = -1.0 + rng.next_gaussian(0.1);
        }
        for dt in &mut dt_bias {
            *dt = rng.next_gaussian(0.05);
        }

        (
            GatedDeltaNetWeights {
                in_proj_qkv,
                in_proj_qkv_rows: qkv_dim,
                in_proj_qkv_cols: hidden,
                in_proj_z,
                in_proj_z_rows: output_dim,
                in_proj_z_cols: hidden,
                in_proj_b,
                in_proj_b_rows: value_heads,
                in_proj_b_cols: hidden,
                in_proj_a,
                in_proj_a_rows: value_heads,
                in_proj_a_cols: hidden,
                a_log,
                dt_bias,
                conv1d_weight,
                conv_dim: qkv_dim,
                kernel_size,
                norm_weight,
                out_proj,
                out_proj_rows: hidden,
                out_proj_cols: output_dim,
            },
            cfg,
        )
    }

    fn assert_close_slice(a: &[f32], b: &[f32], tol: f32, name: &str) {
        assert_eq!(a.len(), b.len(), "length mismatch for {name}");
        for (idx, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (av - bv).abs();
            assert!(
                diff <= tol,
                "{name} mismatch at {idx}: left={av}, right={bv}, diff={diff}, tol={tol}"
            );
        }
    }

    fn scalar_conv1d_silu(
        new_input: &[f32],
        conv_buffer: &mut [f32],
        conv_weight: &[f32],
        output: &mut [f32],
        conv_dim: usize,
        kernel_size: usize,
    ) {
        let buf_len = kernel_size.saturating_sub(1);
        for ch in 0..conv_dim {
            let w_offset = ch * kernel_size;
            let row = &mut conv_buffer[ch * buf_len..(ch + 1) * buf_len];
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

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 2,
            max_shrink_iters: 0,
            .. ProptestConfig::default()
        })]

        #[test]
        fn fused_matches_reference(
            input in prop::collection::vec(-1.0f32..1.0, 2048),
            seed in 0u64..1000,
        ) {
            let (weights, cfg) = make_test_weights(seed);
            let mut state_ref = GatedDeltaNetState::new(&cfg);
            let mut state_fused = state_ref.clone();
            let mut scratch_ref = GatedDeltaNetScratch::default();
            let mut scratch_fused = GatedDeltaNetFusedScratch::default();
            let mut output_ref = vec![0.0f32; cfg.hidden_size];
            let mut output_fused = vec![0.0f32; cfg.hidden_size];

            gated_delta_net_step(
                &input,
                &mut state_ref,
                &weights,
                &cfg,
                &mut scratch_ref,
                &mut output_ref,
            );
            gated_delta_net_step_fused(
                &input,
                &mut state_fused,
                &weights,
                &cfg,
                &mut scratch_fused,
                &mut output_fused,
                &crate::lora_hook::NoopLoraHook,
                0,
            );

            for i in 0..cfg.hidden_size {
                prop_assert!(
                    (output_ref[i] - output_fused[i]).abs() < 1e-4,
                    "output mismatch at [{i}]: ref={}, fused={}",
                    output_ref[i],
                    output_fused[i]
                );
            }

            for i in 0..state_ref.s_matrices.len() {
                prop_assert!(
                    (state_ref.s_matrices[i] - state_fused.s_matrices[i]).abs() < 1e-4,
                    "S mismatch at [{i}]: ref={}, fused={}",
                    state_ref.s_matrices[i],
                    state_fused.s_matrices[i]
                );
            }

            for i in 0..state_ref.conv_buffer.len() {
                prop_assert!(
                    (state_ref.conv_buffer[i] - state_fused.conv_buffer[i]).abs() < 1e-7,
                    "conv buffer mismatch at [{i}]: ref={}, fused={}",
                    state_ref.conv_buffer[i],
                    state_fused.conv_buffer[i]
                );
            }
        }
    }

    #[test]
    fn multi_step_accumulation_matches_reference() {
        let (weights, cfg) = make_test_weights(7);
        let mut rng = XorShift64::new(123456789);
        let mut state_ref = GatedDeltaNetState::new(&cfg);
        let mut state_fused = state_ref.clone();
        let mut scratch_ref = GatedDeltaNetScratch::default();
        let mut scratch_fused = GatedDeltaNetFusedScratch::default();
        let mut output_ref = vec![0.0f32; cfg.hidden_size];
        let mut output_fused = vec![0.0f32; cfg.hidden_size];
        let mut input = vec![0.0f32; cfg.hidden_size];

        for step in 0..10 {
            for v in &mut input {
                *v = rng.next_gaussian(0.5);
            }

            gated_delta_net_step(
                &input,
                &mut state_ref,
                &weights,
                &cfg,
                &mut scratch_ref,
                &mut output_ref,
            );
            gated_delta_net_step_fused(
                &input,
                &mut state_fused,
                &weights,
                &cfg,
                &mut scratch_fused,
                &mut output_fused,
                &crate::lora_hook::NoopLoraHook,
                0,
            );

            assert_close_slice(
                &output_ref,
                &output_fused,
                1e-4,
                &format!("multi-step output step {step}"),
            );
            assert_close_slice(
                &state_ref.s_matrices,
                &state_fused.s_matrices,
                1e-4,
                &format!("multi-step state step {step}"),
            );
            assert_close_slice(
                &state_ref.conv_buffer,
                &state_fused.conv_buffer,
                1e-7,
                &format!("multi-step conv step {step}"),
            );
        }
    }

    #[test]
    fn simd_helpers_match_scalar_reference() {
        let mut rng = XorShift64::new(999);

        let mut l2_ref = vec![0.0f32; 128];
        let mut l2_simd = vec![0.0f32; 128];
        for v in &mut l2_ref {
            *v = rng.next_gaussian(0.7);
        }
        l2_simd.copy_from_slice(&l2_ref);
        l2_normalize_vec(&mut l2_ref);
        simd_l2_normalize(&mut l2_simd);
        assert_close_slice(&l2_ref, &l2_simd, 1e-6, "simd_l2_normalize");

        let key_dim = 128usize;
        let value_dim = 128usize;
        let mut s = vec![0.0f32; key_dim * value_dim];
        let mut k = vec![0.0f32; key_dim];
        let mut delta = vec![0.0f32; value_dim];
        rng.fill_gaussian(&mut s, 0.3);
        rng.fill_gaussian(&mut k, 0.3);
        rng.fill_gaussian(&mut delta, 0.3);

        let mut matvec_ref = vec![0.0f32; value_dim];
        let mut matvec_simd = vec![0.0f32; value_dim];
        scalar_matvec_transpose(&s, &k, &mut matvec_ref, key_dim, value_dim);
        simd_matvec_transpose(&s, &k, &mut matvec_simd, key_dim, value_dim);
        assert_close_slice(&matvec_ref, &matvec_simd, 1e-5, "simd_matvec_transpose");

        let g = 0.731f32;
        let mut update_ref = s.clone();
        let mut update_simd = s.clone();
        scalar_decay_and_rank1_update(&mut update_ref, &k, &delta, g, key_dim, value_dim);
        simd_decay_and_rank1_update(&mut update_simd, &k, &delta, g, key_dim, value_dim);
        assert_close_slice(
            &update_ref,
            &update_simd,
            1e-5,
            "simd_decay_and_rank1_update",
        );

        let mut x = vec![0.0f32; value_dim];
        let mut z = vec![0.0f32; value_dim];
        let mut gamma = vec![0.0f32; value_dim];
        rng.fill_gaussian(&mut x, 0.4);
        rng.fill_gaussian(&mut z, 0.4);
        for g in &mut gamma {
            *g = 1.0 + rng.next_gaussian(0.05);
        }
        let mut norm_ref = vec![0.0f32; value_dim];
        let mut norm_simd = vec![0.0f32; value_dim];
        gated_rms_norm(&x, &z, &gamma, &mut norm_ref, 1e-6);
        simd_gated_rms_norm(&x, &z, &gamma, &mut norm_simd, 1e-6);
        assert_close_slice(&norm_ref, &norm_simd, 1e-5, "simd_gated_rms_norm");
    }

    #[test]
    fn conv1d_silu_fused_matches_scalar_reference() {
        let conv_dim = 64usize;
        let kernel_size = 4usize;
        let mut rng = XorShift64::new(1234);
        let mut new_input = vec![0.0f32; conv_dim];
        let mut conv_weight = vec![0.0f32; conv_dim * kernel_size];
        let mut buf_ref = vec![0.0f32; conv_dim * (kernel_size - 1)];
        let mut buf_fused = vec![0.0f32; conv_dim * (kernel_size - 1)];
        let mut out_ref = vec![0.0f32; conv_dim];
        let mut out_fused = vec![0.0f32; conv_dim];
        rng.fill_gaussian(&mut new_input, 0.5);
        rng.fill_gaussian(&mut conv_weight, 0.2);
        rng.fill_gaussian(&mut buf_ref, 0.2);
        buf_fused.copy_from_slice(&buf_ref);

        scalar_conv1d_silu(
            &new_input,
            &mut buf_ref,
            &conv_weight,
            &mut out_ref,
            conv_dim,
            kernel_size,
        );
        conv1d_silu_fused(
            &new_input,
            &mut buf_fused,
            &conv_weight,
            &mut out_fused,
            conv_dim,
            kernel_size,
        );

        assert_close_slice(&out_ref, &out_fused, 1e-7, "conv1d output");
        assert_close_slice(&buf_ref, &buf_fused, 1e-7, "conv1d buffer");
    }

    #[test]
    fn zero_input_produces_zero_output() {
        let (weights, cfg) = make_test_weights(21);
        let input = vec![0.0f32; cfg.hidden_size];
        let mut state_ref = GatedDeltaNetState::new(&cfg);
        let mut state_fused = state_ref.clone();
        let mut scratch_ref = GatedDeltaNetScratch::default();
        let mut scratch_fused = GatedDeltaNetFusedScratch::default();
        let mut output_ref = vec![0.0f32; cfg.hidden_size];
        let mut output_fused = vec![0.0f32; cfg.hidden_size];

        gated_delta_net_step(
            &input,
            &mut state_ref,
            &weights,
            &cfg,
            &mut scratch_ref,
            &mut output_ref,
        );
        gated_delta_net_step_fused(
            &input,
            &mut state_fused,
            &weights,
            &cfg,
            &mut scratch_fused,
            &mut output_fused,
            &crate::lora_hook::NoopLoraHook,
            0,
        );

        assert!(output_ref.iter().all(|v| v.abs() < 1e-8));
        assert!(output_fused.iter().all(|v| v.abs() < 1e-8));
        assert_close_slice(&output_ref, &output_fused, 1e-8, "zero-input output");
        assert_close_slice(
            &state_ref.s_matrices,
            &state_fused.s_matrices,
            1e-8,
            "zero-input state",
        );
    }

    #[test]
    fn large_values_remain_finite_and_match_reference() {
        let (weights, cfg) = make_test_weights(42);
        let input: Vec<f32> = (0..cfg.hidden_size)
            .map(|i| if i % 2 == 0 { 1.0e4 } else { -1.0e4 })
            .collect();
        let mut state_ref = GatedDeltaNetState::new(&cfg);
        let mut state_fused = state_ref.clone();
        let mut scratch_ref = GatedDeltaNetScratch::default();
        let mut scratch_fused = GatedDeltaNetFusedScratch::default();
        let mut output_ref = vec![0.0f32; cfg.hidden_size];
        let mut output_fused = vec![0.0f32; cfg.hidden_size];

        gated_delta_net_step(
            &input,
            &mut state_ref,
            &weights,
            &cfg,
            &mut scratch_ref,
            &mut output_ref,
        );
        gated_delta_net_step_fused(
            &input,
            &mut state_fused,
            &weights,
            &cfg,
            &mut scratch_fused,
            &mut output_fused,
            &crate::lora_hook::NoopLoraHook,
            0,
        );

        assert!(output_ref.iter().all(|v| v.is_finite()));
        assert!(output_fused.iter().all(|v| v.is_finite()));
        assert!(state_ref.s_matrices.iter().all(|v| v.is_finite()));
        assert!(state_fused.s_matrices.iter().all(|v| v.is_finite()));
        // Large inputs (1e4) amplify f32 rounding differences between scalar
        // and SIMD paths — use a slightly looser tolerance.
        // Tolerance accommodates x86/ARM FMA accumulation order differences on large values
        assert_close_slice(&output_ref, &output_fused, 5e-3, "large-value output");
        assert_close_slice(
            &state_ref.s_matrices,
            &state_fused.s_matrices,
            2e-3,
            "large-value state",
        );
    }

    #[test]
    fn single_head_isolation_matches_reference() {
        let mut cfg = Qwen35Config::qwen35_2b();
        cfg.hidden_size = 64;
        cfg.linear_num_key_heads = 1;
        cfg.linear_num_value_heads = Some(1);
        cfg.linear_key_head_dim = 16;
        cfg.linear_value_head_dim = 16;
        cfg.linear_conv_kernel_dim = 4;

        let (weights, cfg) = make_test_weights_for_cfg(cfg, 314159);
        let mut rng = XorShift64::new(271828);
        let mut input = vec![0.0f32; cfg.hidden_size];
        rng.fill_gaussian(&mut input, 0.5);

        let mut state_ref = GatedDeltaNetState::new(&cfg);
        let mut state_fused = state_ref.clone();
        let mut scratch_ref = GatedDeltaNetScratch::default();
        let mut scratch_fused = GatedDeltaNetFusedScratch::default();
        let mut output_ref = vec![0.0f32; cfg.hidden_size];
        let mut output_fused = vec![0.0f32; cfg.hidden_size];

        gated_delta_net_step(
            &input,
            &mut state_ref,
            &weights,
            &cfg,
            &mut scratch_ref,
            &mut output_ref,
        );
        gated_delta_net_step_fused(
            &input,
            &mut state_fused,
            &weights,
            &cfg,
            &mut scratch_fused,
            &mut output_fused,
            &crate::lora_hook::NoopLoraHook,
            0,
        );

        assert_close_slice(&output_ref, &output_fused, 1e-4, "single-head output");
        assert_close_slice(
            &state_ref.s_matrices,
            &state_fused.s_matrices,
            1e-4,
            "single-head state",
        );
        assert_close_slice(
            &state_ref.conv_buffer,
            &state_fused.conv_buffer,
            1e-7,
            "single-head conv",
        );
    }

    #[test]
    fn test_fused_asymmetric_heads_nonzero_output() {
        let mut cfg = Qwen35Config::qwen35_2b();
        cfg.linear_num_value_heads = Some(48);
        let (weights, cfg) = make_test_weights_for_cfg(cfg, 42);

        let mut state = GatedDeltaNetState::new(&cfg);
        let mut scratch = GatedDeltaNetFusedScratch::default();
        let mut output = vec![0.0f32; cfg.hidden_size];

        let mut rng = XorShift64::new(123);
        let mut input = vec![0.0f32; cfg.hidden_size];
        rng.fill_gaussian(&mut input, 0.1);

        gated_delta_net_step_fused(
            &input,
            &mut state,
            &weights,
            &cfg,
            &mut scratch,
            &mut output,
            &crate::lora_hook::NoopLoraHook,
            0,
        );

        let output_dim = cfg.linear_output_dim();
        assert_eq!(output_dim, 48 * 128);

        for v_head in 0..48 {
            let start = v_head * 128;
            let head_slice = &scratch.gated_norm_buf[start..start + 128];
            let energy: f32 = head_slice.iter().map(|x| x * x).sum();
            assert!(
                energy > 0.0,
                "v_head {v_head} output is zero — asymmetric loop likely not iterating all value heads"
            );
        }
    }

    #[test]
    fn test_fused_asymmetric_s_matrix_independence() {
        let mut cfg = Qwen35Config::qwen35_2b();
        cfg.linear_num_value_heads = Some(48);
        let (weights, cfg) = make_test_weights_for_cfg(cfg, 77);

        let mut state = GatedDeltaNetState::new(&cfg);
        let mut scratch = GatedDeltaNetFusedScratch::default();
        let mut output = vec![0.0f32; cfg.hidden_size];

        let mut rng = XorShift64::new(456);
        let mut input = vec![0.0f32; cfg.hidden_size];
        rng.fill_gaussian(&mut input, 0.1);

        gated_delta_net_step_fused(
            &input,
            &mut state,
            &weights,
            &cfg,
            &mut scratch,
            &mut output,
            &crate::lora_hook::NoopLoraHook,
            0,
        );

        let s_size = 128 * 128;
        let s0 = &state.s_matrices[0..s_size];
        let s1 = &state.s_matrices[s_size..2 * s_size];
        assert_ne!(
            s0, s1,
            "v_head 0 and v_head 1 should have different S-matrices"
        );
    }

    #[test]
    fn test_fused_symmetric_backward_compat_produces_nonzero() {
        let (weights, cfg) = make_test_weights(99);
        let mut state = GatedDeltaNetState::new(&cfg);
        let mut scratch = GatedDeltaNetFusedScratch::default();
        let mut output = vec![0.0f32; cfg.hidden_size];

        let mut rng = XorShift64::new(789);
        let mut input = vec![0.0f32; cfg.hidden_size];
        rng.fill_gaussian(&mut input, 0.1);

        gated_delta_net_step_fused(
            &input,
            &mut state,
            &weights,
            &cfg,
            &mut scratch,
            &mut output,
            &crate::lora_hook::NoopLoraHook,
            0,
        );

        let output_dim = cfg.linear_output_dim();
        assert_eq!(output_dim, 16 * 128);

        for h in 0..16 {
            let start = h * 128;
            let head_slice = &scratch.gated_norm_buf[start..start + 128];
            let energy: f32 = head_slice.iter().map(|x| x * x).sum();
            assert!(energy > 0.0, "head {h} output is zero in symmetric config");
        }
    }

    #[test]
    fn test_fused_decay_gate_finite_on_exp_overflow() {
        // Mirror of gdn::test_decay_gate_finite_on_exp_overflow for the fused path's
        // own compute_decay_gate. `a_log.exp()` overflows f32 to +inf for a_log > ~88;
        // paired with softplus(very_negative) = 0.0 the product was `inf * 0.0` = NaN,
        // which poisons the recurrent state and every subsequent token.
        let g = compute_decay_gate(100.0, -200.0, 0.0);
        assert!(
            g.is_finite() && (0.0..=1.0).contains(&g),
            "fused decay gate must stay finite in [0,1] under exp overflow, got {g}"
        );
        assert!((g - 1.0).abs() < 1e-6, "expected g≈1 for dt≈0, got {g}");

        let g0 = compute_decay_gate(100.0, 5.0, 0.0);
        assert!(
            g0.is_finite() && g0 >= 0.0,
            "fused decay gate must stay finite, got {g0}"
        );
        assert!(g0 < 1e-6, "expected g≈0 for huge decay rate, got {g0}");
    }

    // The safe public SIMD wrappers below dispatch to unsafe AVX2/NEON kernels
    // that load `key_dim`/`value_dim`/`x.len()` lanes via raw pointers. A
    // length-mismatched call must fail closed (panic) rather than read OOB. The
    // guards are release-active `assert!`s, so these `#[should_panic]` tests hold
    // in both debug and release builds.

    /// Mutation-sensitive test: verifies that the 4 decay params (a_log, dt_bias,
    /// alpha, beta) are indexed per VALUE head, not per key head. With ratio=3,
    /// value heads {3,4,5} share k_head=1.
    ///
    /// Design: beta is IDENTICAL within each key group (in_proj_b rows use k_head
    /// index so same weight → same beta). V rows are identical within a group
    /// (uniform in_proj_qkv). S matrices are pre-seeded equal within a group.
    /// After one step the ONLY source of divergence between S[3] and S[4] is the
    /// decay gate: `S_new = g*S_old + outer(k, (v - g*kv_mem)*beta)`. With distinct
    /// a_log and alpha, g3≠g4, so S_new[3]≠S_new[4].
    ///
    /// If indexing reverts to [k_head]: g3=g4=g5 (all use k_head=1) and with
    /// equal beta/v/S_old the update is identical → S_new[3]=S_new[4] → assertion FAILS.
    ///
    /// Mutation-verification protocol (NEVER use `git checkout` to revert):
    ///   1. Edit: change `weights.a_log[h]` → `weights.a_log[k_head]` (and alpha/dt_bias)
    ///   2. `touch crates/inference/src/attention/gdn_fused.rs`
    ///   3. `cargo test -p lattice-inference test_fused_asymmetric_decay_params` → MUST FAIL
    ///   4. Forward-patch back to `[h]`; re-run → MUST PASS
    #[test]
    fn test_fused_asymmetric_decay_params_are_value_head_indexed() {
        // key=4, value=12, ratio=3. Value heads {3,4,5} share k_head=1.
        let mut cfg = Qwen35Config::qwen35_2b();
        cfg.hidden_size = 16;
        cfg.linear_num_key_heads = 4;
        cfg.linear_num_value_heads = Some(12);
        cfg.linear_key_head_dim = 4;
        cfg.linear_value_head_dim = 2;
        cfg.linear_conv_kernel_dim = 1;

        let key_heads = cfg.linear_num_key_heads;
        let value_heads = cfg.linear_num_value_heads();
        let ratio = value_heads / key_heads; // 3
        let hidden = cfg.hidden_size;
        let qkv_dim = cfg.linear_qkv_dim(); // 4*4*2 + 12*2 = 56
        let output_dim = cfg.linear_output_dim(); // 12*2 = 24
        let key_dim = cfg.linear_key_head_dim;
        let value_dim = cfg.linear_value_head_dim;
        let kernel_size = cfg.linear_conv_kernel_dim;

        // in_proj_a: each value head h has a DISTINCT leading weight so alpha[h] differs.
        // in_proj_b: each head uses the SAME weight as its key head so beta[h] is equal
        // within a key group — removing beta as a driver of S divergence, isolating decay.
        let mut in_proj_a = vec![0.0_f32; value_heads * hidden];
        let mut in_proj_b = vec![0.0_f32; value_heads * hidden];
        for h in 0..value_heads {
            in_proj_a[h * hidden] = (h as f32 + 1.0) * 0.3; // distinct per value head
            let k_head = h / ratio;
            in_proj_b[h * hidden] = (k_head as f32 + 1.0) * 0.2; // same within key group
        }

        // Distinct a_log and dt_bias per value head (the core parameters being fixed).
        let a_log: Vec<f32> = (0..value_heads).map(|h| (h as f32) * 0.5 - 2.5).collect();
        let dt_bias: Vec<f32> = (0..value_heads).map(|h| (h as f32) * 0.1 - 0.5).collect();

        // QKV, Z, out weights: uniform so V rows are identical across value heads.
        let in_proj_qkv = vec![0.01_f32; qkv_dim * hidden];
        let in_proj_z = vec![0.01_f32; output_dim * hidden];
        let conv1d_weight = vec![1.0_f32; qkv_dim * kernel_size];
        let out_proj = vec![0.01_f32; hidden * output_dim];
        let norm_weight = vec![1.0_f32; value_dim];

        let weights = GatedDeltaNetWeights {
            in_proj_qkv,
            in_proj_qkv_rows: qkv_dim,
            in_proj_qkv_cols: hidden,
            in_proj_z,
            in_proj_z_rows: output_dim,
            in_proj_z_cols: hidden,
            in_proj_b,
            in_proj_b_rows: value_heads,
            in_proj_b_cols: hidden,
            in_proj_a,
            in_proj_a_rows: value_heads,
            in_proj_a_cols: hidden,
            a_log,
            dt_bias,
            conv1d_weight,
            conv_dim: qkv_dim,
            kernel_size,
            norm_weight,
            out_proj,
            out_proj_rows: hidden,
            out_proj_cols: output_dim,
        };

        let mut state = GatedDeltaNetState::new(&cfg);
        let s_size = key_dim * value_dim; // 4*2 = 8

        // Pre-seed S matrices for group {3,4,5} with equal non-zero values so the
        // decay term `g * S_old` is non-trivial and different g values drive divergence.
        for j in 0..s_size {
            state.s_matrices[3 * s_size + j] = 0.5;
            state.s_matrices[4 * s_size + j] = 0.5;
            state.s_matrices[5 * s_size + j] = 0.5;
        }

        let mut scratch = GatedDeltaNetFusedScratch::default();
        let mut output = vec![0.0_f32; hidden];
        let input: Vec<f32> = (0..hidden)
            .map(|i| (i as f32 + 1.0) / (hidden as f32))
            .collect();

        gated_delta_net_step_fused(
            &input,
            &mut state,
            &weights,
            &cfg,
            &mut scratch,
            &mut output,
            &crate::lora_hook::NoopLoraHook,
            0,
        );

        assert!(
            scratch.alpha_proj.len() >= value_heads,
            "alpha_proj len {} < value_heads {}; ensure_capacity must use value_heads",
            scratch.alpha_proj.len(),
            value_heads
        );
        assert!(
            scratch.beta_proj.len() >= value_heads,
            "beta_proj len {} < value_heads {}",
            scratch.beta_proj.len(),
            value_heads
        );

        // With correct value-head indexing: a_log[3]≠a_log[4] and alpha_proj[3]≠alpha_proj[4]
        // → g3≠g4.  With regression to k_head indexing: g3=g4=g5 (all k_head=1).
        // S[3] and S[4] start equal (both seeded 0.5).  S_new = g*S_old + outer(k, delta).
        // delta = (v - g*kv_mem)*beta where v,beta,kv_mem are equal for h=3,4 by construction.
        // Therefore divergence in S_new[3] vs S_new[4] is SOLELY from g3 ≠ g4.
        let s3 = state.s_matrices[3 * s_size..4 * s_size].to_vec();
        let s4 = state.s_matrices[4 * s_size..5 * s_size].to_vec();
        assert_ne!(
            s3, s4,
            "S[h=3] and S[h=4] must diverge after one step (decay gates g3≠g4 drive it); \
             regression to k_head indexing collapses g3=g4 → S stays equal"
        );
        let s4 = state.s_matrices[4 * s_size..5 * s_size].to_vec();
        let s5 = state.s_matrices[5 * s_size..6 * s_size].to_vec();
        assert_ne!(
            s4, s5,
            "S[h=4] and S[h=5] must also diverge (decay gates g4≠g5)"
        );
    }

    #[test]
    #[should_panic]
    fn simd_matvec_transpose_rejects_short_k() {
        let s = [0.0f32; 8];
        let k = [0.0f32; 1]; // key_dim = 2 but only 1 element
        let mut kv_mem = [0.0f32; 4];
        simd_matvec_transpose(&s, &k, &mut kv_mem, 2, 4);
    }

    // ADR-080 C4: `s` is read through a raw pointer (`row_ptr.add(j)` /
    // `s.as_ptr().wrapping_add(...)`) in the AVX2/NEON kernels with no per-element bounds
    // check, unlike `k` (indexed via safe `k[i]`, which Rust already bounds-checks on its
    // own). A short `k` test above is not mutation-sensitive to this guard specifically —
    // it's caught by the incidental `k[i]` panic even with the validator removed. This test
    // exercises the case the validator uniquely guards: a too-short `s`.
    #[test]
    #[should_panic(expected = "gdn_matvec_transpose")]
    fn simd_matvec_transpose_rejects_short_s() {
        let s = [0.0f32; 7]; // key_dim*value_dim = 8, one short
        let k = [0.0f32; 2];
        let mut kv_mem = [0.0f32; 4];
        simd_matvec_transpose(&s, &k, &mut kv_mem, 2, 4);
    }

    #[test]
    #[should_panic]
    fn simd_decay_and_rank1_update_rejects_short_delta() {
        let mut s = [0.0f32; 8];
        let k = [0.0f32; 2];
        let delta = [0.0f32; 1]; // value_dim = 4 but only 1 element
        simd_decay_and_rank1_update(&mut s, &k, &delta, 0.5, 2, 4);
    }

    // ADR-080 C4: same rationale as `simd_matvec_transpose_rejects_short_s` above — `s` is
    // read/written through a raw pointer in the AVX2/NEON kernels with no bounds check, and a
    // short `k`/`delta` test doesn't exercise that guard since `k[i]`/`delta[j]` are indexed
    // via safe indexing that panics on its own even with the validator removed.
    #[test]
    #[should_panic(expected = "gdn_decay_and_rank1_update")]
    fn simd_decay_and_rank1_update_rejects_short_s() {
        let mut s = [0.0f32; 7]; // key_dim*value_dim = 8, one short
        let k = [0.0f32; 2];
        let delta = [0.0f32; 4];
        simd_decay_and_rank1_update(&mut s, &k, &delta, 0.5, 2, 4);
    }

    #[test]
    #[should_panic]
    fn simd_gated_rms_norm_rejects_short_gamma() {
        let x = [1.0f32; 8];
        let z = [0.0f32; 8];
        let gamma: [f32; 0] = []; // must equal x.len()
        let mut out = [0.0f32; 8];
        simd_gated_rms_norm(&x, &z, &gamma, &mut out, 1e-6);
    }

    // -----------------------------------------------------------------------
    // The SHIPPING scalar/AVX2/NEON l2-normalize paths must fail closed
    // identically to the reference helper.
    // -----------------------------------------------------------------------

    /// Table test for `scalar_l2_normalize` and, on the architecture the test runs on,
    /// the concrete SIMD backend it dispatches to (AVX2 on x86_64, NEON on aarch64) plus
    /// the public `simd_l2_normalize` dispatcher — all three must agree.
    #[test]
    fn simd_l2_normalize_backends_fail_closed_table() {
        struct Case {
            name: &'static str,
            input: [f32; 4],
            expect_all_zero: bool,
        }
        let cases = [
            Case {
                name: "nan_lane",
                input: [f32::NAN, 1.0, 0.0, 2.0],
                expect_all_zero: true,
            },
            Case {
                name: "pos_inf_lane",
                input: [f32::INFINITY, 1.0, 0.0, 2.0],
                expect_all_zero: true,
            },
            Case {
                name: "neg_inf_lane",
                input: [1.0, f32::NEG_INFINITY, 0.0, 2.0],
                expect_all_zero: true,
            },
            Case {
                name: "all_zero",
                input: [0.0, 0.0, 0.0, 0.0],
                expect_all_zero: true,
            },
            Case {
                name: "very_small_finite",
                input: [1e-7, 0.0, 0.0, 0.0],
                expect_all_zero: false,
            },
            Case {
                name: "ordinary_finite",
                input: [3.0, 4.0, 0.0, 0.0],
                expect_all_zero: false,
            },
        ];

        for case in &cases {
            let mut scalar_out = case.input;
            scalar_l2_normalize(&mut scalar_out);
            let mut dispatch_out = case.input;
            simd_l2_normalize(&mut dispatch_out);

            assert!(
                scalar_out.iter().all(|v| v.is_finite()),
                "case {}: scalar_l2_normalize output must be fully finite, got {:?}",
                case.name,
                scalar_out
            );
            assert!(
                dispatch_out.iter().all(|v| v.is_finite()),
                "case {}: simd_l2_normalize (dispatcher) output must be fully finite, got {:?}",
                case.name,
                dispatch_out
            );
            assert_close_slice(
                &scalar_out,
                &dispatch_out,
                1e-6,
                &format!("case {}: scalar vs dispatcher", case.name),
            );

            if case.expect_all_zero {
                assert!(
                    scalar_out.iter().all(|&v| v == 0.0),
                    "case {}: scalar_l2_normalize must zero the whole vector, got {:?}",
                    case.name,
                    scalar_out
                );
                assert!(
                    dispatch_out.iter().all(|&v| v == 0.0),
                    "case {}: simd_l2_normalize must zero the whole vector, got {:?}",
                    case.name,
                    dispatch_out
                );
            }

            #[cfg(target_arch = "x86_64")]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    let mut avx2_out = case.input;
                    // SAFETY: guarded by runtime feature detection above.
                    unsafe { simd_l2_normalize_avx2(&mut avx2_out) };
                    assert!(
                        avx2_out.iter().all(|v| v.is_finite()),
                        "case {}: simd_l2_normalize_avx2 output must be fully finite, got {:?}",
                        case.name,
                        avx2_out
                    );
                    assert_close_slice(
                        &scalar_out,
                        &avx2_out,
                        1e-6,
                        &format!("case {}: scalar vs avx2", case.name),
                    );
                    if case.expect_all_zero {
                        assert!(
                            avx2_out.iter().all(|&v| v == 0.0),
                            "case {}: simd_l2_normalize_avx2 must zero the whole vector, got {:?}",
                            case.name,
                            avx2_out
                        );
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    let mut neon_out = case.input;
                    // SAFETY: guarded by runtime feature detection above.
                    unsafe { simd_l2_normalize_neon(&mut neon_out) };
                    assert!(
                        neon_out.iter().all(|v| v.is_finite()),
                        "case {}: simd_l2_normalize_neon output must be fully finite, got {:?}",
                        case.name,
                        neon_out
                    );
                    assert_close_slice(
                        &scalar_out,
                        &neon_out,
                        1e-6,
                        &format!("case {}: scalar vs neon", case.name),
                    );
                    if case.expect_all_zero {
                        assert!(
                            neon_out.iter().all(|&v| v == 0.0),
                            "case {}: simd_l2_normalize_neon must zero the whole vector, got {:?}",
                            case.name,
                            neon_out
                        );
                    }
                }
            }
        }
    }

    /// State-isolation proof through the SHIPPING fused path
    /// (`gated_delta_net_step_fused`) using the REAL Qwen3.5 conv kernel size
    /// (`linear_conv_kernel_dim == 4`, i.e. `Qwen35Config::qwen35_2b()` unmodified — no
    /// `kernel_dim = 1` override), so `state.conv_buffer`'s rolling history is genuinely
    /// exercised, not sidestepped.
    ///
    /// ## Conv-window mechanics (read from `apply_causal_conv1d` in `attention/gdn.rs`
    /// before writing this test)
    ///
    /// `apply_causal_conv1d` stores each step's RAW per-channel projection value in a
    /// rolling `buf_len = kernel_size - 1` (= 3) history *before* L2-normalization runs;
    /// the guard only sanitizes the L2-normalize consumer, it never retroactively cleans
    /// what conv1d already pushed into `conv_buffer`. A raw value written at step `t0`
    /// is read back as one of the `kernel_size` (= 4) taps at steps `t0, t0+1, t0+2,
    /// t0+3` and is fully evicted from the window starting step `t0+4`. So a single NaN
    /// raw projection at step 0 contaminates the reduced norm at steps 0-3 (not just
    /// step 0), and step 4 onward is unaffected by it.
    ///
    /// ## What "poisoned" and "reference" mean here
    ///
    /// Both runs use `weights_window` (head-0's entire K row block zeroed) for ALL FOUR
    /// window steps (0-3), not just step 0 — the poisoned run additionally sets exactly
    /// one weight entry to NaN, ONLY at step 0. Steps 4-7 (four clean tokens, the number
    /// needed to observe the poisoned value fully age out of the `kernel_size == 4`
    /// conv window) use ordinary `weights_clean`, identical in both runs.
    ///
    /// This is a deliberate, disclosed choice, not a narrowing of the claim: holding the
    /// K row at true zero for the *whole* contamination window (not just step 0) is the
    /// only reference construction for which "bit-identical to the poisoned run" is
    /// mathematically achievable. If steps 1-3 used ordinary (nonzero) clean weights
    /// instead, the two runs would necessarily diverge during the window: the guard
    /// forces the WHOLE 128-lane K vector to zero the instant any one lane's reduced
    /// norm goes non-finite, while a genuinely undisturbed run's other 127 lanes carry
    /// real, non-zero signal at those same steps. A single corrupted lane forcing a
    /// whole-vector zero is the documented ADR-080 contract (see PR body "Contract
    /// decision"); it is not, and cannot be, bit-identical to what an always-clean run
    /// would have produced at those specific steps -- the recurrent state update is not
    /// self-correcting, so a real K contribution written at step 1-3 in a "genuinely
    /// never touched" reference could never be un-written by a later step either way.
    /// Holding K at true zero through the full window on BOTH sides isolates exactly the
    /// contract this PR fixes (whole-vector zero-by-assignment on a non-finite reduced
    /// norm) from that separate, inherent, whole-vector-zero-granularity information
    /// cost, and is the literal question this test answers: does the guard's
    /// zeroing take the identical numerical path as true-zero weights, for as long as
    /// the corruption persists in the conv window, and does state fully re-converge
    /// (`state.snapshot()`: `s_matrices` AND `conv_buffer`) once the window closes?
    ///
    /// ## Assertions
    ///
    /// - Every step (0-7): output and `s_matrices` are finite AND bit-identical between
    ///   poisoned and reference runs (fail-closed: the NaN never leaks past the guard
    ///   into the recurrent state or the output).
    /// - Post-step snapshots 0-2: the poisoned run's raw `conv_buffer` still contains
    ///   exactly the one tracked NaN (raw storage is pre-guard by design), so the
    ///   COMPLETE snapshot differs from the reference there — the assertions pin this
    ///   expected divergence explicitly rather than claiming full-state identity.
    /// - Post-step snapshot 3 onward: step 0's poisoned value has aged out of the
    ///   conv window, and the complete `state.snapshot()` (both `s_matrices` and
    ///   `conv_buffer`) is finite and bit-identical to the reference unconditionally
    ///   (no guard involvement needed) -- the concrete "recovers to bit-identical once
    ///   aged out of the conv window" proof this test exists to provide.
    ///
    /// Mutation-sensitive: reverting the `!norm_sq.is_finite()` guard in
    /// `simd_l2_normalize` (any backend) makes the poisoned run's step-0 output/state
    /// NaN, which corrupts `state.s_matrices` and every subsequent step once written --
    /// so every assertion from step 0 onward fails immediately. Note the CPU guard's
    /// condition is `!norm_sq.is_finite()` only (no `> 1e-12` near-zero threshold) --
    /// see `attention::gdn_fused::simd_l2_normalize*` vs. Metal's `isfinite(sg_buf[0]) &&
    /// sg_buf[0] > 1e-12f`. This test's zero-norm case never exercises that difference:
    /// this test's "invalid" case is a true NaN (violates `is_finite()` on both
    /// backends), not a tiny nonzero norm (where the two backends would diverge). Do not
    /// read this test as evidence the two backends share one near-zero contract; they do
    /// not (see PR body "CPU vs. Metal near-zero contract").
    #[test]
    fn gated_delta_net_step_fused_k_poison_window_state_isolation_production_kernel() {
        let cfg = Qwen35Config::qwen35_2b();
        assert_eq!(
            cfg.linear_conv_kernel_dim, 4,
            "this test exists specifically to exercise the REAL production conv kernel \
             size, not a degenerate override -- if qwen35_2b()'s default ever changes \
             this test must be updated to keep testing the real value, not silently \
             test a stale one"
        );
        let kernel_size = cfg.linear_conv_kernel_dim;
        let window = kernel_size; // # of dispatch steps a single poisoned raw value contaminates
        let clean_tail = 4; // number of clean tokens needed to fully evict the poisoned value from the kernel_size=4 conv window
        let total_steps = window + clean_tail;

        let (base_weights, cfg) = make_test_weights_for_cfg(cfg, 77);
        let hidden = cfg.hidden_size;
        let key_dim = cfg.linear_key_head_dim;
        let num_heads = cfg.linear_num_key_heads;
        let q_total = num_heads * key_dim;
        let k0_row_start = q_total; // head-0 K rows begin right after all Q rows

        let build_window_weights = |poison: bool| -> GatedDeltaNetWeights {
            let mut w = base_weights.clone();
            for r in k0_row_start..k0_row_start + key_dim {
                let row = &mut w.in_proj_qkv[r * hidden..(r + 1) * hidden];
                row.fill(0.0);
            }
            if poison {
                // One NaN lane in the first zeroed K row: input[0] * NaN = NaN
                // regardless of input[0]'s value (0 * NaN = NaN under IEEE-754), and
                // every other row stays exactly 0.0 * finite = 0.0.
                w.in_proj_qkv[k0_row_start * hidden] = f32::NAN;
            }
            w
        };
        let weights_poison = build_window_weights(true);
        let weights_zero = build_window_weights(false);
        let (weights_clean, _) = make_test_weights_for_cfg(cfg.clone(), 78);

        let mut rng = XorShift64::new(99);
        let mut inputs: Vec<Vec<f32>> = Vec::new();
        for _ in 0..total_steps {
            let mut v = vec![0.0f32; hidden];
            rng.fill_gaussian(&mut v, 0.1);
            inputs.push(v);
        }

        type OutputsAndSnapshots = (Vec<Vec<f32>>, Vec<(Vec<f32>, Vec<f32>)>); // (s_matrices, conv_buffer) per step

        let run = |step0_weights: &GatedDeltaNetWeights,
                   window_weights: &GatedDeltaNetWeights|
         -> OutputsAndSnapshots {
            let mut state = GatedDeltaNetState::new(&cfg);
            let mut scratch = GatedDeltaNetFusedScratch::default();
            let mut outputs = Vec::with_capacity(total_steps);
            let mut snapshots = Vec::with_capacity(total_steps);

            for step in 0..total_steps {
                let w = if step == 0 {
                    step0_weights
                } else if step < window {
                    window_weights
                } else {
                    &weights_clean
                };
                let mut output = vec![0.0f32; hidden];
                gated_delta_net_step_fused(
                    &inputs[step],
                    &mut state,
                    w,
                    &cfg,
                    &mut scratch,
                    &mut output,
                    &crate::lora_hook::NoopLoraHook,
                    0,
                );
                outputs.push(output);
                snapshots.push(state.snapshot());
            }
            (outputs, snapshots)
        };

        // Both runs use the zero-K window weights at steps 1..window; only step 0
        // differs (NaN vs true zero in one lane).
        let (poisoned_outputs, poisoned_snapshots) = run(&weights_poison, &weights_zero);
        let (reference_outputs, reference_snapshots) = run(&weights_zero, &weights_zero);

        // `apply_causal_conv1d` stores the RAW (pre-guard) value: the single NaN raw
        // projection written by step 0 is a genuine, expected, transient resident of
        // `conv_buffer` for exactly `buf_len = kernel_size - 1` (== 3) POST-step
        // snapshots (0, 1, 2) -- it shifts one slot left after every step and is fully
        // evicted from the buffer once `buf_len` shifts have happened, i.e. by the
        // snapshot taken after step `buf_len - 1` (== 2)'s successor, step `buf_len`
        // (== 3). `conv_buffer` is NOT guard-sanitized (only the L2-normalize consumer
        // is), so asserting it stays finite at every step would be asserting something
        // the code does not do and does not need to do -- the guard's job is to stop the
        // raw NaN from being *read* as a valid vector, not to scrub where it is stored.
        let raw_taint_snapshots = kernel_size.saturating_sub(1); // buf_len; snapshots 0..raw_taint_snapshots hold the raw NaN
        for step in 0..total_steps {
            let non_finite = poisoned_outputs[step]
                .iter()
                .filter(|v| !v.is_finite())
                .count();
            assert_eq!(
                non_finite, 0,
                "step {step}: poisoned run must fail closed (finite output), found {non_finite} non-finite element(s) -- a NaN K lane leaked past simd_l2_normalize"
            );
            let (s_matrices, _conv_buffer) = &poisoned_snapshots[step];
            let non_finite_s = s_matrices.iter().filter(|v| !v.is_finite()).count();
            assert_eq!(
                non_finite_s, 0,
                "step {step}: poisoned run's s_matrices must fail closed (finite), found {non_finite_s} non-finite element(s) -- s_matrices is only ever updated from the GUARDED (post-L2-normalize) K vector, never from raw conv_buffer directly, so it must stay finite even while conv_buffer transiently does not"
            );

            assert_eq!(
                poisoned_outputs[step], reference_outputs[step],
                "step {step}: poisoned-K run output must be bit-identical to the explicit-zero-weight reference"
            );
            assert_eq!(
                poisoned_snapshots[step].0, reference_snapshots[step].0,
                "step {step}: poisoned-K run s_matrices must be bit-identical to the explicit-zero-weight reference"
            );

            let (poisoned_conv, reference_conv) =
                (&poisoned_snapshots[step].1, &reference_snapshots[step].1);
            let reference_non_finite = reference_conv.iter().filter(|v| !v.is_finite()).count();
            assert_eq!(
                reference_non_finite, 0,
                "step {step}: the explicit-zero-weight reference never has a NaN raw \
                 projection at all, so its conv_buffer must always be fully finite"
            );
            if step < raw_taint_snapshots {
                // Inside the raw-storage taint window: exactly one conv_buffer element
                // (the shifting slot holding step 0's raw NaN) legitimately differs --
                // NaN in the poisoned run's raw storage, 0.0 in the reference's. Assert
                // that difference is exactly accounted for and nothing else diverges.
                let poisoned_non_finite = poisoned_conv.iter().filter(|v| !v.is_finite()).count();
                assert_eq!(
                    poisoned_non_finite,
                    1,
                    "step {step}: expected exactly one transient raw-NaN resident in \
                     conv_buffer (step 0's poisoned projection, still shifting through \
                     the buf_len={} window), found {poisoned_non_finite}",
                    kernel_size - 1
                );
                let mismatches = poisoned_conv
                    .iter()
                    .zip(reference_conv.iter())
                    .filter(|(p, r)| {
                        if p.is_nan() {
                            **r != 0.0
                        } else {
                            p.to_bits() != r.to_bits()
                        }
                    })
                    .count();
                assert_eq!(
                    mismatches, 0,
                    "step {step}: every conv_buffer element other than the one \
                     transient raw-NaN slot must be bit-identical between the \
                     poisoned and reference runs (found {mismatches} unexplained \
                     mismatch(es) beyond the expected NaN-vs-0.0 slot)"
                );
            } else {
                // Window fully closed: conv_buffer must now be completely finite and
                // bit-identical -- the concrete "recovers to bit-identical once aged out
                // of the conv window" proof this test provides.
                let poisoned_non_finite = poisoned_conv.iter().filter(|v| !v.is_finite()).count();
                assert_eq!(
                    poisoned_non_finite,
                    0,
                    "step {step}: conv_buffer must be fully finite once the raw-NaN \
                     resident has aged out of the buf_len={} rolling window, found \
                     {poisoned_non_finite} non-finite element(s)",
                    kernel_size - 1
                );
                assert_eq!(
                    poisoned_conv, reference_conv,
                    "step {step}: conv_buffer must be bit-identical between poisoned \
                     and reference runs once the window has fully closed"
                );
            }
        }

        assert!(
            total_steps > window,
            "test must run strictly more steps than the conv window to prove post-window convergence"
        );
    }
}

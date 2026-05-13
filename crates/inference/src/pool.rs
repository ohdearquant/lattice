use crate::forward::cpu::simd_config;

/// **Unstable**: internal pooling kernel; SIMD dispatch details may change.
///
/// Dispatches to SIMD (NEON/AVX2) when available, falls back to scalar.
pub fn mean_pool(
    hidden_states: &[f32],
    attention_mask: &[u32],
    seq_len: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let expected_hidden = seq_len
        .checked_mul(hidden_size)
        .expect("invariant: seq_len * hidden_size must fit usize");
    assert_eq!(
        hidden_states.len(),
        expected_hidden,
        "hidden_states length must match seq_len * hidden_size"
    );
    assert_eq!(
        attention_mask.len(),
        seq_len,
        "attention_mask length must match seq_len"
    );

    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is enabled by runtime config. The assertions above
            // prove hidden_states and attention_mask cover all indexed rows.
            return unsafe { mean_pool_neon(hidden_states, attention_mask, seq_len, hidden_size) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx2_enabled && config.fma_enabled {
            // SAFETY: AVX2+FMA are enabled by runtime config. The assertions
            // above prove hidden_states and attention_mask cover all indexed rows.
            return unsafe { mean_pool_avx2(hidden_states, attention_mask, seq_len, hidden_size) };
        }
    }

    mean_pool_scalar(hidden_states, attention_mask, seq_len, hidden_size)
}

fn mean_pool_scalar(
    hidden_states: &[f32],
    attention_mask: &[u32],
    seq_len: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; hidden_size];
    let mut count = 0u32;

    for i in 0..seq_len {
        if attention_mask[i] != 0 {
            let row = &hidden_states[i * hidden_size..(i + 1) * hidden_size];
            for (out, &value) in output.iter_mut().zip(row.iter()) {
                *out += value;
            }
            count += 1;
        }
    }

    if count > 0 {
        let inv_count = 1.0 / count as f32;
        for value in &mut output {
            *value *= inv_count;
        }
    }

    output
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// # Safety
///
/// Caller must ensure NEON is available, `hidden_states.len() == seq_len *
/// hidden_size`, and `attention_mask.len() == seq_len`. Pointer arithmetic is
/// bounded by those lengths; NEON loads are unaligned-safe for `f32`.
unsafe fn mean_pool_neon(
    hidden_states: &[f32],
    attention_mask: &[u32],
    seq_len: usize,
    hidden_size: usize,
) -> Vec<f32> {
    use std::arch::aarch64::*;

    let mut output = vec![0.0f32; hidden_size];
    let mut count = 0u32;
    let out_ptr = output.as_mut_ptr();
    let chunks = hidden_size / 16;

    for i in 0..seq_len {
        if attention_mask[i] == 0 {
            continue;
        }
        count += 1;
        let row_ptr = hidden_states.as_ptr().add(i * hidden_size);

        // SIMD accumulation: 4 accumulators x 4 lanes = 16 elements/iter
        for c in 0..chunks {
            let off = c * 16;
            let a0 = vld1q_f32(out_ptr.add(off) as *const f32);
            let b0 = vld1q_f32(row_ptr.add(off));
            vst1q_f32(out_ptr.add(off), vaddq_f32(a0, b0));

            let a1 = vld1q_f32(out_ptr.add(off + 4) as *const f32);
            let b1 = vld1q_f32(row_ptr.add(off + 4));
            vst1q_f32(out_ptr.add(off + 4), vaddq_f32(a1, b1));

            let a2 = vld1q_f32(out_ptr.add(off + 8) as *const f32);
            let b2 = vld1q_f32(row_ptr.add(off + 8));
            vst1q_f32(out_ptr.add(off + 8), vaddq_f32(a2, b2));

            let a3 = vld1q_f32(out_ptr.add(off + 12) as *const f32);
            let b3 = vld1q_f32(row_ptr.add(off + 12));
            vst1q_f32(out_ptr.add(off + 12), vaddq_f32(a3, b3));
        }

        // Handle remaining 4-element chunks
        let rem16_start = chunks * 16;
        let rem4_chunks = (hidden_size - rem16_start) / 4;
        for c in 0..rem4_chunks {
            let off = rem16_start + c * 4;
            let a = vld1q_f32(out_ptr.add(off) as *const f32);
            let b = vld1q_f32(row_ptr.add(off));
            vst1q_f32(out_ptr.add(off), vaddq_f32(a, b));
        }

        // Scalar remainder
        let rem_start = rem16_start + rem4_chunks * 4;
        for j in rem_start..hidden_size {
            *out_ptr.add(j) += *row_ptr.add(j);
        }
    }

    if count > 0 {
        let inv_count = 1.0 / count as f32;
        let vinv = vdupq_n_f32(inv_count);

        for c in 0..chunks {
            let off = c * 16;
            vst1q_f32(
                out_ptr.add(off),
                vmulq_f32(vld1q_f32(out_ptr.add(off) as *const f32), vinv),
            );
            vst1q_f32(
                out_ptr.add(off + 4),
                vmulq_f32(vld1q_f32(out_ptr.add(off + 4) as *const f32), vinv),
            );
            vst1q_f32(
                out_ptr.add(off + 8),
                vmulq_f32(vld1q_f32(out_ptr.add(off + 8) as *const f32), vinv),
            );
            vst1q_f32(
                out_ptr.add(off + 12),
                vmulq_f32(vld1q_f32(out_ptr.add(off + 12) as *const f32), vinv),
            );
        }

        let rem16_start = chunks * 16;
        let rem4_chunks = (hidden_size - rem16_start) / 4;
        for c in 0..rem4_chunks {
            let off = rem16_start + c * 4;
            vst1q_f32(
                out_ptr.add(off),
                vmulq_f32(vld1q_f32(out_ptr.add(off) as *const f32), vinv),
            );
        }

        let rem_start = rem16_start + rem4_chunks * 4;
        for j in rem_start..hidden_size {
            *out_ptr.add(j) *= inv_count;
        }
    }

    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
/// # Safety
///
/// Caller must ensure AVX2+FMA are available, `hidden_states.len() == seq_len *
/// hidden_size`, and `attention_mask.len() == seq_len`. Pointer arithmetic is
/// bounded by those lengths; AVX loads use unaligned load intrinsics.
unsafe fn mean_pool_avx2(
    hidden_states: &[f32],
    attention_mask: &[u32],
    seq_len: usize,
    hidden_size: usize,
) -> Vec<f32> {
    use std::arch::x86_64::*;

    let mut output = vec![0.0f32; hidden_size];
    let mut count = 0u32;
    let out_ptr = output.as_mut_ptr();
    let chunks = hidden_size / 32;

    for i in 0..seq_len {
        if attention_mask[i] == 0 {
            continue;
        }
        count += 1;
        let row_ptr = hidden_states.as_ptr().add(i * hidden_size);

        // SIMD accumulation: 4 accumulators x 8 lanes = 32 elements/iter
        for c in 0..chunks {
            let off = c * 32;
            let a0 = _mm256_loadu_ps(out_ptr.add(off) as *const f32);
            let b0 = _mm256_loadu_ps(row_ptr.add(off));
            _mm256_storeu_ps(out_ptr.add(off), _mm256_add_ps(a0, b0));

            let a1 = _mm256_loadu_ps(out_ptr.add(off + 8) as *const f32);
            let b1 = _mm256_loadu_ps(row_ptr.add(off + 8));
            _mm256_storeu_ps(out_ptr.add(off + 8), _mm256_add_ps(a1, b1));

            let a2 = _mm256_loadu_ps(out_ptr.add(off + 16) as *const f32);
            let b2 = _mm256_loadu_ps(row_ptr.add(off + 16));
            _mm256_storeu_ps(out_ptr.add(off + 16), _mm256_add_ps(a2, b2));

            let a3 = _mm256_loadu_ps(out_ptr.add(off + 24) as *const f32);
            let b3 = _mm256_loadu_ps(row_ptr.add(off + 24));
            _mm256_storeu_ps(out_ptr.add(off + 24), _mm256_add_ps(a3, b3));
        }

        // Handle remaining 8-element chunks
        let rem32_start = chunks * 32;
        let rem8_chunks = (hidden_size - rem32_start) / 8;
        for c in 0..rem8_chunks {
            let off = rem32_start + c * 8;
            let a = _mm256_loadu_ps(out_ptr.add(off) as *const f32);
            let b = _mm256_loadu_ps(row_ptr.add(off));
            _mm256_storeu_ps(out_ptr.add(off), _mm256_add_ps(a, b));
        }

        // Scalar remainder
        let rem_start = rem32_start + rem8_chunks * 8;
        for j in rem_start..hidden_size {
            *out_ptr.add(j) += *row_ptr.add(j);
        }
    }

    if count > 0 {
        let inv_count = 1.0 / count as f32;
        let vinv = _mm256_set1_ps(inv_count);

        for c in 0..chunks {
            let off = c * 32;
            _mm256_storeu_ps(
                out_ptr.add(off),
                _mm256_mul_ps(_mm256_loadu_ps(out_ptr.add(off) as *const f32), vinv),
            );
            _mm256_storeu_ps(
                out_ptr.add(off + 8),
                _mm256_mul_ps(_mm256_loadu_ps(out_ptr.add(off + 8) as *const f32), vinv),
            );
            _mm256_storeu_ps(
                out_ptr.add(off + 16),
                _mm256_mul_ps(_mm256_loadu_ps(out_ptr.add(off + 16) as *const f32), vinv),
            );
            _mm256_storeu_ps(
                out_ptr.add(off + 24),
                _mm256_mul_ps(_mm256_loadu_ps(out_ptr.add(off + 24) as *const f32), vinv),
            );
        }

        let rem32_start = chunks * 32;
        let rem8_chunks = (hidden_size - rem32_start) / 8;
        for c in 0..rem8_chunks {
            let off = rem32_start + c * 8;
            _mm256_storeu_ps(
                out_ptr.add(off),
                _mm256_mul_ps(_mm256_loadu_ps(out_ptr.add(off) as *const f32), vinv),
            );
        }

        let rem_start = rem32_start + rem8_chunks * 8;
        for j in rem_start..hidden_size {
            *out_ptr.add(j) *= inv_count;
        }
    }

    output
}

/// **Unstable**: internal pooling kernel for decoder-only models; may be merged
/// with `mean_pool` behind a strategy enum.
///
/// Used by decoder-only embedding models (Qwen3-Embedding) where the last
/// token's representation captures the full sequence meaning.
pub fn last_token_pool(
    hidden_states: &[f32],
    attention_mask: &[u32],
    seq_len: usize,
    hidden_size: usize,
) -> Vec<f32> {
    debug_assert_eq!(hidden_states.len(), seq_len * hidden_size);
    debug_assert_eq!(attention_mask.len(), seq_len);

    // Find the last non-padding position.
    let last_pos = (0..seq_len)
        .rev()
        .find(|&i| attention_mask[i] != 0)
        .unwrap_or(seq_len.saturating_sub(1));

    hidden_states[last_pos * hidden_size..(last_pos + 1) * hidden_size].to_vec()
}

/// Extract the `[CLS]` token hidden state (first token) for cross-encoder scoring.
///
/// Returns a copy of `hidden_states[0..hidden_size]`.
pub fn cls_pool(hidden_states: &[f32], seq_len: usize, hidden_size: usize) -> Vec<f32> {
    assert!(seq_len > 0, "seq_len must be non-zero for cls_pool");
    let expected_hidden = seq_len
        .checked_mul(hidden_size)
        .expect("invariant: seq_len * hidden_size must fit usize");
    assert_eq!(
        hidden_states.len(),
        expected_hidden,
        "hidden_states length must match seq_len * hidden_size"
    );
    hidden_states[..hidden_size].to_vec()
}

/// **Unstable**: internal normalization kernel; SIMD dispatch may change.
///
/// Dispatches to SIMD (NEON/AVX2) for both the dot-product norm computation
/// and the scale operation, falling back to scalar.
pub fn l2_normalize(vector: &mut [f32]) {
    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is enabled by runtime config. `vector` is a unique
            // mutable slice and the kernel bounds pointer arithmetic by len().
            unsafe { l2_normalize_neon(vector) };
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx2_enabled && config.fma_enabled {
            // SAFETY: AVX2+FMA are enabled by runtime config. `vector` is a
            // unique mutable slice and the kernel bounds pointer arithmetic by len().
            unsafe { l2_normalize_avx2(vector) };
            return;
        }
    }

    l2_normalize_scalar(vector);
}

fn l2_normalize_scalar(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv_norm = 1.0 / norm;
        for v in vector.iter_mut() {
            *v *= inv_norm;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// # Safety
///
/// Caller must ensure NEON is available. `vector` must be the only mutable
/// reference to its allocation; all loads and stores are bounded by
/// `vector.len()`.
unsafe fn l2_normalize_neon(vector: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = vector.len();
    let ptr = vector.as_mut_ptr();
    let chunks = n / 16;

    // --- Compute squared norm using SIMD ---
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    for c in 0..chunks {
        let off = c * 16;
        let v0 = vld1q_f32(ptr.add(off) as *const f32);
        acc0 = vfmaq_f32(acc0, v0, v0);
        let v1 = vld1q_f32(ptr.add(off + 4) as *const f32);
        acc1 = vfmaq_f32(acc1, v1, v1);
        let v2 = vld1q_f32(ptr.add(off + 8) as *const f32);
        acc2 = vfmaq_f32(acc2, v2, v2);
        let v3 = vld1q_f32(ptr.add(off + 12) as *const f32);
        acc3 = vfmaq_f32(acc3, v3, v3);
    }

    let combined = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let mut norm_sq = vaddvq_f32(combined);

    // Handle remaining 4-element chunks
    let rem16_start = chunks * 16;
    let rem4_chunks = (n - rem16_start) / 4;
    for c in 0..rem4_chunks {
        let off = rem16_start + c * 4;
        let v = vld1q_f32(ptr.add(off) as *const f32);
        norm_sq += vaddvq_f32(vmulq_f32(v, v));
    }

    // Scalar remainder
    let rem_start = rem16_start + rem4_chunks * 4;
    for i in rem_start..n {
        let v = *ptr.add(i);
        norm_sq += v * v;
    }

    let norm = norm_sq.sqrt();
    if norm == 0.0 {
        return;
    }

    let inv_norm = 1.0 / norm;
    let vinv = vdupq_n_f32(inv_norm);

    // --- Scale using SIMD ---
    for c in 0..chunks {
        let off = c * 16;
        vst1q_f32(
            ptr.add(off),
            vmulq_f32(vld1q_f32(ptr.add(off) as *const f32), vinv),
        );
        vst1q_f32(
            ptr.add(off + 4),
            vmulq_f32(vld1q_f32(ptr.add(off + 4) as *const f32), vinv),
        );
        vst1q_f32(
            ptr.add(off + 8),
            vmulq_f32(vld1q_f32(ptr.add(off + 8) as *const f32), vinv),
        );
        vst1q_f32(
            ptr.add(off + 12),
            vmulq_f32(vld1q_f32(ptr.add(off + 12) as *const f32), vinv),
        );
    }

    for c in 0..rem4_chunks {
        let off = rem16_start + c * 4;
        vst1q_f32(
            ptr.add(off),
            vmulq_f32(vld1q_f32(ptr.add(off) as *const f32), vinv),
        );
    }

    for i in rem_start..n {
        *ptr.add(i) *= inv_norm;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
/// # Safety
///
/// Caller must ensure AVX2+FMA are available. `vector` must be the only mutable
/// reference to its allocation; all unaligned loads and stores are bounded by
/// `vector.len()`.
unsafe fn l2_normalize_avx2(vector: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = vector.len();
    let ptr = vector.as_mut_ptr();
    let chunks = n / 32;

    // --- Compute squared norm using SIMD ---
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    for c in 0..chunks {
        let off = c * 32;
        let v0 = _mm256_loadu_ps(ptr.add(off) as *const f32);
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        let v1 = _mm256_loadu_ps(ptr.add(off + 8) as *const f32);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        let v2 = _mm256_loadu_ps(ptr.add(off + 16) as *const f32);
        acc2 = _mm256_fmadd_ps(v2, v2, acc2);
        let v3 = _mm256_loadu_ps(ptr.add(off + 24) as *const f32);
        acc3 = _mm256_fmadd_ps(v3, v3, acc3);
    }

    let combined = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut norm_sq = hsum_m256_local(combined);

    // Handle remaining 8-element chunks
    let rem32_start = chunks * 32;
    let rem8_chunks = (n - rem32_start) / 8;
    for c in 0..rem8_chunks {
        let off = rem32_start + c * 8;
        let v = _mm256_loadu_ps(ptr.add(off) as *const f32);
        norm_sq += hsum_m256_local(_mm256_mul_ps(v, v));
    }

    // Scalar remainder
    let rem_start = rem32_start + rem8_chunks * 8;
    for i in rem_start..n {
        let v = *ptr.add(i);
        norm_sq += v * v;
    }

    let norm = norm_sq.sqrt();
    if norm == 0.0 {
        return;
    }

    let inv_norm = 1.0 / norm;
    let vinv = _mm256_set1_ps(inv_norm);

    // --- Scale using SIMD ---
    for c in 0..chunks {
        let off = c * 32;
        _mm256_storeu_ps(
            ptr.add(off),
            _mm256_mul_ps(_mm256_loadu_ps(ptr.add(off) as *const f32), vinv),
        );
        _mm256_storeu_ps(
            ptr.add(off + 8),
            _mm256_mul_ps(_mm256_loadu_ps(ptr.add(off + 8) as *const f32), vinv),
        );
        _mm256_storeu_ps(
            ptr.add(off + 16),
            _mm256_mul_ps(_mm256_loadu_ps(ptr.add(off + 16) as *const f32), vinv),
        );
        _mm256_storeu_ps(
            ptr.add(off + 24),
            _mm256_mul_ps(_mm256_loadu_ps(ptr.add(off + 24) as *const f32), vinv),
        );
    }

    for c in 0..rem8_chunks {
        let off = rem32_start + c * 8;
        _mm256_storeu_ps(
            ptr.add(off),
            _mm256_mul_ps(_mm256_loadu_ps(ptr.add(off) as *const f32), vinv),
        );
    }

    for i in rem_start..n {
        *ptr.add(i) *= inv_norm;
    }
}

/// Local horizontal sum for AVX2 — mirrors the one in layers.rs but avoids
/// cross-module visibility issues.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
/// # Safety
///
/// Caller must ensure AVX is available for the `__m256` input value. This helper
/// only rearranges lanes inside the provided register and does not dereference
/// memory.
unsafe fn hsum_m256_local(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mean_pool_respects_attention_mask() {
        let hidden_states = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
            100.0, 200.0, // padding token
        ];
        let attention_mask = vec![1, 1, 0];
        let pooled = mean_pool(&hidden_states, &attention_mask, 3, 2);
        assert_eq!(pooled, vec![2.0, 3.0]);
    }

    #[test]
    fn test_mean_pool_all_masked() {
        let hidden_states = vec![1.0, 2.0, 3.0, 4.0];
        let attention_mask = vec![0, 0];
        let pooled = mean_pool(&hidden_states, &attention_mask, 2, 2);
        assert_eq!(pooled, vec![0.0, 0.0]);
    }

    #[test]
    fn test_mean_pool_simd_matches_scalar() {
        // Use hidden_size=384 (BGE-small) to exercise SIMD paths.
        let hidden_size = 384;
        let seq_len = 32;
        let hidden_states = make_deterministic_vec(seq_len * hidden_size, 0xB001);
        // Mix of 1s and 0s in the mask
        let attention_mask: Vec<u32> = (0..seq_len).map(|i| if i < 24 { 1 } else { 0 }).collect();

        let simd_result = mean_pool(&hidden_states, &attention_mask, seq_len, hidden_size);
        let scalar_result = mean_pool_scalar(&hidden_states, &attention_mask, seq_len, hidden_size);

        assert_eq!(simd_result.len(), scalar_result.len());
        for i in 0..hidden_size {
            assert_relative_eq!(simd_result[i], scalar_result[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_l2_normalize_unit_length() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert_relative_eq!(v[0], 0.6, epsilon = 1e-6);
        assert_relative_eq!(v[1], 0.8, epsilon = 1e-6);
        let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut v = vec![0.0; 384];
        l2_normalize(&mut v);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_l2_normalize_simd_matches_scalar() {
        // Use 384-dim vector to exercise SIMD paths.
        let mut simd_v = make_deterministic_vec(384, 0xA2B0);
        let mut scalar_v = simd_v.clone();

        l2_normalize(&mut simd_v);
        l2_normalize_scalar(&mut scalar_v);

        for i in 0..384 {
            assert_relative_eq!(simd_v[i], scalar_v[i], epsilon = 1e-5);
        }

        // Verify unit norm
        let norm: f32 = simd_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
    }

    // --- cls_pool tests ---

    #[test]
    fn test_cls_pool_extracts_first_token() {
        // seq_len=2, hidden_size=3: CLS vector is the first hidden_size floats
        let hidden_states = vec![
            0.1f32, 0.2, 0.3, // token 0 ([CLS])
            0.4, 0.5, 0.6, // token 1
        ];
        let cls = cls_pool(&hidden_states, 2, 3);
        assert_eq!(cls, vec![0.1f32, 0.2, 0.3]);
        // non-CLS tokens must not appear in output
        assert_eq!(cls.len(), 3);
    }

    #[test]
    fn test_cls_pool_single_token() {
        let hidden_states = vec![1.0f32, 2.0, 3.0];
        let cls = cls_pool(&hidden_states, 1, 3);
        assert_eq!(cls, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn test_cls_pool_longer_sequence_ignores_later_tokens() {
        // Even with many tokens, only first hidden_size values are returned
        let hidden_size = 2;
        let seq_len = 4;
        let hidden_states: Vec<f32> = (0..seq_len * hidden_size).map(|i| i as f32 * 0.1).collect();
        let cls = cls_pool(&hidden_states, seq_len, hidden_size);
        assert_eq!(cls, vec![0.0f32, 0.1]);
    }

    #[test]
    #[should_panic(expected = "seq_len must be non-zero")]
    fn test_cls_pool_zero_seq_len_panics() {
        cls_pool(&[], 0, 4);
    }

    /// Deterministic pseudo-random vector for reproducible tests.
    fn make_deterministic_vec(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed ^ (len as u32).wrapping_mul(0x9E37_79B9);
        if state == 0 {
            state = 0xA341_316C;
        }
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let unit = state as f32 / u32::MAX as f32;
            out.push(unit * 0.04 - 0.02);
        }
        out
    }
}

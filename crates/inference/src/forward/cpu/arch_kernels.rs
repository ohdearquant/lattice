// ---------------------------------------------------------------------------
// AVX2+FMA matmul_bt — 4 accumulators × 8 lanes = 32 elements/iter
// ---------------------------------------------------------------------------
/// **Unstable**: AVX2+FMA matmul; intrinsic selection and unroll factor may change.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn matmul_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let a_row = a.as_ptr().add(i * k);
        let c_row = &mut c[i * n..(i + 1) * n];
        for j in 0..n {
            let b_row = b.as_ptr().add(j * k);
            c_row[j] = dot_product_avx2(a_row, b_row, k);
        }
    }

    #[inline]
    unsafe fn dot_product_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
        use std::arch::x86_64::*;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let chunks = len / 32;
        for chunk in 0..chunks {
            let offset = chunk * 32;
            let a0 = _mm256_loadu_ps(a.add(offset));
            let b0 = _mm256_loadu_ps(b.add(offset));
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);

            let a1 = _mm256_loadu_ps(a.add(offset + 8));
            let b1 = _mm256_loadu_ps(b.add(offset + 8));
            acc1 = _mm256_fmadd_ps(a1, b1, acc1);

            let a2 = _mm256_loadu_ps(a.add(offset + 16));
            let b2 = _mm256_loadu_ps(b.add(offset + 16));
            acc2 = _mm256_fmadd_ps(a2, b2, acc2);

            let a3 = _mm256_loadu_ps(a.add(offset + 24));
            let b3 = _mm256_loadu_ps(b.add(offset + 24));
            acc3 = _mm256_fmadd_ps(a3, b3, acc3);
        }

        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut sum = hsum_m256(acc);

        let rem8_start = chunks * 32;
        let rem8_end = (len / 8) * 8;
        for offset in (rem8_start..rem8_end).step_by(8) {
            let av = _mm256_loadu_ps(a.add(offset));
            let bv = _mm256_loadu_ps(b.add(offset));
            sum += hsum_m256(_mm256_mul_ps(av, bv));
        }

        for offset in rem8_end..len {
            sum += *a.add(offset) * *b.add(offset);
        }

        sum
    }
}

// ---------------------------------------------------------------------------
// AVX-512 matmul_bt — 4 accumulators × 16 lanes = 64 elements/iter
// ---------------------------------------------------------------------------
/// **Unstable**: AVX-512 matmul; requires avx512f feature flag, interface may change.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn matmul_avx512(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let a_row = a.as_ptr().add(i * k);
        let c_row = &mut c[i * n..(i + 1) * n];
        for j in 0..n {
            let b_row = b.as_ptr().add(j * k);
            c_row[j] = dot_product_avx512(a_row, b_row, k);
        }
    }

    #[inline]
    unsafe fn dot_product_avx512(a: *const f32, b: *const f32, len: usize) -> f32 {
        use std::arch::x86_64::*;

        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        let chunks = len / 64;
        for chunk in 0..chunks {
            let offset = chunk * 64;
            // SAFETY: Caller guarantees a and b have at least `len` elements.
            // offset + 48..63 < chunks*64 <= len.
            let a0 = _mm512_loadu_ps(a.add(offset));
            let b0 = _mm512_loadu_ps(b.add(offset));
            acc0 = _mm512_fmadd_ps(a0, b0, acc0);

            let a1 = _mm512_loadu_ps(a.add(offset + 16));
            let b1 = _mm512_loadu_ps(b.add(offset + 16));
            acc1 = _mm512_fmadd_ps(a1, b1, acc1);

            let a2 = _mm512_loadu_ps(a.add(offset + 32));
            let b2 = _mm512_loadu_ps(b.add(offset + 32));
            acc2 = _mm512_fmadd_ps(a2, b2, acc2);

            let a3 = _mm512_loadu_ps(a.add(offset + 48));
            let b3 = _mm512_loadu_ps(b.add(offset + 48));
            acc3 = _mm512_fmadd_ps(a3, b3, acc3);
        }

        // Combine the four accumulators into one and reduce.
        let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        let mut sum = _mm512_reduce_add_ps(acc);

        // Handle remaining 16-element chunks.
        let rem16_start = chunks * 64;
        let rem16_end = (len / 16) * 16;
        for offset in (rem16_start..rem16_end).step_by(16) {
            let av = _mm512_loadu_ps(a.add(offset));
            let bv = _mm512_loadu_ps(b.add(offset));
            sum += _mm512_reduce_add_ps(_mm512_mul_ps(av, bv));
        }

        // Scalar remainder.
        for offset in rem16_end..len {
            sum += *a.add(offset) * *b.add(offset);
        }

        sum
    }
}

// ---------------------------------------------------------------------------
// NEON matmul_bt — 4 accumulators × 4 lanes = 16 elements/iter
// ---------------------------------------------------------------------------
/// **Unstable**: NEON matmul; intrinsic selection and unroll factor may change.
///
/// # Safety
///
/// Caller must ensure this runs only when NEON is available and that `a`, `b`,
/// and `c` cover the `[m, k]`, `[n, k]`, and `[m, n]` regions respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn matmul_neon(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let a_row = a.as_ptr().add(i * k);
        let c_row = &mut c[i * n..(i + 1) * n];
        for (j, c_val) in c_row.iter_mut().enumerate().take(n) {
            let b_row = b.as_ptr().add(j * k);
            *c_val = dot_product_neon(a_row, b_row, k);
        }
    }

    #[inline]
    unsafe fn dot_product_neon(a: *const f32, b: *const f32, len: usize) -> f32 {
        use std::arch::aarch64::*;

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let chunks = len / 16;
        for chunk in 0..chunks {
            let offset = chunk * 16;
            let a0 = vld1q_f32(a.add(offset));
            let b0 = vld1q_f32(b.add(offset));
            acc0 = vfmaq_f32(acc0, a0, b0);

            let a1 = vld1q_f32(a.add(offset + 4));
            let b1 = vld1q_f32(b.add(offset + 4));
            acc1 = vfmaq_f32(acc1, a1, b1);

            let a2 = vld1q_f32(a.add(offset + 8));
            let b2 = vld1q_f32(b.add(offset + 8));
            acc2 = vfmaq_f32(acc2, a2, b2);

            let a3 = vld1q_f32(a.add(offset + 12));
            let b3 = vld1q_f32(b.add(offset + 12));
            acc3 = vfmaq_f32(acc3, a3, b3);
        }

        let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut sum = vaddvq_f32(acc);

        let rem4_start = chunks * 16;
        let rem4_end = (len / 4) * 4;
        for offset in (rem4_start..rem4_end).step_by(4) {
            let av = vld1q_f32(a.add(offset));
            let bv = vld1q_f32(b.add(offset));
            sum += vaddvq_f32(vmulq_f32(av, bv));
        }

        for offset in rem4_end..len {
            sum += *a.add(offset) * *b.add(offset);
        }

        sum
    }
}

// ---------------------------------------------------------------------------
// Shared x86_64 horizontal sum helper
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn hsum_m256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

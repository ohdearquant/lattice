/// **Unstable**: RMS normalization in-place; may add SIMD path.
///
/// RMS normalization: x = x * gamma / rms(x), where rms = sqrt(mean(x^2) + eps).
///
/// Unlike LayerNorm, RMSNorm does not center (no beta/mean subtraction).
pub fn rms_norm(x: &mut [f32], gamma: &[f32], hidden: usize, eps: f32) {
    let num_tokens = x.len() / hidden;
    debug_assert_eq!(x.len(), num_tokens * hidden);
    debug_assert_eq!(gamma.len(), hidden);

    for t in 0..num_tokens {
        let row = &mut x[t * hidden..(t + 1) * hidden];

        // Compute mean of squares.
        let mut sum_sq = 0.0f32;
        for &v in row.iter() {
            sum_sq += v * v;
        }
        let rms = (sum_sq / hidden as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Scale by gamma / rms.
        for (v, &g) in row.iter_mut().zip(gamma.iter()) {
            *v = *v * inv_rms * g;
        }
    }
}

/// **Unstable**: SiLU in-place; may gain SIMD path.
///
/// SiLU activation: x = x * sigmoid(x).
#[inline]
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v * (1.0 / (1.0 + (-*v).exp()));
    }
}

/// **Unstable**: element-wise multiply in-place; may gain SIMD path.
///
/// Element-wise multiply: a *= b.
#[inline]
pub fn elementwise_mul(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (av, &bv) in a.iter_mut().zip(b.iter()) {
        *av *= bv;
    }
}

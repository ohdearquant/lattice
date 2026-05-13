/// Qwen3.5 uses a shifted RMSNorm where gamma is offset by 1.0.
/// This is different from standard RMSNorm (used by Qwen3/Llama) which uses plain gamma.
pub(crate) fn qwen35_rms_norm(x: &mut [f32], gamma: &[f32], hidden: usize, eps: f32) {
    let num_tokens = x.len() / hidden;
    debug_assert_eq!(x.len(), num_tokens * hidden);
    debug_assert_eq!(gamma.len(), hidden);

    for t in 0..num_tokens {
        let row = &mut x[t * hidden..(t + 1) * hidden];
        let mut sum_sq = 0.0f32;
        for &v in row.iter() {
            sum_sq += v * v;
        }
        let rms = (sum_sq / hidden as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for (v, &g) in row.iter_mut().zip(gamma.iter()) {
            *v = *v * inv_rms * (1.0 + g);
        }
    }
}

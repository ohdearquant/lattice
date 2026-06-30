//! Shifted RMSNorm implementation used by the Qwen3.5 forward path.
/// Qwen3.5 uses a shifted RMSNorm: `x * rsqrt(mean(x²) + eps) * (1 + gamma)`.
/// Empirically verified by PPL — plain `gamma` (the standard convention used
/// by Qwen3/Llama and MLX `nn.RMSNorm`) produces nonsense logits on this
/// model. MLX must absorb the +1 into weights at load time.
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

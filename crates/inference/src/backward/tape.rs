// Forward pass that caches all activations needed for the backward pass.
// The scope for this milestone: only layer-23 (the last GQA layer) is trained.
// Layers 0..23 are a frozen prefix — we run them forward and save the residual
// stream at layer-23's input. From there we cache everything needed to
// differentiate through layer-23 and the final head.

/// Cached activations for a single GQA layer (layer-23 in the full model).
pub struct LayerCache {
    /// Residual stream input before pre-attention RMSNorm: [hidden]
    pub residual_pre_attn: Vec<f32>,
    /// Pre-attention RMSNorm output (i.e. the normed hidden): [hidden]
    pub normed_pre_attn: Vec<f32>,
    /// inv_rms values for the pre-attention RMSNorm: scalar per token, stored [1]
    pub inv_rms_pre_attn: f32,
    /// Pre-FFN residual: [hidden]
    pub residual_pre_ffn: Vec<f32>,
    /// Normed pre-FFN: [hidden]
    pub normed_pre_ffn: Vec<f32>,
    /// inv_rms for pre-FFN norm: scalar
    pub inv_rms_pre_ffn: f32,
    /// Gate pre-activation (before silu) for SwiGLU: [inter]
    pub gate_pre: Vec<f32>,
    /// Up pre-activation: [inter]
    pub up_pre: Vec<f32>,
    /// Attention output (after o_proj) before residual add: [hidden]
    pub attn_out: Vec<f32>,
    /// FFN output (after down_proj) before residual add: [hidden]
    pub ffn_out: Vec<f32>,
}

/// Activations from the full sequence forward pass through one GQA layer.
/// Stored per-token so the backward can iterate.
pub struct SequenceLayerCache {
    pub tokens: Vec<LayerCache>,
    /// Q pre-rope per-token, packed: [seq_len * q_dim]
    pub q_pre_rope: Vec<f32>,
    /// K pre-rope per-token: [seq_len * kv_dim]
    pub k_pre_rope: Vec<f32>,
    /// V per-token: [seq_len * kv_dim]
    pub v: Vec<f32>,
    /// Softmax probs per position: vec of length seq_len, each element is
    /// a flat [num_q_heads * (t+1)] probs vector for position t.
    pub softmax_probs: Vec<Vec<f32>>,
    /// Context (after softmax-weighted sum, before o_proj): [seq_len * q_dim]
    pub context: Vec<f32>,
    /// h_q = A_q x per-token: [seq_len * rank]
    pub h_q: Vec<f32>,
    /// h_v = A_v x per-token: [seq_len * rank]
    pub h_v: Vec<f32>,
    /// Q after rope: per-token head layout [seq_len][q_dim]
    pub q_after_rope: Vec<Vec<f32>>,
}

/// Top-level activation tape covering the trained segment.
pub struct BackwardTape {
    /// Hidden state at the input of layer-23 (= output of layer-22 after residual).
    /// Shape [hidden] — we treat the prefix as frozen so we only need this boundary.
    pub layer23_input: Vec<f32>,
    /// Per-sequence cache for layer-23.
    pub layer23: SequenceLayerCache,
    /// Final RMSNorm normed output: [hidden]
    pub final_normed: Vec<f32>,
    /// inv_rms for the final norm: scalar
    pub inv_rms_final: f32,
    /// Logits: [vocab_size]
    pub logits: Vec<f32>,
}

/// RMSNorm forward + cache: returns (normed, inv_rms).
///
/// Convention: shifted gamma — y_i = x_i * (1 + w_i) * inv_rms
/// Matches `qwen35_rms_norm` in norm.rs (required for Qwen3.5 pre-attn and pre-FFN norms).
pub fn rms_norm_forward(x: &[f32], w: &[f32], eps: f32) -> (Vec<f32>, f32) {
    let d = x.len();
    let mean_sq: f32 = x.iter().map(|xi| xi * xi).sum::<f32>() / d as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let normed: Vec<f32> = x
        .iter()
        .zip(w.iter())
        .map(|(xi, wi)| xi * (1.0 + wi) * inv_rms)
        .collect();
    (normed, inv_rms)
}

/// SwiGLU forward: returns (output [hidden], gate_pre [inter], up_pre [inter]).
pub fn swiglu_forward(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    hidden: usize,
    inter: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut gate_pre = vec![0.0f32; inter];
    let mut up_pre = vec![0.0f32; inter];
    for i in 0..inter {
        gate_pre[i] = w_gate[i * hidden..(i + 1) * hidden]
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum();
        up_pre[i] = w_up[i * hidden..(i + 1) * hidden]
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum();
    }

    // silu(gate) * up
    let mixed: Vec<f32> = gate_pre
        .iter()
        .zip(up_pre.iter())
        .map(|(&g, &u)| {
            let s = 1.0 / (1.0 + (-g).exp());
            g * s * u
        })
        .collect();

    // down_proj
    let mut out = vec![0.0f32; hidden];
    for i in 0..hidden {
        out[i] = w_down[i * inter..(i + 1) * inter]
            .iter()
            .zip(mixed.iter())
            .map(|(a, b)| a * b)
            .sum();
    }
    (out, gate_pre, up_pre)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_forward_roundtrip() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        // Use w=0.5 (nonzero, non-unity) so shifted vs plain gamma produce distinct results.
        // Shifted convention: y_i = x_i * (1 + w_i) * inv_rms = x_i * 1.5 * inv_rms
        let w = vec![0.5f32; 4];
        let (normed, inv_rms) = rms_norm_forward(&x, &w, 1e-6);
        let mean_sq: f32 = x.iter().map(|xi| xi * xi).sum::<f32>() / 4.0;
        let expected_inv = 1.0 / (mean_sq + 1e-6f32).sqrt();
        assert!((inv_rms - expected_inv).abs() < 1e-6, "inv_rms mismatch");
        // Shifted gamma: expected = x_i * (1 + w_i) * inv_rms = x_i * 1.5 * inv_rms
        let expected_norm: Vec<f32> = x.iter().map(|xi| xi * 1.5 * expected_inv).collect();
        for (a, b) in normed.iter().zip(expected_norm.iter()) {
            assert!((a - b).abs() < 1e-5, "normed mismatch {a} vs {b}");
        }
    }

    #[test]
    fn swiglu_forward_smoke() {
        let hidden = 2;
        let inter = 3;
        let x = vec![1.0f32, -0.5];
        let w_gate = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
        let w_up = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];
        let w_down = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let (out, gate_pre, up_pre) = swiglu_forward(&x, &w_gate, &w_up, &w_down, hidden, inter);
        assert_eq!(out.len(), hidden);
        assert_eq!(gate_pre.len(), inter);
        assert_eq!(up_pre.len(), inter);
    }
}

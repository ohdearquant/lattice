//! Scalar projection, normalization, activation and attention helpers shared by vision encoders.

/// Batch matrix-vector: A [rows, cols] applied to each column of X [n, cols].
/// Output is [n, rows].
pub fn batch_matvec(a: &[f32], x: &[f32], n: usize, rows: usize, cols: usize) -> Vec<f32> {
    // Guard the output allocation itself against usize overflow (ADR-080 C4, same rationale
    // as `matmul()` in forward/cpu/matmul.rs): an unchecked `n * rows` can wrap in release,
    // producing a too-small allocation that would then pass a subsequent `>=` length check
    // on the same wrapped value.
    assert!(
        n.checked_mul(rows).is_some(),
        "vit_batch_matvec: output shape overflow: n*rows"
    );
    let mut out = vec![0.0f32; n * rows];
    // Modeled as C = A @ B^T with m=n (batch size), k=cols, n=rows (X is the "a" operand,
    // A is the "b" operand — [rows, cols], transposed in the multiply — out is the "c"
    // operand, already safely allocated above). Release-active, overflow-first,
    // oversized-scratch-allowed, replacing the previous exact-match `assert_eq!` pair.
    crate::forward::cpu::validate_gemm_bt(
        x.len(),
        a.len(),
        out.len(),
        n,
        cols,
        rows,
        "vit_batch_matvec",
    );
    for i in 0..n {
        let xi = &x[i * cols..(i + 1) * cols];
        let yi = &mut out[i * rows..(i + 1) * rows];
        for r in 0..rows {
            let row = &a[r * cols..(r + 1) * cols];
            let mut acc = 0.0f32;
            for (a_val, x_val) in row.iter().zip(xi.iter()) {
                acc += a_val * x_val;
            }
            yi[r] = acc;
        }
    }
    out
}

/// LayerNorm: normalize x by (x - mean) / sqrt(var + eps), then scale by weight + bias.
pub fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let n = x.len();
    assert_eq!(weight.len(), n);
    assert_eq!(bias.len(), n);

    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();

    for (i, v) in x.iter_mut().enumerate() {
        *v = (*v - mean) * inv_std * weight[i] + bias[i];
    }
}

/// GELU activation (approximation matching PyTorch's tanh approximation).
///
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// Softmax over a slice of logits (in-place).
///
/// ADR-080 C1: routes the fail-closed row contract through the shared
/// row-finalizer (#741). Previously this had NO guard at all — a NaN or
/// `+inf` logit propagated NaN through the unconditional division into every
/// element of the row and onward through the rest of the ViT forward pass.
pub fn softmax_inplace(x: &mut [f32]) {
    let (max, any_nan) = crate::attention::softmax_row::row_max_and_any_nan(x);
    if crate::attention::softmax_row::row_fails_closed_pre_exp(max, any_nan) {
        x.fill(0.0);
        return;
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    crate::attention::softmax_row::finalize_row(x, sum);
}

/// Apply rotate-half RoPE to one `[head_dim]` vector in place, given
/// `cos`/`sin` each `[head_dim]` (matches `apply_rotary_pos_emb_vision` in
/// the HF reference: `x_embed = x * cos + rotate_half(x) * sin`, where
/// `rotate_half(x) = cat(-x[half:], x[:half])`).
///
/// Shared by CPU vision encoders and their Metal setup paths to preserve
/// the rotate-half convention (ADR-069 S3b's "zero independent convention
/// decisions" contract).
pub fn apply_rope_inplace(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    for i in 0..half {
        let a = x[i];
        let b = x[half + i];
        x[i] = a * cos[i] + (-b) * sin[i];
        x[half + i] = b * cos[half + i] + a * sin[half + i];
    }
}

/// Standard full multi-head self-attention (no causal mask, no windowing —
/// the entire `[n, ...]` sequence is one attention segment).
/// `qkv`: `[n, 3 * hidden]` with per-row layout `[Q(hidden) | K(hidden) | V(hidden)]`,
/// each of Q/K/V split into `n_heads` contiguous `head_dim`-wide chunks.
pub fn multihead_attention_full(
    qkv: &[f32],
    n: usize,
    hidden: usize,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n * hidden];

    for h in 0..n_heads {
        let mut q_h = vec![0.0f32; n * head_dim];
        let mut k_h = vec![0.0f32; n * head_dim];
        let mut v_h = vec![0.0f32; n * head_dim];
        for i in 0..n {
            let base = i * 3 * hidden;
            q_h[i * head_dim..(i + 1) * head_dim]
                .copy_from_slice(&qkv[base + h * head_dim..base + (h + 1) * head_dim]);
            k_h[i * head_dim..(i + 1) * head_dim].copy_from_slice(
                &qkv[base + hidden + h * head_dim..base + hidden + (h + 1) * head_dim],
            );
            v_h[i * head_dim..(i + 1) * head_dim].copy_from_slice(
                &qkv[base + 2 * hidden + h * head_dim..base + 2 * hidden + (h + 1) * head_dim],
            );
        }

        let mut scores = vec![0.0f32; n * n];
        for i in 0..n {
            let qi = &q_h[i * head_dim..(i + 1) * head_dim];
            for j in 0..n {
                let kj = &k_h[j * head_dim..(j + 1) * head_dim];
                let dot: f32 = qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
                scores[i * n + j] = dot * scale;
            }
        }
        for i in 0..n {
            softmax_inplace(&mut scores[i * n..(i + 1) * n]);
        }
        for i in 0..n {
            let attn_row = &scores[i * n..(i + 1) * n];
            for j in 0..head_dim {
                let mut acc = 0.0f32;
                for k in 0..n {
                    acc += attn_row[k] * v_h[k * head_dim + j];
                }
                out[i * hidden + h * head_dim + j] = acc;
            }
        }
    }

    out
}

/// Error-function approximation (Abramowitz & Stegun 7.1.26, max absolute
/// error ~1.5e-7), computed in f64 for headroom above f32 output precision.
/// `std` has no stable `erf`, and exact (non-tanh-approximated) GELU is the
/// merger's HF-pinned activation (see module docs), so it is implemented
/// here rather than reusing [`gelu`].
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

/// Exact GELU: `0.5 * x * (1 + erf(x / sqrt(2)))` — matches PyTorch's
/// `nn.GELU()` default (`approximate="none"`), which is what
/// `Qwen3_5VisionPatchMerger.act_fn` hardcodes.
#[inline(always)]
pub fn gelu_exact(x: f32) -> f32 {
    let xd = x as f64;
    (0.5 * xd * (1.0 + erf(xd / std::f64::consts::SQRT_2))) as f32
}

#[cfg(test)]
mod tests;

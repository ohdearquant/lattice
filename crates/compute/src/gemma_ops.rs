//! Shared Gemma normalization, activation, projection and rotary-embedding kernels.

use crate::forward::cpu::{elementwise_mul, matmul_bt, rms_norm};
#[doc(inline)]
pub use crate::gemma_rope::{
    gemma4_rope_cos_sin, gemma4_rope_inv_freq, vision_axial_rope_inv_freq,
};

/// Standard RMSNorm: `x * rsqrt(mean(x^2) + eps) * gamma` -- plain gamma
/// scaling, **not** lattice's Qwen3.5 shifted `(1 + gamma)` variant
/// (`lattice_inference::model::qwen35::qwen35_rms_norm`), exercised by
/// `mutation_shifted_rms_norm_fails_golden`, retained in
/// `crates/inference/src/model/gemma4_ops/tests.rs`.
///
/// Reuses `crate::forward::cpu::rms_norm` as-is: this is **bounded f32
/// parity with `Gemma4RMSNorm.forward`, not bit-for-bit equivalence**. The
/// reference computes `pow(mean(x^2) + eps, -0.5)` (a single `pow` call);
/// this kernel computes `sqrt(mean(x^2) + eps)` then reciprocal (two
/// rounding steps), and its NEON path reduces `sum(x^2)` in 4-wide partial
/// sums merged pairwise -- a different accumulation order than either the
/// reference or this kernel's own scalar fallback. The golden fixture set
/// covers this at both a 1/24-scaled synthetic width (`rms_norm.json`,
/// hidden=64) and the real E2B `hidden_size=1536` (`rms_norm_wide.json`,
/// `tests::rms_norm_wide_matches_hf_golden`), each within its manifest's
/// predeclared tolerance -- see
/// `crates/compute/tests/fixtures/gemma4/stage3/manifest.json`'s
/// `rms_norm`/`rms_norm_wide` `tolerance_justification` for the structural
/// bound.
pub fn gemma4_rms_norm(x: &mut [f32], gamma: &[f32], hidden: usize, eps: f32) {
    rms_norm(x, gamma, hidden, eps);
}

/// Exact tanh-approximate GELU (`gelu_pytorch_tanh`), matching HF's
/// `ACT2FN["gelu_pytorch_tanh"]` to f32-tanh rounding.
///
/// Deliberately **not** `crate::forward::cpu::gelu`: that kernel's
/// `fast_tanh` is a rational approximation of tanh (see its doc comment for
/// the measured bound), while this stage is checked against HF with a
/// predeclared 1e-5 max-abs-diff tolerance for `geglu_mlp`. This uses
/// `f32::tanh` (libm, correctly rounded to a few ULPs) instead -- same GELU
/// formula, exact tanh.
fn gelu_tanh_exact(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// In-place exact tanh-approximate GELU (`gelu_pytorch_tanh`) over a whole
/// slice -- the same `gelu_tanh_exact` scalar op [`gemma4_geglu_mlp`] uses
/// internally, exposed for the per-layer-embedding gate activation
/// (`Gemma4TextDecoderLayer.forward`'s `self.act_fn(per_layer_input_gate(x))`,
/// `modeling_gemma4.py:1451`), which is a plain activation with no MLP
/// wiring around it.
pub fn gemma4_gelu_tanh(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = gelu_tanh_exact(*v);
    }
}

/// GeGLU MLP: `down(gelu_tanh(gate(x)) * up(x))`
/// (`Gemma4TextMLP.forward`, `modeling_gemma4.py:1079-1081`).
///
/// `gate_w`/`up_w` are `[intermediate, hidden]` and `down_w` is
/// `[hidden, intermediate]` (`nn.Linear` weight layout, `[out, in]`) --
/// exactly the layout `crate::forward::cpu::matmul_bt` expects as its `B`
/// argument.
///
/// `gate_scratch`/`up_scratch` are caller-provided `tokens*intermediate`
/// buffers (their contents on entry are irrelevant -- both are fully
/// overwritten by the first `matmul_bt` before being read) so that a
/// per-layer caller can reuse one allocation across every layer of a
/// forward pass instead of allocating two fresh vectors per call.
#[allow(clippy::too_many_arguments)]
pub fn gemma4_geglu_mlp(
    x: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    tokens: usize,
    hidden: usize,
    intermediate: usize,
    gate_scratch: &mut [f32],
    up_scratch: &mut [f32],
    out: &mut [f32],
) {
    assert_eq!(x.len(), tokens * hidden, "x must be tokens*hidden");
    assert_eq!(
        gate_w.len(),
        intermediate * hidden,
        "gate_w must be intermediate*hidden"
    );
    assert_eq!(
        up_w.len(),
        intermediate * hidden,
        "up_w must be intermediate*hidden"
    );
    assert_eq!(
        down_w.len(),
        hidden * intermediate,
        "down_w must be hidden*intermediate"
    );
    assert_eq!(out.len(), tokens * hidden, "out must be tokens*hidden");
    assert_eq!(
        gate_scratch.len(),
        tokens * intermediate,
        "gate_scratch must be tokens*intermediate"
    );
    assert_eq!(
        up_scratch.len(),
        tokens * intermediate,
        "up_scratch must be tokens*intermediate"
    );

    matmul_bt(x, gate_w, gate_scratch, tokens, hidden, intermediate);
    matmul_bt(x, up_w, up_scratch, tokens, hidden, intermediate);
    for v in gate_scratch.iter_mut() {
        *v = gelu_tanh_exact(*v);
    }
    elementwise_mul(gate_scratch, up_scratch);
    matmul_bt(gate_scratch, down_w, out, tokens, intermediate, hidden);
}

/// Per-head RMSNorm applied to Q and K (each with a learned scale weight);
/// V gets the same RMSNorm shape but **unscaled** -- no learned weight,
/// mathematically `rms_norm(v, ones, head_dim, eps)`.
/// (`Gemma4TextAttention.__init__` constructs `q_norm`/`k_norm` with
/// `with_scale` defaulted `True` and `v_norm = Gemma4RMSNorm(head_dim, eps,
/// with_scale=False)` at `modeling_gemma4.py:1208-1212`; `forward` applies
/// `q_norm` then RoPE (`:1240-1241`), `k_norm` then RoPE (`:1257`), and
/// `v_norm` with **no** RoPE at all (`:1260`) -- V never rotates, see
/// [`gemma4_apply_rope`].)
///
/// Substituting a scaled norm for V (i.e. reusing `q_gamma`/`k_gamma`-style
/// weights on V) is this stage's V-norm negative test
/// (`tests::mutation_scaled_v_norm_fails_golden`).
pub fn gemma4_qk_norm_v_unscaled(
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    q_gamma: &[f32],
    k_gamma: &[f32],
    head_dim: usize,
    eps: f32,
) {
    rms_norm(q, q_gamma, head_dim, eps);
    rms_norm(k, k_gamma, head_dim, eps);
    let ones = vec![1.0f32; head_dim];
    rms_norm(v, &ones, head_dim, eps);
}

/// Apply RoPE to a `[seq_len, heads, head_dim]` tensor (row-major, `head_dim`
/// fastest-varying) via `x * cos + rotate_half(x) * sin`
/// (`apply_rotary_pos_emb`, `modeling_gemma4.py:787-806`).
///
/// `rotate_half` (`modeling_gemma4.py:780-785`) pairs dimension `i` with
/// dimension `head_dim/2 + i` (**stride-half**: `x1 = x[..half]`,
/// `x2 = x[half..]`, `rotate = cat(-x2, x1)`), **not** the interleaved
/// `(2i, 2i+1)` convention -- this repo's worst historical bug class
/// (`CLAUDE.md` "RoPE pairing alt-forward variants"; ADR-082's own Risks
/// section names this exact failure mode). Verified directly against the
/// pinned source line-for-line above, not inferred from a comment.
///
/// Swapping the local/global `theta` passed to [`gemma4_rope_inv_freq`]
/// before calling this is this stage's dual-RoPE negative test
/// (`tests::mutation_swapped_theta_fails_golden`).
pub fn gemma4_apply_rope(
    x: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    seq_len: usize,
    heads: usize,
    head_dim: usize,
) {
    assert_eq!(
        x.len(),
        seq_len * heads * head_dim,
        "x must be seq_len*heads*head_dim"
    );
    assert_eq!(
        cos.len(),
        seq_len * head_dim,
        "cos must be seq_len*head_dim"
    );
    assert_eq!(
        sin.len(),
        seq_len * head_dim,
        "sin must be seq_len*head_dim"
    );
    // Stride-half pairing requires an even head_dim; an odd width would
    // leave the last lane outside every (i, half+i) pair and silently
    // diverge from the rotate-half reference.
    assert!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "head_dim must be even and > 0 for stride-half RoPE, got {head_dim}"
    );
    let half = head_dim / 2;
    for t in 0..seq_len {
        let cos_row = &cos[t * head_dim..(t + 1) * head_dim];
        let sin_row = &sin[t * head_dim..(t + 1) * head_dim];
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            let row = &mut x[base..base + head_dim];
            // Same arithmetic as materializing `rotate_half(row)` into a
            // temporary buffer then computing `row * cos + rotated * sin`
            // element-wise, just without the heap allocation: each pair
            // `(i, half+i)` only reads its own pre-update values (`x1`,
            // `x2`), so writing both outputs of a pair before moving to the
            // next pair is order-independent and bit-identical to the
            // original two-pass form (even head_dim enforced above; pairs
            // are disjoint only in that domain).
            for i in 0..half {
                let x1 = row[i];
                let x2 = row[half + i];
                row[i] = x1 * cos_row[i] - x2 * sin_row[i];
                row[half + i] = x2 * cos_row[half + i] + x1 * sin_row[half + i];
            }
        }
    }
}

/// Axial 2-D RoPE applied in place to a `[seq, heads, head_dim]` tensor (row-major,
/// `head_dim` fastest-varying).
///
/// This mirrors the reference's `apply_multidimensional_rope` with `ndim = 2`
/// (`modeling_gemma4.py:843-891`) rather than the single-axis rotate-half in
/// [`gemma4_apply_rope`]. Each head splits into two `head_dim / 2` parts — the first
/// rotated by the axis-0 (x) angle, the second by axis-1 (y) — and **within each part**
/// the rotate-half pairing ties lane `j` to lane `j + head_dim / 4`. At `head_dim = 64`
/// that is the reference's "two 32-channel halves, lane `j` paired with `j + 16`" form.
///
/// A whole-64 stride-half pairing, the layout [`gemma4_apply_rope`] implements, is a
/// different map: it pairs lane `j` with `j + 32`, so the two functions must not be
/// substituted for one another even though both are built from `rotate_half`.
pub fn vision_axial_rope_apply(
    x: &mut [f32],
    positions: &[[usize; 2]],
    inv_freq: &[f32],
    heads: usize,
    head_dim: usize,
) {
    let per_axis = head_dim / 4;
    assert!(
        per_axis > 0 && head_dim.is_multiple_of(4),
        "head_dim must be a positive multiple of 4 for axial RoPE"
    );
    assert_eq!(
        inv_freq.len(),
        per_axis * 2,
        "inv_freq must hold two per-axis blocks"
    );
    assert_eq!(
        x.len(),
        positions.len() * heads * head_dim,
        "x must be seq*heads*head_dim"
    );
    let part = head_dim / 2;
    for (t, &[x_coord, y_coord]) in positions.iter().enumerate() {
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            for axis in 0..2 {
                let coordinate = if axis == 0 { x_coord } else { y_coord } as f32;
                let offset = base + axis * part;
                for j in 0..per_axis {
                    let angle = coordinate * inv_freq[axis * per_axis + j];
                    let (sine, cosine) = angle.sin_cos();
                    let low = offset + j;
                    let high = offset + per_axis + j;
                    let u = x[low];
                    let v = x[high];
                    x[low] = u * cosine - v * sine;
                    x[high] = v * cosine + u * sine;
                }
            }
        }
    }
}

/// Scaled embedding lookup: `embedding[id] * sqrt(hidden_size)`
/// (`Gemma4TextScaledWordEmbedding.forward`,
/// `modeling_gemma4.py:1468-1469`; `embed_scale` is
/// `config.hidden_size**0.5`, a Python float, instantiated at
/// `modeling_gemma4.py:1601-1602` and applied after being cast to the
/// embedding weight's dtype -- computed and applied here in f32 throughout,
/// matching this stage's f32 reference).
pub fn gemma4_scaled_embedding(ids: &[u32], embed_weight: &[f32], hidden: usize, out: &mut [f32]) {
    assert_eq!(
        out.len(),
        ids.len() * hidden,
        "out must be ids.len()*hidden"
    );
    assert!(
        embed_weight.len().is_multiple_of(hidden),
        "embed_weight must be a whole number of hidden-sized rows"
    );
    let scale = (hidden as f32).sqrt();
    for (t, &id) in ids.iter().enumerate() {
        let row_start = id as usize * hidden;
        let row = &embed_weight[row_start..row_start + hidden];
        let out_row = &mut out[t * hidden..(t + 1) * hidden];
        for (o, &v) in out_row.iter_mut().zip(row.iter()) {
            *o = v * scale;
        }
    }
}

/// Final logit softcapping: `cap * tanh(logits / cap)`
/// (`modeling_gemma4.py:1889-1892`: divide, tanh, multiply, in that order --
/// mirrored here rather than the algebraically-equivalent `cap *
/// (logits/cap).tanh()` reassociation, though at this stage's tight
/// tolerance the two are indistinguishable).
pub fn gemma4_logit_softcap(logits: &mut [f32], cap: f32) {
    for v in logits.iter_mut() {
        *v /= cap;
        *v = v.tanh();
        *v *= cap;
    }
}

#[cfg(test)]
mod tests;

//! Gemma 4 rotary position embedding tables.
//!
//! The text-layer inverse-frequency table (default and proportional RoPE), its cos/sin
//! expansion, and the vision tower's axial inverse-frequency table. Applying the rotation to a
//! tensor is applied in [`crate::gemma_ops`]; `lattice_inference::model::gemma4_ops` re-exports these functions at
//! their original paths.

/// `angle.sin_cos()` behind one function body, so every RoPE table builder rounds the same way.
///
/// Apple's libm can return a different last bit from `__sincosf_stret` than from separate
/// `sinf`/`cosf` calls for the same argument, and LLVM chooses between the two per call site
/// after inlining. Two tables built by separate loops over the same angles can then differ by
/// one ulp. `inline(never)` keeps a single lowering for every caller in the process.
#[inline(never)]
pub fn rope_sin_cos(angle: f32) -> (f32, f32) {
    angle.sin_cos()
}

/// Build a per-layer-type RoPE inverse-frequency table of length
/// `head_dim / 2`.
///
/// Local (sliding) layers pass `partial_rotary_factor: None` (the
/// `rope_type="default"` path,
/// `Gemma4TextRotaryEmbedding.compute_default_rope_parameters`,
/// `modeling_gemma4.py:1120-1156`): every slot is
/// `1 / theta^(2i / head_dim)`.
///
/// Global (full-attention) layers pass `Some(partial_rotary_factor)` (the
/// `rope_type="proportional"` path, `_compute_proportional_rope_parameters`,
/// `transformers/modeling_rope_utils.py:187-252`): only the first
/// `(partial_rotary_factor * head_dim / 2) as usize` slots are the computed
/// frequency; the rest are **exactly zero**. A zero frequency makes
/// `cos=1, sin=0` for that dimension pair in [`gemma4_rope_cos_sin`] /
/// [`crate::gemma_ops::gemma4_apply_rope`] -- i.e. those dimensions pass
/// through unrotated,
/// rather than being truncated from the tensor. `cos`/`sin` stay
/// `head_dim`-wide either way.
pub fn gemma4_rope_inv_freq(
    head_dim: usize,
    theta: f64,
    partial_rotary_factor: Option<f32>,
) -> Vec<f32> {
    assert!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "head_dim must be a positive even number"
    );
    let half = head_dim / 2;
    let rope_angles = match partial_rotary_factor {
        Some(factor) => ((factor * head_dim as f32) / 2.0) as usize,
        None => half,
    };
    (0..half)
        .map(|i| {
            if i < rope_angles {
                (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Build `[positions.len() * head_dim]` cos/sin tables from an inv_freq
/// table (`Gemma4TextRotaryEmbedding.forward`,
/// `modeling_gemma4.py:1163-1172`: `freqs = position ⊗ inv_freq`,
/// `emb = cat(freqs, freqs)`, `cos = emb.cos()`, `sin = emb.sin()` --
/// `attention_scaling` is `1.0` for both the `"default"` and
/// `"proportional"` RoPE types used here, so it is omitted).
pub fn gemma4_rope_cos_sin(inv_freq: &[f32], positions: &[u32]) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let head_dim = half * 2;
    let mut cos = vec![0f32; positions.len() * head_dim];
    let mut sin = vec![0f32; positions.len() * head_dim];
    for (t, &pos) in positions.iter().enumerate() {
        for i in 0..half {
            let angle = pos as f32 * inv_freq[i];
            let (s, c) = rope_sin_cos(angle);
            cos[t * head_dim + i] = c;
            cos[t * head_dim + half + i] = c;
            sin[t * head_dim + i] = s;
            sin[t * head_dim + half + i] = s;
        }
    }
    (cos, sin)
}

/// Axial 2-D RoPE inverse frequencies for a vision head: the first `head_dim / 4`
/// entries are the axis-0 (x) frequencies and the next `head_dim / 4` are axis-1 (y).
/// With theta 100 and a 64-wide head this is `100^(-2j/32)` for `j = 0..16`, x then y
/// (`Gemma4VisionRotaryEmbedding.compute_axial_rope_parameters`,
/// `modeling_gemma4.py:739-760`, `spatial_dim = head_dim // 2 = 32`).
pub fn vision_axial_rope_inv_freq(head_dim: usize, theta: f64) -> Vec<f32> {
    assert!(
        head_dim > 0 && head_dim.is_multiple_of(4),
        "head_dim must be a positive multiple of 4 for axial RoPE"
    );
    let per_axis = head_dim / 4;
    let mut inv_freq = Vec::with_capacity(per_axis * 2);
    for _ in 0..2 {
        for j in 0..per_axis {
            inv_freq.push((1.0 / theta.powf(2.0 * j as f64 / (2 * per_axis) as f64)) as f32);
        }
    }
    inv_freq
}

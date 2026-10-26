//! Gemma math primitives and named shared-kernel reexports.

#[doc(inline)]
pub use lattice_compute::gemma_ops::{
    gemma4_apply_rope, gemma4_geglu_mlp, gemma4_gelu_tanh, gemma4_qk_norm_v_unscaled,
    gemma4_rms_norm, vision_axial_rope_apply,
};
#[doc(inline)]
pub use lattice_compute::gemma_rope::{
    gemma4_rope_cos_sin, gemma4_rope_inv_freq, vision_axial_rope_inv_freq,
};

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

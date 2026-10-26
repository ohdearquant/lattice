//! Gemma math primitives and named shared-kernel reexports.

#[doc(inline)]
pub use lattice_compute::gemma_ops::{
    gemma4_apply_rope, gemma4_geglu_mlp, gemma4_gelu_tanh, gemma4_logit_softcap,
    gemma4_qk_norm_v_unscaled, gemma4_rms_norm, gemma4_scaled_embedding, vision_axial_rope_apply,
};
#[doc(inline)]
pub use lattice_compute::gemma_rope::{
    gemma4_rope_cos_sin, gemma4_rope_inv_freq, vision_axial_rope_inv_freq,
};

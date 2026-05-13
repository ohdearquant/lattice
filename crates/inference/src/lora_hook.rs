//! LoRA adapter hook for the inference forward pass.
//!
//! Defines a trait that the forward pass calls after each linear projection.
//! This lives in foundation/inference (not platform/tune) so the dependency
//! direction stays correct: platform/tune implements this trait.
//!
//! The default [`NoopLoraHook`] does nothing — zero overhead when no adapter is loaded.

/// **Unstable**: trait for LoRA adapter injection into linear projections.
///
/// The inference forward pass calls `apply()` after each `matmul_bt`.
/// If a LoRA adapter exists for the given (layer, module), it adds:
/// `output += scale * B @ (A @ x)`
pub trait LoraHook: Send + Sync {
    /// **Unstable**: apply LoRA delta to a projection output in-place.
    ///
    /// # Arguments
    /// * `layer_idx` - Transformer layer index (0-based)
    /// * `module` - Projection name. Full-attention layers (GQA): `"q_proj"`, `"k_proj"`,
    ///   `"v_proj"`, `"o_proj"`. Linear-attention layers (GDN): `"in_proj_qkv"`,
    ///   `"in_proj_z"`, `"in_proj_b"`, `"in_proj_a"`, `"out_proj"`.
    ///   MLP (all layers): `"gate_proj"`, `"up_proj"`, `"down_proj"`.
    /// * `x` - Input activation (the same input that was passed to the base projection)
    /// * `output` - Base projection output to modify in-place
    fn apply(&self, layer_idx: usize, module: &str, x: &[f32], output: &mut [f32]);
}

/// No-op implementation. Used when no adapter is loaded.
/// The compiler should inline and eliminate these calls entirely.
pub struct NoopLoraHook;

impl LoraHook for NoopLoraHook {
    #[inline(always)]
    fn apply(&self, _layer_idx: usize, _module: &str, _x: &[f32], _output: &mut [f32]) {}
}

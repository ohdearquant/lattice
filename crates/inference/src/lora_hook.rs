//! LoRA adapter hook for the inference forward pass.
//!
//! Defines a trait that the forward pass calls after each linear projection.
//! This lives in foundation/inference (not platform/tune) so the dependency
//! direction stays correct: platform/tune implements this trait.
//!
//! The default `NoopLoraHook` does nothing — zero overhead when no adapter is loaded.

use crate::model::qwen35_config::Qwen35Config;

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
    ///   BERT: `"query"`, `"key"`, `"value"`, `"attn_output"`, `"ffn_intermediate"`, `"ffn_output"`.
    /// * `x` - Input activation (the same input that was passed to the base projection)
    /// * `output` - Base projection output to modify in-place
    fn apply(&self, layer_idx: usize, module: &str, x: &[f32], output: &mut [f32]);

    /// **Unstable**: self-check this hook's declared rank/shape against a
    /// Qwen3.5 model's geometry before it is installed.
    ///
    /// [`crate::model::qwen35::Qwen35Model::set_lora`] calls this before
    /// swapping the hook in, so a mismatched adapter is rejected instead of
    /// silently corrupting a projection's output prefix (or panicking past a
    /// `debug_assert` in a release build). Default: no-op (trusts the
    /// caller) — real adapters with known geometry (e.g.
    /// `lattice_tune::lora::LoraAdapter`) override it.
    fn validate_against(&self, _config: &Qwen35Config) -> Result<(), String> {
        Ok(())
    }
}

/// No-op implementation. Used when no adapter is loaded.
/// The compiler should inline and eliminate these calls entirely.
pub struct NoopLoraHook;

impl LoraHook for NoopLoraHook {
    #[inline(always)]
    fn apply(&self, _layer_idx: usize, _module: &str, _x: &[f32], _output: &mut [f32]) {}
}

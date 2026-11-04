//! LoRA adapter hook for the inference forward pass.
//!
//! Defines a trait that the forward pass calls after each linear projection.
//! The contract lives in host-side compute so inference and training adapters
//! can share it without making compute depend on either implementation.
//!
//! The default `NoopLoraHook` does nothing — zero overhead when no adapter is loaded.

pub use lattice_lora::{
    GDN_LORA_MODULES, GDN_LORA_MODULES_SERVABLE, LoraProjectionShape, LoraTargetGeometry,
    gdn_lora_module_is_servable,
};

/// **Unstable**: trait for LoRA adapter injection into linear projections.
///
/// The inference forward pass calls `apply()` for each projected row after a
/// `matmul_bt`.
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
    /// * `x` - One input row (the same activation passed to the base projection)
    /// * `output` - The corresponding base projection output row to modify in-place
    fn apply(&self, layer_idx: usize, module: &str, x: &[f32], output: &mut [f32]);

    /// **Unstable**: check this hook's declared projection geometry before installation or use.
    ///
    /// Model installers and hooked scoring entry points call this before changing
    /// state or running a forward pass. Hooks with known dimensions should reject
    /// invalid layer indices, unknown modules, and input/output width mismatches.
    /// The default accepts the caller's geometry without checking it.
    ///
    /// This may be called more than once for one request. Implement it as a
    /// repeatable read of declared dimensions, without one-shot side effects.
    fn validate_geometry(&self, _geometry: &dyn LoraTargetGeometry) -> Result<(), String> {
        Ok(())
    }

    /// **Unstable**: whether this hook has anything to apply for
    /// `(layer_idx, module)`.
    ///
    /// `apply_lora_rows` calls this once per projection, before its
    /// per-row loop, so a hook with nothing to do for this projection (the
    /// default no-adapter case) skips the loop — and the one virtual call
    /// per token row it would otherwise cost — entirely. Default: `true`
    /// (assume active; correct but not optimized for hooks that don't
    /// override it). [`NoopLoraHook`] overrides this to `false`.
    fn is_active(&self, _layer_idx: usize, _module: &str) -> bool {
        true
    }
}

/// Apply an active LoRA hook to each input/output row pair of one projection.
///
/// Geometry is checked before activity, including for an inactive hook. Activity is
/// resolved once per projection; active hooks receive contiguous rows in input order.
///
/// # Panics
/// Panics if either row width is zero, either slice contains an incomplete row, or
/// the input and output row counts differ.
pub fn apply_lora_rows(
    lora: &dyn LoraHook,
    layer_idx: usize,
    module: &str,
    input: &[f32],
    output: &mut [f32],
    input_row_width: usize,
    output_row_width: usize,
) {
    assert!(input_row_width > 0, "LoRA input row width must be non-zero");
    assert!(
        output_row_width > 0,
        "LoRA output row width must be non-zero"
    );
    assert_eq!(
        input.len() % input_row_width,
        0,
        "LoRA input must contain complete rows"
    );
    assert_eq!(
        output.len() % output_row_width,
        0,
        "LoRA output must contain complete rows"
    );
    assert_eq!(
        input.len() / input_row_width,
        output.len() / output_row_width,
        "LoRA input and output row counts must match"
    );

    // Resolved once per projection rather than once per row: on the default
    // (no-adapter) path this skips straight past the per-row loop below
    // instead of paying one virtual dispatch per token.
    if !lora.is_active(layer_idx, module) {
        return;
    }

    for (input_row, output_row) in input
        .chunks_exact(input_row_width)
        .zip(output.chunks_exact_mut(output_row_width))
    {
        lora.apply(layer_idx, module, input_row, output_row);
    }
}

/// No-op implementation. Used when no adapter is loaded.
/// The compiler should inline and eliminate these calls entirely.
pub struct NoopLoraHook;

impl LoraHook for NoopLoraHook {
    #[inline(always)]
    fn apply(&self, _layer_idx: usize, _module: &str, _x: &[f32], _output: &mut [f32]) {}

    #[inline(always)]
    fn is_active(&self, _layer_idx: usize, _module: &str) -> bool {
        false
    }
}

#[cfg(test)]
mod tests;

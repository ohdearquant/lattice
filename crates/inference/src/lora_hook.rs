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

    /// **Unstable**: self-check this hook's declared projection geometry
    /// against a BERT cross-encoder model's dimensions before it is used
    /// for hooked scoring.
    ///
    /// [`crate::model::cross_encoder::CrossEncoderModel::score_with_hook`]
    /// calls this before the forward pass (and before any row is sliced),
    /// so a mismatched adapter is rejected with a recoverable error instead
    /// of `apply_lora` slicing `output[..lora.d_out]` out of bounds past a
    /// `debug_assert` that release builds compile out. Default: no-op
    /// (trusts the caller) — real adapters with known geometry (e.g.
    /// `lattice_tune::lora::LoraAdapter`) override it.
    fn validate_against_bert(
        &self,
        _num_hidden_layers: usize,
        _hidden_size: usize,
        _intermediate_size: usize,
    ) -> Result<(), String> {
        Ok(())
    }
}

pub(crate) fn apply_lora_rows(
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RowSensitiveHook {
        calls: AtomicUsize,
    }

    impl LoraHook for RowSensitiveHook {
        fn apply(&self, _layer_idx: usize, _module: &str, x: &[f32], output: &mut [f32]) {
            assert_eq!(x.len(), 2);
            assert_eq!(output.len(), 3);
            self.calls.fetch_add(1, Ordering::Relaxed);
            output.fill(x[0]);
        }
    }

    #[test]
    fn applies_lora_to_each_flattened_token_row() {
        let hook = RowSensitiveHook {
            calls: AtomicUsize::new(0),
        };
        let input = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let mut output = [0.0; 9];

        apply_lora_rows(&hook, 0, "projection", &input, &mut output, 2, 3);

        assert_eq!(hook.calls.load(Ordering::Relaxed), 3);
        assert_eq!(output, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
    }
}

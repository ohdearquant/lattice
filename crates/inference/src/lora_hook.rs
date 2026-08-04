//! LoRA adapter hook for the inference forward pass.
//!
//! Defines a trait that the forward pass calls after each linear projection.
//! This lives in foundation/inference (not platform/tune) so the dependency
//! direction stays correct: platform/tune implements this trait.
//!
//! The default `NoopLoraHook` does nothing — zero overhead when no adapter is loaded.

use crate::model::qwen35_config::Qwen35Config;

/// Input and output dimensions for one LoRA-targetable linear projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoraProjectionShape {
    /// Width of the activation consumed by the projection.
    pub d_in: usize,
    /// Width of the projection output.
    pub d_out: usize,
}

/// Return the LoRA projection shape for a Qwen3.5 layer and module.
///
/// Full-attention modules are valid only on full-attention layers, Gated
/// DeltaNet modules are valid only on linear-attention layers, and MLP modules
/// are valid on both. The returned geometry includes `in_proj_a` and
/// `in_proj_b`; a backend that cannot apply those projections must enforce that
/// capability restriction separately.
pub fn qwen35_projection_shape(
    config: &Qwen35Config,
    layer_idx: usize,
    module: &str,
) -> Result<LoraProjectionShape, String> {
    if layer_idx >= config.num_hidden_layers {
        return Err(format!(
            "layer {layer_idx} is out of range for Qwen3.5 model with {} layers",
            config.num_hidden_layers
        ));
    }

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let is_full = config.is_full_attention(layer_idx);
    let (d_in, d_out) = match (module, is_full) {
        ("q_proj", true) => (hidden, 2 * config.full_q_dim()),
        ("k_proj", true) => (hidden, config.full_kv_dim()),
        ("v_proj", true) => (hidden, config.full_kv_dim()),
        ("o_proj", true) => (config.full_q_dim(), hidden),
        ("in_proj_qkv", false) => (hidden, config.linear_qkv_dim()),
        ("in_proj_z", false) => (hidden, config.linear_output_dim()),
        ("in_proj_b" | "in_proj_a", false) => (hidden, config.linear_num_value_heads()),
        ("out_proj", false) => (config.linear_output_dim(), hidden),
        ("gate_proj", _) | ("up_proj", _) => (hidden, intermediate),
        ("down_proj", _) => (intermediate, hidden),
        ("q_proj" | "k_proj" | "v_proj" | "o_proj", false) => {
            return Err(format!(
                "module '{module}' is a full-attention projection but layer {layer_idx} is GDN"
            ));
        }
        ("in_proj_qkv" | "in_proj_z" | "in_proj_b" | "in_proj_a" | "out_proj", true) => {
            return Err(format!(
                "module '{module}' is a GDN projection but layer {layer_idx} is full-attention"
            ));
        }
        _ => return Err(format!("unknown LoRA module '{module}'")),
    };

    Ok(LoraProjectionShape { d_in, d_out })
}

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
    /// This is the hook's OWN check, and the default below returns `Ok(())`:
    /// a hook that does not override it is trusted, and nothing else on the
    /// scoring path re-checks its geometry.
    ///
    /// [`crate::model::cross_encoder::CrossEncoderModel::score_with_hook`]
    /// and `score_batch_with_hook` call this before the forward pass (and
    /// before any row is sliced) and map an `Err` to
    /// [`crate::error::InferenceError::InvalidInput`]. So it is an
    /// OVERRIDING implementation — `lattice_tune::lora::LoraAdapter` is the
    /// one in this workspace — that makes a mismatched adapter fail with a
    /// recoverable error instead of `apply_lora` slicing
    /// `output[..lora.d_out]` out of bounds past a `debug_assert` that
    /// release builds compile out.
    ///
    /// Implement it for any hook whose geometry is known. Leaving it at the
    /// default opts that hook out of the rejection, not into it.
    ///
    /// This may be called more than once for a single request: a batch of N
    /// documents calls it N+1 times, once at the batch boundary and once per
    /// document. Implement it as a repeatable read of declared dimensions, not
    /// as a one-shot operation with side effects.
    fn validate_against_bert(
        &self,
        _num_hidden_layers: usize,
        _hidden_size: usize,
        _intermediate_size: usize,
    ) -> Result<(), String> {
        Ok(())
    }

    /// **Unstable**: whether this hook has anything to apply for
    /// `(layer_idx, module)`.
    ///
    /// [`apply_lora_rows`] calls this once per projection, before its
    /// per-row loop, so a hook with nothing to do for this projection (the
    /// default no-adapter case) skips the loop — and the one virtual call
    /// per token row it would otherwise cost — entirely. Default: `true`
    /// (assume active; correct but not optimized for hooks that don't
    /// override it). [`NoopLoraHook`] overrides this to `false`.
    fn is_active(&self, _layer_idx: usize, _module: &str) -> bool {
        true
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
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn qwen35_projection_shape_covers_full_attention_modules() {
        let config = Qwen35Config::qwen35_0_8b();
        let full_layer = 3;
        let cases = [
            (
                "q_proj",
                LoraProjectionShape {
                    d_in: 1024,
                    d_out: 4096,
                },
            ),
            (
                "k_proj",
                LoraProjectionShape {
                    d_in: 1024,
                    d_out: 512,
                },
            ),
            (
                "v_proj",
                LoraProjectionShape {
                    d_in: 1024,
                    d_out: 512,
                },
            ),
            (
                "o_proj",
                LoraProjectionShape {
                    d_in: 2048,
                    d_out: 1024,
                },
            ),
        ];

        for (module, expected) in cases {
            assert_eq!(
                qwen35_projection_shape(&config, full_layer, module),
                Ok(expected),
                "unexpected shape for {module}"
            );
        }
    }

    #[test]
    fn qwen35_projection_shape_covers_gdn_modules_including_alpha_and_beta() {
        let config = Qwen35Config::qwen35_0_8b();
        let gdn_layer = 0;
        let cases = [
            (
                "in_proj_qkv",
                LoraProjectionShape {
                    d_in: 1024,
                    d_out: 6144,
                },
            ),
            (
                "in_proj_z",
                LoraProjectionShape {
                    d_in: 1024,
                    d_out: 2048,
                },
            ),
            (
                "in_proj_b",
                LoraProjectionShape {
                    d_in: 1024,
                    d_out: 16,
                },
            ),
            (
                "in_proj_a",
                LoraProjectionShape {
                    d_in: 1024,
                    d_out: 16,
                },
            ),
            (
                "out_proj",
                LoraProjectionShape {
                    d_in: 2048,
                    d_out: 1024,
                },
            ),
        ];

        for (module, expected) in cases {
            assert_eq!(
                qwen35_projection_shape(&config, gdn_layer, module),
                Ok(expected),
                "unexpected shape for {module}"
            );
        }
    }

    #[test]
    fn qwen35_projection_shape_uses_value_heads_for_gdn_alpha_and_beta() {
        let config = Qwen35Config::qwen36_35b_a3b();
        assert_eq!(config.linear_num_key_heads, 16);
        assert_eq!(config.linear_num_value_heads(), 32);

        for module in ["in_proj_b", "in_proj_a"] {
            assert_eq!(
                qwen35_projection_shape(&config, 0, module),
                Ok(LoraProjectionShape {
                    d_in: config.hidden_size,
                    d_out: 32,
                }),
                "{module} must use the asymmetric value-head count"
            );
        }
    }

    #[test]
    fn qwen35_projection_shape_accepts_mlp_modules_on_both_layer_types() {
        let config = Qwen35Config::qwen35_0_8b();
        let cases = [
            (
                "gate_proj",
                LoraProjectionShape {
                    d_in: 1024,
                    d_out: 3584,
                },
            ),
            (
                "up_proj",
                LoraProjectionShape {
                    d_in: 1024,
                    d_out: 3584,
                },
            ),
            (
                "down_proj",
                LoraProjectionShape {
                    d_in: 3584,
                    d_out: 1024,
                },
            ),
        ];

        for layer_idx in [0, 3] {
            for (module, expected) in cases {
                assert_eq!(
                    qwen35_projection_shape(&config, layer_idx, module),
                    Ok(expected),
                    "unexpected shape for {module} on layer {layer_idx}"
                );
            }
        }
    }

    #[test]
    fn qwen35_projection_shape_rejects_wrong_layer_types() {
        let config = Qwen35Config::qwen35_0_8b();

        assert_eq!(
            qwen35_projection_shape(&config, 0, "q_proj"),
            Err("module 'q_proj' is a full-attention projection but layer 0 is GDN".to_string())
        );
        assert_eq!(
            qwen35_projection_shape(&config, 3, "in_proj_qkv"),
            Err(
                "module 'in_proj_qkv' is a GDN projection but layer 3 is full-attention"
                    .to_string()
            )
        );
    }

    #[test]
    fn qwen35_projection_shape_rejects_unknown_and_out_of_range_targets() {
        let config = Qwen35Config::qwen35_0_8b();

        assert_eq!(
            qwen35_projection_shape(&config, 3, "q_porj"),
            Err("unknown LoRA module 'q_porj'".to_string())
        );
        assert_eq!(
            qwen35_projection_shape(&config, config.num_hidden_layers, "q_proj"),
            Err("layer 24 is out of range for Qwen3.5 model with 24 layers".to_string())
        );
    }

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

    struct InactiveHook {
        calls: AtomicUsize,
    }

    impl LoraHook for InactiveHook {
        fn apply(&self, _layer_idx: usize, _module: &str, _x: &[f32], _output: &mut [f32]) {
            self.calls.fetch_add(1, Ordering::Relaxed);
        }

        fn is_active(&self, _layer_idx: usize, _module: &str) -> bool {
            false
        }
    }

    /// A hook that reports `is_active == false` for a projection must never
    /// have `apply` dispatched for any row of that projection: `apply_lora_rows`
    /// checks activity once, before the per-row loop, not per row.
    #[test]
    fn skips_the_per_row_loop_entirely_when_the_hook_is_inactive() {
        let hook = InactiveHook {
            calls: AtomicUsize::new(0),
        };
        let input = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let mut output = [7.0; 9];

        apply_lora_rows(&hook, 0, "projection", &input, &mut output, 2, 3);

        assert_eq!(
            hook.calls.load(Ordering::Relaxed),
            0,
            "apply must not be called for any row when is_active is false"
        );
        assert_eq!(output, [7.0; 9], "output must be untouched");
    }

    /// `NoopLoraHook` (the default when no adapter is loaded) must report
    /// itself inactive for every projection, so the default (no-adapter)
    /// path stays off the per-row virtual-dispatch cost.
    #[test]
    fn noop_hook_reports_inactive_for_any_projection() {
        let hook = NoopLoraHook;
        assert!(!hook.is_active(0, "query"));
        assert!(!hook.is_active(41, "ffn_output"));
    }
}

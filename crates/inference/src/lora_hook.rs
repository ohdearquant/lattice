//! LoRA adapter hook for the inference forward pass.
//!
//! Defines a trait that the forward pass calls after each linear projection.
//! This lives in foundation/inference (not platform/tune) so the dependency
//! direction stays correct: platform/tune implements this trait.
//!
//! The default `NoopLoraHook` does nothing — zero overhead when no adapter is loaded.

use crate::model::qwen35_config::Qwen35Config;

/// Every Gated DeltaNet projection a LoRA adapter can target, in the order the
/// trainers emit them.
///
/// This is the geometry set: [`qwen35_projection_shape`] returns a shape for all
/// five. Whether a given backend can APPLY one is a separate question, answered
/// by [`GDN_LORA_MODULES_SERVABLE`].
pub const GDN_LORA_MODULES: [&str; 5] = [
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
];

/// The Gated DeltaNet projections the Metal forward can actually apply.
///
/// `in_proj_a` and `in_proj_b` are absent on purpose. Both are consumed inside
/// the fused GDN recurrence kernel, which reads the base weights directly, so
/// there is no post-projection row for a LoRA delta to be added to. That is a
/// capability statement about the kernel, not a validation slip.
///
/// THIS EXISTS TO BE THE ONLY COPY. The trainers pick a module set and the Metal
/// loader refuses one; when those two lists are written independently they drift,
/// and the drift is silent until load time -- an adapter trains to completion and
/// then cannot be served, with the loss having been minimised for a delta the
/// server will never apply. A trainer that selects from this constant and a
/// loader that refuses by this constant cannot disagree.
///
/// Widening this set is a kernel change, not an edit here: see the doc comment on
/// `load_lora_adapter` in the Metal forward.
pub const GDN_LORA_MODULES_SERVABLE: [&str; 3] = ["in_proj_qkv", "in_proj_z", "out_proj"];

/// Whether `module` is a GDN projection the Metal forward can apply.
///
/// False for a non-GDN module name too, so callers that mean "is this GDN module
/// servable" must already know they hold a GDN module name; [`GDN_LORA_MODULES`]
/// is the membership test for that.
#[must_use]
pub fn gdn_lora_module_is_servable(module: &str) -> bool {
    GDN_LORA_MODULES_SERVABLE.contains(&module)
}

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
mod tests;

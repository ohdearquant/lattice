//! The projection geometry a LoRA adapter is checked against, and the Gated DeltaNet
//! projection names shared by the trainers and the Metal loader.

/// Every Gated DeltaNet projection a LoRA adapter can target, in the order the
/// trainers emit them.
///
/// This is the geometry set: `lattice_inference::lora_hook::qwen35_projection_shape`
/// returns a shape for all
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

/// **Unstable**: the projection geometry a LoRA hook is checked against before it is installed.
pub trait LoraTargetGeometry {
    /// Number of layers a hook may target; valid layer indices are `0..num_layers()`.
    fn num_layers(&self) -> usize;
    /// Shape of `module` on `layer_idx`, or an error saying why that pair is not a projection of this model.
    fn projection_shape(
        &self,
        layer_idx: usize,
        module: &str,
    ) -> Result<LoraProjectionShape, String>;
}

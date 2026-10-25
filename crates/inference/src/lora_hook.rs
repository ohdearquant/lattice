//! LoRA adapter hooks, re-exported from `lattice-compute`.
//!
//! Named exports retain the existing public surface; the row helper stays crate-private.

pub(crate) use lattice_compute::lora_hook::apply_lora_rows;
#[doc(inline)]
pub use lattice_compute::lora_hook::{
    GDN_LORA_MODULES, GDN_LORA_MODULES_SERVABLE, LoraHook, LoraProjectionShape, NoopLoraHook,
    gdn_lora_module_is_servable, qwen35_projection_shape,
};

#[cfg(test)]
mod tests {
    #[test]
    fn row_helper_is_the_same_function_item() {
        fn same_function<F>(_: F, _: F) {}
        same_function(
            super::apply_lora_rows,
            lattice_compute::lora_hook::apply_lora_rows,
        );
    }
}

//! Weight storage for Lattice inference: safetensors parsing and validation, the f16, f32, q3, q4 and
//! q8 weight formats, trusted memory mapping, bounded file reads, and model-directory format detection.
//!
//! `lattice-inference` re-exports the supported items at their original `lattice_inference::weights`
//! and `lattice_inference::model_format` paths.

// Weight loaders and quantizers take many shape and buffer arguments, and loops index several
// arrays at once; the same allowances as the inference crate.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

pub mod bounded_config;
pub mod bounded_read;
pub mod consumed;
pub mod f16_encode;
pub mod f16_weights;
pub mod f32_weights;
pub mod half_bits;
pub mod ingress;
pub mod mmap_trust;
pub mod model_format;
/// Experimental CPU decoder for MXFP4 blocks.
#[doc(hidden)]
pub mod mxfp4;
/// Experimental CPU decoder for row-major NVFP4 blocks.
#[doc(hidden)]
pub mod nvfp4;
#[cfg_attr(not(test), allow(dead_code))]
pub mod prepared_safetensors;
#[cfg_attr(not(test), allow(dead_code))]
pub mod prepared_safetensors_inventory;
pub mod q3_weights;
pub mod q4_weights;
pub mod q8_weights;
pub mod safetensors_layout;

// Re-export from f32_weights (the primary/base weight types)
pub use self::f32_weights::*;

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod test_fixtures {
    pub mod bert_namespace;
}

/// Storage glue used by the `lattice-inference` facade.
///
/// Not part of the supported API. The facade calls these free functions by their full path; they are
/// not re-exported by that facade.
#[doc(hidden)]
pub mod __internal {
    use crate::f32_weights::SafetensorsFile;
    use lattice_inference_core::InferenceError;
    use std::fs::File;
    use std::path::Path;

    /// Parse safetensors from an already-open file without reopening its path.
    pub fn safetensors_from_open_file(
        file: File,
        path: &Path,
    ) -> Result<SafetensorsFile, InferenceError> {
        SafetensorsFile::from_open_file(file, path)
    }

    /// Read a BF16 tensor payload without widening or copying its backing bytes.
    pub fn safetensors_bf16_payload<'a>(
        file: &'a SafetensorsFile,
        name: &str,
    ) -> Result<&'a [u8], InferenceError> {
        file.bf16_payload(name)
    }

    /// Count tensors currently holding a cached f32 conversion.
    pub fn safetensors_cached_f32_tensor_count(file: &SafetensorsFile) -> usize {
        file.cached_f32_tensor_count()
    }
}

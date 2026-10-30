//! **Compatibility path.** Re-exported from `lattice-model-qwen3-embedding` so that code written against the pre-split `lattice-inference`
//! keeps compiling. Not a stable API; removed at 1.0. Import from `lattice-model-qwen3-embedding` directly.

#[cfg(feature = "wgpu-gpu")]
pub use lattice_model_qwen3_embedding::gpu::{
    GpuForwardError, GpuModelState, GpuRuntimeConfig, Qwen3Config, Qwen3LayerWeights, Qwen3Weights,
};

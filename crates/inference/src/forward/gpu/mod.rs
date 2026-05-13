//! GPU forward path for a Qwen3-style decoder using `wgpu` compute pipelines.
//!
//! Design notes:
//!   - Decoder weights live in persistent GPU buffers created once in `GpuModelState::new()`.
//!   - Embedding gather is performed on CPU and uploaded as one dense hidden-state
//!     buffer per call (sparse embedding lookup stays host-side).
//!   - `runtime.max_seq_len` is a practical attention allocation cap. This implementation
//!     materializes an explicit `[num_heads, seq, seq]` score buffer (not flash-attention).

#[cfg(feature = "wgpu-gpu")]
mod inner;

#[cfg(feature = "wgpu-gpu")]
pub use inner::{
    GpuForwardError, GpuModelState, GpuRuntimeConfig, Qwen3Config, Qwen3LayerWeights, Qwen3Weights,
};

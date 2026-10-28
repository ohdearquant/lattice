//! GPU forward path for a Qwen3-style decoder using `wgpu` compute pipelines.
//!
//! Design notes:
//!   - Decoder weights live in persistent GPU buffers created once in `GpuModelState::new()`.
//!   - Embedding gather is performed on CPU and uploaded as one dense hidden-state
//!     buffer per call (sparse embedding lookup stays host-side).
//!   - `runtime.max_seq_len` is a practical attention allocation cap. This implementation
//!     materializes an explicit `[num_heads, seq, seq]` score buffer (not flash-attention).
//!
//! **Compatibility path.** This module keeps the pre-split `lattice-inference` paths compiling; the
//! code it names lives in the crates that own it. These low-level paths are not a stable API and
//! will be removed at 1.0.

#[cfg(feature = "wgpu-gpu")]
mod inner;

#[cfg(feature = "wgpu-gpu")]
pub use inner::{
    GpuForwardError, GpuModelState, GpuRuntimeConfig, Qwen3Config, Qwen3LayerWeights, Qwen3Weights,
};

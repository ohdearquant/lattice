//! Inner WGPU forward module index and API-type re-exports.
//!
//! **Compatibility path.** This module keeps the pre-split `lattice-inference` paths compiling; the
//! code it names lives in the crates that own it. These low-level paths are not a stable API and
//! will be removed at 1.0.
mod api;
mod bind_groups;
mod buffers;
mod dims;
mod dispatch;
mod params;
mod pipelines;
mod shaders;
mod state;
mod util;

#[cfg(test)]
mod tests;

pub use api::{GpuForwardError, GpuRuntimeConfig, Qwen3Config, Qwen3LayerWeights, Qwen3Weights};
pub use state::GpuModelState;

#[cfg(test)]
pub use util::build_rope_tables;

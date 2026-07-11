//! Weight module index for f16, f32, q4, and q8 weights, with f32 weight re-exports.
pub mod f16_weights;
pub mod f32_weights;
pub(crate) mod ingress;
pub mod q4_weights;
pub mod q8_weights;

// Re-export from f32_weights (the primary/base weight types)
pub use self::f32_weights::*;

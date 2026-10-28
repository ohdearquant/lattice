//! Reverse-mode gradients (VJPs) for LoRA training on Lattice transformer layers.
//!
//! - [`ops`]: linear, LoRA, RMSNorm, SwiGLU and RoPE vector-Jacobian products.
//! - [`tape`]: forward helpers that keep the activations the backward pass needs.
//! - [`attention_gqa`]: gated grouped-query attention forward with cache and its backward.
//! - [`gdn`]: GatedDeltaNet forward with saved state and its backward, including LoRA gradients
//!   for the five GDN projections.
//! - [`gradcheck`]: a tiny all-GQA model and a finite-difference gradient check over the chain.
//! - `dispatch`: test helpers, built for this crate's tests and under the `test-utils` feature.
//!
//! The CPU kernels come from `lattice-compute` and the Qwen3.5 configuration from `lattice-arch`.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod attention_gqa;
#[cfg(any(test, feature = "test-utils"))]
pub mod dispatch;
pub mod gdn;
pub mod gradcheck;
pub mod ops;
pub mod tape;

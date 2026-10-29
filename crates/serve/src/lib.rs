//! HTTP serving for lattice: the `lattice` CLI, the `lattice_serve` daemon and the serving
//! runtimes they share. Built on `lattice-inference`.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

/// Shared HTTP serving contract (error envelope, `finish_reason`, `max_tokens`
/// zero-rejection, `/v1/models` body) consumed by both the `lattice` unified
/// server and the `lattice_serve` daemon binaries (ADR-080 cluster C2).
/// Requires the `serve` feature (axum/tokio/futures).
pub mod serve;
/// CPU serving runtimes for `lattice serve`: Gemma 4 E2B text and the traced
/// Qwen3.5 CPU entries. Not a stable API.
#[doc(hidden)]
pub mod serving_cpu;
/// Shared request-local execution host for CPU-backed `lattice serve` routes.
/// Not a stable API.
#[doc(hidden)]
pub mod serving_cpu_host;
/// Pooled embedding service interfaces used by serving entry points.
#[doc(hidden)]
pub mod serving_embeddings;
/// Builds the serving worker's runtime and request preparation for a loaded
/// model. Not a stable API.
#[doc(hidden)]
pub mod serving_factory;
/// Shared request facade for the HTTP serving binaries. Not a stable API.
#[doc(hidden)]
pub mod serving_host;
mod serving_preparation;
/// Selects serving providers from bounded checkpoint metadata.
/// Requires the `serve` feature.
#[doc(hidden)]
pub mod serving_provider;
pub(crate) mod serving_runtime;
pub(crate) mod serving_runtime_contract;
/// Dedicated-thread host for thread-affine serving runtimes. Not a stable API.
#[doc(hidden)]
pub mod serving_worker;

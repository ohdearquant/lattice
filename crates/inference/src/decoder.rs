//! Shared driver observation and family measurement compatibility paths.
//!
//! **Compatibility re-exports.** The items this module re-exports from `lattice-arch` and
//! `lattice-model-qwen35` keep code written against the pre-split `lattice-inference` compiling.
//! Those low-level re-exports are not a stable API and will be removed at 1.0; import them from the
//! owning crate directly.

#[cfg(all(feature = "qwen35", feature = "bench-internals"))]
#[doc(inline)]
pub use lattice_model_qwen35::decoder::bench_support;

#[cfg(feature = "serving-support")]
pub(crate) mod driver {
    pub(crate) use lattice_arch::decoder::driver::DriverTrace;
}

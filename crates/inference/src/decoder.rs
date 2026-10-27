//! Shared driver observation and family measurement compatibility paths.

#[cfg(all(feature = "qwen35", feature = "bench-internals"))]
#[doc(inline)]
pub use lattice_model_qwen35::decoder::bench_support;

#[cfg(feature = "serving-support")]
pub(crate) mod driver {
    pub(crate) use lattice_arch::decoder::driver::DriverTrace;
}

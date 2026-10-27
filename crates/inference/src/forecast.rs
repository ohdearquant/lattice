//! Time-series forecasting: shared context preparation and native forecasting models,
//! re-exported from `lattice-forecast`.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-forecast` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-forecast` directly.

#[doc(inline)]
pub use lattice_forecast::{
    Chronos2, Chronos2Config, Chronos2Series, PredictOptions, QuantileForecast, Reverso,
    ReversoConfig, Toto2, Toto2Config, Toto2Options, Toto2Variate,
};
#[doc(inline)]
pub use lattice_forecast::{chronos2, prep, reverso, toto2};

//! Time-series forecasting: shared context preparation and native forecasting models.
//!
//! Models take a univariate history and return point forecasts. Context preparation (missing
//! values, padding, striding, resampling) lives in [`prep`] so every model prepares its input the
//! same way its reference evaluation does.

mod fft;
pub mod prep;
pub mod reverso;

pub use reverso::{PredictOptions, Reverso, ReversoConfig};

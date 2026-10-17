//! Time-series forecasting: shared context preparation and native forecasting models.
//!
//! Reverso returns point forecasts of a univariate history; Chronos-2 and Toto 2.0 return quantile
//! forecasts and can forecast related series jointly. Reverso's context preparation (missing values,
//! padding, striding, resampling) lives in [`prep`] so it matches its reference evaluation;
//! Chronos-2 and Toto 2.0 prepare their own input inside the model, as their references do.

pub mod chronos2;
mod fft;
pub mod prep;
pub mod reverso;
pub mod toto2;

pub use chronos2::{Chronos2, Chronos2Config, Chronos2Series, QuantileForecast};
pub use reverso::{PredictOptions, Reverso, ReversoConfig};
pub use toto2::{Toto2, Toto2Config, Toto2Options, Toto2Variate};

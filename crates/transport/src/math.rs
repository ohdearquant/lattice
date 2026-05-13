//! Small math shim for transport module.
//!
//! Delegates to platform `std` implementations. Kept as a thin wrapper to
//! centralize all floating-point operations used by the Sinkhorn solvers.

#[inline]
pub fn abs(x: f32) -> f32 {
    x.abs()
}

#[inline]
pub fn exp(x: f32) -> f32 {
    x.exp()
}

#[inline]
pub fn ln(x: f32) -> f32 {
    x.ln()
}

#[inline]
pub fn log1p(x: f32) -> f32 {
    x.ln_1p()
}

#[inline]
pub fn sqrt(x: f32) -> f32 {
    x.sqrt()
}

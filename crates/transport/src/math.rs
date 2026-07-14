//! Small math shim for transport floating-point operations.

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

use super::api::checked_mul;
use super::api::{GpuForwardError, Result};

pub(super) fn build_rope_tables(
    head_dim: usize,
    max_seq_len: usize,
    theta: f64,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut cos = Vec::with_capacity(max_seq_len * half_dim);
    let mut sin = Vec::with_capacity(max_seq_len * half_dim);
    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

pub(super) fn bytes_f32(len: usize) -> Result<u64> {
    let bytes = checked_mul(len, std::mem::size_of::<f32>(), "byte count")?;
    u64::try_from(bytes)
        .map_err(|_| GpuForwardError::Limit("buffer size does not fit in u64".to_string()))
}

pub(super) fn to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| GpuForwardError::Limit(format!("{name}={value} does not fit in u32")))
}

#[inline]
pub(super) fn ceil_div_u32(x: u32, y: u32) -> u32 {
    (x + y - 1) / y
}

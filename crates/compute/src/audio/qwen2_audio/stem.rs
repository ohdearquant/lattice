//! Dense, padded Qwen2-Audio convolution stem over channel-major f32 data.

use lattice_inference_core::InferenceError;

const KERNEL: usize = 3;
const MAX_ELEMENTS: usize = 16_000_000;
const MAX_MACS: u64 = 10_000_000_000;

/// **Unstable**: synthetic stem results, both channel-major.
pub struct StemOutput {
    /// First convolution and GELU, `[hidden, frames]`.
    pub first: Vec<f32>,
    /// Second convolution and GELU, `[hidden, ceil(frames / 2)]`.
    pub second: Vec<f32>,
    /// Output time dimension after the second convolution.
    pub output_frames: usize,
}

/// **Unstable**: explicitly supplied dense convolution parameters.
pub struct StemWeights<'a> {
    /// Conv1 `[hidden, input_channels, 3]`.
    pub conv1: &'a [f32],
    /// Conv1 bias `[hidden]`.
    pub bias1: &'a [f32],
    /// Conv2 `[hidden, hidden, 3]`.
    pub conv2: &'a [f32],
    /// Conv2 bias `[hidden]`.
    pub bias2: &'a [f32],
}

/// Apply the two padded convolutions and error-function GELUs.
///
/// `input` is channel-major `[input_channels, frames]`; weights are row-major
/// `[output_channel, input_channel, kernel]`. This is a bounded scalar CPU
/// baseline, not a model loader or a performance claim.
///
/// # Errors
///
/// Rejects empty dimensions, invalid lengths, non-finite values, overflow,
/// excessive work/allocation, or non-finite computed values.
pub fn run(
    input: &[f32],
    input_channels: usize,
    frames: usize,
    hidden: usize,
    weights: &StemWeights<'_>,
) -> Result<StemOutput, InferenceError> {
    let invalid = |s: &str| InferenceError::InvalidInput(s.to_owned());
    if input_channels == 0 || frames == 0 || hidden == 0 {
        return Err(invalid("audio stem dimensions must be nonzero"));
    }
    let product = |a: usize, b: usize| {
        a.checked_mul(b)
            .filter(|&n| n <= MAX_ELEMENTS)
            .ok_or_else(|| invalid("audio stem shape exceeds bounds"))
    };
    let in_len = product(input_channels, frames)?;
    let first_len = product(hidden, frames)?;
    let out_frames = frames.div_ceil(2);
    let second_len = product(hidden, out_frames)?;
    let w1_len = product(product(hidden, input_channels)?, KERNEL)?;
    let w2_len = product(product(hidden, hidden)?, KERNEL)?;
    if input.len() != in_len
        || weights.conv1.len() != w1_len
        || weights.conv2.len() != w2_len
        || weights.bias1.len() != hidden
        || weights.bias2.len() != hidden
    {
        return Err(invalid("audio stem tensor shape mismatch"));
    }
    if [
        input,
        weights.conv1,
        weights.conv2,
        weights.bias1,
        weights.bias2,
    ]
    .into_iter()
    .flatten()
    .any(|x| !x.is_finite())
    {
        return Err(invalid("audio stem input or weights are non-finite"));
    }
    let m1 = u64::try_from(w1_len)
        .ok()
        .and_then(|n| n.checked_mul(frames as u64));
    let m2 = u64::try_from(w2_len)
        .ok()
        .and_then(|n| n.checked_mul(out_frames as u64));
    if m1
        .zip(m2)
        .and_then(|(a, b)| a.checked_add(b))
        .is_none_or(|n| n > MAX_MACS)
    {
        return Err(invalid("audio stem exceeds work budget"));
    }
    let mut first = allocate(first_len)?;
    let mut second = allocate(second_len)?;
    conv_gelu(
        input,
        input_channels,
        frames,
        hidden,
        weights.conv1,
        weights.bias1,
        1,
        &mut first,
    )?;
    conv_gelu(
        &first,
        hidden,
        frames,
        hidden,
        weights.conv2,
        weights.bias2,
        2,
        &mut second,
    )?;
    Ok(StemOutput {
        first,
        second,
        output_frames: out_frames,
    })
}

fn allocate(n: usize) -> Result<Vec<f32>, InferenceError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(n)
        .map_err(|_| InferenceError::InvalidInput("audio stem allocation failed".into()))?;
    output.resize(n, 0.0);
    Ok(output)
}

fn exact_gelu(x: f32) -> f32 {
    (0.5 * f64::from(x) * libm::erfc(-f64::from(x) * std::f64::consts::FRAC_1_SQRT_2)) as f32
}

#[allow(clippy::too_many_arguments)]
fn conv_gelu(
    input: &[f32],
    input_channels: usize,
    frames: usize,
    hidden: usize,
    weights: &[f32],
    bias: &[f32],
    stride: usize,
    output: &mut [f32],
) -> Result<(), InferenceError> {
    let out_frames = frames.div_ceil(stride);
    for out_ch in 0..hidden {
        for t in 0..out_frames {
            let mut sum = bias[out_ch];
            for in_ch in 0..input_channels {
                for k in 0..KERNEL {
                    let padded_pos = t * stride + k;
                    if padded_pos > 0 && padded_pos - 1 < frames {
                        sum += input[in_ch * frames + padded_pos - 1]
                            * weights[(out_ch * input_channels + in_ch) * KERNEL + k];
                    }
                }
            }
            let value = exact_gelu(sum);
            if !sum.is_finite() || !value.is_finite() {
                return Err(InferenceError::InvalidInput(
                    "audio stem produced non-finite value".into(),
                ));
            }
            output[out_ch * out_frames + t] = value;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;

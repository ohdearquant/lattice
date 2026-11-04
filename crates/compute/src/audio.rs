//! CPU feature extraction for the Whisper and Qwen2-Audio log-mel profile.
//!
//! This accepts mono PCM already sampled at 16 kHz. It is a standalone
//! frontend; it does not implement an audio model, resampling or media decoding.

use lattice_inference_core::InferenceError;
use std::f64::consts::PI;

const SAMPLE_RATE: usize = 16_000;
const FFT_SIZE: usize = 400;
const HOP: usize = 160;
const FREQUENCIES: usize = FFT_SIZE / 2 + 1;
const MELS: usize = 128;

/// **Unstable**: Whisper-style log-mel features in mel-major `[128, frames]` order.
#[derive(Debug, Clone)]
pub struct WhisperLogMel {
    /// Flat mel-major feature values.
    pub values: Vec<f32>,
    /// Number of time frames in each mel channel.
    pub frames: usize,
}

/// **Unstable**: reusable tables for the 16 kHz, 400-point Whisper log-mel frontend.
///
/// Reuse the plan across clips to amortize the FFT and filter table setup.
pub struct WhisperLogMelPlan {
    window: [f64; FFT_SIZE],
    roots: [(f64, f64); FFT_SIZE],
    filters: Vec<f64>,
}

impl Default for WhisperLogMelPlan {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperLogMelPlan {
    /// Build the periodic Hann window, FFT roots and Slaney mel filters.
    pub fn new() -> Self {
        let window =
            std::array::from_fn(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / FFT_SIZE as f64).cos());
        let roots = std::array::from_fn(|i| {
            let (sin, cos) = (2.0 * PI * i as f64 / FFT_SIZE as f64).sin_cos();
            (cos, sin)
        });
        let low_mel = hz_to_mel(0.0);
        let high_mel = hz_to_mel((SAMPLE_RATE / 2) as f64);
        let edges: Vec<f64> = (0..MELS + 2)
            .map(|i| mel_to_hz(low_mel + (high_mel - low_mel) * i as f64 / (MELS + 1) as f64))
            .collect();
        let mut filters = vec![0.0; MELS * FREQUENCIES];
        for m in 0..MELS {
            let (left, center, right) = (edges[m], edges[m + 1], edges[m + 2]);
            for k in 0..FREQUENCIES {
                let hz = SAMPLE_RATE as f64 * k as f64 / FFT_SIZE as f64;
                let triangle = ((hz - left) / (center - left))
                    .min((right - hz) / (right - center))
                    .max(0.0);
                filters[m * FREQUENCIES + k] = triangle * 2.0 / (right - left);
            }
        }
        Self {
            window,
            roots,
            filters,
        }
    }

    fn frame_spectrum(&self, windowed: &[f64; FFT_SIZE], spectrum: &mut [(f64, f64); FFT_SIZE]) {
        fft_real(windowed, 0, 1, spectrum, &self.roots);
    }

    /// Extract the unpadded Whisper log-mel tensor for a mono PCM clip.
    ///
    /// Frames use centered reflection padding and omit the final STFT frame.
    /// `sample_rate` must be exactly 16,000; this method does not resample.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for a wrong sample rate, non-finite PCM,
    /// input too short for 200-sample reflection padding, shape overflow or
    /// allocation failure.
    pub fn extract(
        &self,
        samples: &[f32],
        sample_rate: usize,
    ) -> Result<WhisperLogMel, InferenceError> {
        let invalid = |message: &str| InferenceError::InvalidInput(message.to_string());
        if sample_rate != SAMPLE_RATE {
            return Err(invalid("Whisper log-mel requires 16 kHz PCM"));
        }
        if samples.len() <= FFT_SIZE / 2 {
            return Err(invalid(
                "Whisper reflection padding requires more than 200 samples",
            ));
        }
        if samples.len() > usize::MAX - FFT_SIZE || samples.iter().any(|sample| !sample.is_finite())
        {
            return Err(invalid(
                "PCM length overflows frame indexing or contains a non-finite value",
            ));
        }
        let frames = samples.len() / HOP;
        let len = MELS
            .checked_mul(frames)
            .ok_or_else(|| invalid("log-mel shape overflow"))?;
        let mut mel = Vec::new();
        mel.try_reserve_exact(len)
            .map_err(|_| invalid("log-mel allocation failed"))?;
        mel.resize(len, 0.0f64);
        let mut windowed = [0.0f64; FFT_SIZE];
        let mut power = [0.0f64; FREQUENCIES];
        let mut spectrum = [(0.0f64, 0.0f64); FFT_SIZE];
        for frame in 0..frames {
            let start = frame * HOP;
            for (i, sample) in windowed.iter_mut().enumerate() {
                let padded = start + i;
                let index = reflect_index(padded, samples.len());
                *sample = f64::from(samples[index]) * self.window[i];
            }
            self.frame_spectrum(&windowed, &mut spectrum);
            for (value, &(re, im)) in power.iter_mut().zip(&spectrum) {
                *value = re * re + im * im;
            }
            for m in 0..MELS {
                let energy: f64 = power
                    .iter()
                    .zip(&self.filters[m * FREQUENCIES..(m + 1) * FREQUENCIES])
                    .map(|(value, weight)| value * weight)
                    .sum();
                mel[m * frames + frame] = energy.max(1e-10).log10();
            }
        }
        let max = mel.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| invalid("log-mel output allocation failed"))?;
        values.extend(
            mel.into_iter()
                .map(|value| ((value.max(max - 8.0) + 4.0) / 4.0) as f32),
        );
        Ok(WhisperLogMel { values, frames })
    }
}

fn fft_real(
    input: &[f64; FFT_SIZE],
    offset: usize,
    stride: usize,
    output: &mut [(f64, f64)],
    roots: &[(f64, f64); FFT_SIZE],
) {
    let size = output.len();
    if size == 1 {
        output[0] = (input[offset], 0.0);
        return;
    }

    let radix = if size.is_multiple_of(2) { 2 } else { 5 };
    let child_size = size / radix;
    for (lane, child) in output.chunks_exact_mut(child_size).enumerate() {
        fft_real(input, offset + lane * stride, stride * radix, child, roots);
    }

    let root_stride = FFT_SIZE / size;
    for k in 0..child_size {
        // Copy this residue group before scattering, so no input lane is overwritten.
        let mut lanes = [(0.0, 0.0); 5];
        for (lane, value) in lanes.iter_mut().take(radix).enumerate() {
            *value = output[lane * child_size + k];
        }
        for group in 0..radix {
            let bin = k + group * child_size;
            let (mut re, mut im) = lanes[0];
            for (lane, &(lane_re, lane_im)) in lanes.iter().take(radix).enumerate().skip(1) {
                let (root_re, root_im) = roots[(lane * bin * root_stride) % FFT_SIZE];
                re += lane_re * root_re - lane_im * root_im;
                im += lane_re * root_im + lane_im * root_re;
            }
            output[bin] = (re, im);
        }
    }
}

fn reflect_index(padded: usize, length: usize) -> usize {
    if padded < FFT_SIZE / 2 {
        FFT_SIZE / 2 - padded
    } else {
        let index = padded - FFT_SIZE / 2;
        if index < length {
            index
        } else {
            length - 2 - (index - length)
        }
    }
}

fn hz_to_mel(hz: f64) -> f64 {
    const LINEAR_SCALE: f64 = 200.0 / 3.0;
    if hz < 1_000.0 {
        hz / LINEAR_SCALE
    } else {
        15.0 + (hz / 1_000.0).ln() / (6.4_f64.ln() / 27.0)
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const LINEAR_SCALE: f64 = 200.0 / 3.0;
    if mel < 15.0 {
        mel * LINEAR_SCALE
    } else {
        1_000.0 * ((mel - 15.0) * (6.4_f64.ln() / 27.0)).exp()
    }
}

#[cfg(test)]
mod tests;

/// Qwen2-Audio's bounded single-clip processor profile.
pub mod qwen2_audio;

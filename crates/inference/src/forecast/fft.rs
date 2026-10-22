//! Complex radix-2 FFT in f64, and the circular convolution that long-convolution layers use.
//!
//! The transform runs in f64 because it is cheap at the lengths forecasting models use (a few
//! thousand points per channel) and it keeps the convolution's own rounding far below the f32
//! values it consumes and produces.

use std::f64::consts::PI;

use crate::error::InferenceError;

/// Tables for an in-place iterative radix-2 FFT of one power-of-two length.
#[derive(Debug, Clone)]
pub(crate) struct Fft {
    n: usize,
    /// `exp(-2πik/n)` for `k < n/2`.
    twiddles: Vec<(f64, f64)>,
    bit_reverse: Vec<usize>,
}

/// The spectrum of a real kernel, computed once and reused for every convolution with it.
#[derive(Debug, Clone)]
pub(crate) struct KernelSpectrum {
    re: Vec<f64>,
    im: Vec<f64>,
}

/// Reusable work buffers for [`Fft::circular_convolve`].
#[derive(Debug, Clone)]
pub(crate) struct FftScratch {
    re: Vec<f64>,
    im: Vec<f64>,
}

impl Fft {
    pub(crate) fn new(n: usize) -> Result<Self, InferenceError> {
        if n < 2 || !n.is_power_of_two() {
            return Err(InferenceError::InvalidInput(format!(
                "FFT length {n} is not a power of two of at least 2"
            )));
        }
        let bits = n.trailing_zeros();
        let bit_reverse = (0..n)
            .map(|i| i.reverse_bits() >> (usize::BITS - bits))
            .collect();
        let twiddles = (0..n / 2)
            .map(|k| {
                let angle = -2.0 * PI * k as f64 / n as f64;
                (angle.cos(), angle.sin())
            })
            .collect();
        Ok(Self {
            n,
            twiddles,
            bit_reverse,
        })
    }

    pub(crate) fn scratch(&self) -> FftScratch {
        FftScratch {
            re: vec![0.0; self.n],
            im: vec![0.0; self.n],
        }
    }

    /// Unnormalized DFT in place: `X[k] = Σ x[t]·exp(∓2πikt/n)`, the minus sign when
    /// `inverse` is false. The inverse direction does not divide by `n`.
    pub(crate) fn transform(&self, re: &mut [f64], im: &mut [f64], inverse: bool) {
        let n = self.n;
        debug_assert_eq!(re.len(), n);
        debug_assert_eq!(im.len(), n);
        for (i, &j) in self.bit_reverse.iter().enumerate() {
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut size = 2;
        while size <= n {
            let half = size / 2;
            let step = n / size;
            for start in (0..n).step_by(size) {
                for k in 0..half {
                    let (wr, wi) = self.twiddles[k * step];
                    let wi = if inverse { -wi } else { wi };
                    let a = start + k;
                    let b = a + half;
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            size *= 2;
        }
    }

    /// Spectrum of a real kernel of exactly `n` taps.
    pub(crate) fn kernel_spectrum(&self, kernel: &[f32]) -> KernelSpectrum {
        debug_assert_eq!(kernel.len(), self.n);
        let mut re: Vec<f64> = kernel.iter().map(|&v| f64::from(v)).collect();
        let mut im = vec![0.0; self.n];
        self.transform(&mut re, &mut im, false);
        KernelSpectrum { re, im }
    }

    /// Circular convolution of `n` real samples with a kernel:
    /// `out[t] = Σ_s x[s]·kernel[(t - s) mod n]`.
    pub(crate) fn circular_convolve(
        &self,
        x: &[f32],
        kernel: &KernelSpectrum,
        out: &mut [f32],
        scratch: &mut FftScratch,
    ) {
        let n = self.n;
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(out.len(), n);
        let (re, im) = (&mut scratch.re, &mut scratch.im);
        for (dst, &v) in re.iter_mut().zip(x) {
            *dst = f64::from(v);
        }
        im.fill(0.0);
        self.transform(re, im, false);
        for (((r, i), &c), &d) in re
            .iter_mut()
            .zip(im.iter_mut())
            .zip(&kernel.re)
            .zip(&kernel.im)
        {
            let (a, b) = (*r, *i);
            *r = a * c - b * d;
            *i = a * d + b * c;
        }
        self.transform(re, im, true);
        let scale = 1.0 / n as f64;
        for (dst, &v) in out.iter_mut().zip(re.iter()) {
            *dst = (v * scale) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn rejects_lengths_that_are_not_powers_of_two() {
        for n in [0, 1, 3, 6, 2047] {
            assert!(Fft::new(n).is_err(), "length {n} must be refused");
        }
        assert!(Fft::new(2).is_ok());
    }

    #[test]
    fn transform_matches_a_direct_dft() {
        let n = 64;
        let fft = Fft::new(n).unwrap();
        let mut seed = 7;
        let x: Vec<f64> = (0..n).map(|_| f64::from(lcg(&mut seed))).collect();
        let y: Vec<f64> = (0..n).map(|_| f64::from(lcg(&mut seed))).collect();
        let (mut re, mut im) = (x.clone(), y.clone());
        fft.transform(&mut re, &mut im, false);
        for k in 0..n {
            let (mut sr, mut si) = (0.0, 0.0);
            for t in 0..n {
                let a = -2.0 * PI * (k * t) as f64 / n as f64;
                sr += x[t] * a.cos() - y[t] * a.sin();
                si += x[t] * a.sin() + y[t] * a.cos();
            }
            assert!((re[k] - sr).abs() < 1e-9, "re[{k}] {} vs {sr}", re[k]);
            assert!((im[k] - si).abs() < 1e-9, "im[{k}] {} vs {si}", im[k]);
        }
    }

    #[test]
    fn circular_convolution_matches_the_direct_sum() {
        let n = 128;
        let fft = Fft::new(n).unwrap();
        let mut seed = 11;
        let x: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
        let k: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
        let spectrum = fft.kernel_spectrum(&k);
        let mut out = vec![0.0f32; n];
        let mut scratch = fft.scratch();
        fft.circular_convolve(&x, &spectrum, &mut out, &mut scratch);
        for t in 0..n {
            let direct: f64 = (0..n)
                .map(|s| f64::from(x[s]) * f64::from(k[(t + n - s) % n]))
                .sum();
            assert!(
                (f64::from(out[t]) - direct).abs() < 1e-5,
                "out[{t}] {} vs {direct}",
                out[t]
            );
        }
    }

    #[test]
    fn convolving_with_a_shifted_impulse_rotates_the_signal() {
        let n = 16;
        let fft = Fft::new(n).unwrap();
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut k = vec![0.0f32; n];
        k[3] = 1.0;
        let mut out = vec![0.0f32; n];
        fft.circular_convolve(&x, &fft.kernel_spectrum(&k), &mut out, &mut fft.scratch());
        for t in 0..n {
            assert!((out[t] - x[(t + n - 3) % n]).abs() < 1e-5);
        }
    }
}

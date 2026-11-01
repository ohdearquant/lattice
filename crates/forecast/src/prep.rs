//! Context preparation shared by forecasting models: missing-value imputation, padding to a
//! fixed window, striding, and resampling a forecast back to its requested length.
//!
//! Missing values are `NaN`. The arithmetic follows `numpy.interp` (f64, constant extension past
//! both ends) because published forecasting evaluations prepare their contexts with it, and a
//! model reproduces its published scores only when it sees the same context.

/// Linear interpolation of `x` over the increasing points `(xp, fp)`, as `numpy.interp`:
/// constant extension below `xp[0]` and above the last point, and an exact hit returns its value.
/// `xp` must be non-empty and increasing.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    debug_assert!(!xp.is_empty() && xp.len() == fp.len());
    let last = xp.len() - 1;
    if x < xp[0] {
        return fp[0];
    }
    if x >= xp[last] {
        return fp[last];
    }
    let j = xp.partition_point(|&p| p <= x) - 1;
    if xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let y = slope * (x - xp[j]) + fp[j];
    if !y.is_nan() {
        return y;
    }
    let y = slope * (x - xp[j + 1]) + fp[j + 1];
    if y.is_nan() && fp[j] == fp[j + 1] {
        fp[j]
    } else {
        y
    }
}

/// Replaces each `NaN` with the nearest earlier non-`NaN` value. A leading run stays `NaN`.
pub fn forward_fill(x: &mut [f32]) {
    let mut last = f32::NAN;
    for v in x.iter_mut() {
        if v.is_nan() {
            *v = last;
        } else {
            last = *v;
        }
    }
}

/// Replaces each `NaN` with the nearest later non-`NaN` value. A trailing run stays `NaN`.
pub fn backward_fill(x: &mut [f32]) {
    let mut next = f32::NAN;
    for v in x.iter_mut().rev() {
        if v.is_nan() {
            *v = next;
        } else {
            next = *v;
        }
    }
}

/// Fills every `NaN`: linear interpolation between observed points (constant past the first and
/// last) when at least two are observed; otherwise forward fill, then backward fill, then zero.
pub fn impute_missing(x: &mut [f32]) {
    if !x.iter().any(|v| v.is_nan()) {
        return;
    }
    let observed: Vec<usize> = (0..x.len()).filter(|&i| !x[i].is_nan()).collect();
    if observed.len() >= 2 {
        let xp: Vec<f64> = observed.iter().map(|&i| i as f64).collect();
        let fp: Vec<f64> = observed.iter().map(|&i| f64::from(x[i])).collect();
        for (i, v) in x.iter_mut().enumerate() {
            *v = interp(i as f64, &xp, &fp) as f32;
        }
        return;
    }
    forward_fill(x);
    backward_fill(x);
    for v in x.iter_mut() {
        if v.is_nan() {
            *v = 0.0;
        }
    }
}

/// The last `n` values of `x`, or `x` left-padded to `n` with its first value (which may be
/// `NaN`, left for imputation) when it is shorter. An empty `x` pads with zeros.
pub fn left_pad(x: &[f32], n: usize) -> Vec<f32> {
    if x.len() >= n {
        return x[x.len() - n..].to_vec();
    }
    let fill = x.first().copied().unwrap_or(0.0);
    let mut out = vec![fill; n - x.len()];
    out.extend_from_slice(x);
    out
}

/// Every `factor`-th value starting from the first (`x[::factor]`). A factor of 0 or 1 copies.
pub fn stride(x: &[f32], factor: usize) -> Vec<f32> {
    x.iter().step_by(factor.max(1)).copied().collect()
}

/// `n` evenly spaced points from 0 to 1 inclusive, computed as `numpy.linspace(0, 1, n)`.
fn unit_linspace(n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![0.0],
        _ => {
            let step = 1.0 / (n - 1) as f64;
            let mut out: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
            out[n - 1] = 1.0;
            out
        }
    }
}

/// Resamples `seq` to `new_len` points by linear interpolation over a shared unit interval, as
/// `numpy.interp(linspace(0, 1, new_len), linspace(0, 1, len(seq)), seq)`. An empty `seq` yields
/// an empty result.
pub fn resample_linear(seq: &[f32], new_len: usize) -> Vec<f32> {
    if seq.is_empty() {
        return Vec::new();
    }
    let xp = unit_linspace(seq.len());
    let fp: Vec<f64> = seq.iter().map(|&v| f64::from(v)).collect();
    unit_linspace(new_len)
        .into_iter()
        .map(|x| interp(x, &xp, &fp) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
    }

    #[test]
    fn interp_extends_constant_and_hits_points_exactly() {
        let xp = [1.0, 3.0, 4.0];
        let fp = [10.0, 20.0, -2.0];
        assert_eq!(interp(0.0, &xp, &fp), 10.0);
        assert_eq!(interp(1.0, &xp, &fp), 10.0);
        assert_eq!(interp(2.0, &xp, &fp), 15.0);
        assert_eq!(interp(3.0, &xp, &fp), 20.0);
        assert_eq!(interp(3.5, &xp, &fp), 9.0);
        assert_eq!(interp(9.0, &xp, &fp), -2.0);
    }

    #[test]
    fn imputation_interpolates_between_observed_points() {
        let mut x = [f32::NAN, 1.0, f32::NAN, 3.0, f32::NAN];
        impute_missing(&mut x);
        assert!(same(&x, &[1.0, 1.0, 2.0, 3.0, 3.0]));
    }

    #[test]
    fn a_single_observation_fills_everything() {
        let mut x = [f32::NAN, f32::NAN, 4.5, f32::NAN];
        impute_missing(&mut x);
        assert!(same(&x, &[4.5, 4.5, 4.5, 4.5]));
    }

    #[test]
    fn no_observation_fills_with_zero() {
        let mut x = [f32::NAN; 3];
        impute_missing(&mut x);
        assert!(same(&x, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn left_pad_repeats_the_first_value_and_truncates_from_the_left() {
        assert!(same(&left_pad(&[2.0, 3.0], 4), &[2.0, 2.0, 2.0, 3.0]));
        assert!(same(&left_pad(&[1.0, 2.0, 3.0], 2), &[2.0, 3.0]));
        assert!(same(&left_pad(&[], 2), &[0.0, 0.0]));
    }

    #[test]
    fn stride_starts_at_the_first_value() {
        assert!(same(
            &stride(&[0.0, 1.0, 2.0, 3.0, 4.0], 2),
            &[0.0, 2.0, 4.0]
        ));
        assert!(same(&stride(&[0.0, 1.0], 1), &[0.0, 1.0]));
    }

    #[test]
    fn resample_keeps_endpoints_and_interpolates_between() {
        let out = resample_linear(&[0.0, 10.0], 5);
        assert!(same(&out, &[0.0, 2.5, 5.0, 7.5, 10.0]));
        assert!(same(&resample_linear(&[7.0], 3), &[7.0, 7.0, 7.0]));
    }
}

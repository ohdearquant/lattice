//! Scalar and vector operations shared by the decoder.

/// Epsilon of the weightless RMS norm inside the block and message mixtures. The reference
/// builds those norms with their default epsilon, not the configured one.
pub(crate) const MIXTURE_NORM_EPS: f32 = 1e-6;
/// Added to an L2 norm before it divides a clip threshold.
const L2_CLIP_OFFSET: f32 = 1e-6;
/// Added to the sum of the selected routing scores before it divides them.
pub(crate) const ROUTE_SUM_OFFSET: f32 = 1e-20;

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn rms_factor(x: &[f32], eps: f32) -> f32 {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    1.0 / (mean + eps).sqrt()
}

/// `x * rsqrt(mean(x^2) + eps)`, without a weight.
pub(crate) fn rms_normalize_into(x: &[f32], eps: f32, out: &mut [f32]) {
    let factor = rms_factor(x, eps);
    for (o, v) in out.iter_mut().zip(x) {
        *o = v * factor;
    }
}

/// `weight * (x * rsqrt(mean(x^2) + eps))`; the weight has no `1 +` offset.
pub(crate) fn rms_norm_weighted_into(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let factor = rms_factor(x, eps);
    for ((o, v), w) in out.iter_mut().zip(x).zip(weight) {
        *o = w * (v * factor);
    }
}

/// `cap * tanh(x / cap)`.
pub(crate) fn softcap(x: f32, cap: f32) -> f32 {
    (x / cap).tanh() * cap
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Scales `v` down to L2 norm `threshold` when it is longer; a shorter vector is unchanged.
pub(crate) fn l2_clip(v: &mut [f32], threshold: f32) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let factor = (threshold / (norm + L2_CLIP_OFFSET)).min(1.0);
    for x in v.iter_mut() {
        *x *= factor;
    }
}

/// `1 + alpha * ln(max((position + 1) / floor, 1))`, the query scale of the global layers.
pub(crate) fn log_scale(position: usize, alpha: f32, floor: usize) -> f32 {
    let ratio = ((position + 1) as f32 / floor as f32).max(1.0);
    1.0 + alpha * ratio.ln()
}

/// Softmax of `logits` in place. A `sink` logit joins the normalization and takes probability
/// mass, but has no entry in the output.
pub(crate) fn softmax_with_sink(logits: &mut [f32], sink: Option<f32>) {
    let mut max = sink.unwrap_or(f32::NEG_INFINITY);
    for &logit in logits.iter() {
        max = max.max(logit);
    }
    let mut sum = sink.map_or(0.0, |s| (s - max).exp());
    for logit in logits.iter_mut() {
        *logit = (*logit - max).exp();
        sum += *logit;
    }
    for logit in logits.iter_mut() {
        *logit /= sum;
    }
}

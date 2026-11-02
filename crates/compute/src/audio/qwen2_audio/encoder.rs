//! Bounded, synthetic Qwen2-Audio encoder and projector assembly.
use super::{layer, stem};
use lattice_inference_core::InferenceError;

const MAX_ELEMENTS: usize = 16_000_000;
const MAX_MACS: u64 = 10_000_000_000;
const MAX_LAYERS: usize = 128;
const EPS: f32 = 1e-5;

/// **Unstable**: synthetic encoder dimensions and real mel-frame count.
pub struct EncoderConfig {
    /// Input mel channels.
    pub input_channels: usize,
    /// Physical mel-frame count, including padding.
    pub frames: usize,
    /// Real mel frames, a nonempty prefix of the physical input.
    pub valid_frames: usize,
    /// Encoder hidden width.
    pub hidden: usize,
    /// Number of attention heads.
    pub heads: usize,
    /// Feed-forward intermediate width.
    pub ffn: usize,
    /// Projector output width.
    pub output: usize,
}

/// **Unstable**: caller-supplied synthetic parameters.
pub struct EncoderWeights<'a> {
    /// Convolution stem.
    pub stem: stem::StemWeights<'a>,
    /// Position table `[ceil(frames/2), hidden]`, time-major.
    pub positions: &'a [f32],
    /// Ordered transformer layers.
    pub layers: &'a [layer::LayerWeights<'a>],
    /// Final normalization scale `[hidden]`.
    pub norm_weight: &'a [f32],
    /// Final normalization bias `[hidden]`.
    pub norm_bias: &'a [f32],
    /// Projector weight `[output, hidden]`.
    pub projector_weight: &'a [f32],
    /// Projector bias `[output]`.
    pub projector_bias: &'a [f32],
}

/// **Unstable**: intermediate arrays for synthetic differential checks.
pub struct EncoderOutput {
    /// Second stem result, channel-major `[hidden, time]`.
    pub stem: Vec<f32>,
    /// Position-added input to the layer stack, `[time, hidden]`.
    pub positioned: Vec<f32>,
    /// Output of the last layer or position-added input when the stack is empty.
    pub layers: Vec<f32>,
    /// Pairwise pooled states `[floor(time/2), hidden]`.
    pub pooled: Vec<f32>,
    /// Final normalized states, same shape as `pooled`.
    pub normalized: Vec<f32>,
    /// Projector output `[floor(time/2), output]`.
    pub projected: Vec<f32>,
    /// Physical post-convolution frame count.
    pub time: usize,
    /// Valid post-pool token count; physical output is not truncated to this count.
    pub valid_tokens: usize,
}

fn invalid(message: &str) -> InferenceError {
    InferenceError::InvalidInput(message.to_owned())
}
fn product(a: usize, b: usize) -> Result<usize, InferenceError> {
    a.checked_mul(b)
        .filter(|&v| v <= MAX_ELEMENTS)
        .ok_or_else(|| invalid("audio encoder shape exceeds bounds"))
}
fn allocate(n: usize) -> Result<Vec<f32>, InferenceError> {
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| invalid("audio encoder allocation failed"))?;
    out.resize(n, 0.0);
    Ok(out)
}
fn finite(values: &[f32]) -> Result<(), InferenceError> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("audio encoder encountered non-finite values"));
    }
    Ok(())
}
fn check_layer(
    w: &layer::LayerWeights<'_>,
    hidden: usize,
    ffn: usize,
) -> Result<(), InferenceError> {
    let hh = product(hidden, hidden)?;
    let fh = product(ffn, hidden)?;
    for (values, len) in [
        (w.norm1_weight, hidden),
        (w.norm1_bias, hidden),
        (w.q_weight, hh),
        (w.q_bias, hidden),
        (w.k_weight, hh),
        (w.v_weight, hh),
        (w.v_bias, hidden),
        (w.out_weight, hh),
        (w.out_bias, hidden),
        (w.norm2_weight, hidden),
        (w.norm2_bias, hidden),
        (w.fc1_weight, fh),
        (w.fc1_bias, ffn),
        (w.fc2_weight, fh),
        (w.fc2_bias, hidden),
    ] {
        if values.len() != len {
            return Err(invalid("audio encoder layer weight shape mismatch"));
        }
        finite(values)?;
    }
    Ok(())
}

fn check_work(
    c: &EncoderConfig,
    time: usize,
    valid: usize,
    conv1_len: usize,
    conv2_len: usize,
    projector_len: usize,
    layer_count: usize,
) -> Result<(), InferenceError> {
    let n = |x: usize| u64::try_from(x).ok();
    let work = n(conv1_len)
        .and_then(|a| n(c.frames).and_then(|b| a.checked_mul(b)))
        .and_then(|a| {
            n(conv2_len)
                .and_then(|b| n(time).and_then(|t| b.checked_mul(t)))
                .and_then(|b| a.checked_add(b))
        })
        .and_then(|a| {
            let t = n(time)?;
            let h = n(c.hidden)?;
            let f = n(c.ffn)?;
            let v = n(valid)?;
            let per_layer = t
                .checked_mul(h)?
                .checked_mul(h)?
                .checked_mul(4)?
                .checked_add(t.checked_mul(h)?.checked_mul(f)?.checked_mul(2)?)?
                .checked_add(t.checked_mul(v)?.checked_mul(h)?.checked_mul(2)?)?;
            a.checked_add(per_layer.checked_mul(n(layer_count)?)?)
        })
        .and_then(|a| {
            n(time / 2)
                .and_then(|t| n(projector_len).and_then(|p| t.checked_mul(p)))
                .and_then(|p| a.checked_add(p))
        });
    if work.is_none_or(|v| v > MAX_MACS) {
        return Err(invalid("audio encoder exceeds aggregate work budget"));
    }

    Ok(())
}

/// Run a bounded synthetic stem, layer stack, pool, final norm and projector.
///
/// Inputs are channel-major `[input_channels, frames]`; positions and layer
/// outputs are time-major. The valid-key count is `ceil(valid_frames/2)`;
/// padded queries are still evaluated. Pooling precedes final normalization.
/// A zero-layer stack is a synthetic fixture, not a real checkpoint.
///
/// # Errors
/// Rejects invalid or nonfinite inputs, shapes, work budgets or allocation failures.
pub fn run(
    input: &[f32],
    c: &EncoderConfig,
    w: &EncoderWeights<'_>,
) -> Result<EncoderOutput, InferenceError> {
    if c.input_channels == 0
        || c.frames == 0
        || c.hidden == 0
        || c.heads == 0
        || c.ffn == 0
        || c.output == 0
        || !c.hidden.is_multiple_of(c.heads)
        || c.valid_frames == 0
        || c.valid_frames > c.frames
        || w.layers.len() > MAX_LAYERS
    {
        return Err(invalid("audio encoder invalid dimensions or valid length"));
    }
    let time = c.frames.div_ceil(2);
    if time < 2 {
        return Err(invalid(
            "audio encoder needs at least two convolution frames for pooling",
        ));
    }
    let valid = c.valid_frames.div_ceil(2);
    let th = product(time, c.hidden)?;
    let pooled_len = product(time / 2, c.hidden)?;
    let projected_len = product(time / 2, c.output)?;
    let conv1_len = product(product(c.hidden, c.input_channels)?, 3)?;
    let conv2_len = product(product(c.hidden, c.hidden)?, 3)?;
    let projector_len = product(c.output, c.hidden)?;
    product(c.frames, c.hidden)?;
    product(time, valid)?;
    product(time, c.ffn)?;
    if input.len() != product(c.input_channels, c.frames)?
        || w.stem.conv1.len() != conv1_len
        || w.stem.bias1.len() != c.hidden
        || w.stem.conv2.len() != conv2_len
        || w.stem.bias2.len() != c.hidden
        || w.positions.len() != th
        || w.norm_weight.len() != c.hidden
        || w.norm_bias.len() != c.hidden
        || w.projector_weight.len() != projector_len
        || w.projector_bias.len() != c.output
    {
        return Err(invalid("audio encoder tensor shape mismatch"));
    }
    check_work(
        c,
        time,
        valid,
        conv1_len,
        conv2_len,
        projector_len,
        w.layers.len(),
    )?;
    for values in [
        input,
        w.stem.conv1,
        w.stem.bias1,
        w.stem.conv2,
        w.stem.bias2,
        w.positions,
        w.norm_weight,
        w.norm_bias,
        w.projector_weight,
        w.projector_bias,
    ] {
        finite(values)?;
    }
    for layer in w.layers {
        check_layer(layer, c.hidden, c.ffn)?;
    }

    let stem = stem::run(input, c.input_channels, c.frames, c.hidden, &w.stem)?.second;
    let mut positioned = allocate(th)?;
    for t in 0..time {
        for h in 0..c.hidden {
            positioned[t * c.hidden + h] = stem[h * time + t] + w.positions[t * c.hidden + h];
        }
    }
    finite(&positioned)?;
    let mut states = allocate(th)?;
    states.copy_from_slice(&positioned);
    for weights in w.layers {
        states = layer::run(&states, time, c.hidden, c.heads, c.ffn, valid, weights)?.output;
    }
    let mut pooled = allocate(pooled_len)?;
    for t in 0..time / 2 {
        for h in 0..c.hidden {
            pooled[t * c.hidden + h] =
                (states[2 * t * c.hidden + h] + states[(2 * t + 1) * c.hidden + h]) * 0.5;
        }
    }
    finite(&pooled)?;
    let mut normalized = allocate(pooled_len)?;
    for (src, dst) in pooled
        .chunks_exact(c.hidden)
        .zip(normalized.chunks_exact_mut(c.hidden))
    {
        let mean = src.iter().sum::<f32>() / c.hidden as f32;
        let var = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c.hidden as f32;
        let inv = (var + EPS).sqrt().recip();
        for h in 0..c.hidden {
            dst[h] = (src[h] - mean) * inv * w.norm_weight[h] + w.norm_bias[h];
        }
    }
    finite(&normalized)?;
    let mut projected = allocate(projected_len)?;
    for t in 0..time / 2 {
        for o in 0..c.output {
            let mut sum = w.projector_bias[o];
            for h in 0..c.hidden {
                sum += normalized[t * c.hidden + h] * w.projector_weight[o * c.hidden + h];
            }
            projected[t * c.output + o] = sum;
        }
    }
    finite(&projected)?;
    Ok(EncoderOutput {
        stem,
        positioned,
        layers: states,
        pooled,
        normalized,
        projected,
        time,
        valid_tokens: valid / 2,
    })
}

#[cfg(test)]
mod tests;

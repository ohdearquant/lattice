//! Bounded f32 evaluation of one synthetic Qwen2-Audio encoder layer.
use lattice_inference_core::InferenceError;

const MAX_ELEMENTS: usize = 16_000_000;
const MAX_MACS: u64 = 10_000_000_000;
const EPS: f32 = 1e-5;

/// **Unstable**: time-major intermediate arrays for one synthetic encoder layer.
pub struct LayerOutput {
    /// First layer-normalization output, `[time, hidden]`.
    pub norm1: Vec<f32>,
    /// Unscaled query projection, `[time, hidden]`.
    pub query: Vec<f32>,
    /// Key projection, `[time, hidden]`.
    pub key: Vec<f32>,
    /// Value projection, `[time, hidden]`.
    pub value: Vec<f32>,
    /// Output projection of concatenated attention heads, `[time, hidden]`.
    pub attention: Vec<f32>,
    /// First residual sum, `[time, hidden]`.
    pub residual: Vec<f32>,
    /// Second layer-normalization output, `[time, hidden]`.
    pub norm2: Vec<f32>,
    /// GELU output after the first FFN projection, `[time, ffn]`.
    pub activated: Vec<f32>,
    /// Final residual sum, `[time, hidden]`.
    pub output: Vec<f32>,
}

/// **Unstable**: all row-major synthetic weights and affine parameters.
pub struct LayerWeights<'a> {
    /// First layer-normalization scale and bias, each `[hidden]`.
    pub norm1_weight: &'a [f32],
    /// First layer-normalization bias.
    pub norm1_bias: &'a [f32],
    /// Query projection `[hidden, hidden]` and bias `[hidden]`.
    pub q_weight: &'a [f32],
    /// Query bias.
    pub q_bias: &'a [f32],
    /// Key projection `[hidden, hidden]`; there is no key bias.
    pub k_weight: &'a [f32],
    /// Value projection `[hidden, hidden]` and bias `[hidden]`.
    pub v_weight: &'a [f32],
    /// Value bias.
    pub v_bias: &'a [f32],
    /// Output projection `[hidden, hidden]` and bias `[hidden]`.
    pub out_weight: &'a [f32],
    /// Output bias.
    pub out_bias: &'a [f32],
    /// Second layer-normalization scale and bias, each `[hidden]`.
    pub norm2_weight: &'a [f32],
    /// Second layer-normalization bias.
    pub norm2_bias: &'a [f32],
    /// First FFN projection `[ffn, hidden]` and bias `[ffn]`.
    pub fc1_weight: &'a [f32],
    /// First FFN bias.
    pub fc1_bias: &'a [f32],
    /// Second FFN projection `[hidden, ffn]` and bias `[hidden]`.
    pub fc2_weight: &'a [f32],
    /// Second FFN bias.
    pub fc2_bias: &'a [f32],
}

fn invalid(message: &str) -> InferenceError {
    InferenceError::InvalidInput(message.to_owned())
}
fn product(a: usize, b: usize) -> Result<usize, InferenceError> {
    a.checked_mul(b)
        .filter(|&n| n <= MAX_ELEMENTS)
        .ok_or_else(|| invalid("audio layer shape exceeds bounds"))
}
fn allocate(n: usize) -> Result<Vec<f32>, InferenceError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(n)
        .map_err(|_| invalid("audio layer allocation failed"))?;
    values.resize(n, 0.0);
    Ok(values)
}
fn norm(x: &[f32], scale: &[f32], bias: &[f32], hidden: usize) -> Result<Vec<f32>, InferenceError> {
    let mut y = allocate(x.len())?;
    y.copy_from_slice(x);
    crate::forward::cpu::layer_norm(&mut y, scale, bias, hidden, EPS);
    finite(&y)?;
    Ok(y)
}
fn linear(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    time: usize,
    input: usize,
    output: usize,
) -> Result<Vec<f32>, InferenceError> {
    let mut y = allocate(product(time, output)?)?;
    crate::forward::cpu::matmul_bt(x, weight, &mut y, time, input, output);
    if let Some(bias) = bias {
        for row in y.chunks_exact_mut(output) {
            for (value, bias) in row.iter_mut().zip(bias) {
                *value += bias;
            }
        }
    }
    finite(&y)?;
    Ok(y)
}
fn finite(x: &[f32]) -> Result<(), InferenceError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(invalid("audio layer encountered non-finite values"));
    }
    Ok(())
}

/// Evaluate one pre-normalized noncausal encoder layer in f32.
///
/// `input` is `[time, hidden]` after adding position embeddings. `valid_keys`
/// is the number of non-padding keys in the contiguous prefix, from 1 to time.
/// Padded queries are still evaluated. This bounded baseline does not load a model.
///
/// # Errors
/// Returns an error for invalid dimensions, shape, finite values, budget, or allocation.
pub fn run(
    input: &[f32],
    time: usize,
    hidden: usize,
    heads: usize,
    ffn: usize,
    valid_keys: usize,
    w: &LayerWeights<'_>,
) -> Result<LayerOutput, InferenceError> {
    if time == 0
        || hidden == 0
        || heads == 0
        || ffn == 0
        || !hidden.is_multiple_of(heads)
        || valid_keys == 0
        || valid_keys > time
    {
        return Err(invalid("audio layer invalid dimensions or valid-key count"));
    }
    let th = product(time, hidden)?;
    let tf = product(time, ffn)?;
    let hh = product(hidden, hidden)?;
    let fh = product(ffn, hidden)?;
    product(time, valid_keys)?;
    if input.len() != th
        || [
            w.norm1_weight,
            w.norm1_bias,
            w.q_bias,
            w.v_bias,
            w.out_bias,
            w.norm2_weight,
            w.norm2_bias,
            w.fc2_bias,
        ]
        .iter()
        .any(|x| x.len() != hidden)
        || [w.q_weight, w.k_weight, w.v_weight, w.out_weight]
            .iter()
            .any(|x| x.len() != hh)
        || [w.fc1_weight, w.fc2_weight].iter().any(|x| x.len() != fh)
        || w.fc1_bias.len() != ffn
    {
        return Err(invalid("audio layer tensor shape mismatch"));
    }
    let work = (time as u64)
        .checked_mul(hidden as u64)
        .and_then(|v| v.checked_mul(hidden as u64))
        .and_then(|v| v.checked_mul(4))
        .and_then(|v| {
            (time as u64)
                .checked_mul(hidden as u64)
                .and_then(|a| a.checked_mul(ffn as u64))
                .and_then(|a| a.checked_mul(2))
                .and_then(|a| v.checked_add(a))
        })
        .and_then(|v| {
            (time as u64)
                .checked_mul(valid_keys as u64)
                .and_then(|a| a.checked_mul(hidden as u64))
                .and_then(|a| a.checked_mul(2))
                .and_then(|a| v.checked_add(a))
        });
    if work.is_none_or(|v| v > MAX_MACS) {
        return Err(invalid("audio layer exceeds work budget"));
    }
    for values in [
        input,
        w.norm1_weight,
        w.norm1_bias,
        w.q_weight,
        w.q_bias,
        w.k_weight,
        w.v_weight,
        w.v_bias,
        w.out_weight,
        w.out_bias,
        w.norm2_weight,
        w.norm2_bias,
        w.fc1_weight,
        w.fc1_bias,
        w.fc2_weight,
        w.fc2_bias,
    ] {
        finite(values)?;
    }
    let norm1 = norm(input, w.norm1_weight, w.norm1_bias, hidden)?;
    let query = linear(&norm1, w.q_weight, Some(w.q_bias), time, hidden, hidden)?;
    let key = linear(&norm1, w.k_weight, None, time, hidden, hidden)?;
    let value = linear(&norm1, w.v_weight, Some(w.v_bias), time, hidden, hidden)?;
    let mut context = allocate(th)?;
    let mut scores = allocate(valid_keys)?;
    let width = hidden / heads;
    let scale = (width as f32).sqrt().recip();
    for t in 0..time {
        for h in 0..heads {
            let offset = h * width;
            for k in 0..valid_keys {
                let mut sum = 0.0;
                for d in 0..width {
                    sum += (query[t * hidden + offset + d] * scale) * key[k * hidden + offset + d];
                }
                scores[k] = sum;
            }
            finite(&scores)?;
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0;
            for s in &mut scores {
                *s = (*s - max).exp();
                denominator += *s;
            }
            if !denominator.is_finite() || denominator == 0.0 {
                return Err(invalid("audio layer invalid softmax"));
            }
            for k in 0..valid_keys {
                let probability = scores[k] / denominator;
                for d in 0..width {
                    context[t * hidden + offset + d] +=
                        probability * value[k * hidden + offset + d];
                }
            }
        }
    }
    finite(&context)?;
    let attention = linear(
        &context,
        w.out_weight,
        Some(w.out_bias),
        time,
        hidden,
        hidden,
    )?;
    let mut residual = allocate(th)?;
    for i in 0..th {
        residual[i] = input[i] + attention[i];
    }
    finite(&residual)?;
    let norm2 = norm(&residual, w.norm2_weight, w.norm2_bias, hidden)?;
    let mut activated = linear(&norm2, w.fc1_weight, Some(w.fc1_bias), time, hidden, ffn)?;
    debug_assert_eq!(activated.len(), tf);
    for v in &mut activated {
        *v = super::stem::exact_gelu(*v);
    }
    finite(&activated)?;
    let ff2 = linear(
        &activated,
        w.fc2_weight,
        Some(w.fc2_bias),
        time,
        ffn,
        hidden,
    )?;
    let mut output = allocate(th)?;
    for i in 0..th {
        output[i] = residual[i] + ff2[i];
    }
    finite(&output)?;
    Ok(LayerOutput {
        norm1,
        query,
        key,
        value,
        attention,
        residual,
        norm2,
        activated,
        output,
    })
}

#[cfg(test)]
mod tests;

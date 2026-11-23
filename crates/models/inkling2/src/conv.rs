//! Short causal depthwise convolutions with a residual and an output scale.

/// The inputs of one convolution that later tokens still read: the last `kernel - 1`, oldest
/// first. It starts as zeros, the left padding of a prompt.
#[derive(Debug, Clone)]
pub(crate) struct ConvHistory {
    channels: usize,
    values: Vec<f32>,
}

impl ConvHistory {
    pub(crate) fn new(channels: usize, kernel: usize) -> Self {
        Self {
            channels,
            values: vec![0.0; channels * (kernel - 1)],
        }
    }

    fn context(&self, inputs: &[f32]) -> Vec<f32> {
        let mut context = Vec::with_capacity(self.values.len() + inputs.len());
        context.extend_from_slice(&self.values);
        context.extend_from_slice(inputs);
        context
    }

    fn keep_tail(&mut self, context: &[f32]) {
        let start = context.len() - self.values.len();
        self.values.copy_from_slice(&context[start..]);
    }

    #[cfg(test)]
    pub(crate) fn values(&self) -> &[f32] {
        &self.values
    }
}

fn output_divisor(taps: usize) -> f32 {
    (taps as f64 + 1.0).sqrt() as f32
}

/// The static form: `y_t[c] = (sum_i taps[c][i] x_{t - kernel + 1 + i}[c] + x_t[c]) /
/// sqrt(kernel + 1)`, with `taps` stored `[channels, kernel]` and `inputs` and `out` holding
/// whole tokens of `channels` elements.
pub(crate) fn static_conv(
    history: &mut ConvHistory,
    taps: &[f32],
    kernel: usize,
    inputs: &[f32],
    out: &mut [f32],
) {
    let channels = history.channels;
    let context = history.context(inputs);
    let divisor = output_divisor(kernel);
    for (t, (row_out, row_in)) in out
        .chunks_exact_mut(channels)
        .zip(inputs.chunks_exact(channels))
        .enumerate()
    {
        for (channel, (o, &x)) in row_out.iter_mut().zip(row_in).enumerate() {
            let channel_taps = &taps[channel * kernel..(channel + 1) * kernel];
            let mut acc = 0.0f32;
            for (i, &tap) in channel_taps.iter().enumerate() {
                acc += tap * context[(t + i) * channels + channel];
            }
            *o = (acc + x) / divisor;
        }
    }
    history.keep_tail(&context);
}

/// The dynamic form: the tap at backward distance `lag` for channel `c` of key/value head `h` at
/// token `t` is `static_taps[lag][c] + token_taps[t][h * kernel + lag]`, and
/// `y_t[c] = (sum_lag tap x_{t - lag}[c] + x_t[c]) / sqrt(2 * kernel + 1)`.
pub(crate) fn dynamic_conv(
    history: &mut ConvHistory,
    static_taps: &[f32],
    token_taps: &[f32],
    heads: usize,
    kernel: usize,
    inputs: &[f32],
    out: &mut [f32],
) {
    let channels = history.channels;
    let per_head = channels / heads;
    let context = history.context(inputs);
    let divisor = output_divisor(2 * kernel);
    for (t, (row_out, row_in)) in out
        .chunks_exact_mut(channels)
        .zip(inputs.chunks_exact(channels))
        .enumerate()
    {
        let taps_t = &token_taps[t * heads * kernel..(t + 1) * heads * kernel];
        for (channel, (o, &x)) in row_out.iter_mut().zip(row_in).enumerate() {
            let head = channel / per_head;
            let mut acc = 0.0f32;
            for lag in 0..kernel {
                let tap = static_taps[lag * channels + channel] + taps_t[head * kernel + lag];
                acc += tap * context[(t + kernel - 1 - lag) * channels + channel];
            }
            *o = (acc + x) / divisor;
        }
    }
    history.keep_tail(&context);
}

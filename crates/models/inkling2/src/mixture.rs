//! Attention over RMS-normalized states with learned queries, and the gate that mixes a message
//! into a hidden state. The block mixtures replace the residual sum at the attention and MLP
//! inputs; the message mixture carries the previous pass into the next.

use crate::ops::{MIXTURE_NORM_EPS, dot, rms_normalize_into, softmax_with_sink};
use crate::weights::MessageGateWeights;

/// States normalized once, then attended by any number of queries. The normalized states are
/// both the keys and the values, and the logits are scaled by `hidden^-0.5`.
pub(crate) struct NormalizedStates {
    hidden: usize,
    rows: Vec<f32>,
    scale: f32,
}

impl NormalizedStates {
    pub(crate) fn new(hidden: usize, states: &[&[f32]]) -> Self {
        let mut rows = vec![0.0f32; states.len() * hidden];
        for (row, state) in rows.chunks_exact_mut(hidden).zip(states) {
            rms_normalize_into(state, MIXTURE_NORM_EPS, row);
        }
        Self {
            hidden,
            rows,
            scale: (hidden as f64).powf(-0.5) as f32,
        }
    }

    pub(crate) fn attend(&self, query: &[f32], out: &mut [f32]) {
        let mut weights: Vec<f32> = self
            .rows
            .chunks_exact(self.hidden)
            .map(|row| dot(query, row) * self.scale)
            .collect();
        softmax_with_sink(&mut weights, None);
        out.fill(0.0);
        for (weight, row) in weights.iter().zip(self.rows.chunks_exact(self.hidden)) {
            for (o, v) in out.iter_mut().zip(row) {
                *o += weight * v;
            }
        }
    }
}

/// `out = (1 - tanh(forget z)) * hidden + tanh(gain z) * message` with
/// `z = down [n(hidden); n(message)]`; the un-normalized states enter the output.
pub(crate) fn message_gate(
    gate: &MessageGateWeights,
    eps: f32,
    hidden: &[f32],
    message: &[f32],
    out: &mut [f32],
) {
    let width = hidden.len();
    let mut joined = vec![0.0f32; 2 * width];
    rms_normalize_into(hidden, eps, &mut joined[..width]);
    rms_normalize_into(message, eps, &mut joined[width..]);
    let mut low = vec![0.0f32; gate.down.rows()];
    gate.down.matvec(&joined, &mut low);
    let mut forget = vec![0.0f32; width];
    gate.forget.matvec(&low, &mut forget);
    let mut gain = vec![0.0f32; width];
    gate.gain.matvec(&low, &mut gain);
    for (i, o) in out.iter_mut().enumerate() {
        *o = (1.0 - forget[i].tanh()) * hidden[i] + gain[i].tanh() * message[i];
    }
}

//! Sigmoid-routed experts with a selection bias and a softcapped SwiGLU.

use lattice_inference_core::InferenceError;

use crate::ops::{ROUTE_SUM_OFFSET, sigmoid, softcap};
use crate::tensor::DenseMatrix;
use crate::weights::MoeWeights;

pub(crate) struct MoeParams {
    pub(crate) top_k: usize,
    pub(crate) route_scale: f32,
    pub(crate) gate_cap: f32,
    pub(crate) up_cap: f32,
    pub(crate) intermediate: usize,
}

/// The chosen experts in increasing order with their weights. Experts are chosen by score plus
/// selection bias; the weights use the scores alone, normalized over the chosen experts and
/// multiplied by the route scale.
pub(crate) fn route(
    p: &MoeParams,
    router: &DenseMatrix,
    selection_bias: &[f32],
    x: &[f32],
) -> Vec<(usize, f32)> {
    let mut scores = vec![0.0f32; router.rows()];
    router.matvec(x, &mut scores);
    for score in scores.iter_mut() {
        *score = sigmoid(*score);
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        (scores[b] + selection_bias[b])
            .total_cmp(&(scores[a] + selection_bias[a]))
            .then(a.cmp(&b))
    });
    order.truncate(p.top_k);
    order.sort_unstable();
    let sum: f32 = order.iter().map(|&expert| scores[expert]).sum();
    order
        .into_iter()
        .map(|expert| {
            (
                expert,
                scores[expert] / (sum + ROUTE_SUM_OFFSET) * p.route_scale,
            )
        })
        .collect()
}

/// `softcap(gate) * sigmoid(gate) * softcap(up)`: the sigmoid sees the raw gate.
pub(crate) fn softcapped_swiglu(gate: f32, up: f32, gate_cap: f32, up_cap: f32) -> f32 {
    softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap)
}

/// The weighted sum of the chosen experts for one token, accumulated in increasing expert order.
pub(crate) fn moe(
    p: &MoeParams,
    w: &MoeWeights,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), InferenceError> {
    out.fill(0.0);
    let mut gate_up = vec![0.0f32; 2 * p.intermediate];
    let mut activation = vec![0.0f32; p.intermediate];
    let mut down = vec![0.0f32; out.len()];
    for (expert, weight) in route(p, &w.router, &w.selection_bias, x) {
        w.gate_up.matvec(expert, x, &mut gate_up)?;
        for (a, pair) in activation.iter_mut().zip(gate_up.chunks_exact(2)) {
            *a = softcapped_swiglu(pair[0], pair[1], p.gate_cap, p.up_cap);
        }
        w.down.matvec(expert, &activation, &mut down)?;
        for (o, d) in out.iter_mut().zip(&down) {
            *o += weight * d;
        }
    }
    Ok(())
}

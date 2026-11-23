//! Attention of one layer: the tokens of a call attend to themselves and to the keys kept in
//! the cache slot of the layer application.

use lattice_arch::inkling2_config::AttentionSpan;

use crate::cache::CacheSlot;
use crate::conv::{ConvHistory, dynamic_conv, static_conv};
use crate::ops::{dot, l2_clip, log_scale, rms_normalize_into, sigmoid, softmax_with_sink};
use crate::tensor::DenseMatrix;
use crate::weights::{AttentionWeights, KvConvolution};

/// Geometry and constants of the attention of one layer.
pub(crate) struct AttentionParams {
    pub(crate) span: AttentionSpan,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) hidden: usize,
    pub(crate) d_rel: usize,
    pub(crate) extent: usize,
    pub(crate) kernel: usize,
    pub(crate) window: Option<usize>,
    pub(crate) qk_clip: f32,
    pub(crate) rel_clip: f32,
    pub(crate) eps: f32,
    pub(crate) log_alpha: f32,
    pub(crate) log_floor: usize,
}

/// Inputs of the query, key and value projections, `tokens * hidden` values each. The query
/// input also feeds the relative-position states, the sink logits and the output gate.
pub(crate) struct AttentionInputs<'a> {
    pub(crate) query: &'a [f32],
    pub(crate) key: &'a [f32],
    pub(crate) value: &'a [f32],
}

/// Attention for the tokens at positions `start..start + tokens`, written to `out`
/// (`tokens * hidden`). The new keys and values are appended to `slot`, and a sliding slot
/// then drops the ones no later token can see.
pub(crate) fn attention(
    p: &AttentionParams,
    w: &AttentionWeights,
    inputs: &AttentionInputs<'_>,
    start: usize,
    slot: &mut CacheSlot,
    out: &mut [f32],
) {
    let tokens = inputs.query.len() / p.hidden;
    let kv_width = p.kv_heads * p.head_dim;
    let mut keys = project_kv(
        p,
        &w.key,
        &w.key_convolution,
        inputs.key,
        &mut slot.key_history,
    );
    for head in keys.chunks_exact_mut(p.head_dim) {
        l2_clip(head, p.qk_clip);
    }
    let mut values = project_kv(
        p,
        &w.value,
        &w.value_convolution,
        inputs.value,
        &mut slot.value_history,
    );
    let mut normalized = vec![0.0f32; p.head_dim];
    for head in values.chunks_exact_mut(p.head_dim) {
        rms_normalize_into(head, p.eps, &mut normalized);
        head.copy_from_slice(&normalized);
    }
    slot.keys.extend_from_slice(&keys);
    slot.values.extend_from_slice(&values);

    let q_width = p.heads * p.head_dim;
    let group = p.heads / p.kv_heads;
    let logit_scale = (p.head_dim as f64).powf(-0.5) as f32;
    let global = p.span == AttentionSpan::Global;
    let mut query = vec![0.0f32; q_width];
    let mut relative = vec![0.0f32; p.heads * p.d_rel];
    let mut bias = vec![0.0f32; p.heads * p.extent];
    let mut sinks = vec![0.0f32; p.heads];
    let mut gate = vec![0.0f32; q_width];
    let mut mixed = vec![0.0f32; q_width];
    let mut logits = Vec::new();
    for t in 0..tokens {
        let position = start + t;
        let q_in = &inputs.query[t * p.hidden..(t + 1) * p.hidden];
        let tau = if global {
            log_scale(position, p.log_alpha, p.log_floor)
        } else {
            1.0
        };
        w.query.matvec(q_in, &mut query);
        scale_clip(&mut query, &w.query_scale, p.head_dim, p.qk_clip, tau);
        w.relative.matvec(q_in, &mut relative);
        scale_clip(&mut relative, &w.relative_scale, p.d_rel, p.rel_clip, tau);
        fill_relative_bias(
            &relative,
            p.d_rel,
            &w.relative_profiles,
            p.extent,
            &mut bias,
        );
        w.sink.matvec(q_in, &mut sinks);
        w.output_gate.matvec(q_in, &mut gate);

        let first = slot.first_position;
        let lowest = match p.window {
            Some(window) => first.max((position + 1).saturating_sub(window)),
            None => first,
        };
        for head in 0..p.heads {
            let q_head = &query[head * p.head_dim..(head + 1) * p.head_dim];
            let kv_offset = (head / group) * p.head_dim;
            let head_bias = &bias[head * p.extent..(head + 1) * p.extent];
            logits.clear();
            for key_position in lowest..=position {
                let row = (key_position - first) * kv_width + kv_offset;
                let key = &slot.keys[row..row + p.head_dim];
                logits.push(
                    dot(q_head, key) * logit_scale + bias_at(head_bias, position - key_position),
                );
            }
            softmax_with_sink(&mut logits, Some(sinks[head]));
            let out_head = &mut mixed[head * p.head_dim..(head + 1) * p.head_dim];
            out_head.fill(0.0);
            for (offset, weight) in logits.iter().enumerate() {
                let row = (lowest + offset - first) * kv_width + kv_offset;
                for (o, v) in out_head.iter_mut().zip(&slot.values[row..row + p.head_dim]) {
                    *o += weight * v;
                }
            }
        }
        for (m, g) in mixed.iter_mut().zip(&gate) {
            *m *= sigmoid(*g);
        }
        w.output
            .matvec(&mixed, &mut out[t * p.hidden..(t + 1) * p.hidden]);
    }
    slot.evict();
}

/// Projects and convolves the key or value inputs, `tokens * kv_heads * head_dim` values. The
/// per-token taps of a dynamic convolution are projected from the same inputs.
fn project_kv(
    p: &AttentionParams,
    matrix: &DenseMatrix,
    convolution: &KvConvolution,
    inputs: &[f32],
    history: &mut ConvHistory,
) -> Vec<f32> {
    let tokens = inputs.len() / p.hidden;
    let width = p.kv_heads * p.head_dim;
    let mut projected = vec![0.0f32; tokens * width];
    for (row_out, row_in) in projected
        .chunks_exact_mut(width)
        .zip(inputs.chunks_exact(p.hidden))
    {
        matrix.matvec(row_in, row_out);
    }
    let mut out = vec![0.0f32; tokens * width];
    match convolution {
        KvConvolution::Static { taps } => {
            static_conv(history, taps, p.kernel, &projected, &mut out)
        }
        KvConvolution::Dynamic {
            static_taps,
            projection,
        } => {
            let per_token = p.kv_heads * p.kernel;
            let mut token_taps = vec![0.0f32; tokens * per_token];
            for (row_out, row_in) in token_taps
                .chunks_exact_mut(per_token)
                .zip(inputs.chunks_exact(p.hidden))
            {
                projection.matvec(row_in, row_out);
            }
            dynamic_conv(
                history,
                static_taps,
                &token_taps,
                p.kv_heads,
                p.kernel,
                &projected,
                &mut out,
            );
        }
    }
    out
}

/// Multiplies by the learned scale, clips each head of `width` values to L2 norm `threshold`,
/// then multiplies by `tau`.
fn scale_clip(states: &mut [f32], scale: &[f32], width: usize, threshold: f32, tau: f32) {
    for (v, s) in states.iter_mut().zip(scale) {
        *v *= s;
    }
    for head in states.chunks_exact_mut(width) {
        l2_clip(head, threshold);
        for v in head.iter_mut() {
            *v *= tau;
        }
    }
}

/// `bias[h][d] = sum_j relative[h][j] * profiles[j][d]` for every head and every backward
/// distance `d < extent`.
pub(crate) fn fill_relative_bias(
    relative: &[f32],
    d_rel: usize,
    profiles: &DenseMatrix,
    extent: usize,
    bias: &mut [f32],
) {
    let mut profile = vec![0.0f32; extent];
    bias.fill(0.0);
    for j in 0..d_rel {
        profiles.row_into(j, &mut profile);
        for (head_relative, head_bias) in relative
            .chunks_exact(d_rel)
            .zip(bias.chunks_exact_mut(extent))
        {
            let weight = head_relative[j];
            for (b, value) in head_bias.iter_mut().zip(&profile) {
                *b += weight * value;
            }
        }
    }
}

/// The bias of one head at a backward distance; zero at or past the extent.
pub(crate) fn bias_at(head_bias: &[f32], distance: usize) -> f32 {
    head_bias.get(distance).copied().unwrap_or(0.0)
}

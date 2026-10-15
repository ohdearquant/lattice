use crate::attention::flash::{TiledAttentionBuffers, TiledAttentionConfig};
use crate::error::InferenceError;
use crate::forward::cpu::matmul_bt;
use crate::model::qwen35::resize;
use core::mem::size_of;

fn checked_float_product(a: usize, b: usize) -> Result<usize, InferenceError> {
    let len = a.checked_mul(b).ok_or_else(|| {
        InferenceError::InvalidInput("prefill attention tile dimensions overflow usize".into())
    })?;
    if len > isize::MAX as usize / size_of::<f32>() {
        return Err(InferenceError::InvalidInput(
            "prefill attention tile exceeds addressable f32 storage".into(),
        ));
    }
    Ok(len)
}

pub(super) fn prepare_buffers(
    config: &TiledAttentionConfig,
    buffers: &mut TiledAttentionBuffers,
) -> Result<(), InferenceError> {
    let br = config.tile_size_q.max(1);
    let bc = config.tile_size_kv.max(1);
    let q_len = checked_float_product(br, config.head_dim)?;
    let kv_len = checked_float_product(bc, config.head_dim)?;
    let scores_len = checked_float_product(br, bc)?;
    let rows_len = checked_float_product(br, 1)?;

    for (values, len) in [
        (&mut buffers.q_tile, q_len),
        (&mut buffers.k_tile, kv_len),
        (&mut buffers.v_tile, kv_len),
        (&mut buffers.s_tile, scores_len),
        (&mut buffers.output, q_len),
        (&mut buffers.row_max, rows_len),
        (&mut buffers.row_sum, rows_len),
    ] {
        resize(values, len);
    }
    Ok(())
}

trait WorkObserver {
    fn packed_q(&mut self, _q_start: usize, _q_rows: usize, _head_dim: usize) {}

    fn packed_kv(&mut self, _q_start: usize, _kv_start: usize, _kv_rows: usize, _head_dim: usize) {}

    fn scored(
        &mut self,
        _q_start: usize,
        _kv_start: usize,
        _q_rows: usize,
        _kv_rows: usize,
        _head_dim: usize,
    ) {
    }
}

struct Noop;

impl WorkObserver for Noop {}

pub(super) fn causal_attention_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    q_stride: usize,
    kv_stride: usize,
    out_stride: usize,
    q_head_offset: usize,
    kv_head_offset: usize,
    head_dim: usize,
    scale: f32,
    config: &TiledAttentionConfig,
    buffers: &mut TiledAttentionBuffers,
) {
    causal_attention_head_with_observer(
        q,
        k,
        v,
        output,
        seq_len,
        q_stride,
        kv_stride,
        out_stride,
        q_head_offset,
        kv_head_offset,
        head_dim,
        scale,
        config,
        buffers,
        &mut Noop,
    );
}

fn causal_attention_head_with_observer<O: WorkObserver>(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    q_stride: usize,
    kv_stride: usize,
    out_stride: usize,
    q_head_offset: usize,
    kv_head_offset: usize,
    head_dim: usize,
    scale: f32,
    config: &TiledAttentionConfig,
    buffers: &mut TiledAttentionBuffers,
    observer: &mut O,
) {
    assert_eq!(head_dim, config.head_dim);
    let tile_q = config.tile_size_q.max(1);
    let tile_kv = config.tile_size_kv.max(1);
    let TiledAttentionBuffers {
        q_tile,
        k_tile,
        v_tile,
        s_tile,
        output: acc,
        row_max,
        row_sum,
        ..
    } = buffers;

    for q_start in (0..seq_len).step_by(tile_q) {
        let q_rows = tile_q.min(seq_len - q_start);
        let q_end = q_start + q_rows;
        for qi in 0..q_rows {
            let src = (q_start + qi) * q_stride + q_head_offset;
            q_tile[qi * head_dim..(qi + 1) * head_dim].copy_from_slice(&q[src..src + head_dim]);
        }
        observer.packed_q(q_start, q_rows, head_dim);
        acc[..q_rows * head_dim].fill(0.0);
        row_max[..q_rows].fill(f32::NEG_INFINITY);
        row_sum[..q_rows].fill(0.0);

        for kv_start in (0..q_end).step_by(tile_kv) {
            let kv_rows = tile_kv.min(q_end - kv_start);
            for ki in 0..kv_rows {
                let src = (kv_start + ki) * kv_stride + kv_head_offset;
                k_tile[ki * head_dim..(ki + 1) * head_dim].copy_from_slice(&k[src..src + head_dim]);
                v_tile[ki * head_dim..(ki + 1) * head_dim].copy_from_slice(&v[src..src + head_dim]);
            }
            observer.packed_kv(q_start, kv_start, kv_rows, head_dim);
            matmul_bt(
                &q_tile[..q_rows * head_dim],
                &k_tile[..kv_rows * head_dim],
                &mut s_tile[..q_rows * kv_rows],
                q_rows,
                head_dim,
                kv_rows,
            );
            observer.scored(q_start, kv_start, q_rows, kv_rows, head_dim);

            for qi in 0..q_rows {
                let q_row = q_start + qi;
                if q_row < kv_start {
                    continue;
                }
                let valid = kv_rows.min(q_row + 1 - kv_start);
                let scores = &mut s_tile[qi * kv_rows..qi * kv_rows + valid];
                let mut local_max = f32::NEG_INFINITY;
                let mut poisoned = false;
                for score in scores.iter_mut() {
                    *score *= scale;
                    if score.is_nan() || *score == f32::INFINITY {
                        poisoned = true;
                    }
                    if *score > local_max {
                        local_max = *score;
                    }
                }
                let new_max = row_max[qi].max(local_max);
                let alpha = if row_sum[qi] == 0.0 {
                    0.0
                } else {
                    (row_max[qi] - new_max).exp()
                };
                for value in &mut acc[qi * head_dim..(qi + 1) * head_dim] {
                    *value *= alpha;
                }
                row_sum[qi] *= alpha;
                for score in scores.iter_mut() {
                    *score = (*score - new_max).exp();
                    row_sum[qi] += *score;
                }
                // NaN remains sticky across later tiles, including an all-minus-inf prefix.
                if poisoned {
                    row_sum[qi] = f32::NAN;
                }
                row_max[qi] = new_max;
            }

            for ki in 0..kv_rows {
                let kv_pos = kv_start + ki;
                for qi in 0..q_rows {
                    // A masked future value must never enter even a zero-weight product.
                    if kv_pos > q_start + qi {
                        continue;
                    }
                    let weight = s_tile[qi * kv_rows + ki];
                    for d in 0..head_dim {
                        acc[qi * head_dim + d] += weight * v_tile[ki * head_dim + d];
                    }
                }
            }
        }

        for qi in 0..q_rows {
            let out_start = (q_start + qi) * out_stride + q_head_offset;
            let out = &mut output[out_start..out_start + head_dim];
            if !row_sum[qi].is_finite() || row_sum[qi] <= 0.0 {
                out.fill(0.0);
            } else {
                let inv_sum = 1.0 / row_sum[qi];
                for d in 0..head_dim {
                    out[d] = acc[qi * head_dim + d] * inv_sum;
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "batch_prefill_attention_tests.rs"]
mod tests;

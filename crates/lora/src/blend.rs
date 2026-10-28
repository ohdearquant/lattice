//! Host-side concatenation of LoRA factors with already-effective weights.

use crate::{
    LoraError, LoraLayer, LoraLayerData, accumulate_rank, check_buffer_lengths,
    check_rank_total_cap,
};

/// Concatenate a validated projection's factors using already-effective weights.
///
/// Each entry is `(A, B, rank, effective_weight)`. Matrices are row-major,
/// with A shaped `(rank, d_in)` and B shaped `(d_out, rank)`. Source order
/// determines the vertical A and horizontal B concatenation order. This is
/// the legacy bridge for callers that have already folded any adapter scale
/// into their mixture weight; it does not apply another scale.
///
/// # Errors
///
/// Returns a validation error for rank overflow, rank-cap violations,
/// disagreement with `rank_total`, invalid source lengths, or allocation
/// size overflow. The caller retains its own grouping and dimension checks.
pub fn blend_effective_weights<'a, I>(
    ctx: &str,
    entries: I,
    rank_total: usize,
    d_in: usize,
    d_out: usize,
) -> Result<LoraLayer, LoraError>
where
    I: Iterator<Item = (&'a [f32], &'a [f32], usize, f32)>,
{
    let entries: Vec<_> = entries.collect();
    blend_slice(ctx, &entries, rank_total, d_in, d_out)
}

/// Blend shared host factors without allocating a temporary factor list.
///
/// The shape, rank and buffer checks are the same as [`blend_effective_weights`].
///
/// # Errors
///
/// Returns a validation error when rank, buffer lengths or output sizes are invalid.
pub fn blend_layer_effective_weights(
    ctx: &str,
    entries: &[(&LoraLayer, f32)],
    rank_total: usize,
    d_in: usize,
    d_out: usize,
) -> Result<LoraLayer, LoraError> {
    blend_slice(ctx, entries, rank_total, d_in, d_out)
}

/// Blend shared GPU-load factor records on the host without a temporary factor list.
///
/// Weights are already effective; no additional scale is applied. The shape, rank
/// and buffer checks are the same as [`blend_effective_weights`].
///
/// # Errors
///
/// Returns a validation error when rank, buffer lengths or output sizes are invalid.
pub fn blend_layer_data_effective_weights(
    ctx: &str,
    entries: &[(&LoraLayerData, f32)],
    rank_total: usize,
    d_in: usize,
    d_out: usize,
) -> Result<LoraLayer, LoraError> {
    blend_slice(ctx, entries, rank_total, d_in, d_out)
}

trait Factor {
    fn parts(&self) -> (&[f32], &[f32], usize, f32);
}

impl Factor for (&[f32], &[f32], usize, f32) {
    #[inline]
    fn parts(&self) -> (&[f32], &[f32], usize, f32) {
        *self
    }
}

impl Factor for (&LoraLayer, f32) {
    #[inline]
    fn parts(&self) -> (&[f32], &[f32], usize, f32) {
        (&self.0.a, &self.0.b, self.0.rank, self.1)
    }
}

impl Factor for (&LoraLayerData, f32) {
    #[inline]
    fn parts(&self) -> (&[f32], &[f32], usize, f32) {
        (&self.0.a, &self.0.b, self.0.rank, self.1)
    }
}

#[inline]
fn blend_slice<T: Factor>(
    ctx: &str,
    entries: &[T],
    rank_total: usize,
    d_in: usize,
    d_out: usize,
) -> Result<LoraLayer, LoraError> {
    let mut actual_rank = 0usize;
    for entry in entries {
        let (_, _, rank, _) = entry.parts();
        actual_rank = accumulate_rank(actual_rank, rank, ctx).map_err(LoraError::Validation)?;
    }
    check_rank_total_cap(actual_rank, ctx).map_err(LoraError::Validation)?;
    if actual_rank != rank_total {
        return Err(LoraError::Validation(format!(
            "{ctx}: planned rank {rank_total} does not match source rank {actual_rank}"
        )));
    }
    for (idx, entry) in entries.iter().enumerate() {
        let (a, b, rank, _) = entry.parts();
        check_buffer_lengths(ctx, idx, rank, d_in, d_out, a.len(), b.len())
            .map_err(LoraError::Validation)?;
    }
    let a_buf_len = rank_total.checked_mul(d_in).ok_or_else(|| {
        LoraError::Validation(format!("{ctx}: rank_total * d_in overflowed usize"))
    })?;
    let b_buf_len = d_out.checked_mul(rank_total).ok_or_else(|| {
        LoraError::Validation(format!("{ctx}: d_out * rank_total overflowed usize"))
    })?;

    let mut a_blend = Vec::with_capacity(a_buf_len);
    for entry in entries {
        let (a, _, _, _) = entry.parts();
        a_blend.extend_from_slice(a);
    }

    let mut b_blend = vec![0.0f32; b_buf_len];
    let mut col_offset = 0usize;
    for entry in entries {
        let (_, b, r_e, eff_weight) = entry.parts();
        for row in 0..d_out {
            let dst_start = row * rank_total + col_offset;
            let src_start = row * r_e;
            for c in 0..r_e {
                b_blend[dst_start + c] = eff_weight * b[src_start + c];
            }
        }
        col_offset += r_e;
    }

    Ok(LoraLayer {
        a: a_blend,
        b: b_blend,
        d_in,
        d_out,
        rank: rank_total,
    })
}

#[cfg(test)]
mod tests;

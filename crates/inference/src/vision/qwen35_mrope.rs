//! Pure-Rust M-RoPE position-id and cos/sin table builder for Qwen3.5
//! vision-language decoding (ADR-069 Stage 5a).
//!
//! This module builds the per-physical-token `(t, h, w)` position triples
//! and the interleaved-axis cos/sin rotation tables that the six
//! full-attention GQA layers consume. It performs **no decoder work**: no
//! embedding substitution, no attention, no cache writes. Wiring this output
//! into the decoder forward pass is a separate stage.
//!
//! The algorithm and every numeric constant here were differentially
//! verified against a pinned HF `transformers` reference run (see the
//! stage's recon and probe artifacts) before implementation: the worked
//! text+image+text toy position table, an 82-token HF-probed position
//! table with `rope_delta`, the 32-lane interleaved cos/sin schedule, and
//! the decode-time position rule are all reproduced exactly by the tests
//! below.

use crate::error::InferenceError;
use crate::vision::qwen35_vit::GridThw;

/// Per-physical-token `(t, h, w)` M-RoPE position triples for one sequence,
/// plus the `rope_delta` used to resume position bookkeeping at decode time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MRopePositions {
    /// One `(t, h, w)` triple per physical token in the input sequence,
    /// in input order.
    pub positions: Vec<(u32, u32, u32)>,
    /// `max(position_ids) + 1 - physical_input_length`. Added to the
    /// physical KV-cache length at decode time to recover the logical
    /// M-RoPE coordinate (see [`decode_position`]).
    pub rope_delta: i64,
}

/// Per-physical-token cos/sin rotation rows, `rope_half` lanes each, built
/// from the interleaved T/H/W axis-selection schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct MRopeTables {
    /// `cos[token][lane]`, `lane` in `0..rope_half`.
    pub cos: Vec<Vec<f32>>,
    /// `sin[token][lane]`, `lane` in `0..rope_half`.
    pub sin: Vec<Vec<f32>>,
}

/// Build per-physical-token `(t, h, w)` M-RoPE position ids for an expanded
/// token stream containing zero or more image runs.
///
/// `input_ids` is the full, already-expanded decoder token stream (one
/// `image_token_id` entry per post-merger visual row — the processor's
/// expansion, not a placeholder count). `grids` supplies the unmerged
/// `(T, H, W)` patch-grid shape for each image run, in the order those runs
/// appear in `input_ids`.
///
/// Text runs advance all three axes together, one position per token. Each
/// image run consumes the next entry in `grids`, starts all axes at the
/// current position, sweeps the merged `(T, H/m, W/m)` grid row-major, and
/// advances the shared position counter by `max(H/m, W/m)` afterward — not
/// by the number of image-pad tokens consumed.
///
/// The advance is spatial-only. HF (v5.12.1) splits a video into `t = 1`
/// frame grids *before* position assignment and then advances by the spatial
/// maximum, so folding `T` into the advance here would diverge from the
/// reference on the very path it appears to serve. This API is image-only;
/// video re-enters with the video path, implemented against those semantics
/// and accepted on HF numeric parity.
///
/// Fails closed (`InvalidInput`) on: empty `input_ids`, zero merge size,
/// zero-dimension grids, grids not divisible by the merge size, image runs
/// whose length does not match their grid, missing or leftover grids, and
/// any arithmetic that would overflow the position space.
pub fn build_position_ids(
    input_ids: &[u32],
    image_token_id: u32,
    grids: &[GridThw],
    spatial_merge_size: usize,
) -> Result<MRopePositions, InferenceError> {
    if spatial_merge_size == 0 {
        return Err(InferenceError::InvalidInput(
            "spatial_merge_size must be > 0".to_string(),
        ));
    }
    if input_ids.is_empty() {
        return Err(InferenceError::InvalidInput(
            "input_ids must not be empty".to_string(),
        ));
    }
    let m = spatial_merge_size;

    fn advance(pos: u32, by: usize, what: &str) -> Result<u32, InferenceError> {
        u32::try_from(by)
            .ok()
            .and_then(|v| pos.checked_add(v))
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "position overflow advancing {what} by {by} from {pos}"
                ))
            })
    }

    let mut positions = Vec::with_capacity(input_ids.len());
    let mut current_pos: u32 = 0;
    let mut grid_idx = 0usize;
    let mut i = 0usize;

    while i < input_ids.len() {
        if input_ids[i] == image_token_id {
            let grid = *grids.get(grid_idx).ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "image-pad run at physical index {i} has no matching grid \
                     (only {} grid(s) supplied)",
                    grids.len()
                ))
            })?;
            grid_idx += 1;

            if grid.t == 0 || grid.h == 0 || grid.w == 0 {
                return Err(InferenceError::InvalidInput(format!(
                    "grid {grid:?} has a zero dimension"
                )));
            }
            if !grid.h.is_multiple_of(m) || !grid.w.is_multiple_of(m) {
                return Err(InferenceError::InvalidInput(format!(
                    "grid {grid:?} is not divisible by spatial_merge_size {m}"
                )));
            }
            let (lt, lh, lw) = (grid.t, grid.h / m, grid.w / m);
            let run_len = lt
                .checked_mul(lh)
                .and_then(|x| x.checked_mul(lw))
                .filter(|&len| len > 0)
                .ok_or_else(|| {
                    InferenceError::InvalidInput(format!(
                        "grid {grid:?} with m={m} yields an overflowing or zero merged run length"
                    ))
                })?;
            let run_end = i.checked_add(run_len).ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "image-pad run at physical index {i} with length {run_len} overflows"
                ))
            })?;

            if run_end > input_ids.len()
                || input_ids[i..run_end].iter().any(|&t| t != image_token_id)
            {
                return Err(InferenceError::InvalidInput(format!(
                    "image-pad run starting at physical index {i} does not have the \
                     expected length {run_len} (= T*H*W/m^2 for grid {grid:?}, m={m})"
                )));
            }

            for t in 0..lt {
                for h in 0..lh {
                    for w in 0..lw {
                        positions.push((
                            advance(current_pos, t, "image T axis")?,
                            advance(current_pos, h, "image H axis")?,
                            advance(current_pos, w, "image W axis")?,
                        ));
                    }
                }
            }

            current_pos = advance(current_pos, lh.max(lw), "post-image position")?;
            i = run_end;
        } else {
            positions.push((current_pos, current_pos, current_pos));
            current_pos = advance(current_pos, 1, "text position")?;
            i += 1;
        }
    }

    if grid_idx != grids.len() {
        return Err(InferenceError::InvalidInput(format!(
            "{} grid(s) supplied but only {grid_idx} image run(s) found in input_ids",
            grids.len()
        )));
    }

    let max_pos = positions
        .iter()
        .flat_map(|&(t, h, w)| [t, h, w])
        .max()
        .unwrap_or(0);
    let rope_delta = (max_pos as i64 + 1) - (input_ids.len() as i64);

    Ok(MRopePositions {
        positions,
        rope_delta,
    })
}

/// Build the interleaved-axis cos/sin rotation tables for every token in
/// `positions`.
///
/// `rope_half = (head_dim as f32 * partial_rotary_factor) as usize / 2`
/// (32 for Qwen3.5-0.8B). `mrope_section` must have exactly 3 entries
/// declaring the exact realized (T, H, W) lane counts, summing to `rope_half`.
/// Every lane starts on T; H overwrites lanes `1,4,...` while `lane / 3 < H`,
/// and W overwrites lanes `2,5,...` while `lane / 3 < W`. The declaration is
/// rejected unless this strided assignment realizes all three declared counts.
/// `inv_freq[i] = theta^(-2*i / rope_dim)`, `rope_dim = 2 * rope_half`.
pub fn build_cos_sin(
    positions: &MRopePositions,
    head_dim: usize,
    partial_rotary_factor: f32,
    theta: f32,
    mrope_section: &[usize],
) -> Result<MRopeTables, InferenceError> {
    if mrope_section.len() != 3 {
        return Err(InferenceError::InvalidInput(format!(
            "mrope_section must have exactly 3 entries (T,H,W), got {}",
            mrope_section.len()
        )));
    }

    if !theta.is_finite() || theta <= 0.0 {
        return Err(InferenceError::InvalidInput(format!(
            "theta must be finite and positive, got {theta}"
        )));
    }

    let rope_dim_exact = head_dim as f64 * f64::from(partial_rotary_factor);
    if !rope_dim_exact.is_finite()
        || rope_dim_exact <= 0.0
        || rope_dim_exact.fract() != 0.0
        || rope_dim_exact > head_dim as f64
    {
        return Err(InferenceError::InvalidInput(format!(
            "head_dim*partial_rotary_factor must be a positive integer <= head_dim, \
             got {rope_dim_exact} (head_dim={head_dim}, factor={partial_rotary_factor})"
        )));
    }
    let rope_dim = rope_dim_exact as usize;
    if !rope_dim.is_multiple_of(2) {
        return Err(InferenceError::InvalidInput(format!(
            "head_dim*partial_rotary_factor must be even, got {rope_dim}"
        )));
    }
    let rope_half = rope_dim / 2;

    let section_sum = mrope_section
        .iter()
        .try_fold(0usize, |acc, &c| acc.checked_add(c))
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!("mrope_section {mrope_section:?} sum overflows"))
        })?;
    if section_sum != rope_half {
        return Err(InferenceError::InvalidInput(format!(
            "mrope_section {mrope_section:?} sums to {section_sum}, expected rope_half={rope_half}"
        )));
    }

    let mut realized = [0usize; 3];
    for i in 0..rope_half {
        let axis = match (i % 3, i / 3) {
            (1, section_idx) if section_idx < mrope_section[1] => 1,
            (2, section_idx) if section_idx < mrope_section[2] => 2,
            _ => 0,
        };
        realized[axis] += 1;
    }
    if realized != mrope_section {
        return Err(InferenceError::InvalidInput(format!(
            "mrope_section {mrope_section:?} does not match realized {realized:?} \
             for rope_half={rope_half}"
        )));
    }

    let inv_freq: Vec<f32> = (0..rope_half)
        .map(|i| theta.powf(-2.0 * i as f32 / rope_dim as f32))
        .collect();

    let mut cos = Vec::with_capacity(positions.positions.len());
    let mut sin = Vec::with_capacity(positions.positions.len());

    for &(t, h, w) in &positions.positions {
        let mut cos_row = Vec::with_capacity(rope_half);
        let mut sin_row = Vec::with_capacity(rope_half);
        for i in 0..rope_half {
            let axis_val = match (i % 3, i / 3) {
                (1, section_idx) if section_idx < mrope_section[1] => h,
                (2, section_idx) if section_idx < mrope_section[2] => w,
                _ => t,
            };
            let angle = axis_val as f32 * inv_freq[i];
            cos_row.push(angle.cos());
            sin_row.push(angle.sin());
        }
        cos.push(cos_row);
        sin.push(sin_row);
    }

    Ok(MRopeTables { cos, sin })
}

/// The decode-time M-RoPE coordinate (all three axes equal) for the next
/// generated token, given the physical KV-cache length and the prefill's
/// `rope_delta`. Physical cache length counts every physical token
/// (including image pads and delimiters) and is not itself a RoPE
/// coordinate. Errors when the result is negative or exceeds `u32::MAX`.
pub fn decode_position(physical_cache_len: usize, rope_delta: i64) -> Result<u32, InferenceError> {
    let len = i64::try_from(physical_cache_len).map_err(|_| {
        InferenceError::InvalidInput(format!(
            "physical_cache_len={physical_cache_len} is not representable as i64"
        ))
    })?;
    let raw = len.checked_add(rope_delta).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "decode position overflow: physical_cache_len={physical_cache_len} + \
             rope_delta={rope_delta}"
        ))
    })?;
    u32::try_from(raw).map_err(|_| {
        InferenceError::InvalidInput(format!(
            "decode position {raw} (physical_cache_len={physical_cache_len} + \
             rope_delta={rope_delta}) is not representable as a u32 coordinate"
        ))
    })
}

#[cfg(test)]
mod tests;

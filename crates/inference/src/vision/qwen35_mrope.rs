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
/// advances the shared position counter by `max(H, W) / m` afterward — not
/// by the number of image-pad tokens consumed. See RECON sec. 2
/// "Position-id construction".
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
    let m = spatial_merge_size;

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

            if !grid.h.is_multiple_of(m) || !grid.w.is_multiple_of(m) {
                return Err(InferenceError::InvalidInput(format!(
                    "grid {grid:?} is not divisible by spatial_merge_size {m}"
                )));
            }
            let (lt, lh, lw) = (grid.t, grid.h / m, grid.w / m);
            let run_len = lt * lh * lw;
            let run_end = i + run_len;

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
                            current_pos + t as u32,
                            current_pos + h as u32,
                            current_pos + w as u32,
                        ));
                    }
                }
            }

            current_pos += lh.max(lw) as u32;
            i = run_end;
        } else {
            positions.push((current_pos, current_pos, current_pos));
            current_pos += 1;
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
/// (T, H, W lane counts) summing to `rope_half`; lane `i` selects axis
/// `i % 3` (the cyclic `T,H,W,T,H,W,...` schedule — RECON sec. 2 confirms
/// this reduces to exactly the given per-axis counts for `[11,11,10]`).
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

    let rope_dim = (head_dim as f32 * partial_rotary_factor) as usize;
    if rope_dim == 0 || !rope_dim.is_multiple_of(2) {
        return Err(InferenceError::InvalidInput(format!(
            "head_dim*partial_rotary_factor must be a positive even integer, got {rope_dim}"
        )));
    }
    let rope_half = rope_dim / 2;

    let section_sum: usize = mrope_section.iter().sum();
    if section_sum != rope_half {
        return Err(InferenceError::InvalidInput(format!(
            "mrope_section {mrope_section:?} sums to {section_sum}, expected rope_half={rope_half}"
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
            let axis_val = match i % 3 {
                0 => t,
                1 => h,
                _ => w,
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
/// coordinate — see RECON sec. 3.
pub fn decode_position(physical_cache_len: usize, rope_delta: i64) -> Result<u32, InferenceError> {
    let raw = physical_cache_len as i64 + rope_delta;
    u32::try_from(raw).map_err(|_| {
        InferenceError::InvalidInput(format!(
            "decode position underflow: physical_cache_len={physical_cache_len} + \
             rope_delta={rope_delta} = {raw} is negative"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {b}, got {a} (diff {})",
            (a - b).abs()
        );
    }

    // ---- Test 1: RECON worked toy (sec. 2) ----
    // A B <vs> <img><img><img><img> <ve> C D, grid (1,4,4), m=2.
    const VISION_START: u32 = 900;
    const VISION_END: u32 = 901;
    const IMAGE_PAD: u32 = 902;
    const TOKEN_A: u32 = 1;
    const TOKEN_B: u32 = 2;
    const TOKEN_C: u32 = 3;
    const TOKEN_D: u32 = 4;

    #[test]
    fn recon_worked_toy_table() {
        let input_ids = [
            TOKEN_A,
            TOKEN_B,
            VISION_START,
            IMAGE_PAD,
            IMAGE_PAD,
            IMAGE_PAD,
            IMAGE_PAD,
            VISION_END,
            TOKEN_C,
            TOKEN_D,
        ];
        let grids = [GridThw { t: 1, h: 4, w: 4 }];
        let result = build_position_ids(&input_ids, IMAGE_PAD, &grids, 2).unwrap();

        let expected = [
            (0, 0, 0),
            (1, 1, 1),
            (2, 2, 2),
            (3, 3, 3),
            (3, 3, 4),
            (3, 4, 3),
            (3, 4, 4),
            (5, 5, 5),
            (6, 6, 6),
            (7, 7, 7),
        ];
        assert_eq!(result.positions, expected);
        // trailing text resumes at 5, confirmed by the last three triples.
        assert_eq!(result.positions[7].0, 5);
    }

    // ---- Test 2: HF probe golden (probe_positions_result.json) ----
    // 4 text tokens, image run of 64 (grid 1,16,16 unmerged, m=2 -> merged
    // 1,8,8), then 14 trailing text tokens. 82 tokens total.
    fn probe_golden_input_ids() -> Vec<u32> {
        let mut ids = vec![TOKEN_A; 4];
        ids.extend(std::iter::repeat_n(IMAGE_PAD, 64));
        ids.extend(vec![TOKEN_A; 14]);
        ids
    }

    #[test]
    fn hf_probe_golden_positions() {
        let input_ids = probe_golden_input_ids();
        assert_eq!(input_ids.len(), 82);
        let grids = [GridThw { t: 1, h: 16, w: 16 }];
        let result = build_position_ids(&input_ids, IMAGE_PAD, &grids, 2).unwrap();

        assert_eq!(result.positions.len(), 82);
        // Text prefix 0-3.
        assert_eq!(result.positions[0], (0, 0, 0));
        assert_eq!(result.positions[1], (1, 1, 1));
        assert_eq!(result.positions[2], (2, 2, 2));
        assert_eq!(result.positions[3], (3, 3, 3));
        // First image pad (physical index 4) -> (4,4,4).
        assert_eq!(result.positions[4], (4, 4, 4));
        // physical index 5 -> (4,4,5) per golden W-sweep.
        assert_eq!(result.positions[5], (4, 4, 5));
        // physical index 12 -> next H row: (4,5,4).
        assert_eq!(result.positions[12], (4, 5, 4));
        // physical index 67 (last image pad) -> (4,11,11).
        assert_eq!(result.positions[67], (4, 11, 11));
        // trailing text resumes at 12, not the physical index 68.
        assert_eq!(result.positions[68], (12, 12, 12));
        assert_eq!(result.positions[69], (13, 13, 13));
        assert_eq!(result.positions[70], (14, 14, 14));
        assert_eq!(result.positions[71], (15, 15, 15));

        assert_eq!(result.rope_delta, -56);

        let decoded = decode_position(82, result.rope_delta).unwrap();
        assert_eq!(decoded, 26);
    }

    // ---- Test 3: lane schedule with section [11,11,10] ----
    // Drives build_cos_sin itself (not a restatement of the formula) with
    // theta=1.0 -- inv_freq[i] = 1.0^anything = 1.0 for every lane, so
    // angle == axis_val exactly and cos/sin directly reveal which axis a
    // lane selected. T/H/W are distinct small values chosen so their
    // cos values are pairwise well-separated (no aliasing near the 1e-4
    // tolerance).
    #[test]
    fn lane_schedule_matches_section_counts() {
        let section = [11usize, 11, 10];
        let (t, h, w) = (2u32, 3u32, 5u32);
        let positions = MRopePositions {
            positions: vec![(t, h, w)],
            rope_delta: 0,
        };
        let tables = build_cos_sin(&positions, 256, 0.25, 1.0, &section).unwrap();

        for lane in 0..32usize {
            let expected_axis = match lane % 3 {
                0 => t,
                1 => h,
                _ => w,
            };
            let expected_cos = (expected_axis as f32).cos();
            let expected_sin = (expected_axis as f32).sin();
            assert_close(tables.cos[0][lane], expected_cos, 1e-4);
            assert_close(tables.sin[0][lane], expected_sin, 1e-4);
        }

        // T lanes: 0,3,...,30 (11). H lanes: 1,4,...,31 (11). W lanes:
        // 2,5,...,29 (10) -- exactly the counts in `section`.
        let t_lanes: Vec<usize> = (0..32).filter(|i| i % 3 == 0).collect();
        let h_lanes: Vec<usize> = (0..32).filter(|i| i % 3 == 1).collect();
        let w_lanes: Vec<usize> = (0..32).filter(|i| i % 3 == 2).collect();
        assert_eq!(t_lanes.len(), section[0]);
        assert_eq!(h_lanes.len(), section[1]);
        assert_eq!(w_lanes.len(), section[2]);
        assert_eq!(t_lanes, (0..=30).step_by(3).collect::<Vec<_>>());
        assert_eq!(h_lanes, (1..=31).step_by(3).collect::<Vec<_>>());
        assert_eq!(w_lanes, (2..=29).step_by(3).collect::<Vec<_>>());
    }

    // ---- Test 4: cos/sin numerics vs probe_mrope_lanes_result.json ----
    #[test]
    fn cos_sin_numerics_match_hf_probe() {
        let positions = MRopePositions {
            positions: vec![(4, 4, 4)],
            rope_delta: 0,
        };
        let tables = build_cos_sin(&positions, 256, 0.25, 1e7, &[11, 11, 10]).unwrap();
        assert_close(tables.cos[0][0], -0.653644, 1e-4);
        assert_close(tables.sin[0][0], -0.756802, 1e-4);
        assert_close(tables.cos[0][1], -0.748892, 1e-4);
        assert_close(tables.cos[0][2], 0.109877, 1e-4);
        assert_close(tables.cos[0][3], 0.635073, 1e-4);
    }

    // ---- Test 5: text-only reduction ----
    #[test]
    fn text_only_reduces_to_1d_table() {
        let input_ids = [TOKEN_A, TOKEN_B, TOKEN_C, TOKEN_D];
        let result = build_position_ids(&input_ids, IMAGE_PAD, &[], 2).unwrap();
        for (idx, &(t, h, w)) in result.positions.iter().enumerate() {
            assert_eq!(t as usize, idx);
            assert_eq!(h as usize, idx);
            assert_eq!(w as usize, idx);
        }
        assert_eq!(result.rope_delta, 0);

        let theta = 1e7_f32;
        let head_dim = 256;
        let partial_rotary_factor = 0.25;
        let section = [11usize, 11, 10];
        let tables =
            build_cos_sin(&result, head_dim, partial_rotary_factor, theta, &section).unwrap();

        let rope_dim = (head_dim as f32 * partial_rotary_factor) as usize;
        let rope_half = rope_dim / 2;
        for (token_idx, &(t, h, w)) in result.positions.iter().enumerate() {
            assert_eq!(t, h);
            assert_eq!(h, w);
            for lane in 0..rope_half {
                let inv_freq = theta.powf(-2.0 * lane as f32 / rope_dim as f32);
                let expected_angle = t as f32 * inv_freq;
                assert_close(tables.cos[token_idx][lane], expected_angle.cos(), 1e-5);
                assert_close(tables.sin[token_idx][lane], expected_angle.sin(), 1e-5);
            }
        }
    }

    // ---- Test 6: fail-closed negatives ----
    #[test]
    fn rejects_image_run_length_mismatch() {
        // grid (1,4,4) with m=2 expects 4 image-pad tokens; supply 3.
        let input_ids = [IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, TOKEN_A];
        let grids = [GridThw { t: 1, h: 4, w: 4 }];
        let err = build_position_ids(&input_ids, IMAGE_PAD, &grids, 2).unwrap_err();
        assert!(matches!(err, InferenceError::InvalidInput(_)));
    }

    #[test]
    fn rejects_leftover_grids() {
        let input_ids = [TOKEN_A, TOKEN_B];
        let grids = [GridThw { t: 1, h: 4, w: 4 }];
        let err = build_position_ids(&input_ids, IMAGE_PAD, &grids, 2).unwrap_err();
        assert!(matches!(err, InferenceError::InvalidInput(_)));
    }

    #[test]
    fn rejects_missing_grid_for_image_run() {
        let input_ids = [IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD];
        let err = build_position_ids(&input_ids, IMAGE_PAD, &[], 2).unwrap_err();
        assert!(matches!(err, InferenceError::InvalidInput(_)));
    }

    #[test]
    fn rejects_zero_merge_size() {
        let input_ids = [TOKEN_A];
        let err = build_position_ids(&input_ids, IMAGE_PAD, &[], 0).unwrap_err();
        assert!(matches!(err, InferenceError::InvalidInput(_)));
    }

    #[test]
    fn rejects_mrope_section_sum_mismatch() {
        let positions = MRopePositions {
            positions: vec![(0, 0, 0)],
            rope_delta: 0,
        };
        let err = build_cos_sin(&positions, 256, 0.25, 1e7, &[10, 10, 10]).unwrap_err();
        assert!(matches!(err, InferenceError::InvalidInput(_)));
    }

    #[test]
    fn rejects_mrope_section_wrong_axis_count() {
        let positions = MRopePositions {
            positions: vec![(0, 0, 0)],
            rope_delta: 0,
        };
        let err = build_cos_sin(&positions, 256, 0.25, 1e7, &[16, 16]).unwrap_err();
        assert!(matches!(err, InferenceError::InvalidInput(_)));
    }

    #[test]
    fn decode_position_rejects_negative() {
        let err = decode_position(0, -5).unwrap_err();
        assert!(matches!(err, InferenceError::InvalidInput(_)));
    }
}

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

#[test]
fn post_image_advance_is_spatial_only_even_when_t_is_largest() {
    // t=3 is deliberately larger than the merged spatial extents
    // (h/m = w/m = 2) so the two conventions give different answers:
    // spatial-only lands the next text token at 5 + max(2, 2) = 7,
    // folding T in would land it at 5 + max(3, 2, 2) = 8.
    //
    // HF (v5.12.1) splits a video into t = 1 frame grids *before*
    // assigning positions and then advances by the spatial maximum, so
    // spatial-only is the reference-aligned answer. This asserts the
    // discriminating case, not merely a case both conventions satisfy.
    let mut input_ids = vec![TOKEN_A; 5];
    input_ids.extend(std::iter::repeat_n(IMAGE_PAD, 12));
    input_ids.push(TOKEN_B);

    let grids = [GridThw { t: 3, h: 4, w: 4 }];
    let result = build_position_ids(&input_ids, IMAGE_PAD, &grids, 2).unwrap();

    assert_eq!(result.positions[5], (5, 5, 5));
    // last image-pad token: inside the sweep, unaffected by the advance
    assert_eq!(result.positions[16], (7, 6, 6));
    // the token after the image run: this is the discriminator
    assert_eq!(result.positions[17], (7, 7, 7));
}

fn assert_hf_lane_schedule(section: [usize; 3], rope_half: usize, expected_counts: [usize; 3]) {
    let (t, h, w) = (2u32, 3u32, 5u32);
    let positions = MRopePositions {
        positions: vec![(t, h, w)],
        rope_delta: 0,
    };
    let tables = build_cos_sin(&positions, rope_half * 2, 1.0, 1.0, &section).unwrap();

    let mut expected_axes = vec![0usize; rope_half];
    for (axis, offset) in [(1usize, 1usize), (2, 2)] {
        let end = (section[axis] * 3).min(rope_half);
        for lane in (offset..end).step_by(3) {
            expected_axes[lane] = axis;
        }
    }

    let actual_counts = [
        expected_axes.iter().filter(|&&axis| axis == 0).count(),
        expected_axes.iter().filter(|&&axis| axis == 1).count(),
        expected_axes.iter().filter(|&&axis| axis == 2).count(),
    ];
    assert_eq!(actual_counts, expected_counts);

    for (lane, axis) in expected_axes.into_iter().enumerate() {
        let expected_axis = match axis {
            0 => t,
            1 => h,
            2 => w,
            _ => unreachable!(),
        };
        let expected_cos = (expected_axis as f32).cos();
        let expected_sin = (expected_axis as f32).sin();
        assert_close(tables.cos[0][lane], expected_cos, 1e-4);
        assert_close(tables.sin[0][lane], expected_sin, 1e-4);
    }
}

#[test]
fn lane_schedule_matches_hf_saturating_overwrite() {
    for (section, rope_half, expected_counts) in [
        ([11, 11, 10], 32, [11, 11, 10]),
        ([22, 21, 21], 64, [22, 21, 21]),
        ([20, 6, 6], 32, [20, 6, 6]),
    ] {
        assert_hf_lane_schedule(section, rope_half, expected_counts);
    }
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
    let tables = build_cos_sin(&result, head_dim, partial_rotary_factor, theta, &section).unwrap();

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

#[test]
fn rejects_zero_dimension_grid_instead_of_looping() {
    // A zero-sized grid used to produce a zero-length run that never
    // advanced the scan cursor (infinite loop). Must fail closed.
    for grid in [
        GridThw { t: 0, h: 2, w: 2 },
        GridThw { t: 1, h: 0, w: 2 },
        GridThw { t: 1, h: 2, w: 0 },
    ] {
        let err = build_position_ids(&[IMAGE_PAD], IMAGE_PAD, &[grid], 2).unwrap_err();
        assert!(matches!(err, InferenceError::InvalidInput(_)));
    }
}

#[test]
fn rejects_overflowing_grid_arithmetic() {
    // run_end = i + run_len must not wrap: usize::MAX * 1 * 1 run.
    let input_ids = [TOKEN_A, IMAGE_PAD];
    let grids = [GridThw {
        t: usize::MAX,
        h: 1,
        w: 1,
    }];
    let err = build_position_ids(&input_ids, IMAGE_PAD, &grids, 1).unwrap_err();
    assert!(matches!(err, InferenceError::InvalidInput(_)));

    // lt * lh * lw itself must not wrap either.
    let grids = [GridThw {
        t: usize::MAX,
        h: 2,
        w: 2,
    }];
    let err = build_position_ids(&input_ids, IMAGE_PAD, &grids, 1).unwrap_err();
    assert!(matches!(err, InferenceError::InvalidInput(_)));
}

#[test]
fn rejects_empty_input_ids() {
    let err = build_position_ids(&[], IMAGE_PAD, &[], 2).unwrap_err();
    assert!(matches!(err, InferenceError::InvalidInput(_)));
}

#[test]
fn rejects_fractional_rope_dim() {
    // 256 * 0.3 = 76.8 must error, not silently truncate to 76.
    let positions = MRopePositions {
        positions: vec![(0, 0, 0)],
        rope_delta: 0,
    };
    let err = build_cos_sin(&positions, 256, 0.3, 1e7, &[13, 13, 12]).unwrap_err();
    assert!(matches!(err, InferenceError::InvalidInput(_)));
}

#[test]
fn rejects_non_finite_or_non_positive_theta() {
    let positions = MRopePositions {
        positions: vec![(0, 0, 0)],
        rope_delta: 0,
    };
    for theta in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
        let err = build_cos_sin(&positions, 256, 0.25, theta, &[11, 11, 10]).unwrap_err();
        assert!(matches!(err, InferenceError::InvalidInput(_)));
    }
}

#[test]
fn rejects_mrope_section_sum_overflow() {
    let positions = MRopePositions {
        positions: vec![(0, 0, 0)],
        rope_delta: 0,
    };
    let err = build_cos_sin(&positions, 256, 0.25, 1e7, &[usize::MAX, 1, 1]).unwrap_err();
    assert!(matches!(err, InferenceError::InvalidInput(_)));
}

#[test]
fn decode_position_rejects_out_of_range() {
    // usize::MAX must not wrap through i64 into a small Ok value.
    let err = decode_position(usize::MAX, 2).unwrap_err();
    assert!(matches!(err, InferenceError::InvalidInput(_)));
    // A positive result above u32::MAX is out of range, not "negative".
    let err = decode_position(u32::MAX as usize + 10, 0).unwrap_err();
    assert!(matches!(err, InferenceError::InvalidInput(_)));
}

#[test]
fn shipped_config_tables_are_bit_identical() {
    let cfg = crate::model::qwen35_config::Qwen35Config::from_config_json_str(include_str!(
        "../../../tests/fixtures/qwen35_0_8b_config.json"
    ))
    .unwrap();
    let params = cfg.rope_parameters.as_ref().unwrap();
    let section = params.mrope_section.as_ref().unwrap();
    assert_eq!(section, &[11, 11, 10]);
    let positions = MRopePositions {
        positions: vec![(0, 0, 0), (2, 3, 5), (4, 11, 11)],
        rope_delta: 0,
    };
    let tables = build_cos_sin(
        &positions,
        cfg.head_dim,
        params.partial_rotary_factor.unwrap(),
        params.rope_theta as f32,
        section,
    )
    .unwrap();
    let bits: Vec<u32> = tables
        .cos
        .iter()
        .chain(&tables.sin)
        .flatten()
        .map(|x| x.to_bits())
        .collect();
    let rope_dim = cfg.head_dim as f32 * params.partial_rotary_factor.unwrap();
    let mut expected_cos = Vec::new();
    let mut expected_sin = Vec::new();
    for &(t, h, w) in &positions.positions {
        for lane in 0..32 {
            let inv_freq = (params.rope_theta as f32).powf(-2.0 * lane as f32 / rope_dim);
            let angle = [t, h, w][lane % 3] as f32 * inv_freq;
            expected_cos.push(angle.cos().to_bits());
            expected_sin.push(angle.sin().to_bits());
        }
    }
    expected_cos.extend(expected_sin);
    assert_eq!(bits, expected_cos);
}

fn assert_rejects_unrealizable_section(section: [usize; 3], realized: [usize; 3]) {
    let rope_half: usize = section.iter().sum();
    for positions in [vec![], vec![(2, 3, 5)]] {
        let positions = MRopePositions {
            positions,
            rope_delta: 0,
        };
        let err = build_cos_sin(&positions, rope_half * 2, 1.0, 1e7, &section)
            .expect_err("declared axis counts must match the realized schedule");
        let InferenceError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains(&format!("{section:?}")), "{message}");
        assert!(
            message.contains(&format!("realized {realized:?}")),
            "{message}"
        );
        assert!(
            message.contains(&format!("rope_half={rope_half}")),
            "{message}"
        );
    }
}

#[test]
fn rejects_unrealizable_section_width_cap() {
    assert_rejects_unrealizable_section([10, 11, 11], [11, 11, 10]);
}

#[test]
fn rejects_unrealizable_section_temporal_remainder() {
    assert_rejects_unrealizable_section([11, 10, 11], [12, 10, 10]);
}

#[test]
fn rejects_unrealizable_section_zero_temporal() {
    assert_rejects_unrealizable_section([0, 16, 16], [11, 11, 10]);
}

#[test]
fn rejects_unrealizable_section_saturated_spatial_caps() {
    assert_rejects_unrealizable_section([16, 24, 24], [22, 21, 21]);
}

use super::*;
use crate::vision::qwen35_vit::GridThw;

fn base_request() -> Qwen35VisionRequest {
    // One image run: grid (1,4,4), merge_size=2 -> 4 post-merger rows.
    let mut input_ids = vec![10u32, 11, 248_053];
    input_ids.extend(std::iter::repeat_n(248_056u32, 4));
    input_ids.push(248_054);
    input_ids.push(12);
    Qwen35VisionRequest {
        input_ids,
        image_grids: vec![GridThw { t: 1, h: 4, w: 4 }],
        post_merger_rows: vec![0.1f32; 4 * 8],
        image_token_id: 248_056,
        spatial_merge_size: 2,
        decoder_hidden_size: 8,
    }
}

#[test]
fn valid_request_passes() {
    let req = base_request();
    assert_eq!(req.image_pad_count(), 4);
    req.validate().expect("valid request");
}

#[test]
fn zero_decoder_hidden_size_rejected() {
    let mut req = base_request();
    req.decoder_hidden_size = 0;
    assert!(req.validate().is_err());
}

#[test]
fn zero_spatial_merge_size_rejected() {
    let mut req = base_request();
    req.spatial_merge_size = 0;
    assert!(req.validate().is_err());
}

#[test]
fn empty_input_ids_rejected() {
    let mut req = base_request();
    req.input_ids.clear();
    assert!(req.validate().is_err());
}

#[test]
fn zero_dim_grid_rejected() {
    let mut req = base_request();
    req.image_grids[0] = GridThw { t: 0, h: 4, w: 4 };
    assert!(req.validate().is_err());
}

#[test]
fn row_times_hidden_overflow_rejected() {
    let mut req = base_request();
    // Force the rows*hidden_size product past usize::MAX.
    req.decoder_hidden_size = usize::MAX / 2;
    assert!(req.validate().is_err());
}

#[test]
fn grid_patch_count_overflow_rejected() {
    let mut req = base_request();
    req.image_grids[0] = GridThw {
        t: usize::MAX,
        h: usize::MAX,
        w: 2,
    };
    assert!(req.validate().is_err());
}

#[test]
fn non_finite_visual_scalar_rejected_nan() {
    let mut req = base_request();
    req.post_merger_rows[0] = f32::NAN;
    assert!(req.validate().is_err());
}

#[test]
fn non_finite_visual_scalar_rejected_inf() {
    let mut req = base_request();
    req.post_merger_rows[3] = f32::INFINITY;
    assert!(req.validate().is_err());
}

#[test]
fn empty_grid_list_with_no_images_is_valid() {
    let req = Qwen35VisionRequest {
        input_ids: vec![1, 2, 3],
        image_grids: vec![],
        post_merger_rows: vec![],
        image_token_id: 248_056,
        spatial_merge_size: 2,
        decoder_hidden_size: 8,
    };
    req.validate().expect("text-only request is valid");
}

#[test]
fn grid_list_length_mismatched_against_image_runs_rejected() {
    let mut req = base_request();
    // Two grids supplied but only one contiguous image-pad run exists.
    req.image_grids.push(GridThw { t: 1, h: 2, w: 2 });
    assert!(req.validate().is_err());
}

#[test]
fn mismatched_pad_count_vs_grid_rejected() {
    let mut req = base_request();
    // Drop one image-pad token so the run no longer matches T*H*W/m^2.
    let first_pad = req
        .input_ids
        .iter()
        .position(|&id| id == req.image_token_id)
        .expect("has a pad token");
    req.input_ids.remove(first_pad);
    assert!(req.validate().is_err());
}

#[test]
fn post_merger_buffer_wrong_length_rejected() {
    let mut req = base_request();
    req.post_merger_rows.pop();
    assert!(req.validate().is_err());
}

#[test]
fn grid_not_divisible_by_merge_size_rejected() {
    let mut req = base_request();
    req.image_grids[0] = GridThw { t: 1, h: 3, w: 4 };
    // 3 is not divisible by spatial_merge_size=2.
    assert!(req.validate().is_err());
}

/// HF-probed golden (probes/RESULTS.md, probe 2 + probe 3): an 82-token
/// prefill with a text prefix of length 4, then an image with unmerged
/// grid `(1,16,16)` / merge_size=2 (merged LLM grid `1,8,8`, 64
/// post-merger rows). The first image-pad physical token lands at
/// M-RoPE position `(4,4,4)`, and its 32-lane cos/sin row matches the
/// checkpoint's `theta=1e7`, `partial_rotary_factor=0.25`,
/// `mrope_section=[11,11,10]` config exactly (float32-level agreement).
/// This test drives the request's `build_mrope_tables` wiring method,
/// not the already-covered S5a unit functions directly.
#[test]
fn build_mrope_tables_matches_hf_probe_golden_through_wired_path() {
    use crate::model::qwen35_config::{Qwen35Config, RopeParams};

    let mut input_ids: Vec<u32> = vec![10, 11, 248_053, 12];
    input_ids.extend(std::iter::repeat_n(248_056u32, 64));
    input_ids.push(248_054);

    let req = Qwen35VisionRequest {
        input_ids,
        image_grids: vec![GridThw { t: 1, h: 16, w: 16 }],
        post_merger_rows: vec![0.0f32; 64 * 1024],
        image_token_id: 248_056,
        spatial_merge_size: 2,
        decoder_hidden_size: 1024,
    };
    req.validate().expect("probe-shaped request is valid");

    let cfg = Qwen35Config {
        head_dim: 256,
        rope_parameters: Some(RopeParams {
            rope_theta: 1.0e7,
            partial_rotary_factor: Some(0.25),
            mrope_section: Some(vec![11, 11, 10]),
            mrope_interleaved: Some(true),
        }),
        ..Qwen35Config::default()
    };

    let (positions, tables) = req.build_mrope_tables(&cfg).expect("builds tables");

    // Probe 2: first image-pad token (physical index 4) lands at (4,4,4).
    assert_eq!(positions.positions[4], (4, 4, 4));

    // Probe 3: 32-lane cos/sin row at that position, float32-level match.
    let cos = &tables.cos[4];
    let sin = &tables.sin[4];
    assert!((cos[0] - (-0.653_644)).abs() < 1e-4, "cos[0]={}", cos[0]);
    assert!((sin[0] - (-0.756_802)).abs() < 1e-4, "sin[0]={}", sin[0]);
    assert!((cos[1] - (-0.748_892)).abs() < 1e-4, "cos[1]={}", cos[1]);
    assert!((cos[2] - 0.109_877).abs() < 1e-4, "cos[2]={}", cos[2]);
    assert!((cos[3] - 0.635_073).abs() < 1e-4, "cos[3]={}", cos[3]);
}

/// Text-only non-regression, driven through `build_mrope_tables` (the
/// new decoder-facing wiring), not just the already-covered S5a unit
/// (`text_only_reduces_to_1d_table` in `qwen35_mrope`): with no image
/// runs, every physical token has all three M-RoPE axes equal to its
/// sequential position, so the interleaved axis-selection collapses to
/// plain 1-D stride-half RoPE — `cos[lane] = cos(pos * theta^(-2*lane/64))`
/// for every lane regardless of which axis it nominally reads from.
#[test]
fn build_mrope_tables_text_only_collapses_to_1d() {
    use crate::model::qwen35_config::{Qwen35Config, RopeParams};

    let req = Qwen35VisionRequest {
        input_ids: vec![5, 6, 7, 8, 9],
        image_grids: vec![],
        post_merger_rows: vec![],
        image_token_id: 248_056,
        spatial_merge_size: 2,
        decoder_hidden_size: 1024,
    };
    req.validate().expect("text-only request is valid");

    let cfg = Qwen35Config {
        head_dim: 256,
        rope_parameters: Some(RopeParams {
            rope_theta: 1.0e7,
            partial_rotary_factor: Some(0.25),
            mrope_section: Some(vec![11, 11, 10]),
            mrope_interleaved: Some(true),
        }),
        ..Qwen35Config::default()
    };

    let (positions, tables) = req.build_mrope_tables(&cfg).expect("builds tables");

    for (token_idx, &(t, h, w)) in positions.positions.iter().enumerate() {
        assert_eq!(t, token_idx as u32);
        assert_eq!(h, token_idx as u32);
        assert_eq!(w, token_idx as u32);

        let pos = f64::from(token_idx as u32);
        for lane in 0..32usize {
            let inv_freq = 1.0e7_f64.powf(-2.0 * lane as f64 / 64.0);
            let angle = pos * inv_freq;
            let expected_cos = angle.cos() as f32;
            let expected_sin = angle.sin() as f32;
            assert!(
                (tables.cos[token_idx][lane] - expected_cos).abs() < 1e-4,
                "token {token_idx} lane {lane}: cos {} vs expected {expected_cos}",
                tables.cos[token_idx][lane]
            );
            assert!(
                (tables.sin[token_idx][lane] - expected_sin).abs() < 1e-4,
                "token {token_idx} lane {lane}: sin {} vs expected {expected_sin}",
                tables.sin[token_idx][lane]
            );
        }
    }
}

#[test]
fn build_mrope_tables_rejects_config_without_mrope_section() {
    use crate::model::qwen35_config::Qwen35Config;

    let req = base_request();
    let cfg = Qwen35Config::default(); // no rope_parameters -> no mrope_section
    assert!(req.build_mrope_tables(&cfg).is_err());
}

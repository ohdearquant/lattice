//! Bit-exact goldens for the PaddleOCR-VL image preprocessing pipeline.
//!
//! Holds `lattice_inference::vision::paddleocr_preprocess` against the
//! committed fixture `fixtures/paddleocr_vl/preprocess/preprocess_goldens.json`,
//! produced with PIL 12.3.0 on a synthetic HWC uint8 image. No model
//! checkpoint and no feature flag are required; the test runs under default
//! features on pure CPU.
//!
//! Input image formula (must reproduce `input_sha256` per case):
//! `pixel[y][x][c] = (x*7 + y*13 + c*101 + (x*y) % 29 + (x/3)*(c+1)) % 256`
//! with integer division.

use lattice_inference::vision::paddleocr_preprocess::{
    PaddleOcrImageProcessorConfig, preprocess_rgb8, resize_bicubic_rgb8, smart_resize,
};
use sha2::{Digest, Sha256};
const FIXTURE: &str = "tests/fixtures/paddleocr_vl/preprocess/preprocess_goldens.json";

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn golden_input_rgb(height: usize, width: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(height * width * 3);
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let v = (x * 7 + y * 13 + c * 101 + (x * y) % 29 + (x / 3) * (c + 1)) % 256;
                out.push(v as u8);
            }
        }
    }
    out
}

/// Reference patch extraction for the goldens: reslice the (sha256-verified)
/// resized uint8 bytes with the spec's layout — raster patch order, each
/// patch `[c, py, px]`, f32 rescale by 1/255 and per-channel normalize —
/// without going through the module under test. Used to pin the patch order:
/// the fixture's first/last patch values are identical under raster and
/// column-major order, so only non-corner patches distinguish them.
fn reference_patch(
    resized: &[u8],
    dst_w: usize,
    py: usize,
    px: usize,
    cfg: &PaddleOcrImageProcessorConfig,
) -> Vec<f32> {
    let patch = cfg.patch_size;
    let mut out = Vec::with_capacity(3 * patch * patch);
    for c in 0..3 {
        for yy in 0..patch {
            for xx in 0..patch {
                let byte = resized[((py * patch + yy) * dst_w + px * patch + xx) * 3 + c];
                let v = (byte as f64 * cfg.rescale_factor) as f32;
                out.push((v - cfg.image_mean[c]) / cfg.image_std[c]);
            }
        }
    }
    out
}

fn expect_exact(label: &str, got: f32, want: f64) {
    assert!(
        got == want as f32,
        "{label}: got {got:?} ({:?}), want {want:?} ({:?})",
        got.to_bits(),
        (want as f32).to_bits()
    );
}

fn expect_one_ulp(label: &str, got: f32, want: f64) {
    // The fixture records the reference's vectorized float32 aggregate. A
    // scalar traversal can choose a different reduction order by one ULP;
    // sampled tensor values remain exact bit comparisons.
    let want = want as f32;
    assert!(
        got.to_bits().abs_diff(want.to_bits()) <= 1,
        "{label}: got {got:?} ({:?}), want {want:?} ({:?}) differ by more than one f32 ULP",
        got.to_bits(),
        want.to_bits()
    );
}

fn scalar_mean_abs(values: &[f32]) -> f32 {
    (values.iter().map(|value| value.abs() as f64).sum::<f64>() / values.len() as f64) as f32
}

#[test]
fn paddleocr_vl_preprocess_goldens() {
    let fixture_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let raw = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", fixture_path.display()));
    let json: serde_json::Value = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("cannot parse fixture {FIXTURE}: {e}"));

    let pil_version = json["pil_version"].as_str().expect("pil_version string");
    assert!(
        pil_version == "12.3.0",
        "unexpected pil_version {pil_version}"
    );

    let cases = json["cases"]
        .as_array()
        .expect("cases array in fixture")
        .clone();
    assert!(!cases.is_empty(), "fixture has no cases");

    let cfg = PaddleOcrImageProcessorConfig::paddleocr_vl_defaults();

    for case in &cases {
        let id = case["id"].as_str().expect("case id").to_string();
        let input_hw: [usize; 2] = {
            let arr = case["input_hw"].as_array().expect("input_hw array");
            [
                arr[0].as_u64().expect("input_hw[0]") as usize,
                arr[1].as_u64().expect("input_hw[1]") as usize,
            ]
        };
        let (height, width) = (input_hw[0], input_hw[1]);
        let resized_hw: [usize; 2] = {
            let arr = case["resized_hw"].as_array().expect("resized_hw array");
            [
                arr[0].as_u64().expect("resized_hw[0]") as usize,
                arr[1].as_u64().expect("resized_hw[1]") as usize,
            ]
        };
        let grid_thw: [usize; 3] = {
            let arr = case["grid_thw"].as_array().expect("grid_thw array");
            [
                arr[0].as_u64().expect("grid_thw[0]") as usize,
                arr[1].as_u64().expect("grid_thw[1]") as usize,
                arr[2].as_u64().expect("grid_thw[2]") as usize,
            ]
        };
        let input_sha256 = case["input_sha256"]
            .as_str()
            .expect("input_sha256")
            .to_string();
        let resized_sha256 = case["resized_sha256"]
            .as_str()
            .expect("resized_sha256")
            .to_string();
        let num_patches = case["num_patches"].as_u64().expect("num_patches") as usize;
        let first_patch_first8: Vec<f64> = case["first_patch_first8"]
            .as_array()
            .expect("first_patch_first8 array")
            .iter()
            .map(|v| v.as_f64().expect("f64 element"))
            .collect();
        let last_patch_first8: Vec<f64> = case["last_patch_first8"]
            .as_array()
            .expect("last_patch_first8 array")
            .iter()
            .map(|v| v.as_f64().expect("f64 element"))
            .collect();
        let mean_abs = case["mean_abs"].as_f64().expect("mean_abs f64");
        let resized_first8_u8: Vec<u8> = case["resized_first8_u8"]
            .as_array()
            .expect("resized_first8_u8 array")
            .iter()
            .map(|v| v.as_u64().expect("u8 element") as u8)
            .collect();
        let resized_last8_u8: Vec<u8> = case["resized_last8_u8"]
            .as_array()
            .expect("resized_last8_u8 array")
            .iter()
            .map(|v| v.as_u64().expect("u8 element") as u8)
            .collect();

        // 1. The regenerated input must match the fixture; anything after is
        //    meaningless if the formula drifted.
        let rgb = golden_input_rgb(height, width);
        let in_sha = hex_sha256(&rgb);
        assert_eq!(
            in_sha, input_sha256,
            "[{id}] input image formula mismatch: got {in_sha}, fixture {input_sha256}"
        );

        // 2. smart_resize must reproduce the PIL resize target.
        let (dst_h, dst_w) =
            smart_resize(height, width, cfg.factor(), cfg.min_pixels, cfg.max_pixels)
                .unwrap_or_else(|e| panic!("[{id}] smart_resize failed: {e}"));
        assert_eq!(
            (dst_h, dst_w),
            (resized_hw[0], resized_hw[1]),
            "[{id}] smart_resize: got ({dst_h}, {dst_w}), fixture ({}, {})",
            resized_hw[0],
            resized_hw[1]
        );

        // 3. The bicubic resize must be bit-exact against PIL 12.3.0.
        let resized = resize_bicubic_rgb8(&rgb, height, width, dst_h, dst_w)
            .unwrap_or_else(|e| panic!("[{id}] resize_bicubic_rgb8 failed: {e}"));
        let resized_sha = hex_sha256(&resized);
        let got_first8: Vec<u8> = resized[..8.min(resized.len())].to_vec();
        assert_eq!(
            got_first8, resized_first8_u8,
            "[{id}] resized first 8 bytes mismatch"
        );
        let got_last8 = resized[resized.len() - 8..].to_vec();
        assert_eq!(
            got_last8, resized_last8_u8,
            "[{id}] resized last 8 bytes mismatch"
        );
        assert_eq!(
            resized_sha, resized_sha256,
            "[{id}] resized_sha256 mismatch: got {resized_sha}, fixture {resized_sha256}; \
             first 8 bytes got {got_first8:?}, fixture {resized_first8_u8:?}"
        );

        // 4. Full preprocess: grid, patch count, and f32 values.
        let out = preprocess_rgb8(&cfg, &rgb, height, width)
            .unwrap_or_else(|e| panic!("[{id}] preprocess_rgb8 failed: {e}"));
        assert_eq!(
            out.resized_hw,
            (resized_hw[0], resized_hw[1]),
            "[{id}] resized_hw mismatch"
        );
        assert_eq!(
            out.grid_thw,
            grid_thw.into(),
            "[{id}] grid_thw: got {:?}, fixture {grid_thw:?}",
            out.grid_thw
        );
        let n_patches = out.pixel_values.len() / (3 * 14 * 14);
        assert_eq!(
            n_patches, num_patches,
            "[{id}] num_patches: got {n_patches}, fixture {num_patches}"
        );

        let patch_len = 3 * 14 * 14;
        for (i, (got, want)) in out.pixel_values[..8]
            .iter()
            .zip(&first_patch_first8)
            .enumerate()
        {
            expect_exact(&format!("[{id}] first_patch_first8[{i}]"), *got, *want);
        }
        let last_base = patch_len * (num_patches - 1);
        for (i, (got, want)) in out.pixel_values[last_base..last_base + 8]
            .iter()
            .zip(&last_patch_first8)
            .enumerate()
        {
            expect_exact(&format!("[{id}] last_patch_first8[{i}]"), *got, *want);
        }
        let got_mean_abs = scalar_mean_abs(&out.pixel_values);
        expect_one_ulp(&format!("[{id}] mean_abs"), got_mean_abs, mean_abs);

        // 5. Patch order: two non-corner patches, re-sliced independently
        //    from the verified resized bytes. The first and last patches
        //    coincide under raster and column-major order, so the corner
        //    checks alone cannot catch a transposed patch grid.
        let ref01 = reference_patch(&resized, dst_w, 0, 1, &cfg);
        for (i, (got, want)) in out.pixel_values[patch_len..2 * patch_len]
            .iter()
            .zip(&ref01)
            .enumerate()
        {
            expect_exact(&format!("[{id}] patch(0,1)[{i}]"), *got, *want as f64);
        }
        let ref10 = reference_patch(&resized, dst_w, 1, 0, &cfg);
        for (i, (got, want)) in out.pixel_values
            [patch_len * grid_thw[2]..patch_len * grid_thw[2] + patch_len]
            .iter()
            .zip(&ref10)
            .enumerate()
        {
            expect_exact(&format!("[{id}] patch(1,0)[{i}]"), *got, *want as f64);
        }
    }
}

#[test]
fn paddleocr_vl_config_from_json() {
    let cfg = PaddleOcrImageProcessorConfig::from_preprocessor_json_str(
        r#"{
            "min_pixels": 112896,
            "max_pixels": 1003520,
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 1,
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5]
        }"#,
    )
    .expect("config parses");
    let d = PaddleOcrImageProcessorConfig::paddleocr_vl_defaults();
    assert_eq!(cfg.min_pixels, d.min_pixels);
    assert_eq!(cfg.max_pixels, d.max_pixels);
    assert_eq!(cfg.patch_size, d.patch_size);
    assert_eq!(cfg.merge_size, d.merge_size);
    assert_eq!(cfg.factor(), 28);
    assert_eq!(cfg.image_mean, [0.5, 0.5, 0.5]);
    assert_eq!(cfg.image_std, [0.5, 0.5, 0.5]);
    assert!((cfg.rescale_factor - 1.0 / 255.0).abs() < 1e-12);

    let missing = PaddleOcrImageProcessorConfig::from_preprocessor_json_str(r#"{"min_pixels": 1}"#)
        .expect_err("missing keys must fail");
    assert!(
        missing.to_string().contains("max_pixels"),
        "error should name the missing key: {missing}"
    );

    let bad = PaddleOcrImageProcessorConfig::from_preprocessor_json_str(
        r#"{"min_pixels": 112896, "max_pixels": 1003520, "patch_size": 14, "merge_size": 2, "image_mean": [0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "rescale_factor": 1/255}"#,
    )
    .expect_err("malformed json must fail");
    assert!(
        bad.to_string().contains("preprocessor_config"),
        "got: {bad}"
    );
}

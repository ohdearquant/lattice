use super::*;

#[test]
fn smart_resize_1246x560_rounds_half_to_even() {
    // 1246/28 = 44.5: round-half-to-even gives 44 -> 1232. A half-up
    // rounding would give 45 -> 1260, so this pins the rounding mode.
    let (h, w) = smart_resize(1246, 560, 28, 112_896, 1_003_520).expect("smart_resize");
    assert_eq!((h, w), (1232, 560), "44.5 must round to 44 (half-to-even)");
}

#[test]
fn smart_resize_tiny_row_scales_up() {
    // height < factor branch: width is re-derived from the aspect ratio.
    let (h, w) = smart_resize(20, 100, 28, 112_896, 1_003_520).expect("smart_resize");
    assert_eq!((h, w), (168, 756));
}

#[test]
fn smart_resize_large_image_scales_down() {
    let (h, w) = smart_resize(2000, 1500, 28, 112_896, 1_003_520).expect("smart_resize");
    assert_eq!((h, w), (1148, 840));
}

#[test]
fn smart_resize_rejects_zero_rounded_dimension() {
    let err = smart_resize(4096, 100, 28, 1, 784).expect_err("zero target width");
    assert!(matches!(err, InferenceError::Inference(_)));
    let message = err.to_string();
    assert!(message.contains("width"), "got: {message}");
    assert!(message.contains("4096x100"), "got: {message}");
}

#[test]
fn smart_resize_keeps_nonzero_rounded_dimensions() {
    let resized = smart_resize(4096, 4096, 28, 1, 784).expect("nonzero target");
    assert_eq!(resized, (28, 28));
}

#[test]
fn smart_resize_rejects_extreme_aspect_ratio() {
    let err = smart_resize(1000, 1, 28, 112_896, 1_003_520).expect_err("ratio 1000 > 200");
    assert!(matches!(err, InferenceError::Inference(_)));
}

#[test]
fn smart_resize_rejects_zero_dims() {
    assert!(smart_resize(0, 10, 28, 112_896, 1_003_520).is_err());
    assert!(smart_resize(10, 0, 28, 112_896, 1_003_520).is_err());
    assert!(smart_resize(10, 10, 0, 112_896, 1_003_520).is_err());
    assert!(smart_resize(10, 10, 28, 112_896, 0).is_err());
}

#[test]
fn defaults_match_shipped_config() {
    let cfg = PaddleOcrImageProcessorConfig::paddleocr_vl_defaults();
    cfg.validate().expect("shipped defaults validate");
    assert_eq!(cfg.min_pixels, 112_896);
    assert_eq!(cfg.max_pixels, 1_003_520);
    assert_eq!(cfg.patch_size, 14);
    assert_eq!(cfg.merge_size, 2);
    assert_eq!(cfg.factor(), 28);
    assert_eq!(cfg.image_mean, [0.5, 0.5, 0.5]);
    assert_eq!(cfg.image_std, [0.5, 0.5, 0.5]);
    assert!((cfg.rescale_factor - 1.0 / 255.0).abs() < 1e-12);
}

#[test]
fn parse_preprocessor_config_round_trips_defaults() {
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
    .expect("parses");
    let d = PaddleOcrImageProcessorConfig::paddleocr_vl_defaults();
    assert_eq!(cfg.min_pixels, d.min_pixels);
    assert_eq!(cfg.max_pixels, d.max_pixels);
    assert_eq!(cfg.factor(), d.factor());
    assert!(
        (cfg.rescale_factor - d.rescale_factor).abs() < 1e-12,
        "rescale_factor should be 1/255"
    );
}

#[test]
fn parse_preprocessor_config_missing_field_is_an_error() {
    for json in [
        r#"{"min_pixels": 112896}"#,
        r#"{"min_pixels": 112896, "max_pixels": 1, "patch_size": 14, "merge_size": 2, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}"#,
        r#"{"min_pixels": "big", "max_pixels": 1, "patch_size": 14, "merge_size": 2, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "rescale_factor": 0.004}"#,
        r#"{"min_pixels": 1, "max_pixels": 1, "patch_size": 14, "merge_size": 2, "image_mean": [0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "rescale_factor": 0.004}"#,
    ] {
        let err =
            PaddleOcrImageProcessorConfig::from_preprocessor_json_str(json).expect_err("must fail");
        assert!(
            err.to_string().contains("preprocessor_config"),
            "got: {err}"
        );
    }
}

#[test]
fn config_validation_accepts_and_rejects_each_boundary() {
    let mut cfg = PaddleOcrImageProcessorConfig::paddleocr_vl_defaults();

    cfg.min_pixels = 0;
    assert!(cfg.validate().is_err());
    cfg.min_pixels = 1;
    assert!(cfg.validate().is_ok());

    cfg.min_pixels = 1;
    cfg.max_pixels = 1;
    assert!(cfg.validate().is_ok());
    cfg.min_pixels = 2;
    assert!(cfg.validate().is_err());
    cfg.min_pixels = 1;
    cfg.max_pixels = MAX_PIXEL_BUDGET;
    assert!(cfg.validate().is_ok());
    cfg.max_pixels = MAX_PIXEL_BUDGET + 1;
    assert!(cfg.validate().is_err());
    cfg.max_pixels = 1_003_520;

    cfg.patch_size = 0;
    assert!(cfg.validate().is_err());
    cfg.patch_size = 1;
    assert!(cfg.validate().is_ok());
    cfg.merge_size = 0;
    assert!(cfg.validate().is_err());
    cfg.merge_size = 1;
    assert!(cfg.validate().is_ok());

    cfg.patch_size = MAX_RESIZE_FACTOR;
    cfg.merge_size = 1;
    assert!(cfg.validate().is_ok());
    cfg.patch_size = MAX_RESIZE_FACTOR;
    cfg.merge_size = 2;
    assert!(cfg.validate().is_err());
    cfg.patch_size = usize::MAX;
    cfg.merge_size = 2;
    assert!(cfg.validate().is_err());

    cfg.patch_size = 14;
    cfg.merge_size = 2;
    cfg.rescale_factor = f64::MIN_POSITIVE;
    assert!(cfg.validate().is_ok());
    cfg.rescale_factor = 0.0;
    assert!(cfg.validate().is_err());
    cfg.rescale_factor = f64::INFINITY;
    assert!(cfg.validate().is_err());
}

#[test]
fn preprocess_rejects_max_pixels_before_resize_allocation() {
    let mut cfg = PaddleOcrImageProcessorConfig::paddleocr_vl_defaults();
    cfg.min_pixels = usize::MAX;
    cfg.max_pixels = usize::MAX;
    assert!(
        cfg.validate().is_err(),
        "invalid budget must fail validation"
    );
    assert!(preprocess_rgb8(&cfg, &[0, 0, 0], 1, 1).is_err());
}

#[test]
fn preprocess_rescales_through_f64_before_f32() {
    let cfg = PaddleOcrImageProcessorConfig {
        min_pixels: 1,
        max_pixels: 1,
        patch_size: 1,
        merge_size: 1,
        image_mean: [0.0; 3],
        image_std: [1.0; 3],
        rescale_factor: 1.0 / 255.0,
    };
    let out = preprocess_rgb8(&cfg, &[254, 0, 0], 1, 1).expect("1x1 preprocess");
    assert_eq!(out.pixel_values[0], (254.0f64 / 255.0) as f32);
}

#[test]
fn preprocess_admits_full_resolution_phone_geometry_before_buffer_check() {
    let cfg = PaddleOcrImageProcessorConfig::paddleocr_vl_defaults();
    let err = preprocess_rgb8(&cfg, &[], 8064, 6048).expect_err("empty RGB buffer");
    assert!(matches!(err, InferenceError::ShapeMismatch { .. }));
}

#[test]
fn smart_resize_bounds_full_resolution_phone_output() {
    let cfg = PaddleOcrImageProcessorConfig::paddleocr_vl_defaults();
    let (height, width) =
        smart_resize(6048, 8064, 28, cfg.min_pixels, cfg.max_pixels).expect("smart resize");
    assert!(height <= MAX_RESIZED_IMAGE_SIDE);
    assert!(width <= MAX_RESIZED_IMAGE_SIDE);
    assert!(height * width <= cfg.max_pixels);
}

#[test]
fn oversized_preprocessor_json_is_rejected_before_read() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("preprocessor_config.json");
    let file = std::fs::File::create(&path).expect("create config");
    file.set_len(MAX_CONFIG_JSON_BYTES + 1)
        .expect("extend config");
    let err =
        PaddleOcrImageProcessorConfig::from_preprocessor_json(&path).expect_err("oversized config");
    assert!(err.to_string().contains("MAX_CONFIG_JSON_BYTES"));
}

#[test]
fn preprocessor_json_rejects_a_directory() {
    let directory = tempfile::tempdir().expect("tempdir");
    let err = PaddleOcrImageProcessorConfig::from_preprocessor_json(directory.path())
        .expect_err("a directory must be rejected as a preprocessor config file");
    assert!(
        err.to_string().contains("is not a regular file"),
        "wrong error: {err}"
    );
}

#[test]
fn image_dimensions_and_buffer_lengths_are_bounded() {
    let cfg = PaddleOcrImageProcessorConfig::paddleocr_vl_defaults();
    for &(height, width) in &[(MAX_INPUT_IMAGE_SIDE + 1, 1), (16_384, 4_097)] {
        assert!(preprocess_rgb8(&cfg, &[], height, width).is_err());
        assert!(resize_bicubic_rgb8(&[], height, width, 1, 1).is_err());
    }
    assert!(resize_bicubic_rgb8(&[0, 0, 0], 1, 1, 1, MAX_RESIZED_IMAGE_SIDE + 1).is_err());
    assert!(resize_bicubic_rgb8(&[0, 0, 0], 1, 1, 1, 1).is_ok());
    assert!(preprocess_rgb8(&cfg, &[0, 0, 0], 1, 1).is_ok());
}

#[test]
fn resize_bicubic_rejects_destination_over_pixel_budget() {
    let err = resize_bicubic_rgb8(
        &[0, 0, 0],
        1,
        1,
        MAX_RESIZED_IMAGE_SIDE,
        MAX_RESIZED_IMAGE_SIDE,
    )
    .expect_err("destination over pixel budget");
    let message = err.to_string();
    assert!(message.contains("max_pixels"), "got: {message}");
}

#[test]
fn parse_preprocessor_json_missing_file_is_io_error() {
    let err = PaddleOcrImageProcessorConfig::from_preprocessor_json(Path::new(
        "/nonexistent/preprocessor_config.json",
    ))
    .expect_err("missing file must fail");
    assert!(matches!(err, InferenceError::Io(_)));
}

#[test]
fn bicubic_kernel_values() {
    assert_eq!(bicubic_kernel(0.0), 1.0);
    assert_eq!(bicubic_kernel(1.0), 0.0);
    assert_eq!(bicubic_kernel(2.0), 0.0);
    assert_eq!(bicubic_kernel(5.0), 0.0);
    // |x| = 0.5: ((1.5)(0.5) - 2.5)(0.25) + 1 = 0.5625
    assert!((bicubic_kernel(0.5) - 0.5625).abs() < 1e-12);
    // |x| = 1.5: ((-3.5)(1.5) + 8)(1.5) - 4 = 0.125; * a(-0.5) = -0.0625
    assert!((bicubic_kernel(1.5) + 0.0625).abs() < 1e-12);
    assert_eq!(bicubic_kernel(-0.5), bicubic_kernel(0.5));
}

#[test]
fn clip8_bias_and_clamp() {
    // Bias alone: 2^21 >> 22 = 0.
    assert_eq!(clip8(ACC_BIAS), 0);
    // 2^21 + 1*2^22 = 1.5 -> truncates to 1.
    assert_eq!(clip8(ACC_BIAS + (1 << PRECISION_BITS)), 1);
    // 255 exactly: 2^21 + 255*2^22 = 256.5 -> 256 -> clamped to 255.
    assert_eq!(clip8(ACC_BIAS + 255 * (1 << PRECISION_BITS)), 255);
    // Negative accumulates clamp to 0 (arithmetic shift keeps sign).
    assert_eq!(clip8(ACC_BIAS - (1 << PRECISION_BITS) * 2), 0);
}

#[test]
fn resize_identity_on_small_image_is_stable() {
    // Same dimensions: a bicubic resize with scale 1.0 reproduces the
    // input up to the integer kernel (center taps dominate). We check
    // determinism and shape only — the exact 1:1 pass is not required to
    // be the identity.
    let src: Vec<u8> = (0..64 * 64 * 3).map(|i| (i % 256) as u8).collect();
    let a = resize_bicubic_rgb8(&src, 64, 64, 64, 64).expect("resize");
    let b = resize_bicubic_rgb8(&src, 64, 64, 64, 64).expect("resize");
    assert_eq!(a.len(), 64 * 64 * 3);
    assert_eq!(a, b);
}

#[test]
fn preprocess_rejects_bad_lengths() {
    let cfg = PaddleOcrImageProcessorConfig::paddleocr_vl_defaults();
    let err = preprocess_rgb8(&cfg, &[0u8; 3], 2, 2).expect_err("short input");
    assert!(matches!(err, InferenceError::ShapeMismatch { .. }));
}

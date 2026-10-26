use super::*;

/// Create a minimal valid 1×1 PNG (8-byte signature + minimal IHDR/IDAT/IEND).
/// We synthesize a tiny solid-color image for unit tests.
fn make_test_png(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
    // Create solid-color RgbImage and encode to PNG bytes.
    let mut img = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            img.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    img.write_to(&mut cursor, image::ImageFormat::Png)
        .expect("test PNG encode");
    buf
}

#[test]
fn preprocess_patch_count_448_16() {
    let cfg = PreprocessConfig::qwen3_vl();
    let png = make_test_png(64, 64, 128, 64, 32);
    let tensor = preprocess(&png, &cfg).expect("preprocess");
    assert_eq!(tensor.n_patches, 784); // (448/16)^2
    assert_eq!(tensor.patch_hw, 16);
    assert_eq!(tensor.patches.len(), 784 * 16 * 16 * 3);
}

#[test]
fn preprocess_patch_len_helper() {
    let cfg = PreprocessConfig::qwen3_vl();
    let png = make_test_png(32, 32, 0, 0, 0);
    let tensor = preprocess(&png, &cfg).expect("preprocess");
    assert_eq!(tensor.patch_len(), 16 * 16 * 3); // 768
}

#[test]
fn preprocess_normalization_is_applied() {
    let cfg = PreprocessConfig::qwen3_vl();
    // Solid white image (255, 255, 255) → after /255.0 → 1.0 for all channels.
    let png = make_test_png(32, 32, 255, 255, 255);
    let tensor = preprocess(&png, &cfg).expect("preprocess");
    // Check that at least the first pixel of the first patch is normalized.
    let v_r = tensor.patches[0];
    let expected_r = (1.0_f32 - QWEN3_VL_IMAGE_MEAN[0]) / QWEN3_VL_IMAGE_STD[0];
    assert!(
        (v_r - expected_r).abs() < 1e-4,
        "channel R: got {v_r}, expected {expected_r}"
    );
}

#[test]
fn preprocess_black_image_normalized() {
    let cfg = PreprocessConfig::qwen3_vl();
    // Solid black (0, 0, 0) → raw = 0.0 → normalized = (0 - mean) / std
    let png = make_test_png(32, 32, 0, 0, 0);
    let tensor = preprocess(&png, &cfg).expect("preprocess");
    let v_r = tensor.patches[0];
    let expected_r = (0.0_f32 - QWEN3_VL_IMAGE_MEAN[0]) / QWEN3_VL_IMAGE_STD[0];
    assert!(
        (v_r - expected_r).abs() < 1e-4,
        "black image channel R: got {v_r}, expected {expected_r}"
    );
}

#[test]
fn preprocess_invalid_config_indivisible() {
    let mut cfg = PreprocessConfig::qwen3_vl();
    cfg.image_size = 449; // not divisible by 16
    let png = make_test_png(32, 32, 0, 0, 0);
    assert!(preprocess(&png, &cfg).is_err());
}

#[test]
fn preprocess_invalid_bytes() {
    let cfg = PreprocessConfig::qwen3_vl();
    let result = preprocess(b"not an image", &cfg);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), VisionError::ImageDecode(_)));
}

#[test]
fn preprocess_dimensions_custom_config() {
    // 64×64 image with 8×8 patches → 64 patches
    let cfg = PreprocessConfig {
        image_size: 64,
        patch_size: 8,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
    };
    let png = make_test_png(32, 32, 128, 128, 128);
    let tensor = preprocess(&png, &cfg).expect("preprocess 64/8");
    assert_eq!(tensor.n_patches, 64); // (64/8)^2
    assert_eq!(tensor.patch_hw, 8);
    assert_eq!(tensor.patches.len(), 64 * 8 * 8 * 3);
}

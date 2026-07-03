//! CPU-side image preprocessing for the Qwen3-VL vision encoder.
//!
//! Pipeline: decode bytes → resize to image_size×image_size (bilinear) →
//! per-channel normalize (ImageNet-style) → extract patch grid.
//!
//! Output: `ImageTensor` with `n_patches` rows, each row containing
//! `patch_size * patch_size * 3` f32 values in HWC row-major order.
//!
//! ## Normalization constants (Qwen3-VL)
//!
//! Mean: [0.4814546, 0.4578275, 0.4082107]
//! Std:  [0.2686295, 0.2612877, 0.2757770]
//!
//! These are the standard ImageNet statistics used by the Qwen VL family.
//! See `Qwen/Qwen3-VL-7B-Instruct` processor_config.json.

use image::{DynamicImage, ImageReader, RgbImage};
use std::io::Cursor;

use super::VisionError;

/// Per-channel normalization constants for Qwen3-VL.
///
/// Source: `Qwen/Qwen3-VL-7B-Instruct` processor_config.json,
/// `image_mean` / `image_std` fields.
pub const QWEN3_VL_IMAGE_MEAN: [f32; 3] = [0.481_454_6, 0.457_827_5, 0.408_210_7];
pub const QWEN3_VL_IMAGE_STD: [f32; 3] = [0.268_629_5, 0.261_287_7, 0.275_777];

/// Preprocessing configuration.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Target image side length (pixels). Both height and width become this value.
    pub image_size: u32,
    /// Patch side length (pixels).
    pub patch_size: u32,
    /// Per-channel mean for normalization ([R, G, B]).
    pub mean: [f32; 3],
    /// Per-channel standard deviation ([R, G, B]).
    pub std: [f32; 3],
}

impl PreprocessConfig {
    /// Qwen3-VL defaults: 448×448 image, 16×16 patches, ImageNet stats.
    pub fn qwen3_vl() -> Self {
        Self {
            image_size: 448,
            patch_size: 16,
            mean: QWEN3_VL_IMAGE_MEAN,
            std: QWEN3_VL_IMAGE_STD,
        }
    }
}

/// Preprocessed image ready for the ViT encoder.
#[derive(Debug, Clone)]
pub struct ImageTensor {
    /// Flattened patch data: `[n_patches, patch_size * patch_size * 3]` row-major.
    ///
    /// Each row corresponds to one spatial patch in raster order (row-major over
    /// the patch grid). Within a row, values are ordered HWC: for each pixel in
    /// the patch (row-major), the three channel values appear consecutively.
    /// All values are normalized (mean-subtracted, std-divided).
    pub patches: Vec<f32>,
    /// Number of patches: `(image_size / patch_size)^2`.
    pub n_patches: usize,
    /// Number of pixels within a single patch side (patch_size).
    pub patch_hw: usize,
}

impl ImageTensor {
    /// Flattened length for one patch: `patch_hw * patch_hw * 3`.
    pub fn patch_len(&self) -> usize {
        self.patch_hw * self.patch_hw * 3
    }
}

/// Decode, resize, normalize, and extract patch grid from raw image bytes.
///
/// Accepts JPEG, PNG, and any other format supported by the `image` crate.
/// The image is always converted to RGB24 before processing.
///
/// # Errors
///
/// Returns `VisionError::ImageDecode` if the bytes cannot be decoded.
/// Returns `VisionError::InvalidConfig` if `cfg.image_size` is not divisible
/// by `cfg.patch_size`, or if any std component is zero.
pub fn preprocess(image_bytes: &[u8], cfg: &PreprocessConfig) -> Result<ImageTensor, VisionError> {
    // Validate config
    if cfg.patch_size == 0 {
        return Err(VisionError::InvalidConfig("patch_size must be > 0".into()));
    }
    if !cfg.image_size.is_multiple_of(cfg.patch_size) {
        return Err(VisionError::InvalidConfig(format!(
            "image_size {} must be divisible by patch_size {}",
            cfg.image_size, cfg.patch_size
        )));
    }
    for (i, &s) in cfg.std.iter().enumerate() {
        if s <= 0.0 {
            return Err(VisionError::InvalidConfig(format!(
                "std[{i}]={s} must be > 0"
            )));
        }
    }

    // Decode image
    let reader = ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .map_err(|e| VisionError::ImageDecode(format!("format detection failed: {e}")))?;
    let img: DynamicImage = reader
        .decode()
        .map_err(|e| VisionError::ImageDecode(format!("decode failed: {e}")))?;

    // Resize to image_size × image_size (bilinear)
    let size = cfg.image_size;
    let resized: RgbImage = img
        .resize_exact(size, size, image::imageops::FilterType::Triangle)
        .into_rgb8();

    // Extract patches
    let patch_size = cfg.patch_size as usize;
    let image_size = cfg.image_size as usize;
    let patches_per_side = image_size / patch_size;
    let n_patches = patches_per_side * patches_per_side;
    let patch_len = patch_size * patch_size * 3;

    let mut patches = vec![0.0f32; n_patches * patch_len];

    for py in 0..patches_per_side {
        for px in 0..patches_per_side {
            let patch_idx = py * patches_per_side + px;
            let patch_slice = &mut patches[patch_idx * patch_len..(patch_idx + 1) * patch_len];

            let pixel_start_y = py * patch_size;
            let pixel_start_x = px * patch_size;

            let mut out_idx = 0usize;
            for dy in 0..patch_size {
                for dx in 0..patch_size {
                    let px_coord = (pixel_start_x + dx) as u32;
                    let py_coord = (pixel_start_y + dy) as u32;
                    let pixel = resized.get_pixel(px_coord, py_coord);
                    for c in 0..3usize {
                        let raw = pixel[c] as f32 / 255.0;
                        patch_slice[out_idx] = (raw - cfg.mean[c]) / cfg.std[c];
                        out_idx += 1;
                    }
                }
            }
        }
    }

    Ok(ImageTensor {
        patches,
        n_patches,
        patch_hw: patch_size,
    })
}

#[cfg(test)]
mod tests {
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
}

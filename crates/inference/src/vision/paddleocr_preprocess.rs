//! Image preprocessing for PaddleOCR-VL-1.6 vision input.
//!
//! Reproduces the `PaddleOCR-VL` image processor (`preprocessor_config.json`)
//! for a single RGB8 image:
//!
//! 1. `smart_resize` — snap the image to a multiple of `patch_size *
//!    merge_size` (factor 28 for the shipped config) while keeping the pixel
//!    count inside `[min_pixels, max_pixels]`.
//! 2. Bicubic resize on uint8 RGB, bit-exact against Pillow 12.3.0's
//!    `Image.resize` with `Image.BICUBIC` (a port of Pillow's
//!    `src/libImaging/Resample.c`: separable horizontal-then-vertical passes,
//!    per-output fixed-point coefficient tables with 22-bit precision).
//! 3. Rescale by `rescale_factor` (1/255) and per-channel normalize
//!    `(x - mean[c]) / std[c]` in f32.
//! 4. Patchify into `[n_patches, 3, patch, patch]` in raster patch order
//!    (row-major over the `grid_h x grid_w` patch grid), with
//!    `grid_thw = (1, grid_h, grid_w)`.
//!
//! The bit-exactness of steps 1-2 is held by the pure-CPU goldens test in
//! `tests/paddleocr_vl_preprocess_goldens_test.rs`, which checks the resized
//! uint8 image byte-for-byte (sha256) against a fixture produced with
//! PIL 12.3.0.

use crate::error::InferenceError;
use std::path::Path;

/// Fixed-point coefficient precision (bits), matching Pillow's
/// `PRECISION_BITS` in `Resample.c`.
const PRECISION_BITS: i32 = 22;

/// Bicubic kernel constant `a` (CubicConvolution "Keys" curve).
const BICUBIC_A: f64 = -0.5;

/// Image processor configuration from `preprocessor_config.json`.
#[derive(Debug, Clone)]
pub struct PaddleOcrImageProcessorConfig {
    /// Minimum pixel count after smart-resize; smaller images are scaled up.
    pub min_pixels: usize,
    /// Maximum pixel count after smart-resize; larger images are scaled down.
    pub max_pixels: usize,
    /// Spatial patch side length in pixels (14 for the shipped config).
    pub patch_size: usize,
    /// Spatial merge factor (2 for the shipped config).
    pub merge_size: usize,
    /// Per-channel mean used for normalization ([R, G, B]).
    pub image_mean: [f32; 3],
    /// Per-channel standard deviation used for normalization ([R, G, B]).
    pub image_std: [f32; 3],
    /// Multiplicative rescale applied to uint8 values before normalization
    /// (1/255 for the shipped config).
    pub rescale_factor: f32,
}

impl PaddleOcrImageProcessorConfig {
    /// Parse a `preprocessor_config.json` document.
    ///
    /// Required fields: `min_pixels`, `max_pixels`, `patch_size`,
    /// `merge_size`, `image_mean`, `image_std`, `rescale_factor`. Other
    /// fields (e.g. `temporal_patch_size`, `resample`) are ignored.
    pub fn from_preprocessor_json_str(json: &str) -> Result<Self, InferenceError> {
        let value: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            InferenceError::Inference(format!("invalid preprocessor_config json: {e}"))
        })?;

        let usize_field = |key: &str| -> Result<usize, InferenceError> {
            let v = value.get(key).ok_or_else(|| {
                InferenceError::Inference(format!(
                    "preprocessor_config is missing required field `{key}`"
                ))
            })?;
            let n = v.as_u64().ok_or_else(|| {
                InferenceError::Inference(format!(
                    "preprocessor_config field `{key}` must be a non-negative integer"
                ))
            })?;
            usize::try_from(n).map_err(|_| {
                InferenceError::Inference(format!(
                    "preprocessor_config field `{key}` does not fit in usize"
                ))
            })
        };

        let f32_array = |key: &str| -> Result<[f32; 3], InferenceError> {
            let arr = value.get(key).ok_or_else(|| {
                InferenceError::Inference(format!(
                    "preprocessor_config is missing required field `{key}`"
                ))
            })?;
            let list = arr.as_array().ok_or_else(|| {
                InferenceError::Inference(format!(
                    "preprocessor_config field `{key}` must be an array of 3 numbers"
                ))
            })?;
            if list.len() != 3 {
                return Err(InferenceError::Inference(format!(
                    "preprocessor_config field `{key}` must have exactly 3 elements, found {}",
                    list.len()
                )));
            }
            let mut out = [0.0f32; 3];
            for (i, item) in list.iter().enumerate() {
                let f = item.as_f64().ok_or_else(|| {
                    InferenceError::Inference(format!(
                        "preprocessor_config field `{key}[{i}]` must be a number"
                    ))
                })?;
                out[i] = f as f32;
            }
            Ok(out)
        };

        Ok(Self {
            min_pixels: usize_field("min_pixels")?,
            max_pixels: usize_field("max_pixels")?,
            patch_size: usize_field("patch_size")?,
            merge_size: usize_field("merge_size")?,
            image_mean: f32_array("image_mean")?,
            image_std: f32_array("image_std")?,
            rescale_factor: value
                .get("rescale_factor")
                .and_then(serde_json::Value::as_f64)
                .map(|v| v as f32)
                .ok_or_else(|| {
                    InferenceError::Inference(
                        "preprocessor_config is missing required field `rescale_factor`".into(),
                    )
                })?,
        })
    }

    /// Parse a `preprocessor_config.json` file.
    pub fn from_preprocessor_json(path: &Path) -> Result<Self, InferenceError> {
        let raw = std::fs::read_to_string(path).map_err(InferenceError::Io)?;
        Self::from_preprocessor_json_str(&raw)
    }

    /// Shipped PaddleOCR-VL-1.6 defaults: factor 28, 14 px patches with a
    /// 2x2 spatial merge, 1/255 rescale and [0.5, 0.5, 0.5] mean/std.
    pub fn paddleocr_vl_defaults() -> Self {
        Self {
            min_pixels: 112_896,
            max_pixels: 1_003_520,
            patch_size: 14,
            merge_size: 2,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            rescale_factor: 1.0 / 255.0,
        }
    }

    /// `patch_size * merge_size`: the smart-resize divisibility factor (28
    /// for the shipped config).
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }
}

/// Snap `(height, width)` to a multiple of `factor` so the pixel count lands
/// in `[min_pixels, max_pixels]`, following the reference `smart_resize`:
///
/// ```python
/// if height < factor: width = round(width*factor/height); height = factor
/// if width  < factor: height = round(height*factor/width); width = factor
/// h_bar = round(height/factor)*factor; w_bar = round(width/factor)*factor
/// if h_bar*w_bar > max_pixels: beta = sqrt(height*width/max_pixels)
///     h_bar = floor(height/beta/factor)*factor; w_bar = floor(width/beta/factor)*factor
/// elif h_bar*w_bar < min_pixels: beta = sqrt(min_pixels/(height*width))
///     h_bar = ceil(height*beta/factor)*factor; w_bar = ceil(width*beta/factor)*factor
/// ```
///
/// `round` is Python's round-half-to-even. All arithmetic is f64. Returns
/// the resized `(h_bar, w_bar)`.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), InferenceError> {
    if height == 0 || width == 0 || factor == 0 || max_pixels == 0 {
        return Err(InferenceError::Inference(format!(
            "smart_resize requires positive height, width, factor and max_pixels (got {height}x{width}, factor={factor}, max_pixels={max_pixels})"
        )));
    }

    let (mut h, mut w) = (height as f64, width as f64);
    let f = factor as f64;

    if h < f {
        w = (w * f / h).round_ties_even();
        h = f;
    }
    if w < f {
        h = (h * f / w).round_ties_even();
        w = f;
    }

    let (lo, hi) = if h < w { (h, w) } else { (w, h) };
    if hi / lo > 200.0 {
        return Err(InferenceError::Inference(format!(
            "smart_resize aspect ratio {lo:.0}x{hi:.0} too extreme (max/min = {:.4}, limit 200)",
            hi / lo
        )));
    }

    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;

    let pixels = h_bar * w_bar;
    let max_pixels = max_pixels as f64;
    let min_pixels = min_pixels as f64;
    if pixels > max_pixels {
        let beta = (h * w / max_pixels).sqrt();
        h_bar = (h / beta / f).floor() * f;
        w_bar = (w / beta / f).floor() * f;
    } else if pixels < min_pixels {
        let beta = (min_pixels / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }

    Ok((h_bar as usize, w_bar as usize))
}

/// Bicubic convolution kernel with constant `a = BICUBIC_A` (-0.5):
/// `|x| < 1: ((a+2)|x| - (a+3))|x|^2 + 1`, `1 <= |x| < 2: (((|x|-5)|x| + 8)|x| - 4)a`,
/// else 0.
fn bicubic_kernel(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((BICUBIC_A + 2.0) * x - (BICUBIC_A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * BICUBIC_A
    } else {
        0.0
    }
}

/// Per-output fixed-point filter taps for one separable pass, as produced by
/// Pillow's `precompute_coeffs` (8 bpc normalization): `xmin` is the first
/// input index in the input, `xmax` the tap count, and `k_int` the 22-bit
/// fixed-point taps (`PRECISION_BITS = 22`).
struct SpanCoeffs {
    xmin: usize,
    xmax: usize,
    k_int: Vec<i32>,
}

/// Compute per-output coefficient tables for one separable pass over
/// `in_size` input samples producing `out_size` output samples.
///
/// `scale = in_size / out_size`, `filterscale = max(scale, 1.0)`,
/// `support = 2.0 * filterscale` (bicubic support is 2.0); per output index
/// `xx`: `center = (xx + 0.5) * scale`, tap distance
/// `((x + xmin - center) + 0.5) / filterscale` evaluated through
/// [`bicubic_kernel`]; the taps for one output are normalized to sum to 1
/// (when the sum is nonzero) and quantized to 22-bit fixed point with
/// half-away-from-zero truncation, mirroring the C reference.
fn precompute_coeffs(in_size: usize, out_size: usize) -> Vec<SpanCoeffs> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = (support.ceil() as usize) * 2 + 1;
    let in_size_i64 = in_size as i64;

    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let ss = 1.0 / filterscale;
            // C `(int)` cast truncates toward zero; then clamp to the input span.
            let xmin = ((center - support + 0.5) as i64).max(0);
            let xmax = ((center + support + 0.5) as i64).min(in_size_i64) - xmin;
            let xmax = xmax.max(0) as usize;

            let mut weights = vec![0.0f64; xmax];
            for x in 0..xmax {
                weights[x] = bicubic_kernel((x as f64 + xmin as f64 - center + 0.5) * ss);
            }
            let ww: f64 = weights.iter().sum();
            if ww != 0.0 {
                for w in weights.iter_mut() {
                    *w /= ww;
                }
            }

            let mut k_int = vec![0i32; ksize];
            let scale22 = 1i32 << PRECISION_BITS;
            for (i, w) in weights.iter().enumerate() {
                k_int[i] = if *w < 0.0 {
                    (-0.5 + *w * scale22 as f64) as i32
                } else {
                    (0.5 + *w * scale22 as f64) as i32
                };
            }
            SpanCoeffs {
                xmin: xmin as usize,
                xmax,
                k_int,
            }
        })
        .collect()
}

/// Fixed-point accumulation scale: `1 << 21`, i.e. half of the 22-bit
/// fixed-point grid, so that rounding is a truncation after the bias.
const ACC_BIAS: i32 = 1 << 21;

/// `clip8` from the C reference: arithmetic right shift to undo the
/// fixed-point scale, then clamp to `[0, 255]`.
fn clip8(ss: i32) -> u8 {
    (ss >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Bicubic resize of an interleaved HWC uint8 RGB image to `dst_h x dst_w`,
/// bit-exact against Pillow 12.3.0 `Image.resize(..., Image.BICUBIC)` with
/// `reducing_gap=None` (no pre-reduction).
///
/// Two separable passes, each channel independent: a horizontal pass over
/// the source producing a `dst_w x src_h` uint8 intermediate, then a
/// vertical pass over that intermediate.
pub fn resize_bicubic_rgb8(
    src: &[u8],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Result<Vec<u8>, InferenceError> {
    let expected = src_h.saturating_mul(src_w).saturating_mul(3);
    if src.len() != expected {
        return Err(InferenceError::ShapeMismatch {
            name: "resize_bicubic_rgb8 input".into(),
            expected: vec![src_h, src_w, 3],
            actual: vec![src.len()],
        });
    }
    if src_h == 0 || src_w == 0 || dst_h == 0 || dst_w == 0 {
        return Err(InferenceError::Inference(
            "resize_bicubic_rgb8 requires nonzero dimensions".into(),
        ));
    }

    let x_coeffs = precompute_coeffs(src_w, dst_w);
    let y_coeffs = precompute_coeffs(src_h, dst_h);

    // Horizontal pass: src (src_h x src_w) -> mid (src_h x dst_w), HWC.
    let mut mid = vec![0u8; src_h * dst_w * 3];
    for c in 0..3 {
        for y in 0..src_h {
            for (xx, coeffs) in (0..dst_w).zip(x_coeffs.iter()) {
                let mut ss: i32 = ACC_BIAS;
                for (x, tap) in (0..coeffs.xmax).zip(coeffs.k_int.iter()) {
                    let px = src[(y * src_w + coeffs.xmin + x) * 3 + c] as i32;
                    ss += px * *tap;
                }
                mid[(y * dst_w + xx) * 3 + c] = clip8(ss);
            }
        }
    }

    // Vertical pass: mid (src_h x dst_w) -> out (dst_h x dst_w), HWC.
    let mut out = vec![0u8; dst_h * dst_w * 3];
    for c in 0..3 {
        for x in 0..dst_w {
            for (yy, coeffs) in (0..dst_h).zip(y_coeffs.iter()) {
                let mut ss: i32 = ACC_BIAS;
                for (y, tap) in (0..coeffs.xmax).zip(coeffs.k_int.iter()) {
                    let px = mid[((coeffs.xmin + y) * dst_w + x) * 3 + c] as i32;
                    ss += px * *tap;
                }
                out[(yy * dst_w + x) * 3 + c] = clip8(ss);
            }
        }
    }

    Ok(out)
}

/// Output of [`preprocess_rgb8`]: patchified, normalized f32 tensor.
///
/// `pixel_values` is `[n_patches, 3, patch, patch]` flattened, in raster
/// patch order (row-major over the `grid_h x grid_w` grid); `grid_thw` is
/// `(1, grid_h, grid_w)`; `resized_hw` is the smart-resized
/// `(height, width)` the image was bicubically resized to.
#[derive(Debug)]
pub struct PreprocessedImage {
    /// `[n_patches, 3, patch, patch]` flattened f32 pixel values.
    pub pixel_values: Vec<f32>,
    /// `(temporal, grid_h, grid_w)`; temporal is always 1 for images.
    pub grid_thw: (usize, usize, usize),
    /// `(height, width)` after smart-resize.
    pub resized_hw: (usize, usize),
}

/// Run the full PaddleOCR-VL image processor on one RGB8 image
/// (`rgb` is interleaved HWC, length `height * width * 3`):
/// smart-resize, bit-exact bicubic resize, rescale + normalize in f32,
/// patchify to `[n_patches, 3, patch, patch]` in raster patch order.
pub fn preprocess_rgb8(
    cfg: &PaddleOcrImageProcessorConfig,
    rgb: &[u8],
    height: usize,
    width: usize,
) -> Result<PreprocessedImage, InferenceError> {
    let patch = cfg.patch_size;
    if patch == 0 || cfg.merge_size == 0 {
        return Err(InferenceError::Inference(
            "patch_size and merge_size must be nonzero".into(),
        ));
    }
    let expected = height.saturating_mul(width).saturating_mul(3);
    if rgb.len() != expected {
        return Err(InferenceError::ShapeMismatch {
            name: "preprocess_rgb8 input".into(),
            expected: vec![height, width, 3],
            actual: vec![rgb.len()],
        });
    }

    let (dst_h, dst_w) = smart_resize(height, width, cfg.factor(), cfg.min_pixels, cfg.max_pixels)?;
    if dst_h % patch != 0 || dst_w % patch != 0 {
        return Err(InferenceError::Inference(format!(
            "resized image {dst_h} x {dst_w} is not divisible by patch_size {patch}"
        )));
    }
    let resized = resize_bicubic_rgb8(rgb, height, width, dst_h, dst_w)?;

    let grid_h = dst_h / patch;
    let grid_w = dst_w / patch;
    let n_patches = grid_h * grid_w;
    let patch_len = 3 * patch * patch;
    let mut pixel_values = vec![0.0f32; n_patches * patch_len];

    for py_out in 0..grid_h {
        let py = py_out;
        for px_out in 0..grid_w {
            let px = px_out;
            let patch_base = py_out * grid_w + px;
            let out_base = patch_base * patch_len;
            for c in 0..3 {
                let mean = cfg.image_mean[c];
                let std = cfg.image_std[c];
                let mut o = out_base + c * patch * patch;
                for yy in 0..patch {
                    let row = dst_w * (py * patch + yy);
                    for xx in 0..patch {
                        let v =
                            resized[(row + px * patch + xx) * 3 + c] as f32 * cfg.rescale_factor;
                        pixel_values[o] = (v - mean) / std;
                        o += 1;
                    }
                }
            }
        }
    }

    Ok(PreprocessedImage {
        pixel_values,
        grid_thw: (1, grid_h, grid_w),
        resized_hw: (dst_h, dst_w),
    })
}

#[cfg(test)]
mod tests {
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
            let err = PaddleOcrImageProcessorConfig::from_preprocessor_json_str(json)
                .expect_err("must fail");
            assert!(
                err.to_string().contains("preprocessor_config"),
                "got: {err}"
            );
        }
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
}

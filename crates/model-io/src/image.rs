//! Bounded image decoding and shared vision errors.

#[cfg(feature = "image")]
use image::{DynamicImage, ImageReader};
#[cfg(feature = "image")]
use std::io::Cursor;

/// Errors produced by the vision encoder pipeline.
///
/// Marked `#[non_exhaustive]` so that distinguishing a new failure mode is a
/// minor change rather than a major one. Adding `DimensionsExceeded` was a
/// breaking change only because this enum was exhaustive; that cost is paid
/// once here.
#[derive(Debug)]
#[non_exhaustive]
pub enum VisionError {
    /// Raw image bytes could not be decoded (unsupported format, corrupt data).
    ImageDecode(String),
    /// A weight or activation tensor had unexpected dimensions.
    ShapeMismatch {
        expected: usize,
        actual: usize,
        context: String,
    },
    /// A configuration value is invalid (zero dimension, indivisible sizes, etc.).
    InvalidConfig(String),
    /// The decoded image's declared pixel dimensions, pre-merge patch grid,
    /// or resulting preprocessed buffer exceed a serving-side budget --
    /// distinct from [`Self::ImageDecode`] (the image itself is well-formed,
    /// just larger than this server will process) and from
    /// [`Self::InvalidConfig`] (a checkpoint/config problem, not a
    /// per-request one). Callers map this to a client-safe, retriable
    /// rejection rather than the generic image-decode error.
    DimensionsExceeded(String),
    /// I/O error during weight loading.
    Io(std::io::Error),
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImageDecode(msg) => write!(f, "Vision image decode error: {msg}"),
            Self::ShapeMismatch {
                expected,
                actual,
                context,
            } => write!(
                f,
                "Vision shape mismatch ({context}): expected {expected}, got {actual}"
            ),
            Self::InvalidConfig(msg) => write!(f, "Vision invalid config: {msg}"),
            Self::DimensionsExceeded(msg) => write!(f, "Vision dimensions exceeded: {msg}"),
            Self::Io(e) => write!(f, "Vision I/O error: {e}"),
        }
    }
}

impl std::error::Error for VisionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for VisionError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(feature = "image")]
#[derive(Clone, Copy)]
struct DecodeLimits {
    pixels: u64,
    allocation_bytes: u64,
}

#[cfg(feature = "image")]
const EG2_MAX_DECODED_PIXELS: u64 = 8192 * 8192;
#[cfg(feature = "image")]
const EG2_MAX_DECODE_ALLOCATION_BYTES: u64 = 512 * 1024 * 1024;

#[cfg(feature = "image")]
const IMAGE_DECODE_LIMITS: DecodeLimits = DecodeLimits {
    pixels: EG2_MAX_DECODED_PIXELS,
    allocation_bytes: EG2_MAX_DECODE_ALLOCATION_BYTES,
};

/// Decodes PNG or JPEG input after checking the declared pixel count.
///
/// Pixel count is checked before pixel decoding, with no per-side cap. The image crate's
/// decoder allocation limit is additionally applied, but is a best-effort limit.
#[cfg(feature = "image")]
pub fn decode_image_bounded(image_bytes: &[u8]) -> Result<DynamicImage, VisionError> {
    decode_image_with_limits(image_bytes, IMAGE_DECODE_LIMITS)
}

#[cfg(feature = "image")]
fn decode_image_with_limits(
    image_bytes: &[u8],
    budget: DecodeLimits,
) -> Result<DynamicImage, VisionError> {
    if image_bytes.is_empty() {
        return Err(VisionError::ImageDecode("encoded image is empty".into()));
    }
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(budget.allocation_bytes);
    let mut header = ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .map_err(|e| VisionError::ImageDecode(format!("format detection failed: {e}")))?;
    if !matches!(
        header.format(),
        Some(image::ImageFormat::Png | image::ImageFormat::Jpeg)
    ) {
        return Err(VisionError::ImageDecode(
            "only PNG and JPEG images are supported".into(),
        ));
    }
    header.limits(limits.clone());
    let (width, height) = header.into_dimensions().map_err(|e| {
        VisionError::ImageDecode(format!(
            "dimension read failed under allocation limit {} bytes: {e}",
            budget.allocation_bytes
        ))
    })?;
    let pixels = u64::from(width) * u64::from(height);
    if width == 0 || height == 0 || pixels > budget.pixels {
        return Err(VisionError::DimensionsExceeded(format!(
            "declared image {width}x{height} has {pixels} pixels; pixel limit is {}",
            budget.pixels
        )));
    }
    let mut reader = ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .map_err(|e| VisionError::ImageDecode(format!("format detection failed: {e}")))?;
    reader.limits(limits);
    reader.decode().map_err(|e| {
        VisionError::ImageDecode(format!(
            "decode of image {width}x{height} failed under allocation limit {} bytes: {e}",
            budget.allocation_bytes
        ))
    })
}

#[cfg(all(test, feature = "image"))]
mod tests;

//! Validated RGB input and image patch geometry for EmbeddingGemma 2.

use crate::error::InferenceError;

mod resize;

const RGB_CHANNELS: usize = 3;
const PATCH_SIZE: usize = 16;
const POOLING_KERNEL_SIZE: usize = 3;
const REFERENCE_PATCH_CAPACITY: usize = 2520;
const PATCH_VALUES: usize = PATCH_SIZE * PATCH_SIZE * RGB_CHANNELS;
const RGB_RESCALE: f32 = 1.0 / 255.0;

/// Decoded RGB8 image with validated, tightly packed row-major pixels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmbeddingGemma2Rgb8Image {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl EmbeddingGemma2Rgb8Image {
    /// Creates an RGB image, rejecting zero dimensions, overflow and a wrong byte count.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self, InferenceError> {
        let expected = rgb_byte_count(width, height)?;
        if data.len() != expected {
            return Err(invalid_image(format!(
                "RGB byte count must be {expected} for {width}x{height}, got {}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Creates RGB pixels from RGBA8 bytes, retaining RGB channels and discarding alpha.
    pub fn from_rgba8(width: usize, height: usize, data: Vec<u8>) -> Result<Self, InferenceError> {
        let expected = channel_byte_count(width, height, 4)?;
        if data.len() != expected {
            return Err(invalid_image(format!(
                "RGBA byte count must be {expected}, got {}",
                data.len()
            )));
        }
        let mut rgb = rgb_buffer(width, height)?;
        for pixel in data.chunks_exact(4) {
            rgb.extend_from_slice(&pixel[..3]);
        }
        Self::new(width, height, rgb)
    }

    /// Creates RGB pixels by repeating each grayscale8 value across all three channels.
    pub fn from_grayscale8(
        width: usize,
        height: usize,
        data: Vec<u8>,
    ) -> Result<Self, InferenceError> {
        let expected = channel_byte_count(width, height, 1)?;
        if data.len() != expected {
            return Err(invalid_image(format!(
                "grayscale byte count must be {expected}, got {}",
                data.len()
            )));
        }
        let mut rgb = rgb_buffer(width, height)?;
        for value in data {
            rgb.extend_from_slice(&[value, value, value]);
        }
        Self::new(width, height, rgb)
    }

    /// Decodes bounded PNG or JPEG input with RGB8, RGBA8 or grayscale8 channels.
    ///
    /// Declared pixel count is limited before decoding, with no per-side cap. Decoder
    /// allocations have an additional best-effort limit. Higher bit depths are rejected.
    pub fn from_encoded(image_bytes: &[u8]) -> Result<Self, InferenceError> {
        let image = crate::vision::preprocess::decode_image_bounded(image_bytes)
            .map_err(|error| invalid_image(format!("encoded image decode failed: {error}")))?;
        Self::from_decoded(image)
    }

    fn from_decoded(image: image::DynamicImage) -> Result<Self, InferenceError> {
        let width = usize::try_from(image.width()).map_err(|_| {
            invalid_image("decoded image width exceeds the supported integer range")
        })?;
        let height = usize::try_from(image.height()).map_err(|_| {
            invalid_image("decoded image height exceeds the supported integer range")
        })?;
        match image {
            image::DynamicImage::ImageRgb8(pixels) => Self::new(width, height, pixels.into_raw()),
            image::DynamicImage::ImageRgba8(pixels) => {
                Self::from_rgba8(width, height, pixels.into_raw())
            }
            image::DynamicImage::ImageLuma8(pixels) => {
                Self::from_grayscale8(width, height, pixels.into_raw())
            }
            other => Err(invalid_image(format!(
                "decoded image color {:?} is unsupported; RGB8, RGBA8 or grayscale8 is required",
                other.color()
            ))),
        }
    }

    /// Image width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Tightly packed RGB bytes, in row-major pixel order.
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

fn invalid_image(message: impl Into<String>) -> InferenceError {
    InferenceError::InvalidInput(format!("EmbeddingGemma 2 image: {}", message.into()))
}

fn rgb_byte_count(width: usize, height: usize) -> Result<usize, InferenceError> {
    channel_byte_count(width, height, RGB_CHANNELS)
}

fn channel_byte_count(
    width: usize,
    height: usize,
    channels: usize,
) -> Result<usize, InferenceError> {
    if width == 0 || height == 0 {
        return Err(invalid_image("image dimensions must be positive"));
    }
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| invalid_image("RGB dimensions overflow the byte count"))
}

fn rgb_buffer(width: usize, height: usize) -> Result<Vec<u8>, InferenceError> {
    let mut data = Vec::new();
    data.try_reserve_exact(rgb_byte_count(width, height)?)
        .map_err(|error| invalid_image(format!("cannot allocate RGB pixels: {error}")))?;
    Ok(data)
}

#[allow(dead_code, reason = "The image forward path is not yet connected.")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PreparationRuleVersion {
    AspectPreserving48PatchBudget2520V1,
}

#[allow(dead_code, reason = "The image forward path is not yet connected.")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResizeKernelVersion {
    AlreadyResizedRgb8V1,
    TorchCpuUint8BicubicAaV1,
}

#[allow(dead_code, reason = "The image forward path is not yet connected.")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProcessorIdentity {
    rule: PreparationRuleVersion,
    kernel: ResizeKernelVersion,
}

#[allow(dead_code, reason = "The image forward path is not yet connected.")]
impl ProcessorIdentity {
    pub(crate) const fn already_resized_rgb8() -> Self {
        Self {
            rule: PreparationRuleVersion::AspectPreserving48PatchBudget2520V1,
            kernel: ResizeKernelVersion::AlreadyResizedRgb8V1,
        }
    }

    pub(crate) const fn torch_cpu_uint8_bicubic_aa() -> Self {
        Self {
            rule: PreparationRuleVersion::AspectPreserving48PatchBudget2520V1,
            kernel: ResizeKernelVersion::TorchCpuUint8BicubicAaV1,
        }
    }

    pub(crate) const fn rule(self) -> PreparationRuleVersion {
        self.rule
    }

    pub(crate) const fn kernel(self) -> ResizeKernelVersion {
        self.kernel
    }
}

#[allow(dead_code, reason = "The image forward path is not yet connected.")]
#[derive(Debug)]
pub(crate) struct PreparedImage {
    patch_values: Vec<f32>,
    positions: Vec<[usize; 2]>,
    grid_rows: usize,
    grid_cols: usize,
    valid_count: usize,
    soft_rows: usize,
    reference_capacity: usize,
    processor_identity: ProcessorIdentity,
}

#[allow(dead_code, reason = "The image forward path is not yet connected.")]
impl PreparedImage {
    pub(crate) fn new(
        patch_values: Vec<f32>,
        positions: Vec<[usize; 2]>,
        grid_rows: usize,
        grid_cols: usize,
        valid_count: usize,
        soft_rows: usize,
        reference_capacity: usize,
        processor_identity: ProcessorIdentity,
    ) -> Result<Self, InferenceError> {
        if grid_rows == 0 || grid_cols == 0 {
            return Err(invalid_image("patch grid dimensions must be positive"));
        }
        if !grid_rows.is_multiple_of(POOLING_KERNEL_SIZE)
            || !grid_cols.is_multiple_of(POOLING_KERNEL_SIZE)
        {
            return Err(invalid_image("patch grid sides must be multiples of 3"));
        }
        let grid_count = grid_rows
            .checked_mul(grid_cols)
            .ok_or_else(|| invalid_image("patch grid count overflow"))?;
        if valid_count != grid_count || valid_count > REFERENCE_PATCH_CAPACITY {
            return Err(invalid_image(format!(
                "valid patch count {valid_count} must equal grid count {grid_count} and be at most {REFERENCE_PATCH_CAPACITY}"
            )));
        }
        if soft_rows != valid_count / (POOLING_KERNEL_SIZE * POOLING_KERNEL_SIZE) {
            return Err(invalid_image(
                "soft rows must equal valid patch count divided by 9",
            ));
        }
        if reference_capacity != REFERENCE_PATCH_CAPACITY {
            return Err(invalid_image("reference patch capacity must be 2520"));
        }
        let expected_values = valid_count
            .checked_mul(PATCH_VALUES)
            .ok_or_else(|| invalid_image("patch value count overflow"))?;
        if patch_values.len() != expected_values {
            return Err(invalid_image(format!(
                "patch value count must be {expected_values}, got {}",
                patch_values.len()
            )));
        }
        if !patch_values.iter().all(|value| value.is_finite()) {
            return Err(invalid_image("patch values must be finite"));
        }
        if positions.len() != valid_count {
            return Err(invalid_image(format!(
                "position count must be {valid_count}, got {}",
                positions.len()
            )));
        }
        for (row, position) in positions.iter().enumerate() {
            if *position != [row % grid_cols, row / grid_cols] {
                return Err(invalid_image(format!(
                    "patch position at row {row} must be raster (x,y) coordinates"
                )));
            }
        }
        Ok(Self {
            patch_values,
            positions,
            grid_rows,
            grid_cols,
            valid_count,
            soft_rows,
            reference_capacity,
            processor_identity,
        })
    }

    pub(crate) fn patch_values(&self) -> &[f32] {
        &self.patch_values
    }

    pub(crate) fn positions(&self) -> &[[usize; 2]] {
        &self.positions
    }

    pub(crate) fn grid(&self) -> [usize; 2] {
        [self.grid_rows, self.grid_cols]
    }

    pub(crate) fn valid_count(&self) -> usize {
        self.valid_count
    }

    pub(crate) fn soft_rows(&self) -> usize {
        self.soft_rows
    }

    pub(crate) fn reference_capacity(&self) -> usize {
        self.reference_capacity
    }

    pub(crate) fn processor_identity(&self) -> ProcessorIdentity {
        self.processor_identity
    }
}

#[allow(dead_code, reason = "The image forward path is not yet connected.")]
pub(crate) fn aspect_ratio_preserving_size(
    height: usize,
    width: usize,
) -> Result<(usize, usize), InferenceError> {
    get_aspect_ratio_preserving_size(
        height,
        width,
        PATCH_SIZE,
        REFERENCE_PATCH_CAPACITY,
        POOLING_KERNEL_SIZE,
    )
}

#[allow(dead_code, reason = "The image forward path is not yet connected.")]
fn get_aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize), InferenceError> {
    if height == 0 || width == 0 || patch_size == 0 || max_patches == 0 || pooling_kernel_size == 0
    {
        return Err(invalid_image(
            "resize dimensions and parameters must be positive",
        ));
    }
    let total_pixels = height
        .checked_mul(width)
        .ok_or_else(|| invalid_image("source image area overflow"))?;
    let target_pixels = patch_size
        .checked_mul(patch_size)
        .and_then(|patch_area| max_patches.checked_mul(patch_area))
        .ok_or_else(|| invalid_image("target image area overflow"))?;
    let side_multiple = pooling_kernel_size
        .checked_mul(patch_size)
        .ok_or_else(|| invalid_image("resize side multiple overflow"))?;
    let pooling_area = pooling_kernel_size
        .checked_mul(pooling_kernel_size)
        .ok_or_else(|| invalid_image("pooling area overflow"))?;
    let factor = ((target_pixels as f64) / (total_pixels as f64)).sqrt();
    let mut target_height = floor_to_multiple(factor * height as f64, side_multiple)?;
    let mut target_width = floor_to_multiple(factor * width as f64, side_multiple)?;
    if target_height == 0 && target_width == 0 {
        return Err(invalid_image("resize would produce a zero by zero image"));
    }
    let max_side_length = (max_patches / pooling_area)
        .checked_mul(side_multiple)
        .ok_or_else(|| invalid_image("maximum resize side overflow"))?;
    if target_height == 0 {
        target_height = side_multiple;
        target_width =
            capped_aspect_side(width as f64 / height as f64, side_multiple, max_side_length)?;
    } else if target_width == 0 {
        target_width = side_multiple;
        target_height =
            capped_aspect_side(height as f64 / width as f64, side_multiple, max_side_length)?;
    }
    let resized_pixels = target_height
        .checked_mul(target_width)
        .ok_or_else(|| invalid_image("resized image area overflow"))?;
    if resized_pixels > target_pixels {
        return Err(invalid_image("resized image exceeds the patch budget"));
    }
    Ok((target_height, target_width))
}

fn floor_to_multiple(value: f64, multiple: usize) -> Result<usize, InferenceError> {
    floor_integer(value / multiple as f64)?
        .checked_mul(multiple)
        .ok_or_else(|| invalid_image("rounded resize side overflow"))
}

fn capped_aspect_side(ratio: f64, multiple: usize, cap: usize) -> Result<usize, InferenceError> {
    if ratio >= (cap / multiple) as f64 {
        return Ok(cap);
    }
    floor_integer(ratio)?
        .checked_mul(multiple)
        .ok_or_else(|| invalid_image("aspect-ratio resize side overflow"))
}

fn floor_integer(value: f64) -> Result<usize, InferenceError> {
    let rounded = value.floor();
    if !rounded.is_finite() || rounded < 0.0 || rounded >= usize::MAX as f64 {
        return Err(invalid_image(
            "resize dimension is outside the supported integer range",
        ));
    }
    Ok(rounded as usize)
}

#[allow(dead_code, reason = "The image forward path is not yet connected.")]
pub(crate) fn pack_already_resized_rgb8(
    image: &EmbeddingGemma2Rgb8Image,
) -> Result<PreparedImage, InferenceError> {
    pack_rgb8_with_identity(image, ProcessorIdentity::already_resized_rgb8())
}

#[allow(dead_code, reason = "The image forward path is not yet connected.")]
pub(crate) fn prepare_image(
    image: &EmbeddingGemma2Rgb8Image,
) -> Result<PreparedImage, InferenceError> {
    let (height, width) = aspect_ratio_preserving_size(image.height, image.width)?;
    let resized = resize::resize_rgb8_bicubic_aa(image, width, height)?;
    pack_rgb8_with_identity(&resized, ProcessorIdentity::torch_cpu_uint8_bicubic_aa())
}

fn pack_rgb8_with_identity(
    image: &EmbeddingGemma2Rgb8Image,
    processor_identity: ProcessorIdentity,
) -> Result<PreparedImage, InferenceError> {
    let side_multiple = PATCH_SIZE * POOLING_KERNEL_SIZE;
    if !image.height.is_multiple_of(side_multiple) || !image.width.is_multiple_of(side_multiple) {
        return Err(invalid_image(
            "already-resized RGB sides must be multiples of 48",
        ));
    }
    let grid_rows = image.height / PATCH_SIZE;
    let grid_cols = image.width / PATCH_SIZE;
    let valid_count = grid_rows
        .checked_mul(grid_cols)
        .ok_or_else(|| invalid_image("patch grid count overflow"))?;
    if valid_count > REFERENCE_PATCH_CAPACITY {
        return Err(invalid_image(
            "already-resized RGB exceeds the 2520-patch budget",
        ));
    }
    let values_len = valid_count
        .checked_mul(PATCH_VALUES)
        .ok_or_else(|| invalid_image("patch value count overflow"))?;
    let mut patch_values = Vec::new();
    patch_values
        .try_reserve_exact(values_len)
        .map_err(|error| invalid_image(format!("cannot allocate patch values: {error}")))?;
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(valid_count)
        .map_err(|error| invalid_image(format!("cannot allocate patch positions: {error}")))?;
    for grid_y in 0..grid_rows {
        for grid_x in 0..grid_cols {
            positions.push([grid_x, grid_y]);
            for dy in 0..PATCH_SIZE {
                for dx in 0..PATCH_SIZE {
                    let pixel_y = grid_y * PATCH_SIZE + dy;
                    let pixel_x = grid_x * PATCH_SIZE + dx;
                    let offset = (pixel_y * image.width + pixel_x) * RGB_CHANNELS;
                    for channel in 0..RGB_CHANNELS {
                        patch_values.push(f32::from(image.data[offset + channel]) * RGB_RESCALE);
                    }
                }
            }
        }
    }
    PreparedImage::new(
        patch_values,
        positions,
        grid_rows,
        grid_cols,
        valid_count,
        valid_count / (POOLING_KERNEL_SIZE * POOLING_KERNEL_SIZE),
        REFERENCE_PATCH_CAPACITY,
        processor_identity,
    )
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod reference_tests;

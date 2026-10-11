use super::EmbeddingGemma2Rgb8Image;
use crate::error::InferenceError;

struct FilterRow {
    first: usize,
    coefficients: Vec<i16>,
}

struct AxisFilter {
    rows: Vec<FilterRow>,
    precision: u32,
}

fn invalid(message: &str) -> InferenceError {
    InferenceError::InvalidInput(format!("EmbeddingGemma 2 image resize: {message}"))
}

fn cubic_weight(distance: f64) -> f64 {
    let x = distance.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

fn round_coefficient(value: f64) -> i32 {
    if value < 0.0 {
        (value - 0.5) as i32
    } else {
        (value + 0.5) as i32
    }
}

fn axis_filter(input: usize, output: usize) -> Result<AxisFilter, InferenceError> {
    if input == 0 || output == 0 || input > i32::MAX as usize || output > 13_440 {
        return Err(invalid("invalid or unsupported axis dimensions"));
    }
    let scale = input as f64 / output as f64;
    let support = 2.0 * scale.max(1.0);
    let inverse_scale = 1.0 / scale.max(1.0);
    let max_taps = (support.ceil() as usize)
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| invalid("filter support overflow"))?;
    let mut normalized = Vec::new();
    normalized
        .try_reserve_exact(output)
        .map_err(|_| invalid("filter allocation failed"))?;
    let mut largest = 0.0_f64;
    for destination in 0..output {
        let center = scale * (destination as f64 + 0.5);
        let first = ((center - support + 0.5) as i64).max(0) as usize;
        let last = ((center + support + 0.5) as usize).min(input);
        let taps = last.saturating_sub(first).min(max_taps);
        if taps == 0 {
            return Err(invalid("empty filter support"));
        }
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(taps)
            .map_err(|_| invalid("weight allocation failed"))?;
        let mut total = 0.0;
        for offset in 0..taps {
            let distance = (offset as f64 + first as f64 - center + 0.5) * inverse_scale;
            let weight = cubic_weight(distance);
            weights.push(weight);
            total += weight;
        }
        if !total.is_finite() || total == 0.0 {
            return Err(invalid("invalid filter normalization"));
        }
        for weight in &mut weights {
            *weight /= total;
            largest = largest.max(*weight);
        }
        normalized.push((first, weights));
    }
    let mut precision = 0;
    while precision < 22 {
        let next = (0.5 + largest * f64::from(1_u32 << (precision + 1))) as i64;
        if next >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    if precision == 0 {
        return Err(invalid("invalid fixed-point precision"));
    }
    let multiplier = f64::from(1_u32 << precision);
    let mut rows = Vec::new();
    rows.try_reserve_exact(output)
        .map_err(|_| invalid("filter allocation failed"))?;
    for (first, weights) in normalized {
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(weights.len())
            .map_err(|_| invalid("coefficient allocation failed"))?;
        for weight in weights {
            let value = weight * multiplier;
            let integer = round_coefficient(value);
            if !value.is_finite() || integer < i32::from(i16::MIN) || integer > i32::from(i16::MAX)
            {
                return Err(invalid("fixed-point coefficient out of range"));
            }
            coefficients.push(integer as i16);
        }
        rows.push(FilterRow {
            first,
            coefficients,
        });
    }
    Ok(AxisFilter { rows, precision })
}

fn rgb_buffer(width: usize, height: usize) -> Result<Vec<u8>, InferenceError> {
    let length = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| invalid("image byte count overflow"))?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(length)
        .map_err(|_| invalid("image allocation failed"))?;
    pixels.resize(length, 0);
    Ok(pixels)
}

fn filtered_pixel(row: &FilterRow, precision: u32, mut value: impl FnMut(usize) -> u8) -> u8 {
    let mut sum = 1_i64 << (precision - 1);
    for (offset, coefficient) in row.coefficients.iter().enumerate() {
        sum += i64::from(value(row.first + offset)) * i64::from(*coefficient);
    }
    (sum >> precision).clamp(0, 255) as u8
}

pub(super) fn resize_rgb8_bicubic_aa(
    image: &EmbeddingGemma2Rgb8Image,
    width: usize,
    height: usize,
) -> Result<EmbeddingGemma2Rgb8Image, InferenceError> {
    if width == 0 || height == 0 || width > 13_440 || height > 13_440 {
        return Err(invalid("invalid target dimensions"));
    }
    let input_width = image.width();
    let input_height = image.height();
    let horizontal = if width != input_width {
        let filter = axis_filter(input_width, width)?;
        let mut pixels = rgb_buffer(width, input_height)?;
        for y in 0..input_height {
            for x in 0..width {
                for channel in 0..3 {
                    pixels[(y * width + x) * 3 + channel] =
                        filtered_pixel(&filter.rows[x], filter.precision, |source_x| {
                            image.data()[(y * input_width + source_x) * 3 + channel]
                        });
                }
            }
        }
        Some(pixels)
    } else {
        None
    };
    let intermediate = horizontal.as_deref().unwrap_or(image.data());
    let mut output = rgb_buffer(width, height)?;
    if height != input_height {
        let filter = axis_filter(input_height, height)?;
        for y in 0..height {
            for x in 0..width {
                for channel in 0..3 {
                    output[(y * width + x) * 3 + channel] =
                        filtered_pixel(&filter.rows[y], filter.precision, |source_y| {
                            intermediate[(source_y * width + x) * 3 + channel]
                        });
                }
            }
        }
    } else {
        output.copy_from_slice(intermediate);
    }
    EmbeddingGemma2Rgb8Image::new(width, height, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_cubic_has_negative_lobes_and_support_two() {
        assert_eq!(cubic_weight(0.0), 1.0);
        assert_eq!(cubic_weight(0.5), 0.5625);
        assert_eq!(cubic_weight(1.5), -0.0625);
        assert_eq!(cubic_weight(-1.5), -0.0625);
        assert_eq!(cubic_weight(2.0), 0.0);
    }

    #[test]
    fn coefficient_rounding_preserves_half_unit_addition() {
        let below_half = f64::from_bits(0.5_f64.to_bits() - 1);
        assert_eq!(below_half.round(), 0.0);
        assert_eq!((-below_half).round(), -0.0);
        assert_eq!(round_coefficient(below_half), 1);
        assert_eq!(round_coefficient(-below_half), -1);
        assert_eq!(round_coefficient(0.49), 0);
        assert_eq!(round_coefficient(-0.49), 0);
        assert_eq!(round_coefficient(1.5), 2);
        assert_eq!(round_coefficient(-1.5), -2);
    }

    #[test]
    fn normalized_half_pixel_weights_have_reference_fixed_point_values() {
        // For 2→3, the edge weights normalize to [81/76,-5/76]; the middle is [1/2,1/2].
        let filter = axis_filter(2, 3).unwrap();
        assert_eq!(filter.precision, 14);
        assert_eq!(filter.rows[0].first, 0);
        assert_eq!(filter.rows[0].coefficients, [17_462, -1_078]);
        assert_eq!(filter.rows[1].coefficients, [8_192, 8_192]);
        assert_eq!(filter.rows[2].coefficients, [-1_078, 17_462]);
    }

    #[test]
    fn uint8_resampling_rounds_and_clips_after_each_axis() {
        let image =
            EmbeddingGemma2Rgb8Image::new(2, 2, vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2]).unwrap();
        let resized = resize_rgb8_bicubic_aa(&image, 1, 1).unwrap();
        assert_eq!(resized.data(), [2, 2, 2]);
        let ramp = EmbeddingGemma2Rgb8Image::new(2, 1, vec![0, 0, 0, 255, 255, 255]).unwrap();
        assert_eq!(
            resize_rgb8_bicubic_aa(&ramp, 3, 1).unwrap().data(),
            [0, 0, 0, 128, 128, 128, 255, 255, 255]
        );
    }

    #[test]
    fn antialias_downsampling_expands_support_and_renormalizes_edges() {
        let filter = axis_filter(8, 2).unwrap();
        assert!(filter.rows[0].coefficients.len() > 4);
        assert_eq!(filter.rows[0].first, 0);
        let image = EmbeddingGemma2Rgb8Image::new(8, 1, vec![123; 24]).unwrap();
        assert_eq!(
            resize_rgb8_bicubic_aa(&image, 2, 1).unwrap().data(),
            [123; 6]
        );
    }

    #[test]
    fn unchanged_dimensions_preserve_every_byte() {
        let image = EmbeddingGemma2Rgb8Image::new(2, 2, (0..12).collect()).unwrap();
        assert_eq!(
            resize_rgb8_bicubic_aa(&image, 2, 2).unwrap().data(),
            image.data()
        );
    }

    #[test]
    fn resize_rejects_invalid_dimensions() {
        let image = EmbeddingGemma2Rgb8Image::new(1, 1, vec![0; 3]).unwrap();
        for (width, height) in [(0, 1), (1, 0), (13_441, 1), (1, 13_441)] {
            assert!(resize_rgb8_bicubic_aa(&image, width, height).is_err());
        }
        assert!(axis_filter(0, 1).is_err());
    }
}

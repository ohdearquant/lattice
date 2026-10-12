use super::*;

fn assert_invalid<T: std::fmt::Debug>(result: Result<T, InferenceError>, message: &str) {
    match result {
        Err(InferenceError::InvalidInput(actual)) => assert!(actual.contains(message), "{actual}"),
        other => panic!("expected InvalidInput containing {message:?}, got {other:?}"),
    }
}

fn valid_parts() -> (Vec<f32>, Vec<[usize; 2]>) {
    let values = vec![0.0; 9 * PATCH_VALUES];
    let positions = (0..9).map(|row| [row % 3, row / 3]).collect();
    (values, positions)
}

fn prepared(values: Vec<f32>, positions: Vec<[usize; 2]>) -> Result<PreparedImage, InferenceError> {
    PreparedImage::new(
        values,
        positions,
        3,
        3,
        9,
        1,
        2520,
        ProcessorIdentity::already_resized_rgb8(),
    )
}

#[test]
fn rgb8_rejects_zero_dimensions() {
    assert_invalid(EmbeddingGemma2Rgb8Image::new(0, 1, vec![]), "positive");
    assert_invalid(EmbeddingGemma2Rgb8Image::new(1, 0, vec![]), "positive");
}

#[test]
fn rgb8_rejects_area_and_channel_count_overflow() {
    assert_invalid(
        EmbeddingGemma2Rgb8Image::new(usize::MAX, 2, vec![]),
        "overflow",
    );
    assert_invalid(
        EmbeddingGemma2Rgb8Image::new(usize::MAX / 2, 1, vec![]),
        "overflow",
    );
}

#[test]
fn rgb8_rejects_wrong_byte_count() {
    assert_invalid(
        EmbeddingGemma2Rgb8Image::new(2, 1, vec![0; 5]),
        "byte count",
    );
    assert_invalid(
        EmbeddingGemma2Rgb8Image::new(2, 1, vec![0; 7]),
        "byte count",
    );
}

#[test]
fn rgb8_preserves_dimensions_and_rgb_bytes() {
    let bytes = vec![0, 1, 2, 253, 254, 255];
    let image = EmbeddingGemma2Rgb8Image::new(2, 1, bytes.clone()).unwrap();
    assert_eq!(image.width(), 2);
    assert_eq!(image.height(), 1);
    assert_eq!(image.data(), bytes);
}

#[test]
fn rgba8_conversion_discards_alpha_without_blending_rgb() {
    let rgba = vec![9, 20, 31, 0, 8, 19, 30, 1, 7, 18, 29, 128, 6, 17, 28, 255];
    let image = EmbeddingGemma2Rgb8Image::from_rgba8(4, 1, rgba).unwrap();
    assert_eq!(image.data(), [9, 20, 31, 8, 19, 30, 7, 18, 29, 6, 17, 28]);
}

#[test]
fn grayscale8_conversion_repeats_the_original_channel() {
    let image = EmbeddingGemma2Rgb8Image::from_grayscale8(4, 1, vec![0, 1, 128, 255]).unwrap();
    assert_eq!(
        image.data(),
        [0, 0, 0, 1, 1, 1, 128, 128, 128, 255, 255, 255]
    );
}

#[test]
fn conversions_validate_dimensions_and_mode_byte_counts() {
    assert_invalid(
        EmbeddingGemma2Rgb8Image::from_rgba8(0, 1, vec![]),
        "positive",
    );
    assert_invalid(
        EmbeddingGemma2Rgb8Image::from_grayscale8(1, 0, vec![]),
        "positive",
    );
    assert_invalid(
        EmbeddingGemma2Rgb8Image::from_rgba8(1, 1, vec![0; 3]),
        "RGBA byte count",
    );
    assert_invalid(
        EmbeddingGemma2Rgb8Image::from_rgba8(1, 1, vec![0; 5]),
        "RGBA byte count",
    );
    assert_invalid(
        EmbeddingGemma2Rgb8Image::from_grayscale8(1, 1, vec![]),
        "grayscale byte count",
    );
    assert_invalid(
        EmbeddingGemma2Rgb8Image::from_grayscale8(1, 1, vec![0; 2]),
        "grayscale byte count",
    );
    assert_invalid(
        EmbeddingGemma2Rgb8Image::from_rgba8(usize::MAX / 2, 1, vec![]),
        "overflow",
    );
    assert_invalid(
        EmbeddingGemma2Rgb8Image::from_grayscale8(usize::MAX, 2, vec![]),
        "overflow",
    );
}

fn png_bytes(image: image::DynamicImage) -> Vec<u8> {
    let mut output = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut output, image::ImageFormat::Png)
        .unwrap();
    output.into_inner()
}

#[test]
fn encoded_rgb_rgba_and_grayscale_pngs_reuse_validated_conversion() {
    let cases = [
        (
            image::DynamicImage::ImageRgb8(
                image::RgbImage::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap(),
            ),
            vec![1, 2, 3, 4, 5, 6],
        ),
        (
            image::DynamicImage::ImageRgba8(
                image::RgbaImage::from_raw(2, 1, vec![1, 2, 3, 0, 4, 5, 6, 255]).unwrap(),
            ),
            vec![1, 2, 3, 4, 5, 6],
        ),
        (
            image::DynamicImage::ImageLuma8(
                image::GrayImage::from_raw(2, 1, vec![0, 255]).unwrap(),
            ),
            vec![0, 0, 0, 255, 255, 255],
        ),
    ];
    for (source, expected) in cases {
        let result = EmbeddingGemma2Rgb8Image::from_encoded(&png_bytes(source)).unwrap();
        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 1);
        assert_eq!(result.data(), expected);
    }
}

#[test]
fn encoded_failures_use_typed_invalid_input() {
    for bytes in [
        b"".as_slice(),
        b"not an image".as_slice(),
        b"GIF89a\x01\x00\x01\x00".as_slice(),
    ] {
        assert_invalid(
            EmbeddingGemma2Rgb8Image::from_encoded(bytes),
            "encoded image decode failed",
        );
    }
}

#[test]
fn encoded_higher_bit_depth_is_rejected() {
    let pixels =
        image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(1, 1, vec![65535]).unwrap();
    let bytes = png_bytes(image::DynamicImage::ImageLuma16(pixels));
    assert_invalid(
        EmbeddingGemma2Rgb8Image::from_encoded(&bytes),
        "unsupported",
    );
}

#[test]
fn source_geometry_table() {
    let cases = [
        // sqrt(645120/192²)*192 ≈803.19; floor(803.19/48)*48=768; 48²/9=256.
        ((192, 192), (768, 768), [48, 48], 2304, 256),
        // sqrt(645120/384²)*384 ≈803.19; both floors give768; 48²=2304.
        ((384, 384), (768, 768), [48, 48], 2304, 256),
        // sqrt(645120/768²)*768 ≈803.19; both floors give768; 2304/9=256.
        ((768, 768), (768, 768), [48, 48], 2304, 256),
        // sqrt(645120/(480*640))*[480,640] ≈[695.59,927.45]; floors [672,912]; 42*57/9=266.
        ((480, 640), (672, 912), [42, 57], 2394, 266),
        // sqrt(645120/(672*912))*[672,912] ≈[689.47,935.70]; floors [672,912]; 42*57=2394.
        ((672, 912), (672, 912), [42, 57], 2394, 266),
        // 672*960=645120, factor1; grid42*60=2520, then2520/9=280.
        ((672, 960), (672, 960), [42, 60], 2520, 280),
        // sqrt(645120/(1080*1920))*[1080,1920] ≈[602.40,1070.93]; floors [576,1056]; 36*66/9=264.
        ((1080, 1920), (576, 1056), [36, 66], 2376, 264),
        // sqrt(645120/20000)*[100,200] ≈[567.94,1135.89]; floors [528,1104]; 33*69/9=253.
        ((100, 200), (528, 1104), [33, 69], 2277, 253),
        // Height floors0; it becomes48 and width=min(floor(100000/1)*48,280*48)=13440; 3*840/9=280.
        ((1, 100000), (48, 13440), [3, 840], 2520, 280),
    ];
    for ((height, width), expected, grid, valid_count, soft_rows) in cases {
        let resized = aspect_ratio_preserving_size(height, width).unwrap();
        assert_eq!(resized, expected, "source {height}x{width}");
        assert_eq!([resized.0 / 16, resized.1 / 16], grid);
        assert_eq!(grid[0] * grid[1], valid_count);
        assert_eq!(valid_count / 9, soft_rows);
    }
}

#[test]
fn source_geometry_symmetric_extreme_aspect() {
    // Width floors0; it becomes48 and height=min(floor(100000/1)*48,280*48)=13440.
    assert_eq!(
        aspect_ratio_preserving_size(100000, 1).unwrap(),
        (13440, 48)
    );
}

#[test]
fn source_geometry_caps_extreme_side_before_integer_multiplication() {
    assert_eq!(
        aspect_ratio_preserving_size(1, usize::MAX / 3).unwrap(),
        (48, 13440)
    );
}

#[test]
fn source_geometry_rejects_zero_dimensions_and_parameters() {
    for (height, width, patch, budget, pooling) in [
        (0, 1, 16, 2520, 3),
        (1, 0, 16, 2520, 3),
        (1, 1, 0, 2520, 3),
        (1, 1, 16, 0, 3),
        (1, 1, 16, 2520, 0),
    ] {
        assert_invalid(
            get_aspect_ratio_preserving_size(height, width, patch, budget, pooling),
            "positive",
        );
    }
}

#[test]
fn source_geometry_rejects_both_rounded_sides_zero() {
    // Target area16²=256 gives ideal16x16; floor(16/48)*48=0 in both axes.
    assert_invalid(
        get_aspect_ratio_preserving_size(1, 1, 16, 1, 3),
        "zero by zero",
    );
}

#[test]
fn source_geometry_rejects_checked_arithmetic_overflow() {
    assert_invalid(
        aspect_ratio_preserving_size(usize::MAX, 2),
        "source image area overflow",
    );
    assert_invalid(
        get_aspect_ratio_preserving_size(1, 1, usize::MAX, 1, 1),
        "target image area overflow",
    );
    assert_invalid(
        get_aspect_ratio_preserving_size(1, 1, 16, usize::MAX, 3),
        "target image area overflow",
    );
    assert_invalid(
        get_aspect_ratio_preserving_size(1, 1, 2, 1, usize::MAX),
        "resize side multiple overflow",
    );
    assert_invalid(
        get_aspect_ratio_preserving_size(1, 1, 1, 1, usize::MAX),
        "pooling area overflow",
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn source_geometry_rejects_float_rounding_over_budget() {
    // f64 rounds (2^54-1) to2^54; sqrt gives2^27, so integer target area2^54 exceeds the true budget.
    assert_invalid(
        get_aspect_ratio_preserving_size(1, 1, 1, (1usize << 54) - 1, 1),
        "exceeds the patch budget",
    );
}

#[test]
fn source_geometry_rejects_unsupported_float_integer_conversion() {
    assert_invalid(floor_integer(f64::NAN), "supported integer range");
    assert_invalid(floor_integer(f64::INFINITY), "supported integer range");
    assert_invalid(floor_integer(-1.0), "supported integer range");
    assert_invalid(floor_integer(usize::MAX as f64), "supported integer range");
    assert_eq!(floor_integer(1.9).unwrap(), 1);
}

#[test]
fn packing_preserves_raster_patch_pixel_channel_order_without_padding() {
    let (width, height) = (96, 48);
    let mut bytes = Vec::new();
    for y in 0..height {
        for x in 0..width {
            bytes.extend_from_slice(&[x as u8, y as u8, (x + 2 * y) as u8]);
        }
    }
    let image = EmbeddingGemma2Rgb8Image::new(width, height, bytes).unwrap();
    let result = pack_already_resized_rgb8(&image).unwrap();
    assert_eq!(result.grid(), [3, 6]);
    assert_eq!(result.valid_count(), 18);
    assert_eq!(result.soft_rows(), 2);
    assert_eq!(result.reference_capacity(), 2520);
    assert_eq!(result.patch_values().len(), 18 * 768);
    assert_eq!(result.positions().len(), 18);
    for row in 0..18 {
        let [grid_x, grid_y] = [row % 6, row / 6];
        assert_eq!(result.positions()[row], [grid_x, grid_y]);
        for dy in 0..16 {
            for dx in 0..16 {
                let x = grid_x * 16 + dx;
                let y = grid_y * 16 + dy;
                let expected = [x as u8, y as u8, (x + 2 * y) as u8];
                for (channel, byte) in expected.into_iter().enumerate() {
                    let offset = row * 768 + (dy * 16 + dx) * 3 + channel;
                    assert_eq!(
                        result.patch_values()[offset],
                        f32::from(byte) * (1.0 / 255.0)
                    );
                }
            }
        }
    }
    assert_eq!(
        result.processor_identity().rule(),
        PreparationRuleVersion::AspectPreserving48PatchBudget2520V1
    );
    assert_eq!(
        result.processor_identity().kernel(),
        ResizeKernelVersion::AlreadyResizedRgb8V1
    );
}

#[test]
fn packing_rescales_uint8_endpoints() {
    let mut bytes = vec![0; 48 * 48 * 3];
    bytes[0..3].copy_from_slice(&[0, 128, 255]);
    let image = EmbeddingGemma2Rgb8Image::new(48, 48, bytes).unwrap();
    let result = pack_already_resized_rgb8(&image).unwrap();
    assert_eq!(result.patch_values()[0], 0.0);
    assert_eq!(result.patch_values()[1], 128.0 * (1.0 / 255.0));
    assert_eq!(result.patch_values()[2], 1.0);
}

#[test]
fn preparation_of_square_constant_uses_reference_geometry_and_identity() {
    let mut bytes = Vec::new();
    for _ in 0..192 * 192 {
        bytes.extend_from_slice(&[0, 128, 255]);
    }
    let image = EmbeddingGemma2Rgb8Image::new(192, 192, bytes).unwrap();
    let result = prepare_image(&image).unwrap();
    assert_eq!(result.grid(), [48, 48]);
    assert_eq!(result.valid_count(), 2304);
    assert_eq!(result.soft_rows(), 256);
    assert_eq!(result.reference_capacity(), 2520);
    assert_eq!(result.patch_values().len(), 2304 * 768);
    for pixel in result.patch_values().chunks_exact(3) {
        assert_eq!(pixel, [0.0, 128.0 * (1.0 / 255.0), 1.0]);
    }
    assert_eq!(
        result.processor_identity().kernel(),
        ResizeKernelVersion::TorchCpuUint8BicubicAaV1
    );
}

#[test]
fn preparation_bypasses_resize_at_exact_patch_budget() {
    let (width, height) = (960, 672);
    let mut bytes = Vec::new();
    for y in 0..height {
        for x in 0..width {
            bytes.extend_from_slice(&[(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]);
        }
    }
    let image = EmbeddingGemma2Rgb8Image::new(width, height, bytes).unwrap();
    let direct = pack_already_resized_rgb8(&image).unwrap();
    let prepared = prepare_image(&image).unwrap();
    assert_eq!(prepared.grid(), [42, 60]);
    assert_eq!(prepared.valid_count(), 2520);
    assert_eq!(prepared.soft_rows(), 280);
    assert_eq!(prepared.patch_values(), direct.patch_values());
    assert_eq!(prepared.positions(), direct.positions());
    assert_eq!(
        prepared.processor_identity().rule(),
        PreparationRuleVersion::AspectPreserving48PatchBudget2520V1
    );
    assert_eq!(
        prepared.processor_identity().kernel(),
        ResizeKernelVersion::TorchCpuUint8BicubicAaV1
    );
}

#[test]
fn packing_rejects_unaligned_dimensions_and_excess_capacity() {
    for (width, height) in [(47, 48), (48, 47)] {
        let image =
            EmbeddingGemma2Rgb8Image::new(width, height, vec![0; width * height * 3]).unwrap();
        assert_invalid(pack_already_resized_rgb8(&image), "multiples of 48");
    }
    let image = EmbeddingGemma2Rgb8Image::new(816, 816, vec![0; 816 * 816 * 3]).unwrap();
    assert_invalid(pack_already_resized_rgb8(&image), "2520-patch budget");
}

#[test]
fn prepared_rejects_zero_grid_dimensions() {
    for (rows, cols) in [(0, 3), (3, 0)] {
        assert_invalid(
            PreparedImage::new(
                vec![],
                vec![],
                rows,
                cols,
                0,
                0,
                2520,
                ProcessorIdentity::already_resized_rgb8(),
            ),
            "positive",
        );
    }
}

#[test]
fn prepared_rejects_grid_sides_not_divisible_by_pooling_size() {
    for (rows, cols) in [(2, 3), (3, 2)] {
        assert_invalid(
            PreparedImage::new(
                vec![],
                vec![],
                rows,
                cols,
                6,
                0,
                2520,
                ProcessorIdentity::already_resized_rgb8(),
            ),
            "multiples of 3",
        );
    }
}

#[test]
fn prepared_rejects_grid_count_overflow() {
    assert_invalid(
        PreparedImage::new(
            vec![],
            vec![],
            usize::MAX - usize::MAX % 3,
            3,
            0,
            0,
            2520,
            ProcessorIdentity::already_resized_rgb8(),
        ),
        "patch grid count overflow",
    );
}

#[test]
fn prepared_rejects_inconsistent_or_excess_valid_count() {
    let (values, positions) = valid_parts();
    assert_invalid(
        PreparedImage::new(
            values,
            positions,
            3,
            3,
            8,
            0,
            2520,
            ProcessorIdentity::already_resized_rgb8(),
        ),
        "valid patch count",
    );
    assert_invalid(
        PreparedImage::new(
            vec![],
            vec![],
            51,
            51,
            2601,
            289,
            2520,
            ProcessorIdentity::already_resized_rgb8(),
        ),
        "at most 2520",
    );
}

#[test]
fn prepared_rejects_inconsistent_soft_rows() {
    let (values, positions) = valid_parts();
    assert_invalid(
        PreparedImage::new(
            values,
            positions,
            3,
            3,
            9,
            2,
            2520,
            ProcessorIdentity::already_resized_rgb8(),
        ),
        "soft rows",
    );
}

#[test]
fn prepared_rejects_wrong_reference_capacity() {
    let (values, positions) = valid_parts();
    assert_invalid(
        PreparedImage::new(
            values,
            positions,
            3,
            3,
            9,
            1,
            2519,
            ProcessorIdentity::already_resized_rgb8(),
        ),
        "reference patch capacity",
    );
}

#[test]
fn prepared_rejects_wrong_patch_value_length() {
    let (mut values, positions) = valid_parts();
    values.pop();
    assert_invalid(prepared(values, positions), "patch value count");
    let (mut values, positions) = valid_parts();
    values.push(0.0);
    assert_invalid(prepared(values, positions), "patch value count");
}

#[test]
fn prepared_rejects_nonfinite_patch_values() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let (mut values, positions) = valid_parts();
        values[1] = bad;
        assert_invalid(prepared(values, positions), "finite");
    }
}

#[test]
fn prepared_rejects_wrong_position_length() {
    let (values, mut positions) = valid_parts();
    positions.pop();
    assert_invalid(prepared(values, positions), "position count");
    let (values, mut positions) = valid_parts();
    positions.push([0, 3]);
    assert_invalid(prepared(values, positions), "position count");
}

#[test]
fn prepared_rejects_nonraster_duplicate_or_outside_positions() {
    for changed in [[1, 0], [3, 0], [0, 3], [usize::MAX, usize::MAX]] {
        let (values, mut positions) = valid_parts();
        positions[0] = changed;
        assert_invalid(prepared(values, positions), "raster (x,y)");
    }
    let (values, mut positions) = valid_parts();
    positions.swap(1, 3);
    assert_invalid(prepared(values, positions), "raster (x,y)");
}

#[test]
fn prepared_validated_value_preserves_identity_and_data() {
    let (values, positions) = valid_parts();
    let result = prepared(values.clone(), positions.clone()).unwrap();
    assert_eq!(result.patch_values(), values);
    assert_eq!(result.positions(), positions);
    assert_eq!(result.grid(), [3, 3]);
    assert_eq!(result.valid_count(), 9);
    assert_eq!(result.soft_rows(), 1);
    assert_eq!(result.reference_capacity(), 2520);
    assert_eq!(
        result.processor_identity(),
        ProcessorIdentity::already_resized_rgb8()
    );
}

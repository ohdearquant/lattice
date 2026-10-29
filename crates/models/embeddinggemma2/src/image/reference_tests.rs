//! Processor digests observed on x86_64 CPU with PyTorch 2.14.1 and Torchvision 0.29.1.
//! The primary processor probe agrees across AVX512, AVX2 and DEFAULT capabilities.
//! Synthetic recipe source SHA-256:
//! 10a0445192aa9ea0bee003f1ffc87258351048dea9165535130406d1189c6eaa.

use sha2::{Digest, Sha256};

use super::{
    EmbeddingGemma2Rgb8Image, PreparationRuleVersion, PreparedImage, ResizeKernelVersion,
    aspect_ratio_preserving_size, prepare_image, resize,
};

#[derive(Clone, Copy, Debug)]
enum Recipe {
    BlackSquare,
    WhiteAligned,
    RedSmall,
    GradientNonmultiple,
    CheckerAboveBudget,
    BudgetAligned,
    ExtremeLandscape,
    RgbaNonmultiple,
    GraySmall,
}

impl Recipe {
    fn source_geometry(self) -> (usize, usize, usize) {
        match self {
            Self::BlackSquare => (32, 32, 3),
            Self::WhiteAligned => (48, 48, 3),
            Self::RedSmall => (16, 16, 3),
            Self::GradientNonmultiple => (197, 263, 3),
            Self::CheckerAboveBudget => (1024, 1536, 3),
            Self::BudgetAligned => (672, 960, 3),
            Self::ExtremeLandscape => (1, 100_000, 3),
            Self::RgbaNonmultiple => (101, 157, 4),
            Self::GraySmall => (45, 61, 1),
        }
    }
}

struct Expected {
    source_sha256: &'static str,
    converted_hwc_sha256: Option<&'static str>,
    resized_hw: (usize, usize),
    grid_hw: [usize; 2],
    valid_count: usize,
    soft_rows: usize,
    resized_chw_sha256: Option<&'static str>,
    padded_patch_sha256: &'static str,
}

fn synthetic_source(recipe: Recipe) -> Vec<u8> {
    let (height, width, channels) = recipe.source_geometry();
    let mut source = Vec::with_capacity(height * width * channels);
    for y in 0..height {
        for x in 0..width {
            match recipe {
                Recipe::BlackSquare => source.extend_from_slice(&[0, 0, 0]),
                Recipe::WhiteAligned => source.extend_from_slice(&[255, 255, 255]),
                Recipe::RedSmall => source.extend_from_slice(&[255, 0, 0]),
                Recipe::GradientNonmultiple | Recipe::RgbaNonmultiple => {
                    source.extend_from_slice(&[
                        (255 * x / (width - 1)) as u8,
                        (255 * y / (height - 1)) as u8,
                        (255 * (x + y) / (width + height - 2)) as u8,
                    ]);
                    if matches!(recipe, Recipe::RgbaNonmultiple) {
                        source.push(((3 * x + 5 * y) % 256) as u8);
                    }
                }
                Recipe::CheckerAboveBudget | Recipe::BudgetAligned => {
                    let tile = if matches!(recipe, Recipe::CheckerAboveBudget) {
                        7
                    } else {
                        16
                    };
                    let value = (255 * ((x / tile + y / tile) % 2)) as u8;
                    source.extend_from_slice(&[value, value, value]);
                }
                Recipe::ExtremeLandscape => {
                    let value = (255 * x / (width - 1)) as u8;
                    source.extend_from_slice(&[value, value, value]);
                }
                Recipe::GraySmall => {
                    source.push((255 * (x + 2 * y) / ((width - 1) + 2 * (height - 1))) as u8);
                }
            }
        }
    }
    source
}

fn logical_chw_sha256(image: &EmbeddingGemma2Rgb8Image) -> String {
    let mut digest = Sha256::new();
    let mut plane = Vec::with_capacity(image.width() * image.height());
    for channel in 0..3 {
        plane.clear();
        for pixel in image.data().chunks_exact(3) {
            plane.push(pixel[channel]);
        }
        digest.update(&plane);
    }
    format!("{:x}", digest.finalize())
}

fn padded_patch_sha256(prepared: &PreparedImage, valid_count: usize) -> String {
    let mut digest = Sha256::new();
    let mut row_bytes = [0_u8; 768 * 4];
    for row in prepared.patch_values().chunks_exact(768) {
        for (value, bytes) in row.iter().zip(row_bytes.chunks_exact_mut(4)) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        digest.update(row_bytes);
    }
    // The reference tensor is padded; production patch storage must stay unpadded.
    let zero_row = [0_u8; 768 * 4];
    for _ in valid_count..2520 {
        digest.update(zero_row);
    }
    format!("{:x}", digest.finalize())
}

fn assert_primary_processor(recipe: Recipe, expected: Expected) {
    let (source_height, source_width, channels) = recipe.source_geometry();
    let source = synthetic_source(recipe);
    assert_eq!(source.len(), source_height * source_width * channels);
    // Bind the original RGB, RGBA or L bytes before the real mode conversion.
    assert_eq!(
        format!("{:x}", Sha256::digest(&source)),
        expected.source_sha256,
        "{recipe:?}: original source bytes"
    );
    let image = match recipe {
        Recipe::RgbaNonmultiple => {
            EmbeddingGemma2Rgb8Image::from_rgba8(source_width, source_height, source)
        }
        Recipe::GraySmall => {
            EmbeddingGemma2Rgb8Image::from_grayscale8(source_width, source_height, source)
        }
        _ => EmbeddingGemma2Rgb8Image::new(source_width, source_height, source),
    }
    .unwrap();
    assert_eq!(
        (image.height(), image.width()),
        (source_height, source_width)
    );
    assert_eq!(image.data().len(), source_height * source_width * 3);
    let converted_hwc_sha256 = format!("{:x}", Sha256::digest(image.data()));
    if let Some(observed_sha256) = expected.converted_hwc_sha256 {
        assert_eq!(
            converted_hwc_sha256, observed_sha256,
            "{recipe:?}: converted RGB HWC bytes before resizing"
        );
    } else {
        // No resize entry was captured for this source; its raw RGB digest is the control.
        assert!(matches!(recipe, Recipe::BudgetAligned));
        assert_eq!(converted_hwc_sha256, expected.source_sha256);
    }

    let resized_hw = aspect_ratio_preserving_size(image.height(), image.width()).unwrap();
    assert_eq!(resized_hw, expected.resized_hw, "{recipe:?}: aspect sizing");
    let resized = resize::resize_rgb8_bicubic_aa(&image, resized_hw.1, resized_hw.0).unwrap();
    assert_eq!((resized.height(), resized.width()), expected.resized_hw);
    assert_eq!(
        resized.data().len(),
        expected.resized_hw.0 * expected.resized_hw.1 * 3
    );
    if let Some(resized_chw_sha256) = expected.resized_chw_sha256 {
        // The probe hashes contiguous logical [3,H,W], not the HWC storage bytes.
        assert_eq!(
            logical_chw_sha256(&resized),
            resized_chw_sha256,
            "{recipe:?}: resized CHW uint8 bytes"
        );
    } else {
        // This budget-aligned input bypasses resizing in the primary processor.
        assert!(matches!(recipe, Recipe::BudgetAligned));
        assert_eq!(resized_hw, (source_height, source_width));
        assert_eq!(
            resized.data(),
            image.data(),
            "{recipe:?}: exact bypass bytes"
        );
    }

    let prepared = prepare_image(&image).unwrap();
    assert_eq!(prepared.grid(), expected.grid_hw, "{recipe:?}: patch grid");
    assert_eq!(
        prepared.valid_count(),
        expected.valid_count,
        "{recipe:?}: valid patches"
    );
    assert_eq!(
        prepared.soft_rows(),
        expected.soft_rows,
        "{recipe:?}: soft rows"
    );
    assert_eq!(prepared.reference_capacity(), 2520);
    assert_eq!(prepared.patch_values().len(), expected.valid_count * 768);
    assert!(
        prepared
            .patch_values()
            .iter()
            .all(|value| value.is_finite())
    );
    assert_eq!(prepared.positions().len(), expected.valid_count);
    for (row, position) in prepared.positions().iter().enumerate() {
        assert_eq!(
            *position,
            [row % expected.grid_hw[1], row / expected.grid_hw[1]],
            "{recipe:?}: raster (x,y) position at patch {row}"
        );
    }
    assert_eq!(
        prepared.processor_identity().rule(),
        PreparationRuleVersion::AspectPreserving48PatchBudget2520V1
    );
    assert_eq!(
        prepared.processor_identity().kernel(),
        ResizeKernelVersion::TorchCpuUint8BicubicAaV1
    );
    assert_eq!(
        padded_patch_sha256(&prepared, expected.valid_count),
        expected.padded_patch_sha256,
        "{recipe:?}: little-endian float32 patches with reference-only zero padding"
    );
}

#[test]
fn processor_reference_black_square() {
    assert_primary_processor(
        Recipe::BlackSquare,
        Expected {
            source_sha256: "e80232b4d18d0bb7e794be263ba937626f383f9917d4b8a737ba893a8f752293",
            converted_hwc_sha256: Some(
                "e80232b4d18d0bb7e794be263ba937626f383f9917d4b8a737ba893a8f752293",
            ),
            resized_hw: (768, 768),
            grid_hw: [48, 48],
            valid_count: 2304,
            soft_rows: 256,
            resized_chw_sha256: Some(
                "f0a0e8d753bbee26646b9cedd3cad510258b922274c4085b8da7d4f08e4cca3f",
            ),
            padded_patch_sha256: "4845f5b0aeaf0473275506754fac525ec7a4f1d788f9cd7a0604f4a51d744cf8",
        },
    );
}

#[test]
fn processor_reference_white_aligned() {
    assert_primary_processor(
        Recipe::WhiteAligned,
        Expected {
            source_sha256: "82eec5ad244bc79d0380bffcca90f9121334b8494ad52e593baff095a42b9583",
            converted_hwc_sha256: Some(
                "82eec5ad244bc79d0380bffcca90f9121334b8494ad52e593baff095a42b9583",
            ),
            resized_hw: (768, 768),
            grid_hw: [48, 48],
            valid_count: 2304,
            soft_rows: 256,
            resized_chw_sha256: Some(
                "c2d08aee47c53dc84cbe71d8d96a0a7ad13eca4c7433219318094f893d2008aa",
            ),
            padded_patch_sha256: "f868d0aba027d3d1051db8e5b84c5213ffdc5e11e09b4c0196d739b679db50f0",
        },
    );
}

#[test]
fn processor_reference_red_small() {
    assert_primary_processor(
        Recipe::RedSmall,
        Expected {
            source_sha256: "fa8dc828bdf11c88a2f25d1ffc454a0dce698bd7ebe1338bd1402966b43b50c1",
            converted_hwc_sha256: Some(
                "fa8dc828bdf11c88a2f25d1ffc454a0dce698bd7ebe1338bd1402966b43b50c1",
            ),
            resized_hw: (768, 768),
            grid_hw: [48, 48],
            valid_count: 2304,
            soft_rows: 256,
            resized_chw_sha256: Some(
                "b753e5fa82de48e940df4af0eb7e9e4c377d917fec7108c4ae50a39f456cb6fb",
            ),
            padded_patch_sha256: "d759465d297c13d45ea6dd542e7ba0d57d857da57798c2b26896fc27e7a95a03",
        },
    );
}

#[test]
fn processor_reference_gradient_nonmultiple() {
    assert_primary_processor(
        Recipe::GradientNonmultiple,
        Expected {
            source_sha256: "f3beab3fe9f51c1daed59c6ea2ad4552fa6da9f650dc04155d7096303650b6dc",
            converted_hwc_sha256: Some(
                "f3beab3fe9f51c1daed59c6ea2ad4552fa6da9f650dc04155d7096303650b6dc",
            ),
            resized_hw: (672, 912),
            grid_hw: [42, 57],
            valid_count: 2394,
            soft_rows: 266,
            resized_chw_sha256: Some(
                "1f94b4fc8ab9756e41bc056d9d0ec74ecdb244d545014810c0c8b03124a8ace7",
            ),
            padded_patch_sha256: "1bd710b1c911fc691d17951c378145e17ead8b7dbd1c4c0d7155afc1c2f46427",
        },
    );
}

#[test]
fn processor_reference_checker_above_budget() {
    assert_primary_processor(
        Recipe::CheckerAboveBudget,
        Expected {
            source_sha256: "0ac8ce6fa28f656f4a2cec83e5a88843d0729b9bb33368a1bf7701d7d9181214",
            converted_hwc_sha256: Some(
                "0ac8ce6fa28f656f4a2cec83e5a88843d0729b9bb33368a1bf7701d7d9181214",
            ),
            resized_hw: (624, 960),
            grid_hw: [39, 60],
            valid_count: 2340,
            soft_rows: 260,
            resized_chw_sha256: Some(
                "ae283f344b71ee1ba80422ac7022c9718dd9da3c870369fa9bfbe5328312de67",
            ),
            padded_patch_sha256: "3f1f1329e79eb55c088cd36b41303ff11caadf566d56a04e9a47755f788c0d50",
        },
    );
}

#[test]
fn processor_reference_budget_aligned() {
    assert_primary_processor(
        Recipe::BudgetAligned,
        Expected {
            source_sha256: "16da6a3abbb5593c3f2c8f8e228ec7a87f4e5b9222e8a1f6d91e960728b4a949",
            converted_hwc_sha256: None,
            resized_hw: (672, 960),
            grid_hw: [42, 60],
            valid_count: 2520,
            soft_rows: 280,
            resized_chw_sha256: None,
            padded_patch_sha256: "625c03c1726eae74caee39bfb24ef79a8a6017dedf9a5bc54034e1d1c0049191",
        },
    );
}

#[test]
fn processor_reference_extreme_landscape() {
    assert_primary_processor(
        Recipe::ExtremeLandscape,
        Expected {
            source_sha256: "e8e22af6fdb917238d1a16434690ef4377f445818f7bb4c7246a98e513cda90d",
            converted_hwc_sha256: Some(
                "e8e22af6fdb917238d1a16434690ef4377f445818f7bb4c7246a98e513cda90d",
            ),
            resized_hw: (48, 13_440),
            grid_hw: [3, 840],
            valid_count: 2520,
            soft_rows: 280,
            resized_chw_sha256: Some(
                "82a35df61df3f8e92dd952abeb2c5fd2b7dc6f77313458eadb5e57a77a19cd1e",
            ),
            padded_patch_sha256: "ec76a7f486f6ecb1431e00118098b17b2f463c2f5dbaeaa149743fcafc833ba8",
        },
    );
}

#[test]
fn processor_reference_rgba_nonmultiple() {
    assert_primary_processor(
        Recipe::RgbaNonmultiple,
        Expected {
            source_sha256: "f7d069072ba7f7e0b26f65c3adaa3512c344d142dba14b89989e30bb23fb2fe5",
            converted_hwc_sha256: Some(
                "e85d65ee77f372f49b58c060563f7ef4f2aeccb46d0b1e7ae04ae72c49959e9f",
            ),
            resized_hw: (624, 960),
            grid_hw: [39, 60],
            valid_count: 2340,
            soft_rows: 260,
            resized_chw_sha256: Some(
                "ec8995a3fd76ffe8beb9830b62c24bd9c165f6e965144eb141189a928cdd221b",
            ),
            padded_patch_sha256: "46ed36c9b6199be624c7c611e347d6f6aa6d1596f85c9c14146785c799c25966",
        },
    );
}

#[test]
fn processor_reference_gray_small() {
    assert_primary_processor(
        Recipe::GraySmall,
        Expected {
            source_sha256: "552a43fa02b6fa5b154d7ba3e1b3dc74079f0bc4ff33e0f5339824aec68259b1",
            converted_hwc_sha256: Some(
                "94f3f09c9c25555ce7d6c3c71b3ca19a24fba62e1594a6051be0f37fd12a3b68",
            ),
            resized_hw: (672, 912),
            grid_hw: [42, 57],
            valid_count: 2394,
            soft_rows: 266,
            resized_chw_sha256: Some(
                "4d9a9a4cb7f13ef0f502eee23504b5d9f18d2c6833c7dda04361625f2c66c179",
            ),
            padded_patch_sha256: "d5081c860aeb6656e39b7c30b6874ac7304b7db21f62906fc72459f4aa0773f8",
        },
    );
}

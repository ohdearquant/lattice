use super::*;

const PACKED: [u8; 8] = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
const VALUES: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn reference_scale(code: u8) -> f64 {
    let exponent = (code >> 3) & 15;
    let mantissa = code & 7;
    if exponent == 0 {
        f64::from(mantissa) * 2.0_f64.powi(-9)
    } else {
        (1.0 + f64::from(mantissa) / 8.0) * 2.0_f64.powi(i32::from(exponent) - 7)
    }
}

#[test]
fn all_nibbles_and_valid_scales_match_independent_reference() {
    for scale in 0..=0x7f {
        for global in [0.0, 0.25, 1.0, 16.0, f32::MAX] {
            let mut output = [42.0; NVFP4_BLOCK_LEN];
            decode_nvfp4_block(&PACKED, scale, global, &mut output).unwrap();
            if scale == 0x7f {
                assert!(output.iter().all(|v| v.is_nan()));
                continue;
            }
            for (index, &actual) in output.iter().enumerate() {
                let expected = (VALUES[index] * reference_scale(scale) * f64::from(global)) as f32;
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "scale={scale:#x}, global={global}, index={index}"
                );
            }
        }
    }
}

#[test]
fn scale_specials_and_global_multiplier_are_explicit() {
    assert_eq!(reference_scale(0x01), 2.0_f64.powi(-9));
    assert_eq!(reference_scale(0x38), 1.0);
    assert_eq!(reference_scale(0x7e), 448.0);
    let mut output = [42.0; NVFP4_BLOCK_LEN];
    decode_nvfp4_block(&PACKED, 0x38, 2.0, &mut output).unwrap();
    assert_eq!(output[7], 12.0);
    assert_eq!(output[8].to_bits(), (-0.0_f32).to_bits());
    decode_nvfp4_block(&PACKED, 0x7e, f32::MAX, &mut output).unwrap();
    assert!(output[7].is_infinite());
    assert!(output[15].is_infinite() && output[15].is_sign_negative());
}

#[test]
fn row_major_blocks_use_their_scales_and_global_multiplier() {
    let packed = [0x21; NVFP4_BLOCK_BYTES * 4];
    let scales = [0x38, 0x40, 0x48, 0x50];
    let mut output = [0.0; NVFP4_BLOCK_LEN * 4];
    dequantize_nvfp4(2, 32, &packed, &scales, 0.5, &mut output).unwrap();
    for (block, factor) in output
        .chunks_exact(NVFP4_BLOCK_LEN)
        .zip([0.5, 1.0, 2.0, 4.0])
    {
        for pair in block.chunks_exact(2) {
            assert_eq!(pair, [0.5 * factor, factor]);
        }
    }
}

#[test]
fn invalid_block_scale_or_global_never_changes_output() {
    for scale in 0x80..=0xff {
        let mut output = [17.0; NVFP4_BLOCK_LEN];
        assert!(decode_nvfp4_block(&PACKED, scale, 1.0, &mut output).is_err());
        assert_eq!(output, [17.0; NVFP4_BLOCK_LEN]);
    }
    for global in [-1.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut output = [17.0; NVFP4_BLOCK_LEN];
        assert!(decode_nvfp4_block(&PACKED, 0x38, global, &mut output).is_err());
        assert_eq!(output, [17.0; NVFP4_BLOCK_LEN]);
    }
}

#[test]
fn malformed_matrix_fails_before_any_write() {
    let packed = [0x21; NVFP4_BLOCK_BYTES * 2];
    let mut output = [17.0; NVFP4_BLOCK_LEN * 2];
    assert!(dequantize_nvfp4(1, 31, &packed, &[0x38, 0x38], 1.0, &mut output).is_err());
    assert!(dequantize_nvfp4(1, 32, &packed[..15], &[0x38, 0x38], 1.0, &mut output).is_err());
    assert!(dequantize_nvfp4(1, 32, &packed, &[0x38], 1.0, &mut output).is_err());
    assert!(dequantize_nvfp4(1, 32, &packed, &[0x38, 0x80], 1.0, &mut output).is_err());
    assert!(dequantize_nvfp4(1, 32, &packed, &[0x38, 0x38], 1.0, &mut output[..31]).is_err());
    assert!(dequantize_nvfp4(usize::MAX, 16, &packed, &[0x38, 0x38], 1.0, &mut output).is_err());
    assert_eq!(output, [17.0; NVFP4_BLOCK_LEN * 2]);
}

#[test]
fn empty_dimensions_require_empty_buffers() {
    assert!(dequantize_nvfp4(0, 16, &[], &[], 1.0, &mut []).is_ok());
    assert!(dequantize_nvfp4(3, 0, &[], &[], 1.0, &mut []).is_ok());
    assert!(dequantize_nvfp4(0, 16, &[0], &[], 1.0, &mut []).is_err());
}

#[cfg(feature = "f16")]
#[test]
fn ue4m3_matches_existing_e4m3_decoder_for_positive_encodings() {
    for code in 0..=0x7f {
        let actual = ue4m3(code);
        let existing = crate::half_bits::f8_e4m3_bits_to_f32(code);
        if code == 0x7f {
            assert!(actual.is_nan() && existing.is_nan());
        } else {
            assert_eq!(actual.to_bits(), existing.to_bits(), "code={code:#x}");
        }
    }
}

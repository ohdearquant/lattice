use super::*;

#[test]
fn block_decodes_every_nibble_in_low_first_order() {
    let packed = [
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc,
        0xfe,
    ];
    let expected: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let mut output = [42.0; MXFP4_BLOCK_LEN];
    decode_mxfp4_block(&packed, 127, &mut output);
    for (i, &actual) in output.iter().enumerate() {
        assert_eq!(actual.to_bits(), expected[i % 16].to_bits(), "element {i}");
    }
}

#[test]
fn all_scale_codes_match_independent_f64_reference() {
    let packed = [
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc,
        0xfe,
    ];
    let values: [f64; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    for scale in 0..=u8::MAX {
        let mut output = [42.0; MXFP4_BLOCK_LEN];
        decode_mxfp4_block(&packed, scale, &mut output);
        if scale == 255 {
            assert!(output.iter().all(|x| x.is_nan()));
            continue;
        }
        let factor = 2.0_f64.powi(i32::from(scale) - 127);
        for (index, &actual) in output.iter().enumerate() {
            let expected = (values[index % 16] * factor) as f32;
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "scale={scale} element={index}"
            );
        }
    }
}

#[test]
fn matrix_scales_and_blocks_follow_row_major_order() {
    let mut packed = vec![0x21; MXFP4_BLOCK_BYTES];
    packed.extend([0x87; MXFP4_BLOCK_BYTES]);
    let mut output = vec![99.0; 64];
    dequantize_mxfp4(2, 32, &packed, &[127, 128], &mut output).unwrap();
    for pair in output[..32].chunks_exact(2) {
        assert_eq!(pair, [0.5, 1.0]);
    }
    for pair in output[32..].chunks_exact(2) {
        assert_eq!(pair[0], 12.0);
        assert_eq!(pair[1].to_bits(), (-0.0_f32).to_bits());
    }
}

#[test]
fn malformed_shapes_fail_before_any_output_mutation() {
    let packed = [0x21; MXFP4_BLOCK_BYTES];
    let scales = [127];
    let mut output = [17.0; MXFP4_BLOCK_LEN];
    assert!(dequantize_mxfp4(1, 31, &packed, &scales, &mut output).is_err());
    assert!(dequantize_mxfp4(1, 32, &packed[..15], &scales, &mut output).is_err());
    assert!(dequantize_mxfp4(1, 32, &packed, &[], &mut output).is_err());
    assert!(dequantize_mxfp4(1, 32, &packed, &scales, &mut output[..31]).is_err());
    assert!(dequantize_mxfp4(usize::MAX, 32, &packed, &scales, &mut output).is_err());
    assert_eq!(output, [17.0; MXFP4_BLOCK_LEN]);
}

#[test]
fn empty_dimensions_require_empty_buffers() {
    assert!(dequantize_mxfp4(0, 32, &[], &[], &mut []).is_ok());
    assert!(dequantize_mxfp4(3, 0, &[], &[], &mut []).is_ok());
    assert!(dequantize_mxfp4(0, 32, &[0], &[], &mut []).is_err());
}

#[test]
fn two_blocks_per_row_use_their_own_scales() {
    let packed = [0x11; MXFP4_BLOCK_BYTES * 4];
    let scales = [127, 128, 129, 130];
    let mut output = [0.0; MXFP4_BLOCK_LEN * 4];
    dequantize_mxfp4(2, 64, &packed, &scales, &mut output).unwrap();
    for (block, expected) in output
        .chunks_exact(MXFP4_BLOCK_LEN)
        .zip([0.5, 1.0, 2.0, 4.0])
    {
        assert!(block.iter().all(|&value| value == expected));
    }
}

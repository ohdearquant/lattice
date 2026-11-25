use super::*;

fn reference_e4m3(code: u8) -> Option<f64> {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = (code >> 3) & 0x0f;
    let mantissa = code & 7;
    if exponent == 0x0f && mantissa == 7 {
        return None;
    }
    let magnitude = if exponent == 0 {
        f64::from(mantissa) * 2.0_f64.powi(-9)
    } else {
        (1.0 + f64::from(mantissa) / 8.0) * 2.0_f64.powi(i32::from(exponent) - 7)
    };
    Some(sign * magnitude)
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state >> 33
}

#[test]
fn every_element_and_scale_code_matches_independent_reference() {
    let mut elements = [0u8; MXFP8_BLOCK_LEN];
    for scale in 0..=u8::MAX {
        for start in (0..=255usize).step_by(MXFP8_BLOCK_LEN) {
            for (index, element) in elements.iter_mut().enumerate() {
                *element = (start + index) as u8;
            }
            let mut output = [42.0; MXFP8_BLOCK_LEN];
            decode_mxfp8_block(&elements, scale, &mut output);
            for (index, &actual) in output.iter().enumerate() {
                let code = elements[index];
                match (scale, reference_e4m3(code)) {
                    (255, _) | (_, None) => {
                        assert!(actual.is_nan(), "scale={scale} code={code:#04x}");
                    }
                    (_, Some(value)) => {
                        let expected = (value * 2.0_f64.powi(i32::from(scale) - 127)) as f32;
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "scale={scale} code={code:#04x}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn matrix_blocks_use_their_own_scales_in_row_major_order() {
    let elements = [0x38; MXFP8_BLOCK_LEN * 4];
    let scales = [127, 128, 129, 126];
    let mut output = [0.0; MXFP8_BLOCK_LEN * 4];
    dequantize_mxfp8(2, 64, &elements, &scales, &mut output).unwrap();
    for (block, expected) in output
        .chunks_exact(MXFP8_BLOCK_LEN)
        .zip([1.0, 2.0, 4.0, 0.5])
    {
        assert!(block.iter().all(|&value| value == expected));
    }
}

#[test]
fn malformed_shapes_fail_before_any_output_mutation() {
    let elements = [0x38; MXFP8_BLOCK_LEN];
    let scales = [127];
    let mut output = [17.0; MXFP8_BLOCK_LEN];
    assert!(dequantize_mxfp8(1, 31, &elements, &scales, &mut output).is_err());
    assert!(dequantize_mxfp8(1, 32, &elements[..31], &scales, &mut output).is_err());
    assert!(dequantize_mxfp8(1, 32, &elements, &[], &mut output).is_err());
    assert!(dequantize_mxfp8(1, 32, &elements, &scales, &mut output[..31]).is_err());
    assert!(dequantize_mxfp8(usize::MAX, 32, &elements, &scales, &mut output).is_err());
    assert_eq!(output, [17.0; MXFP8_BLOCK_LEN]);
}

#[test]
fn empty_dimensions_require_empty_buffers() {
    assert!(dequantize_mxfp8(0, 32, &[], &[], &mut []).is_ok());
    assert!(dequantize_mxfp8(3, 0, &[], &[], &mut []).is_ok());
    assert!(dequantize_mxfp8(0, 32, &[0], &[], &mut []).is_err());
}

#[test]
fn matvec_matches_dequantize_then_sequential_dot_bit_for_bit() {
    let (rows, cols) = (5, 96);
    let mut state = 0x9e37_79b9_u64;
    let elements: Vec<u8> = (0..rows * cols)
        .map(|_| match lcg(&mut state) as u8 {
            0x7f | 0xff => 0,
            code => code,
        })
        .collect();
    let scales: Vec<u8> = (0..rows * cols / MXFP8_BLOCK_LEN)
        .map(|_| 120 + (lcg(&mut state) % 16) as u8)
        .collect();
    let x: Vec<f32> = (0..cols)
        .map(|_| (lcg(&mut state) % 2001) as f32 / 1000.0 - 1.0)
        .collect();
    let mut dense = vec![0.0; rows * cols];
    dequantize_mxfp8(rows, cols, &elements, &scales, &mut dense).unwrap();
    let mut out = vec![f32::NAN; rows];
    matvec_mxfp8(rows, cols, &elements, &scales, &x, &mut out).unwrap();
    for row in 0..rows {
        let mut expected = 0.0f32;
        for col in 0..cols {
            expected += dense[row * cols + col] * x[col];
        }
        assert_eq!(out[row].to_bits(), expected.to_bits(), "row {row}");
    }
    assert!(out.iter().all(|value| value.is_finite() && *value != 0.0));
}

#[test]
fn matvec_rejects_mismatched_inputs_before_writing() {
    let elements = [0x38; MXFP8_BLOCK_LEN];
    let x = [1.0; MXFP8_BLOCK_LEN];
    let mut out = [7.0; 1];
    assert!(matvec_mxfp8(1, 32, &elements, &[127], &x[..31], &mut out).is_err());
    assert!(matvec_mxfp8(1, 32, &elements, &[127], &x, &mut []).is_err());
    assert!(matvec_mxfp8(1, 32, &elements, &[], &x, &mut out).is_err());
    assert!(matvec_mxfp8(1, 31, &elements[..31], &[127], &x[..31], &mut out).is_err());
    assert_eq!(out, [7.0]);
    let mut zero_cols = [7.0; 3];
    matvec_mxfp8(3, 0, &[], &[], &[], &mut zero_cols).unwrap();
    assert_eq!(zero_cols, [0.0; 3]);
}

use super::*;

#[test]
fn accepts_finite_values_including_signed_zero_and_subnormal() {
    let values = [0.0f32, -0.0, 1.0, -1.0, f32::MIN_POSITIVE / 2.0];
    reset_decoded_f32_finite_check_count();
    let tensor = IngestedTensor::decoded_f32("test", "t", &[5], "F32", &values);
    assert!(validate_ingested_tensor(tensor).is_ok());
    let checks = decoded_f32_finite_check_count();
    assert_eq!(checks, values.len());
    assert_ne!(checks, 0, "the test scan observable must collect checks");
}

#[test]
fn rejects_nan() {
    let values = [1.0f32, f32::NAN, 3.0];
    let tensor = IngestedTensor::decoded_f32("test", "t", &[3], "F32", &values);
    let err = validate_ingested_tensor(tensor).expect_err("NaN must be rejected");
    let msg = err.to_string();
    assert!(msg.contains('t'), "error should name the tensor: {msg}");
    assert!(
        msg.contains("element index 1"),
        "error should point at the offending index: {msg}"
    );
}

#[test]
fn rejects_positive_infinity() {
    let values = [1.0f32, f32::INFINITY];
    let tensor = IngestedTensor::decoded_f32("test", "t", &[2], "F16", &values);
    let err = validate_ingested_tensor(tensor).expect_err("+inf must be rejected");
    assert!(err.to_string().contains("element index 1"));
}

#[test]
fn rejects_negative_infinity() {
    let values = [f32::NEG_INFINITY, 2.0];
    let tensor = IngestedTensor::decoded_f32("test", "t", &[2], "BF16", &values);
    let err = validate_ingested_tensor(tensor).expect_err("-inf must be rejected");
    assert!(err.to_string().contains("element index 0"));
}

#[test]
fn rejects_shape_product_overflow() {
    let values: [f32; 0] = [];
    let huge_shape = [usize::MAX, 2];
    let tensor = IngestedTensor::decoded_f32("test", "t", &huge_shape, "F32", &values);
    let err = validate_ingested_tensor(tensor).expect_err("overflow must be rejected");
    assert!(err.to_string().contains("overflows"));
}

#[test]
fn rejects_element_count_mismatch() {
    let values = [1.0f32, 2.0, 3.0];
    // Shape says 4 elements, payload only has 3.
    let tensor = IngestedTensor::decoded_f32("test", "t", &[4], "F32", &values);
    let err = validate_ingested_tensor(tensor).expect_err("count mismatch must be rejected");
    assert!(err.to_string().contains("does not match"));
}

#[test]
fn q8_rejects_non_finite_source_with_tensor_attribution() {
    let source_values = [1.0f32, 2.0, f32::NAN, 4.0];
    let tensor = IngestedTensor::q8_source(
        "in-memory Q8 quantization",
        "q_proj",
        &[2, 2],
        &source_values,
    );

    let err = validate_ingested_tensor(tensor).expect_err("NaN must be rejected");
    match err {
        InferenceError::InvalidInput(msg) => {
            assert!(
                msg.contains("q_proj"),
                "error should name the tensor: {msg}"
            );
            assert!(
                msg.contains("element index 2"),
                "error should point at the offending source value: {msg}"
            );
        }
        other => panic!("expected InvalidInput, got: {other}"),
    }
}

#[test]
fn q8_rejects_quantized_geometry_mismatch() {
    let data = [1i8, 2, 3];
    let scales = [0.5f32, 1.0];
    let tensor = IngestedTensor::q8(
        "in-memory Q8 quantization",
        "q_proj",
        &[2, 2],
        &data,
        &scales,
    );

    let err = validate_ingested_tensor(tensor).expect_err("short Q8 data must be rejected");
    assert!(err.to_string().contains("data element count 3"));

    let data = [1i8, 2, 3, 4];
    let scales = [0.5f32];
    let tensor = IngestedTensor::q8(
        "in-memory Q8 quantization",
        "q_proj",
        &[2, 2],
        &data,
        &scales,
    );
    let err = validate_ingested_tensor(tensor).expect_err("short scales must be rejected");
    assert!(err.to_string().contains("scale count 1"));
}

#[test]
fn q8_rejects_non_positive_scale() {
    let data = [1i8, 2, 3, 4];

    for bad in [0.0f32, -1.0] {
        let scales = [0.5f32, bad];
        let tensor = IngestedTensor::q8(
            "in-memory Q8 quantization",
            "q_proj",
            &[2, 2],
            &data,
            &scales,
        );

        let err =
            validate_ingested_tensor(tensor).expect_err("non-positive Q8 scale must be rejected");
        assert!(
            err.to_string().contains("non-positive scale"),
            "unexpected error for scale {bad}: {err}"
        );
    }
}

#[test]
fn q8_rejects_non_finite_scale() {
    let data = [1i8, 2, 3, 4];
    let scales = [0.5f32, f32::INFINITY];
    let tensor = IngestedTensor::q8(
        "in-memory Q8 quantization",
        "q_proj",
        &[2, 2],
        &data,
        &scales,
    );

    let err = validate_ingested_tensor(tensor).expect_err("non-finite Q8 scale must be rejected");
    assert!(err.to_string().contains("non-finite scale"));
}

/// Mutation sensitivity: with the `!v.is_finite()` scan removed (or
/// replaced by an always-`Ok` stub), this test's NaN input would pass
/// validation. Verified by temporarily deleting the scan and confirming
/// `rejects_nan` above goes red before restoring it (see PR description
/// for the revert/touch/restore cycle).
#[test]
fn all_of_a_finite_tensor_is_scanned_not_just_the_first_element() {
    let mut values = vec![1.0f32; 64];
    values[63] = f32::NAN;
    let tensor = IngestedTensor::decoded_f32("test", "t", &[64], "F32", &values);
    let err = validate_ingested_tensor(tensor).expect_err("tail NaN must be caught");
    assert!(err.to_string().contains("element index 63"));
}

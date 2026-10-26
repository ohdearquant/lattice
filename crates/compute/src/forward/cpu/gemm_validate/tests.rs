use super::*;

#[test]
fn validate_gemm_nn_accepts_oversized_buffers() {
    // m=1, k=2, n=2: exact footprints are a=2, b=4, c=2. Pass every buffer oversized.
    validate_gemm_nn(3, 5, 4, 1, 2, 2, "test");
}

#[test]
#[should_panic(expected = "a too short for m*k")]
fn validate_gemm_nn_rejects_short_a() {
    validate_gemm_nn(1, 4, 2, 1, 2, 2, "test");
}

#[test]
#[should_panic(expected = "b too short for k*n")]
fn validate_gemm_nn_rejects_short_b() {
    validate_gemm_nn(2, 3, 2, 1, 2, 2, "test");
}

#[test]
#[should_panic(expected = "c too short for m*n")]
fn validate_gemm_nn_rejects_short_c() {
    validate_gemm_nn(2, 4, 1, 1, 2, 2, "test");
}

#[test]
#[should_panic(expected = "shape overflow: m*k")]
fn validate_gemm_nn_rejects_overflow() {
    validate_gemm_nn(2, 2, 2, 2, usize::MAX, 2, "test");
}

#[test]
fn validate_gemm_bt_accepts_oversized_buffers() {
    // m=1, k=2, n=2: exact footprints are a=2, b=4 (n*k), c=2.
    validate_gemm_bt(3, 5, 4, 1, 2, 2, "test");
}

#[test]
#[should_panic(expected = "b too short for n*k")]
fn validate_gemm_bt_rejects_short_b() {
    validate_gemm_bt(2, 3, 2, 1, 2, 2, "test");
}

#[test]
#[should_panic(expected = "shape overflow: n*k")]
fn validate_gemm_bt_rejects_overflow() {
    validate_gemm_bt(2, 2, 2, 2, 2, usize::MAX, "test");
}

#[test]
fn validate_gemm_strided_shape_accepts_exact() {
    validate_gemm_strided_shape(1, 2, 2, 2, 2, 2, false, "test");
    validate_gemm_strided_shape(1, 2, 2, 2, 2, 2, true, "test");
}

#[test]
#[should_panic(expected = "ldb too small for row extent n")]
fn validate_gemm_strided_shape_rejects_short_ldb_nn() {
    validate_gemm_strided_shape(1, 2, 4, 2, 2, 4, false, "test");
}

#[test]
#[should_panic(expected = "ldb too small for row extent k")]
fn validate_gemm_strided_shape_rejects_short_ldb_bt() {
    validate_gemm_strided_shape(1, 4, 2, 4, 2, 2, true, "test");
}

#[test]
#[should_panic(expected = "shape overflow: n*k")]
fn validate_gemm_strided_shape_rejects_overflow() {
    validate_gemm_strided_shape(2, 2, usize::MAX, 2, 2, 2, true, "test");
}

#[test]
fn validate_ternary_matvec_args_accepts_oversized() {
    validate_ternary_matvec_args(5, 4, 20, 4, 3, 4, 1, "test");
}

#[test]
#[should_panic(expected = "x_q too short for k")]
fn validate_ternary_matvec_args_rejects_short_x_q() {
    validate_ternary_matvec_args(3, 4, 20, 4, 3, 4, 1, "test");
}

#[test]
#[should_panic(expected = "alphas too short for n")]
fn validate_ternary_matvec_args_rejects_short_alphas() {
    validate_ternary_matvec_args(5, 2, 20, 4, 3, 4, 1, "test");
}

#[test]
#[should_panic(expected = "packed_w too short for n*packed_row_bytes")]
fn validate_ternary_matvec_args_rejects_short_packed_w() {
    validate_ternary_matvec_args(5, 4, 2, 4, 3, 4, 1, "test");
}

#[test]
#[should_panic(expected = "output too short for n")]
fn validate_ternary_matvec_args_rejects_short_output() {
    validate_ternary_matvec_args(5, 4, 20, 2, 3, 4, 1, "test");
}

#[test]
#[should_panic(expected = "shape overflow: n*packed_row_bytes")]
fn validate_ternary_matvec_args_rejects_overflow() {
    validate_ternary_matvec_args(5, 4, 20, 4, usize::MAX, 4, usize::MAX, "test");
}

use super::*;

// --- sgemm_nn_ab: release-active argument validation (ADR-080 C4 held finding) ---

#[test]
#[should_panic(expected = "a too short for m*k")]
fn sgemm_nn_ab_rejects_short_a() {
    let a = [0.0f32; 1]; // needs m*k = 2
    let b = [0.0f32; 2];
    let mut c = [0.0f32; 1];
    sgemm_nn_ab(&a, &b, &mut c, 1, 1, 2, 1.0, 0.0);
}

#[test]
#[should_panic(expected = "b too short for k*n")]
fn sgemm_nn_ab_rejects_short_b() {
    let a = [0.0f32; 2];
    let b = [0.0f32; 1]; // needs k*n = 2
    let mut c = [0.0f32; 1];
    sgemm_nn_ab(&a, &b, &mut c, 1, 2, 1, 1.0, 0.0);
}

#[test]
#[should_panic(expected = "shape overflow: m*k")]
fn sgemm_nn_ab_rejects_overflow() {
    let a = [0.0f32; 2];
    let b = [0.0f32; 2];
    let mut c = [0.0f32; 2];
    sgemm_nn_ab(&a, &b, &mut c, usize::MAX, 2, 2, 1.0, 0.0);
}

#[test]
fn sgemm_nn_ab_accepts_oversized_buffers_and_computes_correctly() {
    // m=1, k=2, n=2: C = alpha * A@B + beta*C. A=[1,2], B=[[1,0],[0,1]] (row-major k x n).
    let a = [1.0f32, 2.0, 99.0]; // oversized by 1
    let b = [1.0f32, 0.0, 0.0, 1.0, 99.0]; // oversized by 1
    let mut c = [0.0f32, 0.0, 99.0]; // oversized by 1
    sgemm_nn_ab(&a, &b, &mut c, 1, 2, 2, 1.0, 0.0);
    assert!((c[0] - 1.0).abs() < 1e-6);
    assert!((c[1] - 2.0).abs() < 1e-6);
}

// --- accelerate_matmul_bt / accelerate_matmul: overflow-first + oversized-allowed ---

#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "b too short for n*k")]
fn accelerate_matmul_bt_rejects_short_b() {
    let a = [0.0f32; 2];
    let b = [0.0f32; 1]; // needs n*k = 2
    let mut c = [0.0f32; 1];
    accelerate_matmul_bt(&a, &b, &mut c, 1, 1, 2);
}

#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "shape overflow: n*k")]
fn accelerate_matmul_bt_rejects_overflow() {
    let a = [0.0f32; 2];
    let b = [0.0f32; 2];
    let mut c = [0.0f32; 2];
    accelerate_matmul_bt(&a, &b, &mut c, 2, usize::MAX, 2);
}

#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "b too short for k*n")]
fn accelerate_matmul_rejects_short_b() {
    let a = [0.0f32; 2];
    let b = [0.0f32; 1]; // needs k*n = 2
    let mut c = [0.0f32; 1];
    accelerate_matmul(&a, &b, &mut c, 1, 1, 2);
}

// --- strided GEMM wrappers: shape/stride check before pointer dereference ---

#[test]
#[should_panic(expected = "ldb too small for row extent n")]
fn sgemm_nn_strided_rejects_short_ldb() {
    let a = [0.0f32; 4];
    let b = [0.0f32; 4];
    let mut c = [0.0f32; 4];
    // SAFETY: never reached — the validator panics before any pointer is dereferenced.
    unsafe {
        sgemm_nn_strided(a.as_ptr(), 2, b.as_ptr(), 1, c.as_mut_ptr(), 2, 1, 2, 2);
    }
}

#[test]
#[should_panic(expected = "ldb too small for row extent k")]
fn sgemm_bt_strided_rejects_short_ldb() {
    let a = [0.0f32; 4];
    let b = [0.0f32; 4];
    let mut c = [0.0f32; 4];
    // SAFETY: never reached — the validator panics before any pointer is dereferenced.
    unsafe {
        sgemm_bt_strided(a.as_ptr(), 2, b.as_ptr(), 1, c.as_mut_ptr(), 2, 1, 2, 2);
    }
}

// --- CBLAS i32 ABI guard (ADR-080 C4, #796 round 1 finding 2): a non-overflowing usize
// above i32::MAX passes every usize-level shape/length check (zero-extent shapes make
// every canonical product zero), so it must be caught by cblas_dim's i32::try_from
// conversion before reaching the FFI call, not the shape validator.

#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "n=2147483648 exceeds i32::MAX")]
fn accelerate_matmul_rejects_n_above_i32_max() {
    let huge_n = i32::MAX as usize + 1;
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let mut c: [f32; 0] = [];
    // m=0, k=0: every canonical product (m*k, k*n, m*n) is zero, so validate_gemm_nn's
    // usize-level checks all pass trivially. Only cblas_dim's i32 conversion should reject.
    accelerate_matmul(&a, &b, &mut c, 0, huge_n, 0);
}

#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "n=2147483648 exceeds i32::MAX")]
fn accelerate_matmul_bt_rejects_n_above_i32_max() {
    let huge_n = i32::MAX as usize + 1;
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let mut c: [f32; 0] = [];
    accelerate_matmul_bt(&a, &b, &mut c, 0, huge_n, 0);
}

#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "ldb=2147483648 exceeds i32::MAX")]
fn sgemm_nn_strided_rejects_ldb_above_i32_max() {
    let huge_ldb = i32::MAX as usize + 1;
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let mut c: [f32; 0] = [];
    // m=0, k=0, n=0: validate_gemm_strided_shape's product-overflow checks pass trivially,
    // and lda/ldb/ldc >= their row extents (0) also pass; only cblas_dim should reject.
    // SAFETY: never reached — cblas_dim panics before any pointer is dereferenced.
    unsafe {
        sgemm_nn_strided(
            a.as_ptr(),
            0,
            b.as_ptr(),
            huge_ldb,
            c.as_mut_ptr(),
            0,
            0,
            0,
            0,
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "m=2147483648 exceeds i32::MAX")]
fn sgemm_nn_ab_rejects_m_above_i32_max() {
    let huge_m = i32::MAX as usize + 1;
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let mut c: [f32; 0] = [];
    // n=0, k=0: every canonical product is zero, so validate_gemm_nn's usize-level checks
    // all pass trivially; only cblas_dim's i32 conversion of m should reject.
    sgemm_nn_ab(&a, &b, &mut c, huge_m, 0, 0, 1.0, 0.0);
}

// Accelerate must not receive zero leading dimensions. These tests use real
// input allocations even when the operation has no logical input elements.
#[cfg(target_os = "macos")]
#[test]
fn sgemm_nn_ab_empty_output_leaves_c_untouched() {
    let input = [1.0f32; 6];
    let mut c = [3.0f32, 4.0, 5.0];
    sgemm_nn_ab(&[], &input, &mut c, 0, 2, 3, 1.0, 0.0);
    assert_eq!(c, [3.0, 4.0, 5.0]);
    sgemm_nn_ab(&input, &[], &mut c, 2, 0, 3, 1.0, 0.0);
    assert_eq!(c, [3.0, 4.0, 5.0]);
}

#[cfg(target_os = "macos")]
#[test]
fn sgemm_nn_ab_empty_product_scales_c_without_reading_zero_beta() {
    for beta in [0.0, -0.0] {
        let mut c = [f32::NAN, -2.0, 3.0, f32::NAN, 91.0];
        sgemm_nn_ab(&[], &[], &mut c, 2, 2, 0, f32::NAN, beta);
        assert_eq!(&c[..4], &[0.0; 4]);
        assert_eq!(c[4], 91.0);
    }
    let mut c = [1.0, 2.0, -3.0, 4.0, 91.0];
    sgemm_nn_ab(&[], &[], &mut c, 2, 2, 0, f32::INFINITY, 2.0);
    assert_eq!(c, [2.0, 4.0, -6.0, 8.0, 91.0]);
    let mut c = [1.0, 91.0];
    sgemm_nn_ab(&[], &[], &mut c, 1, 1, 0, 1.0, f32::NAN);
    assert!(c[0].is_nan());
    assert_eq!(c[1], 91.0);
}

#[cfg(target_os = "macos")]
#[test]
fn sgemm_strided_empty_shapes_preserve_output_and_padding() {
    type Strided =
        unsafe fn(*const f32, usize, *const f32, usize, *mut f32, usize, usize, usize, usize);
    let input = [1.0f32; 8];
    for (operation, transposed) in [
        (sgemm_nn_strided as Strided, false),
        (sgemm_bt_strided as Strided, true),
    ] {
        let mut c = [3.0f32, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        // SAFETY: Input and output pointers refer to live allocations large
        // enough for the described rows; these shapes have no output elements.
        unsafe {
            operation(
                input.as_ptr(),
                3,
                input.as_ptr(),
                3,
                c.as_mut_ptr(),
                2,
                0,
                2,
                3,
            );
            operation(
                input.as_ptr(),
                3,
                input.as_ptr(),
                3,
                c.as_mut_ptr(),
                0,
                2,
                0,
                3,
            );
        }
        assert_eq!(c, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let mut c = [f32::NAN, -9.0, 91.0, 92.0, 7.0, f32::NAN, 95.0, 96.0];
        let ldb = if transposed { 0 } else { 2 };
        // SAFETY: The output has two writable rows of two elements at stride
        // four; k=0 means neither input has logical elements to read.
        unsafe {
            operation(
                input.as_ptr(),
                0,
                input.as_ptr(),
                ldb,
                c.as_mut_ptr(),
                4,
                2,
                2,
                0,
            );
        }
        assert_eq!(c, [0.0, 0.0, 91.0, 92.0, 0.0, 0.0, 95.0, 96.0]);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "b too short for k*n")]
fn sgemm_nn_ab_empty_output_still_validates_inputs_first() {
    let mut c = [0.0f32; 1];
    sgemm_nn_ab(&[], &[], &mut c, 0, 2, 3, 1.0, 0.0);
}

#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "ldc=2147483648 exceeds i32::MAX")]
fn sgemm_bt_strided_empty_output_still_checks_all_abi_dimensions() {
    let input = [0.0f32; 1];
    let mut c = [0.0f32; 1];
    // SAFETY: ABI conversion rejects the oversized stride before any pointer use.
    unsafe {
        sgemm_bt_strided(
            input.as_ptr(),
            0,
            input.as_ptr(),
            0,
            c.as_mut_ptr(),
            i32::MAX as usize + 1,
            0,
            0,
            0,
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn sgemm_strided_nonempty_controls_keep_ffi_layout() {
    type Strided =
        unsafe fn(*const f32, usize, *const f32, usize, *mut f32, usize, usize, usize, usize);
    let a = [1.0f32, 2.0, 3.0, 4.0];
    for (operation, b) in [
        (sgemm_nn_strided as Strided, [1.0f32, 2.0, 3.0, 4.0]),
        (sgemm_bt_strided as Strided, [1.0f32, 3.0, 2.0, 4.0]),
    ] {
        let mut c = [f32::NAN, f32::NAN, 91.0, f32::NAN, f32::NAN, 92.0];
        // SAFETY: A and B each contain the described 2x2 matrices; C has two
        // writable rows of two elements at stride three.
        unsafe { operation(a.as_ptr(), 2, b.as_ptr(), 2, c.as_mut_ptr(), 3, 2, 2, 2) };
        assert_eq!(c, [7.0, 10.0, 91.0, 15.0, 22.0, 92.0]);
    }
    let mut c = [1.0f32, 2.0, 3.0, 4.0, 91.0];
    sgemm_nn_ab(&a, &a, &mut c, 2, 2, 2, 1.0, 2.0);
    assert_eq!(c, [9.0, 14.0, 21.0, 30.0, 91.0]);
}

#[cfg(not(target_os = "macos"))]
#[test]
fn sgemm_nn_ab_fallback_empty_product_zero_beta_overwrites_nonfinite_c() {
    for beta in [0.0, -0.0] {
        let mut c = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 91.0];
        sgemm_nn_ab(&[], &[], &mut c, 1, 3, 0, 1.0, beta);
        assert_eq!(c, [0.0, 0.0, 0.0, 91.0]);
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn sgemm_nn_ab_fallback_zero_beta_ignores_nonfinite_c() {
    let a = [1.0, 2.0, 3.0, 4.0];
    for beta in [0.0, -0.0] {
        let mut c = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 91.0];
        sgemm_nn_ab(&a, &a, &mut c, 2, 2, 2, 2.0, beta);
        assert_eq!(c, [14.0, 20.0, 30.0, 44.0, 91.0]);
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn sgemm_nn_ab_fallback_nonzero_beta_accumulates_c() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let mut c = [2.0, 4.0, 6.0, 8.0, 91.0];
    sgemm_nn_ab(&a, &a, &mut c, 2, 2, 2, 2.0, 0.5);
    assert_eq!(c, [15.0, 22.0, 33.0, 48.0, 91.0]);
    let mut empty = [2.0, -4.0, 91.0];
    sgemm_nn_ab(&[], &[], &mut empty, 1, 2, 0, 1.0, 0.5);
    assert_eq!(empty, [1.0, -2.0, 91.0]);
}

#[cfg(not(target_os = "macos"))]
#[test]
fn sgemm_nn_ab_fallback_empty_product_ignores_nonfinite_alpha() {
    for beta in [0.0, -0.0] {
        let mut c = [f32::NAN, f32::INFINITY, 91.0];
        sgemm_nn_ab(&[], &[], &mut c, 1, 2, 0, f32::NAN, beta);
        assert_eq!(c, [0.0, 0.0, 91.0]);
    }
    let mut c = [2.0, -4.0, 91.0];
    sgemm_nn_ab(&[], &[], &mut c, 1, 2, 0, f32::NAN, 0.5);
    assert_eq!(c, [1.0, -2.0, 91.0]);
}

#[cfg(not(target_os = "macos"))]
#[test]
fn sgemm_nn_ab_fallback_zero_alpha_ignores_nonfinite_inputs() {
    let a = [f32::NAN, f32::INFINITY];
    let b = [f32::NEG_INFINITY, f32::NAN];
    for alpha in [0.0, -0.0] {
        for beta in [0.0, -0.0] {
            let mut c = [f32::NAN, 91.0];
            sgemm_nn_ab(&a, &b, &mut c, 1, 1, 2, alpha, beta);
            assert_eq!(c, [0.0, 91.0]);
        }
        let mut c = [4.0, 91.0];
        sgemm_nn_ab(&a, &b, &mut c, 1, 1, 2, alpha, 0.5);
        assert_eq!(c, [2.0, 91.0]);
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
#[should_panic]
fn sgemm_nn_ab_fallback_zero_alpha_still_validates_inputs() {
    sgemm_nn_ab(&[], &[1.0], &mut [1.0], 1, 1, 1, 0.0, 0.0);
}

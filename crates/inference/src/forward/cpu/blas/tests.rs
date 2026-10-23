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

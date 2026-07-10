//! Accelerate-backed and fallback SGEMM helpers for transposed-B and non-transposed matrix multiplies.
// ===================================================================
// Apple Accelerate framework BLAS binding (macOS only)
// ===================================================================
//
// On macOS, we dispatch matmul and matmul_bt to Apple's Accelerate
// framework (cblas_sgemm), which uses the AMX coprocessor on Apple
// Silicon for near-theoretical f32 GEMM throughput (100-300 Gelem/s),
// roughly 3-10x faster than hand-written NEON intrinsics (~25-30 Gelem/s).
//
// This is a zero-dependency approach: Accelerate.framework is a system
// framework available on all macOS installations.

// `validate_gemm_bt` is only used by the macOS `accelerate_matmul_bt` FFI wrapper below; the
// non-macOS fallback path in this file only uses `validate_gemm_nn`/`validate_gemm_strided_shape`,
// so an unconditional import trips `-D warnings` unused-import on non-macOS CI legs (#796 round 1).
#[cfg(target_os = "macos")]
use super::gemm_validate::validate_gemm_bt;
use super::gemm_validate::{validate_gemm_nn, validate_gemm_strided_shape};
#[cfg(not(target_os = "macos"))]
use super::matmul::{matmul_bt, matmul_into};

#[cfg(target_os = "macos")]
mod accelerate {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        /// cblas_sgemm: Single-precision General Matrix Multiply
        /// C = alpha * op(A) * op(B) + beta * C
        pub fn cblas_sgemm(
            order: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );

    }

    pub const CBLAS_ROW_MAJOR: i32 = 101;
    pub const CBLAS_NO_TRANS: i32 = 111;
    pub const CBLAS_TRANS: i32 = 112;
}

/// Convert a GEMM dimension or leading dimension to CBLAS's `i32` ABI type, failing closed
/// (release-active) instead of silently wrapping via an unchecked `as i32` cast.
///
/// `validate_gemm_nn`/`validate_gemm_bt`/`validate_gemm_strided_shape` check `usize` shape-
/// product overflow and buffer-length bounds, but a non-overflowing `usize` value above
/// `i32::MAX` still passes those checks (e.g. `m=0, k=0, n=i32::MAX as usize + 1`: every
/// canonical product is zero, so every length check is trivially satisfied) and would
/// otherwise reach `cblas_sgemm` as a negative parameter (#796 round 1 finding 2). Every
/// Accelerate call site converts each `m`/`n`/`k`/`lda`/`ldb`/`ldc` through this helper
/// immediately before the FFI call, so the checked contract covers the full path to CBLAS,
/// not just the `usize`-level shape check.
#[cfg(target_os = "macos")]
#[inline]
fn cblas_dim(value: usize, param: &'static str, op: &'static str) -> i32 {
    i32::try_from(value)
        .unwrap_or_else(|_| panic!("{op}: {param}={value} exceeds i32::MAX (CBLAS ABI limit)"))
}

/// Accelerate-backed C = A @ B^T.
///
/// A is m x k (row-major), B is n x k (row-major, transposed in the multiply),
/// output is m x n (row-major).
///
/// In BLAS terms: C = 1.0 * A * B^T + 0.0 * C
///   - A: NoTrans, lda = k
///   - B: Trans (stored n x k, transposed to k x n), ldb = k
///   - C: ldc = n
#[cfg(target_os = "macos")]
pub(super) fn accelerate_matmul_bt(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    // Release-active, overflow-first, oversized-scratch-allowed contract (ADR-080 C4) —
    // validated BEFORE the unsafe Accelerate FFI call below.
    validate_gemm_bt(
        a.len(),
        b.len(),
        output.len(),
        m,
        k,
        n,
        "accelerate_matmul_bt",
    );

    // Note: cblas_sgemv was benchmarked for M=1 and was SLOWER than sgemm
    // (4.98 vs 5.57 tok/s). Accelerate's sgemm appears to use multi-threaded
    // AMX tiling even for M=1, while sgemv dispatches single-threaded.
    // Keeping sgemm for all M values.

    // SAFETY: Pointers are valid for the dimensions specified, verified by
    // the assertions above. cblas_sgemm reads exactly m*k from A, n*k from B,
    // and writes exactly m*n to C. The Accelerate framework is thread-safe
    // for non-overlapping output regions.
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            accelerate::CBLAS_NO_TRANS,
            accelerate::CBLAS_TRANS,
            cblas_dim(m, "m", "accelerate_matmul_bt"),
            cblas_dim(n, "n", "accelerate_matmul_bt"),
            cblas_dim(k, "k", "accelerate_matmul_bt"),
            1.0,
            a.as_ptr(),
            cblas_dim(k, "lda", "accelerate_matmul_bt"),
            b.as_ptr(),
            cblas_dim(k, "ldb", "accelerate_matmul_bt"),
            0.0,
            output.as_mut_ptr(),
            cblas_dim(n, "ldc", "accelerate_matmul_bt"),
        );
    }
}

/// Accelerate-backed C = A @ B (non-transposed).
///
/// A is m x k (row-major), B is k x n (row-major), output is m x n (row-major).
///
/// In BLAS terms: C = 1.0 * A * B + 0.0 * C
///   - A: NoTrans, lda = k
///   - B: NoTrans, ldb = n
///   - C: ldc = n
#[cfg(target_os = "macos")]
pub(super) fn accelerate_matmul(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    // Release-active, overflow-first, oversized-scratch-allowed contract (ADR-080 C4) —
    // validated BEFORE the unsafe Accelerate FFI call below.
    validate_gemm_nn(a.len(), b.len(), output.len(), m, k, n, "accelerate_matmul");

    // SAFETY: Same safety argument as accelerate_matmul_bt. Pointers are valid
    // for the specified dimensions, verified by assertions above.
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            accelerate::CBLAS_NO_TRANS,
            accelerate::CBLAS_NO_TRANS,
            cblas_dim(m, "m", "accelerate_matmul"),
            cblas_dim(n, "n", "accelerate_matmul"),
            cblas_dim(k, "k", "accelerate_matmul"),
            1.0,
            a.as_ptr(),
            cblas_dim(k, "lda", "accelerate_matmul"),
            b.as_ptr(),
            cblas_dim(n, "ldb", "accelerate_matmul"),
            0.0,
            output.as_mut_ptr(),
            cblas_dim(n, "ldc", "accelerate_matmul"),
        );
    }
}

// ===================================================================
// Strided GEMM wrappers for zero-copy sub-matrix operations
// ===================================================================
//
// These functions allow BLAS to operate on interleaved sub-matrices
// (e.g. Q/K/V packed in a single fused buffer) without copying data.
// The leading dimension (stride) tells BLAS how many floats to skip
// between consecutive rows, which can exceed the column count when
// operating on a sub-matrix of a larger buffer.

/// **Unstable**: strided GEMM B^T via Accelerate; signature tied to macOS/Accelerate API.
///
/// Strided GEMM: C = A @ B^T with custom leading dimensions.
/// A is [m, k] submatrix with stride lda between rows.
/// B is [n, k] submatrix with stride ldb between rows.
/// C is [m, n] submatrix with stride ldc between rows.
/// This enables operating on submatrices without copying.
///
/// # Safety
///
/// Caller must ensure `a`, `b`, and `c` are valid for the strided matrix
/// regions described by `m`, `n`, `k`, `lda`, `ldb`, and `ldc`, and that `c`
/// is writable for all output elements.
#[cfg(target_os = "macos")]
pub unsafe fn sgemm_bt_strided(
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    // Release-active, overflow-first shape/stride check (ADR-080 C4) — the caller contract
    // (raw pointers) means we cannot verify buffer length, but shape-product overflow and an
    // undersized leading dimension are both checkable and are exactly the class of malformed
    // input that would otherwise cross into the unsafe FFI call below.
    validate_gemm_strided_shape(m, k, n, lda, ldb, ldc, true, "sgemm_bt_strided");

    // SAFETY: Caller guarantees that a[row * lda + col] is valid for
    // row in 0..m, col in 0..k; similarly for b and c. The Accelerate
    // framework reads/writes only within these bounds.
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            accelerate::CBLAS_NO_TRANS,
            accelerate::CBLAS_TRANS,
            cblas_dim(m, "m", "sgemm_bt_strided"),
            cblas_dim(n, "n", "sgemm_bt_strided"),
            cblas_dim(k, "k", "sgemm_bt_strided"),
            1.0,
            a,
            cblas_dim(lda, "lda", "sgemm_bt_strided"),
            b,
            cblas_dim(ldb, "ldb", "sgemm_bt_strided"),
            0.0,
            c,
            cblas_dim(ldc, "ldc", "sgemm_bt_strided"),
        );
    }
}

/// **Unstable**: strided GEMM A@B via Accelerate; signature tied to macOS/Accelerate API.
///
/// Strided GEMM: C = A @ B with custom leading dimensions.
/// A is [m, k] submatrix with stride lda between rows.
/// B is [k, n] submatrix with stride ldb between rows.
/// C is [m, n] submatrix with stride ldc between rows.
///
/// # Safety
///
/// Caller must ensure `a`, `b`, and `c` are valid for the strided matrix
/// regions described by `m`, `n`, `k`, `lda`, `ldb`, and `ldc`, and that `c`
/// is writable for all output elements.
#[cfg(target_os = "macos")]
pub unsafe fn sgemm_nn_strided(
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    // Release-active, overflow-first shape/stride check (ADR-080 C4) — see sgemm_bt_strided.
    validate_gemm_strided_shape(m, k, n, lda, ldb, ldc, false, "sgemm_nn_strided");

    // SAFETY: Same argument as sgemm_bt_strided. Caller guarantees all
    // pointer+stride combinations are within valid allocations.
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            accelerate::CBLAS_NO_TRANS,
            accelerate::CBLAS_NO_TRANS,
            cblas_dim(m, "m", "sgemm_nn_strided"),
            cblas_dim(n, "n", "sgemm_nn_strided"),
            cblas_dim(k, "k", "sgemm_nn_strided"),
            1.0,
            a,
            cblas_dim(lda, "lda", "sgemm_nn_strided"),
            b,
            cblas_dim(ldb, "ldb", "sgemm_nn_strided"),
            0.0,
            c,
            cblas_dim(ldc, "ldc", "sgemm_nn_strided"),
        );
    }
}

/// **Unstable**: scaled A@B via Accelerate; alpha/beta semantics may change.
///
/// Accelerate-backed C = alpha * A @ B + beta * C (non-transposed, with scaling).
///
/// A is m x k (row-major), B is k x n (row-major), C is m x n (row-major).
/// Unlike `accelerate_matmul`, this exposes alpha and beta for fused operations
/// such as rank-1 state updates: S = g*S + k*delta^T via beta=g, alpha=1.
#[cfg(target_os = "macos")]
pub fn sgemm_nn_ab(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    // Release-active, overflow-first, oversized-scratch-allowed contract (ADR-080 C4,
    // held finding: this was previously `debug_assert!`-gated, so a short slice on a
    // release build's non-macOS scalar copy merely tripped Rust's own bounds checks while
    // this cfg-gated Accelerate FFI copy crossed into unsafe code with undersized
    // pointers). Validated BEFORE the unsafe FFI call below.
    validate_gemm_nn(a.len(), b.len(), c.len(), m, k, n, "sgemm_nn_ab");

    // SAFETY: Pointers valid for specified dimensions per the release-active check above.
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            accelerate::CBLAS_NO_TRANS,
            accelerate::CBLAS_NO_TRANS,
            cblas_dim(m, "m", "sgemm_nn_ab"),
            cblas_dim(n, "n", "sgemm_nn_ab"),
            cblas_dim(k, "k", "sgemm_nn_ab"),
            alpha,
            a.as_ptr(),
            cblas_dim(k, "lda", "sgemm_nn_ab"),
            b.as_ptr(),
            cblas_dim(n, "ldb", "sgemm_nn_ab"),
            beta,
            c.as_mut_ptr(),
            cblas_dim(n, "ldc", "sgemm_nn_ab"),
        );
    }
}

/// **Unstable**: non-macOS scalar fallback for scaled A@B; will be replaced by portable BLAS.
///
/// Non-macOS fallback: C = alpha * A @ B + beta * C.
#[cfg(not(target_os = "macos"))]
pub fn sgemm_nn_ab(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    // Release-active, overflow-first, oversized-scratch-allowed contract (ADR-080 C4) —
    // matches the macOS `sgemm_nn_ab` variant so both platform copies share the same
    // release-active guarantee instead of only the FFI copy being upgraded.
    validate_gemm_nn(a.len(), b.len(), c.len(), m, k, n, "sgemm_nn_ab");

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
}

/// **Unstable**: non-macOS fallback for strided B^T GEMM; copies rows, may be replaced.
///
/// Non-macOS fallback: C = A @ B^T with strided access.
/// Copies strided rows into contiguous buffers, then delegates to matmul_bt.
///
/// # Safety
/// Caller must guarantee that pointer+stride combinations are within valid allocations.
#[cfg(not(target_os = "macos"))]
pub unsafe fn sgemm_bt_strided(
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    // Release-active, overflow-first shape/stride check (ADR-080 C4) — validated before any
    // of the unsafe strided reads/writes below. Previously this fallback had NO validation
    // at all (unlike the macOS FFI copy of this function).
    validate_gemm_strided_shape(m, k, n, lda, ldb, ldc, true, "sgemm_bt_strided");

    // Copy strided A into contiguous buffer.
    let mut a_contig = vec![0.0f32; m * k];
    for row in 0..m {
        // SAFETY: Caller guarantees a[row * lda..row * lda + k] is readable.
        let src = unsafe { std::slice::from_raw_parts(a.add(row * lda), k) };
        a_contig[row * k..(row + 1) * k].copy_from_slice(src);
    }
    // Copy strided B into contiguous buffer.
    let mut b_contig = vec![0.0f32; n * k];
    for row in 0..n {
        // SAFETY: Caller guarantees b[row * ldb..row * ldb + k] is readable.
        let src = unsafe { std::slice::from_raw_parts(b.add(row * ldb), k) };
        b_contig[row * k..(row + 1) * k].copy_from_slice(src);
    }
    // Compute into contiguous output, then scatter back with stride.
    let mut c_contig = vec![0.0f32; m * n];
    matmul_bt(&a_contig, &b_contig, &mut c_contig, m, k, n);
    for row in 0..m {
        // SAFETY: Caller guarantees c[row * ldc..row * ldc + n] is writable.
        let dst = unsafe { std::slice::from_raw_parts_mut(c.add(row * ldc), n) };
        dst.copy_from_slice(&c_contig[row * n..(row + 1) * n]);
    }
}

/// **Unstable**: non-macOS fallback for strided A@B GEMM; copies rows, may be replaced.
///
/// Non-macOS fallback: C = A @ B (non-transposed) with strided access.
///
/// # Safety
/// Caller must guarantee that pointer+stride combinations are within valid allocations.
#[cfg(not(target_os = "macos"))]
pub unsafe fn sgemm_nn_strided(
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    // Release-active, overflow-first shape/stride check (ADR-080 C4) — validated before any
    // of the unsafe strided reads/writes below.
    validate_gemm_strided_shape(m, k, n, lda, ldb, ldc, false, "sgemm_nn_strided");

    let mut a_contig = vec![0.0f32; m * k];
    for row in 0..m {
        // SAFETY: Caller guarantees a[row * lda..row * lda + k] is readable.
        let src = unsafe { std::slice::from_raw_parts(a.add(row * lda), k) };
        a_contig[row * k..(row + 1) * k].copy_from_slice(src);
    }
    let mut b_contig = vec![0.0f32; k * n];
    for row in 0..k {
        // SAFETY: Caller guarantees b[row * ldb..row * ldb + n] is readable.
        let src = unsafe { std::slice::from_raw_parts(b.add(row * ldb), n) };
        b_contig[row * n..(row + 1) * n].copy_from_slice(src);
    }
    let mut c_contig = vec![0.0f32; m * n];
    matmul_into(&a_contig, &b_contig, &mut c_contig, m, k, n);
    for row in 0..m {
        // SAFETY: Caller guarantees c[row * ldc..row * ldc + n] is writable.
        let dst = unsafe { std::slice::from_raw_parts_mut(c.add(row * ldc), n) };
        dst.copy_from_slice(&c_contig[row * n..(row + 1) * n]);
    }
}

#[cfg(test)]
mod tests {
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
}

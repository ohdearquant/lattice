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
    assert!(a.len() >= m * k, "A too small: {} < {}", a.len(), m * k);
    assert!(b.len() >= n * k, "B too small: {} < {}", b.len(), n * k);
    assert!(
        output.len() >= m * n,
        "C too small: {} < {}",
        output.len(),
        m * n
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
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            0.0,
            output.as_mut_ptr(),
            n as i32,
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
    assert!(a.len() >= m * k, "A too small: {} < {}", a.len(), m * k);
    assert!(b.len() >= k * n, "B too small: {} < {}", b.len(), k * n);
    assert!(
        output.len() >= m * n,
        "C too small: {} < {}",
        output.len(),
        m * n
    );

    // SAFETY: Same safety argument as accelerate_matmul_bt. Pointers are valid
    // for the specified dimensions, verified by assertions above.
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            accelerate::CBLAS_NO_TRANS,
            accelerate::CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            output.as_mut_ptr(),
            n as i32,
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
    // SAFETY: Caller guarantees that a[row * lda + col] is valid for
    // row in 0..m, col in 0..k; similarly for b and c. The Accelerate
    // framework reads/writes only within these bounds.
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            accelerate::CBLAS_NO_TRANS,
            accelerate::CBLAS_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            lda as i32,
            b,
            ldb as i32,
            0.0,
            c,
            ldc as i32,
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
    // SAFETY: Same argument as sgemm_bt_strided. Caller guarantees all
    // pointer+stride combinations are within valid allocations.
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            accelerate::CBLAS_NO_TRANS,
            accelerate::CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            lda as i32,
            b,
            ldb as i32,
            0.0,
            c,
            ldc as i32,
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
    debug_assert!(a.len() >= m * k, "A too small: {} < {}", a.len(), m * k);
    debug_assert!(b.len() >= k * n, "B too small: {} < {}", b.len(), k * n);
    debug_assert!(c.len() >= m * n, "C too small: {} < {}", c.len(), m * n);

    // SAFETY: Pointers valid for specified dimensions per debug_asserts above.
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            accelerate::CBLAS_NO_TRANS,
            accelerate::CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            beta,
            c.as_mut_ptr(),
            n as i32,
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
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= m * n);

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

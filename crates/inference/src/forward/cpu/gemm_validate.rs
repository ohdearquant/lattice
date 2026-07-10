//! Shared checked GEMM-family argument validator (ADR-080 cluster C4, #784).
//!
//! Canonical contract, originally established in `matmul.rs` (#368, #218/#224 precedent)
//! and now shared by every safe GEMM/GEMV/matvec entry point in this crate:
//!
//! 1. **Overflow-first**: shape-product overflow (`m*k`, `n*k`/`k*n`, `m*n`) is checked with
//!    `checked_mul` BEFORE that product is used to index or size a buffer. A wrapped product
//!    can otherwise make a subsequent `>=` length check pass spuriously on a malformed shape
//!    (the #367 lesson).
//! 2. **Release-active**: real `assert!`, never `debug_assert!`. These wrappers are safe `pub`
//!    functions reachable from arbitrary (possibly model-/config-controlled) call sites; a
//!    check that disappears in release leaves the unsafe SIMD/FFI kernel behind it exposed to
//!    undersized-pointer UB.
//! 3. **Oversized-scratch-prefix allow-list**: the check on each input/output buffer is `>=`,
//!    never `==`. Callers may legitimately pass a reused scratch buffer that is longer than the
//!    exact logical footprint; only the buffer being *too short* is a shape error. Content
//!    beyond the logical footprint is unspecified and may be clobbered by the callee.
//!
//! Backend kernels (SIMD tile variants, Accelerate/BLAS FFI, Metal/WGPU shaders) stay separate
//! implementations; this module only centralizes the pre-dispatch argument check that gates
//! entry into them.

/// Validate arguments for `C = A @ B` (row-major, `A: [m, k]`, `B: [k, n]`, `C: [m, n]`).
///
/// `op` is a short static label included in panic messages to identify the failing call site.
#[inline]
pub(crate) fn validate_gemm_nn(
    a_len: usize,
    b_len: usize,
    c_len: usize,
    m: usize,
    k: usize,
    n: usize,
    op: &'static str,
) {
    assert!(m.checked_mul(k).is_some(), "{op}: shape overflow: m*k");
    assert!(k.checked_mul(n).is_some(), "{op}: shape overflow: k*n");
    assert!(m.checked_mul(n).is_some(), "{op}: shape overflow: m*n");
    assert!(a_len >= m * k, "{op}: a too short for m*k");
    assert!(b_len >= k * n, "{op}: b too short for k*n");
    assert!(c_len >= m * n, "{op}: c too short for m*n");
}

/// Validate arguments for `C = A @ B^T` (row-major, `A: [m, k]`, `B: [n, k]`, `C: [m, n]`).
#[inline]
pub(crate) fn validate_gemm_bt(
    a_len: usize,
    b_len: usize,
    c_len: usize,
    m: usize,
    k: usize,
    n: usize,
    op: &'static str,
) {
    assert!(m.checked_mul(k).is_some(), "{op}: shape overflow: m*k");
    assert!(n.checked_mul(k).is_some(), "{op}: shape overflow: n*k");
    assert!(m.checked_mul(n).is_some(), "{op}: shape overflow: m*n");
    assert!(a_len >= m * k, "{op}: a too short for m*k");
    assert!(b_len >= n * k, "{op}: b too short for n*k");
    assert!(c_len >= m * n, "{op}: c too short for m*n");
}

/// Validate shape + leading-dimension arguments for a strided GEMM (pointer-based, no slice
/// length available to the callee). Checks shape-product overflow and that every leading
/// dimension is at least as large as the row extent it strides over — the same class of
/// overflow-first, release-active guard as [`validate_gemm_nn`]/[`validate_gemm_bt`], scoped to
/// what a raw-pointer strided API can check without slice lengths.
///
/// `transposed_b`: `true` for `A @ B^T` layouts (`B` stored `[n, k]`, `ldb >= k`), `false` for
/// `A @ B` layouts (`B` stored `[k, n]`, `ldb >= n`).
#[inline]
pub(crate) fn validate_gemm_strided_shape(
    m: usize,
    k: usize,
    n: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    transposed_b: bool,
    op: &'static str,
) {
    assert!(m.checked_mul(k).is_some(), "{op}: shape overflow: m*k");
    assert!(n.checked_mul(k).is_some(), "{op}: shape overflow: n*k");
    assert!(m.checked_mul(n).is_some(), "{op}: shape overflow: m*n");
    assert!(lda >= k, "{op}: lda too small for row extent k");
    if transposed_b {
        assert!(ldb >= k, "{op}: ldb too small for row extent k");
    } else {
        assert!(ldb >= n, "{op}: ldb too small for row extent n");
    }
    assert!(ldc >= n, "{op}: ldc too small for row extent n");
}

/// Validate arguments for the BitNet ternary matvec kernels (`matvec_ternary_scalar` /
/// `matvec_ternary_neon`). Packed ternary layout is distinct from row-major f32 GEMM, so this
/// is its own contract, not a reuse of `validate_gemm_*`: weights are packed 4-per-byte
/// (`packed_row_bytes(k)` bytes per row), and the activation/alpha/output vectors are addressed
/// densely by `k`/`n` rather than a 2-D stride.
///
/// `x_q_len`/`alphas_len`/`packed_w_len`/`output_len` all follow the same oversized-allowed
/// `>=` contract as the GEMM validators above: each is checked against its minimum required
/// footprint (`packed_w` against `n * packed_row_bytes(k)`), and a caller-supplied buffer
/// longer than that minimum is accepted — only rows/elements within the required footprint
/// are ever read (see `validate_ternary_matvec_args_accepts_oversized` below).
#[inline]
pub(crate) fn validate_ternary_matvec_args(
    x_q_len: usize,
    alphas_len: usize,
    packed_w_len: usize,
    output_len: usize,
    n: usize,
    k: usize,
    packed_row_bytes: usize,
    op: &'static str,
) {
    assert!(
        n.checked_mul(packed_row_bytes).is_some(),
        "{op}: shape overflow: n*packed_row_bytes"
    );
    assert!(x_q_len >= k, "{op}: x_q too short for k");
    assert!(alphas_len >= n, "{op}: alphas too short for n");
    assert!(
        packed_w_len >= n * packed_row_bytes,
        "{op}: packed_w too short for n*packed_row_bytes"
    );
    assert!(output_len >= n, "{op}: output too short for n");
}

#[cfg(test)]
mod tests {
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
}

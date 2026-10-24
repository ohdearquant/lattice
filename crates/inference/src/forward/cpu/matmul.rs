//! General matmul helpers, preallocated matmul, transposed-B matmul, scalar fallbacks, and m=1 specialization.
use super::gemm_validate::{validate_gemm_bt, validate_gemm_nn};
#[cfg(not(target_os = "macos"))]
use super::simd::simd_config;

#[cfg(all(not(target_os = "macos"), target_arch = "aarch64"))]
use super::arch_kernels::matmul_neon;
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
use super::arch_kernels::{matmul_avx2, matmul_avx512};
#[cfg(target_os = "macos")]
use super::blas::{accelerate_matmul, accelerate_matmul_bt};
#[cfg(not(target_os = "macos"))]
use super::tiled::matmul_bt_tiled;

#[cfg(all(test, feature = "train-backward"))]
std::thread_local! {
    static OBSERVED_MATMUL_CALLS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(all(test, feature = "train-backward"))]
pub(crate) fn observe_matmul_calls<T>(work: impl FnOnce() -> T) -> (T, usize) {
    struct Restore(Option<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OBSERVED_MATMUL_CALLS.with(|calls| {
                let nested = calls.get().unwrap_or(0);
                calls.set(self.0.map(|previous| previous.saturating_add(nested)));
            });
        }
    }
    let previous = OBSERVED_MATMUL_CALLS.with(|calls| calls.replace(Some(0)));
    let restore = Restore(previous);
    let result = work();
    let count = OBSERVED_MATMUL_CALLS.with(|calls| calls.get().unwrap_or(0));
    drop(restore);
    (result, count)
}

/// **Unstable**: general matmul C = A*B; dispatches to platform BLAS or SIMD fallback.
///
/// General matrix multiplication: C = A * B.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // Guard the allocation itself: an overflowed m*n would silently allocate a tiny Vec
    // whose length would then pass matmul_into's c.len() >= m*n check (same wrapped value).
    // Checked here so the vec! and all downstream guards use the same trustworthy product.
    assert!(
        m.checked_mul(n).is_some(),
        "matmul output shape overflow: m*n"
    );
    let mut c = vec![0.0f32; m * n];
    matmul_into(a, b, &mut c, m, k, n);
    c
}

/// **Unstable**: matmul into pre-allocated buffer; dispatch logic may change.
///
/// Matrix multiply into a pre-allocated output buffer.
pub fn matmul_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(all(test, feature = "train-backward"))]
    OBSERVED_MATMUL_CALLS.with(|calls| {
        if let Some(count) = calls.get() {
            calls.set(Some(count.saturating_add(1)));
        }
    });
    // Release-active, overflow-first, oversized-scratch-allowed contract (#368, ADR-080 C4) —
    // see `gemm_validate` for the shared rationale. Some callers pass reused scratch buffers
    // longer than the exact footprint; that is sound (the check is `>=`). Note the output
    // suffix beyond m*n is NOT part of the result and may be clobbered (matmul_scalar zeroes
    // the full c slice) — callers needing suffix preservation must pass &mut c[..m*n].
    validate_gemm_nn(a.len(), b.len(), c.len(), m, k, n, "matmul");

    // GPU dispatch is NOT in this hot path — per-call buffer creation is too slow.
    // GPU acceleration requires the full forward pass to run on-device.
    // See gpu_gemm.rs for standalone GPU GEMM (used in benchmarks).

    #[cfg(target_os = "macos")]
    {
        accelerate_matmul(a, b, c, m, n, k);
    }

    #[cfg(not(target_os = "macos"))]
    matmul_scalar(a, b, c, m, k, n);
}

/// **Unstable**: matmul C = A @ B^T; primary inference kernel, dispatch strategy evolving.
///
/// Matrix multiply with transposed B: C = A @ B^T.
pub fn matmul_bt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Release-active, overflow-first, oversized-scratch-allowed contract (#368, ADR-080 C4).
    // Note: B is stored transposed, so its footprint is n*k, not k*n. Some callers pass
    // reused scratch buffers longer than the exact footprint; that is sound (the check is
    // `>=`). The output suffix beyond m*n is NOT part of the result and is unspecified by
    // contract: on x86_64 the dispatch below writes only c[..m*n], while on the other
    // non-macOS targets matmul_bt_tiled zeroes the full c slice it is given, so callers
    // needing suffix preservation there must pass &mut c[..m*n].
    validate_gemm_bt(a.len(), b.len(), c.len(), m, k, n, "matmul_bt");

    // CPU path only — Accelerate AMX on macOS, SIMD/scalar elsewhere.
    #[cfg(target_os = "macos")]
    {
        accelerate_matmul_bt(a, b, c, m, n, k);
    }

    // x86_64 fallback: hand-written SIMD with tiling for large matrices, rows split across
    // the rayon pool for prefill-sized problems, and a single row's columns split across it
    // for large decode-step products.
    #[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
    {
        // ADR-099 F1 step 1: stock `gemm` for every multi-row product. Single rows (decode
        // steps) keep the existing kernels and their column split.
        #[cfg(feature = "x86-gemm")]
        {
            if m > 1 {
                matmul_bt_gemm(a, b, c, m, k, n);
                return;
            }
        }
        matmul_bt_x86(a, b, c, m, k, n, ROW_THREAD_MIN_WORK, COL_THREAD_MIN_WORK);
    }

    // Other non-macOS targets: hand-written SIMD with tiling for large matrices.
    #[cfg(all(not(target_os = "macos"), not(target_arch = "x86_64")))]
    {
        // Use cache-blocked (tiled) path for large matrices where blocking pays off.
        // Two conditions must be met:
        //   1. Total work m*n*k >= 1024*1024 (below this, overhead dominates).
        //   2. K >= 128 (the shared dimension must be large enough that B-rows don't fit
        //      in L1 cache naturally). When K is small (e.g. 32), each B-row is only
        //      128 bytes and fits in L1 without tiling. Tiling would only change the
        //      accumulation order and introduce unnecessary numerical differences.
        // All m rows go to the tiled kernel without a row split: on aarch64 its NEON edge
        // branch vectorises partial TILE_I row tiles along K, so small m is not forced
        // scalar there.
        let total_work = (m as u64) * (n as u64) * (k as u64);
        if total_work >= 1024 * 1024 && k >= super::tiled::TILE_K {
            matmul_bt_tiled(a, b, c, m, k, n);
            return;
        }

        let config = simd_config();

        #[cfg(target_arch = "aarch64")]
        {
            if config.neon_enabled {
                // SAFETY: NEON is available on aarch64 and the runtime gate ensures this path.
                unsafe {
                    matmul_neon(a, b, c, m, k, n);
                    return;
                }
            }
        }

        matmul_bt_scalar(a, b, c, m, k, n);
    }
}

/// `matmul_bt` through stock `gemm` 0.18.2 (ADR-099 F1). It allocates its packing storage
/// inside the call, the one allocation per call that ADR-099 admits. It runs on the calling
/// thread: `gemm`'s parallel path allocates once per worker, which that exception does not admit.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64", feature = "x86-gemm"))]
fn matmul_bt_gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c[..m * n].fill(0.0);
        return;
    }
    // SAFETY: `matmul_bt` validated that a holds m*k, b holds n*k and c holds m*n elements,
    // with no overflow. C is m x n row-major (column stride 1, row stride n), A is m x k
    // row-major, and B^T is read from the n x k row-major B (row stride 1, column stride k).
    // alpha = 0 with read_dst = false overwrites C without reading it.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            k as isize,
            1,
            0.0,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
}

/// Multiply-add count (m*n*k) at and above which `matmul_bt` splits its rows across the
/// rayon pool on x86_64: twice the smallest shape measured to gain from the split
/// (16 x 1024 x 3584). Only rows are split, and only on x86_64; other targets are unmeasured.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
const ROW_THREAD_MIN_WORK: u64 = 117_440_512;

/// Smallest row chunk worth handing to another thread; shorter chunks lose to the split's cost.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
const ROW_THREAD_MIN_CHUNK: usize = 16;

/// Rows per chunk when `matmul_bt` should split across `threads`, else `None`.
///
/// `None` for fewer than two threads, fewer than 32 rows, an empty shared or output
/// dimension, work (m*n*k) below `min_work`, or when the chunk would cover every row. A chunk
/// is a multiple of TILE_I rows, so every row stays on the kernel the unsplit call picks
/// for it (the tiled kernel takes whole TILE_I row tiles, the direct kernel the remainder).
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
fn row_chunk_len(m: usize, k: usize, n: usize, threads: usize, min_work: u64) -> Option<usize> {
    if threads < 2 || m < 32 || k == 0 || n == 0 {
        return None;
    }
    let work = (m as u64).saturating_mul(n as u64).saturating_mul(k as u64);
    if work < min_work {
        return None;
    }
    let tile = super::tiled::TILE_I;
    let chunk = (m.div_ceil(threads).div_ceil(tile) * tile).max(ROW_THREAD_MIN_CHUNK);
    (chunk < m).then_some(chunk)
}

/// Multiply-add count (n*k) at and above which a single-row `matmul_bt` splits its output
/// columns across the rayon pool on x86_64: twice the smallest shape measured to gain from the
/// split at 2, 4 and 8 threads (1 x 1024 x 256, split/serial 0.89 / 0.77 / 0.73 with B read
/// from DRAM, 8-vCPU AMD EPYC Zen 4; 1 x 1024 x 128 lost at every width). Only x86_64 splits.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
const COL_THREAD_MIN_WORK: u64 = 524_288;

/// Column chunks are whole multiples of this many outputs, one 64-byte cache line of f32, so
/// no two threads write the same line of C.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
const COL_THREAD_MIN_CHUNK: usize = 16;

/// Columns per chunk when a single-row `matmul_bt` should split its output columns across
/// `threads`, else `None`.
///
/// `None` unless m is 1 (a column chunk of a longer product is not contiguous in C and the
/// kernels take no output stride), when the caller already runs on a rayon worker thread
/// (`in_worker`; it may be one of several parallel tasks, and splitting again would put more
/// tasks on the pool than it has threads; this includes a call made inside
/// `ThreadPool::install`), for fewer than two threads, an empty shared or output dimension,
/// work (n*k) below `min_work`, or when the chunk would cover every column.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
fn col_chunk_len(
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
    in_worker: bool,
    min_work: u64,
) -> Option<usize> {
    if m != 1 || in_worker || threads < 2 || k == 0 || n == 0 {
        return None;
    }
    let work = (n as u64).saturating_mul(k as u64);
    if work < min_work {
        return None;
    }
    let chunk = n.div_ceil(threads).div_ceil(COL_THREAD_MIN_CHUNK) * COL_THREAD_MIN_CHUNK;
    (chunk < n).then_some(chunk)
}

/// Whether the tiled kernel takes the leading rows of an m x k x n product.
///
/// Decided once for the whole product: a row chunk that recomputed this from its own,
/// smaller, m would send rows to the direct kernel that the unsplit call gives to the tiled one.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
fn tiled_applies(m: usize, k: usize, n: usize) -> bool {
    // Use cache-blocked (tiled) path for large matrices where blocking pays off.
    // Two conditions must be met:
    //   1. Total work m*n*k >= 1024*1024 (below this, overhead dominates).
    //   2. K >= 128 (the shared dimension must be large enough that B-rows don't fit
    //      in L1 cache naturally). When K is small (e.g. 32), each B-row is only
    //      128 bytes and fits in L1 without tiling. Tiling would only change the
    //      accumulation order and introduce unnecessary numerical differences.
    let total_work = (m as u64) * (n as u64) * (k as u64);
    total_work >= 1024 * 1024 && k >= super::tiled::TILE_K
}

/// How `matmul_bt_x86` ran a product.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Split {
    Serial,
    /// Row chunks of this many rows, run in parallel.
    Rows(usize),
    /// A single row's output columns in chunks of this many columns, run in parallel.
    Columns(usize),
}

/// x86_64 `matmul_bt`: the serial kernels over the whole product, over row chunks run in
/// parallel when `row_chunk_len` says the product is large enough, or, for a single row, over
/// column chunks run in parallel when `col_chunk_len` says so. Each output row is computed
/// from its own A row and B alone, and row chunks start on TILE_I boundaries, so a row split
/// is bit-identical to the serial call; see `matmul_bt_x86_columns` for the column split.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn matmul_bt_x86(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    min_work: u64,
    col_min_work: u64,
) -> Split {
    use rayon::prelude::*;

    let tiled = tiled_applies(m, k, n);
    let threads = rayon::current_num_threads();
    if let Some(rows) = row_chunk_len(m, k, n, threads, min_work) {
        c[..m * n]
            .par_chunks_mut(rows * n)
            .zip(a[..m * k].par_chunks(rows * k))
            .for_each(|(c_chunk, a_chunk)| {
                matmul_bt_x86_rows(a_chunk, b, c_chunk, c_chunk.len() / n, k, n, tiled);
            });
        return Split::Rows(rows);
    }
    let in_worker = rayon::current_thread_index().is_some();
    if let Some(cols) = col_chunk_len(m, k, n, threads, in_worker, col_min_work) {
        matmul_bt_x86_columns(a, b, c, k, n, cols, tiled);
        return Split::Columns(cols);
    }
    matmul_bt_x86_rows(a, b, c, m, k, n, tiled);
    Split::Serial
}

/// A single row's `matmul_bt` over column chunks of `cols` outputs, run across the rayon pool.
///
/// Each output column is computed from the A row and its own B row alone, and a single row is
/// never a whole TILE_I tile, so every chunk runs the same direct kernel the unsplit call runs
/// on the same columns: the result is bit-identical to the serial call.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
fn matmul_bt_x86_columns(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    k: usize,
    n: usize,
    cols: usize,
    tiled: bool,
) {
    use rayon::prelude::*;

    c[..n]
        .par_chunks_mut(cols)
        .zip(b[..n * k].par_chunks(cols * k))
        .for_each(|(c_chunk, b_chunk)| {
            matmul_bt_x86_rows(&a[..k], b_chunk, c_chunk, 1, k, c_chunk.len(), tiled);
        });
}

/// Serial x86_64 `matmul_bt` over `m` rows; `tiled` is `tiled_applies` for the whole product.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
fn matmul_bt_x86_rows(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    tiled: bool,
) {
    // The tiled AVX2 kernel computes a full TILE_I x TILE_J tile with SIMD only when
    // its K-tile has at least 16 elements; every other tile, including every partial
    // TILE_I row tile, runs a scalar loop (the whole tiled call is scalar when
    // AVX2+FMA is not detected). So the tiled path takes only the largest
    // multiple of TILE_I rows, and the remaining rows (all of them when m < TILE_I,
    // as in every decode step) go through the direct kernels below: AVX-512F, then
    // AVX2, then the scalar reference when no SIMD feature is detected. Partial
    // TILE_J column tiles and short K tiles inside the tiled part are still scalar.
    // The tiled call gets exactly c[..full * n] because it zeroes the whole slice it
    // is given.
    let full = if tiled {
        m - m % super::tiled::TILE_I
    } else {
        0
    };
    if full > 0 {
        matmul_bt_tiled(&a[..full * k], b, &mut c[..full * n], full, k, n);
    }
    if full < m {
        matmul_bt_direct(
            &a[full * k..m * k],
            b,
            &mut c[full * n..m * n],
            m - full,
            k,
            n,
        );
    }
}

/// Direct (untiled) transposed-B matmul on x86_64: the widest SIMD kernel the CPU
/// supports, else the scalar reference. Each output element depends only on its own
/// A row and B row.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
fn matmul_bt_direct(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let config = simd_config();

    if config.avx512f_enabled && config.fma_enabled {
        // SAFETY: The runtime feature checks above guarantee AVX-512F+FMA support.
        unsafe {
            matmul_avx512(a, b, c, m, k, n);
            return;
        }
    }
    if config.avx2_enabled && config.fma_enabled {
        // SAFETY: The runtime feature checks above guarantee AVX2+FMA support.
        unsafe {
            matmul_avx2(a, b, c, m, k, n);
            return;
        }
    }

    matmul_bt_scalar(a, b, c, m, k, n);
}

/// **Unstable**: scalar matmul reference; used for non-SIMD targets and testing.
///
/// Scalar reference implementation of A * B.
pub fn matmul_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    c.fill(0.0);

    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            let b_row = &b[p * n..(p + 1) * n];
            let c_row = &mut c[i * n..(i + 1) * n];
            for j in 0..n {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
pub fn matmul_bt_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 1 {
        matmul_bt_scalar_m1(a, b, c, k, n);
        return;
    }
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        for j in 0..n {
            let b_row = &b[j * k..(j + 1) * k];
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            let mut s3 = 0.0f32;
            let unrolled = k / 4;
            for p in 0..unrolled {
                let off = p * 4;
                s0 += a_row[off] * b_row[off];
                s1 += a_row[off + 1] * b_row[off + 1];
                s2 += a_row[off + 2] * b_row[off + 2];
                s3 += a_row[off + 3] * b_row[off + 3];
            }
            for p in (unrolled * 4)..k {
                s0 += a_row[p] * b_row[p];
            }
            c_row[j] = (s0 + s1) + (s2 + s3);
        }
    }
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
#[inline]
fn matmul_bt_scalar_m1(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
    let a_row = &a[..k];
    let unrolled8 = k / 8;
    for j in 0..n {
        let b_row = &b[j * k..(j + 1) * k];
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut s4 = 0.0f32;
        let mut s5 = 0.0f32;
        let mut s6 = 0.0f32;
        let mut s7 = 0.0f32;
        for p in 0..unrolled8 {
            let off = p * 8;
            s0 += a_row[off] * b_row[off];
            s1 += a_row[off + 1] * b_row[off + 1];
            s2 += a_row[off + 2] * b_row[off + 2];
            s3 += a_row[off + 3] * b_row[off + 3];
            s4 += a_row[off + 4] * b_row[off + 4];
            s5 += a_row[off + 5] * b_row[off + 5];
            s6 += a_row[off + 6] * b_row[off + 6];
            s7 += a_row[off + 7] * b_row[off + 7];
        }
        for p in (unrolled8 * 8)..k {
            s0 += a_row[p] * b_row[p];
        }
        c[j] = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- release-active bounds guards (#368) ---

    /// A too-short `b` must panic in both debug AND release builds (release-active assert).
    /// m=1, k=2, n=2: b must have n*k=4 elements; passing [] triggers the guard.
    #[test]
    #[should_panic(expected = "too short for n*k")]
    fn matmul_bt_short_b_panics_in_release() {
        let a = [0.0f32; 2]; // a.len() = m*k = 1*2 = 2 ✓
        let b: [f32; 0] = []; // b.len() = 0 < n*k = 4  ✗
        let mut c = [0.0f32; 2];
        matmul_bt(&a, &b, &mut c, 1, 2, 2);
    }

    /// A shape product that would overflow usize must panic before any memory access.
    /// m=2, k=usize::MAX, n=2: m*k overflows on the first overflow check.
    #[test]
    #[should_panic(expected = "shape overflow")]
    fn matmul_shape_overflow_panics() {
        let a = [0.0f32; 2];
        let b = [0.0f32; 2];
        let mut c = [0.0f32; 1];
        matmul_bt(&a, &b, &mut c, 2, usize::MAX, 2);
    }

    /// Callers may pass reused scratch buffers whose length EXCEEDS the exact footprint.
    /// `>=` is the correct check: this must NOT panic and must produce correct results
    /// in c[0..m*n]. The content of extra slots is unspecified across platforms.
    #[test]
    fn matmul_bt_oversized_c_does_not_panic() {
        // m=1, k=2, n=2: result c is m*n=2 elements. Pass c of length 3 (one extra).
        // a = [1, 2], b = [[1, 0], [0, 1]] stored row-major (transposed B).
        // c[0] = dot(a, b_row0) = 1*1 + 2*0 = 1
        // c[1] = dot(a, b_row1) = 1*0 + 2*1 = 2
        let a = [1.0f32, 2.0];
        let b = [1.0f32, 0.0, 0.0, 1.0]; // n=2 rows of k=2
        let mut c = [0.0f32; 3]; // intentionally oversized (3 > m*n=2)
        matmul_bt(&a, &b, &mut c, 1, 2, 2);
        assert!(
            (c[0] - 1.0).abs() < 1e-6,
            "c[0] should be 1.0, got {}",
            c[0]
        );
        assert!(
            (c[1] - 2.0).abs() < 1e-6,
            "c[1] should be 2.0, got {}",
            c[1]
        );
        // Extra slot c[2]: content is platform-defined, we only guarantee no panic and
        // correct values in c[0..m*n]. This test proves the >= bound is correct.
    }

    // A zero dimension is an empty GEMM, not a shape error: Accelerate aborts the process on a
    // zero leading dimension, so each case runs through the public entry point on every backend.
    // The result region is c[..m*n]; the suffix beyond it is not part of the contract (see the
    // `matmul_bt` doc), so these tests assert nothing about it. The sentinel only proves that a
    // nonempty result region is overwritten.
    const SENTINEL: f32 = 7.5;

    #[test]
    fn matmul_bt_zero_rows_returns() {
        let (m, k, n) = (0usize, 3usize, 2usize);
        let b = [1.0f32; 6];
        let mut c = [SENTINEL; 4];
        matmul_bt(&[], &b, &mut c, m, k, n);
        // m*n == 0: the result region is empty, so the property is that the call returns.
    }

    #[test]
    fn matmul_bt_zero_columns_returns() {
        let (m, k, n) = (2usize, 3usize, 0usize);
        let a = [1.0f32; 6];
        let mut c = [SENTINEL; 4];
        matmul_bt(&a, &[], &mut c, m, k, n);
        // m*n == 0: the result region is empty, so the property is that the call returns.
    }

    #[test]
    fn matmul_bt_zero_inner_dimension_zero_fills_output() {
        let (m, k, n) = (3usize, 0usize, 2usize);
        let mut c = [SENTINEL; 9];
        matmul_bt(&[], &[], &mut c, m, k, n);
        assert!(
            c[..m * n].iter().all(|&v| v == 0.0),
            "k=0 must zero c[..m*n], got {:?}",
            &c[..m * n]
        );
    }

    #[test]
    fn matmul_bt_zero_inner_dimension_single_row_zero_fills_output() {
        let mut c = [SENTINEL; 5];
        matmul_bt(&[], &[], &mut c, 1, 0, 4);
        assert!(c[..4].iter().all(|&v| v == 0.0));
    }

    #[cfg(not(target_os = "macos"))]
    fn lcg_vec(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32) * 0.04 - 0.02
            })
            .collect()
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn every_small_m_matches_scalar_reference() {
        let (k, n) = (256usize, 1024usize);
        let b = lcg_vec(n * k, 0x0E55);
        for m in 1..=9usize {
            let a = lcg_vec(m * k, 0x0F66 + m as u32);
            let mut got = vec![0.0f32; m * n];
            let mut want = vec![0.0f32; m * n];
            matmul_bt(&a, &b, &mut got, m, k, n);
            matmul_bt_scalar(&a, &b, &mut want, m, k, n);
            for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 1e-4, "m={m} idx={idx}: got {g}, want {w}");
            }
        }
    }

    // --- x86_64: small-m rows must not run through the tiled kernel's scalar edge loop ---

    #[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
    mod small_m_dispatch {
        use super::*;

        fn bits(v: &[f32]) -> Vec<u32> {
            v.iter().map(|x| x.to_bits()).collect()
        }

        /// m=1 with n*k >= 1M: the whole call must equal column chunks that are each below
        /// the tiled threshold (direct kernel). The direct kernels compute every output
        /// column from its own A row and B row only, so the two are bit-identical.
        #[test]
        fn m1_large_matches_direct_kernel_per_column_chunk() {
            let (m, k, n) = (1usize, 1024usize, 2048usize);
            let chunk = 512usize;
            assert!((m * k * chunk) < 1024 * 1024 && (m * k * n) >= 1024 * 1024);
            let a = lcg_vec(m * k, 0x0A11);
            let b = lcg_vec(n * k, 0x0B22);

            let mut whole = vec![0.0f32; m * n];
            matmul_bt(&a, &b, &mut whole, m, k, n);

            let mut chunked = vec![0.0f32; m * n];
            for j0 in (0..n).step_by(chunk) {
                matmul_bt(
                    &a,
                    &b[j0 * k..(j0 + chunk) * k],
                    &mut chunked[j0..j0 + chunk],
                    m,
                    k,
                    chunk,
                );
            }
            assert_eq!(bits(&whole), bits(&chunked));
        }

        /// m=5 with work >= 1M: rows 0..4 take the tiled kernel exactly as a 4-row call
        /// does, and row 4 takes the direct kernel exactly as a 1-row call does.
        #[test]
        #[cfg_attr(
            all(feature = "x86-gemm", target_arch = "x86_64", not(target_os = "macos")),
            ignore = "x86-gemm runs multi-row products through gemm and single rows through the existing kernel, so they agree to about 1e-7, not bit for bit"
        )]
        fn m5_splits_into_tiled_rows_and_direct_row() {
            let (m, k, n) = (5usize, 1024usize, 512usize);
            let a = lcg_vec(m * k, 0x0C33);
            let b = lcg_vec(n * k, 0x0D44);

            let mut whole = vec![0.0f32; m * n];
            matmul_bt(&a, &b, &mut whole, m, k, n);

            let mut head = vec![0.0f32; 4 * n];
            matmul_bt(&a[..4 * k], &b, &mut head, 4, k, n);
            assert_eq!(bits(&whole[..4 * n]), bits(&head));

            let mut tail = vec![0.0f32; n];
            matmul_bt(&a[4 * k..], &b, &mut tail, 1, k, n);
            assert_eq!(bits(&whole[4 * n..]), bits(&tail));
        }

        /// A `c` longer than m*n keeps its suffix untouched, for m < TILE_I and for an m
        /// with a remainder, at a size where the tiled path is selected.
        #[test]
        fn oversized_c_suffix_is_untouched() {
            let (k, n) = (1024usize, 1024usize);
            let b = lcg_vec(n * k, 0x1077);
            let sentinel = f32::from_bits(0x7FC0_BEEF);
            for m in [1usize, 3, 4, 5, 7] {
                let a = lcg_vec(m * k, 0x1188 + m as u32);
                let extra = 16usize;
                let mut c = vec![sentinel; m * n + extra];
                matmul_bt(&a, &b, &mut c, m, k, n);
                assert!(
                    c[m * n..].iter().all(|x| x.to_bits() == sentinel.to_bits()),
                    "m={m}: suffix beyond m*n was written"
                );
                assert!(
                    c[..m * n].iter().all(|x| x.is_finite()),
                    "m={m}: result region was not fully written"
                );
            }
        }
    }

    // --- x86_64: row-split dispatch ---

    #[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
    mod row_threads {
        use super::*;

        fn bits(v: &[f32]) -> Vec<u32> {
            v.iter().map(|x| x.to_bits()).collect()
        }

        fn serial(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
            let mut c = vec![0.0f32; m * n];
            matmul_bt_x86_rows(a, b, &mut c, m, k, n, tiled_applies(m, k, n));
            c
        }

        fn in_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("rayon pool")
                .install(f)
        }

        /// The split result and how the dispatcher reports having run it.
        fn threaded(
            a: &[f32],
            b: &[f32],
            m: usize,
            k: usize,
            n: usize,
            min_work: u64,
        ) -> (Vec<f32>, Split) {
            in_pool(4, || {
                let mut c = vec![0.0f32; m * n];
                let used = matmul_bt_x86(a, b, &mut c, m, k, n, min_work, COL_THREAD_MIN_WORK);
                (c, used)
            })
        }

        fn assert_bit_identical(m: usize, k: usize, n: usize) {
            let a = lcg_vec(m * k, 0x2A11 + m as u32);
            let b = lcg_vec(n * k, 0x2B22 + n as u32);
            let (got, used) = threaded(&a, &b, m, k, n, 0);
            assert!(
                matches!(used, Split::Rows(_)),
                "m={m} k={k} n={n}: the split must engage"
            );
            assert_eq!(
                used,
                row_chunk_len(m, k, n, 4, 0).map_or(Split::Serial, Split::Rows),
                "m={m} k={k} n={n}"
            );
            assert_eq!(
                bits(&got),
                bits(&serial(&a, &b, m, k, n)),
                "m={m} k={k} n={n}"
            );
        }

        #[test]
        fn row_chunk_len_decides_the_split() {
            let min = ROW_THREAD_MIN_WORK;
            assert_eq!(min, 2 * 16 * 1024 * 3584);
            // No threads, one thread, too few rows, too little work.
            assert_eq!(row_chunk_len(64, 1024, 3584, 0, min), None);
            assert_eq!(row_chunk_len(64, 1024, 3584, 1, min), None);
            assert_eq!(row_chunk_len(31, 1024, 3584, 8, 0), None);
            assert_eq!(row_chunk_len(16, 1024, 3584, 8, min), None);
            assert_eq!(row_chunk_len(64, 1024, 1024, 8, min), None);
            // Empty shared or output dimension cannot be chunked.
            assert_eq!(row_chunk_len(64, 0, 64, 8, 0), None);
            assert_eq!(row_chunk_len(64, 64, 0, 8, 0), None);
            // The work threshold is inclusive: 32 x 1024 x 3584 is exactly min_work.
            assert_eq!(row_chunk_len(32, 1024, 3584, 8, min), Some(16));
            assert_eq!(row_chunk_len(32, 1024, 3583, 8, min), None);
            // Prefill shape: ceil(64 / 8) = 8 is raised to the 16-row floor.
            assert_eq!(row_chunk_len(64, 1024, 3584, 8, min), Some(16));
            // ceil(m / threads) that is not a multiple of TILE_I is rounded up.
            assert_eq!(row_chunk_len(100, 64, 8, 3, 0), Some(36));
            assert_eq!(row_chunk_len(200, 64, 8, 4, 0), Some(52));
            assert_eq!(row_chunk_len(512, 1024, 3584, 8, min), Some(64));
        }

        #[test]
        fn row_chunk_len_is_tile_aligned_and_leaves_several_chunks() {
            for m in 32..400usize {
                for threads in 2..=16usize {
                    if let Some(rows) = row_chunk_len(m, 64, 8, threads, 0) {
                        assert_eq!(rows % crate::forward::cpu::tiled::TILE_I, 0);
                        assert!(rows >= 16, "m={m} threads={threads}: rows={rows}");
                        assert!(rows < m, "m={m} threads={threads}: rows={rows}");
                        assert!(
                            m.div_ceil(rows) <= threads,
                            "m={m} threads={threads}: rows={rows}"
                        );
                    } else {
                        panic!("m={m} threads={threads}: expected a split");
                    }
                }
            }
        }

        /// k below and at/above TILE_K, with m values that leave 0..3 remainder rows and
        /// a short last chunk.
        #[test]
        fn threaded_matches_serial_bit_for_bit() {
            for m in [32usize, 33, 37, 64, 130] {
                for k in [64usize, 256] {
                    for n in [8usize, 37] {
                        assert_bit_identical(m, k, n);
                    }
                }
            }
        }

        /// Shapes at which the tiled kernel is selected for the whole product, so the
        /// chunks really run it.
        #[test]
        fn threaded_matches_serial_bit_for_bit_when_tiled_kernel_applies() {
            for (m, k, n) in [
                (33usize, 512usize, 64usize),
                (36, 256, 128),
                (37, 256, 128),
                (64, 256, 64),
                (130, 256, 37),
            ] {
                assert!(tiled_applies(m, k, n), "m={m} k={k} n={n}");
                assert_bit_identical(m, k, n);
            }
        }

        /// The last chunk is 4 rows, below the tiled work gate on its own while the whole
        /// product is above it: those 4 rows must still take the tiled kernel.
        #[test]
        fn short_last_chunk_keeps_the_whole_products_kernel_choice() {
            let (m, k, n) = (36usize, 256usize, 128usize);
            let rows = row_chunk_len(m, k, n, 4, 0).expect("split");
            let last = m - (m.div_ceil(rows) - 1) * rows;
            assert_eq!((rows, last), (16, 4));
            assert!(tiled_applies(m, k, n));
            assert!(!tiled_applies(last, k, n));
            assert_bit_identical(m, k, n);
        }

        /// The dispatcher itself, not just the predicate, stays serial below the work
        /// threshold and on a single-thread pool.
        #[test]
        fn dispatcher_runs_serially_unless_the_split_applies() {
            let (m, k, n) = (64usize, 256usize, 64usize);
            let a = lcg_vec(m * k, 0x2E55);
            let b = lcg_vec(n * k, 0x2F66);
            let want = serial(&a, &b, m, k, n);

            let (got, used) = threaded(&a, &b, m, k, n, ROW_THREAD_MIN_WORK);
            assert_eq!(used, Split::Serial, "below the work threshold");
            assert_eq!(bits(&got), bits(&want));

            let (got, used) = threaded(&a, &b, m, k, n, 0);
            assert_eq!(used, Split::Rows(16), "split forced by min_work 0");
            assert_eq!(bits(&got), bits(&want));

            let (got, used) = in_pool(1, || {
                let mut c = vec![0.0f32; m * n];
                let used = matmul_bt_x86(&a, &b, &mut c, m, k, n, 0, COL_THREAD_MIN_WORK);
                (c, used)
            });
            assert_eq!(used, Split::Serial, "one thread");
            assert_eq!(bits(&got), bits(&want));
        }

        #[test]
        fn threaded_matches_scalar_reference() {
            let (m, k, n) = (130usize, 256usize, 37usize);
            let a = lcg_vec(m * k, 0x2C33);
            let b = lcg_vec(n * k, 0x2D44);
            let (got, used) = threaded(&a, &b, m, k, n, 0);
            assert!(matches!(used, Split::Rows(_)));
            let mut want = vec![0.0f32; m * n];
            matmul_bt_scalar(&a, &b, &mut want, m, k, n);
            for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 1e-4, "idx={idx}: got {g}, want {w}");
            }
        }
    }

    // --- x86_64: single-row column-split dispatch ---

    #[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
    mod column_threads {
        use super::*;
        use rayon::prelude::*;

        fn bits(v: &[f32]) -> Vec<u32> {
            v.iter().map(|x| x.to_bits()).collect()
        }

        fn serial(a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
            let mut c = vec![0.0f32; n];
            matmul_bt_x86_rows(a, b, &mut c, 1, k, n, tiled_applies(1, k, n));
            c
        }

        fn in_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("rayon pool")
                .install(f)
        }

        /// The dispatcher called on this test thread, which is not a rayon worker, so the
        /// global pool's width decides the split.
        fn dispatch(
            a: &[f32],
            b: &[f32],
            k: usize,
            n: usize,
            col_min_work: u64,
        ) -> (Vec<f32>, Split) {
            assert_eq!(rayon::current_thread_index(), None);
            let mut c = vec![0.0f32; n];
            let used = matmul_bt_x86(a, b, &mut c, 1, k, n, ROW_THREAD_MIN_WORK, col_min_work);
            (c, used)
        }

        fn expected(k: usize, n: usize, col_min_work: u64) -> Split {
            col_chunk_len(1, k, n, rayon::current_num_threads(), false, col_min_work)
                .map_or(Split::Serial, Split::Columns)
        }

        #[test]
        fn col_chunk_len_decides_the_split() {
            let min = COL_THREAD_MIN_WORK;
            assert_eq!(min, 2 * 1024 * 256);
            // Only a single row splits by columns.
            assert_eq!(col_chunk_len(2, 1024, 4096, 8, false, 0), None);
            // No threads, one thread, a caller already on a rayon worker.
            assert_eq!(col_chunk_len(1, 1024, 4096, 0, false, 0), None);
            assert_eq!(col_chunk_len(1, 1024, 4096, 1, false, 0), None);
            assert_eq!(col_chunk_len(1, 1024, 4096, 8, true, 0), None);
            assert_eq!(col_chunk_len(1, 1024, 248_320, 8, true, min), None);
            // Empty shared or output dimension cannot be chunked.
            assert_eq!(col_chunk_len(1, 0, 4096, 8, false, 0), None);
            assert_eq!(col_chunk_len(1, 1024, 0, 8, false, 0), None);
            // The work threshold is inclusive: 1024 x 512 is exactly min_work.
            assert_eq!(col_chunk_len(1, 1024, 512, 8, false, min), Some(64));
            assert_eq!(col_chunk_len(1, 1024, 511, 8, false, min), None);
            // The largest shape measured to lose from the split stays serial.
            assert_eq!(col_chunk_len(1, 1024, 128, 8, false, min), None);
            // A chunk that would cover every column is no split.
            assert_eq!(col_chunk_len(1, 64, 16, 8, false, 0), None);
            assert_eq!(col_chunk_len(1, 64, 17, 8, false, 0), Some(16));
            // Output head: 248320 / 8 is already a multiple of 16.
            assert_eq!(col_chunk_len(1, 1024, 248_320, 8, false, min), Some(31_040));
            // ceil(n / threads) that is not a multiple of 16 is rounded up.
            assert_eq!(col_chunk_len(1, 64, 1001, 3, false, 0), Some(336));
        }

        #[test]
        fn col_chunk_len_is_line_aligned_and_fits_the_pool() {
            for n in 17..600usize {
                for threads in 2..=16usize {
                    let cols = col_chunk_len(1, 8, n, threads, false, 0)
                        .unwrap_or_else(|| panic!("n={n} threads={threads}: expected a split"));
                    assert_eq!(cols % COL_THREAD_MIN_CHUNK, 0, "n={n} threads={threads}");
                    assert!(cols < n, "n={n} threads={threads}: cols={cols}");
                    assert!(
                        n.div_ceil(cols) <= threads,
                        "n={n} threads={threads}: cols={cols}"
                    );
                }
            }
        }

        /// Odd n, a last chunk shorter than 16 columns, k below the SIMD width, odd k, and a
        /// product large enough that the tiled kernel is selected for the whole call.
        #[test]
        fn column_split_matches_serial_bit_for_bit() {
            let cases = [
                (7usize, 50usize, 16usize),
                (64, 37, 16),
                (256, 1003, 16),
                (1031, 333, 48),
                (1024, 2049, 528),
            ];
            assert!(
                cases
                    .iter()
                    .any(|&(_, n, cols)| (1..16).contains(&(n % cols)))
            );
            assert!(tiled_applies(1, 1024, 2049));
            for (k, n, cols) in cases {
                let a = lcg_vec(k, 0x3A11 + k as u32);
                let b = lcg_vec(n * k, 0x3B22 + n as u32);
                let got = in_pool(4, || {
                    let mut c = vec![0.0f32; n];
                    matmul_bt_x86_columns(&a, &b, &mut c, k, n, cols, tiled_applies(1, k, n));
                    c
                });
                assert_eq!(
                    bits(&got),
                    bits(&serial(&a, &b, k, n)),
                    "k={k} n={n} cols={cols}"
                );
            }
        }

        #[test]
        fn column_split_matches_scalar_reference() {
            let (k, n) = (256usize, 1003usize);
            let a = lcg_vec(k, 0x3C33);
            let b = lcg_vec(n * k, 0x3D44);
            let got = in_pool(4, || {
                let mut c = vec![0.0f32; n];
                matmul_bt_x86_columns(&a, &b, &mut c, k, n, 16, tiled_applies(1, k, n));
                c
            });
            let mut want = vec![0.0f32; n];
            matmul_bt_scalar(&a, &b, &mut want, 1, k, n);
            for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 1e-4, "idx={idx}: got {g}, want {w}");
            }
        }

        /// Outside the pool the dispatcher splits a single row by columns at and above the
        /// threshold, and the result equals the serial row.
        #[test]
        fn dispatcher_splits_a_single_row_outside_the_pool() {
            for (k, n, min) in [
                (256usize, 1003usize, 0u64),
                (1024, 512, COL_THREAD_MIN_WORK),
            ] {
                let a = lcg_vec(k, 0x3E55 + n as u32);
                let b = lcg_vec(n * k, 0x3F66 + n as u32);
                let (got, used) = dispatch(&a, &b, k, n, min);
                assert_eq!(used, expected(k, n, min), "k={k} n={n}");
                if rayon::current_num_threads() >= 2 {
                    assert!(matches!(used, Split::Columns(_)), "k={k} n={n}: {used:?}");
                }
                assert_eq!(bits(&got), bits(&serial(&a, &b, k, n)), "k={k} n={n}");
            }
        }

        /// Below the threshold the dispatcher stays serial.
        #[test]
        fn dispatcher_stays_serial_below_the_column_threshold() {
            for (k, n) in [(1024usize, 511usize), (1024, 128)] {
                assert!(((n * k) as u64) < COL_THREAD_MIN_WORK);
                let a = lcg_vec(k, 0x4A11 + n as u32);
                let b = lcg_vec(n * k, 0x4B22 + n as u32);
                let (got, used) = dispatch(&a, &b, k, n, COL_THREAD_MIN_WORK);
                assert_eq!(used, Split::Serial, "k={k} n={n}");
                assert_eq!(bits(&got), bits(&serial(&a, &b, k, n)), "k={k} n={n}");
            }
        }

        /// A caller already on a rayon worker, directly under `install` or as one of several
        /// parallel tasks, never gets a column split, even with no work threshold.
        #[test]
        fn dispatcher_does_not_split_columns_inside_a_rayon_worker() {
            let (k, n) = (256usize, 1003usize);
            let a = lcg_vec(k, 0x4C33);
            let b = lcg_vec(n * k, 0x4D44);
            let want = bits(&serial(&a, &b, k, n));

            let (got, used) = in_pool(4, || {
                let mut c = vec![0.0f32; n];
                let used = matmul_bt_x86(&a, &b, &mut c, 1, k, n, ROW_THREAD_MIN_WORK, 0);
                (c, used)
            });
            assert_eq!(used, Split::Serial, "under install");
            assert_eq!(bits(&got), want);

            let tasks: Vec<(Vec<f32>, Split)> = in_pool(4, || {
                (0..4)
                    .into_par_iter()
                    .map(|_| {
                        let mut c = vec![0.0f32; n];
                        let used = matmul_bt_x86(&a, &b, &mut c, 1, k, n, ROW_THREAD_MIN_WORK, 0);
                        (c, used)
                    })
                    .collect()
            });
            for (got, used) in tasks {
                assert_eq!(used, Split::Serial, "inside a parallel iterator");
                assert_eq!(bits(&got), want);
            }
        }
    }
}

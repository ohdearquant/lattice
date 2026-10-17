use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

const OUTPUT_POISON: f32 = f32::from_bits(0x7fc0_0031);
const ATOL: f64 = 2e-5;
const RTOL: f64 = 2e-5;

fn matrix(n: usize, k: usize) -> Q8Matrix {
    Q8Matrix {
        data: (0..n * k)
            .map(|i| ((i * 37 + i / k.max(1) * 19 + 11) % 255) as i16 - 127)
            .map(|value| value as i8)
            .collect(),
        scales: (0..n).map(|row| 0.001 * (row % 7 + 1) as f32).collect(),
        rows: n,
        cols: k,
    }
}

fn activations(m: usize, k: usize) -> Vec<f32> {
    let mut input: Vec<f32> = (0..m * k)
        .map(|i| ((i * 13 + i / k.max(1) * 7 + 5) % 29) as f32 / 17.0 - 0.8)
        .collect();
    input.extend([f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
    input
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn independent_f64(a: &[f32], b: &Q8Matrix, m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut output = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f64;
            for inner in 0..k {
                sum += f64::from(a[row * k + inner]) * f64::from(b.data[col * k + inner]);
            }
            output[row * n + col] = sum * f64::from(b.scales[col]);
        }
    }
    output
}

fn assert_finite_close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "nonfinite lane {i}: {actual} vs {expected}"
        );
        let error = (f64::from(actual) - expected).abs();
        assert!(
            error <= ATOL + RTOL * expected.abs(),
            "lane {i}: {actual} vs {expected}"
        );
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    storage: [[usize; 3]; 2],
    contents: [Vec<u32>; 2],
    calls: usize,
}

fn snapshot(scratch: &Q8MatmulScratch) -> Snapshot {
    #[cfg(target_os = "macos")]
    let contents = [bits(&scratch.b_f32), bits(&scratch.c_tile)];
    #[cfg(not(target_os = "macos"))]
    let contents = [Vec::new(), Vec::new()];
    Snapshot {
        storage: scratch.storage_snapshot(),
        contents,
        calls: scratch.call_count(),
    }
}

fn compare_shape(m: usize, k: usize, n: usize, scratch: &mut Q8MatmulScratch) {
    let a = activations(m, k);
    let b = matrix(n, k);
    let active = m * n;
    let mut actual = vec![OUTPUT_POISON; active + 7];
    let mut frozen = actual.clone();
    let mut compatibility = actual.clone();
    let before_calls = scratch.call_count();
    matmul_bt_q8_with_scratch(&a, &b, &mut actual, m, k, n, scratch);
    old_matmul_bt_q8(&a, &b, &mut frozen, m, k, n);
    matmul_bt_q8(&a, &b, &mut compatibility, m, k, n);
    assert_eq!(scratch.call_count(), before_calls + 1);
    let independent = independent_f64(&a, &b, m, k, n);
    assert_finite_close(&actual[..active], &independent);
    assert_finite_close(&frozen[..active], &independent);
    assert_eq!(
        bits(&actual[..active]),
        bits(&frozen[..active]),
        "old/new m={m} k={k} n={n}"
    );
    assert_eq!(bits(&actual[..active]), bits(&compatibility[..active]));
    for output in [&actual, &frozen, &compatibility] {
        assert!(
            output[active..]
                .iter()
                .all(|v| v.to_bits() == OUTPUT_POISON.to_bits())
        );
    }
    if m > 0 && k > 0 && n > 0 {
        assert!(independent.iter().any(|value| value.abs() > 1e-4));
    }
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(ToString::to_string))
        .expect("shape refusal must carry a string")
}

struct RefusalCase {
    a: Vec<f32>,
    b: Q8Matrix,
    m: usize,
    k: usize,
    n: usize,
    c_len: usize,
    expected: &'static str,
}

fn assert_refusal(case: RefusalCase) {
    let mut scratch = Q8MatmulScratch::default();
    scratch.prepare(2, 3).unwrap();
    scratch.poison();
    let before = snapshot(&scratch);
    let mut actual = vec![OUTPUT_POISON; case.c_len];
    let mut frozen = actual.clone();
    let original = bits(&actual);
    let error = catch_unwind(AssertUnwindSafe(|| {
        matmul_bt_q8_with_scratch(
            &case.a,
            &case.b,
            &mut actual,
            case.m,
            case.k,
            case.n,
            &mut scratch,
        );
    }))
    .expect_err("reused kernel must refuse invalid shape");
    assert!(panic_text(error).contains(case.expected));
    let error = catch_unwind(AssertUnwindSafe(|| {
        old_matmul_bt_q8(&case.a, &case.b, &mut frozen, case.m, case.k, case.n);
    }))
    .expect_err("frozen kernel must refuse invalid shape");
    assert!(panic_text(error).contains(case.expected));
    assert_eq!(bits(&actual), original);
    assert_eq!(bits(&frozen), original);
    assert_eq!(snapshot(&scratch), before);
}

#[test]
fn reused_q8_matches_frozen_kernel_and_independent_oracle() {
    for m in [1, 2, 3] {
        for n in [1, 63, 64, 65, 129] {
            for k in [0, 1, 3, 31, 33, 65] {
                compare_shape(m, k, n, &mut Q8MatmulScratch::default());
            }
        }
    }
    for (m, k, n) in [(0, 3, 65), (2, 3, 0), (0, 0, 0)] {
        compare_shape(m, k, n, &mut Q8MatmulScratch::default());
    }
}

#[test]
fn reused_q8_preserves_exact_nonzero_known_values() {
    let a = [1.0, -0.5, 2.0, 3.0, -2.0, 0.5];
    let b = Q8Matrix {
        data: vec![1, 2, 3, -2, 1, -1],
        scales: vec![0.25, 0.5],
        rows: 2,
        cols: 3,
    };
    let expected = [1.5f32, -2.25, 0.125, -4.25];
    let mut scratch = Q8MatmulScratch::default();
    for m in [1, 2] {
        let mut actual = vec![OUTPUT_POISON; m * 2];
        let mut old = actual.clone();
        matmul_bt_q8_with_scratch(&a, &b, &mut actual, m, 3, 2, &mut scratch);
        old_matmul_bt_q8(&a, &b, &mut old, m, 3, 2);
        assert_eq!(bits(&actual), bits(&expected[..m * 2]));
        assert_eq!(bits(&actual), bits(&old));
    }
}

#[test]
fn reused_q8_overwrites_poisoned_scratch_and_preserves_output_tails() {
    let mut scratch = Q8MatmulScratch::default();
    scratch.prepare(3, 65).unwrap();
    let storage = scratch.storage_snapshot();
    for (m, k, n) in [
        (1, 65, 129),
        (3, 3, 63),
        (2, 33, 65),
        (1, 31, 64),
        (3, 0, 1),
    ] {
        scratch.poison();
        compare_shape(m, k, n, &mut scratch);
        assert_eq!(scratch.storage_snapshot(), storage);
    }
    assert_eq!(scratch.call_count(), 5);
}

#[test]
fn reused_q8_preserves_all_common_shape_refusals_before_mutation() {
    for (m, k, n, expected) in [
        (usize::MAX, 2, 1, "matmul shape overflow: m*k"),
        (1, 2, usize::MAX, "matmul shape overflow: n*k"),
        (usize::MAX, 1, 2, "matmul shape overflow: m*n"),
    ] {
        assert_refusal(RefusalCase {
            a: vec![0.5, -0.25],
            b: matrix(2, 2),
            m,
            k,
            n,
            c_len: 4,
            expected,
        });
    }
    for fault in 0..6 {
        let mut a = activations(2, 3);
        a.truncate(6);
        let mut b = matrix(2, 3);
        let mut c_len = 4;
        let expected = match fault {
            0 => {
                a.truncate(5);
                "A length does not match m * k"
            }
            1 => {
                b.rows = 3;
                "B_q rows do not match n"
            }
            2 => {
                b.cols = 4;
                "B_q cols do not match k"
            }
            3 => {
                b.data.truncate(5);
                "B_q data length does not match n * k"
            }
            4 => {
                b.scales.truncate(1);
                "B_q scales length does not match n"
            }
            5 => {
                c_len = 3;
                "C buffer is too small"
            }
            _ => unreachable!(),
        };
        assert_refusal(RefusalCase {
            a,
            b,
            m: 2,
            k: 3,
            n: 2,
            c_len,
            expected,
        });
    }
}

#[test]
fn reused_q8_counts_successful_calls_exactly_once() {
    let mut scratch = Q8MatmulScratch::default();
    assert_eq!(scratch.call_count(), 0);
    scratch.prepare(2, 33).unwrap();
    assert_eq!(scratch.call_count(), 0);
    for (index, (m, k, n)) in [(1, 3, 65), (2, 33, 129), (1, 1, 1)]
        .into_iter()
        .enumerate()
    {
        compare_shape(m, k, n, &mut scratch);
        assert_eq!(scratch.call_count(), index + 1);
    }
}

#[test]
fn q8_scratch_comparison_rejects_nonfinite_operands() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(catch_unwind(|| assert_finite_close(&[value], &[0.0])).is_err());
        assert!(catch_unwind(|| assert_finite_close(&[0.0], &[f64::from(value)])).is_err());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn mac_q8_prepare_grows_both_buffers_without_shrinking() {
    let mut scratch = Q8MatmulScratch::default();
    scratch.prepare(2, 33).unwrap();
    let first = scratch.storage_snapshot();
    assert_eq!([first[0][1], first[1][1]], [2_112, 128]);
    assert!(first.iter().all(|field| field[2] >= field[1]));
    scratch.prepare(3, 65).unwrap();
    let grown = scratch.storage_snapshot();
    assert_eq!([grown[0][1], grown[1][1]], [4_160, 192]);
    assert!(grown.iter().all(|field| field[2] >= field[1]));
    scratch.poison();
    let high_water = snapshot(&scratch);
    scratch.prepare(1, 3).unwrap();
    assert_eq!(snapshot(&scratch), high_water);
}

#[cfg(target_os = "macos")]
#[test]
fn mac_q8_prepare_rejects_unaddressable_extents_before_mutation() {
    let mut scratch = Q8MatmulScratch::default();
    scratch.prepare(2, 3).unwrap();
    scratch.poison();
    let before = snapshot(&scratch);
    let too_many_floats = isize::MAX as usize / std::mem::size_of::<f32>() + 1;
    let byte_limit_dimension = too_many_floats.div_ceil(64);
    for (m, k, reason) in [
        (1, usize::MAX, "TILE_N*k"),
        (usize::MAX, 16, "m*TILE_N"),
        (1, byte_limit_dimension, "byte extent"),
        (byte_limit_dimension, 16, "byte extent"),
    ] {
        let error = scratch
            .prepare(m, k)
            .expect_err("unaddressable storage must refuse");
        assert!(matches!(&error, InferenceError::InvalidInput(_)));
        assert!(error.to_string().contains(reason));
        assert_eq!(snapshot(&scratch), before);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn mac_q8_tile_overflow_guards_match_frozen_kernel() {
    for (m, k, reason) in [
        (0, usize::MAX, "matmul shape overflow: TILE_N*k"),
        (usize::MAX, 0, "matmul shape overflow: m*TILE_N"),
    ] {
        assert_refusal(RefusalCase {
            a: Vec::new(),
            b: Q8Matrix {
                data: Vec::new(),
                scales: Vec::new(),
                rows: 0,
                cols: k,
            },
            m,
            k,
            n: 0,
            c_len: 0,
            expected: reason,
        });
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn nonmac_q8_prepare_ignores_unused_tile_bounds_without_storage() {
    let mut scratch = Q8MatmulScratch::default();
    for (m, k) in [
        (0, 0),
        (1, 3),
        (usize::MAX, 1),
        (1, usize::MAX),
        (usize::MAX, usize::MAX),
    ] {
        let before = snapshot(&scratch);
        scratch.prepare(m, k).unwrap();
        assert_eq!(scratch.storage_snapshot(), [[0; 3]; 2]);
        assert_eq!(snapshot(&scratch), before);
    }
    compare_shape(1, 3, 65, &mut scratch);
    assert_eq!(scratch.storage_snapshot(), [[0; 3]; 2]);
    assert_eq!(scratch.call_count(), 1);
}

pub fn old_matmul_bt_q8(a: &[f32], b_q: &Q8Matrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    // Guard the dimension products against usize overflow BEFORE any `m * k` /
    // `n * k` / `m * n` is evaluated below: in release those multiplies wrap
    // silently, so an overflowing shape could pass a length check and drive an
    // out-of-bounds access. Mirrors the same guards in `matmul_bt`
    // (forward/cpu/matmul.rs); `matmul_bt_q8` was missing them.
    assert!(m.checked_mul(k).is_some(), "matmul shape overflow: m*k");
    assert!(n.checked_mul(k).is_some(), "matmul shape overflow: n*k");
    assert!(m.checked_mul(n).is_some(), "matmul shape overflow: m*n");
    // Oversized-scratch-prefix allow-list (ADR-080 C4): `>=`, not `assert_eq!`. Every access
    // to `a` below is a bounded sub-slice (`a[i*k..(i+1)*k]`), so a caller-supplied `a`
    // longer than `m*k` is sound — unifies with `matmul_bt`/`matmul_bt_f16`'s contract
    // instead of this Q8 path alone rejecting valid oversized callers. `b_q.rows`/`b_q.cols`
    // are the matrix's own declared shape (not a raw buffer length), so those stay an exact
    // shape-mismatch check, not a buffer-size check.
    assert!(a.len() >= m * k, "A length does not match m * k");
    assert_eq!(b_q.rows, n, "B_q rows do not match n");
    assert_eq!(b_q.cols, k, "B_q cols do not match k");
    assert_eq!(
        b_q.data.len(),
        n * k,
        "B_q data length does not match n * k"
    );
    assert_eq!(b_q.scales.len(), n, "B_q scales length does not match n");
    assert!(c.len() >= m * n, "C buffer is too small");

    // On macOS we tile the N (output-channel) dimension, dequantize each tile
    // of B rows from i8→f32 (applying per-row scales), then delegate to the
    // Accelerate-backed matmul_bt which hits the AMX coprocessor.
    //
    // Tile sizing: 64 rows * 2048 cols * 4 bytes = 512 KB fits comfortably in
    // the M-series 16 MB L2 cache. For M=1 (decode) we write tile results
    // directly into c; for M>1 we scatter from a contiguous tile output buffer.
    #[cfg(target_os = "macos")]
    {
        const TILE_N: usize = 64;
        // The tile scratch buffers are sized by `TILE_N * k` and `m * TILE_N`,
        // which are NOT covered by the `m*k` / `n*k` / `m*n` guards above: when
        // `n < TILE_N` (including `n == 0`) these products can overflow while the
        // guarded ones do not, and a wrapped size would under-allocate the
        // scratch and drive an OOB scatter below. Guard them on the same
        // fail-closed contract.
        assert!(
            TILE_N.checked_mul(k).is_some(),
            "matmul shape overflow: TILE_N*k"
        );
        assert!(
            m.checked_mul(TILE_N).is_some(),
            "matmul shape overflow: m*TILE_N"
        );
        let mut b_f32 = vec![0.0f32; TILE_N * k];
        let mut c_tile = vec![0.0f32; m * TILE_N];

        for tile_start in (0..n).step_by(TILE_N) {
            let tile_n = (n - tile_start).min(TILE_N);

            // Dequantize: convert i8 -> f32 and fold in the per-row scale.
            for j in 0..tile_n {
                let global_j = tile_start + j;
                let q_row = &b_q.data[global_j * k..(global_j + 1) * k];
                let scale = b_q.scales[global_j];
                let dst = &mut b_f32[j * k..(j + 1) * k];
                for t in 0..k {
                    dst[t] = q_row[t] as f32 * scale;
                }
            }

            // Accelerate BLAS GEMM: C_tile[m, tile_n] = A[m, k] @ B_tile[tile_n, k]^T
            crate::forward::cpu::matmul_bt(
                a,
                &b_f32[..tile_n * k],
                &mut c_tile[..m * tile_n],
                m,
                k,
                tile_n,
            );

            // Scatter tile results into correct columns of c.
            if m == 1 {
                // Fast path for decode: c is [1, n], tile writes a contiguous run.
                c[tile_start..tile_start + tile_n].copy_from_slice(&c_tile[..tile_n]);
            } else {
                for i in 0..m {
                    let c_row = &mut c[i * n + tile_start..i * n + tile_start + tile_n];
                    let tile_row = &c_tile[i * tile_n..(i + 1) * tile_n];
                    c_row.copy_from_slice(tile_row);
                }
            }
        }
    }

    // Scalar fallback for non-macOS platforms.
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row = &mut c[i * n..(i + 1) * n];

            for j in 0..n {
                let q_row = &b_q.data[j * k..(j + 1) * k];
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += a_row[t] * q_row[t] as f32;
                }
                c_row[j] = acc * b_q.scales[j];
            }
        }
    }
}

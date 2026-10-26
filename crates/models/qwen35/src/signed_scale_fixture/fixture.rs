use super::{Q4_BLOCK_WEIGHTS, Q4Block, Q4Tensor, q4_f16_to_f32, q4_f32_to_f16};

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }

    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// A `[rows, cols]` tensor whose scales lie in `-[0.05, 0.2]` and biases in
/// `[-0.5, 0.5]`, with random codes and code 15 at weight 1 of every block, so a
/// dropped or clamped sign moves at least one weight per block by `15 * 0.05`.
pub(crate) fn negative_scale_tensor(rows: usize, cols: usize, seed: u64) -> Q4Tensor {
    assert!(rows > 0 && cols > 0 && cols.is_multiple_of(Q4_BLOCK_WEIGHTS));
    let mut rng = Lcg(seed);
    let blocks = (0..rows * cols / Q4_BLOCK_WEIGHTS)
        .map(|_| {
            let scale = q4_f32_to_f16(-(0.05 + 0.15 * rng.unit()));
            let bias = q4_f32_to_f16(rng.unit() - 0.5);
            let mut packed = [0u8; 16];
            for byte in &mut packed {
                *byte = rng.next_u32() as u8;
            }
            packed[0] |= 0xF0;
            Q4Block {
                scale,
                bias,
                packed,
            }
        })
        .collect();
    Q4Tensor {
        blocks,
        shape: vec![rows, cols],
        original_len: rows * cols,
    }
}

/// The same blocks with every scale bit pattern replaced by `f(scale_bits)`.
pub(crate) fn map_scales(tensor: &Q4Tensor, f: impl Fn(u16) -> u16) -> Q4Tensor {
    Q4Tensor {
        blocks: tensor
            .blocks
            .iter()
            .map(|b| Q4Block {
                scale: f(b.scale),
                ..*b
            })
            .collect(),
        shape: tensor.shape.clone(),
        original_len: tensor.original_len,
    }
}

/// What a consumer that drops the scale's sign computes.
pub(crate) fn abs_scales(tensor: &Q4Tensor) -> Q4Tensor {
    map_scales(tensor, |bits| bits & 0x7fff)
}

/// What a consumer that clamps the scale at zero computes.
pub(crate) fn zero_scales(tensor: &Q4Tensor) -> Q4Tensor {
    map_scales(tensor, |_| 0)
}

/// Exact dequantization, `code * scale + bias` in f64 from the stored bits.
pub(crate) fn reference(tensor: &Q4Tensor) -> Vec<f64> {
    let mut out = Vec::with_capacity(tensor.original_len);
    for block in &tensor.blocks {
        let scale = f64::from(q4_f16_to_f32(block.scale));
        let bias = f64::from(q4_f16_to_f32(block.bias));
        for byte in block.packed {
            out.push(f64::from(byte & 0x0f) * scale + bias);
            out.push(f64::from(byte >> 4) * scale + bias);
        }
    }
    out.truncate(tensor.original_len);
    out
}

/// The blocks as raw bytes, in the layout of a `.q4` payload and a Metal weight buffer.
pub(crate) fn block_bytes(tensor: &Q4Tensor) -> Vec<u8> {
    let mut out = Vec::with_capacity(tensor.blocks.len() * super::Q4_BLOCK_BYTES);
    for block in &tensor.blocks {
        out.extend_from_slice(&block.scale.to_ne_bytes());
        out.extend_from_slice(&block.bias.to_ne_bytes());
        out.extend_from_slice(&block.packed);
    }
    out
}

/// `y[i, r] = sum_k x[i, k] * w[r, k]` in f64, for `x` of `m` rows and `w` of `rows` rows.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) fn matmul_reference(w: &[f64], x: &[f32], m: usize, rows: usize, k: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; m * rows];
    for i in 0..m {
        for r in 0..rows {
            y[i * rows + r] = (0..k).map(|j| f64::from(x[i * k + j]) * w[r * k + j]).sum();
        }
    }
    y
}

/// The largest `sum_k |x[i, k] * w[r, k]|` over all outputs, the scale a matmul's
/// rounding error is measured against.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) fn matmul_magnitude(w: &[f64], x: &[f32], m: usize, rows: usize, k: usize) -> f64 {
    let mut max = 0.0f64;
    for i in 0..m {
        for r in 0..rows {
            let s: f64 = (0..k)
                .map(|j| (f64::from(x[i * k + j]) * w[r * k + j]).abs())
                .sum();
            max = max.max(s);
        }
    }
    max
}

/// Activations in `[0.5, 1.0]`: positive, so a sign error in the weights cannot cancel in a sum.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) fn positive_activations(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..len).map(|_| 0.5 + 0.5 * rng.unit()).collect()
}

/// Largest `|a - b|`; a NaN or infinite difference is returned as infinity.
pub(crate) fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    let mut max = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        if !d.is_finite() {
            return f64::INFINITY;
        }
        max = max.max(d);
    }
    max
}

/// One consumer's verdict: the fixture output matches the reference within `tol`,
/// and both controls miss it by more than `10 * tol`.
pub(crate) fn assert_sign_honoured(
    consumer: &str,
    reference: &[f64],
    fixture: &[f64],
    abs_control: &[f64],
    zero_control: &[f64],
    tol: f64,
) {
    let d = max_abs_diff(fixture, reference);
    assert!(
        d <= tol,
        "{consumer}: negative-scale output differs from the reference by {d:e} > {tol:e}"
    );
    for (label, control) in [("|scale|", abs_control), ("zero-scale", zero_control)] {
        let dc = max_abs_diff(control, reference);
        assert!(
            dc > 10.0 * tol,
            "{consumer}: the {label} control is within {dc:e} of the reference (needs > {:e}), \
                 so this fixture cannot tell a sign-dropping consumer from a correct one",
            10.0 * tol
        );
    }
}

#[test]
fn fixture_scales_are_all_negative_and_nonzero() {
    let t = negative_scale_tensor(4, 64, 7);
    assert_eq!(t.blocks.len(), 8);
    for b in &t.blocks {
        let s = q4_f16_to_f32(b.scale);
        assert!(s < 0.0 && s.is_finite(), "scale {s}");
        assert_eq!(b.packed[0] >> 4, 15);
    }
}

#[test]
fn dequantize_row_q4_0_honours_negative_scales() {
    let t = negative_scale_tensor(3, 96, 11);
    let run = |t: &Q4Tensor| -> Vec<f64> {
        super::dequantize_row_q4_0(&block_bytes(t), t.original_len)
            .into_iter()
            .map(f64::from)
            .collect()
    };
    assert_sign_honoured(
        "dequantize_row_q4_0",
        &reference(&t),
        &run(&t),
        &run(&abs_scales(&t)),
        &run(&zero_scales(&t)),
        1e-6,
    );
}

#[test]
fn dequantize_q4_to_f32_honours_negative_scales() {
    let t = negative_scale_tensor(3, 96, 13);
    let run = |t: &Q4Tensor| -> Vec<f64> {
        super::dequantize_q4_to_f32(t)
            .into_iter()
            .map(f64::from)
            .collect()
    };
    assert_sign_honoured(
        "dequantize_q4_to_f32",
        &reference(&t),
        &run(&t),
        &run(&abs_scales(&t)),
        &run(&zero_scales(&t)),
        1e-6,
    );
}

#[test]
fn q4_file_with_negative_scales_loads_and_zero_scale_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let t = negative_scale_tensor(2, 64, 17);
    let path = dir.path().join("neg.q4");
    super::save_q4_file(&path, &t).expect("save");
    let loaded = super::load_q4_file(&path).expect("a negative scale is admitted at ingress");
    assert_eq!(loaded.blocks, t.blocks);

    let zero = zero_scales(&t);
    let zpath = dir.path().join("zero.q4");
    super::save_q4_file(&zpath, &zero).expect("save");
    let err = super::load_q4_file(&zpath).expect_err("a zero scale is refused");
    assert!(
        err.to_string().contains("finite and nonzero"),
        "unexpected error: {err}"
    );
}

#[test]
fn block_ingress_admits_negative_and_refuses_zero_and_non_finite() {
    let check =
        |scale: u16, bias: u16| super::validate_q4_block_metadata("fixture", "t", 0, scale, bias);
    let one = q4_f32_to_f16(1.0);
    assert!(check(q4_f32_to_f16(-0.125), one).is_ok());
    assert!(check(q4_f32_to_f16(-65504.0), one).is_ok());
    assert!(
        check(0x8001, one).is_ok(),
        "negative subnormal scale is nonzero"
    );
    for (label, scale) in [
        ("+0", 0x0000u16),
        ("-0", 0x8000),
        ("+inf", 0x7c00),
        ("-inf", 0xfc00),
        ("nan", 0x7e00),
    ] {
        let err = check(scale, one).expect_err(label);
        assert!(err.to_string().contains("invalid scale"), "{label}: {err}");
    }
    assert!(check(q4_f32_to_f16(-0.125), 0x7e00).is_err(), "NaN bias");
}

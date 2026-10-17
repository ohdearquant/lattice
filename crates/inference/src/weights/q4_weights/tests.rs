use super::*;

fn q4_file_bytes(shape: &[usize], original_len: usize, scale: u16, bias: u16) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHQ4");
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for &dim in shape {
        buf.extend_from_slice(&(dim as u64).to_le_bytes());
    }
    buf.extend_from_slice(&(original_len as u64).to_le_bytes());
    for _ in 0..original_len.div_ceil(32) {
        buf.extend_from_slice(&scale.to_ne_bytes());
        buf.extend_from_slice(&bias.to_ne_bytes());
        buf.extend_from_slice(&[0u8; 16]);
    }
    buf
}

fn f16_file_bytes(shape: &[usize], values: &[u16]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHF1");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for &dim in shape {
        buf.extend_from_slice(&(dim as u64).to_le_bytes());
    }
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for &value in values {
        buf.extend_from_slice(&value.to_le_bytes());
    }
    buf
}

// -----------------------------------------------------------------------
// Test 1: Q4Block is exactly 20 bytes (scale + bias + 16 nibble bytes).
// -----------------------------------------------------------------------
#[test]
fn test_q4_block_size() {
    assert_eq!(std::mem::size_of::<Q4Block>(), 20);
    let b = Q4Block {
        scale: 0,
        bias: 0,
        packed: [0u8; 16],
    };
    let base = std::ptr::from_ref(&b) as usize;
    let packed_off = std::ptr::from_ref(&b.packed) as usize - base;
    assert_eq!(
        packed_off, 4,
        "packed field must start at byte offset 4 (after scale + bias, no padding)"
    );
}

// -----------------------------------------------------------------------
// Test 2: All-zero roundtrip — zeros in must produce zeros out.
// -----------------------------------------------------------------------
#[test]
fn test_quantize_dequantize_zeros() {
    let data = quantize_row_q4_0(&vec![0.0f32; 64]).unwrap();
    let out = dequantize_row_q4_0(&data, 64);
    assert_eq!(out.len(), 64);
    for v in &out {
        assert!(v.abs() < 1e-6, "expected ~0, got {v}");
    }
}

// -----------------------------------------------------------------------
// Test 3: Small positive values roundtrip within quantization tolerance.
// -----------------------------------------------------------------------
#[test]
fn test_quantize_dequantize_small_values() {
    let src: Vec<f32> = (0..32).map(|i| i as f32 * 7.0 / 31.0).collect();
    let data = quantize_row_q4_0(&src).unwrap();
    let out = dequantize_row_q4_0(&data, 32);
    let max_err = src
        .iter()
        .zip(&out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.5,
        "max abs error {max_err:.4} >= 0.5 for small values"
    );
}

// -----------------------------------------------------------------------
// Test 4: Symmetric positive and negative values roundtrip.
// -----------------------------------------------------------------------
#[test]
fn test_quantize_dequantize_symmetric() {
    let src: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 15.5 * 7.0).collect();
    let data = quantize_row_q4_0(&src).unwrap();
    let out = dequantize_row_q4_0(&data, 32);
    let max_err = src
        .iter()
        .zip(&out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.5,
        "max abs error {max_err:.4} >= 0.5 for symmetric values"
    );
}

// -----------------------------------------------------------------------
// Test 5: max/min values map to nibbles 15/0 under asymmetric quantization.
// -----------------------------------------------------------------------
#[test]
fn test_quantize_max_range() {
    // Block with w[0] = 7.0 (max, nibble 15) and w[1] = -7.0 (min, nibble 0), rest 0.
    let mut src = vec![0.0f32; 32];
    src[0] = 7.0;
    src[1] = -7.0;
    let data = quantize_row_q4_0(&src).unwrap();
    // Asymmetric: min=-7, max=7, scale = 14/15 ≈ 0.933
    // q[0] = round((7.0 - (-7.0)) / scale) = round(15) = 15 → low nibble
    // q[1] = round((-7.0 - (-7.0)) / scale) = round(0)  = 0  → high nibble
    // byte[0] = (0 << 4) | 15 = 0x0F.
    // Block layout: bytes 0..2 = scale, bytes 2..4 = bias, byte 4 = packed[0].
    let block_byte0 = data[4];
    assert_eq!(
        block_byte0 & 0x0f,
        15,
        "w[0]=7.0 (max) should produce low nibble 15"
    );
    assert_eq!(
        block_byte0 >> 4,
        0,
        "w[1]=-7.0 (min) should produce high nibble 0"
    );
}

// -----------------------------------------------------------------------
// Test 6: Exactly 32 elements — single block roundtrip.
// Values are in [-7, 7] so scale = 1.0 and max error is < 0.5 per step.
// -----------------------------------------------------------------------
#[test]
fn test_quantize_single_block() {
    // Use values in [-7, 7] so scale = 7/7 = 1.0 and max quantization error = 0.5.
    let src: Vec<f32> = (0..32).map(|i| (i as f32 / 31.0) * 14.0 - 7.0).collect();
    let data = quantize_row_q4_0(&src).unwrap();
    assert_eq!(data.len(), 20, "single block must be 20 bytes");
    let out = dequantize_row_q4_0(&data, 32);
    assert_eq!(out.len(), 32);
    // With scale = 1.0 the max quantization error is 0.5 (half a step).
    // Use threshold 0.51 to account for f16 scale rounding.
    let max_err = src
        .iter()
        .zip(&out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err <= 0.51,
        "max abs error {max_err:.4} > 0.51 for single block"
    );
}

// -----------------------------------------------------------------------
// Test 7: 128 elements = 4 blocks.
// -----------------------------------------------------------------------
#[test]
fn test_quantize_multiple_blocks() {
    let src: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 10.0).collect();
    let data = quantize_row_q4_0(&src).unwrap();
    assert_eq!(data.len(), 4 * 20, "4 blocks must be 80 bytes");
    let out = dequantize_row_q4_0(&data, 128);
    assert_eq!(out.len(), 128);
    let max_err = src
        .iter()
        .zip(&out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.5,
        "max abs error {max_err:.4} >= 0.5 for multiple blocks"
    );
}

// -----------------------------------------------------------------------
// Test 8: f32 → f16 → f32 roundtrip preserves value approximately.
// -----------------------------------------------------------------------
#[test]
fn test_f16_roundtrip() {
    let values = [
        0.0f32,
        1.0,
        -1.0,
        0.5,
        -0.5,
        std::f32::consts::PI,
        100.0,
        -100.0,
        0.001,
        65504.0, // max finite f16
    ];
    for &v in &values {
        let bits = q4_f32_to_f16(v);
        let back = q4_f16_to_f32(bits);
        // f16 has ~3 decimal digits of precision; allow 0.2% relative error
        let rel_err = if v.abs() > 1e-4 {
            (v - back).abs() / v.abs()
        } else {
            (v - back).abs()
        };
        assert!(
            rel_err < 0.004,
            "f16 roundtrip failed for {v}: got {back}, rel_err={rel_err:.6}"
        );
    }
}

// -----------------------------------------------------------------------
// Test 9: f16 helpers handle special values correctly.
// -----------------------------------------------------------------------
#[test]
fn test_f16_special_values() {
    // +0 and -0
    assert_eq!(q4_f32_to_f16(0.0f32), 0x0000);
    assert_eq!(q4_f32_to_f16(-0.0f32), 0x8000);
    assert_eq!(q4_f16_to_f32(0x0000), 0.0f32);

    // +∞ and -∞
    let pos_inf = q4_f32_to_f16(f32::INFINITY);
    assert_eq!(pos_inf, 0x7c00);
    assert!(q4_f16_to_f32(pos_inf).is_infinite() && q4_f16_to_f32(pos_inf) > 0.0);

    let neg_inf = q4_f32_to_f16(f32::NEG_INFINITY);
    assert_eq!(neg_inf, 0xfc00);
    assert!(q4_f16_to_f32(neg_inf).is_infinite() && q4_f16_to_f32(neg_inf) < 0.0);

    // NaN round-trips to NaN
    let nan_bits = q4_f32_to_f16(f32::NAN);
    assert!(
        q4_f16_to_f32(nan_bits).is_nan(),
        "NaN should round-trip to NaN"
    );

    // Overflow → ±∞
    let overflow = q4_f32_to_f16(1.0e10f32);
    assert_eq!(overflow, 0x7c00, "overflow should produce +∞");
}

// -----------------------------------------------------------------------
// Test 10: Nibble packing follows sequential-pairs layout.
// -----------------------------------------------------------------------
#[test]
fn test_nibble_packing_order() {
    // Asymmetric block: w[0]=0.0, w[1]=7.0, rest 0.0.
    // min = 0, max = 7, scale = 7/15 ≈ 0.467, bias = 0.
    // q[0] = round((0-0)/scale) = 0  → low nibble 0
    // q[1] = round((7-0)/scale) = 15 → high nibble 15
    // byte[0] = (15 << 4) | 0 = 0xF0.
    // Layout: bytes 0..2 = scale, 2..4 = bias, 4 = packed[0].
    let mut src = vec![0.0f32; 32];
    src[0] = 0.0;
    src[1] = 7.0;
    let data = quantize_row_q4_0(&src).unwrap();
    let byte0 = data[4];
    assert_eq!(
        byte0, 0xF0,
        "byte[0] should be 0xF0 for w[0]=0.0 (nibble=0), w[1]=7.0 (nibble=15)"
    );

    // Dequant: nibble * scale + bias.
    let out = dequantize_row_q4_0(&data, 32);
    // weight[0] = 0 * 0.467 + 0 = 0 (exact)
    assert!(
        (out[0] - 0.0).abs() < 1e-3,
        "weight[0] should be ~0.0, got {}",
        out[0]
    );
    // weight[1] = 15 * scale + bias. With f16 scale rounding, ~7.0 ± 1 ULP.
    assert!(
        (out[1] - 7.0).abs() < 0.05,
        "weight[1] should be ~7.0, got {}",
        out[1]
    );
}

// -----------------------------------------------------------------------
// Test 11: Multi-row per-row quantization via quantize_tensor_q4_0.
// -----------------------------------------------------------------------
#[test]
fn test_quantize_tensor_rows() {
    let rows = 4usize;
    let cols = 64usize;
    let src: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 - 128.0) / 20.0)
        .collect();
    let data = quantize_tensor_q4_0(&src, rows, cols).unwrap();
    let blocks_per_row = cols.div_ceil(32); // 2 blocks per row of 64 cols
    assert_eq!(
        data.len(),
        rows * blocks_per_row * 20,
        "tensor bytes mismatch"
    );

    // Dequant each row and check roundtrip error.
    for row_idx in 0..rows {
        let row_bytes = &data[row_idx * blocks_per_row * 20..(row_idx + 1) * blocks_per_row * 20];
        let out = dequantize_row_q4_0(row_bytes, cols);
        let row_src = &src[row_idx * cols..(row_idx + 1) * cols];
        let max_err = row_src
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.5,
            "row {row_idx}: max abs error {max_err:.4} >= 0.5"
        );
    }
}

// -----------------------------------------------------------------------
// Additional tests (covering design doc test plan items 2–12 via Q4Tensor API)
// -----------------------------------------------------------------------

/// Build bf16 vals from f32 using the module's own helper.
fn to_bf16(vals: &[f32]) -> Vec<u16> {
    vals.iter()
        .map(|&v| {
            // BF16 = upper 16 bits of f32
            let bits = v.to_bits();
            (bits >> 16) as u16
        })
        .collect()
}

fn bf16_round_trip(v: f32) -> f32 {
    bf16_to_f32((v.to_bits() >> 16) as u16)
}

#[test]
fn test_quantize_dequantize_round_trip_zeros_bf16() {
    let data = vec![0u16; 64];
    let tensor = quantize_bf16_to_q4(&data, &[64]).unwrap();
    let out = dequantize_q4_to_f32(&tensor);
    assert_eq!(out.len(), 64);
    for v in &out {
        assert!(v.abs() < 1e-6, "expected ~0, got {v}");
    }
}

#[test]
fn test_quantize_dequantize_round_trip_positive_bf16() {
    let f32_vals: Vec<f32> = (0..32).map(|i| i as f32 * 7.0 / 31.0).collect();
    let bf16_vals = to_bf16(&f32_vals);
    let tensor = quantize_bf16_to_q4(&bf16_vals, &[32]).unwrap();
    let out = dequantize_q4_to_f32(&tensor);
    // Compare against bf16-rounded originals (bf16 conversion is lossy at input).
    // Threshold 0.51 accounts for f16 scale rounding on top of the 0.5 quantization step.
    let max_err = f32_vals
        .iter()
        .zip(&out)
        .map(|(a, b)| (bf16_round_trip(*a) - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err <= 0.51, "max abs error {max_err:.4} > 0.51");
}

#[test]
fn test_nibble_packing_byte_value_bf16() {
    // Asymmetric: w[0]=0.0, w[1]=7.0, rest 0.0
    // min=0, max=7, scale=7/15, bias=0. q[0]=0 (low), q[1]=15 (high). byte[0]=0xF0.
    let mut f32_vals = [0.0f32; 32];
    f32_vals[0] = 0.0;
    f32_vals[1] = 7.0;
    let bf16_vals = to_bf16(&f32_vals);
    let tensor = quantize_bf16_to_q4(&bf16_vals, &[32]).unwrap();
    assert_eq!(tensor.blocks.len(), 1);
    assert_eq!(
        tensor.blocks[0].packed[0], 0xF0,
        "byte[0] should be 0xF0 for w[0]=0.0 (nibble 0), w[1]=7.0 (nibble 15)"
    );
}

#[test]
fn test_max_value_clamps_to_nibble_15() {
    // w[0]=100.0 → scale = 100/7 ≈ 14.28 → q[0] = round(7.0)+8 = 15
    let mut f32_vals = [0.0f32; 32];
    f32_vals[0] = 100.0;
    let bf16_vals = to_bf16(&f32_vals);
    let tensor = quantize_bf16_to_q4(&bf16_vals, &[32]).unwrap();
    let low_nibble = tensor.blocks[0].packed[0] & 0x0f;
    assert_eq!(low_nibble, 15, "weight[0]=100 should clamp to nibble 15");
}

#[test]
fn test_block_boundary_continuity() {
    // Values chosen so that abs_max per block = 7.0 → scale = 1.0.
    // Every value is at least 1.0 above zero so no value rounds to nibble 8 (zero).
    // Block 0: all positive [1..7] repeated; block 1: all negative [-1..-7] repeated.
    let mut f32_vals = Vec::with_capacity(64);
    for i in 0..32 {
        f32_vals.push((i % 7) as f32 + 1.0);
    } // range [1, 7]
    for i in 0..32 {
        f32_vals.push(-((i % 7) as f32 + 1.0));
    } // range [-7, -1]
    let bf16_vals = to_bf16(&f32_vals);
    let tensor = quantize_bf16_to_q4(&bf16_vals, &[64]).unwrap();
    assert_eq!(tensor.blocks.len(), 2);
    let out = dequantize_q4_to_f32(&tensor);
    // All block-0 values are positive [1..7], scale≈1. Dequant ≥ (1-0.5)*1 = 0.5 > 0.
    for v in &out[0..32] {
        assert!(*v > 0.0, "block 0 weight should be positive, got {v}");
    }
    // All block-1 values are negative [-7..-1].
    for v in &out[32..64] {
        assert!(*v < 0.0, "block 1 weight should be negative, got {v}");
    }
}

#[test]
fn test_save_load_round_trip() {
    let f32_vals: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 4.0).collect();
    let bf16_vals = to_bf16(&f32_vals);
    let original = quantize_bf16_to_q4(&bf16_vals, &[8, 8]).unwrap();
    let path = std::path::PathBuf::from("/tmp/test_q4_round_trip.q4");
    save_q4_file(&path, &original).unwrap();
    let loaded = load_q4_file(&path).unwrap();
    assert_eq!(loaded.shape, original.shape);
    assert_eq!(loaded.original_len, original.original_len);
    assert_eq!(loaded.blocks.len(), original.blocks.len());
    for (a, b) in original.blocks.iter().zip(&loaded.blocks) {
        assert_eq!(a.scale, b.scale, "scale mismatch after load");
        assert_eq!(a.packed, b.packed, "packed mismatch after load");
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_stream_quantize_shard_matches_batch() {
    let f32_vals: Vec<f32> = (0..96).map(|i| i as f32 / 10.0).collect();
    let bf16_vals = to_bf16(&f32_vals);
    let batch_tensor = quantize_bf16_to_q4(&bf16_vals, &[96]).unwrap();
    // Convert bf16 u16s to raw bytes (native endian, matching stream_quantize_shard)
    let bf16_bytes: Vec<u8> = bf16_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let stream_blocks = stream_quantize_shard(&bf16_bytes).unwrap();
    assert_eq!(stream_blocks.len(), batch_tensor.blocks.len());
    for (a, b) in batch_tensor.blocks.iter().zip(&stream_blocks) {
        assert_eq!(a.scale, b.scale, "stream vs batch scale mismatch");
        assert_eq!(a.packed, b.packed, "stream vs batch packed mismatch");
    }
}

#[test]
fn test_shape_preservation() {
    let shape = vec![4usize, 8, 4]; // 128 elements
    let data = vec![0u16; 128];
    let tensor = quantize_bf16_to_q4(&data, &shape).unwrap();
    assert_eq!(tensor.shape, shape);
    assert_eq!(tensor.original_len, 128);
    assert_eq!(tensor.blocks.len(), 4); // 128 / 32 = 4

    let path = std::path::PathBuf::from("/tmp/test_q4_shape.q4");
    save_q4_file(&path, &tensor).unwrap();
    let loaded = load_q4_file(&path).unwrap();
    assert_eq!(loaded.shape, shape);
    assert_eq!(loaded.original_len, 128);
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_round_trip_accuracy_tolerance() {
    // 1024 pseudo-random f32 in [-7, 7] using a simple LCG for reproducibility.
    // With scale ≈ 1.0 per block the theoretical max error per weight is 0.5
    // and expected MAE ≈ 0.25 for uniform random input.
    let mut state = 12345u64;
    let mut f32_vals = Vec::with_capacity(1024);
    for _ in 0..1024 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let v = ((state >> 32) as f32 / u32::MAX as f32) * 14.0 - 7.0;
        f32_vals.push(v);
    }
    let data = quantize_row_q4_0(&f32_vals).unwrap();
    let out = dequantize_row_q4_0(&data, 1024);
    assert_eq!(out.len(), 1024);
    let mae = f32_vals
        .iter()
        .zip(&out)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / 1024.0;
    // Threshold: Q4_0 with scale≈1.0 has MAE ≈ 0.25; allow 0.30 for block edge effects.
    assert!(
        mae < 0.30,
        "mean abs error {mae:.4} >= 0.30 (Q4 MAE for uniform [-7,7] expected ≈ 0.25)"
    );
}

// -----------------------------------------------------------------------
// QuaRot pipeline entry points (ADR-044 step 3c-1)
// -----------------------------------------------------------------------

fn f32_to_bf16_bits(v: f32) -> u16 {
    // BF16 = top 16 bits of f32, round-to-nearest-even.
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn synthetic_f32_uniform(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (state >> 32) as f32 / u32::MAX as f32;
            u * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn quantize_f32_to_q4_shape_and_length() {
    let src = synthetic_f32_uniform(96, 17);
    let q = quantize_f32_to_q4(&src, &[3, 32]).unwrap();
    assert_eq!(q.shape, vec![3, 32]);
    assert_eq!(q.original_len, 96);
    assert_eq!(q.blocks.len(), 3, "96 elems = 3 full Q4 blocks");
}

#[test]
fn quantize_f32_to_q4_pads_partial_block() {
    let src = synthetic_f32_uniform(40, 19);
    let q = quantize_f32_to_q4(&src, &[40]).unwrap();
    assert_eq!(q.original_len, 40);
    assert_eq!(q.blocks.len(), 2, "40 elems = 1 full + 1 partial Q4 block");
}

#[test]
fn quantize_f32_to_q4_partial_block_uses_real_tail_min_max() {
    // Mutation-sensitive: a tail block of [5, 6, 7] zero-padded to 32
    // slots would (pre-fix) fold min/max over the padded zeros too,
    // yielding min=0/max=7 instead of the real min=5/max=7. This test
    // fails if the asymmetric path reverts to computing stats over the
    // padded [f32; 32] array instead of the real `chunk.len()` elements.
    let src = [5.0f32, 6.0, 7.0];
    let q = quantize_f32_to_q4(&src, &[3]).unwrap();
    assert_eq!(q.original_len, 3);
    assert_eq!(q.blocks.len(), 1);

    let block = q.blocks[0];
    assert_eq!(block.scale, q4_f32_to_f16(2.0f32 / 15.0));
    assert_eq!(block.bias, q4_f32_to_f16(5.0));
    assert_ne!(
        block.scale,
        q4_f32_to_f16(7.0f32 / 15.0),
        "partial tail scale must not include padded zero in max-min range"
    );
    assert_ne!(
        block.bias,
        q4_f32_to_f16(0.0),
        "partial tail bias must be the real tail min, not padded zero"
    );
}

#[test]
fn quantize_f64_to_q4_symmetric_partial_block_is_bit_identical_to_padded_block() {
    // Symmetric mode must stay bit-identical to the old always-padded
    // path: adding zeros to a non-empty real chunk can never increase
    // abs_max, so the fixed length-aware helper must produce exactly the
    // same Q4Block as folding over the full zero-padded array.
    let src = [5.0f64, -6.0, 7.0];
    let mut padded = [0.0f32; 32];
    for (dst, src) in padded.iter_mut().zip(src.iter()) {
        *dst = *src as f32;
    }

    let expected = quantize_block_with_mode_len(&padded, 32, true).unwrap();
    let q = quantize_f64_to_q4_mode(&src, &[3], true).unwrap();

    assert_eq!(
        q.blocks[0], expected,
        "symmetric partial blocks must stay byte-identical to the old padded path"
    );
}

/// The largest f32 whose `abs_max / 7` symmetric scale still underflows f16
/// to `+0.0`, scaled to reproduce the magnitude the QuaRot write pass emits
/// on Qwen3.5 (~4.8e-38, against f16's smallest positive subnormal ~5.96e-8).
const TINY_SYMMETRIC_ABS_MAX: f32 = 3.363e-37;

/// Asymmetric counterpart: `range / 15` lands in the same underflowing region.
const TINY_ASYMMETRIC_RANGE: f32 = 1.5e-37;

#[test]
fn symmetric_block_with_underflowing_scale_is_quantizable_and_reloadable() {
    // A Hadamard-rotated block can have a range that is tiny but not
    // exactly zero. `abs_max / 7` is then a positive f32 far below f16's
    // smallest subnormal, so it serializes to +0.0 -- a scale no reader can
    // dequantize with. Such a block must take the degenerate fallback, not
    // be rejected: this is the shape the repository's own `quantize_quarot`
    // write pass produces, via `quantize_f64_to_q4` (symmetric).
    let mut vals = [0.0f32; 32];
    vals[0] = TINY_SYMMETRIC_ABS_MAX;
    vals[7] = -TINY_SYMMETRIC_ABS_MAX / 3.0;
    assert!(
        q4_f16_to_f32(q4_f32_to_f16(TINY_SYMMETRIC_ABS_MAX / 7.0)) == 0.0,
        "fixture must actually underflow f16, else this test proves nothing"
    );

    let block = quantize_block_with_mode_len(&vals, 32, true)
        .expect("a block with a tiny but nonzero range must quantize");

    let scale = q4_f16_to_f32(block.scale);
    assert!(
        scale.is_finite() && scale > 0.0,
        "serialized scale {scale} must be finite and strictly positive"
    );
    assert_eq!(scale, 1.0, "degenerate blocks take the 1.0 fallback");

    // Reconstruction error is bounded by the block's true range, exactly as
    // it already was for an all-zero block.
    let tensor = Q4Tensor {
        blocks: vec![block],
        shape: vec![32],
        original_len: 32,
    };
    let out = dequantize_q4_to_f32(&tensor);
    for (i, (&got, &want)) in out.iter().zip(vals.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 2.0 * TINY_SYMMETRIC_ABS_MAX,
            "element {i}: reconstruction error {} exceeds the block's own range",
            (got - want).abs()
        );
    }
}

#[test]
fn asymmetric_block_with_underflowing_scale_is_quantizable_and_reloadable() {
    let mut vals = [0.0f32; 32];
    vals[3] = TINY_ASYMMETRIC_RANGE;
    assert!(
        q4_f16_to_f32(q4_f32_to_f16(TINY_ASYMMETRIC_RANGE / 15.0)) == 0.0,
        "fixture must actually underflow f16, else this test proves nothing"
    );

    let block = quantize_block_with_mode_len(&vals, 32, false)
        .expect("a block with a tiny but nonzero range must quantize");

    let scale = q4_f16_to_f32(block.scale);
    assert!(
        scale.is_finite() && scale > 0.0,
        "serialized scale {scale} must be finite and strictly positive"
    );
    assert_eq!(scale, 1.0, "degenerate blocks take the 1.0 fallback");

    let tensor = Q4Tensor {
        blocks: vec![block],
        shape: vec![32],
        original_len: 32,
    };
    let out = dequantize_q4_to_f32(&tensor);
    for (i, (&got, &want)) in out.iter().zip(vals.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 2.0 * TINY_ASYMMETRIC_RANGE,
            "element {i}: reconstruction error {} exceeds the block's own range",
            (got - want).abs()
        );
    }
}

#[test]
fn quarot_symmetric_write_pass_survives_a_tiny_range_row() {
    // End to end over the seam CI exercises: a tensor written by the
    // symmetric (QuaRot) quantizer, saved, and read back through the same
    // per-block metadata validation the loader applies. Before the
    // degenerate predicate covered f16 underflow, the write side emitted a
    // scale of +0.0 and the read side rejected the file the project's own
    // quantizer had just produced.
    let mut src = vec![0.0f64; 64];
    src[0] = f64::from(TINY_SYMMETRIC_ABS_MAX);
    src[40] = f64::from(-TINY_SYMMETRIC_ABS_MAX) / 2.0;

    let q = quantize_f64_to_q4(&src, &[64]).expect("symmetric quantize must accept the row");
    for (i, b) in q.blocks.iter().enumerate() {
        let scale = q4_f16_to_f32(b.scale);
        assert!(
            scale.is_finite() && scale > 0.0,
            "block {i} serialized scale {scale} is not a usable f16"
        );
    }

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("tiny_range.q4");
    save_q4_file(&path, &q).unwrap();
    let reloaded = load_q4_file(&path).expect("the loader must accept what the writer emits");
    assert_eq!(reloaded.blocks, q.blocks);
}

// #1380: `load_q4_file` is a raw-open checkpoint loader entry point,
// structurally identical to the three the issue named, that used to
// call plain `File::open` before any file-type check ran. A FIFO
// planted at the `.q4` path -- an attacker or a misconfigured MTP
// sidecar directory -- blocked this call indefinitely. Proves the fix
// the same way as the other loader entry points: the call must return
// (any return) inside a deadline, and the return must specifically
// reject the FIFO as non-regular. `mkfifo` (POSIX, not macOS-only)
// keeps this test portable to the Linux CI legs that run this suite.
#[test]
fn load_q4_file_rejects_a_fifo_without_blocking() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("planted.q4");
    let status = std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .expect("run mkfifo");
    assert!(status.success(), "mkfifo must succeed to set up this test");

    let (tx, rx) = std::sync::mpsc::channel();
    let probe_path = path.clone();
    std::thread::spawn(move || {
        // `Box<dyn Error>` is not `Send` (no `+ Send` bound), so it
        // cannot cross this channel -- map to its `Display` text first,
        // same as the `.f16`/`.q3` siblings below.
        let result = load_q4_file(&probe_path).map_err(|e| e.to_string());
        let _ = tx.send(result);
    });
    let result: Result<Q4Tensor, String> =
        rx.recv_timeout(std::time::Duration::from_secs(5)).expect(
            "load_q4_file did not return within 5s -- it blocked on the planted FIFO, \
                 meaning the open-time regular-file guard regressed",
        );
    let err = result.expect_err("a FIFO must be rejected, not accepted, for a .q4 load");
    assert!(
        err.contains("not a regular file"),
        "expected a not-a-regular-file refusal, got: {err}"
    );
}

// #1380: `load_f16_tensor_file` is the `.f16` sibling of the FIFO
// regression above -- same raw-open shape, same fix.
#[test]
fn load_f16_tensor_file_rejects_a_fifo_without_blocking() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("planted.f16");
    let status = std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .expect("run mkfifo");
    assert!(status.success(), "mkfifo must succeed to set up this test");

    let (tx, rx) = std::sync::mpsc::channel();
    let probe_path = path.clone();
    std::thread::spawn(move || {
        // `Box<dyn Error>` is not `Send` (no `+ Send` bound), so it
        // cannot cross this channel -- map to its `Display` text first.
        let result = load_f16_tensor_file(&probe_path).map_err(|e| e.to_string());
        let _ = tx.send(result);
    });
    let result: Result<(Vec<f32>, Vec<usize>), String> =
        rx.recv_timeout(std::time::Duration::from_secs(5)).expect(
            "load_f16_tensor_file did not return within 5s -- it blocked on the planted \
                 FIFO, meaning the open-time regular-file guard regressed",
        );
    let err = result.expect_err("a FIFO must be rejected, not accepted, for a .f16 load");
    assert!(
        err.contains("not a regular file"),
        "expected a not-a-regular-file refusal, got: {err}"
    );
}

#[test]
fn scale_too_large_for_f16_is_still_rejected() {
    // The degenerate fallback covers only the underflow direction. A range
    // that overflows f16 is a real range that cannot be represented, and
    // substituting 1.0 there would silently mis-quantize the block, so
    // `q4_metadata_bits` must still reject it.
    let mut vals = [0.0f32; 32];
    vals[0] = 1.0e7;
    let err = quantize_block_with_mode_len(&vals, 32, true)
        .expect_err("a scale above f16's maximum must not be silently replaced");
    assert!(
        format!("{err}").contains("strictly positive f16"),
        "unexpected error: {err}"
    );
}

#[test]
fn bias_above_f16_range_is_still_rejected() {
    // Independent of the scale predicate: an asymmetric block whose
    // `min_val` exceeds f16's maximum overflows to infinity as a bias.
    let mut vals = [1.0e5f32; 32];
    vals[0] = 1.0e5 + 1.0;
    let err = quantize_block_with_mode_len(&vals, 32, false)
        .expect_err("a bias outside f16 range must be rejected");
    assert!(
        format!("{err}").contains("finite f16"),
        "unexpected error: {err}"
    );
}

#[test]
fn quantize_f64_to_q4_matches_f32_path_after_downcast() {
    // The f64 wrapper must agree byte-for-byte with the f32 entry under
    // the same symmetry mode. `quantize_f64_to_q4` defaults to symmetric
    // (Hadamard-rotated weights are zero-mean); the unrotated `quantize_
    // f32_to_q4` defaults to asymmetric. Both should produce identical
    // output when called with the same mode flag.
    let src_f64: Vec<f64> = synthetic_f32_uniform(256, 23)
        .into_iter()
        .map(f64::from)
        .collect();
    let src_f32: Vec<f32> = src_f64.iter().map(|&v| v as f32).collect();
    let q_f64 = quantize_f64_to_q4_mode(&src_f64, &[256], false).unwrap();
    let q_f32 = quantize_f32_to_q4(&src_f32, &[256]).unwrap();
    assert_eq!(q_f64.shape, q_f32.shape);
    assert_eq!(q_f64.original_len, q_f32.original_len);
    assert_eq!(
        q_f64.blocks.len(),
        q_f32.blocks.len(),
        "f64 path must produce same block count"
    );
    for (i, (a, b)) in q_f64.blocks.iter().zip(q_f32.blocks.iter()).enumerate() {
        assert_eq!(a.scale, b.scale, "block {i} scale mismatch");
        assert_eq!(a.bias, b.bias, "block {i} bias mismatch");
        assert_eq!(a.packed, b.packed, "block {i} packed mismatch");
    }
}

#[test]
fn quantize_f32_to_q4_matches_bf16_path_when_input_is_bf16_castable() {
    // Control test: when the f32 input has zero mantissa entropy below the
    // BF16 truncation point (i.e., it was already bf16 -> f32), both paths
    // MUST produce identical Q4 tensors. This nails down the equivalence
    // so any divergence in the high-precision test below is provably
    // attributable to BF16 truncation, not to a behavioral difference
    // between the two quantize_* implementations.
    let bf16_bits: Vec<u16> = synthetic_f32_uniform(256, 29)
        .into_iter()
        .map(f32_to_bf16_bits)
        .collect();
    let f32_from_bf16: Vec<f32> = bf16_bits.iter().map(|&b| bf16_to_f32(b)).collect();

    let q_bf16 = quantize_bf16_to_q4(&bf16_bits, &[256]).unwrap();
    let q_f32 = quantize_f32_to_q4(&f32_from_bf16, &[256]).unwrap();
    assert_eq!(q_bf16.blocks.len(), q_f32.blocks.len());
    for (i, (a, b)) in q_bf16.blocks.iter().zip(q_f32.blocks.iter()).enumerate() {
        assert_eq!(a.scale, b.scale, "block {i} scale should match");
        assert_eq!(a.packed, b.packed, "block {i} packed should match");
    }
}

#[test]
fn quantize_f32_to_q4_lower_error_than_bf16_path_on_high_precision_input() {
    // ADR-044 §"Step 3c contract" decision driver: when the source carries
    // >7 bits of mantissa entropy (e.g., the output of an f64 rotation
    // pass), the bf16 route discards information the f32 route preserves.
    //
    // Measurement: take 2048 pseudo-random f32 values uniform in [-1, 1]
    // (23-bit mantissa entropy). Quantize via both paths, dequantize, and
    // compare against the f32 source.
    //
    // Expectation: path (b) `quantize_f32_to_q4` produces strictly lower
    // max abs error AND lower mean abs error than path (a) f32->bf16->Q4.
    let src = synthetic_f32_uniform(2048, 31);
    let bf16_bits: Vec<u16> = src.iter().map(|&v| f32_to_bf16_bits(v)).collect();

    let q_bf16 = quantize_bf16_to_q4(&bf16_bits, &[2048]).unwrap();
    let q_f32 = quantize_f32_to_q4(&src, &[2048]).unwrap();
    let deq_bf16 = dequantize_q4_to_f32(&q_bf16);
    let deq_f32 = dequantize_q4_to_f32(&q_f32);

    let err = |reconstructed: &[f32]| -> (f32, f32) {
        let mut max_err = 0.0_f32;
        let mut sum_err = 0.0_f32;
        for (s, r) in src.iter().zip(reconstructed.iter()) {
            let e = (s - r).abs();
            max_err = max_err.max(e);
            sum_err += e;
        }
        (max_err, sum_err / src.len() as f32)
    };
    let (max_bf16, mean_bf16) = err(&deq_bf16);
    let (max_f32, mean_f32) = err(&deq_f32);

    // Self-documenting measurement print (visible via `cargo test -- --nocapture`).
    // Numbers feed the ADR-044 §"Step 3c contract" Q4 bridge decision record.
    eprintln!(
        "[3c-1 measurement] n=2048 source=f32 uniform [-1,1]: \
             f32_path mean_abs_err={mean_f32:.6} max_abs_err={max_f32:.6}; \
             bf16_path mean_abs_err={mean_bf16:.6} max_abs_err={max_bf16:.6}"
    );

    assert!(
        mean_f32 < mean_bf16,
        "f32 mean abs error ({mean_f32:.6}) should be < bf16 mean abs error ({mean_bf16:.6})"
    );
    assert!(
        max_f32 <= max_bf16,
        "f32 max abs error ({max_f32:.6}) should be <= bf16 max abs error ({max_bf16:.6})"
    );
}

// -----------------------------------------------------------------------
// QuaRot composed rotated+Q4 forward gate (Issue #320)
//
// Exercises the full composition: absorb_rotations (offline rotation
// absorption) → quantize_f64_to_q4 → dequantize_q4_to_f32 → matmul,
// and asserts correctness against an independent f64 reference that
// manually mirrors each step. Mutation-sensitive: perturbing the rotation
// dispatch (pipeline.rs:226-230), absorption helpers (rotation.rs:161-164
// or rotation.rs:184-192), Q4 symmetric scale (q4_weights.rs:277 or :280),
// or Q4 symmetric mode flag (q4_weights.rs:523) must cause failure.
// -----------------------------------------------------------------------
#[test]
fn quarot_rotated_q4_forward_matches_f64_reference() {
    use std::collections::HashMap;

    use crate::quant::quarot::hadamard::RandomizedHadamard;
    use crate::quant::quarot::pipeline::{TensorEntry, absorb_rotations};
    use crate::quant::quarot::plan::RotationPlan;

    const HIDDEN: usize = 32;
    const Q_ROWS: usize = 2; // q_proj input-side [2, 32]
    const O_ROWS: usize = 32; // o_proj output-side [32, 32]

    let q_name = "model.language_model.layers.0.self_attn.q_proj.weight";
    let o_name = "model.language_model.layers.0.self_attn.o_proj.weight";

    fn lcg_f64(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 32) as f64 / u32::MAX as f64 * 2.0 - 1.0
            })
            .collect()
    }

    fn matvec(w: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
        (0..rows)
            .map(|r| {
                w[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }

    // NaN-honest max|a-b|: a `.fold(0.0, f64::max)` silently drops a NaN/Inf
    // operand (IEEE maxNum keeps the non-NaN side), letting a catastrophically
    // wrong output read as 0.0 and slip past a `<= tol` gate. Surface it instead.
    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        let mut max = 0.0_f64;
        for (x, y) in a.iter().zip(b) {
            let d = (x - y).abs();
            if !d.is_finite() {
                return d;
            }
            if d > max {
                max = d;
            }
        }
        max
    }

    // Independent symmetric Q4 dequant reference: re-implements
    // quantize_block_with_mode(symmetric=true) + dequantize_q4_to_f32
    // so that a bug in either production function is visible as a mismatch.
    fn ref_q4_dequant(data: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(data.len());
        for chunk in data.chunks(32) {
            let f32s: Vec<f32> = chunk.iter().map(|&v| v as f32).collect();
            let abs_max = f32s.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale_f32 = if abs_max == 0.0 {
                1.0_f32
            } else {
                abs_max / 7.0
            };
            let bias_f32 = -8.0_f32 * scale_f32;
            // Round-trip through f16 storage exactly as quantize_block_with_mode does.
            let scale_dq = q4_f16_to_f32(q4_f32_to_f16(scale_f32));
            let bias_dq = q4_f16_to_f32(q4_f32_to_f16(bias_f32));
            let inv_scale = 1.0 / scale_f32;
            for &v in &f32s {
                let nibble = ((v * inv_scale).round() + 8.0).clamp(0.0, 15.0) as u8;
                out.push(f64::from(nibble as f32 * scale_dq + bias_dq));
            }
        }
        out
    }

    let q_data_orig = lcg_f64(Q_ROWS * HIDDEN, 0x1111_1111_1111_1111);
    let o_data_orig = lcg_f64(O_ROWS * HIDDEN, 0x2222_2222_2222_2222);
    let x_q = lcg_f64(HIDDEN, 0x3333_3333_3333_3333);
    let x_o = lcg_f64(HIDDEN, 0x4444_4444_4444_4444);

    let rotation = RandomizedHadamard::new(0x3200_0001, HIDDEN).expect("rotation init");
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();

    // ---- Production path: absorb_rotations + quantize_f64_to_q4 + dequantize ----
    let mut tensors: HashMap<String, TensorEntry> = HashMap::new();
    tensors.insert(
        q_name.to_string(),
        TensorEntry {
            name: q_name.to_string(),
            shape: vec![Q_ROWS, HIDDEN],
            data: q_data_orig.clone(),
        },
    );
    tensors.insert(
        o_name.to_string(),
        TensorEntry {
            name: o_name.to_string(),
            shape: vec![O_ROWS, HIDDEN],
            data: o_data_orig.clone(),
        },
    );

    absorb_rotations(&mut tensors, &plan, &rotation).expect("absorb_rotations");

    let q_q4 =
        quantize_f64_to_q4(&tensors[q_name].data, &[Q_ROWS, HIDDEN]).expect("q_proj quantize");
    let o_q4 =
        quantize_f64_to_q4(&tensors[o_name].data, &[O_ROWS, HIDDEN]).expect("o_proj quantize");

    // Shape and block-count sanity: fail loudly if the Q4 bridge is broken
    assert_eq!(q_q4.shape, vec![Q_ROWS, HIDDEN], "q_proj shape");
    assert_eq!(q_q4.original_len, Q_ROWS * HIDDEN, "q_proj original_len");
    assert_eq!(q_q4.blocks.len(), Q_ROWS, "[2,32] must produce 2 Q4 blocks");
    assert_eq!(o_q4.shape, vec![O_ROWS, HIDDEN], "o_proj shape");
    assert_eq!(o_q4.original_len, O_ROWS * HIDDEN, "o_proj original_len");
    assert_eq!(
        o_q4.blocks.len(),
        O_ROWS,
        "[32,32] must produce 32 Q4 blocks"
    );

    let q_deq: Vec<f64> = dequantize_q4_to_f32(&q_q4)
        .into_iter()
        .map(f64::from)
        .collect();
    let o_deq: Vec<f64> = dequantize_q4_to_f32(&o_q4)
        .into_iter()
        .map(f64::from)
        .collect();

    let prod_y_q = matvec(&q_deq, Q_ROWS, HIDDEN, &x_q);
    let prod_y_o = matvec(&o_deq, O_ROWS, HIDDEN, &x_o);

    // ---- Reference path: manual rotation + independent Q4 dequant ----

    // Input-side: apply rotation row-by-row (mirrors absorb_input_rotation_f64)
    let mut q_ref = q_data_orig.clone();
    for r in 0..Q_ROWS {
        rotation
            .apply_f64(&mut q_ref[r * HIDDEN..(r + 1) * HIDDEN])
            .expect("q_proj row rotation");
    }

    // Output-side: apply rotation column-by-column (mirrors absorb_output_rotation_f64)
    let mut o_ref = o_data_orig.clone();
    let mut col_buf = vec![0.0_f64; O_ROWS];
    for c in 0..HIDDEN {
        for r in 0..O_ROWS {
            col_buf[r] = o_ref[r * HIDDEN + c];
        }
        rotation
            .apply_f64(&mut col_buf)
            .expect("o_proj col rotation");
        for r in 0..O_ROWS {
            o_ref[r * HIDDEN + c] = col_buf[r];
        }
    }

    let q_ref_deq = ref_q4_dequant(&q_ref);
    let o_ref_deq = ref_q4_dequant(&o_ref);

    let ref_y_q = matvec(&q_ref_deq, Q_ROWS, HIDDEN, &x_q);
    let ref_y_o = matvec(&o_ref_deq, O_ROWS, HIDDEN, &x_o);

    // ---- Assert ----
    let max_q = max_diff(&prod_y_q, &ref_y_q);
    let max_o = max_diff(&prod_y_o, &ref_y_o);

    eprintln!("[quarot_q4_gate] max_abs_diff q_proj={max_q:.2e} o_proj={max_o:.2e}");

    assert!(
        max_q <= 1e-5,
        "q_proj forward max_abs_diff {max_q:.2e} > 1e-5: composed rotated+Q4 path is broken"
    );
    assert!(
        max_o <= 1e-5,
        "o_proj forward max_abs_diff {max_o:.2e} > 1e-5: composed rotated+Q4 path is broken"
    );
}

#[test]
fn quantize_f32_to_q4_rejects_shape_data_mismatch() {
    let data = synthetic_f32_uniform(64, 41);
    let err = quantize_f32_to_q4(&data, &[3, 32])
        .expect_err("shape claiming 96 elements for 64 values must fail");
    assert!(err.to_string().contains("shape product"));
}

#[test]
fn quantize_f64_to_q4_rejects_shape_data_mismatch() {
    let data: Vec<f64> = synthetic_f32_uniform(64, 43)
        .into_iter()
        .map(f64::from)
        .collect();
    let err = quantize_f64_to_q4(&data, &[3, 32])
        .expect_err("shape claiming 96 elements for 64 values must fail");
    assert!(err.to_string().contains("shape product"));
}

#[test]
fn quantize_bf16_to_q4_rejects_shape_data_mismatch() {
    // Lock the same contract on the pre-existing BF16 entry point — the
    // SafeTensors source format rejects shape/data mismatches and the Q4
    // bridge must not silently weaken that invariant.
    let data: Vec<u16> = (0..64).map(|i| i as u16).collect();
    let err = quantize_bf16_to_q4(&data, &[3, 32])
        .expect_err("shape claiming 96 elements for 64 values must fail");
    assert!(err.to_string().contains("shape product"));
}

#[test]
fn quantize_f32_to_q4_rejects_shape_product_overflow() {
    let data = vec![0.0_f32; 32];
    // usize::MAX * 2 overflows; checked_mul must catch it before the
    // length comparison aliases to a valid length by wraparound.
    let err = quantize_f32_to_q4(&data, &[usize::MAX, 2])
        .expect_err("overflowed shape product must fail");
    assert!(err.to_string().contains("overflows usize"));
}

#[test]
fn quantize_f32_to_q4_block_layout_matches_quantize_row() {
    // Sanity: for input that is an exact multiple of 32, the entry should
    // produce the same per-block byte layout as `quantize_row_q4_0`
    // (which the existing kernels are already validated against).
    let src = synthetic_f32_uniform(128, 37);
    let q = quantize_f32_to_q4(&src, &[128]).unwrap();
    let row_bytes = quantize_row_q4_0(&src).unwrap();
    assert_eq!(row_bytes.len(), q.blocks.len() * 20);
    // SAFETY: Q4Block is #[repr(C)] size 20 (scale + bias + 16 nibbles),
    // alignment 2; byte-cast is valid because target element type is u8.
    let q_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(q.blocks.as_ptr().cast::<u8>(), q.blocks.len() * 20) };
    assert_eq!(q_bytes, row_bytes.as_slice());
}

// -----------------------------------------------------------------------
// Tests for dequantize_row_q4_0 robustness (issue #263)
//
// These tests verify that dequantize_row_q4_0 does NOT panic on
// misaligned or undersized inputs. The function uses chunks_exact(20)
// which silently ignores trailing bytes, so removing the assert_eq!
// alignment check makes the behaviour well-defined on any input.
// -----------------------------------------------------------------------

/// Misaligned input (25 bytes = 1 complete block + 5 remainder bytes) must not panic.
/// The 5 trailing bytes are ignored; only the 1 complete block (32 values) is returned.
#[test]
fn dequantize_row_q4_0_misaligned_does_not_panic() {
    // Build a valid 1-block (20-byte) buffer by quantizing 32 known values.
    let src: Vec<f32> = (0..32).map(|i| (i as f32 / 31.0) * 14.0 - 7.0).collect();
    let mut buf = quantize_row_q4_0(&src).unwrap(); // exactly 20 bytes
    assert_eq!(buf.len(), 20);
    // Append 5 garbage bytes — total 25, which is NOT a multiple of 20.
    buf.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0xFF]);
    assert_eq!(buf.len(), 25);

    // Must not panic; chunks_exact(20) stops after the first complete block.
    let out = dequantize_row_q4_0(&buf, 32);

    // Should return exactly 32 values (one block worth).
    assert_eq!(out.len(), 32);

    // Round-trip tolerance: same threshold used by test_quantize_single_block.
    // With scale ≈ 1.0 (range = 14.0, 15 steps) the max error is ≤ 0.51.
    let max_err = src
        .iter()
        .zip(&out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err <= 0.51,
        "max abs error {max_err:.4} > 0.51 for single-block misaligned input"
    );
}

/// Input shorter than one block (10 bytes < 20) must return an empty Vec.
#[test]
fn dequantize_row_q4_0_truncated_below_one_block() {
    let buf = vec![0xABu8; 10]; // 10 bytes — not even one complete block
    // Must not panic; chunks_exact(20) produces zero chunks → empty output.
    let out = dequantize_row_q4_0(&buf, 32);
    assert!(
        out.is_empty(),
        "expected empty Vec for sub-block input, got {} values",
        out.len()
    );
}

/// Clean 2-block (40-byte) input with n_weights=64 still returns 64 correct values.
/// This is a regression guard: removing the assert must not break the happy path.
#[test]
fn dequantize_row_q4_0_exact_blocks_unchanged() {
    let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
    let data = quantize_row_q4_0(&src).unwrap();
    assert_eq!(data.len(), 40, "2-block input must be 40 bytes");
    let out = dequantize_row_q4_0(&data, 64);
    assert_eq!(out.len(), 64);
    let max_err = src
        .iter()
        .zip(&out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.5,
        "max abs error {max_err:.4} >= 0.5 for exact 2-block input"
    );
}

// -----------------------------------------------------------------------
// Adversarial header guards (weight-loading sweep): a crafted .q4/.f16
// header must yield a clean Err, never an integer-overflow buffer or a
// process-aborting OOM allocation.
// -----------------------------------------------------------------------

#[test]
fn test_q4_rejects_huge_ndim() {
    // ndim = u32::MAX → unguarded Vec::with_capacity(ndim) is a ~34 GB OOM.
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHQ4");
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&u32::MAX.to_le_bytes());
    let path = std::path::PathBuf::from("/tmp/test_q4_huge_ndim.q4");
    std::fs::write(&path, &buf).unwrap();
    let r = load_q4_file(&path);
    std::fs::remove_file(&path).ok();
    assert!(
        r.is_err(),
        "u32::MAX ndim must be rejected, not OOM-aborted"
    );
}

#[test]
fn test_read_q4_header_rejects_huge_ndim() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHQ4");
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&u32::MAX.to_le_bytes());
    let path = std::path::PathBuf::from("/tmp/test_q4_header_huge_ndim.q4");
    std::fs::write(&path, &buf).unwrap();
    let mut file = std::fs::File::open(&path).unwrap();
    let r = read_q4_header(&mut file);
    std::fs::remove_file(&path).ok();
    assert!(
        r.is_err(),
        "u32::MAX ndim in read_q4_header must be rejected"
    );
}

#[test]
fn read_q4_header_positions_cursor_at_first_block() {
    use std::io::Read;

    let first_block = Q4Block {
        scale: q4_f32_to_f16(0.5),
        bias: q4_f32_to_f16(-1.0),
        packed: [0xA5; 16],
    };
    let tensor = Q4Tensor {
        blocks: vec![first_block],
        shape: vec![32],
        original_len: 32,
    };
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("cursor.q4");
    save_q4_file(&path, &tensor).unwrap();

    let mut file = std::fs::File::open(&path).unwrap();
    let header = read_q4_header(&mut file).unwrap();
    assert_eq!(header.payload_offset, 28);
    let mut actual = [0u8; Q4_BLOCK_BYTES];
    file.read_exact(&mut actual).unwrap();

    let mut expected = [0u8; Q4_BLOCK_BYTES];
    expected[0..2].copy_from_slice(&first_block.scale.to_ne_bytes());
    expected[2..4].copy_from_slice(&first_block.bias.to_ne_bytes());
    expected[4..].copy_from_slice(&first_block.packed);
    assert_eq!(actual, expected);
}

#[test]
fn test_q4_rejects_huge_original_len() {
    // original_len = 2^62 → unguarded n_blocks*20 is a ~2.9 EB OOM.
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHQ4");
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
    buf.extend_from_slice(&4u64.to_le_bytes()); // shape[0]
    buf.extend_from_slice(&(1u64 << 62).to_le_bytes()); // original_len
    let path = std::path::PathBuf::from("/tmp/test_q4_huge_len.q4");
    std::fs::write(&path, &buf).unwrap();
    let r = load_q4_file(&path);
    std::fs::remove_file(&path).ok();
    assert!(
        r.is_err(),
        "2^62 original_len must be rejected, not OOM-aborted"
    );
}

#[test]
fn test_q4_rejects_shape_product_mismatch() {
    // shape product (4*16=64) disagrees with original_len (32): the header
    // claims twice as many elements as the block payload covers. The
    // quantize paths reject this via assert_shape_matches_data_len; the
    // loader must too, with a clean Err rather than a Q4Tensor whose shape
    // overstates its data (downstream matmuls would read stale elements).
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHQ4");
    buf.extend_from_slice(&2u32.to_le_bytes()); // version
    buf.extend_from_slice(&2u32.to_le_bytes()); // ndim
    buf.extend_from_slice(&4u64.to_le_bytes()); // shape[0]
    buf.extend_from_slice(&16u64.to_le_bytes()); // shape[1] → product 64
    buf.extend_from_slice(&32u64.to_le_bytes()); // original_len (≠ 64)
    buf.extend_from_slice(&q4_f32_to_f16(1.0).to_ne_bytes());
    buf.extend_from_slice(&q4_f32_to_f16(0.0).to_ne_bytes());
    buf.extend_from_slice(&[0u8; 16]); // one valid block payload
    let path = std::path::PathBuf::from("/tmp/test_q4_shape_mismatch.q4");
    std::fs::write(&path, &buf).unwrap();
    let r = load_q4_file(&path);
    std::fs::remove_file(&path).ok();
    assert!(
        r.is_err(),
        "shape product 64 != original_len 32 must be rejected"
    );
}

#[test]
fn q4_ingress_rejects_invalid_scale_and_bias_metadata() {
    let cases = [
        (q4_f32_to_f16(f32::NAN), q4_f32_to_f16(0.0), "NaN scale"),
        (
            q4_f32_to_f16(f32::INFINITY),
            q4_f32_to_f16(0.0),
            "infinite scale",
        ),
        (
            q4_f32_to_f16(f32::NEG_INFINITY),
            q4_f32_to_f16(0.0),
            "negative infinite scale",
        ),
        (q4_f32_to_f16(0.0), q4_f32_to_f16(0.0), "zero scale"),
        (
            q4_f32_to_f16(-0.0),
            q4_f32_to_f16(0.0),
            "negative zero scale",
        ),
        (q4_f32_to_f16(1.0), q4_f32_to_f16(f32::NAN), "NaN bias"),
    ];

    for (scale, bias, label) in cases {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("invalid_metadata.q4");
        std::fs::write(&path, q4_file_bytes(&[32], 32, scale, bias)).unwrap();

        let err = load_q4_file(&path).expect_err(label);
        assert!(
            err.to_string().contains("block 0"),
            "{label} error must identify its block: {err}"
        );
    }
}

/// Structural proof that `validate_q4_file` does not perform a separate
/// full-payload block scan: a file with a structurally valid
/// header/geometry/extent but deliberately non-finite (NaN) block
/// scale/bias must be *accepted* by `validate_q4_file` alone, because
/// that function only checks header/geometry/extent. The exact same
/// bytes are still rejected by `load_q4_file`, which folds the
/// per-block finite check into the single read-and-decode pass it
/// already performs over the payload — proving the check moved rather
/// than disappeared.
///
/// This is a statement about the preflight in isolation, not about what
/// any load path admits: a consumer that never traverses the payload
/// discharges the same obligation eagerly via `open_and_mmap_q4_file`'s
/// `Q4BlockCheck::Now`, covered by the `mmap_entry_point_*` tests below.
///
/// Mutation sensitivity: re-adding a per-block scan loop to
/// `validate_q4_file` makes `validate_q4_file` itself reject this file,
/// so this test goes red on the first assertion.
#[test]
fn validate_q4_file_does_not_scan_block_payload() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("garbage_blocks.q4");
    std::fs::write(
        &path,
        q4_file_bytes(&[32], 32, q4_f32_to_f16(f32::NAN), q4_f32_to_f16(f32::NAN)),
    )
    .unwrap();

    let mut file = std::fs::File::open(&path).unwrap();
    let result = validate_q4_file(&mut file, &path, Some(&[32]));
    assert!(
        result.is_ok(),
        "validate_q4_file must not scan block payload bytes, but got: {:?}",
        result.err()
    );

    let load_result = load_q4_file(&path);
    assert!(
        load_result.is_err(),
        "load_q4_file must still reject non-finite block metadata, folded into its own \
             single payload read"
    );
}

/// Write a single-block `.q4` file with the given scale/bias bit patterns
/// and return its path, keeping `tmp` alive in the caller.
fn write_single_block_q4(tmp: &tempfile::TempDir, scale: u16, bias: u16) -> std::path::PathBuf {
    let path = tmp.path().join("block_metadata.q4");
    std::fs::write(&path, q4_file_bytes(&[32], 32, scale, bias)).unwrap();
    path
}

/// The no-copy mapping path never decodes a block on the CPU, so a NaN
/// scale reaches the GPU kernel unless the mapping entry point checks it.
/// Exercised without a Metal device: the guard lives in
/// `open_and_mmap_q4_file`, not in the Metal buffer construction that
/// follows it, so a machine with no GPU still runs the assertion instead
/// of skipping past it.
#[test]
fn mmap_entry_point_rejects_nan_scale_when_caller_does_not_traverse() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_single_block_q4(&tmp, q4_f32_to_f16(f32::NAN), q4_f32_to_f16(0.0));

    let Err(err) = open_and_mmap_q4_file(
        &path,
        Some(&[32]),
        Q4BlockCheck::Now {
            tensor_name: "nan scale weight",
        },
    ) else {
        panic!("NaN block scale must not reach a no-copy GPU buffer");
    };
    assert!(
        err.contains("block 0"),
        "rejection must name the offending block: {err}"
    );
}

#[test]
fn mmap_entry_point_rejects_infinite_scale_when_caller_does_not_traverse() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_single_block_q4(&tmp, q4_f32_to_f16(f32::INFINITY), q4_f32_to_f16(0.0));

    let Err(err) = open_and_mmap_q4_file(
        &path,
        Some(&[32]),
        Q4BlockCheck::Now {
            tensor_name: "infinite scale weight",
        },
    ) else {
        panic!("infinite block scale must not reach a no-copy GPU buffer");
    };
    assert!(
        err.contains("block 0"),
        "rejection must name the offending block: {err}"
    );
}

#[test]
fn mmap_entry_point_rejects_nan_bias_when_caller_does_not_traverse() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_single_block_q4(&tmp, q4_f32_to_f16(1.0), q4_f32_to_f16(f32::NAN));

    let Err(err) = open_and_mmap_q4_file(
        &path,
        Some(&[32]),
        Q4BlockCheck::Now {
            tensor_name: "nan bias weight",
        },
    ) else {
        panic!("NaN block bias must not reach a no-copy GPU buffer");
    };
    assert!(
        err.contains("block 0"),
        "rejection must name the offending block: {err}"
    );
}

/// The rejections above must come from the requested check, not from the
/// header preflight both variants share: the identical bytes are accepted
/// under `InCallerTraversal`, where the caller's own decode loop is what
/// discharges the obligation.
#[test]
fn mmap_entry_point_defers_block_check_to_a_traversing_caller() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_single_block_q4(&tmp, q4_f32_to_f16(f32::NAN), q4_f32_to_f16(f32::NAN));

    let result = open_and_mmap_q4_file(
        &path,
        Some(&[32]),
        Q4BlockCheck::InCallerTraversal {
            traversal: "test stand-in for a caller decode loop",
        },
    );
    let (_header, _mmap, checked) = result.unwrap_or_else(|e| {
        panic!(
            "a traversing caller validates during its own pass, so the mapping must be \
                 handed out here: {e}"
        )
    });
    assert!(
        checked.is_none(),
        "a deferred check must not hand out the witness that lets bytes be published \
             to a consumer which never decodes them"
    );
}

/// A well-formed file still maps cleanly under the eager check, and the
/// header it returns still describes the same payload.
#[test]
fn mmap_entry_point_accepts_well_formed_q4_file() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_single_block_q4(&tmp, q4_f32_to_f16(0.25), q4_f32_to_f16(-1.0));

    let (header, mmap, checked) = open_and_mmap_q4_file(
        &path,
        Some(&[32]),
        Q4BlockCheck::Now {
            tensor_name: "well formed weight",
        },
    )
    .expect("a well-formed Q4 file must still load through the eager-check path");
    assert!(
        checked.is_some(),
        "an eagerly checked mapping must yield the witness a no-copy consumer needs"
    );
    assert_eq!(header.shape, vec![32]);
    assert_eq!(header.original_len, 32);
    assert_eq!(
        mmap.len() as u64,
        header.payload_offset + Q4_BLOCK_BYTES as u64
    );
}

#[test]
fn q4_ingress_rejects_trailing_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("trailing.q4");
    let mut bytes = q4_file_bytes(&[32], 32, q4_f32_to_f16(1.0), q4_f32_to_f16(0.0));
    bytes.push(0xAA);
    std::fs::write(&path, bytes).unwrap();

    let err = load_q4_file(&path).expect_err("trailing byte must be rejected");
    assert!(
        err.to_string().contains("trailing"),
        "unexpected error: {err}"
    );
}

#[test]
fn checked_native_loaders_reject_same_numel_transposed_geometry() {
    let tmp = tempfile::tempdir().unwrap();
    let q4_path = tmp.path().join("transposed.q4");
    std::fs::write(
        &q4_path,
        q4_file_bytes(&[2, 32], 64, q4_f32_to_f16(1.0), q4_f32_to_f16(0.0)),
    )
    .unwrap();
    let f16_path = tmp.path().join("transposed.f16");
    std::fs::write(&f16_path, f16_file_bytes(&[2, 32], &[0u16; 64])).unwrap();

    let q4_err = load_q4_file_checked(&q4_path, &[32, 2])
        .expect_err("same-numel transposed Q4 shape must be rejected");
    assert!(q4_err.to_string().contains("expected [32, 2]"));
    let f16_err = load_f16_tensor_file_checked(&f16_path, &[32, 2])
        .expect_err("same-numel transposed F16 shape must be rejected");
    assert!(f16_err.to_string().contains("expected [32, 2]"));
}

// -----------------------------------------------------------------------
// validate_q4_header_payload_bounds (issue #540): the Metal no-copy mmap
// loader has no `read_exact` to fail short on a truncated block payload
// the way `load_q4_file` does, so this helper is the sole fail-closed
// gate before a `.q4` file's mmap is handed to Metal dispatch.
// -----------------------------------------------------------------------

#[test]
fn test_validate_q4_header_payload_bounds_rejects_truncated_payload() {
    // original_len=64 → 2 blocks × 20 bytes = 40 required payload bytes;
    // file ends exactly at payload_offset (zero payload bytes present).
    let header = Q4FileHeader {
        shape: vec![64],
        original_len: 64,
        payload_offset: 28,
    };
    let r = validate_q4_header_payload_bounds(&header, 28, &std::path::PathBuf::from("t.q4"));
    assert!(
        r.is_err(),
        "file truncated to payload_offset must be rejected"
    );
}

#[test]
fn test_validate_q4_header_payload_bounds_rejects_one_byte_short() {
    let header = Q4FileHeader {
        shape: vec![64],
        original_len: 64,
        payload_offset: 28,
    };
    // Required length is payload_offset (28) + 40 = 68; one byte short.
    let r = validate_q4_header_payload_bounds(&header, 67, &std::path::PathBuf::from("t.q4"));
    assert!(r.is_err(), "payload one byte short of required must fail");
}

#[test]
fn test_validate_q4_header_payload_bounds_accepts_exact_length() {
    let header = Q4FileHeader {
        shape: vec![64],
        original_len: 64,
        payload_offset: 28,
    };
    let r = validate_q4_header_payload_bounds(&header, 68, &std::path::PathBuf::from("t.q4"));
    assert!(
        r.is_ok(),
        "file with exactly the required payload bytes must be accepted: {r:?}"
    );
}

#[test]
fn test_validate_q4_header_payload_bounds_rejects_trailing_byte() {
    let header = Q4FileHeader {
        shape: vec![32],
        original_len: 32,
        payload_offset: 28,
    };
    let r = validate_q4_header_payload_bounds(&header, 49, std::path::Path::new("t.q4"));
    let err = r.expect_err("one trailing byte must be rejected");
    assert!(err.to_string().contains("trailing"));
}

#[test]
fn test_validate_q4_header_payload_bounds_rejects_shape_mismatch() {
    let header = Q4FileHeader {
        shape: vec![4, 16], // product 64
        original_len: 32,   // disagrees with shape product
        payload_offset: 36,
    };
    let r = validate_q4_header_payload_bounds(&header, 56, std::path::Path::new("t.q4"));
    assert!(
        r.is_err(),
        "shape product != original_len must be rejected before a payload-length check"
    );
}

#[test]
fn test_validate_q4_header_payload_bounds_rejects_huge_original_len_overflow() {
    // original_len near usize::MAX must not panic on overflow in the
    // block-count/byte-count arithmetic; it must return a clean Err.
    let header = Q4FileHeader {
        shape: vec![usize::MAX],
        original_len: usize::MAX,
        payload_offset: 28,
    };
    let r = validate_q4_header_payload_bounds(&header, 1_000, &std::path::PathBuf::from("t.q4"));
    assert!(
        r.is_err(),
        "huge original_len must be rejected, not panic on overflow"
    );
}

#[test]
fn test_f16_rejects_huge_numel() {
    // numel = 2^63 → unguarded numel*2 overflows usize to 0, silently
    // returning ([], [shape]) — wrong data with no error. Must be Err now.
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHF1");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
    buf.extend_from_slice(&(1u64 << 63).to_le_bytes()); // shape[0]
    buf.extend_from_slice(&(1u64 << 63).to_le_bytes()); // numel
    let path = std::path::PathBuf::from("/tmp/test_f16_huge_numel.f16");
    std::fs::write(&path, &buf).unwrap();
    let r = load_f16_tensor_file(&path);
    std::fs::remove_file(&path).ok();
    assert!(
        r.is_err(),
        "2^63 numel must be rejected, not silently truncated to empty"
    );
}

#[test]
fn test_f16_rejects_shape_numel_mismatch() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHF1");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes()); // ndim
    buf.extend_from_slice(&2u64.to_le_bytes()); // shape[0]
    buf.extend_from_slice(&2u64.to_le_bytes()); // shape[1]
    buf.extend_from_slice(&1u64.to_le_bytes()); // numel
    buf.extend_from_slice(&0u16.to_le_bytes()); // one valid f16 payload value
    let path = std::path::PathBuf::from("/tmp/test_f16_shape_numel_mismatch.f16");
    std::fs::write(&path, &buf).unwrap();
    let r = load_f16_tensor_file(&path);
    std::fs::remove_file(&path).ok();
    let err = r.expect_err("shape product != numel must be rejected");
    assert!(
        err.to_string().contains("shape product"),
        "unexpected error: {err}"
    );
}

#[test]
fn f16_ingress_rejects_non_finite_values_and_trailing_bytes() {
    for (bits, label) in [
        (q4_f32_to_f16(f32::NAN), "NaN"),
        (q4_f32_to_f16(f32::INFINITY), "infinity"),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("non_finite.f16");
        std::fs::write(&path, f16_file_bytes(&[1], &[bits])).unwrap();
        let err = load_f16_tensor_file(&path).expect_err(label);
        assert!(err.to_string().contains("element index 0"));
    }

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("trailing.f16");
    let mut bytes = f16_file_bytes(&[1], &[q4_f32_to_f16(1.0)]);
    bytes.push(0xAA);
    std::fs::write(&path, bytes).unwrap();
    let err = load_f16_tensor_file(&path).expect_err("trailing byte must be rejected");
    assert!(
        err.to_string().contains("trailing"),
        "unexpected error: {err}"
    );
}

#[test]
fn quantizers_return_errors_for_shape_mismatches() {
    assert!(quantize_tensor_q4_0(&[0.0], 1, 2).is_err());
    assert!(quantize_bf16_to_q4(&[0], &[2]).is_err());
    assert!(quantize_f32_to_q4(&[0.0], &[2]).is_err());
    assert!(quantize_f64_to_q4_mode(&[0.0], &[2], true).is_err());
}

#[test]
fn quantizer_rejects_f16_metadata_overflow() {
    let err = quantize_row_q4_0(&[f32::MAX; 32])
        .expect_err("finite source whose serialized bias overflows f16 must be rejected");
    assert!(err.to_string().contains("f16"));

    let mut symmetric = [0.0f64; 32];
    symmetric[0] = 500_000.0;
    let err = quantize_f64_to_q4_mode(&symmetric, &[32], true)
        .expect_err("finite source whose serialized symmetric metadata overflows f16");
    assert!(err.to_string().contains("f16"));
}

#[test]
fn test_f16_rejects_huge_ndim() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHF1");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&u32::MAX.to_le_bytes());
    let path = std::path::PathBuf::from("/tmp/test_f16_huge_ndim.f16");
    std::fs::write(&path, &buf).unwrap();
    let r = load_f16_tensor_file(&path);
    std::fs::remove_file(&path).ok();
    assert!(
        r.is_err(),
        "u32::MAX ndim in .f16 must be rejected, not OOM-aborted"
    );
}

#[test]
fn test_q4_rejects_original_len_near_usize_max() {
    // original_len = usize::MAX - 3, with a single-dim shape equal to
    // original_len so shape_product == original_len and the loader
    // reaches the block-payload guard (not the earlier shape-mismatch
    // guard). n_blocks*20 does not itself overflow u64 at this
    // magnitude, so this exercises the file_len-bound branch of
    // checked_alloc_bytes: it must return a clean Err, never panic or
    // attempt a multi-exabyte allocation. Removing the `checked_mul`
    // guard (reverting to `n_blocks * 20`) does not panic here either
    // since the multiply itself doesn't overflow — this test instead
    // proves the *file_len bound* check is load-bearing on its own.
    let huge = usize::MAX - 3;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHQ4");
    buf.extend_from_slice(&2u32.to_le_bytes()); // version
    buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
    buf.extend_from_slice(&(huge as u64).to_le_bytes()); // shape[0] == original_len
    buf.extend_from_slice(&(huge as u64).to_le_bytes()); // original_len
    let path = std::path::PathBuf::from("/tmp/test_q4_original_len_near_usize_max.q4");
    std::fs::write(&path, &buf).unwrap();
    let r = load_q4_file(&path);
    std::fs::remove_file(&path).ok();
    let err = r.expect_err("original_len near usize::MAX must be rejected, not panic/OOM");
    let msg = err.to_string();
    assert!(
        msg.contains("block payload") || msg.contains("header claims"),
        "expected the block-payload allocation guard to fire, got: {msg}"
    );
}

#[test]
fn test_f16_rejects_numel_whose_byte_count_exceeds_file_len() {
    // numel = usize::MAX / 4: numel*2 does NOT overflow (≈ 2^62), so the
    // checked_mul branch passes and rejection can only come from the
    // file_len-bound branch of checked_alloc_bytes. This pins that branch
    // for the f16 loader specifically — the assertion below must not
    // accept "overflows usize", or a deleted file_len check would go
    // unnoticed (the overflow branch is pinned separately by
    // test_f16_rejects_numel_that_wraps_to_small_value_on_overflow).
    let huge = usize::MAX / 4;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHF1");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
    buf.extend_from_slice(&(huge as u64).to_le_bytes()); // shape[0]
    buf.extend_from_slice(&(huge as u64).to_le_bytes()); // numel
    let path = std::path::PathBuf::from("/tmp/test_f16_numel_exceeds_file_len.f16");
    std::fs::write(&path, &buf).unwrap();
    let r = load_f16_tensor_file(&path);
    std::fs::remove_file(&path).ok();
    let err = r.expect_err("oversized f16 numel must be rejected, not panic/OOM");
    let msg = err.to_string();
    assert!(
        msg.contains("f16 data") && msg.contains("header claims"),
        "expected the f16-data file_len-bound guard to fire, got: {msg}"
    );
}

#[test]
fn test_f16_rejects_numel_that_wraps_to_small_value_on_overflow() {
    // numel = usize::MAX/2 + 5: numel*2 overflows u64 and wraps to a
    // *small* residual (10, mod 2^64) that would sail past the
    // file_len-bound check if `checked_mul` were replaced by a plain
    // wrapping multiply — a silent-corruption bug (an ~empty read
    // reported as success with the wrong shape/numel) rather than the
    // OOM/panic the near-usize::MAX test above guards against. This is
    // the scenario `checked_mul` uniquely defends: the bound check alone
    // cannot catch it because the wrapped byte count looks small.
    let huge = usize::MAX / 2 + 5;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHF1");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
    buf.extend_from_slice(&(huge as u64).to_le_bytes()); // shape[0]
    buf.extend_from_slice(&(huge as u64).to_le_bytes()); // numel
    // Trailing filler: `huge * 2` wraps to a small residue (10 bytes) if
    // `checked_mul` is bypassed, so pad enough real bytes that a buggy
    // wrapping multiply would successfully `read_exact` a plausible
    // (wrong) buffer instead of also failing on a short read — isolating
    // the assertion to the overflow guard itself, not an incidental
    // short-file error.
    buf.extend_from_slice(&[0xABu8; 64]);
    let path = std::path::PathBuf::from("/tmp/test_f16_numel_wraps_to_small_value_on_overflow.f16");
    std::fs::write(&path, &buf).unwrap();
    let r = load_f16_tensor_file(&path);
    std::fs::remove_file(&path).ok();
    let err = r.expect_err(
        "numel whose ×2 wraps to a small value must still be rejected via checked_mul, \
             not silently accepted as a tiny (wrong) allocation",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("overflows usize"),
        "expected the checked_mul overflow branch specifically, got: {msg}"
    );
}

// -----------------------------------------------------------------------
// Non-finite input guard — mutation-sensitive tests (Finding 1, PR #452)
//
// IEEE-754: `NaN > x` and `NaN < x` are always false, so a plain
// `f32::max` / `f32::min` fold over a block that contains NaN silently
// ignores the NaN element and computes scale from the finite elements
// only. The NaN then quantizes to nibble 0 via a saturating cast, so
// no panic occurs and the caller receives a plausible-looking Q4Block
// with a silently wrong entry. The guard at the top of
// `quantize_block_with_mode` must catch this before the fold.
//
// Mutation sensitivity: removing the `if !v.is_finite()` guard converts
// both `Err` returns below to `Ok`, turning `result.is_err()` → false
// and failing the assertion.
// -----------------------------------------------------------------------

#[test]
fn test_quantize_block_rejects_nan_input() {
    // Block with one NaN among otherwise-valid weights must return Err.
    let mut vals = vec![1.0f32; 32];
    vals[7] = f32::NAN;
    let result = quantize_row_q4_0(&vals);
    assert!(
        result.is_err(),
        "NaN in weight block must be rejected with InvalidInput"
    );
}

#[test]
fn test_quantize_block_rejects_inf_input() {
    // Block with one +inf element must return Err; the guard covers both
    // +inf and -inf via `is_finite()` (which returns false for any
    // non-finite value, including NaN, +inf, and -inf).
    let mut vals = vec![1.0f32; 32];
    vals[15] = f32::INFINITY;
    let result = quantize_row_q4_0(&vals);
    assert!(
        result.is_err(),
        "+inf in weight block must be rejected with InvalidInput"
    );
}

// -----------------------------------------------------------------------
// Merge-on-first-load `merged_qkvz_*.q4` cache — content-integrity guard
// (#504 remaining slice: "Merged-Q4 cache: compatibility check is
// size-only — no content integrity on the merged artifact.")
//
// Mutation sensitivity: `merged_qkvz_cache_is_valid`'s size check alone
// (the pre-fix behavior) would accept a same-size tampered/stale merged
// file. `test_merged_qkvz_cache_rejects_same_size_corrupted_payload` and
// `test_merged_qkvz_cache_rejects_same_size_stale_source` are the
// discriminating tests: reverting the fingerprint comparison back to a
// bare `metadata.len() == expected_size` check makes both pass
// incorrectly (`is_valid` would wrongly return `true`), so they fail
// under the reverted code. Verified manually per the task's mutation-test
// protocol — see the session report for the reverse-apply/touch/restore
// proof.
// -----------------------------------------------------------------------

/// Write a minimal valid 2-D `.q4` source file (`shape = [rows, cols]`,
/// `rows * cols` must be a multiple of 32) with content derived from
/// `seed` so distinct seeds produce distinct payload bytes.
fn write_test_q4_source(path: &std::path::Path, rows: usize, cols: usize, seed: f32) {
    let n = rows * cols;
    let f32_vals: Vec<f32> = (0..n).map(|i| (i as f32 + seed) % 7.0 - 3.0).collect();
    let bf16_vals = to_bf16(&f32_vals);
    let tensor = quantize_bf16_to_q4(&bf16_vals, &[rows, cols]).unwrap();
    save_q4_file(path, &tensor).unwrap();
}

/// Build a fresh temp-dir-scoped triple of (qkv_path, z_path, merged_path)
/// for one test, so parallel `cargo test` runs never collide on the same
/// file. `qkv_seed`/`z_seed` control the source payload content.
fn merge_test_paths(name: &str) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("lattice_test_merged_qkvz_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    (dir.join("qkv.q4"), dir.join("z.q4"), dir.join("merged.q4"))
}

#[test]
fn test_merged_qkvz_expected_size_computes_correctly() {
    // 36-byte header each; qkv payload = 100 bytes, z payload = 40 bytes.
    let expected = merged_qkvz_expected_size(136, 76).unwrap();
    assert_eq!(expected, 36 + 100 + 40);
}

#[test]
fn test_merged_qkvz_expected_size_rejects_truncated_source() {
    // A source file shorter than its own 36-byte header must fail
    // closed (`Err`), not underflow/panic via unchecked subtraction.
    let err = merged_qkvz_expected_size(20, 136).unwrap_err();
    assert!(
        err.contains("too small"),
        "expected a too-small error, got: {err}"
    );
}

#[test]
fn test_write_merged_qkvz_then_cache_is_valid() {
    let (qkv_p, z_p, merged_p) = merge_test_paths("valid");
    write_test_q4_source(&qkv_p, 4, 8, 1.0);
    write_test_q4_source(&z_p, 4, 8, 5.0);

    write_merged_qkvz(&qkv_p, &z_p, &merged_p).unwrap();

    let qkv_len = std::fs::metadata(&qkv_p).unwrap().len();
    let z_len = std::fs::metadata(&z_p).unwrap().len();
    let expected_size = merged_qkvz_expected_size(qkv_len, z_len).unwrap();

    assert!(
        merged_qkvz_cache_is_valid(&merged_p, expected_size, &qkv_p, &z_p),
        "freshly written merged cache must validate against its own sources"
    );

    std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
}

#[test]
fn test_write_merged_qkvz_rejects_oversized_source_payload() {
    // A source file whose payload exceeds MAX_Q4_MERGE_PAYLOAD_LEN must
    // fail closed BEFORE any payload allocation — the rebuild path uses
    // the same bounded reader as the validator. `set_len` produces a
    // sparse file, so this asserts the cap without 2 GiB of disk I/O.
    let (qkv_p, z_p, merged_p) = merge_test_paths("oversized_source");
    write_test_q4_source(&qkv_p, 4, 8, 1.0);
    write_test_q4_source(&z_p, 4, 8, 5.0);

    let f = std::fs::File::options().write(true).open(&qkv_p).unwrap();
    f.set_len(36 + MAX_Q4_MERGE_PAYLOAD_LEN + 1).unwrap();
    drop(f);

    let err = write_merged_qkvz(&qkv_p, &z_p, &merged_p)
        .expect_err("oversized source payload must be rejected, not read to EOF");
    assert!(
        err.contains("payload too large"),
        "expected a payload-cap error, got: {err}"
    );
    assert!(
        !merged_p.exists(),
        "no merged artifact may be produced from a rejected source"
    );

    std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
}

#[test]
fn test_write_merged_qkvz_rejects_rank0_qkv_header_instead_of_panicking() {
    // A structurally valid-per-`validate_q4_file` but rank-0 (`ndim = 0`)
    // qkv source: shape = [], original_len = 1 (matches the empty
    // shape's element-count product of 1), one valid block. Before the
    // 2-D guard, `write_merged_qkvz` unconditionally indexed
    // `qkv_hdr.shape[0]`, which panics (denial of service) on this input
    // instead of returning `Err`.
    let (qkv_p, z_p, merged_p) = merge_test_paths("rank0_qkv");
    std::fs::write(
        &qkv_p,
        q4_file_bytes(&[], 1, q4_f32_to_f16(1.0), q4_f32_to_f16(0.0)),
    )
    .unwrap();
    write_test_q4_source(&z_p, 4, 8, 5.0);

    let err = write_merged_qkvz(&qkv_p, &z_p, &merged_p)
        .expect_err("rank-0 qkv header must be rejected, not panic");
    assert!(
        err.contains("not 2-D"),
        "expected a 2-D shape error, got: {err}"
    );
    assert!(
        !merged_p.exists(),
        "no merged artifact may be produced from a rejected source"
    );

    std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
}

#[test]
fn test_write_merged_qkvz_rejects_rank0_z_header_instead_of_panicking() {
    let (qkv_p, z_p, merged_p) = merge_test_paths("rank0_z");
    write_test_q4_source(&qkv_p, 4, 8, 1.0);
    std::fs::write(
        &z_p,
        q4_file_bytes(&[], 1, q4_f32_to_f16(1.0), q4_f32_to_f16(0.0)),
    )
    .unwrap();

    let err = write_merged_qkvz(&qkv_p, &z_p, &merged_p)
        .expect_err("rank-0 z header must be rejected, not panic");
    assert!(
        err.contains("not 2-D"),
        "expected a 2-D shape error, got: {err}"
    );
    assert!(
        !merged_p.exists(),
        "no merged artifact may be produced from a rejected source"
    );

    std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
}

#[test]
fn test_merged_qkvz_cache_rejects_missing_file() {
    let (qkv_p, z_p, merged_p) = merge_test_paths("missing");
    write_test_q4_source(&qkv_p, 4, 8, 1.0);
    write_test_q4_source(&z_p, 4, 8, 5.0);
    let qkv_len = std::fs::metadata(&qkv_p).unwrap().len();
    let z_len = std::fs::metadata(&z_p).unwrap().len();
    let expected_size = merged_qkvz_expected_size(qkv_len, z_len).unwrap();

    // merged_p was never written.
    assert!(!merged_qkvz_cache_is_valid(
        &merged_p,
        expected_size,
        &qkv_p,
        &z_p
    ));

    std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
}

#[test]
fn test_merged_qkvz_cache_rejects_wrong_size() {
    let (qkv_p, z_p, merged_p) = merge_test_paths("wrongsize");
    write_test_q4_source(&qkv_p, 4, 8, 1.0);
    write_test_q4_source(&z_p, 4, 8, 5.0);
    write_merged_qkvz(&qkv_p, &z_p, &merged_p).unwrap();

    // Append a stray byte, making the on-disk size disagree with
    // `expected_size`.
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&merged_p)
            .unwrap();
        f.write_all(&[0xAA]).unwrap();
    }

    let qkv_len = std::fs::metadata(&qkv_p).unwrap().len();
    let z_len = std::fs::metadata(&z_p).unwrap().len();
    let expected_size = merged_qkvz_expected_size(qkv_len, z_len).unwrap();

    assert!(!merged_qkvz_cache_is_valid(
        &merged_p,
        expected_size,
        &qkv_p,
        &z_p
    ));

    std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
}

#[test]
fn test_merged_qkvz_cache_rejects_truncated_file() {
    let (qkv_p, z_p, merged_p) = merge_test_paths("truncated");
    write_test_q4_source(&qkv_p, 4, 8, 1.0);
    write_test_q4_source(&z_p, 4, 8, 5.0);
    write_merged_qkvz(&qkv_p, &z_p, &merged_p).unwrap();

    let full_len = std::fs::metadata(&merged_p).unwrap().len();
    let bytes = std::fs::read(&merged_p).unwrap();
    std::fs::write(&merged_p, &bytes[..bytes.len() - 10]).unwrap();

    let qkv_len = std::fs::metadata(&qkv_p).unwrap().len();
    let z_len = std::fs::metadata(&z_p).unwrap().len();
    let expected_size = merged_qkvz_expected_size(qkv_len, z_len).unwrap();
    assert_eq!(expected_size, full_len, "sanity: source sizes unchanged");

    assert!(!merged_qkvz_cache_is_valid(
        &merged_p,
        expected_size,
        &qkv_p,
        &z_p
    ));

    std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
}

#[test]
fn test_merged_qkvz_cache_rejects_same_size_corrupted_payload() {
    // Same-size bit flip inside the merged payload — the pre-fix
    // size-only check would accept this file unchanged. This is the
    // mutation-sensitive case for the content-integrity fix itself.
    let (qkv_p, z_p, merged_p) = merge_test_paths("corrupted");
    write_test_q4_source(&qkv_p, 4, 8, 1.0);
    write_test_q4_source(&z_p, 4, 8, 5.0);
    write_merged_qkvz(&qkv_p, &z_p, &merged_p).unwrap();

    let qkv_len = std::fs::metadata(&qkv_p).unwrap().len();
    let z_len = std::fs::metadata(&z_p).unwrap().len();
    let expected_size = merged_qkvz_expected_size(qkv_len, z_len).unwrap();
    let full_len = std::fs::metadata(&merged_p).unwrap().len();
    assert_eq!(
        full_len, expected_size,
        "sanity: size unchanged by corruption"
    );

    // Flip one byte well inside the payload region (after the 36-byte
    // header) without changing the file's length.
    let mut bytes = std::fs::read(&merged_p).unwrap();
    let flip_at = bytes.len() - 5;
    bytes[flip_at] ^= 0xFF;
    std::fs::write(&merged_p, &bytes).unwrap();

    assert_eq!(
        std::fs::metadata(&merged_p).unwrap().len(),
        expected_size,
        "sanity: byte flip must not change file size"
    );

    assert!(
        !merged_qkvz_cache_is_valid(&merged_p, expected_size, &qkv_p, &z_p),
        "a same-size, bit-flipped merged payload must fail the content-integrity check \
             even though the size-only check would have accepted it"
    );

    std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
}

#[test]
fn test_merged_qkvz_cache_rejects_same_size_stale_source() {
    // The z source file changes content (e.g. a re-quantize with
    // different weights) but keeps the exact same byte length, so the
    // merged filename (which encodes only sizes) and `expected_size`
    // are both unchanged. The stale merged cache must still be
    // rejected once content is checked.
    let (qkv_p, z_p, merged_p) = merge_test_paths("stale");
    write_test_q4_source(&qkv_p, 4, 8, 1.0);
    write_test_q4_source(&z_p, 4, 8, 5.0);
    write_merged_qkvz(&qkv_p, &z_p, &merged_p).unwrap();

    let qkv_len = std::fs::metadata(&qkv_p).unwrap().len();
    let z_len_before = std::fs::metadata(&z_p).unwrap().len();

    // Re-write z with different content but the same shape (same size).
    write_test_q4_source(&z_p, 4, 8, 99.0);
    let z_len_after = std::fs::metadata(&z_p).unwrap().len();
    assert_eq!(
        z_len_before, z_len_after,
        "sanity: same shape must produce the same file size"
    );

    let expected_size = merged_qkvz_expected_size(qkv_len, z_len_after).unwrap();
    assert_eq!(
        std::fs::metadata(&merged_p).unwrap().len(),
        expected_size,
        "sanity: merged file size still matches (source size unchanged)"
    );

    assert!(
        !merged_qkvz_cache_is_valid(&merged_p, expected_size, &qkv_p, &z_p),
        "a same-size stale source must invalidate the merged cache once content is checked"
    );

    std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
}

#[test]
fn test_merged_qkvz_source_fingerprint_matches_file_fingerprint_after_write() {
    let (qkv_p, z_p, merged_p) = merge_test_paths("fingerprint");
    write_test_q4_source(&qkv_p, 4, 8, 2.0);
    write_test_q4_source(&z_p, 4, 8, 6.0);
    write_merged_qkvz(&qkv_p, &z_p, &merged_p).unwrap();

    let source_fp = merged_qkvz_source_fingerprint(&qkv_p, &z_p).unwrap();
    let file_fp = merged_qkvz_file_fingerprint(&merged_p).unwrap();
    assert_eq!(
        source_fp, file_fp,
        "a freshly written merged file's payload fingerprint must equal its sources' fingerprint"
    );

    std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
}

/// `Display` must not repeat what `source()` already exposes.
///
/// These two assertions fail in opposite directions, which is the point: reverting
/// `Display` to interpolate the wrapped error trips the first, and dropping the
/// `source()` implementation trips the second. A chain-printing consumer needs both
/// halves to hold, and neither is observable from the other.
#[test]
fn f16_load_error_display_does_not_duplicate_its_source() {
    use std::error::Error;

    const CAUSE: &str = "sentinel-cause-text";
    let err = F16LoadError::Other(Box::new(std::io::Error::other(CAUSE)));

    let shown = format!("{err}");
    assert!(
        !shown.contains(CAUSE),
        "Display must describe only this wrapper's own contribution, but it \
             interpolated the wrapped cause: {shown:?}"
    );

    let source = err.source().expect("Other must keep its cause reachable");
    assert!(
        format!("{source}").contains(CAUSE),
        "source() must yield the wrapped cause itself"
    );
}

//! Independent integration tests for the W3 3-bit MLP weight format (issue #420).
//!
//! Written by the tester agent, separate from the implementer's inline unit
//! tests in `weights/w3_weights.rs`. These exercise the *public* API only
//! (`lattice_inference::weights::w3_weights::*`) and target scenarios not
//! covered by the inline suite: per-row group-boundary correctness for
//! non-multiple-of-32 column counts, degenerate (all-same-value) scale
//! handling at the full tensor API, near-boundary block counts, and a
//! save/load round-trip through a real temp file. Every test is designed to
//! fail if the corresponding implementation logic is reverted or broken.
//!
//! Run: cargo test --test w3_weights_integration

use lattice_inference::weights::w3_weights::{
    W3_BLOCK_SIZE, W3_GROUP_SIZE, dequantize_row_w3, dequantize_w3_to_f32, load_w3_file,
    quantize_f32_to_w3, quantize_row_w3, quantize_tensor_w3, save_w3_file,
};

// ---------------------------------------------------------------------------
// Happy path: pack -> dequant round-trip stays within the documented
// half-quantization-step error bound (scale = range / 7, max err = scale / 2).
// ---------------------------------------------------------------------------

#[test]
fn test_happy_path_roundtrip_within_half_step_error_bound() {
    // 256 values spanning a wide, non-trivial range across multiple blocks.
    let n = 256usize;
    let src: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.37).sin() * 100.0 + (i as f32) * 0.5)
        .collect();

    let packed = quantize_row_w3(&src).unwrap();
    assert_eq!(
        packed.len(),
        n.div_ceil(W3_GROUP_SIZE) * W3_BLOCK_SIZE,
        "packed byte length must match block count * 16"
    );

    let out = dequantize_row_w3(&packed, n);
    assert_eq!(out.len(), n);

    // Verify per-block error bound independently (per-group scale varies).
    for (block_idx, chunk) in src.chunks(W3_GROUP_SIZE).enumerate() {
        let min_v = chunk.iter().copied().fold(f32::INFINITY, f32::min);
        let max_v = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = (max_v - min_v) / 7.0;
        let half_step = scale / 2.0;
        let start = block_idx * W3_GROUP_SIZE;
        for (i, &orig) in chunk.iter().enumerate() {
            let got = out[start + i];
            let err = (got - orig).abs();
            assert!(
                err <= half_step + 1e-3,
                "block {block_idx} elem {i}: error {err} exceeds half-step bound {half_step} \
                 (orig={orig}, got={got})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Edge: group boundaries for tensors whose column count is NOT a multiple of
// W3_GROUP_SIZE (32). Each row must be quantized independently — a group must
// never straddle a row boundary. This is mutation-sensitive: if the packer
// were changed to flatten the whole tensor before chunking (instead of
// chunking per row), the padding/block-count arithmetic below would diverge
// and the byte-length assertion (or the row-isolation assertion) would fail.
// ---------------------------------------------------------------------------

#[test]
fn test_group_boundary_non_multiple_of_group_size_per_row() {
    let rows = 3usize;
    let cols = 50usize; // not a multiple of 32 -> 2 blocks/row (32 + 18 padded)
    let src: Vec<f32> = (0..rows * cols).map(|i| (i as f32 - 75.0) / 3.0).collect();

    let packed = quantize_tensor_w3(&src, rows, cols).unwrap();
    let blocks_per_row = cols.div_ceil(W3_GROUP_SIZE);
    assert_eq!(
        blocks_per_row, 2,
        "50 cols must need 2 blocks (32 + 18 padded)"
    );
    assert_eq!(
        packed.len(),
        rows * blocks_per_row * W3_BLOCK_SIZE,
        "per-row block count must not merge across row boundaries"
    );

    // Decode each row independently using its own byte slice and compare
    // against the corresponding source row — this fails if row quantization
    // ever bleeds min/max stats across the row boundary.
    let row_bytes = blocks_per_row * W3_BLOCK_SIZE;
    for r in 0..rows {
        let row_src = &src[r * cols..(r + 1) * cols];
        let row_packed = &packed[r * row_bytes..(r + 1) * row_bytes];
        let row_out = dequantize_row_w3(row_packed, cols);
        assert_eq!(row_out.len(), cols);

        // Per-block tolerance, not per-row: the implementation quantizes in
        // W3_GROUP_SIZE chunks and zero-pads a short trailing chunk, so that
        // chunk's min/max (and thus its error bound) includes the padding
        // zeros, not just the real elements within it.
        for (block_idx, block_src) in row_src.chunks(W3_GROUP_SIZE).enumerate() {
            let mut vals = [0.0f32; W3_GROUP_SIZE];
            vals[..block_src.len()].copy_from_slice(block_src);
            let min_v = vals.iter().copied().fold(f32::INFINITY, f32::min);
            let max_v = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let half_step = (max_v - min_v) / 7.0 / 2.0;
            let start = block_idx * W3_GROUP_SIZE;
            for (i, &orig) in block_src.iter().enumerate() {
                let got = row_out[start + i];
                assert!(
                    (got - orig).abs() <= half_step + 1e-3,
                    "row {r} block {block_idx} elem {i}: error too large ({orig} vs {got}), \
                     row isolation likely broken"
                );
            }
        }
    }
}

#[test]
fn test_group_boundary_exact_multiple_produces_no_padding_block() {
    // Exactly 3 groups' worth of data (96 = 3*32): must produce exactly 3
    // blocks, no extra padded trailing block.
    let src: Vec<f32> = (0..96).map(|i| i as f32).collect();
    let packed = quantize_row_w3(&src).unwrap();
    assert_eq!(
        packed.len(),
        3 * W3_BLOCK_SIZE,
        "exact multiple of group size must not allocate an extra padding block"
    );
}

#[test]
fn test_group_boundary_one_element_past_multiple_adds_one_padded_block() {
    // 97 = 3*32 + 1: must allocate a 4th block (mostly zero-padded).
    let src: Vec<f32> = (0..97).map(|i| i as f32).collect();
    let packed = quantize_row_w3(&src).unwrap();
    assert_eq!(packed.len(), 4 * W3_BLOCK_SIZE);
    let out = dequantize_row_w3(&packed, 97);
    assert_eq!(out.len(), 97);
    // The single real value in the last block must still decode close to its
    // source value despite the other 31 slots in that block being padding
    // zeros (which widen or shift the block's min/max away from the true
    // single-value range).
    let last_val = src[96];
    let decoded_last = out[96];
    assert!(
        (decoded_last - last_val).abs() <= (last_val.abs().max(1.0)),
        "last real element in a padded trailing block decoded too far off: \
         expected ~{last_val}, got {decoded_last}"
    );
}

// ---------------------------------------------------------------------------
// Edge: degenerate / all-same-weight blocks. range == 0 must not divide by
// zero (NaN/inf) and must reconstruct the constant value exactly (within f16
// scale/bias rounding, i.e. tightly).
// ---------------------------------------------------------------------------

#[test]
fn test_degenerate_all_same_nonzero_weights_reconstructs_exactly() {
    let src = vec![3.5f32; 64];
    let tensor = quantize_f32_to_w3(&src, &[64]).unwrap();
    let out = dequantize_w3_to_f32(&tensor);
    assert_eq!(out.len(), 64);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.is_finite(),
            "degenerate constant block must not produce NaN/Inf at {i}"
        );
        assert!(
            (v - 3.5).abs() < 1e-2,
            "constant-value block must reconstruct near-exactly: elem {i} got {v}, want 3.5"
        );
    }
}

#[test]
fn test_degenerate_all_same_negative_weights_reconstructs_exactly() {
    let src = vec![-12.25f32; 32];
    let packed = quantize_row_w3(&src).unwrap();
    let out = dequantize_row_w3(&packed, 32);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "elem {i} must be finite");
        assert!(
            (v - (-12.25)).abs() < 1e-2,
            "elem {i}: got {v}, want -12.25"
        );
    }
}

#[test]
fn test_degenerate_all_zero_weights_reconstructs_zero() {
    let src = vec![0.0f32; 32];
    let packed = quantize_row_w3(&src).unwrap();
    let out = dequantize_row_w3(&packed, 32);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "all-zero block must reconstruct to ~0 at {i}, got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// Edge: zero-length / minimal-length inputs must not panic and must round-trip
// consistently through the tensor-level API too.
// ---------------------------------------------------------------------------

#[test]
fn test_empty_input_produces_empty_output_no_panic() {
    let packed = quantize_row_w3(&[]).unwrap();
    assert!(packed.is_empty(), "empty input must produce zero blocks");
    let out = dequantize_row_w3(&packed, 0);
    assert!(out.is_empty());
}

#[test]
fn test_single_element_input_roundtrips_within_tolerance() {
    let src = vec![42.0f32];
    let packed = quantize_row_w3(&src).unwrap();
    assert_eq!(
        packed.len(),
        W3_BLOCK_SIZE,
        "even 1 element must allocate one full block"
    );
    let out = dequantize_row_w3(&packed, 1);
    assert_eq!(out.len(), 1);
    // Single real value + 31 zero-padding values in the same block: min/max
    // spans [0, 42], so the reconstruction error can be up to one full
    // quantization step (~6), not just half a step.
    assert!(
        (out[0] - 42.0).abs() <= 6.5,
        "single-element block should still land close to 42.0, got {}",
        out[0]
    );
}

// ---------------------------------------------------------------------------
// Full tensor API save/load round trip via a real temp file — exercises the
// on-disk format end to end (not just the in-memory Vec<u8> path already
// covered in the unit tests).
// ---------------------------------------------------------------------------

#[test]
fn test_save_load_file_roundtrip_preserves_values_within_tolerance() {
    let rows = 5usize;
    let cols = 40usize; // non-multiple of 32, exercises padding through file I/O
    let src: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) * 0.1234).cos() * 17.0)
        .collect();
    let tensor = quantize_f32_to_w3(&src, &[rows, cols]).unwrap();

    let mut path = std::env::temp_dir();
    path.push(format!(
        "w3_tester_roundtrip_{}_{}.w3",
        std::process::id(),
        rows * cols
    ));
    save_w3_file(&path, &tensor).unwrap();
    let loaded = load_w3_file(&path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(loaded.shape, vec![rows, cols]);
    assert_eq!(loaded.original_len, rows * cols);
    let out = dequantize_w3_to_f32(&loaded);
    assert_eq!(out.len(), rows * cols);

    // Loose but real bound: values must be within the same order of
    // magnitude as source, not garbage/zeroed/misaligned.
    for (i, (&orig, &got)) in src.iter().zip(&out).enumerate() {
        assert!(
            (got - orig).abs() < 3.0,
            "index {i}: loaded+dequantized value {got} too far from source {orig}"
        );
    }
}

// ---------------------------------------------------------------------------
// Mutation-sensitivity guard: quantized values must actually be quantized
// (limited to <=8 distinct levels per block), not simply pass through the
// f32 input unchanged. This would fail if someone "fixed" a bug by making
// dequantize_row_w3 return the original values directly.
// ---------------------------------------------------------------------------

#[test]
fn test_output_is_actually_quantized_to_at_most_8_levels_per_block() {
    let src: Vec<f32> = (0..32).map(|i| i as f32 * 0.987_654_3).collect();
    let packed = quantize_row_w3(&src).unwrap();
    let out = dequantize_row_w3(&packed, 32);

    let mut distinct: Vec<f32> = Vec::new();
    for &v in &out {
        if !distinct.iter().any(|&d: &f32| (d - v).abs() < 1e-6) {
            distinct.push(v);
        }
    }
    assert!(
        distinct.len() <= 8,
        "a single 32-element W3 block must decode to at most 8 distinct levels, got {}",
        distinct.len()
    );

    // And the output must differ from the raw input (i.e. quantization
    // actually introduced error) for this non-trivial, non-degenerate input.
    let identical = src.iter().zip(&out).all(|(a, b)| (a - b).abs() < 1e-7);
    assert!(
        !identical,
        "quantized output must not be bit-identical to the unquantized source"
    );
}

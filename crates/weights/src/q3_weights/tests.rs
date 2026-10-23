use super::*;

/// The largest per-weight dequant error is one half-step of the block's
/// scale (rounding) plus the f16 scale/bias storage error. For a block whose
/// range is `R` over 7 asymmetric steps, `scale = R/7`, so the round error is
/// bounded by `scale/2 = R/14`. This helper returns that bound with a small
/// f16 slack factor.
fn asym_err_bound(min: f32, max: f32) -> f32 {
    let scale = (max - min) / 7.0;
    scale / 2.0 + scale.abs() * 1e-3 + 1e-6
}

#[test]
fn q3_block_is_sixteen_bytes() {
    assert_eq!(Q3_BLOCK_BYTES, 16);
    assert_eq!(std::mem::size_of::<Q3Block>(), 16);
}

#[test]
fn plane_split_pack_unpack_roundtrips_all_values() {
    // Every one of the 32 slots must round-trip every 3-bit level 0..=7.
    for base in 0u8..8 {
        let mut q = [0u8; 32];
        for (i, slot) in q.iter_mut().enumerate() {
            *slot = ((base as usize + i) % 8) as u8;
        }
        let packed = pack_plane_split(&q);
        for (i, &expected) in q.iter().enumerate() {
            assert_eq!(
                unpack_plane_split(&packed, i),
                expected,
                "value {i} (={expected}) did not survive plane-split pack/unpack"
            );
        }
    }
}

#[test]
fn plane_split_mutation_sensitive_high_bit() {
    // A value >= 4 sets the high-1-bit plane. If the high plane is dropped
    // (a plausible packing bug), the unpacked value loses its +4 and this
    // assertion fails — the test guards the 2+1 split, not just the low plane.
    let mut q = [0u8; 32];
    q[5] = 6; // 0b110: low2 = 2, hi = 1
    q[17] = 7; // 0b111: low2 = 3, hi = 1
    let packed = pack_plane_split(&q);
    assert_eq!(unpack_plane_split(&packed, 5), 6);
    assert_eq!(unpack_plane_split(&packed, 17), 7);
    // The high plane byte for i=5 is packed[8 + 5/8] = packed[8], bit 5 set.
    assert_ne!(packed[8] & (1 << 5), 0, "high bit for value 5 must be set");
    // The high plane byte for i=17 is packed[8 + 17/8] = packed[10], bit 1 set.
    assert_ne!(
        packed[10] & (1 << 1),
        0,
        "high bit for value 17 must be set"
    );
}

#[test]
fn asymmetric_roundtrip_within_scale_half() {
    let src: Vec<f32> = (0..32).map(|i| (i as f32 - 12.0) * 0.37).collect();
    let bytes = quantize_row_q3_0(&src).unwrap();
    assert_eq!(bytes.len(), Q3_BLOCK_BYTES);
    let back = dequantize_row_q3_0(&bytes, src.len());
    let min = src.iter().copied().fold(f32::INFINITY, f32::min);
    let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let bound = asym_err_bound(min, max);
    for (i, (&a, &b)) in src.iter().zip(back.iter()).enumerate() {
        assert!(
            (a - b).abs() <= bound,
            "elem {i}: |{a} - {b}| = {} exceeds bound {bound}",
            (a - b).abs()
        );
    }
}

#[test]
fn symmetric_roundtrip_zero_mean() {
    // Zero-mean, symmetric distribution — the symmetric encode mode's target.
    let src: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.1).collect();
    let t = quantize_f32_to_q3_mode(&src, &[32], true).unwrap();
    let back = dequantize_q3_to_f32(&t);
    let abs_max = src.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    // Symmetric scale = abs_max / 3.5; per-step error bounded by scale/2 plus slack.
    let bound = (abs_max / 3.5) / 2.0 + abs_max * 1e-3 + 1e-6;
    for (i, (&a, &b)) in src.iter().zip(back.iter()).enumerate() {
        assert!(
            (a - b).abs() <= bound,
            "elem {i}: |{a} - {b}| = {} exceeds bound {bound}",
            (a - b).abs()
        );
    }
}

#[test]
fn partial_block_uses_real_tail_min_max() {
    // Only 3 real values; the remaining 29 slots are zero padding that must
    // NOT widen the range used to derive the scale.
    let src = vec![10.0f32, 10.5, 11.0];
    let t = quantize_f32_to_q3(&src, &[3]).unwrap();
    assert_eq!(t.blocks.len(), 1);
    assert_eq!(t.original_len, 3);
    let back = dequantize_q3_to_f32(&t);
    assert_eq!(back.len(), 3);
    let bound = asym_err_bound(10.0, 11.0);
    for (i, (&a, &b)) in src.iter().zip(back.iter()).enumerate() {
        assert!(
            (a - b).abs() <= bound,
            "elem {i}: |{a} - {b}| = {} exceeds bound {bound}",
            (a - b).abs()
        );
    }
}

#[test]
fn constant_block_is_exact() {
    // A zero-range block sets scale = 1.0, bias = min; every weight maps to
    // q = 0 and dequantizes back to exactly `bias`.
    let src = vec![3.5f32; 32];
    let t = quantize_f32_to_q3(&src, &[32]).unwrap();
    let back = dequantize_q3_to_f32(&t);
    for &b in &back {
        assert_eq!(b, 3.5);
    }
}

#[test]
fn rejects_non_finite() {
    let mut src = vec![0.1f32; 32];
    src[7] = f32::NAN;
    assert!(quantize_f32_to_q3(&src, &[32]).is_err());
    src[7] = f32::INFINITY;
    assert!(quantize_f32_to_q3(&src, &[32]).is_err());
}

#[test]
#[should_panic(expected = "must equal data length")]
fn rejects_shape_data_mismatch() {
    let src = vec![0.1f32; 32];
    // shape claims 64 elements but data has 32.
    let _ = quantize_f32_to_q3(&src, &[64]);
}

#[test]
fn bf16_path_matches_f32_path_on_bf16_castable_input() {
    // Values exactly representable in bf16 must quantize identically via
    // both entry points (bf16 decode is lossless for these).
    let bf16_bits: Vec<u16> = (0..64).map(|i| ((i as u16) << 7) | 0x3C00).collect();
    let f32_vals: Vec<f32> = bf16_bits.iter().map(|&b| bf16_to_f32(b)).collect();
    let via_bf16 = quantize_bf16_to_q3(&bf16_bits, &[64]).unwrap();
    let via_f32 = quantize_f32_to_q3(&f32_vals, &[64]).unwrap();
    assert_eq!(via_bf16.blocks, via_f32.blocks);
}

#[test]
fn save_load_roundtrip_preserves_blocks_and_shape() {
    let src: Vec<f32> = (0..96).map(|i| (i as f32 - 40.0) * 0.05).collect();
    let t = quantize_f32_to_q3(&src, &[3, 32]).unwrap();
    let dir = std::env::temp_dir();
    let path = dir.join(format!("lattice_q3_roundtrip_{}.q3", std::process::id()));
    save_q3_file(&path, &t).unwrap();

    // Header parses independently of the block payload.
    let mut file = std::fs::File::open(&path).unwrap();
    let header = read_q3_header(&mut file).unwrap();
    assert_eq!(header.shape, vec![3, 32]);
    assert_eq!(header.original_len, 96);
    assert_eq!(header.payload_offset, 20 + 2 * 8);

    let loaded = load_q3_file(&path).unwrap();
    assert_eq!(loaded.shape, t.shape);
    assert_eq!(loaded.original_len, t.original_len);
    assert_eq!(loaded.blocks, t.blocks);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn read_q3_header_leaves_cursor_at_payload_offset() {
    // Regression for the doc promise on `read_q3_header`: the cursor of
    // the SAME file handle must sit at `payload_offset` on return, not
    // wherever the internal BufReader's last fill left the shared fd.
    // Reading the first block directly off `file` (no re-open, no seek)
    // must reproduce the first block written by `save_q3_file`.
    let src: Vec<f32> = (0..96).map(|i| (i as f32 - 40.0) * 0.05).collect();
    let t = quantize_f32_to_q3(&src, &[3, 32]).unwrap();
    let dir = std::env::temp_dir();
    let path = dir.join(format!("lattice_q3_cursor_{}.q3", std::process::id()));
    save_q3_file(&path, &t).unwrap();

    let mut file = std::fs::File::open(&path).unwrap();
    let header = read_q3_header(&mut file).unwrap();

    assert_eq!(
        std::io::Seek::stream_position(&mut file).unwrap(),
        header.payload_offset,
        "cursor must sit at payload_offset after read_q3_header returns"
    );

    let mut first_block_bytes = [0u8; Q3_BLOCK_BYTES];
    std::io::Read::read_exact(&mut file, &mut first_block_bytes).unwrap();
    let expected: &[u8; Q3_BLOCK_BYTES] = unsafe { &*std::ptr::from_ref(&t.blocks[0]).cast() };
    assert_eq!(
        &first_block_bytes, expected,
        "reading off the same handle post-header must yield the first block, not header tail"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_rejects_bad_magic() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("lattice_q3_badmagic_{}.q3", std::process::id()));
    std::fs::write(&path, b"KHQ4\x01\x00\x00\x00").unwrap();
    assert!(load_q3_file(&path).is_err());
    let _ = std::fs::remove_file(&path);
}

// #1380: `load_q3_file` is the `.q3` sibling of the FIFO regression
// covering `load_q4_file`/`load_f16_tensor_file` -- same raw-open shape
// (a plain `File::open` before any file-type check), same fix. `mkfifo`
// (POSIX, not macOS-only) keeps this test portable to the Linux CI legs
// that run this suite.
#[test]
fn load_q3_file_rejects_a_fifo_without_blocking() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("lattice_q3_fifo_{}.q3", std::process::id()));
    let status = std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .expect("run mkfifo");
    assert!(status.success(), "mkfifo must succeed to set up this test");

    let (tx, rx) = std::sync::mpsc::channel();
    let probe_path = path.clone();
    // A pre-fix blocking `File::open` on this FIFO (no writer ever
    // connects) would hang this thread forever; running the call on a
    // detached thread and racing it against a deadline on
    // `rx.recv_timeout` turns "hangs forever" into an observable test
    // failure instead of an actually-hung test process.
    std::thread::spawn(move || {
        // `Box<dyn Error>` is not `Send` (no `+ Send` bound), so it
        // cannot cross this channel -- map to its `Display` text first.
        let result = load_q3_file(&probe_path).map_err(|e| e.to_string());
        let _ = tx.send(result);
    });
    let result: Result<Q3Tensor, String> =
        rx.recv_timeout(std::time::Duration::from_secs(5)).expect(
            "load_q3_file did not return within 5s -- it blocked on the planted FIFO, \
                 meaning the open-time regular-file guard regressed",
        );
    let err = result.expect_err("a FIFO must be rejected, not accepted, for a .q3 load");
    assert!(
        err.contains("not a regular file"),
        "expected a not-a-regular-file refusal, got: {err}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn four_bpw_storage() {
    // 32 weights in 16 bytes = 4.0 bits per weight (the ADR-072 W3 target).
    let bits_per_weight = (Q3_BLOCK_BYTES * 8) as f32 / 32.0;
    assert_eq!(bits_per_weight, 4.0);
}

// -----------------------------------------------------------------------
// MLP-role fail-closed loader gate
// -----------------------------------------------------------------------

#[test]
fn mlp_role_accepts_gate_up_down_and_fused() {
    assert!(validate_q3_mlp_role("model.layers.3.mlp.gate_proj.weight").is_ok());
    assert!(validate_q3_mlp_role("model.layers.3.mlp.up_proj.weight").is_ok());
    assert!(validate_q3_mlp_role("model.layers.3.mlp.down_proj.weight").is_ok());
    assert!(validate_q3_mlp_role("model.layers.3.mlp.gate_up_proj.weight").is_ok());
}

#[test]
fn mlp_role_rejects_attention_and_gdn_tensors() {
    // Attention and GDN weights must stay Q4/Q8 (ADR-072 P1 scope) — any Q3
    // claim on these tensor names must fail closed, not silently load.
    for name in [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.mlp.experts.gate_up_proj", // MoE routed experts, not the dense MLP path
        "lm_head.weight",
        "model.embed_tokens.weight",
    ] {
        assert!(
            validate_q3_mlp_role(name).is_err(),
            "expected '{name}' to be rejected as outside the Q3 MLP role set"
        );
    }
}

#[test]
fn mlp_role_rejects_near_miss_suffixes() {
    // Exact-suffix contract: `.contains()` on the role substring would
    // wrongly accept these — a LoRA adapter branch, a renamed sibling
    // tensor, and a trailing extra suffix after `.weight`.
    for name in [
        "model.layers.3.mlp.gate_proj.lora_A.weight",
        "model.layers.3.mlp.up_proj_backup.weight",
        "model.layers.3.mlp.up_proj_backup",
        "model.layers.3.mlp.down_proj.weight.extra",
    ] {
        assert!(
            validate_q3_mlp_role(name).is_err(),
            "expected near-miss '{name}' to be rejected by the exact-suffix gate"
        );
    }
}

#[test]
fn mlp_role_requires_terminal_weight_suffix() {
    // Real checkpoints always name dense MLP projections with a terminal
    // `.weight` (see the Q4 loader's `{prefix}.mlp.gate_proj.weight` call
    // sites in metal_qwen35.rs); a role-correct name missing it is not a
    // real tensor name and must fail closed.
    for name in [
        "model.layers.3.mlp.gate_proj",
        "model.layers.3.mlp.up_proj",
        "model.layers.3.mlp.down_proj",
        "model.layers.3.mlp.gate_up_proj",
    ] {
        assert!(
            validate_q3_mlp_role(name).is_err(),
            "expected '{name}' (missing '.weight') to be rejected"
        );
    }
}

// -----------------------------------------------------------------------
// Header payload-bounds fail-closed gate
// -----------------------------------------------------------------------

#[test]
fn header_bounds_rejects_truncated_payload() {
    let header = Q3FileHeader {
        shape: vec![64],
        original_len: 64, // 2 blocks * 16 bytes = 32 bytes required
        payload_offset: 20,
    };
    // File is one byte short of the required 20 + 32 = 52 bytes.
    let err = validate_q3_header_payload_bounds(&header, 51, std::path::Path::new("t.q3"));
    assert!(err.is_err());
}

#[test]
fn header_bounds_accepts_exact_payload() {
    let header = Q3FileHeader {
        shape: vec![64],
        original_len: 64,
        payload_offset: 20,
    };
    assert!(validate_q3_header_payload_bounds(&header, 52, std::path::Path::new("t.q3")).is_ok());
}

#[test]
fn header_bounds_rejects_shape_mismatch() {
    let header = Q3FileHeader {
        shape: vec![63], // disagrees with original_len
        original_len: 64,
        payload_offset: 20,
    };
    assert!(
        validate_q3_header_payload_bounds(&header, 1_000, std::path::Path::new("t.q3")).is_err()
    );
}

// -----------------------------------------------------------------------
// CPU reference GEMV/GEMM — parity oracle sanity + mutation sensitivity
// -----------------------------------------------------------------------

fn synth_q3_weight_matrix(seed: u64, n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut rng = seed;
    let mut next = || -> f32 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 11) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let weights_f32: Vec<f32> = (0..n * k).map(|_| next()).collect();
    let mut packed = Vec::with_capacity(n * (k / 32) * Q3_BLOCK_BYTES);
    let mut deq = Vec::with_capacity(n * k);
    for row in weights_f32.chunks_exact(k) {
        let row_packed = quantize_row_q3_0(row).unwrap();
        deq.extend_from_slice(&dequantize_row_q3_0(&row_packed, k));
        packed.extend_from_slice(&row_packed);
    }
    (packed, deq)
}

#[test]
fn gemv_reference_matches_naive_dequant_dot_product() {
    let (n, k) = (17usize, 64usize);
    let (packed, deq) = synth_q3_weight_matrix(0x1234_5678, n, k);
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 0.32).collect();
    let y = gemv_q3_reference(&x, &packed, n, k);
    for (ni, &yv) in y.iter().enumerate() {
        let expect: f32 = x
            .iter()
            .zip(&deq[ni * k..(ni + 1) * k])
            .map(|(&xv, &wv)| xv * wv)
            .sum();
        assert!((yv - expect).abs() < 1e-3, "row {ni}: {yv} vs {expect}");
    }
}

#[test]
fn gemm_reference_matches_gemv_reference_per_row() {
    // GEMM with M=1 must reduce to GEMV — cross-checks the two oracles
    // against each other in addition to the naive-dot check above.
    let (n, k) = (9usize, 96usize);
    let (packed, _deq) = synth_q3_weight_matrix(0xC0FF_EE11, n, k);
    let x: Vec<f32> = (0..k).map(|i| ((i * 7) % 13) as f32 * 0.05 - 0.3).collect();
    let y_gemv = gemv_q3_reference(&x, &packed, n, k);
    let y_gemm = gemm_q3_reference(&x, &packed, 1, n, k);
    assert_eq!(y_gemv, y_gemm);
}

#[test]
fn gemv_reference_mutation_sensitive_high_plane_bit() {
    // Flip one high-plane bit in the packed weight buffer (the same class
    // of bug the Stage-1 `plane_split_mutation_sensitive_high_bit` test
    // guards at the pack/unpack level) and assert the GEMV parity oracle's
    // output changes — this is the oracle the Metal kernel differential
    // test compares against, so it must itself be sensitive to packing
    // corruption or a corrupted kernel could pass parity vacuously.
    let (n, k) = (4usize, 32usize);
    let (mut packed, _deq) = synth_q3_weight_matrix(0xFEED_FACE, n, k);
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.02 - 0.3).collect();
    let y_before = gemv_q3_reference(&x, &packed, n, k);

    // Row 0's block: bytes [0..16) = [scale(2) bias(2) low2(8) hi(4)].
    // Flip a bit in the high-plane byte at packed offset 4+8+0 = 12.
    packed[12] ^= 0x01;

    let y_after = gemv_q3_reference(&x, &packed, n, k);
    assert_ne!(
        y_before[0], y_after[0],
        "flipping a high-plane bit must change the dequantized GEMV output \
             for the row whose block it belongs to"
    );
}

//! Q3 MLP-only weights: loader and kernel parity.

use super::*;
// -------------------------------------------------------------------
// Q3 MLP-only weight format (ADR-072 P1, #420 Stage 2): loader
// fail-closed gates for `mmap_q3_weight`.
// -------------------------------------------------------------------

#[test]
fn mmap_q3_weight_rejects_non_mlp_tensor_role() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir create");
    let path = tmp.path().join("attn.q3");
    let tensor = crate::weights::q3_weights::Q3Tensor {
        blocks: vec![
            crate::weights::q3_weights::Q3Block {
                scale: 0,
                bias: 0,
                packed: [0u8; 12]
            };
            2
        ],
        shape: vec![64],
        original_len: 64,
    };
    crate::weights::q3_weights::save_q3_file(&path, &tensor).expect("save well-formed q3 file");

    let result = mmap_q3_weight(
        &device,
        &path,
        "model.layers.0.self_attn.q_proj.weight",
        "attn-role-rejected",
    );
    assert!(
        result.is_err(),
        "mmap_q3_weight must reject a non-MLP tensor role even with a well-formed payload"
    );
}

#[test]
fn mmap_q3_weight_rejects_truncated_block_payload_before_dispatch() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir create");
    let path = tmp.path().join("truncated_payload.q3");
    let mut file = std::fs::File::create(&path).expect("create truncated q3");
    use std::io::Write;
    file.write_all(b"KHQ3").expect("write magic");
    file.write_all(&1_u32.to_le_bytes()).expect("write version");
    file.write_all(&1_u32.to_le_bytes()).expect("write ndim");
    file.write_all(&64_u64.to_le_bytes()).expect("write shape");
    file.write_all(&64_u64.to_le_bytes())
        .expect("write original_len");
    file.flush().expect("flush q3 header");
    // original_len=64 -> 2 blocks -> 32 required payload bytes; file
    // ends right at payload_offset, so the payload is entirely missing.

    let result = mmap_q3_weight(
        &device,
        &path,
        "model.layers.0.mlp.gate_proj.weight",
        "truncated-payload-proof",
    );
    assert!(
        result.is_err(),
        "mmap_q3_weight must reject a KHQ3 header whose original_len requires \
                 32 payload bytes but whose file ends at payload_offset"
    );
}

#[test]
fn mmap_q3_weight_accepts_fully_populated_mlp_payload() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir create");
    let path = tmp.path().join("full_payload.q3");
    let tensor = crate::weights::q3_weights::Q3Tensor {
        blocks: vec![
            crate::weights::q3_weights::Q3Block {
                scale: 0,
                bias: 0,
                packed: [0u8; 12]
            };
            2
        ],
        shape: vec![64],
        original_len: 64,
    };
    crate::weights::q3_weights::save_q3_file(&path, &tensor).expect("save well-formed q3 file");

    let result = mmap_q3_weight(
        &device,
        &path,
        "model.layers.0.mlp.down_proj.weight",
        "full-payload-accept",
    );
    assert!(
        result.is_ok(),
        "mmap_q3_weight must accept a well-formed MLP-role payload: {:?}",
        result.err()
    );
}

// #1368: `mmap_q3_weight` must route through the same mmap trust
// boundary `mmap_q4_weight` already does -- proves the wiring, not
// just the underlying `open_trusted_mmap_file` predicate (already
// covered directly in `mmap_trust`'s own tests).
#[cfg(unix)]
#[test]
fn mmap_q3_weight_rejects_a_group_or_other_writable_file() {
    use std::os::unix::fs::PermissionsExt;

    let Some(device) = Device::system_default() else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir create");
    let path = tmp.path().join("writable.q3");
    let tensor = crate::weights::q3_weights::Q3Tensor {
        blocks: vec![
            crate::weights::q3_weights::Q3Block {
                scale: 0,
                bias: 0,
                packed: [0u8; 12]
            };
            2
        ],
        shape: vec![64],
        original_len: 64,
    };
    crate::weights::q3_weights::save_q3_file(&path, &tensor).expect("save well-formed q3 file");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666)).expect("chmod 0o666");

    let result = mmap_q3_weight(
        &device,
        &path,
        "model.layers.0.mlp.down_proj.weight",
        "writable-file-rejected",
    );
    let Err(err) = result else {
        panic!("a group/other-writable Q3 file must be refused")
    };
    assert!(
        err.contains("refusing to load"),
        "expected a trust-boundary refusal, got: {err}"
    );

    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod 0o600");
    let result = mmap_q3_weight(
        &device,
        &path,
        "model.layers.0.mlp.down_proj.weight",
        "owner-only-accepted",
    );
    assert!(
        result.is_ok(),
        "an owner-only Q3 file must still be accepted: {:?}",
        result.err()
    );
}

// -------------------------------------------------------------------
// Q3 Metal kernel parity (ADR-072 P1, #420 Stage 2): `gemv_q3_decode`
// and `gemm_q3_tiled` dequantize plane-split 2+1 Q3 blocks to f16 in
// -kernel and must match the CPU `gemv_q3_reference` / `gemm_q3_reference`
// oracle (`weights::q3_weights`) within an f16-accumulation-driven bound.
// -------------------------------------------------------------------

/// Build a Q3 weight buffer (row-major `[N, K]`, one row of packed
/// blocks per output neuron — matches `gemv_q3_decode`/`gemm_q3_tiled`'s
/// `QW[row * row_bytes + ...]` indexing) and its packed-bytes copy for
/// the CPU reference oracle. Deterministic LCG so runs are reproducible.
fn make_q3_weight_ref(device: &Device, seed: u64, n: usize, k: usize) -> (Buffer, Vec<u8>) {
    assert_eq!(k % 32, 0, "K must be divisible by 32 for Q3 GEMV/GEMM");
    let mut rng = seed;
    let mut next = || -> f32 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 11) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let weights_f32: Vec<f32> = (0..n * k).map(|_| next()).collect();
    use crate::weights::q3_weights::quantize_row_q3_0;
    let mut packed_bytes = Vec::with_capacity(n * (k / 32) * 16);
    for row in weights_f32.chunks_exact(k) {
        packed_bytes.extend_from_slice(&quantize_row_q3_0(row).unwrap());
    }
    let buf = device.new_buffer_with_data(
        packed_bytes.as_ptr() as *const _,
        packed_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    buf.set_label("test_qw_q3");
    (buf, packed_bytes)
}

/// Deterministic activation RNG shared by the Q3 GEMV/GEMM parity tests.
fn q3_test_activation(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = seed;
    let mut next = || -> f32 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 11) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    (0..len).map(|_| next()).collect()
}

#[test]
fn gemv_q3_decode_matches_cpu_reference_across_shapes() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");
    let pipeline = &state.engine.pipelines.gemv_q3;
    let queue = &state.engine.queue;

    use crate::weights::q3_weights::gemv_q3_reference;

    // Shapes exercising tile-adjacent geometry: N=1 (single output row),
    // N=5 (odd, not a multiple of NR=2), N=33 with a 3-block K.
    for &(n, k, seed) in &[
        (1usize, 32usize, 0x1111_u64),
        (5usize, 64usize, 0x2222_u64),
        (33usize, 96usize, 0x3333_u64),
    ] {
        let (qw_buf, packed) = make_q3_weight_ref(&device, seed, n, k);
        let x = q3_test_activation(seed ^ 0xABCD, k);
        let x_buf = device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_buf = device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&qw_buf), 0);
        enc.set_buffer(2, Some(&y_buf), 0);
        enc.set_bytes(3, 4, &(n as u32) as *const u32 as *const _);
        enc.set_bytes(4, 4, &(k as u32) as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(2), 1, 1),
            MTLSize::new(32, 4, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // SAFETY: `cmd.wait_until_completed()` above blocks until the
        // GPU has finished writing `y_buf`, and `y_buf` is a
        // `StorageModeShared` buffer sized `n * 4` bytes — the CPU
        // read of `n` f32s below is in-bounds and happens-after the
        // GPU write.
        let y_gpu: &[f32] =
            unsafe { std::slice::from_raw_parts(y_buf.contents().cast::<f32>(), n) };
        let y_ref = gemv_q3_reference(&x, &packed, n, k);

        for i in 0..n {
            let diff = (y_gpu[i] - y_ref[i]).abs();
            assert!(
                diff < 5e-3,
                "shape N={n} K={k}: row {i}: gpu={} ref={} diff={diff}",
                y_gpu[i],
                y_ref[i]
            );
        }
    }
}

#[test]
fn gemm_q3_tiled_matches_cpu_reference_at_tile_boundaries() {
    let Some(device) = Device::system_default() else {
        return;
    };
    if !device.supports_family(MTLGPUFamily::Apple7) {
        return;
    }
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");
    // On Apple7+ the tiled pipeline MUST exist — a missing pipeline here
    // is the exact regression this gate guards against, so fail loudly
    // rather than `return` (mirrors `gemm_q4_tiled_vs_naive_numeric_differential`).
    let tiled_pipeline = state
        .engine
        .pipelines
        .gemm_q3_tiled
        .as_ref()
        .expect("gemm_q3_tiled must be present on Apple7+ (else this gate passes vacuously)");
    let queue = &state.engine.queue;

    use crate::weights::q3_weights::gemm_q3_reference;

    // Shape A: M=64,N=64,K=64 exact tile multiples (BM=64, BN=32, BK=32).
    // Shape B: M=72,N=48,K=64 non-multiple of BM/BN — exercises the
    //   `gm<M`/`gn<N` boundary zero-pad guards in the tiled kernel.
    // Shape C: M=5,N=9,K=96 small odd M/N with a 3-block K depth.
    // Shape D: M=16,N=32,K=1024 — Qwen3.5-0.8B's dense gate/up
    //   projection input depth (`hidden_size=1024`, dispatch uses
    //   `K=hidden` — metal_qwen35.rs:5086), a production K-depth but
    //   not the deepest Q3-eligible one.
    // Shape E: M=16,N=1024,K=3584 — Qwen3.5-0.8B's down-projection
    //   depth (`intermediate_size=3584`, dispatch uses
    //   `K=intermediate` — metal_qwen35.rs:5112), 112 `BK=32`
    //   iterations — the deepest Q3-eligible production MLP shape.
    //
    // `max_abs_diff` is the same NaN-honest helper used by the Q4 twin
    // below (`gemm_q4_tiled_vs_naive_numeric_differential`) — a
    // `.fold(0.0, f32::max)`-style comparison silently drops a NaN/Inf
    // operand (IEEE maxNum semantics keep the non-NaN side), so a
    // corrupted GPU tile that produces NaN would read as a perfect
    // match. Both operands here read the *same* already-quantized Q3
    // bytes, so source quantization error cancels between GPU and CPU
    // reference; the residual is f16 X/W staging plus f32 reduction
    // order, not quant step size — see the measured envelope below,
    // which replaces the prior hand-waved 0.05 bound.
    let mut envelope: Vec<(usize, usize, usize, f32)> = Vec::new();
    for &(m, n, k, seed) in &[
        (64usize, 64usize, 64usize, 0xAAAA_u64),
        (72usize, 48usize, 64usize, 0xBBBB_u64),
        (5usize, 9usize, 96usize, 0xCCCC_u64),
        (16usize, 32usize, 1024usize, 0xEEEE_u64),
        (16usize, 1024usize, 3584usize, 0xFEED_u64),
    ] {
        let (qw_buf, packed) = make_q3_weight_ref(&device, seed, n, k);
        let x = q3_test_activation(seed ^ 0xDEAD, m * k);
        let x_buf = device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_buf = device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(tiled_pipeline);
        enc.set_buffer(0, Some(&qw_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&y_buf), 0);
        enc.set_bytes(3, 4, &(m as u32) as *const u32 as *const _);
        enc.set_bytes(4, 4, &(n as u32) as *const u32 as *const _);
        enc.set_bytes(5, 4, &(k as u32) as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(32), (m as u64).div_ceil(64), 1),
            MTLSize::new(32, 4, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // SAFETY: completed GPU work (`wait_until_completed` above)
        // makes this `StorageModeShared` buffer's `m * n` f32s visible
        // to the CPU; the slice length matches the buffer's
        // allocated size (`m * n * 4` bytes) from its construction
        // above.
        let y_gpu: &[f32] =
            unsafe { std::slice::from_raw_parts(y_buf.contents().cast::<f32>(), m * n) };
        let y_ref = gemm_q3_reference(&x, &packed, m, n, k);

        let max_diff = max_abs_diff(y_gpu, &y_ref);
        assert!(
            max_diff.is_finite(),
            "shape M={m} N={n} K={k}: max|gpu-ref| is not finite (max_diff={max_diff}) \
                     — the GPU tiled kernel produced a NaN/Inf output that a \
                     `.fold(0.0, f32::max)`-style comparison would have silently \
                     dropped as a false match"
        );
        println!("gemm_q3_tiled envelope: M={m} N={n} K={k} max|gpu-ref|={max_diff:.3e}");
        envelope.push((m, n, k, max_diff));
    }

    // Bound derived from the measured envelope above, not a hand-waved
    // constant: the largest observed shape (M=16 N=1024 K=3584, the
    // Qwen3.5-0.8B down-projection depth — 112 `BK=32` iterations)
    // measured max|gpu-ref| ~= 1.946e-2;
    // 0.037 is ~1.9x that. The headroom covers cross-device variation
    // (different Apple Silicon generations accumulate the f16
    // X/W-staging + f32-reduction residual slightly differently) and
    // future kernel-scheduling changes (e.g. a different tile
    // traversal order), not run-to-run jitter — the seeds and
    // dispatch are deterministic, so a rerun on the same host
    // reproduces the same value, and this bound is not measuring
    // that. It must still catch a real regression (e.g. the NaN and
    // packed-high-plane mutation tests below, which corrupt the
    // GPU-visible bytes and must fail this same bound).
    const MEASURED_ENVELOPE_TOL: f32 = 0.037;
    for &(m, n, k, d) in &envelope {
        assert!(
            d < MEASURED_ENVELOPE_TOL,
            "shape M={m} N={n} K={k}: max|gpu-ref| = {d} exceeds the \
                     measured-envelope bound {MEASURED_ENVELOPE_TOL}"
        );
    }
}

/// Mutation-sensitivity test for the differential comparison itself:
/// corrupt a packed block's `scale` field
/// to an IEEE-754 f16 NaN bit pattern so the GPU kernel's dequantized
/// weight — and therefore its GEMM output — is NaN, then prove the
/// NaN-honest comparison fails loudly instead of silently reading as
/// a perfect match. This is deliberately the inverse of the test
/// above: it proves the comparison mechanism itself is
/// mutation-sensitive, the same way the pack/unpack mutation tests
/// prove the kernel is.
///
/// `#[should_panic(expected = "is not
/// finite")]` could not distinguish "the Apple7 dispatch ran and the
/// NaN-honest comparison caught it" from "no Apple7 device, so this
/// branch panicked with an unrelated message that happened to also
/// contain the substring" — both read as green under
/// `should_panic(expected = ...)`. This test instead dispatches
/// unconditionally past the device gate, then wraps only the
/// comparison in `catch_unwind` and asserts on the caught panic's
/// message, so a panic from the wrong place (or no panic at all)
/// fails the test instead of silently passing it.
///
/// Consequence: on a host with a Metal device but no Apple7 GPU
/// (e.g. GitHub's `macos-latest` paravirtual runner), this test
/// still returns early rather than failing — mirroring
/// `gemm_q3_tiled_matches_cpu_reference_at_tile_boundaries` and
/// `gemm_q4_tiled_enabled_on_apple7_plus` above, and the reasoning
/// `e2e-parity.yml`'s MP2 injection gate documents: Apple7's
/// `simdgroup_matrix` requirement is real hardware, not something
/// `LATTICE_METAL_TEST_ENFORCE` can force on a paravirtual GPU. No CI
/// workflow currently selects this test by name (its module path
/// does not match any existing `--lib` filter), so
/// `LATTICE_METAL_TEST_ENFORCE` below only covers the device-absent
/// branch, consistent with every other Apple7-gated test in this file.
#[test]
fn gemm_q3_tiled_differential_fails_closed_on_nan_scale() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(device) = Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    if !device.supports_family(MTLGPUFamily::Apple7) {
        return;
    }
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");
    let tiled_pipeline = state
        .engine
        .pipelines
        .gemm_q3_tiled
        .as_ref()
        .expect("gemm_q3_tiled must be present on Apple7+");
    let queue = &state.engine.queue;

    use crate::weights::q3_weights::gemm_q3_reference;

    let (m, n, k) = (5usize, 9usize, 96usize);
    let (_qw_buf, packed) = make_q3_weight_ref(&device, 0xF00D_u64, n, k);

    // Corrupt row 0's first block: `scale` occupies packed[0..2] as an
    // IEEE-754 f16 bit pattern (little-endian). 0x7C01 is a quiet-ish
    // NaN encoding (exponent all-ones, nonzero mantissa) — every
    // weight in that block dequantizes to NaN, and the GEMM row that
    // reads it becomes NaN in the GPU output.
    let mut corrupted = packed.clone();
    corrupted[0] = 0x01;
    corrupted[1] = 0x7C;
    let qw_buf = device.new_buffer_with_data(
        corrupted.as_ptr() as *const _,
        corrupted.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let x = q3_test_activation(0xF00D_u64 ^ 0xDEAD, m * k);
    let x_buf = device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_buf = device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(tiled_pipeline);
    enc.set_buffer(0, Some(&qw_buf), 0);
    enc.set_buffer(1, Some(&x_buf), 0);
    enc.set_buffer(2, Some(&y_buf), 0);
    enc.set_bytes(3, 4, &(m as u32) as *const u32 as *const _);
    enc.set_bytes(4, 4, &(n as u32) as *const u32 as *const _);
    enc.set_bytes(5, 4, &(k as u32) as *const u32 as *const _);
    enc.dispatch_thread_groups(
        MTLSize::new((n as u64).div_ceil(32), (m as u64).div_ceil(64), 1),
        MTLSize::new(32, 4, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // SAFETY: completed GPU work (`wait_until_completed` above) makes
    // this `StorageModeShared` buffer's `m * n` f32s visible to the
    // CPU; the slice length matches the buffer's allocated size
    // (`m * n * 4` bytes) from its construction above.
    let y_gpu: &[f32] =
        unsafe { std::slice::from_raw_parts(y_buf.contents().cast::<f32>(), m * n) };
    // Reference computed from the ORIGINAL, uncorrupted bytes: the
    // GPU side alone is corrupted, so any NaN in `y_gpu` is a genuine
    // divergence the comparison must surface, not an artifact of a
    // NaN also present on the reference side.
    let y_ref = gemm_q3_reference(&x, &packed, m, n, k);

    // Wrapped in `catch_unwind` (round-2 fix) rather than
    // `#[should_panic]` on the whole test: everything above this
    // point (the device/Apple7 gate, the real dispatch) has already
    // run unconditionally, so a caught "is not finite" panic here
    // proves the dispatch executed and the comparison caught the
    // corruption — not that the test skipped before doing anything.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let max_diff = max_abs_diff(y_gpu, &y_ref);
        assert!(
            max_diff.is_finite(),
            "max|gpu-ref| is not finite (max_diff={max_diff}) — NaN-scale mutation \
                     correctly detected by the comparison"
        );
    }));
    let payload = result
        .expect_err("expected the NaN-scale mutation to make the differential comparison panic");
    let msg = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("<non-string panic payload>");
    assert!(
        msg.contains("is not finite"),
        "expected the caught panic message to contain \"is not finite\", got: {msg}"
    );
}

#[test]
fn gemv_q3_decode_mutation_sensitive_high_plane_bit() {
    // Guards the Metal kernel's high-plane read: if `gemv_q3_decode`
    // had a bug that dropped the high-1-bit plane (silently truncating
    // every dequant to the 0..=3 range), flipping a high-plane bit in
    // the packed weight buffer would NOT change the kernel's output.
    // This proves the kernel actually consumes `packed[8..12]`, not
    // just `packed[0..8]` — the same bug class the Stage-1
    // `plane_split_mutation_sensitive_high_bit` test guards at the
    // pack/unpack level (weights/q3_weights.rs), now proven through the
    // real GPU dispatch path this Stage adds.
    let Some(device) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");
    let pipeline = &state.engine.pipelines.gemv_q3;
    let queue = &state.engine.queue;

    let (n, k) = (2usize, 32usize);
    let (_qw_buf, packed) = make_q3_weight_ref(&device, 0x9999_u64, n, k);
    let mut corrupted = packed.clone();
    // Row 0's block occupies packed[0..16); its high-plane byte 0
    // (values 0..8) is packed[12] (offset 4 header + 8 low-plane bytes).
    corrupted[12] ^= 0x01;

    let x = q3_test_activation(0x9999_u64 ^ 0x5A5A, k);
    let x_buf = device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let run = |packed_bytes: &[u8]| -> Vec<f32> {
        let qw_buf = device.new_buffer_with_data(
            packed_bytes.as_ptr() as *const _,
            packed_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_buf = device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&qw_buf), 0);
        enc.set_buffer(2, Some(&y_buf), 0);
        enc.set_bytes(3, 4, &(n as u32) as *const u32 as *const _);
        enc.set_bytes(4, 4, &(k as u32) as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(2), 1, 1),
            MTLSize::new(32, 4, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        // SAFETY: completed GPU work (`wait_until_completed` above)
        // makes this `StorageModeShared` buffer's `n` f32s visible to
        // the CPU; the slice length matches the buffer's allocated
        // size (`n * 4` bytes) from its construction above.
        unsafe { std::slice::from_raw_parts(y_buf.contents().cast::<f32>(), n) }.to_vec()
    };

    let y_correct = run(&packed);
    let y_corrupted = run(&corrupted);

    assert_ne!(
        y_correct[0], y_corrupted[0],
        "flipping a high-plane bit in row 0's block must change gemv_q3_decode's \
                 output for row 0 — if it doesn't, the kernel is not reading the \
                 high-1-bit plane (packed[8..12])"
    );
    assert_eq!(
        y_correct[1], y_corrupted[1],
        "corrupting row 0's block must not affect row 1's output"
    );
}

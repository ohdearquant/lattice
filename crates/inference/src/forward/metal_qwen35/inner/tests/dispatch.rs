use super::*;

/// Issue #252 (#238 follow-up): runtime capability probe for the f16
/// KV-cache Metal path. GitHub's hosted `macos-latest` runner exposes a
/// PARAVIRTUAL Metal GPU (not a real Apple7 device), so this test must
/// NOT gate on `supports_family(Apple7)` — it gates on whether the f16
/// KV kernels actually construct and execute, and fails loudly
/// (LATTICE_METAL_TEST_ENFORCE=1) instead of skip-passing when a Metal
/// device is absent.
#[test]
fn f16_kv_metal_path_executes_and_reports_capability() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = Device::system_default() else {
        eprintln!("[METAL_F16_KV_CAPABILITY] supported=false reason=no_metal_device");
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };

    assert!(
        matches!(
            std::env::var("LATTICE_KV_F16").as_deref(),
            Ok("1") | Ok("true")
        ),
        "f16 KV Metal probe must run with LATTICE_KV_F16=1"
    );
    assert!(
        matches!(
            std::env::var("LATTICE_METAL_PATH_PROOF").as_deref(),
            Ok("1") | Ok("true")
        ),
        "f16 KV Metal probe must run with LATTICE_METAL_PATH_PROOF=1"
    );

    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = match MetalQwen35State::new(&weights, &cfg, 16) {
        Ok(state) => state,
        Err(err) => {
            eprintln!(
                "[METAL_F16_KV_CAPABILITY] supported=false reason=construct_failed error={err}"
            );
            panic!("f16 KV Metal capability probe failed during state construction: {err}");
        }
    };

    assert!(state.use_kv_f16, "state did not enable f16 KV cache");
    assert!(state.use_kv_f16, "KV cache layout is not f16");
    assert!(state.path_proof_enabled, "path-proof counters are disabled");

    let expected_kv_bytes = 16 * cfg.full_kv_dim() * 2;
    assert_eq!(
        state.session.kv_cache.k_bufs[0].length() as usize,
        expected_kv_bytes,
        "K cache buffer must use f16 element size"
    );

    state.reset_path_proof_counters();
    let _ = state.forward_prefill(&[1, 2, 3]);
    let decode_pos = state.session.kv_cache.seq_len;
    let _ = state.forward_step(4, decode_pos);

    let proof = state.path_proof_snapshot();
    eprintln!(
        "[METAL_F16_KV_CAPABILITY] supported=true kv_f16={} prefill_kv_batch={} prefill_attn_batched={} decode_kv_copy={} decode_attn_direct={} decode_attn_split_partial={} decode_attn_split_reduce={}",
        proof.kv_f16,
        proof.prefill_kv_batch,
        proof.prefill_attn_batched,
        proof.decode_kv_copy,
        proof.decode_attn_direct,
        proof.decode_attn_split_partial,
        proof.decode_attn_split_reduce,
    );

    assert!(proof.kv_f16, "path proof did not report kv_f16=true");
    assert!(
        proof.prefill_kv_batch > 0,
        "f16 batch KV write did not execute"
    );
    assert!(
        proof.prefill_attn_batched > 0,
        "f16 batched prefill attention did not execute"
    );
    assert!(
        proof.decode_kv_copy > 0,
        "f16 decode KV copy did not execute"
    );
    assert!(
        proof.decode_attn_direct > 0
            || (proof.decode_attn_split_partial > 0 && proof.decode_attn_split_reduce > 0),
        "f16 decode attention did not execute"
    );
}

/// Regression for the Q4 decode dispatch-geometry bug (2026-06-26).
///
/// `gemv_q4_decode` writes NR=2 output rows per threadgroup, so
/// `dispatch_matmul_q4` MUST launch `ceil(N/2)` groups. The prior `ceil(N/4)`
/// left the upper ~half of the rows unwritten — on the Q4 decode/logits path
/// (`final_logits` → `dispatch_matmul` for `QuantFormat::Q4_0`, N = vocab_size)
/// this silently corrupted every token after the first prefill token, because
/// the upper half of the vocabulary logits were never produced.
///
/// N=6 is the minimal shape that exposes it: `ceil(6/4)=2` groups write only
/// rows 0..4 (rows 4,5 dropped); `ceil(6/2)=3` groups write all 6. Pre-fill Y
/// with a sentinel and assert every row is overwritten and matches the CPU
/// dequant reference. Reverting the fix to `div_ceil(4)` fails this test.
#[test]
fn dispatch_matmul_q4_writes_all_rows() {
    // Fail closed under enforce: a CI runner that provisions a Metal GPU but
    // silently skips here would make this regression gate verify nothing —
    // the same silent-skip class as the embed parity gate (#383). The
    // dedicated macOS test step sets LATTICE_METAL_TEST_ENFORCE=1.
    //
    // Note: NO Apple7 family gate. `gemv_q4_decode` uses only `simd_sum` +
    // threadgroup memory, not the Apple7-gated `simdgroup_matrix` MMA path,
    // so it runs on GitHub's paravirtual macOS GPU (which reports a Metal
    // device but NOT Apple7). The tiled-GEMM tests below DO need Apple7 and
    // skip on CI; this decode-geometry test genuinely runs there.
    let enforce = std::env::var("LATTICE_METAL_TEST_ENFORCE").is_ok();
    let Some(device) = Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");

    let (n, k) = (6usize, 64usize);
    let (qw_raw, w_deq) = make_q4_weight_ref(&device, 0x0FF0_1234_u64, n, k);
    let qw = Q4WeightBuf::from_buffer(qw_raw);

    let mut xrng = 0x1234_5678_u64;
    let x: Vec<f32> = (0..k)
        .map(|_| {
            xrng = xrng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((xrng >> 11) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    let x_buf = device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Pre-fill the output with a sentinel far outside the achievable result
    // range (|y| < K = 64 here); any row left untouched keeps it.
    const SENTINEL: f32 = -123_456.0;
    let y_init = vec![SENTINEL; n];
    let y_buf = device.new_buffer_with_data(
        y_init.as_ptr() as *const _,
        (n * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = state.engine.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    state.dispatch_matmul_q4(enc, &x_buf, &qw, &y_buf, 1, n as u32, k as u32);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // SAFETY: StorageModeShared, GPU work completed, size matches allocation.
    let y: &[f32] = unsafe { std::slice::from_raw_parts(y_buf.contents() as *const f32, n) };
    let y_ref = cpu_matmul_ref(&x, &w_deq, 1, n, k);

    for (row, &val) in y.iter().enumerate() {
        assert!(
            val.to_bits() != SENTINEL.to_bits(),
            "row {row} of {n} never written — dispatch grid under-covers \
             gemv_q4_decode (NR=2 rows/group). This is the P0 decode-geometry bug."
        );
    }
    let diff = max_abs_diff(y, &y_ref);
    assert!(
        diff < 1e-3,
        "dispatch_matmul_q4 result diverges from CPU dequant ref: max_abs_diff={diff:.4e}"
    );
}

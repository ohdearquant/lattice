//! `from_q4_dir` refusals on misconfigured checkpoints.

use super::*;
// -------------------------------------------------------------------
// from_q4_dir refuse-on-misconfig (PR #32)
// -------------------------------------------------------------------
//
// `read_quarot_seed_from_index` itself (the `quantize_index.json`
// reader) moved to `crate::quant::quarot::convert` (#504 remaining
// slice 2) — it is pure file-I/O with no Metal dependency, so its
// tests live there where they run under default (CPU-only) features
// instead of requiring `metal-gpu`.

#[test]
fn from_q4_dir_rejects_max_cache_len_above_max_position_embeddings() {
    let _gpu_guard = gpu_test_lock();
    // `from_q4_dir` was missing the
    // `max_cache_len <= cfg.max_position_embeddings` guard that
    // `new_session` enforces. Without it, `--max-cache-len 999999
    // --window 999999` would slip past the strided-PPL aggregator
    // (which only knows the KV cap) and panic / read out-of-bounds
    // GPU memory in `partial_rope_interleaved`. The loader now
    // matches `new_session` and refuses up-front.
    let Some(_) = Device::system_default() else {
        return;
    };
    let (mut cfg, _weights) = tiny_metal_qwen35_fixture();
    cfg.max_position_embeddings = 64; // small RoPE table
    let tmp = std::env::temp_dir().join(format!("lattice-q4-test-mpe-{}", std::process::id()));
    // The validator runs before any file I/O so an empty / non-
    // existent dir is fine for this regression.
    let Err(err) =
        MetalQwen35State::from_q4_dir(&tmp, std::path::Path::new("/dev/null"), &cfg, 1024)
    else {
        panic!("max_cache_len > max_position_embeddings must error")
    };
    assert!(
        err.contains("max_cache_len") && err.contains("max_position_embeddings"),
        "error must name both bounds; got: {err}"
    );
}

#[test]
fn from_q4_dir_rejects_zero_max_cache_len() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    let (cfg, _weights) = tiny_metal_qwen35_fixture();
    let tmp = std::env::temp_dir().join(format!("lattice-q4-test-zero-{}", std::process::id()));
    let Err(err) = MetalQwen35State::from_q4_dir(&tmp, std::path::Path::new("/dev/null"), &cfg, 0)
    else {
        panic!("max_cache_len = 0 must error")
    };
    assert!(
        err.contains("max_cache_len"),
        "error must name max_cache_len; got: {err}"
    );
}

#[test]
fn from_q4_dir_rejects_tied_config_with_lm_head_artifact() {
    let _gpu_guard = gpu_test_lock();
    // The prior fix only caught the
    // `tie_word_embeddings=false` + missing `lm_head.q4` half of
    // the artifact contract. The opposite mismatch —
    // `tie_word_embeddings=true` but a `lm_head_weight.q4`
    // present on disk — was still accepted, with the loader
    // silently ignoring the materialized head (the converter's
    // config-flip was likely skipped because HF Qwen3.5 stores
    // the flag in both nested `text_config` and the top level).
    // The loader now refuses both directions before any tensor
    // I/O.
    let Some(_) = Device::system_default() else {
        return;
    };
    let (cfg, _weights) = tiny_metal_qwen35_fixture();
    assert!(
        cfg.tie_word_embeddings,
        "tiny fixture must be tied for this test"
    );
    let tmp = std::env::temp_dir().join(format!("lattice-q4-test-tied-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("tempdir create");
    // Drop a placeholder lm_head_weight.q4 in the dir; the file
    // contents don't matter — the preflight only checks
    // existence, before any loader touches the bytes.
    let lm_head_file = tmp.join("lm_head_weight.q4");
    std::fs::write(&lm_head_file, b"placeholder").expect("write lm_head placeholder");

    let Err(err) = MetalQwen35State::from_q4_dir(&tmp, std::path::Path::new("/dev/null"), &cfg, 16)
    else {
        panic!("tied config with stray lm_head.q4 must error")
    };
    assert!(
        err.contains("lm_head") && err.contains("tie_word_embeddings"),
        "error must name both lm_head and tie_word_embeddings; got: {err}"
    );
    // Cleanup; ignore errors.
    let _ = std::fs::remove_file(&lm_head_file);
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn from_q4_dir_rejects_untied_config_without_lm_head_artifact() {
    let _gpu_guard = gpu_test_lock();
    // The loader silently fell back to
    // `embed_tokens` when `tie_word_embeddings=false` and
    // `lm_head.q4` was missing. That is exactly the contamination
    // ADR-044 step 3c forbids — a QuaRot artifact missing its
    // materialized fused-and-rotated head would still produce a
    // PPL and a PASS verdict using the wrong logits matrix. The
    // loader now refuses before any tensor I/O.
    let Some(_) = Device::system_default() else {
        return;
    };
    let (mut cfg, _weights) = tiny_metal_qwen35_fixture();
    cfg.tie_word_embeddings = false; // make this an untied config
    let tmp = std::env::temp_dir().join(format!("lattice-q4-test-untied-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("tempdir create");
    // Empty dir → lm_head.weight.q4 does not exist.
    let Err(err) = MetalQwen35State::from_q4_dir(&tmp, std::path::Path::new("/dev/null"), &cfg, 16)
    else {
        panic!("untied config without lm_head artifact must error")
    };
    assert!(
        err.contains("lm_head") && err.contains("tie_word_embeddings"),
        "error must name both lm_head and tie_word_embeddings; got: {err}"
    );
    // Cleanup; ignore errors (the dir may have been removed by other tests).
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn lora_gemv_kernel_matches_cpu_reference() {
    let _gpu_guard = gpu_test_lock();
    let Some(device) = Device::system_default() else {
        return;
    };
    let rank: u32 = 16;
    let k: u32 = 1024;
    let n: u32 = 512;
    let scale: f32 = 0.5;

    // Synthetic data
    let mut rng_state = 0xCAFE_BABE_u64;
    let mut rand_f32 = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 11) as u32 as f32 / u32::MAX as f32) - 0.5
    };
    let x: Vec<f32> = (0..k).map(|_| rand_f32()).collect();
    let a: Vec<f32> = (0..(rank * k)).map(|_| rand_f32()).collect();
    let b: Vec<f32> = (0..(n * rank)).map(|_| rand_f32()).collect();
    let y_init: Vec<f32> = (0..n).map(|_| rand_f32()).collect();

    // CPU reference: y += scale * B @ (A @ x)
    let mut intermediate_cpu = vec![0.0f32; rank as usize];
    for r in 0..rank as usize {
        let row = &a[r * k as usize..(r + 1) * k as usize];
        intermediate_cpu[r] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }
    let mut y_expected = y_init.clone();
    for i in 0..n as usize {
        let row = &b[i * rank as usize..(i + 1) * rank as usize];
        let dot: f32 = row
            .iter()
            .zip(intermediate_cpu.iter())
            .map(|(a, b)| a * b)
            .sum();
        y_expected[i] += scale * dot;
    }

    // GPU execution
    let opts = CompileOptions::new();
    let library = device
        .new_library_with_source(MSL_SOURCE, &opts)
        .expect("MSL compile");
    let pipeline_a = {
        let f = library
            .get_function("lora_gemv_a", None)
            .expect("lora_gemv_a");
        device
            .new_compute_pipeline_state_with_function(&f)
            .expect("pipeline lora_gemv_a")
    };
    let pipeline_b = {
        let f = library
            .get_function("lora_gemv_b_accum", None)
            .expect("lora_gemv_b_accum");
        device
            .new_compute_pipeline_state_with_function(&f)
            .expect("pipeline lora_gemv_b_accum")
    };

    let make_buf = |data: &[f32]| -> Buffer {
        device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let x_buf = make_buf(&x);
    let a_buf = make_buf(&a);
    let b_buf = make_buf(&b);
    let y_buf = make_buf(&y_init);
    let inter_buf = device.new_buffer((rank as u64) * 4, MTLResourceOptions::StorageModeShared);

    let queue = device.new_command_queue();
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();

    // Phase 1
    enc.set_compute_pipeline_state(&pipeline_a);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&a_buf), 0);
    enc.set_buffer(2, Some(&inter_buf), 0);
    enc.set_bytes(3, 4, &rank as *const u32 as *const _);
    enc.set_bytes(4, 4, &k as *const u32 as *const _);
    enc.dispatch_thread_groups(MTLSize::new(rank as u64, 1, 1), MTLSize::new(32, 4, 1));

    // Phase 2
    enc.set_compute_pipeline_state(&pipeline_b);
    enc.set_buffer(0, Some(&inter_buf), 0);
    enc.set_buffer(1, Some(&b_buf), 0);
    enc.set_buffer(2, Some(&y_buf), 0);
    enc.set_bytes(3, 4, &n as *const u32 as *const _);
    enc.set_bytes(4, 4, &rank as *const u32 as *const _);
    enc.set_bytes(5, 4, &scale as *const f32 as *const _);
    enc.dispatch_thread_groups(
        MTLSize::new(n.div_ceil(256) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );

    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // SAFETY: y_buf is StorageModeShared, GPU work completed, length matches allocation.
    let y_gpu: &[f32] =
        unsafe { std::slice::from_raw_parts(y_buf.contents() as *const f32, n as usize) };

    let max_diff = y_expected
        .iter()
        .zip(y_gpu.iter())
        .map(|(e, g)| (e - g).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "LoRA GEMV GPU vs CPU diverged: max_abs_diff={max_diff}"
    );
}

#[test]
fn load_adapter_and_dispatch_lora_if_active() {
    let _gpu_guard = gpu_test_lock();
    let Some(device) = Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state =
        MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");

    let hidden = cfg.hidden_size;
    let rank = 4usize;
    let scale = 2.0f32;

    let mut rng = 0xABCD_u64;
    let mut rand_f32 = || -> f32 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 11) as u32 as f32 / u32::MAX as f32) - 0.5
    };

    let a: Vec<f32> = (0..rank * hidden).map(|_| rand_f32()).collect();
    let b: Vec<f32> = (0..hidden * rank).map(|_| rand_f32()).collect();

    // o_proj: d_in = full_q_dim() = hidden, d_out = hidden (both 512 in tiny fixture)
    let layers = vec![LoraLayerData {
        layer_idx: 0,
        module: "o_proj".into(),
        a: a.clone(),
        b: b.clone(),
        rank,
        d_in: hidden,
        d_out: hidden,
    }];

    // Load without QuaRot rotation (unrotated base)
    state.load_lora_adapter(layers, scale, None).unwrap();
    assert!(state.has_lora_adapter());

    // Dispatch through the production wrapper
    let x: Vec<f32> = (0..hidden).map(|_| rand_f32()).collect();
    let y_init: Vec<f32> = (0..hidden).map(|_| rand_f32()).collect();

    let x_buf = device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_buf = device.new_buffer_with_data(
        y_init.as_ptr() as *const _,
        (y_init.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = state.engine.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    state.dispatch_lora_if_active(enc, &x_buf, 0, &y_buf, 0, 0, "o_proj");
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // CPU reference
    let mut intermediate = vec![0.0f32; rank];
    for r in 0..rank {
        intermediate[r] = a[r * hidden..(r + 1) * hidden]
            .iter()
            .zip(x.iter())
            .map(|(ai, xi)| ai * xi)
            .sum();
    }
    let mut y_expected = y_init.clone();
    for i in 0..hidden {
        let dot: f32 = b[i * rank..(i + 1) * rank]
            .iter()
            .zip(intermediate.iter())
            .map(|(bi, ii)| bi * ii)
            .sum();
        y_expected[i] += scale * dot;
    }

    // SAFETY: y_buf is StorageModeShared, GPU work completed, length matches allocation.
    let y_gpu: &[f32] =
        unsafe { std::slice::from_raw_parts(y_buf.contents() as *const f32, hidden) };
    let max_diff = y_expected
        .iter()
        .zip(y_gpu.iter())
        .map(|(e, g)| (e - g).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "dispatch_lora_if_active GPU vs CPU: max_abs_diff={max_diff}"
    );

    // Verify no-op for wrong layer/module
    let y_buf2 = device.new_buffer_with_data(
        y_init.as_ptr() as *const _,
        (y_init.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cmd2 = state.engine.queue.new_command_buffer();
    let enc2 = cmd2.new_compute_command_encoder();
    state.dispatch_lora_if_active(enc2, &x_buf, 0, &y_buf2, 0, 0, "v_proj");
    enc2.end_encoding();
    cmd2.commit();
    cmd2.wait_until_completed();

    // SAFETY: y_buf2 is StorageModeShared, GPU work completed, length matches allocation.
    let y_noop: &[f32] =
        unsafe { std::slice::from_raw_parts(y_buf2.contents() as *const f32, hidden) };
    assert_eq!(
        y_noop,
        &y_init[..],
        "dispatch for non-loaded module should be no-op"
    );

    // Unload
    state.unload_lora_adapter();
    assert!(!state.has_lora_adapter());
}

#[test]
fn gemm_q4_tiled_enabled_on_apple7_plus() {
    let _gpu_guard = gpu_test_lock();
    // Regression guard for the M>1 prefill GEMM. The simdgroup-matrix
    // tiled Q4 kernel (`gemm_q4_tiled`) must be enabled on every Apple7+
    // GPU (M1 and up). It was previously gated behind Apple9, silently
    // disabling it on M1/M2 and falling back to the ~1.6x-slower naive
    // `gemm_q4`. A silent MSL compile failure would re-trigger the same
    // fallback. Both regress prefill throughput with no visible error,
    // so assert the real gate path keeps the tiled pipeline live.
    let Some(device) = Device::system_default() else {
        return;
    };
    if !device.supports_family(MTLGPUFamily::Apple7) {
        return;
    }
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");
    assert!(
        state.engine.pipelines.gemm_q4_tiled.is_some(),
        "gemm_q4_tiled must be enabled on Apple7+ (got None — gate regressed \
                 or MSL_Q4_TILED_SOURCE failed to compile)"
    );
}

// ── Q4 GEMM numeric differential gate (#271) ─────────────────────────
// The tiled kernel (`gemm_q4_tiled`, PR #270) stores X and W in half-precision
// staging tiles (`simdgroup_half8x8`) while keeping accumulation in f32
// (`simdgroup_float8x8`).  This test catches half-tile precision regressions
// that the greedy-argmax e2e gate can miss when the numeric error is too small
// to flip a token but large enough to corrupt softmax-sensitive downstream ops.
//
// Strategy: build a synthetic Q4 weight buffer in-test (no model file), run both
// the naive f32 kernel (`gemm_q4`) and the tiled half-staging kernel
// (`gemm_q4_tiled`) on identical inputs, and assert both stay within documented
// error envelopes vs an in-Rust f32 reference (dequantize → f32 matmul).
//
// The naive kernel is f32 throughout, so its bound vs reference is dominated by
// Q4 quantization error (~2 ULP at half scale).  The tiled kernel adds half
// multiplicand rounding on top, widening the bound — but accumulators stay f32,
// so the envelope is tighter than a pure-f16 path would be.
//
// Buffer layout (MUST match both kernels): QW is indexed as
//   `QW[col * row_bytes + ib * 20 + ...]` where col ∈ 0..N (output neuron),
//   ib ∈ 0..(K/32) (K-block), and row_bytes = (K/32) * 20.
// Calling `quantize_row_q4_0` row-by-row on a [N, K] matrix produces this layout.

/// Hermetic numeric differential: naive vs tiled vs CPU reference, two shapes.
///
/// Shape A (M=64, N=64, K=64) — exact tile multiple, exercises the main tiled path.
/// Shape B (M=72, N=48, K=64) — non-multiple of BM/BN, exercises the boundary
/// guards (`gm<M` / `gn<N` zero-pad paths) in the tiled kernel.
///
/// Measured on M1 (2026-06-24) with half-tile staging (PR #270):
///   naive_vs_ref  A: 3.3e-6  B: 4.3e-6
///   tiled_vs_ref  A: 2.9e-3  B: 2.7e-3
///   tiled_vs_naive A: 2.9e-3  B: 2.7e-3
///
/// Asserted bounds (headroom over measured, per row):
///   naive_vs_ref  < 1e-3   — naive and the CPU reference both consume the same
///                            already-quantized Q4 bytes, so source quantization
///                            error cancels; the residual is GPU-vs-CPU
///                            implementation and reduction-order difference, not
///                            quant step size (~300× headroom)
///   tiled_vs_ref  < 0.015  — half Xtg staging adds ~3e-3 on top of that residual;
///                            accumulators stay f32 so this is well under a
///                            pure-f16 bound (~5× headroom)
///   tiled_vs_naive < 0.012 — direct bound between the two GPU kernels (~4× headroom)
///
/// If future changes restore f32 staging tiles, all three bounds still pass.
/// If half precision is extended to accumulators, tiled_vs_ref would likely exceed 0.015.
#[test]
fn gemm_q4_tiled_vs_naive_numeric_differential() {
    let _gpu_guard = gpu_test_lock();
    let Some(device) = Device::system_default() else {
        return;
    };
    if !device.supports_family(MTLGPUFamily::Apple7) {
        return;
    }
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");
    // On Apple7+ the tiled pipeline MUST exist (see gemm_q4_tiled_enabled_on_apple7_plus).
    // A missing pipeline here is the exact regression this gate guards against, so fail
    // loudly rather than `return` — a skipped numeric test passes vacuously.
    let tiled_pipeline = state
        .engine
        .pipelines
        .gemm_q4_tiled
        .as_ref()
        .expect("gemm_q4_tiled must be present on Apple7+ (else this gate passes vacuously)");
    let naive_pipeline = &state.engine.pipelines.gemm_q4;
    let queue = &state.engine.queue;

    // Activation RNG (independent seed from weight RNG).
    let mut xrng = 0xDEAD_BEEF_u64;
    let mut next_x = || -> f32 {
        xrng = xrng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((xrng >> 11) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    // ── Shape A: M=64, N=64, K=64 — exact tile multiples ────────────
    {
        let (m, n, k) = (64usize, 64usize, 64usize);
        let (qw_buf, w_deq) = make_q4_weight_ref(&device, 0xABCD_1234_u64, n, k);
        let x: Vec<f32> = (0..m * k).map(|_| next_x()).collect();
        let x_buf = device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Naive dispatch: tg = (ceil(N/2), ceil(M/4), 1)
        let y_naive_buf =
            device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
        run_gemm_q4_kernel(
            queue,
            naive_pipeline,
            &qw_buf,
            &x_buf,
            &y_naive_buf,
            m as u32,
            n as u32,
            k as u32,
            n.div_ceil(2) as u64,
            m.div_ceil(4) as u64,
        );

        // Tiled dispatch: tg = (ceil(N/32), ceil(M/64), 1)
        let y_tiled_buf =
            device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
        run_gemm_q4_kernel(
            queue,
            tiled_pipeline,
            &qw_buf,
            &x_buf,
            &y_tiled_buf,
            m as u32,
            n as u32,
            k as u32,
            n.div_ceil(32) as u64,
            m.div_ceil(64) as u64,
        );

        let y_ref = cpu_matmul_ref(&x, &w_deq, m, n, k);
        // SAFETY: buffers are StorageModeShared, GPU work completed, sizes match allocation.
        let y_naive: &[f32] =
            unsafe { std::slice::from_raw_parts(y_naive_buf.contents() as *const f32, m * n) };
        let y_tiled: &[f32] =
            unsafe { std::slice::from_raw_parts(y_tiled_buf.contents() as *const f32, m * n) };

        let naive_vs_ref = max_abs_diff(y_naive, &y_ref);
        let tiled_vs_ref = max_abs_diff(y_tiled, &y_ref);
        let tiled_vs_naive = max_abs_diff(y_tiled, y_naive);

        eprintln!(
            "[shape A 64×64×64] naive_vs_ref={naive_vs_ref:.4e}  \
                     tiled_vs_ref={tiled_vs_ref:.4e}  tiled_vs_naive={tiled_vs_naive:.4e}"
        );
        // Naive and the CPU reference both consume the same already-quantized Q4
        // bytes (see make_q4_weight_ref), so source quantization error cancels here;
        // the residual is GPU-vs-CPU implementation and reduction-order difference.
        // Measured 3.3e-6 on M1 (2026-06-24). Bound 1e-3 gives ~300× headroom.
        // If this fails, the buffer layout or dequant convention is wrong.
        assert!(
            naive_vs_ref < 1e-3,
            "[shape A] naive_vs_ref={naive_vs_ref:.4e} exceeds 1e-3 — \
                     buffer layout or dequant mismatch (check per-row quantize convention)"
        );
        // Tiled adds half Xtg staging; measured 2.9e-3 on M1. Bound 0.015 gives 5× headroom.
        // If this exceeds 0.015, half precision was extended beyond staging tiles.
        assert!(
            tiled_vs_ref < 0.015,
            "[shape A] tiled_vs_ref={tiled_vs_ref:.4e} exceeds 0.015 — \
                     half-tile staging introduced unexpected precision regression"
        );
        // Direct kernel-vs-kernel bound; measured 2.9e-3 on M1. Bound 0.012 gives 4× headroom.
        assert!(
            tiled_vs_naive < 0.012,
            "[shape A] tiled_vs_naive={tiled_vs_naive:.4e} exceeds 0.012 — \
                     tiled and naive kernels are computing different values"
        );
    }

    // ── Shape B: M=72, N=48, K=64 — non-multiple of BM/BN, tests boundary guards ─
    {
        let (m, n, k) = (72usize, 48usize, 64usize);
        let (qw_buf, w_deq) = make_q4_weight_ref(&device, 0x5EED_CAFE_u64, n, k);
        let x: Vec<f32> = (0..m * k).map(|_| next_x()).collect();
        let x_buf = device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let y_naive_buf =
            device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
        run_gemm_q4_kernel(
            queue,
            naive_pipeline,
            &qw_buf,
            &x_buf,
            &y_naive_buf,
            m as u32,
            n as u32,
            k as u32,
            n.div_ceil(2) as u64,
            m.div_ceil(4) as u64,
        );

        let y_tiled_buf =
            device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
        run_gemm_q4_kernel(
            queue,
            tiled_pipeline,
            &qw_buf,
            &x_buf,
            &y_tiled_buf,
            m as u32,
            n as u32,
            k as u32,
            n.div_ceil(32) as u64,
            m.div_ceil(64) as u64,
        );

        let y_ref = cpu_matmul_ref(&x, &w_deq, m, n, k);
        // SAFETY: same as shape A — StorageModeShared, GPU done, sizes match.
        let y_naive: &[f32] =
            unsafe { std::slice::from_raw_parts(y_naive_buf.contents() as *const f32, m * n) };
        let y_tiled: &[f32] =
            unsafe { std::slice::from_raw_parts(y_tiled_buf.contents() as *const f32, m * n) };

        let naive_vs_ref = max_abs_diff(y_naive, &y_ref);
        let tiled_vs_ref = max_abs_diff(y_tiled, &y_ref);
        let tiled_vs_naive = max_abs_diff(y_tiled, y_naive);

        eprintln!(
            "[shape B 72×48×64] naive_vs_ref={naive_vs_ref:.4e}  \
                     tiled_vs_ref={tiled_vs_ref:.4e}  tiled_vs_naive={tiled_vs_naive:.4e}"
        );
        // Measured 4.3e-6, 2.7e-3, 2.7e-3 on M1 (2026-06-24). Same bounds as shape A.
        assert!(
            naive_vs_ref < 1e-3,
            "[shape B] naive_vs_ref={naive_vs_ref:.4e} exceeds 1e-3"
        );
        assert!(
            tiled_vs_ref < 0.015,
            "[shape B] tiled_vs_ref={tiled_vs_ref:.4e} exceeds 0.015"
        );
        assert!(
            tiled_vs_naive < 0.012,
            "[shape B] tiled_vs_naive={tiled_vs_naive:.4e} exceeds 0.012"
        );
    }
}

#[test]
fn forced_non_apple7_q4_gemm_fallback_dispatches_and_matches_reference() {
    let Some(device) = Device::system_default() else {
        eprintln!("[q4-fallback-proof] SKIPPED: no Metal device present on this machine");
        return;
    };
    if !device.supports_family(MTLGPUFamily::Apple7) {
        eprintln!(
            "[q4-fallback-proof] SKIPPED: device does not support Apple7 (tiled Q4 \
                     pipeline never built, so the fallback selection cannot be discriminated)"
        );
        return;
    }
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state =
        MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");
    assert!(
        state.force_q4_gemm_fallback_for_test(),
        "Apple7+ fixture must start with gemm_q4_tiled so the test-only override \
                 discriminates the non-Apple7 selection"
    );
    MetalQwen35State::reset_q4_gemm_fallback_dispatches_for_test();

    let (m, n, k) = (8usize, 8usize, 64usize);
    let (qw_buf, w_deq) = make_q4_weight_ref(&device, 0x899_u64, n, k);
    let qw = Q4WeightBuf::from_buffer(qw_buf);
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i * 37 % 251) as f32 - 125.0) / 127.0)
        .collect();
    let x_buf = device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_buf = device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);

    let cmd = state.engine.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    state.dispatch_gemm_q4(enc, &x_buf, 0, &qw, &y_buf, 0, m as u32, n as u32, k as u32);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let fallback_dispatches = MetalQwen35State::q4_gemm_fallback_dispatches_for_test();
    assert_eq!(
        fallback_dispatches, 1,
        "forced non-Apple7 selection must execute exactly one naive Q4 GEMM fallback"
    );
    let y_ref = cpu_matmul_ref(&x, &w_deq, m, n, k);
    // SAFETY: the command buffer completed and y_buf contains m*n f32 values.
    let y: &[f32] = unsafe { std::slice::from_raw_parts(y_buf.contents() as *const f32, m * n) };
    let diff = max_abs_diff(y, &y_ref);
    eprintln!(
        "[q4-fallback-proof] forced=true fallback_dispatches={fallback_dispatches} \
                 max_abs_diff={diff:.4e}"
    );
    assert!(
        diff < 1e-3,
        "forced non-Apple7 Q4 GEMM fallback diverged from the f32 dequant reference: \
                 max_abs_diff={diff:.4e}"
    );
}

// ── Q8 GEMM numeric differential gate ────────────────────────────────
// Mirrors the Q4 differential test above, but for the Q8_0 tiled kernel
// (`gemm_q8_tiled`, PR #271 follow-up).  Q8_0 blocks are 34 bytes:
// 2-byte f16 scale (LE) + 32 int8 values.  The tiled kernel stores
// activations and dequantized weights as half in threadgroup memory while
// keeping accumulators in f32.  Q8 quantization error is lower than Q4,
// so all bounds are the same (a subtest that Q8 ≤ Q4 on precision).

/// Helper: decode a Q8_0 scale word (little-endian f16) to f32.
fn f16_bits_to_f32(bits: u16) -> f32 {
    // IEEE 754-2008 half to single conversion.
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as u32;
    if exp == 31 {
        // Inf or NaN
        return f32::from_bits(sign | 0x7f80_0000 | (frac << 13));
    }
    if exp == 0 {
        // Zero or denormalized half
        if frac == 0 {
            return f32::from_bits(sign);
        }
        // Denorm half: value = (-1)^sign * 2^(-14) * (frac / 1024)
        // Find the leading 1 bit of frac (10-bit mantissa stored in low bits of frac).
        // frac is nonzero, in range 1..1023 stored in a u32.
        // leading_zeros counts from bit 31 down; the highest bit of frac is at most bit 9.
        let lz = frac.leading_zeros(); // ≥ 22 since frac < 1024
        let shift = lz - 22; // how many extra shifts to normalize (0..9)
        // After left-shift by (shift+1) the implicit 1 is gone; the remaining bits form
        // the f32 mantissa (22 bits from 23..1).
        let frac32 = (frac << (shift + 1)) & 0x007f_ffff;
        // Unbiased exp for denorm half is -14; further adjusted by normalization shift.
        // f32 exponent = -14 - shift + 127 (biased).
        let exp32 = (127i32 - 14 - shift as i32) as u32;
        return f32::from_bits(sign | (exp32 << 23) | frac32);
    }
    // Normal half: exp32 = exp16 - 15 + 127
    let exp32 = (exp - 15 + 127) as u32;
    f32::from_bits(sign | (exp32 << 23) | (frac << 13))
}

/// Build a Q8_0 weight buffer and f32 reference for shape [N, K].
/// Uses a deterministic LCG; calls `quantize_row_q8_0` row-by-row —
/// matching the kernel index formula `QW[col * (K/32 * 34) + ib * 34]`.
fn make_q8_weight_ref(
    device: &Device,
    seed: u64,
    n: usize, // output dim
    k: usize, // input dim — must be divisible by 32
) -> (Buffer, Vec<f32>) {
    assert_eq!(k % 32, 0, "K must be divisible by 32 for Q8_0 GEMM");
    let mut rng = seed;
    let mut next = || -> f32 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 11) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let weights_f32: Vec<f32> = (0..n * k).map(|_| next()).collect();
    let mut packed_bytes = Vec::with_capacity(n * (k / 32) * 34);
    let mut deq_weights = Vec::with_capacity(n * k);
    for row in weights_f32.chunks_exact(k) {
        let row_packed = quantize_row_q8_0(row).unwrap();
        // Dequantize for CPU reference: each 34-byte block has f16 scale + 32 i8 values.
        for block in row_packed.chunks_exact(34) {
            let scale_bits = (block[0] as u16) | ((block[1] as u16) << 8);
            let scale = f16_bits_to_f32(scale_bits);
            for &byte in &block[2..34] {
                deq_weights.push((byte as i8) as f32 * scale);
            }
        }
        packed_bytes.extend_from_slice(&row_packed);
    }
    let buf = device.new_buffer_with_data(
        packed_bytes.as_ptr() as *const _,
        packed_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    buf.set_label("test_qw_q8");
    (buf, deq_weights)
}

/// Helper used by Q8 differential tests: dispatch a single GEMM kernel.
/// `is_tiled` selects buffer binding order:
///   tiled: buf(0)=QW, buf(1)=X, buf(2)=Y
///   naive: buf(0)=X,  buf(1)=QW, buf(2)=Y
fn run_gemm_q8_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    qw_buf: &Buffer,
    x_buf: &Buffer,
    y_buf: &Buffer,
    m: u32,
    n: u32,
    k: u32,
    tg_x: u64,
    tg_y: u64,
    is_tiled: bool,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    if is_tiled {
        enc.set_buffer(0, Some(qw_buf), 0);
        enc.set_buffer(1, Some(x_buf), 0);
    } else {
        enc.set_buffer(0, Some(x_buf), 0);
        enc.set_buffer(1, Some(qw_buf), 0);
    }
    enc.set_buffer(2, Some(y_buf), 0);
    enc.set_bytes(3, 4, &m as *const u32 as *const _);
    enc.set_bytes(4, 4, &n as *const u32 as *const _);
    enc.set_bytes(5, 4, &k as *const u32 as *const _);
    enc.dispatch_thread_groups(MTLSize::new(tg_x, tg_y, 1), MTLSize::new(32, 4, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

#[test]
fn gemm_q8_tiled_enabled_on_apple7_plus() {
    let _gpu_guard = gpu_test_lock();
    let Some(device) = Device::system_default() else {
        return;
    };
    if !device.supports_family(MTLGPUFamily::Apple7) {
        return;
    }
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");
    assert!(
        state.engine.pipelines.gemm_q8_tiled.is_some(),
        "gemm_q8_tiled must be enabled on Apple7+ (got None — gate regressed \
                 or MSL_Q8_TILED_SOURCE failed to compile)"
    );
}

#[test]
fn gemm_q8_tiled_vs_naive_numeric_differential() {
    let _gpu_guard = gpu_test_lock();
    let Some(device) = Device::system_default() else {
        return;
    };
    if !device.supports_family(MTLGPUFamily::Apple7) {
        return;
    }
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");
    let tiled_pipeline = state
        .engine
        .pipelines
        .gemm_q8_tiled
        .as_ref()
        .expect("gemm_q8_tiled must be present on Apple7+ (else this gate passes vacuously)");
    let naive_pipeline = &state.engine.pipelines.gemm_q8;
    let queue = &state.engine.queue;

    let mut xrng = 0xBEEF_DEAD_u64;
    let mut next_x = || -> f32 {
        xrng = xrng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((xrng >> 11) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    // ── Shape A: M=64, N=64, K=64 — exact tile multiples ────────────
    {
        let (m, n, k) = (64usize, 64usize, 64usize);
        let (qw_buf, w_deq) = make_q8_weight_ref(&device, 0xABCD_5678_u64, n, k);
        let x: Vec<f32> = (0..m * k).map(|_| next_x()).collect();
        let x_buf = device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Naive dispatch: buf(0)=X, buf(1)=QW; tg = (ceil(N/2), ceil(M/4), 1)
        let y_naive_buf =
            device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
        run_gemm_q8_kernel(
            queue,
            naive_pipeline,
            &qw_buf,
            &x_buf,
            &y_naive_buf,
            m as u32,
            n as u32,
            k as u32,
            n.div_ceil(2) as u64,
            m.div_ceil(4) as u64,
            false,
        );

        // Tiled dispatch: buf(0)=QW, buf(1)=X; tg = (ceil(N/32), ceil(M/64), 1)
        let y_tiled_buf =
            device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
        run_gemm_q8_kernel(
            queue,
            tiled_pipeline,
            &qw_buf,
            &x_buf,
            &y_tiled_buf,
            m as u32,
            n as u32,
            k as u32,
            n.div_ceil(32) as u64,
            m.div_ceil(64) as u64,
            true,
        );

        let y_ref = cpu_matmul_ref(&x, &w_deq, m, n, k);
        // SAFETY: StorageModeShared, GPU work completed, sizes match allocation.
        let y_naive: &[f32] =
            unsafe { std::slice::from_raw_parts(y_naive_buf.contents() as *const f32, m * n) };
        let y_tiled: &[f32] =
            unsafe { std::slice::from_raw_parts(y_tiled_buf.contents() as *const f32, m * n) };

        let naive_vs_ref = max_abs_diff(y_naive, &y_ref);
        let tiled_vs_ref = max_abs_diff(y_tiled, &y_ref);
        let tiled_vs_naive = max_abs_diff(y_tiled, y_naive);

        eprintln!(
            "[Q8 shape A 64×64×64] naive_vs_ref={naive_vs_ref:.4e}  \
                     tiled_vs_ref={tiled_vs_ref:.4e}  tiled_vs_naive={tiled_vs_naive:.4e}"
        );
        assert!(
            naive_vs_ref < 1e-3,
            "[Q8 shape A] naive_vs_ref={naive_vs_ref:.4e} exceeds 1e-3 — \
                     buffer layout or dequant mismatch"
        );
        assert!(
            tiled_vs_ref < 0.015,
            "[Q8 shape A] tiled_vs_ref={tiled_vs_ref:.4e} exceeds 0.015 — \
                     half-tile staging introduced unexpected precision regression"
        );
        assert!(
            tiled_vs_naive < 0.012,
            "[Q8 shape A] tiled_vs_naive={tiled_vs_naive:.4e} exceeds 0.012 — \
                     tiled and naive kernels are computing different values"
        );
    }

    // ── Shape B: M=72, N=48, K=64 — non-multiple of BM/BN ───────────
    {
        let (m, n, k) = (72usize, 48usize, 64usize);
        let (qw_buf, w_deq) = make_q8_weight_ref(&device, 0x5EED_1234_u64, n, k);
        let x: Vec<f32> = (0..m * k).map(|_| next_x()).collect();
        let x_buf = device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let y_naive_buf =
            device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
        run_gemm_q8_kernel(
            queue,
            naive_pipeline,
            &qw_buf,
            &x_buf,
            &y_naive_buf,
            m as u32,
            n as u32,
            k as u32,
            n.div_ceil(2) as u64,
            m.div_ceil(4) as u64,
            false,
        );

        let y_tiled_buf =
            device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
        run_gemm_q8_kernel(
            queue,
            tiled_pipeline,
            &qw_buf,
            &x_buf,
            &y_tiled_buf,
            m as u32,
            n as u32,
            k as u32,
            n.div_ceil(32) as u64,
            m.div_ceil(64) as u64,
            true,
        );

        let y_ref = cpu_matmul_ref(&x, &w_deq, m, n, k);
        // SAFETY: same as shape A.
        let y_naive: &[f32] =
            unsafe { std::slice::from_raw_parts(y_naive_buf.contents() as *const f32, m * n) };
        let y_tiled: &[f32] =
            unsafe { std::slice::from_raw_parts(y_tiled_buf.contents() as *const f32, m * n) };

        let naive_vs_ref = max_abs_diff(y_naive, &y_ref);
        let tiled_vs_ref = max_abs_diff(y_tiled, &y_ref);
        let tiled_vs_naive = max_abs_diff(y_tiled, y_naive);

        eprintln!(
            "[Q8 shape B 72×48×64] naive_vs_ref={naive_vs_ref:.4e}  \
                     tiled_vs_ref={tiled_vs_ref:.4e}  tiled_vs_naive={tiled_vs_naive:.4e}"
        );
        assert!(
            naive_vs_ref < 1e-3,
            "[Q8 shape B] naive_vs_ref={naive_vs_ref:.4e} exceeds 1e-3"
        );
        assert!(
            tiled_vs_ref < 0.015,
            "[Q8 shape B] tiled_vs_ref={tiled_vs_ref:.4e} exceeds 0.015"
        );
        assert!(
            tiled_vs_naive < 0.012,
            "[Q8 shape B] tiled_vs_naive={tiled_vs_naive:.4e} exceeds 0.012"
        );
    }
}

/// Edge-scale coverage for the half-tile Q4 GEMM (#271).
///
/// The tiled kernel stores activations and dequantized weights as `half` in
/// threadgroup memory before multiplying.  Large activations (100s–1000s) combined
/// with small weight scales can stress f16 representability: if the staged half
/// value saturates to ±65504 the result diverges from the f32 reference.
///
/// Two properties are checked:
///   1. Large activations (±500) with near-unit weights — relative error must stay
///      well under 2 %.  This is the f32-accumulation guard: if the accumulators were
///      silently downcast to f16, the half mantissa alone would push relative error
///      toward ~1 % and shrink the margin.  (Half *staging* of a ±500 input loses
///      ≤0.25 abs/elem, which is why the absolute diff is ~1 at output magnitude
///      ~4e3 — a 0.02 % relative error, not an accumulation error.)
///   2. A near-zero weight block in isolation (second sub-test below) — proves the
///      small-scale dequant path neither amplifies (|y| stays on the order of tens,
///      not thousands) nor diverges from the f32 reference.  The mixed-block global
///      relative error in (1) cannot isolate this, so (2) zeroes the near-unit block.
///
/// Measured on M1 (2026-06-24): mixed abs=9.6e-1, ref_max=4.21e3, rel=2.3e-4 (bound 2 %).
#[test]
fn gemm_q4_tiled_edge_scale_activation_coverage() {
    let _gpu_guard = gpu_test_lock();
    let Some(device) = Device::system_default() else {
        return;
    };
    if !device.supports_family(MTLGPUFamily::Apple7) {
        return;
    }
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");
    // Fail loudly (not skip) if the pipeline is missing on Apple7+ — see the numeric
    // differential test above for why a `return` here would pass vacuously.
    let tiled_pipeline = state
        .engine
        .pipelines
        .gemm_q4_tiled
        .as_ref()
        .expect("gemm_q4_tiled must be present on Apple7+ (else this gate passes vacuously)");
    let queue = &state.engine.queue;

    let (m, n, k) = (64usize, 32usize, 64usize);
    // Weights designed to produce blocks with two distinct scales: near-unit (~1.0)
    // and near-zero (~0.002).  Block 0 (first 32 elements of each row) is near-unit;
    // block 1 (next 32) is near-zero.  K=64 so exactly 2 blocks per output row.
    let mut w_f32 = vec![0.0f32; n * k];
    for ni in 0..n {
        // First block: values in [-1, 1] with typical scale
        for ki in 0..32 {
            let t = (ni * 32 + ki) as f32 / (n * 32) as f32 * 2.0 - 1.0;
            w_f32[ni * k + ki] = t;
        }
        // Second block: near-zero values — tests that the bias term stays small
        for ki in 32..64 {
            w_f32[ni * k + ki] = ((ni * ki) as f32 / (n * k) as f32 - 0.5) * 0.004;
        }
    }

    use crate::weights::q4_weights::dequantize_row_q4_0;
    let mut packed_bytes = Vec::with_capacity(n * (k / 32) * 20);
    let mut w_deq = Vec::with_capacity(n * k);
    for row in w_f32.chunks_exact(k) {
        let row_packed = quantize_row_q4_0(row).unwrap();
        w_deq.extend_from_slice(&dequantize_row_q4_0(&row_packed, k));
        packed_bytes.extend_from_slice(&row_packed);
    }
    let qw_buf = device.new_buffer_with_data(
        packed_bytes.as_ptr() as *const _,
        packed_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Large activations: scale ~500, mixing positive and negative values.
    let mut arng = 0x1234_5678_u64;
    let x: Vec<f32> = (0..m * k)
        .map(|_| {
            arng = arng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((arng >> 11) as u32 as f32 / u32::MAX as f32) * 1000.0 - 500.0
        })
        .collect();

    let x_buf = device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_tiled_buf = device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
    run_gemm_q4_kernel(
        queue,
        tiled_pipeline,
        &qw_buf,
        &x_buf,
        &y_tiled_buf,
        m as u32,
        n as u32,
        k as u32,
        n.div_ceil(32) as u64,
        m.div_ceil(64) as u64,
    );

    let y_ref = cpu_matmul_ref(&x, &w_deq, m, n, k);
    // SAFETY: StorageModeShared, GPU done, size matches allocation.
    let y_tiled: &[f32] =
        unsafe { std::slice::from_raw_parts(y_tiled_buf.contents() as *const f32, m * n) };

    let abs_diff = max_abs_diff(y_tiled, &y_ref);
    let ref_max = y_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    // Relative error = abs_diff / max(|y_ref|), expected < 0.5 % (half-staging rounding
    // on top of Q4 quantization; accumulators stay f32 so error is bounded by Q4 step).
    let rel_err = if ref_max > 1.0 {
        abs_diff / ref_max
    } else {
        abs_diff
    };
    eprintln!(
        "[edge-scale] tiled_vs_ref abs={abs_diff:.4e}  ref_max={ref_max:.2e}  \
                 rel={rel_err:.4e}"
    );
    assert!(
        rel_err < 0.02,
        "[edge-scale] relative error {rel_err:.4e} exceeds 2 % — \
                 half-tile staging may be saturating or accumulating in f16"
    );

    // ── Isolated near-zero block (#272) ────────────────
    // The mixed-block assertion above is dominated by the near-unit block, so it
    // cannot isolate the near-zero claim. Zero the near-unit block and keep only the
    // near-zero one, so the output magnitude IS the near-zero block's contribution.
    // Proves the small-scale dequant path does not amplify and tracks the f32 ref.
    let mut w_nz = w_f32.clone();
    for ni in 0..n {
        for ki in 0..32 {
            w_nz[ni * k + ki] = 0.0;
        }
    }
    let mut nz_packed = Vec::with_capacity(n * (k / 32) * 20);
    let mut nz_deq = Vec::with_capacity(n * k);
    for row in w_nz.chunks_exact(k) {
        let row_packed = quantize_row_q4_0(row).unwrap();
        nz_deq.extend_from_slice(&dequantize_row_q4_0(&row_packed, k));
        nz_packed.extend_from_slice(&row_packed);
    }
    let nz_qw_buf = device.new_buffer_with_data(
        nz_packed.as_ptr() as *const _,
        nz_packed.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_nz_buf = device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
    run_gemm_q4_kernel(
        queue,
        tiled_pipeline,
        &nz_qw_buf,
        &x_buf,
        &y_nz_buf,
        m as u32,
        n as u32,
        k as u32,
        n.div_ceil(32) as u64,
        m.div_ceil(64) as u64,
    );
    let y_nz_ref = cpu_matmul_ref(&x, &nz_deq, m, n, k);
    // SAFETY: StorageModeShared, GPU done, size matches allocation.
    let y_nz: &[f32] =
        unsafe { std::slice::from_raw_parts(y_nz_buf.contents() as *const f32, m * n) };
    let nz_abs = max_abs_diff(y_nz, &y_nz_ref);
    let nz_ref_max = y_nz_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    eprintln!("[edge-scale near-zero] tiled_vs_ref abs={nz_abs:.4e}  ref_max={nz_ref_max:.2e}");
    // |w|≲0.002 × |x|≲500 × 32 terms ⇒ |y| on the order of tens, never thousands.
    assert!(
        nz_ref_max < 200.0,
        "[edge-scale near-zero] near-zero weights produced |y_ref|={nz_ref_max:.2e} \
                 (expected ≲ tens) — a near-zero Q4 block is being amplified"
    );
    // Tiled must track the f32 reference: half staging of ±500 × |w|≲0.002 ⇒ ~1e-3 abs.
    assert!(
        nz_abs < 1.0,
        "[edge-scale near-zero] tiled_vs_ref abs={nz_abs:.4e} exceeds 1.0 — \
                 near-zero block path diverges from the f32 reference"
    );
}

// ── malformed-load validation tests ──────────────────────────────────
// These tests require a Metal device — they return early if none is available.
// Each case must return Err, never panic.

#[test]
fn lora_all_position_prefill_and_nll_return_typed_errors() {
    let _gpu = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture constructs");
    state
        .load_lora_adapter(vec![make_valid_layer(cfg.hidden_size, 1)], 1.0, None)
        .expect("valid LoRA adapter loads");

    let direct_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.forward_prefill_all_logits(&[1, 2])
    }));
    let direct_result =
        direct_outcome.expect("LoRA all-position prefill must return Err, not panic");
    let direct_error = direct_result.expect_err("active LoRA must reject all-position");
    assert!(
        matches!(
            &direct_error,
            crate::error::InferenceError::Inference(message)
                if message.contains("all-position prefill")
                    && message.contains("LoRA adapters")
        ),
        "unexpected all-position error: {direct_error}"
    );

    let nll_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.compute_token_nlls(&[1, 2])
    }));
    let nll_result = nll_outcome.expect("LoRA compute_token_nlls must return Err, not panic");
    let nll_error = nll_result.expect_err("active LoRA must reject per-token NLLs");
    assert!(
        matches!(
            &nll_error,
            crate::error::InferenceError::Inference(message)
                if message.contains("all-position prefill")
                    && message.contains("LoRA adapters")
        ),
        "unexpected NLL error: {nll_error}"
    );
}

#[test]
fn load_lora_adapter_rejects_short_a() {
    let _gpu_guard = gpu_test_lock();
    let Some(_dev) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    let mut layer = make_valid_layer(hidden, rank);
    layer.a.pop(); // one element short of rank * d_in
    let result = state.load_lora_adapter(vec![layer], 1.0, None);
    assert!(result.is_err(), "short A must return Err");
}

#[test]
fn load_lora_adapter_rejects_short_b() {
    let _gpu_guard = gpu_test_lock();
    let Some(_dev) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    let mut layer = make_valid_layer(hidden, rank);
    layer.b.pop(); // one element short of d_out * rank
    let result = state.load_lora_adapter(vec![layer], 1.0, None);
    assert!(result.is_err(), "short B must return Err");
}

#[test]
fn load_lora_adapter_rejects_zero_rank() {
    let _gpu_guard = gpu_test_lock();
    let Some(_dev) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let hidden = cfg.hidden_size;
    let layer = LoraLayerData {
        layer_idx: 0,
        module: "q_proj".into(),
        a: vec![],
        b: vec![],
        rank: 0,
        d_in: hidden,
        d_out: hidden,
    };
    let result = state.load_lora_adapter(vec![layer], 1.0, None);
    assert!(result.is_err(), "zero rank must return Err");
}

#[test]
fn load_lora_adapter_rejects_zero_d_in() {
    let _gpu_guard = gpu_test_lock();
    let Some(_dev) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let layer = LoraLayerData {
        layer_idx: 0,
        module: "q_proj".into(),
        a: vec![],
        b: vec![],
        rank: 4,
        d_in: 0,
        d_out: cfg.hidden_size,
    };
    let result = state.load_lora_adapter(vec![layer], 1.0, None);
    assert!(result.is_err(), "zero d_in must return Err");
}

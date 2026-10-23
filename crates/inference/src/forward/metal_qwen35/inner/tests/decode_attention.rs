//! Decode-attention kernels against the CPU reference.

use super::*;
// -----------------------------------------------------------------------
// decode_attention parity tests
//
// Compare the old 3-pass reference kernel against the new flash GQA kernel
// for all 6 required configurations and key edge cases.
//
// Tolerance gate (from optimization_plan.md §5):
//   max_abs_diff <= 5e-2, mean_abs_diff <= 5e-3
// Measured in optimizations.md: max_abs_diff < 1e-7 for all 6 configs.
// -----------------------------------------------------------------------

fn xorshift32_parity(state: &mut u32) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    (x as f32 / u32::MAX as f32) - 0.5
}

/// Run old reference kernel and new flash kernel on identical inputs.
/// Returns `(max_abs_diff, mean_abs_diff, nan_count)`.
fn parity_check(
    device: &Device,
    queue: &CommandQueue,
    pipe_ref: &ComputePipelineState,
    pipe_flash: &ComputePipelineState,
    pipe_partial: &ComputePipelineState,
    pipe_reduce: &ComputePipelineState,
    num_q_heads: u32,
    num_kv_heads: u32,
    cache_len: u32,
) -> (f32, f32, usize) {
    const HEAD_DIM: u32 = 256;
    const PARTITION_TOKENS: u32 = 1024;
    const DIRECT_THRESHOLD: u32 = 512;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let q_dim = num_q_heads * HEAD_DIM;
    let kv_dim = num_kv_heads * HEAD_DIM;
    let num_partitions = cache_len.div_ceil(PARTITION_TOKENS);

    let q_buf = device.new_buffer((q_dim as u64) * 4, MTLResourceOptions::StorageModeShared);
    let k_buf = device.new_buffer(
        (cache_len as u64) * (kv_dim as u64) * 4,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer(
        (cache_len as u64) * (kv_dim as u64) * 4,
        MTLResourceOptions::StorageModeShared,
    );
    let out_ref = device.new_buffer((q_dim as u64) * 4, MTLResourceOptions::StorageModeShared);
    let out_flash = device.new_buffer((q_dim as u64) * 4, MTLResourceOptions::StorageModeShared);
    let partials_buf = device.new_buffer(
        (num_partitions as u64) * (num_q_heads as u64) * ((HEAD_DIM + 2) as u64) * 4,
        MTLResourceOptions::StorageModeShared,
    );

    let mut rng: u32 = 0xDEAD_BEEF;
    // SAFETY: buffers are StorageModeShared, freshly allocated, correctly sized.
    unsafe {
        let q = std::slice::from_raw_parts_mut(q_buf.contents() as *mut f32, q_dim as usize);
        for v in q.iter_mut() {
            *v = xorshift32_parity(&mut rng);
        }
        let k = std::slice::from_raw_parts_mut(
            k_buf.contents() as *mut f32,
            (cache_len * kv_dim) as usize,
        );
        for v in k.iter_mut() {
            *v = xorshift32_parity(&mut rng);
        }
        let vd = std::slice::from_raw_parts_mut(
            v_buf.contents() as *mut f32,
            (cache_len * kv_dim) as usize,
        );
        for v in vd.iter_mut() {
            *v = xorshift32_parity(&mut rng);
        }
    }

    // Reference: one threadgroup per Q head
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe_ref);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&k_buf), 0);
        enc.set_buffer(2, Some(&v_buf), 0);
        enc.set_buffer(3, Some(&out_ref), 0);
        enc.set_bytes(4, 4, &cache_len as *const u32 as *const _);
        enc.set_bytes(5, 4, &HEAD_DIM as *const u32 as *const _);
        enc.set_bytes(6, 4, &num_q_heads as *const u32 as *const _);
        enc.set_bytes(7, 4, &num_kv_heads as *const u32 as *const _);
        enc.set_bytes(8, 4, &q_dim as *const u32 as *const _);
        enc.set_bytes(9, 4, &kv_dim as *const u32 as *const _);
        enc.set_bytes(10, 4, &scale as *const f32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(num_q_heads as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Flash: direct (cache_len <= 512) or partitioned (cache_len > 512)
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        if cache_len <= DIRECT_THRESHOLD {
            enc.set_compute_pipeline_state(pipe_flash);
            enc.set_buffer(0, Some(&q_buf), 0);
            enc.set_buffer(1, Some(&k_buf), 0);
            enc.set_buffer(2, Some(&v_buf), 0);
            enc.set_buffer(3, Some(&out_flash), 0);
            enc.set_bytes(4, 4, &cache_len as *const u32 as *const _);
            enc.set_bytes(5, 4, &HEAD_DIM as *const u32 as *const _);
            enc.set_bytes(6, 4, &num_q_heads as *const u32 as *const _);
            enc.set_bytes(7, 4, &num_kv_heads as *const u32 as *const _);
            enc.set_bytes(8, 4, &q_dim as *const u32 as *const _);
            enc.set_bytes(9, 4, &kv_dim as *const u32 as *const _);
            enc.set_bytes(10, 4, &scale as *const f32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(num_kv_heads as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        } else {
            enc.set_compute_pipeline_state(pipe_partial);
            enc.set_buffer(0, Some(&q_buf), 0);
            enc.set_buffer(1, Some(&k_buf), 0);
            enc.set_buffer(2, Some(&v_buf), 0);
            enc.set_buffer(3, Some(&partials_buf), 0);
            enc.set_bytes(4, 4, &cache_len as *const u32 as *const _);
            enc.set_bytes(5, 4, &HEAD_DIM as *const u32 as *const _);
            enc.set_bytes(6, 4, &num_q_heads as *const u32 as *const _);
            enc.set_bytes(7, 4, &num_kv_heads as *const u32 as *const _);
            enc.set_bytes(8, 4, &q_dim as *const u32 as *const _);
            enc.set_bytes(9, 4, &kv_dim as *const u32 as *const _);
            enc.set_bytes(10, 4, &scale as *const f32 as *const _);
            enc.set_bytes(11, 4, &PARTITION_TOKENS as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(num_kv_heads as u64, num_partitions as u64, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.set_compute_pipeline_state(pipe_reduce);
            enc.set_buffer(0, Some(&partials_buf), 0);
            enc.set_buffer(1, Some(&out_flash), 0);
            enc.set_bytes(2, 4, &num_q_heads as *const u32 as *const _);
            enc.set_bytes(3, 4, &num_kv_heads as *const u32 as *const _);
            enc.set_bytes(4, 4, &num_partitions as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(num_kv_heads as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
    }

    // SAFETY: both command buffers completed; buffers are f32 StorageModeShared.
    unsafe {
        let a = std::slice::from_raw_parts(out_ref.contents() as *const f32, q_dim as usize);
        let b = std::slice::from_raw_parts(out_flash.contents() as *const f32, q_dim as usize);
        let mut max_d = 0.0f32;
        let mut sum_d = 0.0f32;
        let mut nans = 0usize;
        for (x, y) in a.iter().zip(b.iter()) {
            if !y.is_finite() {
                nans += 1;
            }
            let d = (x - y).abs();
            if d > max_d {
                max_d = d;
            }
            sum_d += d;
        }
        (max_d, sum_d / a.len() as f32, nans)
    }
}

/// Compile all three flash pipelines and the reference pipeline once.
/// Returns None on non-Metal systems (test is skipped).
fn build_parity_pipelines(
    device: &Device,
) -> Option<(
    ComputePipelineState,
    ComputePipelineState,
    ComputePipelineState,
    ComputePipelineState,
)> {
    let opts = CompileOptions::new();
    let lib_ref = device
        .new_library_with_source(MSL_DECODE_REFERENCE, &opts)
        .ok()?;
    let fn_ref = lib_ref
        .get_function("decode_attention_reference", None)
        .ok()?;
    let pipe_ref = device
        .new_compute_pipeline_state_with_function(&fn_ref)
        .ok()?;

    let lib_flash = device
        .new_library_with_source(MSL_SOURCE, &opts)
        .expect("MSL_SOURCE must compile");
    let pipe_flash = device
        .new_compute_pipeline_state_with_function(
            &lib_flash
                .get_function("decode_attention", None)
                .expect("decode_attention not in MSL_SOURCE"),
        )
        .expect("decode_attention pipeline");
    let pipe_partial = device
        .new_compute_pipeline_state_with_function(
            &lib_flash
                .get_function("decode_attention_flash_partial", None)
                .expect("decode_attention_flash_partial not in MSL_SOURCE"),
        )
        .expect("flash_partial pipeline");
    let pipe_reduce = device
        .new_compute_pipeline_state_with_function(
            &lib_flash
                .get_function("decode_attention_flash_reduce", None)
                .expect("decode_attention_flash_reduce not in MSL_SOURCE"),
        )
        .expect("flash_reduce pipeline");

    Some((pipe_ref, pipe_flash, pipe_partial, pipe_reduce))
}

const STRICT_MAX_ABS_TOL: f32 = 1e-4;

fn assert_decode_attention_parity_case(
    device: &Device,
    queue: &CommandQueue,
    pipes: &(
        ComputePipelineState,
        ComputePipelineState,
        ComputePipelineState,
        ComputePipelineState,
    ),
    num_q_heads: u32,
    num_kv_heads: u32,
    cache_len: u32,
) {
    let (pipe_ref, pipe_flash, pipe_partial, pipe_reduce) = pipes;
    let (max_d, mean_d, nans) = parity_check(
        device,
        queue,
        pipe_ref,
        pipe_flash,
        pipe_partial,
        pipe_reduce,
        num_q_heads,
        num_kv_heads,
        cache_len,
    );
    assert_eq!(
        nans, 0,
        "{num_q_heads}Q/{num_kv_heads}KV cache={cache_len}: NaN in flash output"
    );
    assert!(
        max_d < STRICT_MAX_ABS_TOL,
        "{num_q_heads}Q/{num_kv_heads}KV cache={cache_len}: max_abs_diff={max_d:.3e}, mean_abs_diff={mean_d:.3e}, tolerance={STRICT_MAX_ABS_TOL:.3e}"
    );
}

/// Strict parity gate: required matrix {128,512,2048} × {8Q/2KV,16Q/2KV}, tolerance 1e-4.
#[test]
fn test_decode_attention_parity_required_matrix() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let Some(pipes) = build_parity_pipelines(&device) else {
        return;
    };

    for &(num_q_heads, num_kv_heads) in &[(8u32, 2u32), (16u32, 2u32)] {
        for &cache_len in &[128u32, 512u32, 2048u32] {
            assert_decode_attention_parity_case(
                &device,
                &queue,
                &pipes,
                num_q_heads,
                num_kv_heads,
                cache_len,
            );
        }
    }
}

// --- 6 required configurations ---

#[test]
fn test_decode_attention_parity_8q2kv_512() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let Some((pr, pf, pp, prr)) = build_parity_pipelines(&device) else {
        return;
    };
    let (max_d, _mean_d, nans) = parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 512);
    assert_eq!(nans, 0, "8Q/2KV cache=512: NaN in flash output");
    assert!(
        max_d < STRICT_MAX_ABS_TOL,
        "8Q/2KV cache=512: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
    );
}

#[test]
fn test_decode_attention_parity_8q2kv_4096() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let Some((pr, pf, pp, prr)) = build_parity_pipelines(&device) else {
        return;
    };
    let (max_d, _mean_d, nans) = parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 4096);
    assert_eq!(nans, 0, "8Q/2KV cache=4096: NaN in flash output");
    assert!(
        max_d < STRICT_MAX_ABS_TOL,
        "8Q/2KV cache=4096: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
    );
}

#[test]
#[ignore = "slow: reference kernel ~721ms at cache_len=32768; run with --include-ignored"]
fn test_decode_attention_parity_8q2kv_32768() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let Some((pr, pf, pp, prr)) = build_parity_pipelines(&device) else {
        return;
    };
    let (max_d, _mean_d, nans) = parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 32768);
    assert_eq!(nans, 0, "8Q/2KV cache=32768: NaN in flash output");
    assert!(
        max_d < STRICT_MAX_ABS_TOL,
        "8Q/2KV cache=32768: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
    );
}

#[test]
fn test_decode_attention_parity_16q2kv_512() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let Some((pr, pf, pp, prr)) = build_parity_pipelines(&device) else {
        return;
    };
    let (max_d, _mean_d, nans) = parity_check(&device, &queue, &pr, &pf, &pp, &prr, 16, 2, 512);
    assert_eq!(nans, 0, "16Q/2KV cache=512: NaN in flash output");
    assert!(
        max_d < STRICT_MAX_ABS_TOL,
        "16Q/2KV cache=512: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
    );
}

#[test]
fn test_decode_attention_parity_16q2kv_4096() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let Some((pr, pf, pp, prr)) = build_parity_pipelines(&device) else {
        return;
    };
    let (max_d, _mean_d, nans) = parity_check(&device, &queue, &pr, &pf, &pp, &prr, 16, 2, 4096);
    assert_eq!(nans, 0, "16Q/2KV cache=4096: NaN in flash output");
    assert!(
        max_d < STRICT_MAX_ABS_TOL,
        "16Q/2KV cache=4096: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
    );
}

#[test]
#[ignore = "slow: reference kernel ~723ms at cache_len=32768; run with --include-ignored"]
fn test_decode_attention_parity_16q2kv_32768() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let Some((pr, pf, pp, prr)) = build_parity_pipelines(&device) else {
        return;
    };
    let (max_d, _mean_d, nans) = parity_check(&device, &queue, &pr, &pf, &pp, &prr, 16, 2, 32768);
    assert_eq!(nans, 0, "16Q/2KV cache=32768: NaN in flash output");
    assert!(
        max_d < STRICT_MAX_ABS_TOL,
        "16Q/2KV cache=32768: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
    );
}

// --- Edge cases ---

/// cache_len=1: first-token decode. The only cached token gets weight 1.0,
/// so flash output must equal the old reference exactly (same f32 arithmetic).
#[test]
fn test_decode_attention_edge_cache_len_1() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let Some((pr, pf, pp, prr)) = build_parity_pipelines(&device) else {
        return;
    };
    // Use 8Q/2KV; first token is on the direct path (cache_len=1 <= 512)
    let (max_d, mean_d, nans) = parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 1);
    assert_eq!(nans, 0, "cache_len=1: NaN in output");
    // Tighter tolerance: single-token softmax = 1.0 exactly, output = V[0]
    assert!(
        max_d <= 1e-5,
        "cache_len=1: max_abs_diff={max_d:.3e} (expected near 0)"
    );
    assert!(
        mean_d <= 1e-6,
        "cache_len=1: mean_abs_diff={mean_d:.3e} (expected near 0)"
    );
}

/// cache_len=257: crosses the 256-token TILE_TOKENS boundary (direct path).
#[test]
fn test_decode_attention_edge_cache_len_257() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let Some((pr, pf, pp, prr)) = build_parity_pipelines(&device) else {
        return;
    };
    let (max_d, _mean_d, nans) = parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 257);
    assert_eq!(nans, 0, "cache_len=257: NaN in output");
    assert!(
        max_d < STRICT_MAX_ABS_TOL,
        "cache_len=257: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
    );
}

/// cache_len=513: just above DIRECT_THRESHOLD — forces partitioned path.
#[test]
fn test_decode_attention_edge_cache_len_513() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let Some((pr, pf, pp, prr)) = build_parity_pipelines(&device) else {
        return;
    };
    let (max_d, _mean_d, nans) = parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 513);
    assert_eq!(nans, 0, "cache_len=513: NaN in output");
    assert!(
        max_d < STRICT_MAX_ABS_TOL,
        "cache_len=513: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
    );
}

/// cache_len=1025: crosses PARTITION_TOKENS boundary in the partitioned path.
#[test]
fn test_decode_attention_edge_cache_len_1025() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let Some((pr, pf, pp, prr)) = build_parity_pipelines(&device) else {
        return;
    };
    let (max_d, _mean_d, nans) = parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 1025);
    assert_eq!(nans, 0, "cache_len=1025: NaN in output");
    assert!(
        max_d < STRICT_MAX_ABS_TOL,
        "cache_len=1025: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
    );
}

// ── Route-logic unit tests (pure Rust, no Metal device needed) ────────

#[test]
fn sampling_route_legacy_helper_all_cases() {
    // k=0 → always CPU
    assert_eq!(
        choose_gpu_topk_route(0, true, true),
        GpuTopkRoute::CpuFallback,
        "k=0"
    );
    // compact_env=false → always CPU
    assert_eq!(
        choose_gpu_topk_route(1, false, false),
        GpuTopkRoute::CpuFallback,
        "k=1, compact_env=false"
    );
    assert_eq!(
        choose_gpu_topk_route(50, false, true),
        GpuTopkRoute::CpuFallback,
        "k=50, compact_env=false"
    );
    // k=1, compact_env=true → CpuFallback (GPU 2.70× slower than CPU NEON after R1)
    assert_eq!(
        choose_gpu_topk_route(1, true, false),
        GpuTopkRoute::CpuFallback,
        "k=1, compact_env=true, no select"
    );
    assert_eq!(
        choose_gpu_topk_route(1, true, true),
        GpuTopkRoute::CpuFallback,
        "k=1, compact_env=true, select"
    );
    // k=2..=64, compact_env=true, no selection_env → CpuFallback
    for k in [2usize, 10, 50, 64] {
        assert_eq!(
            choose_gpu_topk_route(k, true, false),
            GpuTopkRoute::CpuFallback,
            "k={k}, no LATTICE_COMPACT_TOPK_SELECT"
        );
    }
    // k=50 with both env flags → HierarchicalK50 (R2 experimental).
    assert_eq!(
        choose_gpu_topk_route(50, true, true),
        GpuTopkRoute::HierarchicalK50,
        "k=50 with both env flags must use HierarchicalK50"
    );
    // Other k>1 values must not enter any GPU route.
    for k in [2usize, 10, 49, 51, 64, 65, 128, 256] {
        assert_eq!(
            choose_gpu_topk_route(k, true, true),
            GpuTopkRoute::CpuFallback,
            "k={k} must not enter any GPU top-k route"
        );
    }
    // k>256 → always CpuFallback
    assert_eq!(
        choose_gpu_topk_route(257, true, true),
        GpuTopkRoute::CpuFallback,
        "k=257"
    );
    assert_eq!(
        choose_gpu_topk_route(1000, true, true),
        GpuTopkRoute::CpuFallback,
        "k=1000"
    );
}

#[test]
fn sampling_route_block_helper_gate() {
    assert_eq!(
        choose_gpu_block_topk_route(8, 0.9, true, false),
        GpuTopkRoute::CpuFallback,
        "top_p<1.0 without the approx gate must stay exact"
    );
    assert_eq!(
        choose_gpu_block_topk_route(8, 0.9, true, true),
        GpuTopkRoute::BlockTopK { local_k: 8 },
        "top_p<1.0 WITH the explicit approx gate may take the block route"
    );
    assert_eq!(
        choose_gpu_block_topk_route(0, 1.0, true, false),
        GpuTopkRoute::CpuFallback,
        "top_k=0 means top-k DISABLED (full-distribution sampling), \
                 not greedy; BlockArgmax here would collapse sampling to top-1"
    );
    assert_eq!(
        choose_gpu_block_topk_route(1, 1.0, true, false),
        GpuTopkRoute::BlockArgmax
    );
    for &k in &[8u32, 16, 40, 64] {
        assert_eq!(
            choose_gpu_block_topk_route(k as usize, 1.0, true, false),
            GpuTopkRoute::BlockTopK { local_k: k },
            "k={k} must map to BlockTopK"
        );
    }
    assert_eq!(
        choose_gpu_block_topk_route(50, 1.0, true, false),
        GpuTopkRoute::CpuFallback,
        "k=50 has no precompiled Stage-1 variant; must fall back exact"
    );
    assert_eq!(
        choose_gpu_block_topk_route(8, 1.0, false, false),
        GpuTopkRoute::CpuFallback,
        "LATTICE_COMPACT_TOPK unset must stay exact regardless of k"
    );
}

fn compact_sampling_config(top_k: usize) -> GenerateConfig {
    GenerateConfig {
        top_k,
        top_p: 1.0,
        repetition_penalty: 1.0,
        ..Default::default()
    }
}

fn compact_sampling_environment() -> SamplingRouteEnvironment {
    SamplingRouteEnvironment {
        compact: true,
        selection: true,
        approximate_top_p: false,
    }
}

/// `SamplingRouteEnvironment::current()` reads `LATTICE_COMPACT_TOPK` through
/// `env_switch_enabled`, by value, not presence (#1614): `=0` must disable
/// compact routing rather than enable it by merely being present.
#[test]
fn sampling_route_environment_reads_compact_topk_by_value_not_presence() {
    let disabled = with_compact_topk_env("0", SamplingRouteEnvironment::current);
    assert!(
        !disabled.compact,
        "LATTICE_COMPACT_TOPK=0 must not enable compact routing by presence"
    );

    let enabled = with_compact_topk_env("1", SamplingRouteEnvironment::current);
    assert!(
        enabled.compact,
        "LATTICE_COMPACT_TOPK=1 must still enable compact routing"
    );
}

#[test]
fn sampling_route_plan_composes_block_first_then_legacy_fallback() {
    for (top_k, expected_route, expected_topk) in [
        (1, GpuTopkRoute::BlockArgmax, 1),
        (8, GpuTopkRoute::BlockTopK { local_k: 8 }, 8),
        (16, GpuTopkRoute::BlockTopK { local_k: 16 }, 16),
        (40, GpuTopkRoute::BlockTopK { local_k: 40 }, 40),
        (64, GpuTopkRoute::BlockTopK { local_k: 64 }, 64),
    ] {
        let plan = plan_sampling_route(
            &compact_sampling_config(top_k),
            false,
            compact_sampling_environment(),
        );
        assert_eq!(
            plan,
            SamplingRoutePlan {
                use_compact: true,
                compact_route: expected_route,
                compact_topk: expected_topk,
            },
            "top_k={top_k}"
        );
    }

    let legacy_plan = plan_sampling_route(
        &compact_sampling_config(50),
        false,
        compact_sampling_environment(),
    );
    assert_eq!(
        legacy_plan,
        SamplingRoutePlan {
            use_compact: true,
            compact_route: GpuTopkRoute::HierarchicalK50,
            compact_topk: 50,
        }
    );
}

/// `plan_sampling_route` no longer takes a caller-distinguishing logprobs
/// policy: every caller (direct, streaming, prefix-cache generation)
/// reaches this function only once its own `check_logprobs_not_set`
/// preflight has already rejected `logprobs: Some(_)` for paths that
/// don't support it, or (streaming) captures logprobs itself and must
/// keep full logits. Both configurations below are reachable states a
/// real caller can pass in: `logprobs: None` (every caller, after its
/// own preflight) and `logprobs: Some(_)` (the streaming caller, which
/// has no such preflight and captures logprobs directly).
#[test]
fn sampling_route_plan_requires_full_logits_when_logprobs_requested() {
    let no_logprobs = compact_sampling_config(1);
    assert!(
        plan_sampling_route(&no_logprobs, false, compact_sampling_environment()).use_compact,
        "no logprobs requested must stay eligible for compact sampling"
    );

    let with_logprobs = GenerateConfig {
        min_p: 0.0,
        logprobs: Some(5),
        ..compact_sampling_config(1)
    };
    assert_eq!(
        plan_sampling_route(&with_logprobs, false, compact_sampling_environment()),
        SamplingRoutePlan {
            use_compact: false,
            compact_route: GpuTopkRoute::CpuFallback,
            compact_topk: 0,
        },
        "a requested logprobs capture requires full logits"
    );
}

#[test]
fn sampling_route_plan_preserves_grammar_and_repetition_gates() {
    let penalized = GenerateConfig {
        min_p: 0.0,
        repetition_penalty: 1.1,
        ..compact_sampling_config(8)
    };
    assert!(
        !plan_sampling_route(&penalized, false, compact_sampling_environment(),).use_compact,
        "non-empty history with a repetition penalty requires full logits"
    );
    assert!(
        plan_sampling_route(&penalized, true, compact_sampling_environment(),).use_compact,
        "an empty history preserves the existing repetition-penalty exception"
    );

    use crate::grammar::{GrammarEngine, GrammarSpec};
    let grammar = GrammarEngine::new(
        &GrammarSpec::Gbnf("root ::= \"a\"\n".to_string()),
        vec![b"a".to_vec()],
    )
    .expect("trivial grammar must compile");
    let constrained = GenerateConfig {
        min_p: 0.0,
        grammar: Some(std::sync::Arc::new(grammar)),
        ..compact_sampling_config(8)
    };
    assert!(
        !plan_sampling_route(&constrained, false, compact_sampling_environment(),).use_compact,
        "grammar-constrained decoding requires full logits"
    );
}

#[test]
fn sampling_route_state_transition_preserves_enabled_result_and_clears_fallback() {
    let sentinel = crate::sampling::Candidate {
        token_id: 17,
        logit: 3.5,
    };
    let mut compact_route = GpuTopkRoute::CpuFallback;
    let mut compact_topk = 0;
    let mut compact_result = vec![sentinel];

    let enabled = plan_sampling_route(
        &compact_sampling_config(8),
        false,
        compact_sampling_environment(),
    );
    assert!(apply_sampling_route_plan(
        enabled,
        &mut compact_route,
        &mut compact_topk,
        &mut compact_result,
    ));
    assert_eq!(compact_route, GpuTopkRoute::BlockTopK { local_k: 8 });
    assert_eq!(compact_topk, 8);
    assert_eq!(
        compact_result,
        vec![sentinel],
        "eligible routing preserves the existing compact_result until prefill replaces it"
    );

    let disabled = plan_sampling_route(
        &compact_sampling_config(8),
        false,
        SamplingRouteEnvironment {
            compact: false,
            selection: true,
            approximate_top_p: true,
        },
    );
    assert!(!apply_sampling_route_plan(
        disabled,
        &mut compact_route,
        &mut compact_topk,
        &mut compact_result,
    ));
    assert_eq!(compact_route, GpuTopkRoute::CpuFallback);
    assert_eq!(compact_topk, 0);
    assert!(
        compact_result.is_empty(),
        "fallback routing must clear stale compact candidates"
    );
}

/// Direct generation's `logprobs: Some(_)` admission is rejected by
/// `prepare_direct_generation`'s shared `GenerationEntryContract::MetalDirect`
/// capability check, strictly before either of the two paths that mutate
/// `InferenceSession::compact_route` / `compact_topk` /
/// `compact_result` can run: `reset_state()` and route planning
/// (`plan_sampling_route`). `generate()` propagates that `Err` via
/// `?` immediately after the `prepare_direct_generation` call, so a rejected
/// request must leave route state untouched.
///
/// This replaces a prior test that only proved the ordering held in
/// the *source text* (`prepare_direct_generation` found lexically before
/// `configure_sampling_route`, with an unbounded "rest of the file"
/// slice as the search space). A source-text match cannot tell
/// whether the matched `configure_sampling_route` call even belongs
/// to `generate()`, and it cannot catch a caller that stores
/// `prepare_direct_generation`'s `Result` in a local, calls
/// `configure_sampling_route` unconditionally, and only applies `?`
/// afterward: textual order is preserved, the guard's `Err` is still
/// returned, yet route state has already been mutated. This test
/// proves the real invariant behaviourally instead: seed route state
/// to values `plan_sampling_route` would never leave in place for a
/// rejected request, run the real `generate()`, and assert the
/// seeded state survives untouched.
///
/// Note the boundary this test does *not* cover: a `max_new_tokens:
/// 0` request short-circuits to `GenerationPreparation::Complete` inside
/// `prepare_direct_generation` *before* the logprobs capability check ever
/// runs, so a zero-budget request with `logprobs: Some(_)` is never
/// rejected at all -- it returns `Ok` and never touches sampling
/// routing either way. That path is untested here and must not be
/// read as if this test covered it.
///
/// Mutation sensitivity: reordering `generate()` to call
/// `plan_sampling_route` before applying `prepare_direct_generation`'s
/// `?` -- even while keeping the *textual* prepare-before-configure
/// ordering -- lets this rejected request mutate `compact_route` /
/// `compact_topk` / `compact_result` before the error is returned, so
/// the "state unchanged" assertions below fail.
#[test]
fn direct_generate_rejects_logprobs_before_configuring_sampling_route() {
    let Some(_) = metal::Device::system_default() else {
        panic!(
            "this behavioural admission-order proof requires a real \
                     Metal device and must fail closed rather than silently \
                     skip -- a skipped run would report a pass over \
                     assertions that never executed"
        );
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    // Sentinel values `plan_sampling_route` would never produce for
    // this (rejected) request: a real route/topk pair, and a
    // non-empty result buffer. If route planning runs at
    // all, `apply_sampling_route_plan` unconditionally overwrites
    // `compact_route`/`compact_topk` from its plan and clears
    // `compact_result` whenever `use_compact` is false -- which it
    // always is here, since `logprobs: Some(_)` forces
    // `logprobs_eligible = false`.
    let sentinel_route = GpuTopkRoute::BlockArgmax;
    let sentinel_topk = 7usize;
    let sentinel_result = vec![crate::sampling::Candidate {
        token_id: 99,
        logit: 1.23,
    }];
    state.session.compact_route = sentinel_route;
    state.session.compact_topk = sentinel_topk;
    state.session.compact_result = sentinel_result.clone();

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: vec![],
        enable_thinking: true,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: Some(0),
    };

    let result = state.generate("a", &tokenizer, &gen_cfg);
    assert!(
        matches!(result, Err(crate::error::InferenceError::InvalidInput(_))),
        "logprobs: Some(_) with a nonzero budget must be rejected with \
                 InvalidInput; got {result:?}"
    );

    assert_eq!(
        state.session.compact_route, sentinel_route,
        "a rejected logprobs request must not mutate compact_route -- \
                 route planning must be unreachable once the \
                 preflight guard errors"
    );
    assert_eq!(
        state.session.compact_topk, sentinel_topk,
        "a rejected logprobs request must not mutate compact_topk"
    );
    assert_eq!(
        state.session.compact_result, sentinel_result,
        "a rejected logprobs request must not mutate compact_result"
    );
}

// ── GPU-CPU argmax parity tests (requires Metal device) ──────────────

#[test]
fn test_gpu_argmax_parity_k1() {
    let _gpu_guard = gpu_test_lock();
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let opts = CompileOptions::new();
    let library = device
        .new_library_with_source(MSL_SOURCE, &opts)
        .expect("MSL compile");
    let first_fn = library
        .get_function("logits_argmax_first", None)
        .expect("logits_argmax_first");
    let merge_fn = library
        .get_function("logits_argmax_merge", None)
        .expect("logits_argmax_merge");
    let first_pipe = device
        .new_compute_pipeline_state_with_function(&first_fn)
        .expect("argmax_first pipeline");
    let merge_pipe = device
        .new_compute_pipeline_state_with_function(&merge_fn)
        .expect("argmax_merge pipeline");

    // Dispatch GPU argmax and return the winning token_id.
    let gpu_argmax = |logits: &[f32]| -> u32 {
        let vocab_size = logits.len() as u32;
        let groups = vocab_size.div_ceil(1024);
        let in_buf = make_buffer(&device, logits, "logits");
        // scratch_a: groups × TopKCandidate (8 bytes = float logit + uint token_id)
        let scratch_a_zeros: Vec<f32> = vec![0.0; (groups as usize) * 2];
        let scratch_a = make_buffer(&device, &scratch_a_zeros, "scratch_a");
        // scratch_b: 1 TopKCandidate
        let scratch_b = make_buffer(&device, &[0.0f32; 2], "scratch_b");

        let cmd = queue.new_command_buffer();
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&first_pipe);
            enc.set_buffer(0, Some(&in_buf), 0);
            enc.set_buffer(1, Some(&scratch_a), 0);
            enc.set_bytes(2, 4, &vocab_size as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(1024, 1, 1));
            enc.set_compute_pipeline_state(&merge_pipe);
            enc.set_buffer(0, Some(&scratch_a), 0);
            enc.set_buffer(1, Some(&scratch_b), 0);
            enc.set_bytes(2, 4, &groups as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();

        // SAFETY: command buffer completed; scratch_b layout is
        // TopKCandidate { float logit; uint token_id; } — token_id at bytes [4..8].
        unsafe {
            let ptr = scratch_b.contents() as *const u8;
            let token_id_bytes = std::slice::from_raw_parts(ptr.add(4), 4);
            u32::from_le_bytes([
                token_id_bytes[0],
                token_id_bytes[1],
                token_id_bytes[2],
                token_id_bytes[3],
            ])
        }
    };

    // Mirror of sampling.rs:argmax_f32 — NaN never beats a real value.
    let cpu_argmax = |logits: &[f32]| -> u32 {
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        best_idx
    };

    struct Case {
        name: &'static str,
        logits: Vec<f32>,
        expect_gpu_eq_cpu: bool,
    }

    // When all values are NaN the CPU returns 0 (initial best_idx) but the
    // GPU returns UINT_MAX (sentinel token_id).  This is a documented
    // divergence; in practice logits never contain only NaN at inference time.
    let cases: Vec<Case> = vec![
        Case {
            name: "single element",
            logits: vec![1.0f32],
            expect_gpu_eq_cpu: true,
        },
        Case {
            name: "small random (5 elem)",
            logits: vec![0.1f32, 0.9, 0.3, 0.7, 0.5],
            expect_gpu_eq_cpu: true,
        },
        Case {
            name: "sorted descending (max first)",
            logits: (0u32..256).rev().map(|i| i as f32).collect(),
            expect_gpu_eq_cpu: true,
        },
        Case {
            name: "reverse sorted (max last)",
            logits: (0u32..256).map(|i| i as f32).collect(),
            expect_gpu_eq_cpu: true,
        },
        Case {
            name: "all same values (tie → lowest id = 0)",
            logits: vec![1.5f32; 4096],
            expect_gpu_eq_cpu: true,
        },
        Case {
            name: "full vocab 248320 random",
            logits: {
                let mut rng = 42u64;
                (0..248_320)
                    .map(|_| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        ((rng as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32
                    })
                    .collect()
            },
            expect_gpu_eq_cpu: true,
        },
        Case {
            name: "partial NaN (non-NaN max wins)",
            logits: vec![1.0f32, f32::NAN, 5.0, f32::NAN, 3.0],
            expect_gpu_eq_cpu: true,
        },
        Case {
            name: "-Inf values (finite max wins)",
            logits: vec![f32::NEG_INFINITY, 2.0f32, f32::NEG_INFINITY, 1.0],
            expect_gpu_eq_cpu: true,
        },
    ];

    let mut failures = Vec::new();
    for case in &cases {
        let cpu = cpu_argmax(&case.logits);
        let gpu = gpu_argmax(&case.logits);
        let matches = gpu == cpu;
        if case.expect_gpu_eq_cpu && !matches {
            failures.push(format!("MISMATCH {}: cpu={cpu} gpu={gpu}", case.name));
        }
    }

    assert!(
        failures.is_empty(),
        "GPU-CPU argmax parity failures:\n{}",
        failures.join("\n")
    );
}

fn assert_text_chat_image_rejected<T>(result: Result<T, crate::error::InferenceError>) {
    match result {
        Err(crate::error::InferenceError::InvalidInput(message)) => {
            assert_eq!(message, TEXT_CHAT_IMAGE_ERROR);
        }
        Err(error) => panic!("inline image returned the wrong error: {error}"),
        Ok(_) => panic!("text-only chat entry point silently accepted an inline image"),
    }
}

#[test]
fn text_chat_image_guard_rejects_without_a_metal_device() {
    assert_text_chat_image_rejected(reject_inline_images_for_text_chat(&[
        ChatMessage::user_with_image("beforeafter", vec![1, 2, 3], "before".len()),
    ]));
}

#[test]
fn text_chat_entrypoints_reject_inline_images_before_generation() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny Metal chat state");
    let tokenizer = minimal_bpe_tokenizer();
    let gen_cfg = GenerateConfig::default();
    let messages = [ChatMessage::user_with_image(
        "beforeafter",
        vec![0x89, b'P', b'N', b'G'],
        "before".len(),
    )];

    assert_text_chat_image_rejected(state.chat_completion(&messages, &tokenizer, &gen_cfg));
    assert_text_chat_image_rejected(state.chat_completion_streaming(
        &messages,
        &tokenizer,
        &gen_cfg,
        |_, _| true,
    ));
    assert_text_chat_image_rejected(state.chat_completion_streaming_with_cancel(
        &messages,
        &tokenizer,
        &gen_cfg,
        |_, _| true,
        || false,
    ));
    assert_text_chat_image_rejected(state.chat_completion_streaming_with_prefix_cache(
        crate::kv_cache::CrossTurnSlotId::DEFAULT,
        &messages,
        &tokenizer,
        &gen_cfg,
        |_, _| true,
    ));
    assert_text_chat_image_rejected(
        state.chat_completion_streaming_with_prefix_cache_and_cancel(
            crate::kv_cache::CrossTurnSlotId::DEFAULT,
            &messages,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || false,
        ),
    );
}

fn rotate_embedding_rows_for_test(
    weights: &mut ModelWeights,
    hidden: usize,
    rot: &crate::quant::quarot::hadamard::RandomizedHadamard,
) {
    for row in weights.embed_tokens.chunks_exact_mut(hidden) {
        rot.apply(row).unwrap();
    }
}

// ADR-051 §"Verification" line 213: MTP draft-logit equivalence test.
// Verifies that counter-rotating MTP inputs (apply_inverse) and re-rotating the
// MTP output (apply) before the shared lm_head produces logits identical to the
// unrotated reference path, within Metal f16/Q8 quantization tolerance.
#[test]
fn mtp_draft_logit_equivalence_with_quarot_counter_rotation() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };

    let (mut cfg, reference_weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;

    let seed = 0x51_51_51_u64;
    let rotation =
        crate::quant::quarot::hadamard::RandomizedHadamard::new(seed, cfg.hidden_size).unwrap();

    // Build rotated weights independently (ModelWeights has no Clone).
    let (_, mut rotated_weights) = tiny_metal_qwen35_fixture();
    rotate_embedding_rows_for_test(&mut rotated_weights, cfg.hidden_size, &rotation);

    let mut reference_state =
        metal_state_with_synthetic_mtp_for_test(&reference_weights, &cfg, None);
    let mut rotated_state =
        metal_state_with_synthetic_mtp_for_test(&rotated_weights, &cfg, Some(rotation.clone()));

    // Inject a realistic pre-final hidden state. Both paths get the same
    // mathematical vector; the rotated path gets R applied so that
    // apply_inverse inside mtp_forward_one recovers the original.
    let h_original: Vec<f32> = (0..cfg.hidden_size)
        .map(|i| ((i as f32) * 0.013).sin() * 0.25)
        .collect();
    let mut h_rotated = h_original.clone();
    rotation.apply(&mut h_rotated).unwrap();
    reference_state.session.last_pre_final_hidden = h_original;
    rotated_state.session.last_pre_final_hidden = h_rotated;

    let reference = reference_state.mtp_forward_one(42_u32, 0_usize);
    let rotated = rotated_state.mtp_forward_one(42_u32, 0_usize);

    assert_eq!(reference.logits.len(), cfg.vocab_size);
    assert_eq!(rotated.logits.len(), cfg.vocab_size);
    let max_abs_diff = reference
        .logits
        .iter()
        .zip(rotated.logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    // Tolerance is wider than the non-rotated golden-logit tests (1e-4) because
    // the Hadamard rotation spreads each embedding row across all 512 dimensions,
    // and the subsequent f16 roundtrip in embed_tokens + RMSNorm amplification
    // (~sqrt(hidden)) introduces ~0.02 per-logit error on the rotated path.
    // A missing or wrong counter-rotation would produce O(10) error, so 0.05
    // is tight enough to catch implementation defects.
    assert!(
        max_abs_diff < 0.05,
        "MTP draft logits diverged after QuaRot counter-rotation: max_abs_diff={max_abs_diff}"
    );
}

/// `mtp_pre_fc_rmsnorm` must apply the shifted `(1.0 + gamma)` convention,
/// not plain `gamma`. Hand-computed: x=[2,2,2,2], gamma=[0.5,0.5,0.5,0.5],
/// eps=0 -> sum_sq=16, mean=4, inv_rms=0.5. Shifted: 2*0.5*(1+0.5)=1.5.
/// Plain gamma would give 2*0.5*0.5=0.5 -- a clean, non-degenerate
/// (non-zero, non-one) numeric difference, not a sign-only tolerance.
#[test]
fn mtp_pre_fc_rmsnorm_applies_shifted_convention() {
    let mut x = vec![2.0f32; 4];
    let gamma = vec![0.5f32; 4];
    MetalQwen35State::mtp_pre_fc_rmsnorm(&mut x, &gamma, 0.0);
    for &v in &x {
        assert!(
            (v - 1.5).abs() < 1e-6,
            "got {v}, expected 1.5 under the shifted (1.0 + gamma) convention \
                     (2 * inv_rms(0.5) * (1.0 + 0.5)); plain gamma would give 0.5"
        );
    }
}

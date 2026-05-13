//! Test simdgroup_multiply_accumulate (MMA) for GEMM on Metal 4.
//! Apple GPUs support 8x8 simdgroup matrix operations since Apple7.

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
fn main() {
    eprintln!("need metal-gpu");
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn main() {
    use metal::*;
    use std::time::Instant;

    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    // SIMD-group GEMM using simdgroup_multiply_accumulate
    // Stages tiles through threadgroup memory for Metal 4 compatibility.
    let msl_simd = r#"
    #include <metal_stdlib>
    #include <metal_simdgroup_matrix>
    using namespace metal;

    kernel void gemm_simd(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C       [[buffer(2)]],
        constant uint& M      [[buffer(3)]],
        constant uint& N      [[buffer(4)]],
        constant uint& K      [[buffer(5)]],
        uint2 tgp             [[threadgroup_position_in_grid]],
        uint sgid             [[simdgroup_index_in_threadgroup]],
        uint lane             [[thread_index_in_simdgroup]],
        uint tid              [[thread_index_in_threadgroup]])
    {
        constexpr uint SG_DIM = 8;
        constexpr uint TG_DIM = 16;
        constexpr uint THREADS = 128; // 4 simdgroups

        // Threadgroup staging area for A and B tiles
        threadgroup float tg_a[TG_DIM][SG_DIM]; // 16x8
        threadgroup float tg_b[TG_DIM][SG_DIM]; // 16x8

        uint sg_row = sgid / 2;
        uint sg_col = sgid % 2;
        uint out_row = tgp.y * TG_DIM + sg_row * SG_DIM;
        uint out_col = tgp.x * TG_DIM + sg_col * SG_DIM;

        // Zero-init accumulator via threadgroup staging
        threadgroup float tg_zero[SG_DIM * SG_DIM];
        if (tid < 64) tg_zero[tid] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 acc;
        simdgroup_load(acc, tg_zero, SG_DIM);

        for (uint k0 = 0; k0 < K; k0 += SG_DIM) {
            // Cooperatively load A tile [TG_DIM × SG_DIM] and B tile [TG_DIM × SG_DIM]
            for (uint idx = tid; idx < TG_DIM * SG_DIM; idx += THREADS) {
                uint r = idx / SG_DIM;
                uint c = idx % SG_DIM;
                uint a_row = tgp.y * TG_DIM + r;
                uint a_col = k0 + c;
                tg_a[r][c] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;

                uint b_row = tgp.x * TG_DIM + r;
                uint b_col = k0 + c;
                tg_b[r][c] = (b_row < N && b_col < K) ? B[b_row * K + b_col] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 a_tile;
            simdgroup_float8x8 b_tile;
            simdgroup_load(a_tile, &tg_a[sg_row * SG_DIM][0], SG_DIM);
            simdgroup_load(b_tile, &tg_b[sg_col * SG_DIM][0], SG_DIM, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Store via threadgroup
        threadgroup float tg_out[SG_DIM][SG_DIM];
        simdgroup_store(acc, &tg_out[0][0], SG_DIM);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Write to global memory
        for (uint idx = lane; idx < SG_DIM * SG_DIM; idx += 32) {
            uint r = idx / SG_DIM;
            uint c = idx % SG_DIM;
            uint gr = out_row + r;
            uint gc = out_col + c;
            if (gr < M && gc < N) {
                C[gr * N + gc] = tg_out[r][c];
            }
        }
    }
    "#;

    let opts = CompileOptions::new();
    match device.new_library_with_source(msl_simd, &opts) {
        Ok(lib) => {
            let func = lib.get_function("gemm_simd", None).unwrap();
            let pipe = device
                .new_compute_pipeline_state_with_function(&func)
                .unwrap();

            let m: u32 = 5;
            let n: u32 = 2048;
            let k: u32 = 1024;
            let a = device.new_buffer(
                (m as u64 * k as u64 * 4),
                MTLResourceOptions::StorageModeShared,
            );
            let b = device.new_buffer(
                (n as u64 * k as u64 * 4),
                MTLResourceOptions::StorageModeShared,
            );
            let c = device.new_buffer(
                (m as u64 * n as u64 * 4),
                MTLResourceOptions::StorageModeShared,
            );

            // Fill
            unsafe {
                let av = std::slice::from_raw_parts_mut(a.contents() as *mut f32, (m * k) as usize);
                for (i, v) in av.iter_mut().enumerate() {
                    *v = (i as f32 * 0.001).sin();
                }
                let bv = std::slice::from_raw_parts_mut(b.contents() as *mut f32, (n * k) as usize);
                for (i, v) in bv.iter_mut().enumerate() {
                    *v = (i as f32 * 0.0001).cos();
                }
            }

            // Warmup
            for _ in 0..5 {
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipe);
                enc.set_buffer(0, Some(&a), 0);
                enc.set_buffer(1, Some(&b), 0);
                enc.set_buffer(2, Some(&c), 0);
                enc.set_bytes(3, 4, &m as *const u32 as *const _);
                enc.set_bytes(4, 4, &n as *const u32 as *const _);
                enc.set_bytes(5, 4, &k as *const u32 as *const _);
                // 4 simdgroups per TG (128 threads), grid covers M/16 × N/16
                let tg_m = 16u64;
                let tg_n = 16u64;
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        (n as u64 + tg_n - 1) / tg_n,
                        (m as u64 + tg_m - 1) / tg_m,
                        1,
                    ),
                    MTLSize::new(128, 1, 1), // 4 simdgroups × 32 threads
                );
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }

            // Benchmark batched
            let rounds = 20;
            let batch = 28;
            let t = Instant::now();
            for _ in 0..rounds {
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for _ in 0..batch {
                    enc.set_compute_pipeline_state(&pipe);
                    enc.set_buffer(0, Some(&a), 0);
                    enc.set_buffer(1, Some(&b), 0);
                    enc.set_buffer(2, Some(&c), 0);
                    enc.set_bytes(3, 4, &m as *const u32 as *const _);
                    enc.set_bytes(4, 4, &n as *const u32 as *const _);
                    enc.set_bytes(5, 4, &k as *const u32 as *const _);
                    let tg_m = 16u64;
                    let tg_n = 16u64;
                    enc.dispatch_thread_groups(
                        MTLSize::new(
                            (n as u64 + tg_n - 1) / tg_n,
                            (m as u64 + tg_m - 1) / tg_m,
                            1,
                        ),
                        MTLSize::new(128, 1, 1),
                    );
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
            let total = t.elapsed().as_micros() as f64 / rounds as f64;
            let per = total / batch as f64;
            eprintln!(
                "simdgroup GEMM(5,2048,1024): {per:.0}us/dispatch ({total:.0}us for {batch})"
            );

            // Correctness check: compute one element on CPU
            unsafe {
                let av = std::slice::from_raw_parts(a.contents() as *const f32, (m * k) as usize);
                let bv = std::slice::from_raw_parts(b.contents() as *const f32, (n * k) as usize);
                let cv = std::slice::from_raw_parts(c.contents() as *const f32, (m * n) as usize);
                // C[0,0] = sum_k A[0,k] * B[0,k]
                let mut expected = 0.0f64;
                for i in 0..k as usize {
                    expected += av[i] as f64 * bv[i] as f64;
                }
                let got = cv[0] as f64;
                eprintln!(
                    "C[0,0] expected={expected:.6} got={got:.6} err={:.2e}",
                    (expected - got).abs()
                );
            }
        }
        Err(e) => {
            eprintln!("simdgroup GEMM compilation failed: {e}");
        }
    }

    // Also benchmark the basic matmul_bt for comparison
    let msl_basic = r#"
    #include <metal_stdlib>
    using namespace metal;
    kernel void matmul_bt(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C       [[buffer(2)]],
        constant uint& M      [[buffer(3)]],
        constant uint& N      [[buffer(4)]],
        constant uint& K      [[buffer(5)]],
        uint2 gid  [[thread_position_in_grid]],
        uint2 lid  [[thread_position_in_threadgroup]])
    {
        constexpr uint TILE = 16;
        threadgroup float tA[TILE][TILE];
        threadgroup float tB[TILE][TILE];
        uint row = gid.y;
        uint col = gid.x;
        float sum = 0.0f;
        for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
            uint ak = t * TILE + lid.x;
            uint bk = t * TILE + lid.y;
            tA[lid.y][lid.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
            tB[lid.y][lid.x] = (col < N && bk < K) ? B[col * K + bk] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < TILE; k++) {
                sum += tA[lid.y][k] * tB[k][lid.x];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row < M && col < N) {
            C[row * N + col] = sum;
        }
    }
    "#;
    let lib2 = device.new_library_with_source(msl_basic, &opts).unwrap();
    let pipe2 = device
        .new_compute_pipeline_state_with_function(&lib2.get_function("matmul_bt", None).unwrap())
        .unwrap();

    let m: u32 = 5;
    let n: u32 = 2048;
    let k: u32 = 1024;
    let a = device.new_buffer(
        (m as u64 * k as u64 * 4),
        MTLResourceOptions::StorageModeShared,
    );
    let b = device.new_buffer(
        (n as u64 * k as u64 * 4),
        MTLResourceOptions::StorageModeShared,
    );
    let c = device.new_buffer(
        (m as u64 * n as u64 * 4),
        MTLResourceOptions::StorageModeShared,
    );
    let rounds = 20;
    let batch = 28;
    let t = Instant::now();
    for _ in 0..rounds {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..batch {
            enc.set_compute_pipeline_state(&pipe2);
            enc.set_buffer(0, Some(&a), 0);
            enc.set_buffer(1, Some(&b), 0);
            enc.set_buffer(2, Some(&c), 0);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(((n as u64) + 15) / 16, ((m as u64) + 15) / 16, 1),
                MTLSize::new(16, 16, 1),
            );
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let total = t.elapsed().as_micros() as f64 / rounds as f64;
    let per = total / batch as f64;
    eprintln!("matmul_bt(5,2048,1024):      {per:.0}us/dispatch ({total:.0}us for {batch})");
}

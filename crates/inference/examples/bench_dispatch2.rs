//! Test if wait_until_completed has a minimum latency floor on Metal 4.
//! Also test real GEMM dispatch overhead.

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

    // Real GEMM kernel (same as our matmul_bt)
    let src = r#"
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

    let opts = CompileOptions::new();
    let lib = device.new_library_with_source(src, &opts).unwrap();
    let func = lib.get_function("matmul_bt", None).unwrap();
    let pipe = device
        .new_compute_pipeline_state_with_function(&func)
        .unwrap();

    // Create buffers for (5, 2048) = A[5,1024] @ B[2048,1024]^T
    let m: u32 = 5;
    let n: u32 = 2048;
    let k: u32 = 1024;
    let a_buf = device.new_buffer(
        (m as u64 * k as u64 * 4),
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = device.new_buffer(
        (n as u64 * k as u64 * 4),
        MTLResourceOptions::StorageModeShared,
    );
    let c_buf = device.new_buffer(
        (m as u64 * n as u64 * 4),
        MTLResourceOptions::StorageModeShared,
    );

    // Fill with random-ish data
    unsafe {
        let a = std::slice::from_raw_parts_mut(a_buf.contents() as *mut f32, (m * k) as usize);
        for (i, v) in a.iter_mut().enumerate() {
            *v = (i as f32 * 0.001).sin();
        }
        let b = std::slice::from_raw_parts_mut(b_buf.contents() as *mut f32, (n * k) as usize);
        for (i, v) in b.iter_mut().enumerate() {
            *v = (i as f32 * 0.002).cos();
        }
    }

    // Warmup
    for _ in 0..5 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&c_buf), 0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &n as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_threads(MTLSize::new(n as u64, m as u64, 1), MTLSize::new(16, 16, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Test 1: Single GEMM per command buffer
    let rounds = 50;
    let t = Instant::now();
    for _ in 0..rounds {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&c_buf), 0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &n as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_threads(MTLSize::new(n as u64, m as u64, 1), MTLSize::new(16, 16, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let us = t.elapsed().as_micros() as f64 / rounds as f64;
    eprintln!("1 GEMM(5,2048,1024) / cmd buffer: {us:.0}us");

    // Test 2: 196 GEMMs (7 per layer × 28 layers) in one cmd buffer
    let t = Instant::now();
    for _ in 0..10 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..196 {
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&a_buf), 0);
            enc.set_buffer(1, Some(&b_buf), 0);
            enc.set_buffer(2, Some(&c_buf), 0);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            enc.dispatch_threads(MTLSize::new(n as u64, m as u64, 1), MTLSize::new(16, 16, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let us = t.elapsed().as_micros() as f64 / 10.0;
    eprintln!(
        "196 GEMMs(5,2048,1024) / 1 cmd: {us:.0}us total, {:.0}us/gemm",
        us / 196.0
    );

    // Test 3: Empty command buffer (baseline sync overhead)
    let t = Instant::now();
    for _ in 0..200 {
        let cmd = queue.new_command_buffer();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let empty_us = t.elapsed().as_micros() as f64 / 200.0;
    eprintln!("Empty cmd buffer roundtrip: {empty_us:.0}us");
}

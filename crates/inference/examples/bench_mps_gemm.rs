//! Benchmark MPSMatrixMultiplication (Apple's optimized GEMM) vs hand-written matmul_bt.
//! Uses raw objc messaging to call MPS APIs since metal 0.33 doesn't expose them.

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
fn main() {
    eprintln!("need metal-gpu");
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn main() {
    use metal::*;
    use objc::runtime::{Class, Object};
    use objc::{msg_send, sel, sel_impl};
    use std::time::Instant;

    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let device_ptr = &*device as *const DeviceRef as *const Object as *mut Object;

    let m: u64 = 5;
    let n: u64 = 2048;
    let k: u64 = 1024;

    // Create Metal buffers for A[M,K], B[K,N] (MPS uses column-major or we transpose), C[M,N]
    // MPS expects: C = alpha * A @ B + beta * C
    // A: [M x K], B: [K x N], C: [M x N]
    // But our B is stored as [N x K] (row-major, each row is one output dim).
    // MPS can handle transposed B: C = A @ B^T where B is [N x K]

    let a_buf = device.new_buffer(m * k * 4, MTLResourceOptions::StorageModeShared);
    let b_buf = device.new_buffer(n * k * 4, MTLResourceOptions::StorageModeShared);
    let c_buf = device.new_buffer(m * n * 4, MTLResourceOptions::StorageModeShared);

    // Fill with data
    unsafe {
        let a = std::slice::from_raw_parts_mut(a_buf.contents() as *mut f32, (m * k) as usize);
        for (i, v) in a.iter_mut().enumerate() {
            *v = (i as f32 * 0.001).sin();
        }
        let b = std::slice::from_raw_parts_mut(b_buf.contents() as *mut f32, (n * k) as usize);
        for (i, v) in b.iter_mut().enumerate() {
            *v = (i as f32 * 0.0001).cos();
        }
    }

    unsafe {
        // Create MPSMatrixDescriptors
        let mps_desc_class =
            Class::get("MPSMatrixDescriptor").expect("MPSMatrixDescriptor not found");

        // A descriptor: M rows × K columns, row-major
        let a_desc: *mut Object = msg_send![mps_desc_class,
            matrixDescriptorWithRows:m
            columns:k
            rowBytes:(k * 4)
            dataType:0x10000020u32]; // MPSDataTypeFloat32 = FloatBit | 32

        // B descriptor: N rows × K columns, row-major (will transpose in multiply)
        let b_desc: *mut Object = msg_send![mps_desc_class,
            matrixDescriptorWithRows:n
            columns:k
            rowBytes:(k * 4)
            dataType:0x10000020u32];

        // C descriptor: M rows × N columns
        let c_desc: *mut Object = msg_send![mps_desc_class,
            matrixDescriptorWithRows:m
            columns:n
            rowBytes:(n * 4)
            dataType:0x10000020u32];

        // Create MPSMatrix objects
        let mps_matrix_class = Class::get("MPSMatrix").expect("MPSMatrix not found");
        let a_ptr = &*a_buf as *const BufferRef as *const Object as *mut Object;
        let b_ptr = &*b_buf as *const BufferRef as *const Object as *mut Object;
        let c_ptr = &*c_buf as *const BufferRef as *const Object as *mut Object;

        let a_mat: *mut Object = msg_send![mps_matrix_class, alloc];
        let a_mat: *mut Object = msg_send![a_mat, initWithBuffer:a_ptr descriptor:a_desc];

        let b_mat: *mut Object = msg_send![mps_matrix_class, alloc];
        let b_mat: *mut Object = msg_send![b_mat, initWithBuffer:b_ptr descriptor:b_desc];

        let c_mat: *mut Object = msg_send![mps_matrix_class, alloc];
        let c_mat: *mut Object = msg_send![c_mat, initWithBuffer:c_ptr descriptor:c_desc];

        // Create MPSMatrixMultiplication: C = alpha * A @ B^T + beta * C
        let mps_mm_class =
            Class::get("MPSMatrixMultiplication").expect("MPSMatrixMultiplication not found");
        let mm: *mut Object = msg_send![mps_mm_class, alloc];
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        let mm: *mut Object = msg_send![mm,
            initWithDevice:device_ptr
            transposeLeft:false
            transposeRight:true
            resultRows:m as usize
            resultColumns:n as usize
            interiorColumns:k as usize
            alpha:alpha
            beta:beta];

        // Warmup
        for _ in 0..5 {
            let cmd = queue.new_command_buffer();
            let cmd_ptr = &*cmd as *const CommandBufferRef as *const Object as *mut Object;
            let _: () = msg_send![mm, encodeToCommandBuffer:cmd_ptr
                leftMatrix:a_mat rightMatrix:b_mat resultMatrix:c_mat];
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Benchmark: single dispatch
        let rounds = 50;
        let t = Instant::now();
        for _ in 0..rounds {
            let cmd = queue.new_command_buffer();
            let cmd_ptr = &*cmd as *const CommandBufferRef as *const Object as *mut Object;
            let _: () = msg_send![mm, encodeToCommandBuffer:cmd_ptr
                leftMatrix:a_mat rightMatrix:b_mat resultMatrix:c_mat];
            cmd.commit();
            cmd.wait_until_completed();
        }
        let single_us = t.elapsed().as_micros() as f64 / rounds as f64;
        eprintln!("MPS GEMM(5,2048,1024) single: {single_us:.0}us");

        // Benchmark: batched (28 encodes in 1 cmd buffer)
        let batch = 28;
        let t = Instant::now();
        for _ in 0..20 {
            let cmd = queue.new_command_buffer();
            let cmd_ptr = &*cmd as *const CommandBufferRef as *const Object as *mut Object;
            for _ in 0..batch {
                let _: () = msg_send![mm, encodeToCommandBuffer:cmd_ptr
                    leftMatrix:a_mat rightMatrix:b_mat resultMatrix:c_mat];
            }
            cmd.commit();
            cmd.wait_until_completed();
        }
        let total = t.elapsed().as_micros() as f64 / 20.0;
        let per = total / batch as f64;
        eprintln!("MPS GEMM(5,2048,1024) batched: {per:.0}us/dispatch ({total:.0}us for {batch})");

        // Correctness check
        {
            let a = std::slice::from_raw_parts(a_buf.contents() as *const f32, (m * k) as usize);
            let b = std::slice::from_raw_parts(b_buf.contents() as *const f32, (n * k) as usize);
            let c = std::slice::from_raw_parts(c_buf.contents() as *const f32, (m * n) as usize);
            let mut expected = 0.0f64;
            for i in 0..k as usize {
                expected += a[i] as f64 * b[i] as f64; // C[0,0] = A[0,:] @ B[0,:]
            }
            let got = c[0] as f64;
            let err = (expected - got).abs();
            eprintln!("C[0,0] expected={expected:.4} got={got:.4} err={err:.2e}");
        }
    }

    // matmul_bt comparison
    let msl = r#"
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
        uint row = gid.y; uint col = gid.x; float sum = 0.0f;
        for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
            uint ak = t * TILE + lid.x; uint bk = t * TILE + lid.y;
            tA[lid.y][lid.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
            tB[lid.y][lid.x] = (col < N && bk < K) ? B[col * K + bk] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < TILE; k++) sum += tA[lid.y][k] * tB[k][lid.x];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row < M && col < N) C[row * N + col] = sum;
    }
    "#;
    let opts = CompileOptions::new();
    let lib = device.new_library_with_source(msl, &opts).unwrap();
    let pipe = device
        .new_compute_pipeline_state_with_function(&lib.get_function("matmul_bt", None).unwrap())
        .unwrap();

    let m32 = m as u32;
    let n32 = n as u32;
    let k32 = k as u32;
    let t = Instant::now();
    for _ in 0..20 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..28 {
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&a_buf), 0);
            enc.set_buffer(1, Some(&b_buf), 0);
            enc.set_buffer(2, Some(&c_buf), 0);
            enc.set_bytes(3, 4, &m32 as *const u32 as *const _);
            enc.set_bytes(4, 4, &n32 as *const u32 as *const _);
            enc.set_bytes(5, 4, &k32 as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((n + 15) / 16, (m + 15) / 16, 1),
                MTLSize::new(16, 16, 1),
            );
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let total = t.elapsed().as_micros() as f64 / 20.0;
    eprintln!(
        "matmul_bt(5,2048,1024) batched: {:.0}us/dispatch ({total:.0}us for 28)",
        total / 28.0
    );
}

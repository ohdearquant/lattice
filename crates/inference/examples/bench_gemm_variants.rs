//! Compare GEMM implementations on Metal 4:
//! 1. Basic matmul_bt (16x16 tiled)
//! 2. Rect rega1x8_r8
//! 3. Rect rega2x4_r16
//! 4. Multiple independent GEMMs to test overlap/parallelism
//! 5. Single encoder vs multiple encoders

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
fn main() {
    eprintln!("need metal-gpu");
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn main() {
    use metal::*;
    use std::time::Instant;

    let device = Device::system_default().unwrap();
    eprintln!("Device: {} | Metal Family: checking...", device.name());

    // Check Metal feature sets
    for family in [
        MTLGPUFamily::Apple7,
        MTLGPUFamily::Apple8,
        MTLGPUFamily::Apple9,
    ] {
        if device.supports_family(family) {
            eprintln!("  Supports: {:?}", family);
        }
    }

    let queue = device.new_command_queue();

    let msl = include_str!("../metal_forward.rs");
    let start = msl.find("r#\"\n").unwrap() + 4;
    let end = msl[start..].find("\"#;").unwrap() + start;
    let msl_src = &msl[start..end];
    let opts = CompileOptions::new();
    let lib = device.new_library_with_source(msl_src, &opts).unwrap();
    let make_pipe = |name: &str| {
        let f = lib.get_function(name, None).unwrap();
        device.new_compute_pipeline_state_with_function(&f).unwrap()
    };

    let matmul_pipe = make_pipe("matmul_bt");
    let gemm_r8 = make_pipe("gemm_rega1x8_r8");
    let gemm_r16 = make_pipe("gemm_rega2x4_r16");

    // Allocate separate buffers for parallel GEMM test
    let make_buf = |sz: u64| device.new_buffer(sz, MTLResourceOptions::StorageModeShared);
    let a_buf = make_buf(5 * 1024 * 4);
    let wq_buf = make_buf(2048 * 1024 * 4);
    let wk_buf = make_buf(1024 * 1024 * 4);
    let wv_buf = make_buf(1024 * 1024 * 4);
    let q_buf = make_buf(5 * 2048 * 4);
    let k_buf = make_buf(5 * 1024 * 4);
    let v_buf = make_buf(5 * 1024 * 4);

    // Fill A with data
    unsafe {
        let a = std::slice::from_raw_parts_mut(a_buf.contents() as *mut f32, 5 * 1024);
        for (i, v) in a.iter_mut().enumerate() {
            *v = (i as f32 * 0.001).sin();
        }
        let w = std::slice::from_raw_parts_mut(wq_buf.contents() as *mut f32, 2048 * 1024);
        for (i, v) in w.iter_mut().enumerate() {
            *v = (i as f32 * 0.0001).cos();
        }
    }

    let m: u32 = 5;
    let n_q: u32 = 2048;
    let n_kv: u32 = 1024;
    let k: u32 = 1024;
    let rounds = 20;

    let dispatch_matmul_bt =
        |enc: &ComputeCommandEncoderRef, w: &BufferRef, out: &BufferRef, n: u32| {
            enc.set_compute_pipeline_state(&matmul_pipe);
            enc.set_buffer(0, Some(&a_buf), 0);
            enc.set_buffer(1, Some(w), 0);
            enc.set_buffer(2, Some(out), 0);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            enc.dispatch_threads(MTLSize::new(n as u64, m as u64, 1), MTLSize::new(16, 16, 1));
        };

    let dispatch_r8 = |enc: &ComputeCommandEncoderRef, w: &BufferRef, out: &BufferRef, n: u32| {
        enc.set_compute_pipeline_state(&gemm_r8);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &n as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize::new((n / 8) as u64, m as u64, 1),
            MTLSize::new(8, 8, 1),
        );
    };

    // Warmup
    for _ in 0..5 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        dispatch_r8(&enc, &wq_buf, &q_buf, n_q);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Test 1: Basic matmul_bt (5,2048,1024)
    let t = Instant::now();
    for _ in 0..rounds {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..28 {
            dispatch_matmul_bt(&enc, &wq_buf, &q_buf, n_q);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let bt_us = t.elapsed().as_micros() as f64 / rounds as f64 / 28.0;
    eprintln!("\n=== GEMM Variant Comparison (M=5, K=1024) ===");
    eprintln!("matmul_bt(5,2048,1024):  {bt_us:.0}us/dispatch");

    // Test 2: Rect r8 (5,2048,1024)
    let t = Instant::now();
    for _ in 0..rounds {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..28 {
            dispatch_r8(&enc, &wq_buf, &q_buf, n_q);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let r8_us = t.elapsed().as_micros() as f64 / rounds as f64 / 28.0;
    eprintln!("gemm_r8(5,2048,1024):    {r8_us:.0}us/dispatch");

    // Test 3: Q+K+V sequential (3 separate dispatches)
    let t = Instant::now();
    for _ in 0..rounds {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..28 {
            dispatch_r8(&enc, &wq_buf, &q_buf, n_q);
            dispatch_r8(&enc, &wk_buf, &k_buf, n_kv);
            dispatch_r8(&enc, &wv_buf, &v_buf, n_kv);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let qkv_total = t.elapsed().as_micros() as f64 / rounds as f64;
    let qkv_per = qkv_total / (28.0 * 3.0);
    eprintln!("\n=== Parallelism Test ===");
    eprintln!("Q+K+V sequential (28 layers): {qkv_total:.0}us total, {qkv_per:.0}us/gemm");

    // Test 4: All 7 GEMMs per layer, single encoder
    let wo_buf = make_buf(1024 * 2048 * 4);
    let wg_buf = make_buf(3072 * 1024 * 4);
    let wu_buf = make_buf(3072 * 1024 * 4);
    let wd_buf = make_buf(1024 * 3072 * 4);
    let gate_buf = make_buf(5 * 3072 * 4);
    let up_buf = make_buf(5 * 3072 * 4);
    let ffn_buf = make_buf(5 * 1024 * 4);
    let n_inter: u32 = 3072;

    let t = Instant::now();
    for _ in 0..rounds {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..28 {
            dispatch_r8(&enc, &wq_buf, &q_buf, n_q); // Q
            dispatch_r8(&enc, &wk_buf, &k_buf, n_kv); // K
            dispatch_r8(&enc, &wv_buf, &v_buf, n_kv); // V
            dispatch_r8(&enc, &wo_buf, &a_buf, n_kv); // O (approx)
            dispatch_r8(&enc, &wg_buf, &gate_buf, n_inter); // Gate
            dispatch_r8(&enc, &wu_buf, &up_buf, n_inter); // Up
            dispatch_r8(&enc, &wd_buf, &ffn_buf, n_kv); // Down
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let all7_total = t.elapsed().as_micros() as f64 / rounds as f64;
    let all7_per = all7_total / (28.0 * 7.0);
    eprintln!("7 GEMMs/layer (28 layers): {all7_total:.0}us total, {all7_per:.0}us/gemm");
    eprintln!("  Expected if serial: {:.0}us", 28.0 * 7.0 * r8_us);
    eprintln!(
        "  Actual / Serial: {:.2}x",
        all7_total / (28.0 * 7.0 * r8_us)
    );

    // Test 5: 28 separate encoders (like real forward) vs 1 encoder
    let t = Instant::now();
    for _ in 0..rounds {
        let cmd = queue.new_command_buffer();
        for _ in 0..28 {
            let enc = cmd.new_compute_command_encoder();
            dispatch_r8(&enc, &wq_buf, &q_buf, n_q);
            dispatch_r8(&enc, &wk_buf, &k_buf, n_kv);
            dispatch_r8(&enc, &wv_buf, &v_buf, n_kv);
            dispatch_r8(&enc, &wo_buf, &a_buf, n_kv);
            dispatch_r8(&enc, &wg_buf, &gate_buf, n_inter);
            dispatch_r8(&enc, &wu_buf, &up_buf, n_inter);
            dispatch_r8(&enc, &wd_buf, &ffn_buf, n_kv);
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
    }
    let enc28_total = t.elapsed().as_micros() as f64 / rounds as f64;
    eprintln!("\n=== Encoder Overhead ===");
    eprintln!("28 encoders × 7 GEMMs: {enc28_total:.0}us");
    eprintln!("1 encoder × 196 GEMMs: {all7_total:.0}us");
    eprintln!("Encoder overhead: {:.0}us total", enc28_total - all7_total);
}

//! Micro-benchmark: Metal dispatch overhead on Metal 4.
//! Tests noop dispatches to isolate per-dispatch and per-encoder overhead.

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
fn main() {
    eprintln!("Requires macOS + metal-gpu feature");
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn main() {
    use metal::*;
    use std::time::Instant;

    let device = Device::system_default().expect("no Metal device");
    eprintln!("Device: {}", device.name());
    let queue = device.new_command_queue();

    let src = r#"
    #include <metal_stdlib>
    using namespace metal;
    kernel void noop(device float* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
        if (gid == 0) out[0] = 1.0f;
    }
    "#;

    let opts = CompileOptions::new();
    let lib = device.new_library_with_source(src, &opts).unwrap();
    let func = lib.get_function("noop", None).unwrap();
    let pipe = device
        .new_compute_pipeline_state_with_function(&func)
        .unwrap();
    let buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    // Warmup
    for _ in 0..10 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&buf), 0);
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Test 1: Single dispatch per command buffer
    let n = 200;
    let t = Instant::now();
    for _ in 0..n {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&buf), 0);
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let single_us = t.elapsed().as_micros() as f64 / n as f64;
    eprintln!("1 dispatch / 1 cmd buffer: {single_us:.0}us per call");

    // Test 2: Many dispatches, single encoder, single cmd buffer
    let dispatches = 500;
    let rounds = 20;
    let t = Instant::now();
    for _ in 0..rounds {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..dispatches {
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&buf), 0);
            enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let batch_us = t.elapsed().as_micros() as f64 / rounds as f64;
    eprintln!(
        "{dispatches} dispatches / 1 encoder: {batch_us:.0}us total, {:.1}us/dispatch",
        batch_us / dispatches as f64
    );

    // Test 3: 28 encoders × 17 dispatches (mimics transformer forward)
    let t = Instant::now();
    for _ in 0..rounds {
        let cmd = queue.new_command_buffer();
        for _ in 0..28 {
            let enc = cmd.new_compute_command_encoder();
            for _ in 0..17 {
                enc.set_compute_pipeline_state(&pipe);
                enc.set_buffer(0, Some(&buf), 0);
                enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            }
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
    }
    let fwd_us = t.elapsed().as_micros() as f64 / rounds as f64;
    eprintln!(
        "28 encoders × 17 dispatches: {fwd_us:.0}us total, {:.1}us/dispatch",
        fwd_us / (28.0 * 17.0)
    );

    // Test 4: Single encoder for everything (no per-layer encoder)
    let t = Instant::now();
    for _ in 0..rounds {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..(28 * 17) {
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&buf), 0);
            enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let flat_us = t.elapsed().as_micros() as f64 / rounds as f64;
    eprintln!(
        "1 encoder × {} dispatches: {flat_us:.0}us total, {:.1}us/dispatch",
        28 * 17,
        flat_us / (28.0 * 17.0)
    );
}

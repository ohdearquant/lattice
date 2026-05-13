//! Micro-benchmark: individual kernel timing on Metal 4.
//! Isolates which kernel is the bottleneck.

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

    // Compile ALL kernels from the inference MSL source
    let msl = include_str!("../metal_forward.rs");
    // Extract the MSL source between r#" and "#;
    let start = msl.find("r#\"\n").unwrap() + 4;
    let end = msl[start..].find("\"#;").unwrap() + start;
    let msl_src = &msl[start..end];

    let opts = CompileOptions::new();
    let lib = device.new_library_with_source(msl_src, &opts).unwrap();

    let make_pipe = |name: &str| -> ComputePipelineState {
        let f = lib.get_function(name, None).unwrap();
        device.new_compute_pipeline_state_with_function(&f).unwrap()
    };

    let rms_pipe = make_pipe("rms_norm");
    let rope_pipe = make_pipe("rope");
    let silu_pipe = make_pipe("silu_mul");
    let copy_pipe = make_pipe("copy_buf");
    let add_pipe = make_pipe("add_buf");
    let attn_pipe = make_pipe("fused_attention");
    let gemm_pipe = make_pipe("gemm_rega1x8_r8");

    // Buffers for seq=5, hidden=1024, q_dim=2048, kv_dim=1024, inter=3072
    let hidden_buf = device.new_buffer(5 * 1024 * 4, MTLResourceOptions::StorageModeShared);
    let gamma_buf = device.new_buffer(1024 * 4, MTLResourceOptions::StorageModeShared);
    let q_buf = device.new_buffer(5 * 2048 * 4, MTLResourceOptions::StorageModeShared);
    let k_buf = device.new_buffer(5 * 1024 * 4, MTLResourceOptions::StorageModeShared);
    let v_buf = device.new_buffer(5 * 1024 * 4, MTLResourceOptions::StorageModeShared);
    let o_buf = device.new_buffer(5 * 2048 * 4, MTLResourceOptions::StorageModeShared);
    let gate_buf = device.new_buffer(5 * 3072 * 4, MTLResourceOptions::StorageModeShared);
    let cos_buf = device.new_buffer(5 * 64 * 4, MTLResourceOptions::StorageModeShared);
    let sin_buf = device.new_buffer(5 * 64 * 4, MTLResourceOptions::StorageModeShared);
    let weight_buf = device.new_buffer(2048 * 1024 * 4, MTLResourceOptions::StorageModeShared);

    let time_kernel = |name: &str, f: &dyn Fn(&CommandBufferRef)| -> f64 {
        // Warmup
        for _ in 0..5 {
            let cmd = queue.new_command_buffer();
            f(cmd);
            cmd.commit();
            cmd.wait_until_completed();
        }
        let rounds = 50;
        let t = Instant::now();
        for _ in 0..rounds {
            let cmd = queue.new_command_buffer();
            f(cmd);
            cmd.commit();
            cmd.wait_until_completed();
        }
        let us = t.elapsed().as_micros() as f64 / rounds as f64;
        eprintln!("{name:>20}: {us:>8.0}us");
        us
    };

    let seq: u32 = 5;
    let hidden: u32 = 1024;
    let q_dim: u32 = 2048;
    let kv_dim: u32 = 1024;
    let inter: u32 = 3072;
    let head_dim: u32 = 128;
    let num_heads: u32 = 16;
    let num_kv: u32 = 8;
    let eps: f32 = 1e-6;

    // RMS norm
    time_kernel("rms_norm(5,1024)", &|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&rms_pipe);
        enc.set_buffer(0, Some(&hidden_buf), 0);
        enc.set_buffer(1, Some(&gamma_buf), 0);
        enc.set_bytes(2, 4, &hidden as *const u32 as *const _);
        enc.set_bytes(3, 4, &seq as *const u32 as *const _);
        enc.set_bytes(4, 4, &eps as *const f32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(seq as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    });

    // RMS norm for Q heads: 5*16=80 rows of 128
    let q_rows: u32 = seq * num_heads;
    time_kernel("rms_norm(80,128)", &|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&rms_pipe);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&gamma_buf), 0);
        enc.set_bytes(2, 4, &head_dim as *const u32 as *const _);
        enc.set_bytes(3, 4, &q_rows as *const u32 as *const _);
        enc.set_bytes(4, 4, &eps as *const f32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(q_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    });

    // RoPE
    time_kernel("rope(5,16,128)", &|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&rope_pipe);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&cos_buf), 0);
        enc.set_buffer(2, Some(&sin_buf), 0);
        enc.set_bytes(3, 4, &seq as *const u32 as *const _);
        enc.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        enc.set_bytes(5, 4, &head_dim as *const u32 as *const _);
        enc.set_bytes(6, 4, &q_dim as *const u32 as *const _);
        let total = seq as u64 * num_heads as u64 * 64;
        enc.dispatch_threads(
            MTLSize::new(((total + 255) / 256) * 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    });

    // GEMM
    let m = seq;
    let n = q_dim;
    let k = hidden;
    time_kernel("gemm(5,2048,1024)", &|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&gemm_pipe);
        enc.set_buffer(0, Some(&hidden_buf), 0);
        enc.set_buffer(1, Some(&weight_buf), 0);
        enc.set_buffer(2, Some(&q_buf), 0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &n as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize::new((n / 8) as u64, m as u64, 1),
            MTLSize::new(8, 8, 1),
        );
        enc.end_encoding();
    });

    // silu_mul
    let count = seq * inter;
    time_kernel("silu_mul(15360)", &|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&silu_pipe);
        enc.set_buffer(0, Some(&gate_buf), 0);
        enc.set_buffer(1, Some(&gate_buf), 0);
        enc.set_bytes(2, 4, &count as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize::new(((count as u64 + 255) / 256) * 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    });

    // copy
    let hcount = seq * hidden;
    time_kernel("copy(5120)", &|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&copy_pipe);
        enc.set_buffer(0, Some(&hidden_buf), 0);
        enc.set_buffer(1, Some(&hidden_buf), 0);
        enc.set_bytes(2, 4, &hcount as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize::new(((hcount as u64 + 255) / 256) * 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    });

    // add
    time_kernel("add(5120)", &|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&add_pipe);
        enc.set_buffer(0, Some(&hidden_buf), 0);
        enc.set_buffer(1, Some(&hidden_buf), 0);
        enc.set_bytes(2, 4, &hcount as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize::new(((hcount as u64 + 255) / 256) * 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    });

    // fused attention
    let scale: f32 = 1.0 / (128.0f32).sqrt();
    #[repr(C)]
    struct FAParams {
        seq_len: u32,
        q_dim4: u32,
        kv_dim4: u32,
        num_kv_heads: u32,
        scale: f32,
        _p0: u32,
        _p1: u32,
        _p2: u32,
    }
    let fa_params = FAParams {
        seq_len: seq,
        q_dim4: q_dim / 4,
        kv_dim4: kv_dim / 4,
        num_kv_heads: num_kv,
        scale,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    time_kernel("fused_attn(5)", &|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&attn_pipe);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&k_buf), 0);
        enc.set_buffer(2, Some(&v_buf), 0);
        enc.set_buffer(3, Some(&o_buf), 0);
        enc.set_bytes(
            4,
            std::mem::size_of::<FAParams>() as u64,
            &fa_params as *const FAParams as *const _,
        );
        let q_blocks = (seq + 3) / 4;
        enc.dispatch_thread_groups(
            MTLSize::new(num_kv as u64, q_blocks as u64, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    });

    // Summary
    eprintln!("\nPer-layer estimate (SHORT, seq=5):");
    eprintln!("  7 GEMMs + 4 norms + 2 ropes + 1 attn + 1 silu + 2 copy + 2 add = ~17 ops");
}

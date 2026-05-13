//! Batched kernel timing — N dispatches of same kernel in 1 cmd buffer.

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

    let rms_pipe = make_pipe("rms_norm");
    let rope_pipe = make_pipe("rope");
    let silu_pipe = make_pipe("silu_mul");
    let copy_pipe = make_pipe("copy_buf");
    let add_pipe = make_pipe("add_buf");
    let attn_pipe = make_pipe("fused_attention");
    let gemm_pipe = make_pipe("gemm_rega1x8_r8");
    let gemm_bf16_pipe = make_pipe("gemm_rega1x8_r8_bf16");

    let hidden_buf = device.new_buffer(5 * 1024 * 4, MTLResourceOptions::StorageModeShared);
    let gamma_buf = device.new_buffer(1024 * 4, MTLResourceOptions::StorageModeShared);
    let q_buf = device.new_buffer(5 * 2048 * 4, MTLResourceOptions::StorageModeShared);
    let k_buf = device.new_buffer(5 * 1024 * 4, MTLResourceOptions::StorageModeShared);
    let v_buf = device.new_buffer(5 * 1024 * 4, MTLResourceOptions::StorageModeShared);
    let o_buf = device.new_buffer(5 * 2048 * 4, MTLResourceOptions::StorageModeShared);
    let gate_buf = device.new_buffer(5 * 3072 * 4, MTLResourceOptions::StorageModeShared);
    let cos_buf = device.new_buffer(512 * 64 * 4, MTLResourceOptions::StorageModeShared);
    let sin_buf = device.new_buffer(512 * 64 * 4, MTLResourceOptions::StorageModeShared);
    let weight_buf = device.new_buffer(2048 * 1024 * 4, MTLResourceOptions::StorageModeShared);
    // BF16 weight buffer (half size)
    let weight_bf16_buf = device.new_buffer(2048 * 1024 * 2, MTLResourceOptions::StorageModeShared);

    let n_batch = 100u32;
    let rounds = 10;

    let seq: u32 = 5;
    let hidden: u32 = 1024;
    let q_dim: u32 = 2048;
    let kv_dim: u32 = 1024;
    let inter: u32 = 3072;
    let head_dim: u32 = 128;
    let num_heads: u32 = 16;
    let num_kv: u32 = 8;
    let eps: f32 = 1e-6;

    let time_batched = |name: &str, dispatch: &dyn Fn(&ComputeCommandEncoderRef)| {
        // Warmup
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..n_batch {
            dispatch(&enc);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let t = Instant::now();
        for _ in 0..rounds {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for _ in 0..n_batch {
                dispatch(&enc);
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let total_us = t.elapsed().as_micros() as f64 / rounds as f64;
        let per_us = total_us / n_batch as f64;
        eprintln!("{name:>25}: {per_us:>6.1}us/dispatch ({total_us:.0}us for {n_batch})");
    };

    time_batched("rms_norm(5,1024)", &|enc| {
        enc.set_compute_pipeline_state(&rms_pipe);
        enc.set_buffer(0, Some(&hidden_buf), 0);
        enc.set_buffer(1, Some(&gamma_buf), 0);
        enc.set_bytes(2, 4, &hidden as *const u32 as *const _);
        enc.set_bytes(3, 4, &seq as *const u32 as *const _);
        enc.set_bytes(4, 4, &eps as *const f32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(seq as u64, 1, 1), MTLSize::new(256, 1, 1));
    });

    let q_rows: u32 = seq * num_heads;
    time_batched("rms_norm(80,128)", &|enc| {
        enc.set_compute_pipeline_state(&rms_pipe);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&gamma_buf), 0);
        enc.set_bytes(2, 4, &head_dim as *const u32 as *const _);
        enc.set_bytes(3, 4, &q_rows as *const u32 as *const _);
        enc.set_bytes(4, 4, &eps as *const f32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(q_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    });

    time_batched("rope(5,16,128)", &|enc| {
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
    });

    time_batched("gemm_r8(5,2048,1024)", &|enc| {
        enc.set_compute_pipeline_state(&gemm_pipe);
        enc.set_buffer(0, Some(&hidden_buf), 0);
        enc.set_buffer(1, Some(&weight_buf), 0);
        enc.set_buffer(2, Some(&q_buf), 0);
        enc.set_bytes(3, 4, &seq as *const u32 as *const _);
        enc.set_bytes(4, 4, &q_dim as *const u32 as *const _);
        enc.set_bytes(5, 4, &hidden as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize::new((q_dim / 8) as u64, seq as u64, 1),
            MTLSize::new(8, 8, 1),
        );
    });

    time_batched("silu_mul(15360)", &|enc| {
        let count = seq * inter;
        enc.set_compute_pipeline_state(&silu_pipe);
        enc.set_buffer(0, Some(&gate_buf), 0);
        enc.set_buffer(1, Some(&gate_buf), 0);
        enc.set_bytes(2, 4, &count as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize::new(((count as u64 + 255) / 256) * 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    });

    time_batched("copy(5120)", &|enc| {
        let count = seq * hidden;
        enc.set_compute_pipeline_state(&copy_pipe);
        enc.set_buffer(0, Some(&hidden_buf), 0);
        enc.set_buffer(1, Some(&hidden_buf), 0);
        enc.set_bytes(2, 4, &count as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize::new(((count as u64 + 255) / 256) * 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    });

    time_batched("add(5120)", &|enc| {
        let count = seq * hidden;
        enc.set_compute_pipeline_state(&add_pipe);
        enc.set_buffer(0, Some(&hidden_buf), 0);
        enc.set_buffer(1, Some(&hidden_buf), 0);
        enc.set_bytes(2, 4, &count as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize::new(((count as u64 + 255) / 256) * 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    });

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
    let scale: f32 = 1.0 / (128.0f32).sqrt();
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
    time_batched("fused_attn(5)", &|enc| {
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
    });

    // BF16 GEMM
    time_batched("gemm_r8_bf16(5,2048,1024)", &|enc| {
        enc.set_compute_pipeline_state(&gemm_bf16_pipe);
        enc.set_buffer(0, Some(&hidden_buf), 0);
        enc.set_buffer(1, Some(&weight_bf16_buf), 0);
        enc.set_buffer(2, Some(&q_buf), 0);
        enc.set_bytes(3, 4, &seq as *const u32 as *const _);
        enc.set_bytes(4, 4, &q_dim as *const u32 as *const _);
        enc.set_bytes(5, 4, &hidden as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize::new((q_dim / 8) as u64, seq as u64, 1),
            MTLSize::new(8, 8, 1),
        );
    });

    // Estimate full layer
    eprintln!("\nEstimate per layer: 7×GEMM + 4×norm + 2×rope + 1×attn + 1×silu + 2×copy + 2×add");
}

//! Quick benchmark: reference vs fused GatedDeltaNet.

use lattice_inference::attention::gdn::*;
use lattice_inference::attention::gdn_fused::*;
use lattice_inference::lora_hook::NoopLoraHook;
use lattice_inference::model::qwen35_config::Qwen35Config;
use std::time::Instant;

fn main() {
    let cfg = Qwen35Config::qwen35_2b();
    let hidden = cfg.hidden_size;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let num_heads = cfg.linear_num_key_heads;
    let kernel_size = cfg.linear_conv_kernel_dim;

    let mut seed = 42u64;
    let mut rand_f32 = || -> f32 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    };
    let make_vec =
        |n: usize, r: &mut dyn FnMut() -> f32| -> Vec<f32> { (0..n).map(|_| r() * 0.01).collect() };

    let weights = GatedDeltaNetWeights {
        in_proj_qkv: make_vec(qkv_dim * hidden, &mut rand_f32),
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z: make_vec(output_dim * hidden, &mut rand_f32),
        in_proj_z_rows: output_dim,
        in_proj_z_cols: hidden,
        in_proj_b: make_vec(num_heads * hidden, &mut rand_f32),
        in_proj_b_rows: num_heads,
        in_proj_b_cols: hidden,
        in_proj_a: make_vec(num_heads * hidden, &mut rand_f32),
        in_proj_a_rows: num_heads,
        in_proj_a_cols: hidden,
        a_log: make_vec(num_heads, &mut rand_f32),
        dt_bias: make_vec(num_heads, &mut rand_f32),
        conv1d_weight: make_vec(qkv_dim * kernel_size, &mut rand_f32),
        conv_dim: qkv_dim,
        kernel_size,
        norm_weight: make_vec(cfg.linear_value_head_dim, &mut rand_f32),
        out_proj: make_vec(hidden * output_dim, &mut rand_f32),
        out_proj_rows: hidden,
        out_proj_cols: output_dim,
    };

    let input: Vec<f32> = make_vec(hidden, &mut rand_f32);
    let iters = 200;

    // Reference
    let ref_us = {
        let mut state = GatedDeltaNetState::new(&cfg);
        let mut scratch = GatedDeltaNetScratch::default();
        let mut output = vec![0.0f32; hidden];
        for _ in 0..10 {
            gated_delta_net_step(
                &input,
                &mut state,
                &weights,
                &cfg,
                &mut scratch,
                &mut output,
            );
        }
        state.reset();
        let t0 = Instant::now();
        for _ in 0..iters {
            gated_delta_net_step(
                &input,
                &mut state,
                &weights,
                &cfg,
                &mut scratch,
                &mut output,
            );
        }
        t0.elapsed().as_micros()
    };

    // Fused
    let fused_us = {
        let mut state = GatedDeltaNetState::new(&cfg);
        let mut scratch = GatedDeltaNetFusedScratch::default();
        let mut output = vec![0.0f32; hidden];
        for _ in 0..10 {
            gated_delta_net_step_fused(
                &input,
                &mut state,
                &weights,
                &cfg,
                &mut scratch,
                &mut output,
                &NoopLoraHook,
                0,
            );
        }
        state.reset();
        let t0 = Instant::now();
        for _ in 0..iters {
            gated_delta_net_step_fused(
                &input,
                &mut state,
                &weights,
                &cfg,
                &mut scratch,
                &mut output,
                &NoopLoraHook,
                0,
            );
        }
        t0.elapsed().as_micros()
    };

    let ref_per = ref_us as f64 / iters as f64;
    let fused_per = fused_us as f64 / iters as f64;
    let speedup = ref_per / fused_per;

    println!("GatedDeltaNet step benchmark ({iters} iterations)");
    println!("  Reference: {ref_us}us total, {ref_per:.1}us/step");
    println!("  Fused:     {fused_us}us total, {fused_per:.1}us/step");
    println!("  Speedup:   {speedup:.2}x");
}

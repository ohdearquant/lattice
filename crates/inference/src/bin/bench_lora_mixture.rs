//! LoRA mixture blend latency benchmark.
//!
//! Measures the CPU cost of blending k LoRA adapters of rank r into one
//! rank-k*r adapter, for k∈{1,4,8} and r∈{1,2}.  This is the overhead added
//! per request when using the mixture path vs the single-adapter path.
//!
//! If `LATTICE_MODEL_DIR` points to a valid Qwen3.5-0.8b Q4 model, the bench
//! additionally loads the blended adapter onto GPU and measures decode tok/s,
//! giving a full end-to-end comparison. It also times the per-selection-change
//! swap cost — blend, GPU adapter unload, GPU adapter load — as three separate
//! phases: a coefficient-only fast path that skips the blend still pays the
//! unload+load cost, and a combined number cannot tell the two apart.
//!
//! # Output
//!
//! ```text
//! BLEND_BENCH r=1 k=1 layers=12 blend_us=<f>
//! BLEND_BENCH r=1 k=4 layers=12 blend_us=<f>
//! BLEND_BENCH r=1 k=8 layers=12 blend_us=<f>
//! BLEND_BENCH r=2 k=1 layers=12 blend_us=<f>
//! BLEND_BENCH r=2 k=4 layers=12 blend_us=<f>
//! BLEND_BENCH r=2 k=8 layers=12 blend_us=<f>
//! ```
//!
//! When a model is available:
//! ```text
//! SWAP_BENCH r=1 k=1 layers=<n> blend_us=<f> unload_us=<f> load_us=<f> iters=<n>
//! SWAP_BENCH r=1 k=4 layers=<n> blend_us=<f> unload_us=<f> load_us=<f> iters=<n>
//! ...
//! DECODE_BENCH r=1 k=1 tok_s=<f>
//! DECODE_BENCH r=1 k=4 tok_s=<f>
//! ...
//! ```
//!
//! `layers` means the same thing in both lines -- the number of blended
//! `LoraLayerData` entries -- but the two `blend_us` figures are NOT comparable
//! at equal `r` and `k`: the CPU section builds `q_proj` and `v_proj` per layer
//! while the GPU section builds `q_proj` only, so it blends half as many entries
//! by construction. Read `layers` before comparing any two blend figures.
//!
//! # Env vars
//!
//!   LATTICE_MODEL_DIR   Q4 model dir (optional; enables GPU decode bench)
//!   BENCH_WARMUP        warmup iterations for blend bench (default 5)
//!   BENCH_ITERS         measured iterations (default 20)
//!   BENCH_NEW_TOKENS    tokens to generate in decode bench (default 32)
#![allow(clippy::field_reassign_with_default)]

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("bench_lora_mixture requires macOS + --features metal-gpu,f16.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        if let Err(e) = run() {
            eprintln!("bench_lora_mixture failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
fn full_attention_layer_indices(
    cfg: &lattice_inference::model::qwen35_config::Qwen35Config,
) -> Vec<usize> {
    use lattice_inference::model::qwen35_config::LayerType;

    cfg.layer_types
        .iter()
        .enumerate()
        .filter_map(|(layer_idx, layer_type)| {
            (*layer_type == LayerType::FullAttention).then_some(layer_idx)
        })
        .collect()
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::forward::metal_qwen35::{LoraLayerData, blend_lora_layer_data};
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use std::time::Instant;

    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    if iters == 0 {
        // Every per-iteration figure below divides by `iters`. At zero that is a
        // NaN printed in a field a reader parses as microseconds, so refuse here
        // rather than emit an unreadable measurement.
        return Err("BENCH_ITERS must be at least 1".into());
    }
    let new_tokens: usize = std::env::var("BENCH_NEW_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let cfg = Qwen35Config::qwen35_0_8b();
    let layer_indices = full_attention_layer_indices(&cfg);
    const D_IN: usize = 1024;
    const D_OUT: usize = 4096;

    for &rank in &[1usize, 2] {
        for &k in &[1usize, 4, 8] {
            // GDN layers have no q_proj or v_proj and are rejected by the runtime.
            let adapters: Vec<Vec<LoraLayerData>> = (0..k)
                .map(|adapter_idx| {
                    layer_indices
                        .iter()
                        .copied()
                        .flat_map(|layer_idx| {
                            let seed = (adapter_idx * cfg.num_hidden_layers + layer_idx) as f32;
                            // q_proj
                            let layer_q = LoraLayerData {
                                layer_idx,
                                module: "q_proj".into(),
                                a: (0..rank * D_IN)
                                    .map(|i| (i as f32 + seed) * 0.001)
                                    .collect(),
                                b: (0..D_OUT * rank)
                                    .map(|i| (i as f32 + seed) * 0.0005)
                                    .collect(),
                                rank,
                                d_in: D_IN,
                                d_out: D_OUT,
                            };
                            // v_proj (same d_in/d_out for simplicity)
                            let layer_v = LoraLayerData {
                                layer_idx,
                                module: "v_proj".into(),
                                a: (0..rank * D_IN)
                                    .map(|i| (i as f32 + seed + 1.0) * 0.001)
                                    .collect(),
                                b: (0..D_OUT * rank)
                                    .map(|i| (i as f32 + seed + 1.0) * 0.0005)
                                    .collect(),
                                rank,
                                d_in: D_IN,
                                d_out: D_OUT,
                            };
                            [layer_q, layer_v]
                        })
                        .collect()
                })
                .collect();

            let weight = 1.0 / k as f32;
            let refs: Vec<(&[LoraLayerData], f32)> =
                adapters.iter().map(|a| (a.as_slice(), weight)).collect();

            // Warmup
            for _ in 0..warmup {
                let _ = blend_lora_layer_data(&refs);
            }

            // Measured iterations
            let start = Instant::now();
            for _ in 0..iters {
                let blended =
                    blend_lora_layer_data(&refs).expect("blend must not fail on synthetic data");
                // Prevent the compiler from eliminating the blend entirely.
                std::hint::black_box(blended.len());
            }
            let elapsed = start.elapsed();
            let blend_us = elapsed.as_micros() as f64 / iters as f64;

            let layers_count = layer_indices.len() * 2; // q_proj + v_proj
            println!("BLEND_BENCH r={rank} k={k} layers={layers_count} blend_us={blend_us:.1}");
        }
    }

    // GPU decode bench: only if a model directory is available.
    if let Ok(model_dir_str) = std::env::var("LATTICE_MODEL_DIR") {
        run_gpu_decode_bench(&model_dir_str, new_tokens, warmup, iters)?;
    } else {
        eprintln!(
            "[bench_lora_mixture] LATTICE_MODEL_DIR not set; skipping GPU decode bench. \
             Set it to a valid Qwen3.5-0.8b Q4 dir to enable decode tok/s measurements."
        );
    }

    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_gpu_decode_bench(
    model_dir_str: &str,
    new_tokens: usize,
    warmup: usize,
    iters: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::GenerateConfig;
    use lattice_inference::forward::metal_qwen35::{
        LoraLayerData, MetalQwen35State, blend_lora_layer_data,
    };
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use lattice_inference::tokenizer::BpeTokenizer;
    use std::time::Instant;

    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();

    let dir = std::path::Path::new(model_dir_str);
    if !dir.exists() {
        eprintln!("[bench_lora_mixture] model dir does not exist: {model_dir_str}");
        return Ok(());
    }

    let tokenizer_path = dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        eprintln!("[bench_lora_mixture] tokenizer.json not found; skipping GPU bench");
        return Ok(());
    }

    eprintln!("[bench_lora_mixture] loading model from {model_dir_str}");

    let cfg = Qwen35Config::from_model_dir(dir).map_err(|e| format!("config.json load: {e}"))?;

    let mut metal = MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg, 512)
        .map_err(|e| format!("Metal Q4 init: {e}"))?;

    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path)?;

    const D_IN: usize = 1024;
    const D_OUT: usize = 4096;
    let num_layers = cfg.num_hidden_layers;
    let layer_indices = full_attention_layer_indices(&cfg);
    let prompt = "Hello";

    for &rank in &[1usize, 2] {
        for &k in &[1usize, 4, 8] {
            let adapters: Vec<Vec<LoraLayerData>> = (0..k)
                .map(|adapter_idx| {
                    layer_indices
                        .iter()
                        .copied()
                        .map(|layer_idx| {
                            let seed = (adapter_idx * num_layers + layer_idx) as f32;
                            LoraLayerData {
                                layer_idx,
                                module: "q_proj".into(),
                                // Zero A → zero LoRA delta, no numerical change to output.
                                a: vec![0.0f32; rank * D_IN],
                                b: vec![seed * 1e-9; D_OUT * rank],
                                rank,
                                d_in: D_IN,
                                d_out: D_OUT,
                            }
                        })
                        .collect()
                })
                .collect();

            let weight = 1.0 / k as f32;
            let refs: Vec<(&[LoraLayerData], f32)> =
                adapters.iter().map(|a| (a.as_slice(), weight)).collect();

            let mut gen_cfg = GenerateConfig::default();
            gen_cfg.max_new_tokens = new_tokens;
            gen_cfg.enable_thinking = false;

            // Swap-cost bench: blend, GPU unload, GPU load, timed as three
            // separate phases in the same order `ResidencyRegistry::apply`
            // uses on a selection change. A combined number (as
            // `generate_with_lora_mixture` below produces internally) can't
            // tell a coefficient-only fast path's savings (skip the blend,
            // still pay the upload) from the full cost.
            let mut swap_blend_us_total = 0.0f64;
            let mut swap_unload_us_total = 0.0f64;
            let mut swap_load_us_total = 0.0f64;
            let mut swap_layers = 0usize;

            // Warmup
            for _ in 0..warmup {
                let blended =
                    blend_lora_layer_data(&refs).expect("blend must not fail on synthetic data");
                metal.unload_lora_adapter();
                metal
                    .load_lora_adapter(blended, 1.0, None)
                    .expect("load must not fail on synthetic data");
            }

            // Measured iterations
            for _ in 0..iters {
                let blend_start = Instant::now();
                let blended =
                    blend_lora_layer_data(&refs).expect("blend must not fail on synthetic data");
                swap_blend_us_total += blend_start.elapsed().as_micros() as f64;
                swap_layers = blended.len();
                std::hint::black_box(swap_layers);

                let unload_start = Instant::now();
                metal.unload_lora_adapter();
                swap_unload_us_total += unload_start.elapsed().as_micros() as f64;

                let load_start = Instant::now();
                metal
                    .load_lora_adapter(blended, 1.0, None)
                    .expect("load must not fail on synthetic data");
                swap_load_us_total += load_start.elapsed().as_micros() as f64;
            }
            // Leave the slot unloaded: `generate_with_lora_mixture` below
            // unloads unconditionally before its own load, but this bench
            // owns the state it just changed rather than relying on that.
            metal.unload_lora_adapter();

            let swap_blend_us = swap_blend_us_total / iters as f64;
            let swap_unload_us = swap_unload_us_total / iters as f64;
            let swap_load_us = swap_load_us_total / iters as f64;
            println!(
                "SWAP_BENCH r={rank} k={k} layers={swap_layers} blend_us={swap_blend_us:.1} \
                 unload_us={swap_unload_us:.1} load_us={swap_load_us:.1} iters={iters}"
            );

            // Warmup
            let _ = metal.generate_with_lora_mixture(&refs, prompt, &tokenizer, &gen_cfg);

            // Measure
            let start = Instant::now();
            let out = metal.generate_with_lora_mixture(&refs, prompt, &tokenizer, &gen_cfg)?;
            let elapsed_s = start.elapsed().as_secs_f64();
            let tok_s = out.generated_tokens as f64 / elapsed_s;

            println!(
                "DECODE_BENCH r={rank} k={k} tok_s={tok_s:.1} generated={}",
                out.generated_tokens
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_inference::model::qwen35_config::Qwen35Config;

    #[test]
    fn qwen35_0_8b_lora_layers_are_full_attention_only() {
        let cfg = Qwen35Config::qwen35_0_8b();

        assert_eq!(
            full_attention_layer_indices(&cfg),
            vec![3, 7, 11, 15, 19, 23]
        );
    }
}

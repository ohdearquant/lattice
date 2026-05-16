//! Metal Qwen3.5 decode throughput benchmark.
//!
//! Measures single-token decode tok/s on M2 Max with both QuaRot Q4 and naive Q8 weights.
//! Requires:
//! - Q4 model at `~/.lattice/models/qwen3.5-0.8b-q4-quarot/`
//! - Q8/safetensors at `~/.lattice/models/qwen3.5-0.8b/`
//! - tokenizer at `~/.lattice/models/qwen3.5-0.8b/tokenizer.json`
//!
//! Run: `cargo bench -p lattice-inference --features metal-gpu,f16 -- metal_decode`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::path::PathBuf;
use std::time::Duration;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_inference::forward::metal_qwen35::{LoraLayerData, MetalQwen35State};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_inference::model::qwen35_config::Qwen35Config;

fn q4_model_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let quarot = PathBuf::from(format!("{home}/.lattice/models/qwen3.5-0.8b-q4-quarot"));
    if quarot.join("config.json").exists() {
        Some(quarot)
    } else {
        None
    }
}

fn safetensors_model_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir = PathBuf::from(format!("{home}/.lattice/models/qwen3.5-0.8b"));
    if dir.join("config.json").exists() {
        Some(dir)
    } else {
        None
    }
}

fn tokenizer_path() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(format!(
        "{home}/.lattice/models/qwen3.5-0.8b/tokenizer.json"
    ));
    if p.exists() { Some(p) } else { None }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn load_q4_state() -> Option<(MetalQwen35State, Qwen35Config)> {
    let dir = q4_model_dir()?;
    let tok = tokenizer_path()?;
    let cfg = Qwen35Config::from_config_json(&dir.join("config.json")).ok()?;
    let state = MetalQwen35State::from_q4_dir(&dir, &tok, &cfg, 4096).ok()?;
    Some((state, cfg))
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn load_q8_state() -> Option<(MetalQwen35State, Qwen35Config)> {
    use lattice_inference::model::qwen35::Qwen35Model;
    let dir = safetensors_model_dir()?;
    let model = Qwen35Model::from_safetensors(&dir).ok()?;
    let cfg = model.config().clone();
    let state = MetalQwen35State::new(model.weights(), &cfg, 4096).ok()?;
    Some((state, cfg))
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn generate_random_lora(cfg: &Qwen35Config, rank: usize) -> Vec<LoraLayerData> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let mut layers = Vec::new();
    let mut rng_state: u64 = 12345;

    let mut next_f32 = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 11) as u32 as f32 / u32::MAX as f32) - 0.5
    };

    for layer_idx in 0..cfg.num_hidden_layers {
        let is_full = cfg.is_full_attention(layer_idx);

        let attn_modules: Vec<(&str, usize, usize)> = if is_full {
            vec![
                ("q_proj", hidden, 2 * cfg.full_q_dim()),
                ("k_proj", hidden, cfg.full_kv_dim()),
                ("v_proj", hidden, cfg.full_kv_dim()),
                ("o_proj", cfg.full_q_dim(), hidden),
            ]
        } else {
            vec![
                ("in_proj_qkv", hidden, cfg.linear_qkv_dim()),
                ("in_proj_z", hidden, cfg.linear_output_dim()),
                ("out_proj", cfg.linear_output_dim(), hidden),
            ]
        };

        let mlp_modules: Vec<(&str, usize, usize)> = vec![
            ("gate_proj", hidden, inter),
            ("up_proj", hidden, inter),
            ("down_proj", inter, hidden),
        ];

        for (module, d_in, d_out) in attn_modules.into_iter().chain(mlp_modules) {
            let a: Vec<f32> = (0..rank * d_in).map(|_| next_f32() * 0.02).collect();
            let b: Vec<f32> = (0..d_out * rank).map(|_| next_f32() * 0.02).collect();
            layers.push(LoraLayerData {
                layer_idx,
                module: module.to_string(),
                a,
                b,
                rank,
                d_in,
                d_out,
            });
        }
    }

    layers
}

fn bench_metal_decode_q4(c: &mut Criterion) {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("SKIP: metal_decode bench requires macOS + metal-gpu feature");
        let _ = c;
        return;
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        let Some((mut state, cfg)) = load_q4_state() else {
            eprintln!(
                "SKIP: Q4 model not found at ~/.lattice/models/qwen3.5-0.8b-q4-quarot\n\
                 (tokenizer expected at ~/.lattice/models/qwen3.5-0.8b/tokenizer.json)"
            );
            return;
        };

        let mut group = c.benchmark_group("metal_decode_q4");
        group.warm_up_time(Duration::from_secs(3));
        group.measurement_time(Duration::from_secs(10));
        group.sample_size(50);

        let prompt_tokens: Vec<u32> = vec![42, 100, 200, 300, 400];
        state.forward_prefill(&prompt_tokens);
        let mut pos = prompt_tokens.len();

        group.throughput(Throughput::Elements(1));
        group.bench_function(BenchmarkId::new("forward_step", "no_adapter"), |b| {
            b.iter(|| {
                let _logits = state.forward_step(42, pos);
                pos += 1;
                if pos > 500 {
                    state.reset_state();
                    state.forward_prefill(&prompt_tokens);
                    pos = prompt_tokens.len();
                }
            });
        });

        state.reset_state();
        let lora_layers = generate_random_lora(&cfg, 8);
        state
            .load_lora_adapter(lora_layers, 1.0, Some(42))
            .expect("load_lora_adapter (rank 8)");
        state.forward_prefill(&prompt_tokens);
        pos = prompt_tokens.len();

        group.bench_function(BenchmarkId::new("forward_step", "lora_rank8"), |b| {
            b.iter(|| {
                let _logits = state.forward_step(42, pos);
                pos += 1;
                if pos > 500 {
                    state.reset_state();
                    state.unload_lora_adapter();
                    state
                        .load_lora_adapter(generate_random_lora(&cfg, 8), 1.0, Some(42))
                        .expect("reload lora_rank8");
                    state.forward_prefill(&prompt_tokens);
                    pos = prompt_tokens.len();
                }
            });
        });

        state.unload_lora_adapter();
        group.finish();
    }
}

fn bench_metal_decode_q8(c: &mut Criterion) {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("SKIP: metal_decode bench requires macOS + metal-gpu feature");
        let _ = c;
        return;
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        let Some((mut state, _cfg)) = load_q8_state() else {
            eprintln!(
                "SKIP: Q8 model not found at ~/.lattice/models/qwen3.5-0.8b\n\
                 (needs config.json + model.safetensors)"
            );
            return;
        };

        let mut group = c.benchmark_group("metal_decode_q8");
        group.warm_up_time(Duration::from_secs(3));
        group.measurement_time(Duration::from_secs(10));
        group.sample_size(50);

        let prompt_tokens: Vec<u32> = vec![42, 100, 200, 300, 400];
        state.forward_prefill(&prompt_tokens);
        let mut pos = prompt_tokens.len();

        group.throughput(Throughput::Elements(1));
        group.bench_function(BenchmarkId::new("forward_step", "no_adapter"), |b| {
            b.iter(|| {
                let _logits = state.forward_step(42, pos);
                pos += 1;
                if pos > 500 {
                    state.reset_state();
                    state.forward_prefill(&prompt_tokens);
                    pos = prompt_tokens.len();
                }
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_metal_decode_q4, bench_metal_decode_q8);
criterion_main!(benches);

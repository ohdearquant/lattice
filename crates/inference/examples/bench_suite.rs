//! Unified benchmark suite with structured JSON output for autoresearch.
//!
//! Runs LLM generation, GDN layer, and embedding benchmarks, then outputs
//! structured results that can be consumed by the krons autoresearch pipeline.
//!
//! Usage:
//!   cargo run --release -p lattice-inference --bin bench_suite
//!   cargo run --release -p lattice-inference --bin bench_suite -- --json
//!   cargo run --release -p lattice-inference --bin bench_suite -- --json --baseline benchmarks/baseline.json

use std::time::Instant;

// ---------------------------------------------------------------------------
// Metric collection
// ---------------------------------------------------------------------------

struct Metric {
    name: &'static str,
    value: f64,
    unit: &'static str,
}

// ---------------------------------------------------------------------------
// GDN benchmark (no model files needed)
// ---------------------------------------------------------------------------

fn bench_gdn() -> Vec<Metric> {
    use lattice_inference::attention::gdn::*;
    use lattice_inference::attention::gdn_fused::*;
    use lattice_inference::model::qwen35_config::Qwen35Config;

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

    // Warmup + reference
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
        t0.elapsed().as_micros() as f64
    };

    // Warmup + fused
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
                &lattice_inference::lora_hook::NoopLoraHook,
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
                &lattice_inference::lora_hook::NoopLoraHook,
                0,
            );
        }
        t0.elapsed().as_micros() as f64
    };

    let ref_per = ref_us / iters as f64;
    let fused_per = fused_us / iters as f64;
    let speedup = ref_per / fused_per;

    vec![
        Metric {
            name: "gdn_ref_us_per_step",
            value: ref_per,
            unit: "us",
        },
        Metric {
            name: "gdn_fused_us_per_step",
            value: fused_per,
            unit: "us",
        },
        Metric {
            name: "gdn_fused_speedup",
            value: speedup,
            unit: "x",
        },
    ]
}

// ---------------------------------------------------------------------------
// LLM benchmark (Qwen3.5-2B)
// ---------------------------------------------------------------------------

fn bench_llm() -> Vec<Metric> {
    let home = std::env::var("HOME").unwrap();
    let model_dir = format!("{home}/.lattice/models/qwen3.5-2b");
    let dir = std::path::Path::new(&model_dir);

    if !dir.join("model.safetensors").exists() {
        eprintln!("[bench_suite] Qwen3.5-2B model not found at {model_dir}, skipping LLM bench");
        return vec![];
    }

    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::GenerateConfig;

    // Load model
    let t_load = Instant::now();
    let model = Qwen35Model::from_safetensors(dir).expect("failed to load Qwen3.5-2B");
    let load_ms = t_load.elapsed().as_millis() as f64;

    let gen_cfg = GenerateConfig {
        max_new_tokens: 20,
        temperature: 0.0,
        top_k: 1,
        seed: Some(42),
        ..Default::default()
    };

    // Warmup
    let _ = model.generate("Hello", &gen_cfg);

    // Short prompt benchmark
    let short_prompt = "The capital of France is";
    let n_runs = 3;
    let mut total_tok = 0usize;
    let mut total_ms = 0f64;
    let mut ttft_ms_sum = 0f64;
    let mut prefill_ms_sum = 0f64;

    for _ in 0..n_runs {
        let t0 = Instant::now();
        let result = model
            .generate(short_prompt, &gen_cfg)
            .expect("generate failed");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let gen_tokens = result.generated_tokens;
        let prompt_tokens = result.prompt_tokens;

        // Estimate TTFT: total_time * (prompt_tokens / (prompt_tokens + gen_tokens))
        // This is approximate — true TTFT requires instrumented prefill
        let prefill_frac = prompt_tokens as f64 / (prompt_tokens + gen_tokens) as f64;
        let est_ttft = elapsed_ms * prefill_frac;
        let est_prefill_per_tok = est_ttft / prompt_tokens as f64;

        total_tok += gen_tokens;
        total_ms += elapsed_ms;
        ttft_ms_sum += est_ttft;
        prefill_ms_sum += est_prefill_per_tok;
    }

    let avg_ms = total_ms / n_runs as f64;
    let tok_per_sec = total_tok as f64 / (total_ms / 1000.0);
    let avg_ttft = ttft_ms_sum / n_runs as f64;
    let avg_prefill_per_tok = prefill_ms_sum / n_runs as f64;

    vec![
        Metric {
            name: "llm_load_ms",
            value: load_ms,
            unit: "ms",
        },
        Metric {
            name: "llm_tok_per_sec",
            value: tok_per_sec,
            unit: "tok/s",
        },
        Metric {
            name: "llm_ttft_ms",
            value: avg_ttft,
            unit: "ms",
        },
        Metric {
            name: "llm_prefill_ms_per_tok",
            value: avg_prefill_per_tok,
            unit: "ms/tok",
        },
        Metric {
            name: "llm_total_ms_20tok",
            value: avg_ms,
            unit: "ms",
        },
    ]
}

// ---------------------------------------------------------------------------
// F16 LLM benchmark (Qwen3.5-2B with half-precision weights)
// ---------------------------------------------------------------------------

#[cfg(feature = "f16")]
fn bench_llm_f16() -> Vec<Metric> {
    let home = std::env::var("HOME").unwrap();
    let model_dir = format!("{home}/.lattice/models/qwen3.5-2b");
    let dir = std::path::Path::new(&model_dir);

    if !dir.join("model.safetensors").exists() {
        eprintln!("[bench_suite] Qwen3.5-2B model not found, skipping f16 LLM bench");
        return vec![];
    }

    use lattice_inference::forward::cpu_f16::generate_f16;
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::rope::RopeTable;
    use lattice_inference::tokenizer::bpe::BpeTokenizer;
    use lattice_inference::weights::SafetensorsFile;
    use lattice_inference::weights::f16_weights::load_f16_weights;

    let cfg = Qwen35Config::qwen35_2b();
    let tokenizer = BpeTokenizer::from_tokenizer_json(&dir.join("tokenizer.json"))
        .expect("failed to load tokenizer");
    let rope_dim = cfg.rope_dim();
    let rope_max = cfg.max_position_embeddings.min(8192);
    let rope = RopeTable::new(rope_dim, rope_max, cfg.rope_theta);
    let sf =
        SafetensorsFile::open(&dir.join("model.safetensors")).expect("failed to open safetensors");

    let t_load = Instant::now();
    let f16_weights = load_f16_weights(&sf, &cfg).expect("failed to load f16 weights");
    let load_ms = t_load.elapsed().as_millis() as f64;

    let gen_cfg = GenerateConfig {
        max_new_tokens: 20,
        temperature: 0.0,
        top_k: 1,
        seed: Some(42),
        ..Default::default()
    };

    // Warmup
    let _ = generate_f16(&f16_weights, &cfg, &tokenizer, &rope, "Hello", &gen_cfg);

    let short_prompt = "The capital of France is";
    let n_runs = 3;
    let mut total_tok = 0usize;
    let mut total_ms = 0f64;

    for _ in 0..n_runs {
        let t0 = Instant::now();
        let result = generate_f16(
            &f16_weights,
            &cfg,
            &tokenizer,
            &rope,
            short_prompt,
            &gen_cfg,
        )
        .expect("f16 generate failed");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_tok += result.generated_tokens;
        total_ms += elapsed_ms;
    }

    let tok_per_sec = total_tok as f64 / (total_ms / 1000.0);
    let avg_ms = total_ms / n_runs as f64;

    vec![
        Metric {
            name: "f16_load_ms",
            value: load_ms,
            unit: "ms",
        },
        Metric {
            name: "f16_tok_per_sec",
            value: tok_per_sec,
            unit: "tok/s",
        },
        Metric {
            name: "f16_total_ms_20tok",
            value: avg_ms,
            unit: "ms",
        },
    ]
}

// ---------------------------------------------------------------------------
// Q8 LLM benchmark (Qwen3.5-2B with INT8 quantized weights)
// ---------------------------------------------------------------------------

fn bench_llm_q8() -> Vec<Metric> {
    let home = std::env::var("HOME").unwrap();
    let model_dir = format!("{home}/.lattice/models/qwen3.5-2b");
    let dir = std::path::Path::new(&model_dir);

    if !dir.join("model.safetensors").exists() {
        eprintln!("[bench_suite] Qwen3.5-2B model not found, skipping Q8 LLM bench");
        return vec![];
    }

    use lattice_inference::forward::cpu_q8::{generate_q8, quantize_from_model};
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::rope::RopeTable;
    use lattice_inference::tokenizer::bpe::BpeTokenizer;

    // Load f32 model first, then quantize
    let model = Qwen35Model::from_safetensors(dir).expect("failed to load Qwen3.5-2B");
    let cfg = Qwen35Config::qwen35_2b();
    let tokenizer = BpeTokenizer::from_tokenizer_json(&dir.join("tokenizer.json"))
        .expect("failed to load tokenizer");
    let rope_dim = cfg.rope_dim();
    let rope_max = cfg.max_position_embeddings.min(8192);
    let rope = RopeTable::new(rope_dim, rope_max, cfg.rope_theta);

    let t_quant = Instant::now();
    let q8_weights = quantize_from_model(&model);
    let quant_ms = t_quant.elapsed().as_millis() as f64;

    // Drop the f32 model to free memory
    drop(model);

    let gen_cfg = GenerateConfig {
        max_new_tokens: 20,
        temperature: 0.0,
        top_k: 1,
        seed: Some(42),
        ..Default::default()
    };

    // Warmup
    let _ = generate_q8(&q8_weights, &cfg, &tokenizer, &rope, "Hello", &gen_cfg);

    let short_prompt = "The capital of France is";
    let n_runs = 3;
    let mut total_tok = 0usize;
    let mut total_ms = 0f64;

    for _ in 0..n_runs {
        let t0 = Instant::now();
        let result = generate_q8(&q8_weights, &cfg, &tokenizer, &rope, short_prompt, &gen_cfg)
            .expect("q8 generate failed");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_tok += result.generated_tokens;
        total_ms += elapsed_ms;
    }

    let tok_per_sec = total_tok as f64 / (total_ms / 1000.0);
    let avg_ms = total_ms / n_runs as f64;

    vec![
        Metric {
            name: "q8_quant_ms",
            value: quant_ms,
            unit: "ms",
        },
        Metric {
            name: "q8_tok_per_sec",
            value: tok_per_sec,
            unit: "tok/s",
        },
        Metric {
            name: "q8_total_ms_20tok",
            value: avg_ms,
            unit: "ms",
        },
    ]
}

// ---------------------------------------------------------------------------
// Q8 NEON LLM benchmark (Qwen3.5-2B on native NEON int8)
// ---------------------------------------------------------------------------

fn bench_llm_q8_neon() -> Vec<Metric> {
    let home = std::env::var("HOME").unwrap();
    let model_dir = format!("{home}/.lattice/models/qwen3.5-2b");
    let dir = std::path::Path::new(&model_dir);

    if !dir.join("model.safetensors").exists() {
        eprintln!("[bench_suite] Qwen3.5-2B model not found, skipping Q8 NEON bench");
        return vec![];
    }

    use lattice_inference::forward::neon_forward::{generate_q8_neon, quantize_model};
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::rope::RopeTable;
    use lattice_inference::tokenizer::bpe::BpeTokenizer;

    let model = Qwen35Model::from_safetensors(dir).expect("failed to load Qwen3.5-2B");
    let cfg = Qwen35Config::qwen35_2b();
    let tokenizer = BpeTokenizer::from_tokenizer_json(&dir.join("tokenizer.json"))
        .expect("failed to load tokenizer");
    let rope_dim = cfg.rope_dim();
    let rope_max = cfg.max_position_embeddings.min(8192);
    let rope = RopeTable::new(rope_dim, rope_max, cfg.rope_theta);

    let t_quant = Instant::now();
    let q8_model = quantize_model(model.weights(), &cfg);
    let quant_ms = t_quant.elapsed().as_millis() as f64;

    drop(model);

    let gen_cfg = GenerateConfig {
        max_new_tokens: 20,
        temperature: 0.0,
        top_k: 1,
        seed: Some(42),
        ..Default::default()
    };

    // Warmup
    let _ = generate_q8_neon(&q8_model, &cfg, &tokenizer, &rope, "Hello", &gen_cfg);

    let short_prompt = "The capital of France is";
    let n_runs = 3;
    let mut total_tok = 0usize;
    let mut total_ms = 0f64;

    for _ in 0..n_runs {
        let t0 = Instant::now();
        let result = generate_q8_neon(&q8_model, &cfg, &tokenizer, &rope, short_prompt, &gen_cfg)
            .expect("q8_neon generate failed");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_tok += result.generated_tokens;
        total_ms += elapsed_ms;
    }

    let tok_per_sec = total_tok as f64 / (total_ms / 1000.0);
    let avg_ms = total_ms / n_runs as f64;

    vec![
        Metric {
            name: "q8_neon_quant_ms",
            value: quant_ms,
            unit: "ms",
        },
        Metric {
            name: "q8_neon_tok_per_sec",
            value: tok_per_sec,
            unit: "tok/s",
        },
        Metric {
            name: "q8_neon_total_ms_20tok",
            value: avg_ms,
            unit: "ms",
        },
    ]
}

// ---------------------------------------------------------------------------
// Metal GPU LLM benchmark (Qwen3.5-2B on Metal)
// ---------------------------------------------------------------------------

#[cfg(feature = "metal-gpu")]
fn bench_llm_metal() -> Vec<Metric> {
    let home = std::env::var("HOME").unwrap();
    let model_dir = format!("{home}/.lattice/models/qwen3.5-2b");
    let dir = std::path::Path::new(&model_dir);

    if !dir.join("model.safetensors").exists() {
        eprintln!("[bench_suite] Qwen3.5-2B model not found, skipping Metal GPU bench");
        return vec![];
    }

    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::tokenizer::bpe::BpeTokenizer;

    let model = Qwen35Model::from_safetensors(dir).expect("failed to load Qwen3.5-2B");
    let cfg = model.config().clone();
    let tokenizer = BpeTokenizer::from_tokenizer_json(&dir.join("tokenizer.json"))
        .expect("failed to load tokenizer");

    let t_init = Instant::now();
    let mut metal_state =
        MetalQwen35State::new(model.weights(), &cfg, 4096).expect("failed to init Metal GPU state");
    let init_ms = t_init.elapsed().as_millis() as f64;

    let gen_cfg = GenerateConfig {
        max_new_tokens: 20,
        temperature: 0.0,
        top_k: 1,
        seed: Some(42),
        ..Default::default()
    };

    // Warmup
    let _ = metal_state.generate("Hello", &tokenizer, &gen_cfg);
    metal_state.reset_state();

    let short_prompt = "The capital of France is";
    let n_runs = 3;
    let mut total_tok = 0usize;
    let mut total_ms = 0f64;

    for _ in 0..n_runs {
        metal_state.reset_state();
        let t0 = Instant::now();
        let result = metal_state.generate(short_prompt, &tokenizer, &gen_cfg);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_tok += result.generated_tokens;
        total_ms += elapsed_ms;
    }

    let tok_per_sec = total_tok as f64 / (total_ms / 1000.0);
    let avg_ms = total_ms / n_runs as f64;

    // Drop f32 model to free memory
    drop(model);

    vec![
        Metric {
            name: "metal_init_ms",
            value: init_ms,
            unit: "ms",
        },
        Metric {
            name: "metal_tok_per_sec",
            value: tok_per_sec,
            unit: "tok/s",
        },
        Metric {
            name: "metal_total_ms_20tok",
            value: avg_ms,
            unit: "ms",
        },
    ]
}

// ---------------------------------------------------------------------------
// Embedding benchmark (Qwen3-0.6B)
// ---------------------------------------------------------------------------

fn bench_embedding() -> Vec<Metric> {
    let home = std::env::var("HOME").unwrap();
    let model_dir = format!("{home}/.lattice/models/qwen3-embedding-0.6b");
    let dir = std::path::Path::new(&model_dir);

    if !dir.join("model.safetensors").exists() {
        eprintln!(
            "[bench_suite] Qwen3-0.6B model not found at {model_dir}, skipping embedding bench"
        );
        return vec![];
    }

    let t_load = Instant::now();
    let model =
        lattice_inference::QwenModel::from_directory(dir).expect("failed to load Qwen3-0.6B");
    let load_ms = t_load.elapsed().as_millis() as f64;

    // Warmup
    let _ = model.encode("warmup").unwrap();

    let short = "hello world";
    let medium = "The Qwen3 embedding model uses a decoder-only transformer architecture with \
                  grouped query attention, rotary position embeddings, SwiGLU FFN, and RMS \
                  normalization for efficient multilingual text representation across 100+ languages.";
    let long_text = "The Qwen3 embedding model uses a decoder-only transformer architecture \
                     with grouped query attention and rotary position embeddings for efficient \
                     multilingual text representation. "
        .repeat(25);

    let mut results = vec![Metric {
        name: "embed_load_ms",
        value: load_ms,
        unit: "ms",
    }];

    for (label, text, n) in [
        ("embed_short_ms", short, 10),
        ("embed_medium_ms", medium, 5),
        ("embed_long_ms", long_text.as_str(), 3),
    ] {
        let t = Instant::now();
        for _ in 0..n {
            let _ = model.encode(text).unwrap();
        }
        let ms = t.elapsed().as_millis() as f64 / n as f64;
        results.push(Metric {
            name: label,
            value: ms,
            unit: "ms",
        });
    }

    results
}

// ---------------------------------------------------------------------------
// Baseline comparison
// ---------------------------------------------------------------------------

fn load_baseline(path: &str) -> Vec<(String, f64)> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[bench_suite] Failed to read baseline {path}: {e}");
            return vec![];
        }
    };

    // Minimal JSON parsing — extract "name": {"value": N} pairs
    let mut entries = Vec::new();
    let metrics_start = content.find("\"metrics\"").unwrap_or(0);
    let section = &content[metrics_start..];

    // Find each metric key and its value
    let mut pos = 0;
    while pos < section.len() {
        // Find next quoted key after "metrics"
        if let Some(key_start) = section[pos..].find('"') {
            let ks = pos + key_start + 1;
            if let Some(key_end) = section[ks..].find('"') {
                let key = &section[ks..ks + key_end];
                // Skip non-metric keys
                if key == "metrics"
                    || key == "version"
                    || key == "created"
                    || key == "device"
                    || key == "commit"
                    || key == "value"
                    || key == "unit"
                    || key == "description"
                {
                    pos = ks + key_end + 1;
                    continue;
                }
                // Look for "value": NUMBER
                let after_key = &section[ks + key_end..];
                if let Some(val_idx) = after_key.find("\"value\"") {
                    let val_section = &after_key[val_idx + 8..];
                    // Skip whitespace and colon
                    let trimmed =
                        val_section.trim_start_matches(|c: char| c == ':' || c.is_whitespace());
                    if !trimmed.starts_with("null") {
                        // Parse number
                        let num_end = trimmed
                            .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
                            .unwrap_or(trimmed.len());
                        if let Ok(val) = trimmed[..num_end].parse::<f64>() {
                            entries.push((key.to_string(), val));
                        }
                    }
                }
                pos = ks + key_end + 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    entries
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

fn print_json(metrics: &[Metric]) {
    // Get commit hash
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    println!("{{");
    println!("  \"commit\": \"{commit}\",");
    println!("  \"timestamp\": {timestamp},");
    println!("  \"device\": \"{}\",", detect_device());
    println!("  \"metrics\": [");

    for (i, m) in metrics.iter().enumerate() {
        let comma = if i + 1 < metrics.len() { "," } else { "" };
        println!(
            "    {{\"name\": \"{}\", \"value\": {:.4}, \"unit\": \"{}\"}}{comma}",
            m.name, m.value, m.unit
        );
    }

    println!("  ]");
    println!("}}");
}

fn print_table(metrics: &[Metric], baseline: &[(String, f64)]) {
    println!("╔══════════════════════════════╦══════════════╦════════╦══════════╗");
    println!("║ Metric                       ║ Value        ║ Unit   ║ vs Base  ║");
    println!("╠══════════════════════════════╬══════════════╬════════╬══════════╣");

    for m in metrics {
        let base_delta = baseline
            .iter()
            .find(|(k, _)| k == m.name)
            .map(|(_, base)| {
                let pct = (m.value - base) / base * 100.0;
                if pct.abs() < 0.5 {
                    "  ~same ".to_string()
                } else if pct > 0.0 {
                    // For tok/s higher is better, for latency lower is better
                    let is_higher_better =
                        m.name.contains("tok_per_sec") || m.name.contains("speedup");
                    if (pct > 0.0) == is_higher_better {
                        format!(" +{pct:>5.1}% ")
                    } else {
                        format!(" {pct:>+5.1}% ")
                    }
                } else {
                    let is_higher_better =
                        m.name.contains("tok_per_sec") || m.name.contains("speedup");
                    if (pct < 0.0) == !is_higher_better {
                        format!(" {pct:>5.1}% ")
                    } else {
                        format!(" {pct:>+5.1}% ")
                    }
                }
            })
            .unwrap_or_else(|| "   N/A  ".to_string());

        println!(
            "║ {:<28} ║ {:>12.2} ║ {:<6} ║{base_delta}║",
            m.name, m.value, m.unit
        );
    }

    println!("╚══════════════════════════════╩══════════════╩════════╩══════════╝");
}

fn detect_device() -> String {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Apple Silicon".to_string())
    }
    #[cfg(not(target_os = "macos"))]
    {
        "Unknown".to_string()
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_mode = args.iter().any(|a| a == "--json");
    let baseline_path = args
        .iter()
        .position(|a| a == "--baseline")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    // Select which benchmarks to run
    let run_all = !args
        .iter()
        .any(|a| a == "--llm" || a == "--gdn" || a == "--embed");
    let run_llm = run_all || args.iter().any(|a| a == "--llm");
    let run_gdn = run_all || args.iter().any(|a| a == "--gdn");
    let run_embed = run_all || args.iter().any(|a| a == "--embed");

    eprintln!("[bench_suite] Starting benchmark suite...");
    let t_total = Instant::now();

    let mut metrics = Vec::new();

    if run_gdn {
        eprintln!("[bench_suite] Running GDN benchmark...");
        metrics.extend(bench_gdn());
    }

    if run_llm {
        eprintln!("[bench_suite] Running LLM benchmark (Qwen3.5-2B)...");
        metrics.extend(bench_llm());

        #[cfg(feature = "f16")]
        {
            eprintln!("[bench_suite] Running F16 LLM benchmark (Qwen3.5-2B)...");
            metrics.extend(bench_llm_f16());
        }

        eprintln!("[bench_suite] Running Q8 LLM benchmark (Qwen3.5-2B)...");
        metrics.extend(bench_llm_q8());

        eprintln!("[bench_suite] Running Q8 NEON LLM benchmark (Qwen3.5-2B)...");
        metrics.extend(bench_llm_q8_neon());

        #[cfg(feature = "metal-gpu")]
        {
            eprintln!("[bench_suite] Running Metal GPU LLM benchmark (Qwen3.5-2B)...");
            metrics.extend(bench_llm_metal());
        }
    }

    if run_embed {
        eprintln!("[bench_suite] Running embedding benchmark (Qwen3-0.6B)...");
        metrics.extend(bench_embedding());
    }

    let total_s = t_total.elapsed().as_secs_f64();
    eprintln!(
        "[bench_suite] Done in {total_s:.1}s ({} metrics collected)",
        metrics.len()
    );

    // Load baseline for comparison
    let baseline = baseline_path.map(|p| load_baseline(p)).unwrap_or_default();

    if json_mode {
        print_json(&metrics);
    } else {
        print_table(&metrics, &baseline);
    }
}

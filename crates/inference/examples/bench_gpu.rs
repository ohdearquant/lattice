//! Benchmark: GPU forward pass for Qwen3-Embedding-0.6B vs CPU.
//!
//! Usage:
//!   cargo run --release -p lattice-inference --example bench_gpu --features "f16,wgpu-gpu"
//!
//! `-- --gpu-only` times the GPU forward alone, for hosts that cannot hold the CPU model
//! beside the GPU copy. It loads no CPU model, drops the checkpoint's f32 view once the
//! host copy is built and the host copy once the weights are uploaded, so it peaks at two
//! f32 copies of the weights and holds only the GPU buffers while timing. It prints no CPU
//! column and no cosine similarity.

use std::path::Path;
use std::time::Instant;

#[cfg(feature = "wgpu-gpu")]
use lattice_inference::forward::gpu::{
    GpuForwardError, GpuModelState, GpuRuntimeConfig, Qwen3Config as GpuQwen3Config,
    Qwen3LayerWeights as GpuLayerWeights, Qwen3Weights as GpuWeights,
};

fn main() {
    #[cfg(not(feature = "wgpu-gpu"))]
    {
        eprintln!("This binary requires the 'wgpu-gpu' feature.");
        std::process::exit(1);
    }

    #[cfg(feature = "wgpu-gpu")]
    if std::env::args().any(|a| a == "--gpu-only") {
        run_gpu_only();
    } else {
        run_bench();
    }
}

#[cfg(feature = "wgpu-gpu")]
fn run_bench() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = format!("{home}/.lattice/models/qwen3-embedding-0.6b");
    let dir = Path::new(&model_dir);

    if !dir.join("model.safetensors").exists() {
        eprintln!("Model not found at {model_dir}");
        eprintln!(
            "Download: huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir {model_dir}"
        );
        std::process::exit(1);
    }

    println!("=== Qwen3-Embedding-0.6B GPU Benchmark ===\n");

    // Load CPU model for comparison
    println!("Loading CPU model...");
    let t0 = Instant::now();
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();
    let cpu_model = lattice_inference::QwenModel::from_directory(dir).unwrap();
    let cpu_load_ms = t0.elapsed().as_millis();
    println!("  CPU model loaded in {cpu_load_ms}ms\n");

    // Extract config
    let cpu_cfg = cpu_model.config();
    let gpu_config = GpuQwen3Config {
        vocab_size: cpu_cfg.vocab_size,
        hidden_size: cpu_cfg.hidden_size,
        num_hidden_layers: cpu_cfg.num_hidden_layers,
        num_attention_heads: cpu_cfg.num_attention_heads,
        num_key_value_heads: cpu_cfg.num_key_value_heads,
        head_dim: cpu_cfg.head_dim,
        intermediate_size: cpu_cfg.intermediate_size,
        max_position_embeddings: cpu_cfg.max_position_embeddings,
        rms_norm_eps: cpu_cfg.rms_norm_eps,
        rope_theta: cpu_cfg.rope_theta,
    };

    // Load weights from safetensors for GPU
    println!("Loading weights from safetensors for GPU upload...");
    let t0 = Instant::now();
    let safetensors =
        lattice_inference::weights::SafetensorsFile::open(&dir.join("model.safetensors")).unwrap();
    let mmap_weights = lattice_inference::model::qwen::weight_loading::load_weights(
        &safetensors,
        cpu_cfg.num_hidden_layers,
        cpu_cfg.hidden_size,
        cpu_cfg.num_attention_heads,
        cpu_cfg.num_key_value_heads,
        cpu_cfg.head_dim,
        cpu_cfg.intermediate_size,
    )
    .unwrap();

    // Convert mmap-backed weights to owned Vec<f32> for GPU
    let gpu_weights = GpuWeights {
        embed_tokens: mmap_weights.embed_tokens.data.to_vec(),
        norm_weight: mmap_weights.norm_weight.data.to_vec(),
        layers: mmap_weights
            .layers
            .iter()
            .map(|l| GpuLayerWeights {
                q_proj_weight: l.q_proj_weight.data.to_vec(),
                k_proj_weight: l.k_proj_weight.data.to_vec(),
                v_proj_weight: l.v_proj_weight.data.to_vec(),
                o_proj_weight: l.o_proj_weight.data.to_vec(),
                q_norm_weight: l.q_norm_weight.data.to_vec(),
                k_norm_weight: l.k_norm_weight.data.to_vec(),
                input_layernorm_weight: l.input_layernorm_weight.data.to_vec(),
                gate_proj_weight: l.gate_proj_weight.data.to_vec(),
                up_proj_weight: l.up_proj_weight.data.to_vec(),
                down_proj_weight: l.down_proj_weight.data.to_vec(),
                post_attention_layernorm_weight: l.post_attention_layernorm_weight.data.to_vec(),
            })
            .collect(),
    };
    let weight_load_ms = t0.elapsed().as_millis();
    println!("  Weights copied in {weight_load_ms}ms\n");

    // Create GPU model state with conservative max_seq_len for benchmark
    let runtime = GpuRuntimeConfig {
        max_seq_len: 512,
        upload_embeddings_to_gpu: false,
    };

    println!("Initializing GPU model state (uploading weights to GPU)...");
    let t0 = Instant::now();
    let gpu_model = match GpuModelState::new(gpu_config.clone(), &gpu_weights, runtime) {
        Ok(m) => m,
        Err(GpuForwardError::NoAdapter) => {
            eprintln!("No GPU adapter found. Cannot run GPU benchmark.");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Failed to initialize GPU: {e}");
            std::process::exit(1);
        }
    };
    let gpu_init_ms = t0.elapsed().as_millis();
    println!("  GPU initialized in {gpu_init_ms}ms");
    println!(
        "  Buffers created: {}",
        gpu_model.user_buffer_creation_count()
    );
    println!();

    // Test texts
    let short = "hello world";
    let medium = "The Qwen3 embedding model uses a decoder-only transformer architecture with grouped query attention, rotary position embeddings, SwiGLU FFN, and RMS normalization for efficient multilingual text representation across 100+ languages.";
    let long_text = "The Qwen3 embedding model uses a decoder-only transformer architecture with grouped query attention and rotary position embeddings for efficient multilingual text representation. ".repeat(5);

    // Warmup CPU
    let _ = cpu_model.encode("warmup").unwrap();

    println!("--- Comparison: CPU encode() vs GPU forward() ---\n");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>10}  {:>10}",
        "Text", "CPU (ms)", "GPU (ms)", "Speedup", "CosSim"
    );
    println!("{:-<64}", "");

    for (label, text) in [
        ("short", short),
        ("medium", medium),
        ("long", long_text.as_str()),
    ] {
        // Tokenize for GPU
        let input = cpu_model.tokenizer().tokenize(text);
        let seq_len = input.real_length;
        let input_ids: Vec<u32> = input.input_ids[..seq_len].to_vec();

        // CPU benchmark
        let n_cpu = match label {
            "short" => 10,
            "medium" => 5,
            _ => 3,
        };
        let t = Instant::now();
        let mut cpu_emb = Vec::new();
        for _ in 0..n_cpu {
            cpu_emb = cpu_model.encode(text).unwrap();
        }
        let cpu_ms = t.elapsed().as_millis() as f64 / n_cpu as f64;

        // GPU warmup
        let _ = gpu_model.forward(&input_ids, seq_len);

        // GPU benchmark
        let n_gpu = match label {
            "short" => 20,
            "medium" => 10,
            _ => 5,
        };
        let t = Instant::now();
        let mut gpu_hidden = Vec::new();
        for _ in 0..n_gpu {
            gpu_hidden = gpu_model.forward(&input_ids, seq_len).unwrap();
        }
        let gpu_ms = t.elapsed().as_millis() as f64 / n_gpu as f64;

        // Extract last-token hidden state from GPU output for comparison.
        // GPU returns full [seq_len, hidden_size], we need the last token.
        let hidden_size = gpu_config.hidden_size;
        let last_tok_start = (seq_len - 1) * hidden_size;
        let gpu_last_hidden = if gpu_hidden.len() >= last_tok_start + hidden_size {
            &gpu_hidden[last_tok_start..last_tok_start + hidden_size]
        } else {
            &gpu_hidden[..hidden_size.min(gpu_hidden.len())]
        };

        // L2 normalize for cosine similarity
        let gpu_norm = l2_norm(gpu_last_hidden);
        let cpu_norm = l2_norm(&cpu_emb);

        // Cosine similarity (cpu_emb is already normalized, but be safe)
        let cos_sim = if gpu_norm > 0.0 && cpu_norm > 0.0 {
            let min_len = gpu_last_hidden.len().min(cpu_emb.len());
            let mut dot = 0.0f64;
            for i in 0..min_len {
                dot += gpu_last_hidden[i] as f64 * cpu_emb[i] as f64;
            }
            dot / (gpu_norm * cpu_norm)
        } else {
            0.0
        };

        let speedup = if gpu_ms > 0.0 {
            cpu_ms / gpu_ms
        } else {
            f64::INFINITY
        };
        println!(
            "{:>10}  {:>10.1}ms  {:>10.1}ms  {:>9.2}x  {:>9.4}",
            label, cpu_ms, gpu_ms, speedup, cos_sim
        );
    }

    println!("\n  Total GPU submits: {}", gpu_model.submit_count());
    println!(
        "  Total GPU buffers created: {}",
        gpu_model.user_buffer_creation_count()
    );
    println!("\nDone.");
}

fn l2_norm(v: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for &x in v {
        sum += (x as f64) * (x as f64);
    }
    sum.sqrt()
}

#[cfg(feature = "wgpu-gpu")]
fn run_gpu_only() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = format!("{home}/.lattice/models/qwen3-embedding-0.6b");
    let dir = Path::new(&model_dir);

    if !dir.join("model.safetensors").exists() {
        eprintln!("Model not found at {model_dir}");
        std::process::exit(1);
    }

    println!("=== Qwen3-Embedding-0.6B GPU Benchmark (GPU only) ===\n");

    let cfg = lattice_inference::model::qwen::QwenConfig::qwen3_embedding_0_6b();
    let tokenizer = lattice_inference::load_tokenizer(dir).unwrap();
    let gpu_config = GpuQwen3Config {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        intermediate_size: cfg.intermediate_size,
        max_position_embeddings: cfg.max_position_embeddings,
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta: cfg.rope_theta,
    };

    println!("Loading weights from safetensors for GPU upload...");
    let t0 = Instant::now();
    let gpu_weights = {
        let safetensors =
            lattice_inference::weights::SafetensorsFile::open(&dir.join("model.safetensors"))
                .unwrap();
        let mmap_weights = lattice_inference::model::qwen::weight_loading::load_weights(
            &safetensors,
            cfg.num_hidden_layers,
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.intermediate_size,
        )
        .unwrap();
        GpuWeights {
            embed_tokens: mmap_weights.embed_tokens.data.to_vec(),
            norm_weight: mmap_weights.norm_weight.data.to_vec(),
            layers: mmap_weights
                .layers
                .iter()
                .map(|l| GpuLayerWeights {
                    q_proj_weight: l.q_proj_weight.data.to_vec(),
                    k_proj_weight: l.k_proj_weight.data.to_vec(),
                    v_proj_weight: l.v_proj_weight.data.to_vec(),
                    o_proj_weight: l.o_proj_weight.data.to_vec(),
                    q_norm_weight: l.q_norm_weight.data.to_vec(),
                    k_norm_weight: l.k_norm_weight.data.to_vec(),
                    input_layernorm_weight: l.input_layernorm_weight.data.to_vec(),
                    gate_proj_weight: l.gate_proj_weight.data.to_vec(),
                    up_proj_weight: l.up_proj_weight.data.to_vec(),
                    down_proj_weight: l.down_proj_weight.data.to_vec(),
                    post_attention_layernorm_weight: l
                        .post_attention_layernorm_weight
                        .data
                        .to_vec(),
                })
                .collect(),
        }
    };
    println!("  Weights copied in {}ms\n", t0.elapsed().as_millis());

    let runtime = GpuRuntimeConfig {
        max_seq_len: 512,
        upload_embeddings_to_gpu: false,
    };

    println!("Initializing GPU model state (uploading weights to GPU)...");
    let t0 = Instant::now();
    let gpu_model = match GpuModelState::new(gpu_config, &gpu_weights, runtime) {
        Ok(m) => m,
        Err(GpuForwardError::NoAdapter) => {
            eprintln!("No GPU adapter found. Cannot run GPU benchmark.");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Failed to initialize GPU: {e}");
            std::process::exit(1);
        }
    };
    drop(gpu_weights);
    println!("  GPU initialized in {}ms\n", t0.elapsed().as_millis());

    let short = "hello world";
    let medium = "The Qwen3 embedding model uses a decoder-only transformer architecture with grouped query attention, rotary position embeddings, SwiGLU FFN, and RMS normalization for efficient multilingual text representation across 100+ languages.";
    let long_text = "The Qwen3 embedding model uses a decoder-only transformer architecture with grouped query attention and rotary position embeddings for efficient multilingual text representation. ".repeat(5);

    println!("{:>10}  {:>12}  {:>8}", "Text", "GPU (ms)", "Tokens");
    println!("{:-<34}", "");

    for (label, text) in [
        ("short", short),
        ("medium", medium),
        ("long", long_text.as_str()),
    ] {
        let input = tokenizer.tokenize(text);
        let seq_len = input.real_length;
        let input_ids: Vec<u32> = input.input_ids[..seq_len].to_vec();

        let _ = gpu_model.forward(&input_ids, seq_len);

        let n_gpu = match label {
            "short" => 20,
            "medium" => 10,
            _ => 5,
        };
        let t = Instant::now();
        for _ in 0..n_gpu {
            gpu_model.forward(&input_ids, seq_len).unwrap();
        }
        let gpu_ms = t.elapsed().as_millis() as f64 / n_gpu as f64;

        println!("{label:>10}  {gpu_ms:>10.1}ms  {seq_len:>8}");
    }

    println!("\n  Total GPU submits: {}", gpu_model.submit_count());
    println!("\nDone.");
}

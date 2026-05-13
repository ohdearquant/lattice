//! Benchmark: Direct Metal forward pass for Qwen3-Embedding-0.6B vs CPU.
//!
//! Usage:
//!   cargo run --release --features "f16,metal-gpu" --bin bench_metal -p lattice-inference

use std::path::Path;
use std::time::Instant;

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("This binary requires macOS + the 'metal-gpu' feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    run_bench();
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_bench() {
    use lattice_inference::QwenModel;
    use lattice_inference::forward::metal::MetalForwardPass;
    use lattice_inference::pool::{l2_normalize, last_token_pool};

    let home = std::env::var("HOME").unwrap();
    let model_dir = format!("{home}/.lattice/models/qwen3-embedding-0.6b");
    let dir = Path::new(&model_dir);

    if !dir.join("model.safetensors").exists() {
        eprintln!("Model not found at {model_dir}");
        eprintln!("Run: cargo run --release --features 'f16,download' --bin bench_embedding");
        std::process::exit(1);
    }

    println!("=== Qwen3-Embedding-0.6B Metal Forward Pass Benchmark ===\n");

    // Load CPU model.
    println!("Loading CPU model...");
    let t0 = Instant::now();
    let cpu_model = QwenModel::from_directory(dir).unwrap();
    let cpu_load_ms = t0.elapsed().as_millis();
    println!(
        "  CPU model loaded in {cpu_load_ms}ms (GPU: {})\n",
        if cpu_model.has_gpu() {
            "Metal"
        } else {
            "CPU-only"
        }
    );

    let cpu_cfg = cpu_model.config();
    let hidden_size = cpu_cfg.hidden_size;

    // Load weights from safetensors for Metal upload.
    println!("Loading weights for Metal upload...");
    let t0 = Instant::now();
    let safetensors =
        lattice_inference::weights::SafetensorsFile::open(&dir.join("model.safetensors")).unwrap();
    let mmap_weights = safetensors
        .load_qwen_weights(
            cpu_cfg.num_hidden_layers,
            cpu_cfg.hidden_size,
            cpu_cfg.num_attention_heads,
            cpu_cfg.num_key_value_heads,
            cpu_cfg.head_dim,
            cpu_cfg.intermediate_size,
        )
        .unwrap();
    let weight_load_ms = t0.elapsed().as_millis();
    println!("  Weights loaded in {weight_load_ms}ms\n");

    // Create Metal forward pass (uploads weights to unified memory).
    println!("Initializing Metal forward pass...");
    let t0 = Instant::now();
    let mut metal = match MetalForwardPass::new(cpu_cfg, &mmap_weights, 2048) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to initialize Metal: {e}");
            std::process::exit(1);
        }
    };
    let metal_init_ms = t0.elapsed().as_millis();
    println!("  Metal initialized in {metal_init_ms}ms\n");

    // Test texts.
    let short = "hello world";
    let medium = "The Qwen3 embedding model uses a decoder-only transformer architecture \
                  with grouped query attention, rotary position embeddings, SwiGLU FFN, \
                  and RMS normalization for efficient multilingual text representation.";
    let long_text = "The Qwen3 embedding model uses a decoder-only transformer architecture \
                     with grouped query attention and rotary position embeddings for efficient \
                     multilingual text representation. "
        .repeat(5);
    let very_long = "The Qwen3 embedding model uses a decoder-only transformer architecture \
                     with grouped query attention, rotary position embeddings, SwiGLU FFN, \
                     and RMS normalization for efficient multilingual text representation. \
                     It supports Matryoshka output dimension truncation and last-token pooling. "
        .repeat(10);

    // Warmup CPU.
    let _ = cpu_model.encode("warmup").unwrap();

    println!("--- Comparison: CPU encode() vs Metal forward() ---\n");
    println!(
        "{:>10}  {:>8}  {:>12}  {:>12}  {:>10}  {:>10}",
        "Text", "Tokens", "CPU (ms)", "Metal (ms)", "Speedup", "CosSim"
    );
    println!("{:-<74}", "");

    for (label, text) in [
        ("short", short),
        ("medium", medium),
        ("long", long_text.as_str()),
        ("very_long", very_long.as_str()),
    ] {
        // Tokenize for Metal path.
        let input = cpu_model.tokenizer().tokenize(text);
        let seq_len = input.real_length;
        let input_ids = &input.input_ids[..seq_len];

        // CPU embedding gather buffer for Metal path.
        let mut hidden_input = vec![0.0f32; seq_len * hidden_size];

        // CPU benchmark (full encode: tokenize + forward + pool + normalize).
        let n_cpu = match label {
            "short" => 10,
            "medium" => 5,
            _ => 3,
        };
        cpu_model.cache_clear();
        let t = Instant::now();
        let mut cpu_emb = Vec::new();
        for _ in 0..n_cpu {
            cpu_model.cache_clear();
            cpu_emb = cpu_model.encode(text).unwrap();
        }
        let cpu_ms = t.elapsed().as_millis() as f64 / n_cpu as f64;

        // For Metal: do embedding gather on CPU, then Metal forward, then CPU pool+norm.
        // We need the raw embedding table. Get it from the mmap weights.
        for (i, &tok_id) in input_ids.iter().enumerate() {
            let tok_id = tok_id as usize;
            let src =
                &mmap_weights.embed_tokens.data[tok_id * hidden_size..(tok_id + 1) * hidden_size];
            hidden_input[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(src);
        }

        // Metal warmup.
        let _ = metal.forward(&hidden_input, seq_len).unwrap();

        // Metal benchmark.
        let n_metal = match label {
            "short" => 20,
            "medium" => 10,
            _ => 5,
        };
        let t = Instant::now();
        let mut metal_hidden = Vec::new();
        for _ in 0..n_metal {
            metal_hidden = metal.forward(&hidden_input, seq_len).unwrap();
        }
        let metal_ms = t.elapsed().as_millis() as f64 / n_metal as f64;

        // Pool + normalize the Metal output.
        let metal_pooled = last_token_pool(
            &metal_hidden,
            &input.attention_mask[..seq_len],
            seq_len,
            hidden_size,
        );
        let mut metal_emb = metal_pooled;
        l2_normalize(&mut metal_emb);

        // Cosine similarity.
        let cos_sim = cosine_similarity(&cpu_emb, &metal_emb);

        let speedup = if metal_ms > 0.0 {
            cpu_ms / metal_ms
        } else {
            f64::INFINITY
        };

        println!(
            "{:>10}  {:>8}  {:>10.1}ms  {:>10.1}ms  {:>9.2}x  {:>9.4}",
            label, seq_len, cpu_ms, metal_ms, speedup, cos_sim
        );
    }

    // --- Batch throughput benchmark ---
    println!("\n--- Batch Throughput ---\n");

    let batch_texts: Vec<String> = vec![
        "hello world".into(),
        "The quick brown fox jumps over the lazy dog".into(),
        "Machine learning enables computers to learn from data and make predictions".into(),
        "Rust programming language provides memory safety without garbage collection".into(),
        "Neural network transformer architecture with multi-head attention mechanism".into(),
        "Database indexing and query optimization for high-performance data retrieval".into(),
        "Cloud computing provides scalable on-demand infrastructure and platform services".into(),
        "Version control systems like git enable distributed collaborative software development"
            .into(),
        "RESTful API design principles for building robust and maintainable web services".into(),
        "Container orchestration with kubernetes enables automated deployment and scaling".into(),
    ];
    let batch_refs: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();

    // Batch encode (sequential GPU forwards)
    cpu_model.cache_clear();
    let rounds = 10;
    let t = Instant::now();
    for _ in 0..rounds {
        cpu_model.cache_clear();
        let _ = cpu_model.encode_batch(&batch_refs).unwrap();
    }
    let batch_ms = t.elapsed().as_millis() as f64 / rounds as f64;
    let per_text = batch_ms / batch_texts.len() as f64;
    let throughput = 1000.0 / per_text;
    println!(
        "Batch of {}: {batch_ms:.0}ms total, {per_text:.1}ms/text",
        batch_texts.len()
    );
    println!("Throughput: {throughput:.0} texts/sec");
    println!(
        "Migration estimate (13K atoms): {:.0}s ({:.1}min)",
        13000.0 / throughput,
        13000.0 / throughput / 60.0
    );

    // Cache hit speed
    let _ = cpu_model.encode("cached text").unwrap();
    let t = Instant::now();
    for _ in 0..1000 {
        let _ = cpu_model.encode("cached text").unwrap();
    }
    let cached_us = t.elapsed().as_micros() as f64 / 1000.0;
    println!("\nCache hit: {cached_us:.1}us");

    println!("\nDone.");
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let min_len = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..min_len {
        dot += a[i] as f64 * b[i] as f64;
        norm_a += (a[i] as f64) * (a[i] as f64);
        norm_b += (b[i] as f64) * (b[i] as f64);
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

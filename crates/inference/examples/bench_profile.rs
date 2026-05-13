//! Per-component profiler for Qwen3-Embedding-0.6B inference.
//!
//! Breaks down forward pass into: tokenize, embed lookup, QKV proj, QK-norm,
//! RoPE, attention, O proj, FFN, final norm, pool, L2 normalize.
//!
//! Usage:
//!   cargo run --release --features f16 --bin bench_profile

use lattice_inference::QwenModel;
use std::time::Instant;

fn main() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = format!("{home}/.lattice/models/qwen3-embedding-0.6b");
    let dir = std::path::Path::new(&model_dir);

    if !dir.join("model.safetensors").exists() {
        eprintln!("Model not found at {model_dir}");
        eprintln!(
            "Download: huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir {model_dir}"
        );
        std::process::exit(1);
    }

    println!("Loading model...");
    let t0 = Instant::now();
    let model = QwenModel::from_directory(dir).unwrap();
    println!("Loaded in {:.0}ms\n", t0.elapsed().as_millis());

    // Warmup.
    let _ = model.encode("warmup").unwrap();

    let short = "hello world";
    let medium = "The Qwen3 embedding model uses a decoder-only transformer architecture with grouped query attention, rotary position embeddings, SwiGLU FFN, and RMS normalization for efficient multilingual text representation across 100+ languages.";
    let long_text = "The Qwen3 embedding model uses a decoder-only transformer architecture with grouped query attention and rotary position embeddings for efficient multilingual text representation. ".repeat(25);

    for (label, text) in [
        ("SHORT", short),
        ("MEDIUM", medium),
        ("LONG", long_text.as_str()),
    ] {
        println!("\n{}", "=".repeat(60));
        println!("=== {label} ({} chars) ===", text.len());

        // Run profiled encode.
        let (_, timings) = model.encode_profiled(text).unwrap();
        timings.print_report(text.len());

        // Also run non-profiled for comparison (overhead check).
        let n = match label {
            "SHORT" => 10,
            "MEDIUM" => 5,
            _ => 3,
        };
        let t = Instant::now();
        for _ in 0..n {
            let _ = model.encode(text).unwrap();
        }
        let unprofiled_ms = t.elapsed().as_millis() as f64 / n as f64;
        println!(
            "\nUnprofiled avg: {unprofiled_ms:.1}ms (profiled: {:.1}ms, overhead: {:.1}%)",
            timings.total_us as f64 / 1000.0,
            ((timings.total_us as f64 / 1000.0) / unprofiled_ms - 1.0) * 100.0,
        );
    }

    // Cache benchmark.
    println!("\n{}", "=".repeat(60));
    println!("=== EMBEDDING CACHE ===\n");

    // Warmup populates cache.
    let _ = model.encode(short).unwrap();
    let _ = model.encode(medium).unwrap();
    let _ = model.encode(long_text.as_str()).unwrap();
    println!("Cache entries: {}", model.cache_size());

    for (label, text) in [
        ("SHORT", short),
        ("MEDIUM", medium),
        ("LONG", long_text.as_str()),
    ] {
        let n = 1000;
        let t = Instant::now();
        for _ in 0..n {
            let _ = model.encode(text).unwrap();
        }
        let ns = t.elapsed().as_nanos() as f64 / n as f64;
        println!("  {label:8} cache hit: {ns:.0}ns ({:.3}us)", ns / 1000.0);
    }

    // Test cache persistence.
    let cache_path = std::path::Path::new("/tmp/lattice_embedding_cache_test.bin");
    let saved = model.cache_save(cache_path).unwrap();
    println!("\n  Saved {saved} entries to {}", cache_path.display());
    let file_size = std::fs::metadata(cache_path).map(|m| m.len()).unwrap_or(0);
    println!("  File size: {:.1}KB", file_size as f64 / 1024.0);

    model.cache_clear();
    assert_eq!(model.cache_size(), 0);
    let loaded = model.cache_load(cache_path).unwrap();
    println!("  Loaded {loaded} entries back");

    // Verify loaded cache works.
    let t = Instant::now();
    for _ in 0..1000 {
        let _ = model.encode(short).unwrap();
    }
    let ns = t.elapsed().as_nanos() as f64 / 1000.0;
    println!("  Post-load cache hit: {ns:.0}ns ({:.3}us)", ns / 1000.0);

    std::fs::remove_file(cache_path).ok();

    // Summary: where is the time going?
    println!("\n\n{}", "=".repeat(60));
    println!("=== OPTIMIZATION TARGETS ===\n");
    println!("Based on the profiles above, the biggest optimization targets are:");
    println!("  1. QKV projections — 3 separate matmul_bt → fuse into 1");
    println!("  2. FFN gate+up — 2 separate matmul_bt → fuse into 1");
    println!("  3. Attention — O(seq^2) scaling, flash attention candidate");
    println!("  4. QK-norm + RoPE — per-position loops, SIMD candidate");
    println!("  5. Weight loading — INT8/INT4 quantization for bandwidth");
}

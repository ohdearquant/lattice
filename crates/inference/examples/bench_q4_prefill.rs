/// Benchmark: Qwen3.6-27B Q4 forward_prefill latency (512-token prompt).
/// Measures the gemm_q4 batch path introduced to eliminate the M-dispatch loop.
///
/// Usage:
///   LATTICE_MODEL_DIR=~/.lattice/models/qwen3.6-27b-q4 \
///   LATTICE_TOKENIZER_DIR=~/.lattice/models/qwen3.6-27b \
///   cargo run --release --example bench_q4_prefill -p lattice-inference --features "f16,metal-gpu"
fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("bench_q4_prefill requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    run();
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use std::time::Instant;

    let home = std::env::var("HOME").expect("HOME not set");
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.6-27b-q4"));
    let tokenizer_dir_str = std::env::var("LATTICE_TOKENIZER_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.6-27b"));

    let dir = std::path::Path::new(&model_dir_str);
    let tokenizer_path = std::path::Path::new(&tokenizer_dir_str).join("tokenizer.json");

    let cfg = if dir.join("config.json").exists() {
        Qwen35Config::from_config_json(&dir.join("config.json")).expect("parse config.json")
    } else {
        Qwen35Config::qwen36_27b()
    };

    eprintln!("[bench] Loading Q4 model from {} ...", dir.display());
    let t0 = Instant::now();
    let mut state = MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg, 4096)
        .expect("from_q4_dir failed — MSL compile or weight load error");
    eprintln!("[bench] Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // 512 synthetic token IDs cycling through mid-vocabulary (stable, repeatable)
    let prompt_ids: Vec<u32> = (0u32..512).map(|i| 100 + (i % 1000)).collect();
    let seq_len = prompt_ids.len();

    let n_warmup = 1usize;
    let n_measure = 3usize;

    eprintln!("[bench] Warmup ({n_warmup} run)...");
    for _ in 0..n_warmup {
        state.reset_state();
        let _ = state.forward_prefill(&prompt_ids);
    }

    eprintln!("[bench] Measuring ({n_measure} runs, seq_len={seq_len})...");
    let mut latencies_ms: Vec<f64> = Vec::with_capacity(n_measure);
    for run in 0..n_measure {
        state.reset_state();
        let t = Instant::now();
        let _ = state.forward_prefill(&prompt_ids);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        latencies_ms.push(ms);
        eprintln!(
            "  run {}: {:.1} ms  ({:.1} tok/s)",
            run + 1,
            ms,
            seq_len as f64 / (ms / 1000.0)
        );
    }

    let mean = latencies_ms.iter().sum::<f64>() / n_measure as f64;
    let min = latencies_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = latencies_ms
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let variance = latencies_ms.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_measure as f64;
    let std_dev = variance.sqrt();

    eprintln!("\n[bench] Results (seq_len={seq_len})");
    eprintln!(
        "  mean:    {mean:.1} ms  ({:.1} tok/s)",
        seq_len as f64 / (mean / 1000.0)
    );
    eprintln!("  min:     {min:.1} ms");
    eprintln!("  max:     {max:.1} ms");
    eprintln!("  std_dev: {std_dev:.1} ms");

    // Structured output for benchmarks.md
    println!(
        "BENCH_RESULT seq_len={seq_len} mean_ms={mean:.1} min_ms={min:.1} max_ms={max:.1} std_dev_ms={std_dev:.1} mean_toks_per_sec={:.1}",
        seq_len as f64 / (mean / 1000.0)
    );
}

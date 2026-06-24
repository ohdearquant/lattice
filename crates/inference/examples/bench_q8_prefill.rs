/// Benchmark: Qwen3.5-0.8B Q8 forward_prefill latency (512-token prompt).
/// Measures the gemm_q8_tiled simdgroup-matrix batch GEMM path on Q8_0 models.
///
/// Usage:
///   LATTICE_MODEL_DIR=~/.lattice/models/qwen3.5-0.8b \
///   cargo run --release --example bench_q8_prefill -p lattice-inference --features "f16,metal-gpu"
fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("bench_q8_prefill requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    run();
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::Qwen35Model;
    use std::time::Instant;

    let home = std::env::var("HOME").expect("HOME not set");
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
    let dir = std::path::Path::new(&model_dir_str);

    if !dir.exists() {
        eprintln!("[bench] Model directory not found: {}", dir.display());
        eprintln!(
            "[bench] Set LATTICE_MODEL_DIR to a Q8 safetensors model directory \
             (e.g. ~/.lattice/models/qwen3.5-0.8b)"
        );
        std::process::exit(1);
    }

    eprintln!("[bench] Loading Q8 model from {} ...", dir.display());
    let t0 = Instant::now();
    let model = Qwen35Model::from_safetensors(dir).expect("from_safetensors failed");
    eprintln!(
        "[bench] Weights loaded in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let t1 = Instant::now();
    let mut state = MetalQwen35State::new(model.weights(), model.config(), 4096)
        .expect("MetalQwen35State::new failed — MSL compile or weight upload error");
    eprintln!(
        "[bench] GPU state initialized in {:.1}s",
        t1.elapsed().as_secs_f64()
    );

    eprintln!("[bench] gemm_q8_tiled: active on Apple7+ (M1 and up)");

    // Synthetic token IDs cycling through mid-vocabulary (stable, repeatable).
    let seq_len: usize = std::env::var("LATTICE_SEQ_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let prompt_ids: Vec<u32> = (0u32..seq_len as u32).map(|i| 100 + (i % 1000)).collect();

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
    let min = latencies_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max = latencies_ms
        .iter()
        .copied()
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

    println!(
        "BENCH_RESULT seq_len={seq_len} mean_ms={mean:.1} min_ms={min:.1} \
         max_ms={max:.1} std_dev_ms={std_dev:.1} mean_toks_per_sec={:.1}",
        seq_len as f64 / (mean / 1000.0)
    );
}

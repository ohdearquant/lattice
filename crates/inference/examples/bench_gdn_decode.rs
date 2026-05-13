/// Baseline benchmark: Qwen3.6-27B Q4 single decode step (`forward_step`) latency.
///
/// Establishes decode-step timing BEFORE GDN GPU optimizations are applied.
/// Run with `LATTICE_PROFILE=1` to emit per-step total times on stderr.
///
/// Usage:
///   LATTICE_MODEL_DIR=~/.lattice/models/qwen3.6-27b-q4 \
///   LATTICE_TOKENIZER_DIR=~/.lattice/models/qwen3.6-27b \
///   LATTICE_PROFILE=1 \
///   cargo run --release --example bench_gdn_decode -p lattice-inference --features "f16,metal-gpu"
fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("bench_gdn_decode requires macOS + metal-gpu feature.");
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

    eprintln!(
        "[bench_gdn_decode] Loading Q4 model from {} ...",
        dir.display()
    );
    let t_load = Instant::now();
    let mut state = MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg, 512)
        .expect("from_q4_dir failed — MSL compile or weight load error");
    eprintln!(
        "[bench_gdn_decode] Model loaded in {:.1}s",
        t_load.elapsed().as_secs_f64()
    );

    // Short 16-token prompt to establish minimal KV cache context.
    // Cycles tokens 100-115 (stable, repeatable, in-vocabulary).
    let prompt_ids: Vec<u32> = (0u32..16).map(|i| 100 + i).collect();
    let prefill_len = prompt_ids.len();

    eprintln!("[bench_gdn_decode] Warmup prefill ({prefill_len} tokens)...");
    state.reset_state();
    let _ = state.forward_prefill(&prompt_ids);

    // Warmup decode: 3 steps at positions prefill_len..prefill_len+3
    // kv_cache.seq_len is private — it stays fixed at prefill_len after forward_prefill.
    // We increment the `position` argument to give correct RoPE embeddings each step.
    // KV writes overwrite the same slot (seq_len slot), which is acceptable for timing.
    for i in 0..3usize {
        let _ = state.forward_step(200 + i as u32, prefill_len + i);
    }

    let n_steps = 10usize;
    eprintln!(
        "[bench_gdn_decode] Measuring {n_steps} decode steps (LATTICE_PROFILE prints per-step total)..."
    );
    let mut step_times_us: Vec<u128> = Vec::with_capacity(n_steps);

    for i in 0..n_steps {
        let pos = prefill_len + i;
        let t = Instant::now();
        let _ = state.forward_step(200 + i as u32, pos);
        step_times_us.push(t.elapsed().as_micros());
    }

    // Statistics
    let total_us: u128 = step_times_us.iter().sum();
    let mean_us = total_us as f64 / n_steps as f64;
    let min_us = *step_times_us.iter().min().unwrap() as f64;
    let max_us = *step_times_us.iter().max().unwrap() as f64;
    let variance = step_times_us
        .iter()
        .map(|&x| (x as f64 - mean_us).powi(2))
        .sum::<f64>()
        / n_steps as f64;
    let std_dev = variance.sqrt();

    eprintln!(
        "\n[bench_gdn_decode] Decode step results ({n_steps} steps, prefill_ctx={prefill_len} tokens)"
    );
    eprintln!(
        "  NOTE: kv_cache.seq_len is private — KV writes use fixed slot; RoPE uses correct positions {prefill_len}..{}",
        prefill_len + n_steps
    );
    eprintln!(
        "  NOTE: LATTICE_PROFILE per-component counters (t_attn_gdn, t_attn_gqa, t_mlp, t_final) are declared but NOT wired — only total is emitted"
    );
    eprintln!(
        "  mean:    {mean_us:.0}us  ({:.2} tok/s)",
        1_000_000.0 / mean_us
    );
    eprintln!("  min:     {min_us:.0}us");
    eprintln!("  max:     {max_us:.0}us");
    eprintln!("  std_dev: {std_dev:.0}us");
    for (i, &us) in step_times_us.iter().enumerate() {
        eprintln!("  step {:2}: {us}us", i + 1);
    }

    // Structured output for baseline.md
    println!(
        "DECODE_BENCH steps={n_steps} ctx={prefill_len} mean_us={mean_us:.0} min_us={min_us:.0} max_us={max_us:.0} std_dev_us={std_dev:.0} mean_toks_per_sec={:.2}",
        1_000_000.0 / mean_us
    );
}

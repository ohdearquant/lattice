/// Benchmark: Layer pruning impact on Qwen3.6-27B Q4 Metal GPU.
/// Tests baseline + 8/12/16-layer removal. Measures prefill throughput (512 tokens)
/// and decode throughput (32 steps), 2 warmup + 3 measured runs each.
///
/// Usage:
///   LATTICE_MODEL_DIR=~/.lattice/models/qwen3.6-27b-q4 \
///   LATTICE_TOKENIZER_DIR=~/.lattice/models/qwen3.6-27b \
///   cargo run --release --example bench_pruning -p lattice-inference --features "f16,metal-gpu"
fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("bench_pruning requires macOS + metal-gpu feature.");
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

    let base_cfg = if dir.join("config.json").exists() {
        Qwen35Config::from_config_json(&dir.join("config.json")).expect("parse config.json")
    } else {
        Qwen35Config::qwen36_27b()
    };

    let num_layers = base_cfg.num_hidden_layers;
    eprintln!("[bench_pruning] Model: {num_layers}-layer Qwen3.6-27B Q4");

    // Layer masks for each configuration.
    // Architecture: [GDN,GDN,GDN,GQA]×16 = 48 GDN + 16 GQA.
    // Pruning from center blocks (5-8, layers 20-35). Preserve layers 0-3 and 60-63.
    //
    // Block layout:
    //   Block 5: layers 20,21,22(GDN),23(GQA)
    //   Block 6: layers 24,25,26(GDN),27(GQA)
    //   Block 7: layers 28,29,30(GDN),31(GQA)
    //   Block 8: layers 32,33,34(GDN),35(GQA)

    // 8 layers removed (12.5%): 8 GDN from center
    let pruned_8: Vec<usize> = vec![28, 29, 30, 32, 33, 34, 36, 37];

    // 12 layers removed (18.75%): type-balanced — 9 GDN + 3 GQA
    let pruned_12: Vec<usize> = vec![24, 25, 26, 28, 29, 30, 32, 33, 34, 27, 31, 35];

    // 16 layers removed (25%, past Gromov): type-balanced — 12 GDN + 4 GQA
    let pruned_16: Vec<usize> = vec![
        20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 23, 27, 31, 35,
    ];

    let make_mask = |prune: &[usize]| -> Vec<bool> {
        let mut m = vec![true; num_layers];
        for &i in prune {
            m[i] = false;
        }
        m
    };

    let configs: &[(&str, Vec<bool>)] = &[
        ("baseline (64 active)", vec![true; num_layers]),
        ("pruned-8  (56 active)", make_mask(&pruned_8)),
        ("pruned-12 (52 active)", make_mask(&pruned_12)),
        ("pruned-16 (48 active)", make_mask(&pruned_16)),
    ];

    // Synthetic prompt: 512 mid-vocab token IDs, deterministic
    let prompt_ids: Vec<u32> = (0u32..512).map(|i| 100 + (i % 1000)).collect();
    let n_warmup = 2usize;
    let n_measure = 3usize;
    let decode_steps = 32usize;

    eprintln!("\nPROMPT_LEN=512  DECODE_STEPS={decode_steps}  WARMUP={n_warmup}  RUNS={n_measure}");
    eprintln!("============================================================");
    println!(
        "BENCH_PRUNING_HEADER label,active_layers,prefill_mean_ms,prefill_tok_per_sec,decode_mean_ms_per_tok,decode_tok_per_sec,load_time_s"
    );

    for (label, mask) in configs {
        let active = mask.iter().filter(|&&x| x).count();
        let mut cfg = base_cfg.clone();
        cfg.apply_layer_mask(mask.clone());

        eprintln!("\n[{label}]  active={active}");

        // Load model with this config
        let t_load = Instant::now();
        eprintln!("  Loading...");
        let mut state = match MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg, 4096) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  LOAD FAILED: {e}");
                println!("BENCH_PRUNING_RESULT {label},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR");
                continue;
            }
        };
        let load_s = t_load.elapsed().as_secs_f64();
        eprintln!("  Loaded in {load_s:.1}s");

        // ---- Prefill benchmark ----
        eprintln!("  Prefill warmup ({n_warmup} runs)...");
        for _ in 0..n_warmup {
            state.reset_state();
            let _ = state.forward_prefill(&prompt_ids);
        }

        eprintln!("  Prefill measuring ({n_measure} runs)...");
        let mut prefill_ms: Vec<f64> = Vec::with_capacity(n_measure);
        for run in 0..n_measure {
            state.reset_state();
            let t = Instant::now();
            let _ = state.forward_prefill(&prompt_ids);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            prefill_ms.push(ms);
            eprintln!(
                "    prefill run {}: {:.0} ms  ({:.1} tok/s)",
                run + 1,
                ms,
                prompt_ids.len() as f64 / (ms / 1000.0)
            );
        }
        let prefill_mean = prefill_ms.iter().sum::<f64>() / n_measure as f64;
        let prefill_tps = prompt_ids.len() as f64 / (prefill_mean / 1000.0);
        eprintln!("  Prefill mean: {prefill_mean:.0} ms  ({prefill_tps:.1} tok/s)");

        // ---- Decode benchmark ----
        // Prime the KV cache with prefill, then measure decode steps
        eprintln!("  Decode warmup ({n_warmup} rounds of {decode_steps} steps)...");
        for _ in 0..n_warmup {
            state.reset_state();
            let _ = state.forward_prefill(&prompt_ids[..32]);
            for step in 0..decode_steps {
                let _ = state.forward_step(100u32 + step as u32, 32 + step);
            }
        }

        eprintln!("  Decode measuring ({n_measure} rounds)...");
        let mut decode_ms_total: Vec<f64> = Vec::with_capacity(n_measure);
        for run in 0..n_measure {
            state.reset_state();
            let _ = state.forward_prefill(&prompt_ids[..32]);
            let t = Instant::now();
            for step in 0..decode_steps {
                let _ = state.forward_step(100u32 + step as u32, 32 + step);
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            decode_ms_total.push(ms);
            eprintln!(
                "    decode run {}: {:.0} ms total  ({:.1} tok/s)",
                run + 1,
                ms,
                decode_steps as f64 / (ms / 1000.0)
            );
        }
        let decode_mean_total = decode_ms_total.iter().sum::<f64>() / n_measure as f64;
        let decode_ms_per_tok = decode_mean_total / decode_steps as f64;
        let decode_tps = decode_steps as f64 / (decode_mean_total / 1000.0);
        eprintln!("  Decode mean: {decode_ms_per_tok:.1} ms/tok  ({decode_tps:.1} tok/s)");

        println!(
            "BENCH_PRUNING_RESULT {label},{active},{prefill_mean:.0},{prefill_tps:.1},{decode_ms_per_tok:.1},{decode_tps:.1},{load_s:.1}"
        );

        drop(state);
    }

    eprintln!("\n[bench_pruning] Done.");
}

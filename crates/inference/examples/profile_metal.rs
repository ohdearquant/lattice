/// Metal GPU benchmark — proper throughput measurement with multiple prompts.
/// Usage: cargo run --release -p lattice-inference --bin profile_metal --features "f16,metal-gpu"

fn main() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = format!("{home}/.lattice/models/qwen3.5-2b");
    let dir = std::path::Path::new(&model_dir);

    if !dir.join("model.safetensors").exists() {
        eprintln!("Model not found at {model_dir}");
        std::process::exit(1);
    }

    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::GenerateConfig;
    use lattice_inference::tokenizer::bpe::BpeTokenizer;

    eprintln!("[bench] Loading model...");
    let t0 = std::time::Instant::now();
    let model = Qwen35Model::from_safetensors(dir).expect("load model");
    let cfg = model.config().clone();
    let tokenizer =
        BpeTokenizer::from_tokenizer_json(&dir.join("tokenizer.json")).expect("load tokenizer");
    eprintln!("[bench] Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    eprintln!("[bench] Initializing Metal GPU state...");
    let t1 = std::time::Instant::now();
    let mut metal = MetalQwen35State::new(model.weights(), &cfg, 4096).expect("init metal");
    eprintln!("[bench] Metal init in {:.1}s", t1.elapsed().as_secs_f64());

    // Build reverse vocab using tokenizer's internal mapping
    let decode = |ids: &[u32]| -> String {
        use lattice_inference::tokenizer::bpe::byte_decode_token;
        ids.iter()
            .filter_map(|id| tokenizer.token_for_id(*id))
            .map(|s| byte_decode_token(s))
            .collect()
    };

    let gen_cfg_greedy = GenerateConfig {
        max_new_tokens: 20,
        temperature: 0.0,
        top_k: 1,
        seed: Some(630),
        ..Default::default()
    };

    // Warmup
    eprintln!("\n[bench] Warmup...");
    let _ = metal.generate("Hello", &tokenizer, &gen_cfg_greedy);
    metal.reset_state();

    // === Benchmark prompts ===
    let prompts = [
        ("Short factual", "The capital of France is"),
        ("Code", "def fibonacci(n):"),
        (
            "Reasoning",
            "If all cats are animals and some animals are pets, then",
        ),
        (
            "Long context",
            "The history of artificial intelligence began in the 1950s when Alan Turing proposed",
        ),
    ];

    eprintln!("\n============================================================");
    eprintln!("  BENCHMARK: Qwen3.5-2B Q8_0 Metal GPU (single-cmd pipeline)");
    eprintln!("============================================================\n");

    for (label, prompt) in &prompts {
        for max_tok in [20, 50] {
            let gen_cfg = GenerateConfig {
                max_new_tokens: max_tok,
                temperature: 0.0,
                top_k: 1,
                seed: Some(630),
                ..Default::default()
            };

            // Run 3 times, take best
            let mut best_tps = 0.0f64;
            let mut best_result = None;
            let n_runs = 3;

            for run in 0..n_runs {
                metal.reset_state();
                let t = std::time::Instant::now();
                let result = metal.generate(prompt, &tokenizer, &gen_cfg);
                let elapsed = t.elapsed();
                let tps = result.generated_tokens as f64 / elapsed.as_secs_f64();
                if tps > best_tps {
                    best_tps = tps;
                    if run == 0 {
                        best_result = Some(result);
                    }
                }
            }

            let result = best_result.unwrap();
            let decoded = decode(&result.token_ids);

            eprintln!("[{label}] {max_tok} tokens:");
            eprintln!("  Prompt ({} tok): \"{prompt}\"", result.prompt_tokens);
            eprintln!(
                "  Generated: {} tokens @ {:.1} tok/s (best of {n_runs})",
                result.generated_tokens, best_tps
            );
            eprintln!(
                "  Token IDs: {:?}",
                &result.token_ids[..result.generated_tokens.min(10)]
            );
            eprintln!("  Decoded: \"{}\"", &decoded[..decoded.len().min(120)]);
            eprintln!();
        }
    }

    // === Throughput vs sequence position ===
    eprintln!("--- Throughput vs Sequence Position (50 tokens) ---");
    metal.reset_state();
    std::env::set_var("LATTICE_PROFILE", "1");
    let gen_cfg_50 = GenerateConfig {
        max_new_tokens: 50,
        temperature: 0.0,
        top_k: 1,
        seed: Some(630),
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let result = metal.generate("The capital of France is", &tokenizer, &gen_cfg_50);
    let total_ms = t.elapsed().as_secs_f64() * 1000.0;
    let avg_tps = result.generated_tokens as f64 / (total_ms / 1000.0);
    eprintln!(
        "[summary] {} tokens in {:.1}ms = {:.1} tok/s average",
        result.generated_tokens, total_ms, avg_tps
    );

    // ── Hidden-readback overhead measurement ─────────────────────────────────
    // TODO(i2): When MetalQwen35State::forward_step_with_hidden and
    //   forward_prefill_with_hidden are implemented, replace this section with:
    //
    //   let n_measure = 5;
    //   let prompt_ids: Vec<u32> = (0..8).collect(); // fixed short prompt
    //
    //   // Measure forward_prefill overhead
    //   metal.reset_state();
    //   let t0 = std::time::Instant::now();
    //   for _ in 0..n_measure { metal.reset_state(); let _ = metal.forward_prefill(&prompt_ids); }
    //   let t_prefill_us = t0.elapsed().as_secs_f64() * 1e6 / n_measure as f64;
    //
    //   metal.reset_state();
    //   let t1 = std::time::Instant::now();
    //   for _ in 0..n_measure { metal.reset_state(); let _ = metal.forward_prefill_with_hidden(&prompt_ids); }
    //   let t_prefill_hidden_us = t1.elapsed().as_secs_f64() * 1e6 / n_measure as f64;
    //
    //   // Measure forward_step overhead
    //   let t2 = std::time::Instant::now();
    //   for _ in 0..n_measure { let _ = metal.forward_step(42, 1); }
    //   let t_step_us = t2.elapsed().as_secs_f64() * 1e6 / n_measure as f64;
    //
    //   let t3 = std::time::Instant::now();
    //   for _ in 0..n_measure { let _ = metal.forward_step_with_hidden(42, 1); }
    //   let t_step_hidden_us = t3.elapsed().as_secs_f64() * 1e6 / n_measure as f64;
    //
    //   eprintln!("\npath,tok_s,hidden_readback_us,accepted_tokens_per_forward,acceptance_rate");
    //   eprintln!("greedy,{:.1},0.0,1.00,1.00", avg_tps);
    //   eprintln!("prefill_overhead,n/a,{:.1},n/a,n/a", t_prefill_hidden_us - t_prefill_us);
    //   eprintln!("step_overhead,n/a,{:.1},n/a,n/a", t_step_hidden_us - t_step_us);
    //   for dl in [2u32, 4, 8] {
    //       eprintln!("mtp_draft_{dl},todo,{:.1},todo,todo", t_step_hidden_us - t_step_us);
    //   }
    eprintln!(
        "\nMTP real-model benchmark skipped: \
         model has no loaded MTP weights or Metal MTP path is disabled"
    );
    eprintln!("  (Wire forward_step_with_hidden + forward_prefill_with_hidden in i2 to enable)");
}

//! Metal GPU benchmark — proper throughput measurement with multiple prompts.
//!
//! Usage:
//! `cargo run --release -p lattice-inference --example profile_metal --features "f16,metal-gpu"`
//!
//! Set `LATTICE_MODEL_DIR` to a safetensors or native Q4 model directory.
//! For a Q4 checkpoint whose tokenizer lives elsewhere, set
//! `LATTICE_TOKENIZER_DIR` to that tokenizer directory.

fn main() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-2b"));
    let tokenizer_dir =
        std::env::var("LATTICE_TOKENIZER_DIR").unwrap_or_else(|_| model_dir.clone());
    let dir = std::path::Path::new(&model_dir);
    let tokenizer_path = std::path::Path::new(&tokenizer_dir).join("tokenizer.json");

    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::model_format::{ModelFormat, detect_format};
    use lattice_inference::tokenizer::bpe::BpeTokenizer;

    let _gpu_guard = lattice_inference::measurement::gpu_test_lock();
    eprintln!("[bench] Loading model...");
    let t0 = std::time::Instant::now();
    let (mut metal, quant_label) = match detect_format(dir) {
        ModelFormat::Q4 => {
            let cfg = Qwen35Config::from_model_dir(dir).expect("load Q4 model config");
            let state = MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg, 4096)
                .expect("initialize Q4 Metal state");
            (state, "Q4_0")
        }
        ModelFormat::Safetensors => {
            let model = Qwen35Model::from_safetensors(dir).expect("load safetensors model");
            let state = MetalQwen35State::new(model.weights(), model.config(), 4096)
                .expect("initialize Q8 Metal state");
            (state, "Q8_0")
        }
        ModelFormat::Unknown => {
            eprintln!("Unrecognized model directory at {model_dir}");
            std::process::exit(1);
        }
    };
    let tokenizer =
        BpeTokenizer::from_tokenizer_json(&tokenizer_path).expect("load tokenizer.json");
    eprintln!("[bench] Model loaded in {:.1}s", t0.elapsed().as_secs_f64());
    eprintln!(
        "[bench] Metal format={quant_label}; MTP weights loaded={}",
        metal.has_mtp()
    );

    // Build reverse vocab using tokenizer's internal mapping
    let decode = |ids: &[u32]| -> String {
        use lattice_inference::tokenizer::bpe::byte_decode_token;
        ids.iter()
            .filter_map(|id| tokenizer.token_for_id(*id))
            .map(byte_decode_token)
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
    eprintln!("  BENCHMARK: Qwen3.5 {quant_label} Metal GPU (single-cmd pipeline)");
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
                let result = metal
                    .generate(prompt, &tokenizer, &gen_cfg)
                    .expect("generation failed");
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
    // SAFETY: single-threaded example; no other thread reads the environment
    // concurrently at this point (set before the generation call below).
    unsafe {
        std::env::set_var("LATTICE_PROFILE", "1");
    }
    let gen_cfg_50 = GenerateConfig {
        max_new_tokens: 50,
        temperature: 0.0,
        top_k: 1,
        seed: Some(630),
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let result = metal
        .generate("The capital of France is", &tokenizer, &gen_cfg_50)
        .expect("generation failed");
    let total_ms = t.elapsed().as_secs_f64() * 1000.0;
    let avg_tps = result.generated_tokens as f64 / (total_ms / 1000.0);
    eprintln!(
        "[summary] {} tokens in {:.1}ms = {:.1} tok/s average",
        result.generated_tokens, total_ms, avg_tps
    );

    // Keep the explicit readback measurements off the verbose per-step profile path.
    // SAFETY: single-threaded example; no other thread reads the environment.
    unsafe {
        std::env::remove_var("LATTICE_PROFILE");
    }

    // ── Hidden-readback overhead measurement ─────────────────────────────────
    let n_measure = 5usize;
    let prompt_ids: Vec<u32> = (0..8).collect();

    let measure_prefill = |state: &mut MetalQwen35State, capture_hidden: bool| -> f64 {
        let start = std::time::Instant::now();
        for _ in 0..n_measure {
            state.reset_state();
            if capture_hidden {
                let output = state
                    .forward_prefill_with_hidden(&prompt_ids)
                    .expect("hidden-returning prefill");
                std::hint::black_box(output);
            } else {
                std::hint::black_box(state.forward_prefill(&prompt_ids));
            }
        }
        start.elapsed().as_secs_f64() * 1e6 / n_measure as f64
    };

    let measure_steps = |state: &mut MetalQwen35State, capture_hidden: bool, steps: usize| -> f64 {
        let mut elapsed = std::time::Duration::ZERO;
        for _ in 0..n_measure {
            state.reset_state();
            std::hint::black_box(state.forward_prefill(&[1]));
            let start = std::time::Instant::now();
            for offset in 0..steps {
                if capture_hidden {
                    let output = state
                        .forward_step_with_hidden(42, 1 + offset)
                        .expect("hidden-returning step");
                    std::hint::black_box(output);
                } else {
                    std::hint::black_box(state.forward_step(42, 1 + offset));
                }
            }
            elapsed += start.elapsed();
        }
        elapsed.as_secs_f64() * 1e6 / n_measure as f64
    };

    let prefill_us = measure_prefill(&mut metal, false);
    let prefill_hidden_us = measure_prefill(&mut metal, true);
    let step_us = measure_steps(&mut metal, false, 1);
    let step_hidden_us = measure_steps(&mut metal, true, 1);

    eprintln!("\npath,tok_s,hidden_readback_us,accepted_tokens_per_forward,acceptance_rate");
    eprintln!("greedy,{avg_tps:.1},0.0,1.00,1.00");
    eprintln!(
        "prefill_overhead,n/a,{:.1},n/a,n/a",
        prefill_hidden_us - prefill_us
    );
    eprintln!("step_overhead,n/a,{:.1},n/a,n/a", step_hidden_us - step_us);
    for draft_len in [2usize, 4, 8] {
        let baseline_us = measure_steps(&mut metal, false, draft_len);
        let with_hidden_us = measure_steps(&mut metal, true, draft_len);
        let tok_s = draft_len as f64 * 1e6 / with_hidden_us;
        eprintln!(
            "mtp_draft_{draft_len},{tok_s:.1},{:.1},n/a,n/a",
            with_hidden_us - baseline_us
        );
    }
    eprintln!(
        "# MTP weights loaded={}; mtp_draft_* isolates explicit target-hidden readback \
         for each requested draft span without changing the live MTP policy",
        metal.has_mtp()
    );
}

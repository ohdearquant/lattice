//! Apples-to-apple decode benchmark (A/B slope method).
//!
//! Runs the REAL Metal e2e path (`MetalQwen35State::chat_completion`, identical
//! to `chat_metal`) at two token counts for a fixed prompt and prints total
//! wall time for each. The harness computes the prefill-canceling slope
//!   decode_tok_per_s = (N2 - N1) / (T(N2) - T(N1))
//! which is methodology-identical to how MLX and Ollama are measured, so the
//! comparison is fair (prefill, model load, and fixed per-call overhead cancel).
//!
//! Env:
//!   LATTICE_MODEL_DIR   model dir (default ~/.lattice/models/qwen3.5-0.8b)
//!   BENCH_N             generated tokens for this run (required)
//!   BENCH_RUNS          repetitions (default 5); median reported by harness
//!
//! Output (one line per run, stderr-free for easy parsing):
//!   RESULT n_req=<N> completion=<actual> total_ms=<f>

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("Requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        if let Err(e) = run() {
            eprintln!("bench_decode_ab failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::forward::metal_qwen35::{ChatMessage, MetalQwen35State};
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::tokenizer::{BpeTokenizer, Tokenizer};

    let home = std::env::var("HOME")?;
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
    let dir = std::path::Path::new(&model_dir_str);

    let n: usize = std::env::var("BENCH_N")
        .expect("BENCH_N required")
        .parse()
        .expect("BENCH_N must be a positive integer");
    let runs: usize = std::env::var("BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    // Detect Q4 dir (no model.safetensors, has .q4 files) — same dispatch as chat_metal.
    let is_q4_dir = !dir.join("model.safetensors").exists()
        && std::fs::read_dir(dir)
            .ok()
            .and_then(|mut entries| {
                entries.find(|e| {
                    e.as_ref()
                        .ok()
                        .and_then(|e| e.file_name().to_str().map(|n| n.ends_with(".q4")))
                        .unwrap_or(false)
                })
            })
            .is_some();

    let tokenizer_dir_str =
        std::env::var("LATTICE_TOKENIZER_DIR").unwrap_or_else(|_| model_dir_str.clone());
    let tokenizer_dir = std::path::Path::new(&tokenizer_dir_str);

    eprintln!(
        "[bench] loading {model_dir_str} ({})",
        if is_q4_dir { "Q4" } else { "safetensors" }
    );

    // Two ownership paths: Q4 owns a separately-loaded tokenizer; safetensors
    // borrows the tokenizer from Qwen35Model. We materialize a single owned
    // BpeTokenizer either way to keep the rest of the bench branch-free.
    let mut metal: MetalQwen35State;
    let tokenizer: BpeTokenizer;

    if is_q4_dir {
        let cfg = if dir.join("config.json").exists() {
            Qwen35Config::from_config_json(&dir.join("config.json"))
                .map_err(|e| format!("config.json parse: {e}"))?
        } else {
            Qwen35Config::qwen35_0_8b()
        };
        metal =
            MetalQwen35State::from_q4_dir(dir, &tokenizer_dir.join("tokenizer.json"), &cfg, 4096)
                .map_err(|e| format!("Metal Q4 init: {e}"))?;
        tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_dir.join("tokenizer.json"))?;
    } else {
        let model = Qwen35Model::from_safetensors(dir).expect("load model");
        let cfg = model.config().clone();
        metal = MetalQwen35State::new(model.weights(), &cfg, 4096).expect("init metal");
        tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_dir.join("tokenizer.json"))?;
    }

    // Greedy, deterministic, no thinking — closest to raw continuation that
    // MLX/Ollama generate. Decode cost per token is what we measure; the
    // slope cancels prompt-format / prefill differences vs the other engines.
    let gen_cfg = GenerateConfig {
        max_new_tokens: n,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: None,
        grammar: None,
        stop_strings: vec![],
    };

    // Same continuation prompt as the original bench, single user turn.
    // Optionally pad to a target context length (BENCH_PROMPT_TOKENS) by
    // repeating filler text, so we can measure decode at real context depth
    // (agentic workload: long context + short response).
    let base = "The quick brown fox jumps over the lazy dog. \
                Once upon a time in a land far away, there lived a wise old owl \
                who knew many secrets. Every morning the sun rose over the \
                mountains and cast long shadows across the quiet valley. ";
    let prompt: String = match std::env::var("BENCH_PROMPT_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(target) if target > 0 => {
            let mut p = String::new();
            while tokenizer.tokenize(&p).real_length < target {
                p.push_str(base);
            }
            p
        }
        _ => base.to_string(),
    };
    eprintln!(
        "[bench] prompt_tokens={}",
        tokenizer.tokenize(&prompt).real_length
    );
    let history = vec![ChatMessage::user(&prompt)];

    // One warmup (not recorded).
    metal.reset_state();
    let _ = metal.chat_completion(&history, &tokenizer, &gen_cfg);

    for _ in 0..runs {
        metal.reset_state();
        let t = std::time::Instant::now();
        let result = metal.chat_completion(&history, &tokenizer, &gen_cfg);
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "RESULT n_req={} completion={} total_ms={:.3}",
            n, result.completion_tokens, total_ms
        );
    }
    Ok(())
}

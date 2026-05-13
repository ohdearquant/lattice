//! Qwen3.5-2B text generation demo.
//!
//! Usage: cargo run --release --bin qwen35_generate -- [--prompt "Hello"] [--max-tokens 64] [--seed 42]

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("What is the meaning of life?");

    let max_tokens: usize = args
        .iter()
        .position(|a| a == "--max-tokens")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let seed: Option<u64> = args
        .iter()
        .position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    let temperature: Option<f32> = args
        .iter()
        .position(|a| a == "--temperature")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    let model_dir = std::env::var("LATTICE_MODEL_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME not set");
            PathBuf::from(home).join(".lattice").join("models")
        })
        .join("qwen3.5-2b");

    println!("Loading Qwen3.5-2B from {model_dir:?}...");
    let t0 = Instant::now();

    let model = match lattice_inference::model::qwen35::Qwen35Model::from_safetensors(&model_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            std::process::exit(1);
        }
    };

    let load_ms = t0.elapsed().as_millis();
    println!("Model loaded in {load_ms}ms\n");

    let mut gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
        max_new_tokens: max_tokens,
        seed,
        ..Default::default()
    };
    if let Some(t) = temperature {
        gen_cfg.temperature = t;
    }

    println!("Prompt: {prompt}");
    println!(
        "Config: temp={}, top_k={}, top_p={}, rep_penalty={}, seed={:?}",
        gen_cfg.temperature, gen_cfg.top_k, gen_cfg.top_p, gen_cfg.repetition_penalty, gen_cfg.seed
    );
    println!("Generating up to {max_tokens} tokens...\n");

    let t1 = Instant::now();
    match model.generate(prompt, &gen_cfg) {
        Ok(output) => {
            let gen_ms = t1.elapsed().as_millis();
            let tokens_per_sec = if gen_ms > 0 {
                output.generated_tokens as f64 / (gen_ms as f64 / 1000.0)
            } else {
                0.0
            };

            println!("--- Generated Text ---");
            println!("{}", output.text);
            println!("--- Stats ---");
            println!("Token IDs: {:?}", output.token_ids);
            println!("Prompt tokens:    {}", output.prompt_tokens);
            println!("Generated tokens: {}", output.generated_tokens);
            println!("Time:             {gen_ms}ms");
            println!("Speed:            {tokens_per_sec:.1} tok/s");
        }
        Err(e) => {
            eprintln!("Generation failed: {e}");
            std::process::exit(1);
        }
    }
}

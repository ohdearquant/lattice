//! Qwen3.5-2B generation with LoRA adapter — proves MLX-trained adapters work in Rust inference.
//!
//! Usage:
//!   cargo run --release -p lattice-tune --features "safetensors,inference-hook" \
//!     --bin generate_lora -- \
//!     --lora platform/tune/output/mlx-qwen35-2b-code-r16/adapter_exported.safetensors \
//!     --prompt "Write a Rust function that checks if a number is prime" \
//!     --max-tokens 64

use std::path::PathBuf;
use std::time::Instant;

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let prompt =
        parse_arg(&args, "--prompt").unwrap_or_else(|| "What is the meaning of life?".to_string());

    let max_tokens: usize = parse_arg(&args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let seed: Option<u64> = parse_arg(&args, "--seed").and_then(|s| s.parse().ok());

    let lora_path: Option<PathBuf> = parse_arg(&args, "--lora").map(PathBuf::from);

    let model_dir = std::env::var("LATTICE_MODEL_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME not set");
            PathBuf::from(home).join(".lattice").join("models")
        })
        .join("qwen3.5-2b");

    // Load base model
    println!("Loading Qwen3.5-2B from {:?}...", model_dir);
    let t0 = Instant::now();

    let mut model =
        match lattice_inference::model::qwen35::Qwen35Model::from_safetensors(&model_dir) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to load model: {e}");
                std::process::exit(1);
            }
        };

    println!("Model loaded in {}ms", t0.elapsed().as_millis());

    // Load LoRA adapter
    if let Some(ref lora) = lora_path {
        println!("Loading LoRA adapter from {:?}...", lora);
        let t_lora = Instant::now();

        match lattice_tune::lora::LoraAdapter::from_safetensors(lora) {
            Ok(adapter) => {
                println!(
                    "  Adapter: {} pairs, rank={}, scale={:.2}, {} parameters",
                    adapter.num_adapted_layers(),
                    adapter.config.rank,
                    adapter.config.scale(),
                    adapter.num_parameters()
                );
                model.set_lora(Box::new(adapter));
                println!(
                    "  LoRA active (loaded in {}ms)",
                    t_lora.elapsed().as_millis()
                );
            }
            Err(e) => {
                eprintln!("Failed to load LoRA adapter: {e}");
                std::process::exit(1);
            }
        }
    } else {
        println!("No LoRA adapter (base model only)");
    }

    // Configure generation
    let mut gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig::default();
    gen_cfg.max_new_tokens = max_tokens;
    gen_cfg.seed = seed;

    println!("\nPrompt: {prompt}");
    println!(
        "Config: temp={}, top_k={}, seed={:?}, max_tokens={}",
        gen_cfg.temperature, gen_cfg.top_k, gen_cfg.seed, max_tokens
    );
    println!("Generating...\n");

    // Generate
    let t1 = Instant::now();
    match model.generate(&prompt, &gen_cfg) {
        Ok(output) => {
            let gen_ms = t1.elapsed().as_millis();
            let tok_s = if gen_ms > 0 {
                output.generated_tokens as f64 / (gen_ms as f64 / 1000.0)
            } else {
                0.0
            };

            println!("--- Output ---");
            println!("{}", output.text);
            println!("--- Stats ---");
            println!(
                "Tokens: {} prompt + {} generated in {}ms ({:.1} tok/s)",
                output.prompt_tokens, output.generated_tokens, gen_ms, tok_s
            );
            if lora_path.is_some() {
                println!("LoRA: ACTIVE");
            }
        }
        Err(e) => {
            eprintln!("Generation failed: {e}");
            std::process::exit(1);
        }
    }
}

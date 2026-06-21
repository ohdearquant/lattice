//! Qwen3.5 generation with LoRA adapter — proves PEFT/MLX-trained adapters work in Rust inference.
//!
//! Usage:
//!   cargo run --release -p lattice-tune --features "safetensors,inference-hook" \
//!     --bin generate_lora -- \
//!     --model-dir ~/.lattice/models/qwen3.5-0.8b \
//!     --lora adapter.safetensors \
//!     --prompt "Write a Rust function that checks if a number is prime" \
//!     --max-tokens 64
//!     [--json]   Emit @@lattice gen_token events for the Lattice Studio app.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// Escape a string as a JSON string literal (including surrounding double quotes).
/// Does NOT depend on serde_json — this is a self-contained, correct escaper.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                // Other ASCII control characters: \u00XX
                let code = c as u32;
                out.push_str(&format!("\\u{code:04x}"));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn default_model_cache() -> PathBuf {
    std::env::var("LATTICE_MODEL_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME not set");
            PathBuf::from(home).join(".lattice").join("models")
        })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let prompt =
        parse_arg(&args, "--prompt").unwrap_or_else(|| "What is the meaning of life?".to_string());

    let max_tokens: usize = parse_arg(&args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let seed: Option<u64> = parse_arg(&args, "--seed").and_then(|s| s.parse().ok());

    let temperature: Option<f32> = parse_arg(&args, "--temperature").and_then(|s| s.parse().ok());

    let lora_path: Option<PathBuf> = parse_arg(&args, "--lora").map(PathBuf::from);

    let emit_json = parse_flag(&args, "--json");

    let model_dir = if let Some(dir) = parse_arg(&args, "--model-dir") {
        PathBuf::from(dir)
    } else {
        let model_name = parse_arg(&args, "--model").unwrap_or_else(|| "qwen3.5-0.8b".to_string());
        default_model_cache().join(model_name)
    };

    // Load base model
    println!("Loading model from {model_dir:?}...");
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
        println!("Loading LoRA adapter from {lora:?}...");
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
                if let Err(e) = adapter.validate_against(model.config()) {
                    eprintln!("LoRA adapter incompatible with loaded model: {e}");
                    std::process::exit(1);
                }
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
    let mut gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
        max_new_tokens: max_tokens,
        seed,
        ..lattice_inference::model::qwen35_config::GenerateConfig::default()
    };
    if let Some(t) = temperature {
        gen_cfg.temperature = t;
    }

    println!("\nPrompt: {prompt}");
    println!(
        "Config: temp={}, top_k={}, seed={:?}, max_tokens={}",
        gen_cfg.temperature, gen_cfg.top_k, gen_cfg.seed, max_tokens
    );
    println!("Generating...\n");

    let t1 = Instant::now();

    if emit_json {
        // Streaming JSON mode: emit @@lattice gen_token events live.
        let mut stdout = std::io::stdout();
        let mut first_token_emitted = false;
        let mut ttft_ms: f64 = 0.0;

        let result = model.generate_streaming(&prompt, &gen_cfg, |delta| {
            if !first_token_emitted {
                ttft_ms = t1.elapsed().as_secs_f64() * 1000.0;
                first_token_emitted = true;
            }
            let token_json = json_escape(delta);
            writeln!(
                stdout,
                "@@lattice {{\"ev\":\"gen_token\",\"token\":{token_json},\"done\":false}}"
            )
            .ok();
            stdout.flush().ok();
        });

        match result {
            Ok(output) => {
                let gen_ms = t1.elapsed().as_millis();
                let tok_s = if gen_ms > 0 {
                    output.generated_tokens as f64 / (gen_ms as f64 / 1000.0)
                } else {
                    0.0
                };
                // Final done event with stats.
                writeln!(
                    stdout,
                    "@@lattice {{\"ev\":\"gen_token\",\"token\":\"\",\"done\":true,\"tok_s\":{tok_s:.1},\"ttft_ms\":{ttft_ms:.1}}}"
                )
                .ok();
                stdout.flush().ok();
            }
            Err(e) => {
                eprintln!("Generation failed: {e}");
                std::process::exit(1);
            }
        }
    } else {
        // Non-JSON mode: original atomic output block, unchanged.
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
}

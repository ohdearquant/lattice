/// Interactive chat with Qwen3.5-2B or Qwen3.6-27B (Q4) on Metal GPU.
///
/// # Usage
///
/// 2B (default):
/// ```
/// cargo run --release -p lattice-inference --bin chat_metal --features "f16,metal-gpu"
/// ```
///
/// 27B Q4:
/// ```
/// LATTICE_MODEL_DIR=~/.lattice/models/qwen3.6-27b-q4 \
/// LATTICE_TOKENIZER_DIR=~/.lattice/models/qwen3.6-27b \
/// cargo run --release -p lattice-inference --bin chat_metal --features "f16,metal-gpu"
/// ```

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("Requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        if let Err(e) = run_chat() {
            eprintln!("chat_metal failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_chat() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::forward::metal_qwen35::{ChatMessage, MetalQwen35State};
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::tokenizer::bpe::BpeTokenizer;
    use std::io::{self, BufRead, Write};

    let home = std::env::var("HOME")?;

    // LATTICE_MODEL_DIR overrides the default 2B model directory.
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-2b"));
    let dir = std::path::Path::new(&model_dir_str);

    // Determine if this is a Q4 directory (has .q4 files but no model.safetensors).
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

    // LATTICE_TOKENIZER_DIR overrides where the tokenizer.json is read from.
    let tokenizer_dir_str =
        std::env::var("LATTICE_TOKENIZER_DIR").unwrap_or_else(|_| model_dir_str.clone());
    let tokenizer_dir = std::path::Path::new(&tokenizer_dir_str);

    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_dir.join("tokenizer.json"))
        .map_err(|e| {
            format!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_dir.display()
            )
        })?;

    let mut metal;
    let model_tag;

    if is_q4_dir {
        // Q4 direct-load path (for 27B or any Q4-quantized model).
        eprintln!("[chat] Detected Q4 model directory: {}", dir.display());

        // Load config: try config.json in the Q4 dir first; fall back to qwen36_27b preset.
        let cfg = if dir.join("config.json").exists() {
            eprintln!("[chat] Loading config from {}/config.json", dir.display());
            Qwen35Config::from_config_json(&dir.join("config.json"))
                .map_err(|e| format!("failed to parse config.json: {e}"))?
        } else {
            eprintln!("[chat] No config.json found; using qwen36_27b preset");
            Qwen35Config::qwen36_27b()
        };

        model_tag = format!(
            "Q4 {}B ({} layers, hidden {})",
            cfg.vocab_size / 1_000,
            cfg.num_hidden_layers,
            cfg.hidden_size,
        );

        eprintln!("[chat] Loading Q4 model from {}...", dir.display());
        let t0 = std::time::Instant::now();
        metal =
            MetalQwen35State::from_q4_dir(dir, &tokenizer_dir.join("tokenizer.json"), &cfg, 4096)
                .map_err(|e| format!("failed to initialize Metal from Q4 dir: {e}"))?;
        eprintln!(
            "[chat] Q4 model loaded in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
    } else {
        // Existing safetensors path (2B model).
        if !dir.join("model.safetensors").exists() {
            eprintln!("Model not found at {model_dir_str}");
            std::process::exit(1);
        }

        eprintln!("[chat] Loading Qwen3.5-2B from {}...", dir.display());
        let t0 = std::time::Instant::now();
        let model = Qwen35Model::from_safetensors(dir).expect("load model");
        let cfg = model.config().clone();
        eprintln!("[chat] Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

        model_tag = format!("Qwen3.5-2B (hidden {})", cfg.hidden_size);

        eprintln!("[chat] Initializing Metal GPU...");
        let t1 = std::time::Instant::now();
        metal = MetalQwen35State::new(model.weights(), &cfg, 4096).expect("init metal");
        eprintln!("[chat] Metal ready in {:.1}s", t1.elapsed().as_secs_f64());
    }

    let gen_cfg = GenerateConfig {
        max_new_tokens: 512,
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        repetition_penalty: 1.1,
        seed: None,
        stop_token_ids: vec![],
        enable_thinking: true,
    };

    let system_msg = ChatMessage::system("You are a helpful assistant. Be concise and direct.");

    eprintln!("\n=== {model_tag} Chat (Metal GPU, Q4) ===");
    eprintln!("Type your message. Empty line or Ctrl-D to quit.\n");

    let stdin = io::stdin();
    let mut history: Vec<ChatMessage> = vec![system_msg];

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }

        let input = line.trim();
        if input.is_empty() {
            break;
        }

        history.push(ChatMessage::user(input));
        metal.reset_state();

        let t = std::time::Instant::now();
        let result = metal.chat_completion(&history, &tokenizer, &gen_cfg);
        let elapsed = t.elapsed();

        let text = result.message.content.trim().to_string();
        println!("{text}");

        let tps = if result.completion_tokens > 0 {
            result.completion_tokens as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        eprintln!(
            "[{} prompt + {} completion in {:.1}ms = {:.1} tok/s]",
            result.prompt_tokens,
            result.completion_tokens,
            elapsed.as_secs_f64() * 1000.0,
            tps,
        );

        history.push(ChatMessage::assistant(text));
    }

    Ok(())
}

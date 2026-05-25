//! Logit dump binary for divergence analysis vs MLX.
//!
//! Loads a safetensors Qwen3.5-0.8B model, runs prefill on a token sequence,
//! and writes every position's full logit vector (f32 LE) to a binary file.
//!
//! Output format: seq_len × vocab_size f32 values, row-major (position-major).
//!
//! Env:
//!   LATTICE_MODEL_DIR  model dir (default ~/.lattice/models/qwen3.5-0.8b)
//!   LATTICE_LOGIT_OUT  output binary file path (default /tmp/lattice_logits.bin)
//!   LATTICE_TOKENS     space-separated token IDs (required)

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("bench_logit_dump requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        if let Err(e) = run() {
            eprintln!("bench_logit_dump failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::Qwen35Model;

    let home = std::env::var("HOME")?;
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
    let out_path = std::env::var("LATTICE_LOGIT_OUT")
        .unwrap_or_else(|_| "/tmp/lattice_logits.bin".to_string());
    let tokens_str = std::env::var("LATTICE_TOKENS")
        .map_err(|_| "LATTICE_TOKENS env var required (space-separated token IDs)")?;

    let tokens: Vec<u32> = tokens_str
        .split_whitespace()
        .map(|s| s.parse::<u32>().expect("token must be u32"))
        .collect();

    if tokens.is_empty() {
        return Err("LATTICE_TOKENS must not be empty".into());
    }

    let dir = std::path::Path::new(&model_dir_str);
    let tok_dir_str =
        std::env::var("LATTICE_TOKENIZER_DIR").unwrap_or_else(|_| model_dir_str.clone());
    let tok_dir = std::path::Path::new(&tok_dir_str);
    eprintln!("[bench_logit_dump] loading {model_dir_str}");
    eprintln!("[bench_logit_dump] tokens: {} positions", tokens.len());

    let is_q4_dir = !dir.join("model.safetensors").exists()
        && std::fs::read_dir(dir)
            .ok()
            .and_then(|mut e| {
                e.find(|e| {
                    e.as_ref()
                        .ok()
                        .and_then(|e| e.file_name().to_str().map(|n| n.ends_with(".q4")))
                        .unwrap_or(false)
                })
            })
            .is_some();

    let (mut metal, cfg) = if is_q4_dir {
        let cfg = lattice_inference::model::qwen35_config::Qwen35Config::from_config_json(
            &dir.join("config.json"),
        )
        .map_err(|e| format!("config.json: {e}"))?;
        let state = MetalQwen35State::from_q4_dir(dir, &tok_dir.join("tokenizer.json"), &cfg, 4096)
            .map_err(|e| format!("from_q4_dir: {e}"))?;
        (state, cfg)
    } else {
        let model = Qwen35Model::from_safetensors(dir).map_err(|e| format!("load model: {e}"))?;
        let cfg = model.config().clone();
        let state = MetalQwen35State::new(model.weights(), &cfg, 4096)
            .map_err(|e| format!("MetalQwen35State::new: {e}"))?;
        (state, cfg)
    };

    eprintln!("[bench_logit_dump] running forward_prefill_all_logits...");
    let logits_flat = metal.forward_prefill_all_logits(&tokens);

    let vocab = cfg.vocab_size;
    let n_pos = tokens.len();
    let expected = n_pos * vocab;
    if logits_flat.len() != expected {
        return Err(format!(
            "expected {} f32 values ({n_pos} × {vocab}), got {}",
            expected,
            logits_flat.len()
        )
        .into());
    }

    eprintln!("[bench_logit_dump] writing {n_pos}×{vocab} f32 → {out_path}");

    let bytes: Vec<u8> = logits_flat.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&out_path, &bytes)?;

    // Echo vocab size and n_pos to stdout for the Python caller to parse.
    println!("VOCAB={vocab}");
    println!("NPOS={n_pos}");
    println!("OUT={out_path}");
    Ok(())
}

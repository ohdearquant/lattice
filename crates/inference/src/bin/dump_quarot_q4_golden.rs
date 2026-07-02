//! QuaRot+Q4 composed-path golden dumper (issue #320).
//!
//! Loads a QuaRot-rotated Q4 artifact directory (as produced by
//! `quantize_quarot`) through the real Metal runtime loader
//! ([`MetalQwen35State::from_q4_dir`]), runs greedy decode for a fixed set
//! of prompts, and prints the resulting token IDs as JSON matching the
//! `quarot_q4_composed_v1` fixture schema consumed by
//! `crates/inference/tests/quarot_q4_composed_golden.rs`.
//!
//! This binary only computes and prints; it never writes the committed
//! fixture file itself. `scripts/gen_quarot_q4_composed_golden.py` owns the
//! freeze/overwrite policy and redirects this binary's stdout to the
//! fixture path.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --bin dump_quarot_q4_golden --features "f16,metal-gpu" -- \
//!   --q4-dir target/quarot-q4-golden/qwen3.5-0.8b-quarot-q4 \
//!   --tokenizer-dir ~/.lattice/models/qwen3.5-0.8b \
//!   --model-id Qwen/Qwen3.5-0.8B \
//!   --quarot-seed 0xCAFE_BABE_DEAD_BEEF \
//!   --model-dir-default ~/.lattice/models/qwen3.5-0.8b \
//!   --converter target/release/quantize_quarot \
//!   --max-new-tokens 8 > golden.json
//! ```

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("dump_quarot_q4_golden requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        if let Err(e) = run() {
            eprintln!("dump_quarot_q4_golden failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::tokenizer::bpe::BpeTokenizer;
    use std::path::PathBuf;

    /// Fixed prompt set. Kept in sync by hand with the committed fixture —
    /// this binary is the source of truth for prompt *content*; the fixture
    /// is the frozen source of truth for expected *token IDs*.
    const PROMPTS: &[(&str, &str)] = &[
        ("short_factual", "The capital of France is"),
        (
            "code",
            "Write a Python function that returns the nth Fibonacci number:\n\ndef fibonacci(n):",
        ),
        (
            "structured_reasoning",
            "Question: A train travels 60 miles in 2 hours. What is its average speed?\nAnswer:",
        ),
    ];

    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    let q4_dir = PathBuf::from(get("--q4-dir").ok_or("--q4-dir is required")?);
    let tokenizer_dir = get("--tokenizer-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| q4_dir.clone());
    let model_id = get("--model-id").unwrap_or_else(|| "Qwen/Qwen3.5-0.8B".to_string());
    let model_dir_default =
        get("--model-dir-default").unwrap_or_else(|| "~/.lattice/models/qwen3.5-0.8b".to_string());
    let quarot_seed = get("--quarot-seed").unwrap_or_else(|| "0xCAFE_BABE_DEAD_BEEF".to_string());
    let converter =
        get("--converter").unwrap_or_else(|| "target/release/quantize_quarot".to_string());
    let max_new_tokens: usize = get("--max-new-tokens")
        .map(|s| s.parse().expect("--max-new-tokens must be usize"))
        .unwrap_or(8);
    let max_cache_len: usize = get("--max-cache-len")
        .map(|s| s.parse().expect("--max-cache-len must be usize"))
        .unwrap_or(4096);

    eprintln!("[dump_quarot_q4_golden] q4_dir={}", q4_dir.display());
    eprintln!(
        "[dump_quarot_q4_golden] tokenizer_dir={}",
        tokenizer_dir.display()
    );

    let cfg = Qwen35Config::from_config_json(&q4_dir.join("config.json"))
        .map_err(|e| format!("failed to parse {}/config.json: {e}", q4_dir.display()))?;
    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_dir.join("tokenizer.json"))
        .map_err(|e| {
            format!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_dir.display()
            )
        })?;

    eprintln!("[dump_quarot_q4_golden] loading Q4 artifact via from_q4_dir...");
    let mut metal = MetalQwen35State::from_q4_dir(
        &q4_dir,
        &tokenizer_dir.join("tokenizer.json"),
        &cfg,
        max_cache_len,
    )
    .map_err(|e| format!("from_q4_dir: {e}"))?;

    let gen_cfg = GenerateConfig {
        max_new_tokens,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
    };

    let mut cases = Vec::with_capacity(PROMPTS.len());
    for (name, prompt) in PROMPTS {
        eprintln!("[dump_quarot_q4_golden] generating case={name}");
        let out = metal.generate(prompt, &tokenizer, &gen_cfg);
        eprintln!(
            "[dump_quarot_q4_golden] case={name} generated_tokens={} stopped={} stop_reason={:?}",
            out.generated_tokens, out.stopped, out.stop_reason
        );
        cases.push(serde_json::json!({
            "name": name,
            "prompt": prompt,
            "expected_generated_ids": out.token_ids,
        }));
    }

    let golden = serde_json::json!({
        "schema_version": 1,
        "model_id": model_id,
        "model_dir_default": model_dir_default,
        "artifact_kind": "lattice-self-quarot-q4-greedy-token-golden",
        "quarot_seed": quarot_seed,
        "converter": converter,
        "max_new_tokens": max_new_tokens,
        "generation": {
            "temperature": gen_cfg.temperature,
            "top_k": gen_cfg.top_k,
            "top_p": gen_cfg.top_p,
            "repetition_penalty": gen_cfg.repetition_penalty,
            "enable_mtp": false,
            "enable_thinking": false,
            "stop_token_ids": Vec::<u32>::new(),
        },
        "cases": cases,
    });

    println!("{}", serde_json::to_string_pretty(&golden)?);
    Ok(())
}

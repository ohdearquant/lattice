//! Debug diagnostic for Qwen3.5-2B — checks each stage of the forward pass.

use lattice_inference::tokenizer::common::Tokenizer;
use std::path::PathBuf;

fn main() {
    let model_dir = std::env::var("LATTICE_MODEL_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME not set");
            PathBuf::from(home).join(".lattice").join("models")
        })
        .join("qwen3.5-2b");

    println!("Loading model...");
    let model = lattice_inference::model::qwen35::Qwen35Model::from_safetensors(&model_dir)
        .expect("Failed to load model");

    let _cfg = model.config();

    // Test tokenization
    let prompt = "The capital of France is";
    let tokenizer = model.tokenizer();
    let input = tokenizer.tokenize(prompt);
    let prompt_ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();
    println!("Prompt: {prompt:?}");
    println!("Token IDs: {prompt_ids:?}");
    println!("Expected:  [760, 6511, 314, 9338, 369]");

    // Run generate with temperature 0 (greedy)
    let gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
        max_new_tokens: 16,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };

    println!("\nGenerating (greedy, 16 tokens)...");
    match model.generate(prompt, &gen_cfg) {
        Ok(output) => {
            println!("Generated: {:?}", output.text);
            println!("Token IDs: {:?}", output.token_ids);
            println!("Prompt tokens: {}", output.prompt_tokens);
            println!("Generated tokens: {}", output.generated_tokens);

            // Decode each token
            for &tid in &output.token_ids {
                let tok = tokenizer.token_for_id(tid).unwrap_or("??");
                let decoded = lattice_inference::tokenizer::bpe::byte_decode_token(tok);
                println!("  {tid} -> {decoded:?}");
            }
        }
        Err(e) => {
            eprintln!("Generation failed: {e}");
        }
    }

    // Single token test — use forward_single_token_debug for per-layer stats
    println!("\n--- Single token 760 ('The') [per-layer stats to stderr] ---");
    let logits_single = model.forward_single_token_debug(760);
    let sm: f32 = logits_single.iter().sum::<f32>() / logits_single.len() as f32;
    let ss: f32 = (logits_single.iter().map(|x| (x - sm).powi(2)).sum::<f32>()
        / logits_single.len() as f32)
        .sqrt();
    println!("Logits: mean={sm:.6}, std={ss:.6}");
    println!("Python: mean=-2.140625, std=2.125000");
    let mut si: Vec<(usize, f32)> = logits_single
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    si.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("Top-5:");
    for (i, (tok, logit)) in si.iter().take(5).enumerate() {
        let name = tokenizer.token_for_id(*tok as u32).unwrap_or("??");
        let decoded = lattice_inference::tokenizer::bpe::byte_decode_token(name);
        println!("  {i}: token_id={tok}, logit={logit:.4}, decoded={decoded:?}");
    }
    println!("Python: 2614(' following')=12.69, 220(' ')=11.50, 7193(' purpose')=11.13");

    // Full prompt
    println!("\n--- Full prompt forward pass ---");
    let logits = model.forward_prompt_debug(&prompt_ids);
    let logit_mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    let logit_std: f32 =
        (logits.iter().map(|x| (x - logit_mean).powi(2)).sum::<f32>() / logits.len() as f32).sqrt();
    println!("Logits: mean={logit_mean:.6}, std={logit_std:.6}");
    println!("Python: mean=-0.085449, std=2.218750");

    // Top-10 tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("\nTop-10 tokens:");
    println!("Python top: 11751(' Paris')=18.875, 279(' the')=16.5, 264(' a')=16.25");
    for (i, (tok, logit)) in indexed.iter().take(10).enumerate() {
        let name = tokenizer.token_for_id(*tok as u32).unwrap_or("??");
        let decoded = lattice_inference::tokenizer::bpe::byte_decode_token(name);
        println!("  {i}: token_id={tok}, logit={logit:.4}, decoded={decoded:?}");
    }

    let nan_count = logits.iter().filter(|x| x.is_nan()).count();
    let inf_count = logits.iter().filter(|x| x.is_infinite()).count();
    if nan_count > 0 || inf_count > 0 {
        println!("WARNING: NaN={nan_count}, Inf={inf_count}");
    }
}

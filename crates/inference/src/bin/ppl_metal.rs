use lattice_inference::forward::metal_qwen35::MetalQwen35State;
use lattice_inference::model::qwen35::{PerplexityConfig, Qwen35Model};
use lattice_inference::tokenizer::{BpeTokenizer, Tokenizer};

fn main() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
    let dir = std::path::Path::new(&model_dir);
    let n_tokens: usize = std::env::var("PPL_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);

    eprintln!("[ppl_metal] loading {model_dir}");
    let model = Qwen35Model::from_safetensors(dir).expect("load model");
    let cfg = model.config().clone();
    let mut metal = MetalQwen35State::new(model.weights(), &cfg, 4096).expect("init metal");
    let tokenizer = BpeTokenizer::from_tokenizer_json(&dir.join("tokenizer.json")).unwrap();

    let corpus_path =
        std::env::var("CORPUS").unwrap_or_else(|_| "/tmp/wikitext2_test.txt".to_string());
    let corpus = std::fs::read_to_string(&corpus_path).expect("read corpus");
    let input = tokenizer.tokenize(&corpus);
    let all_tokens: Vec<u32> = input.input_ids[..input.real_length].to_vec();
    let tokens = &all_tokens[..all_tokens.len().min(n_tokens)];
    eprintln!(
        "[ppl_metal] scoring {} tokens (Metal GPU, Q8 + f16 lm_head)",
        tokens.len()
    );

    let t = std::time::Instant::now();
    let ppl_cfg = PerplexityConfig {
        window: 512,
        stride: 256,
    };
    let report = metal.compute_perplexity(tokens, &ppl_cfg).expect("ppl");
    let elapsed = t.elapsed();

    println!("PPL:     {:.4}", report.ppl);
    println!("NLL:     {:.6}", report.mean_nll);
    println!("Tokens:  {}", report.num_tokens_scored);
    println!("Time:    {:.1}s", elapsed.as_secs_f64());
}

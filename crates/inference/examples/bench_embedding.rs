//! Benchmark: Pure Rust inference vs ONNX Runtime for Qwen3-Embedding-0.6B.
//!
//! Usage:
//!   cargo run --release --features "f16,bench-ort" --bin bench_embedding

use std::time::Instant;

fn main() {
    let home = std::env::var("HOME").unwrap();

    let short = "hello world";
    let medium = "The Qwen3 embedding model uses a decoder-only transformer architecture with grouped query attention, rotary position embeddings, SwiGLU FFN, and RMS normalization for efficient multilingual text representation across 100+ languages.";
    let long_text = "The Qwen3 embedding model uses a decoder-only transformer architecture with grouped query attention and rotary position embeddings for efficient multilingual text representation. ".repeat(25);

    println!("=== Qwen3-Embedding-0.6B Benchmark ===\n");
    println!(
        "Texts: short={}ch, medium={}ch, long={}ch\n",
        short.len(),
        medium.len(),
        long_text.len()
    );

    // --- Pure Rust (lattice-inference) ---
    {
        let model_dir = format!("{home}/.lattice/models/qwen3-embedding-0.6b");
        let dir = std::path::Path::new(&model_dir);
        if dir.join("model.safetensors").exists() {
            println!("--- Pure Rust (Accelerate AMX) ---");
            let t0 = Instant::now();
            let model = lattice_inference::QwenModel::from_directory(dir).unwrap();
            println!("  Load: {:.0}ms", t0.elapsed().as_millis());

            let _ = model.encode("warmup").unwrap();

            for (label, text) in [
                ("short", short),
                ("medium", medium),
                ("long", long_text.as_str()),
            ] {
                let n = match label {
                    "short" => 10,
                    "medium" => 5,
                    _ => 3,
                };
                let t = Instant::now();
                for _ in 0..n {
                    let _ = model.encode(text).unwrap();
                }
                let ms = t.elapsed().as_millis() as f64 / n as f64;
                println!("  {label:8}: {ms:>8.1}ms");
            }
            println!();
        } else {
            println!("--- Pure Rust: SKIPPED (no safetensors) ---\n");
        }
    }

    // --- ONNX Runtime ---
    #[cfg(feature = "bench-ort")]
    {
        use ort::session::{Session, builder::GraphOptimizationLevel};

        let onnx_dir = format!("{home}/.lattice/models/qwen3-embedding-0.6b-onnx");
        let tokenizer_path = format!("{onnx_dir}/tokenizer.json");

        if !std::path::Path::new(&tokenizer_path).exists() {
            println!("--- ONNX: SKIPPED (no tokenizer.json) ---\n");
            return;
        }

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();

        for (tag, filename) in [
            ("ONNX-fp32", "model.onnx"),
            ("ONNX-q8", "model_quantized.onnx"),
        ] {
            let model_path = format!("{onnx_dir}/{filename}");
            if !std::path::Path::new(&model_path).exists() {
                println!("--- {tag}: SKIPPED (not downloaded) ---\n");
                continue;
            }

            println!("--- {tag} (CPU) ---");
            let t0 = Instant::now();
            let mut session = match Session::builder()
                .unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .unwrap()
                .with_intra_threads(4)
                .unwrap()
                .commit_from_file(&model_path)
            {
                Ok(s) => s,
                Err(e) => {
                    println!("  FAILED to load: {e}\n");
                    continue;
                }
            };
            println!("  Load: {:.0}ms", t0.elapsed().as_millis());

            // Print input/output names
            println!(
                "  inputs: {:?}",
                session
                    .inputs()
                    .iter()
                    .map(|i| i.name())
                    .collect::<Vec<_>>()
            );
            println!(
                "  outputs: {:?}",
                session
                    .outputs()
                    .iter()
                    .map(|o| o.name())
                    .collect::<Vec<_>>()
            );

            // Warmup
            run_onnx_encode(&mut session, &tokenizer, "warmup");

            for (label, text) in [
                ("short", short),
                ("medium", medium),
                ("long", long_text.as_str()),
            ] {
                let n = match label {
                    "short" => 10,
                    "medium" => 5,
                    _ => 3,
                };
                let t = Instant::now();
                for _ in 0..n {
                    run_onnx_encode(&mut session, &tokenizer, text);
                }
                let ms = t.elapsed().as_millis() as f64 / n as f64;
                println!("  {label:8}: {ms:>8.1}ms");
            }
            println!();
        }
    }

    #[cfg(not(feature = "bench-ort"))]
    {
        println!("--- ONNX: SKIPPED (compile with --features bench-ort) ---\n");
    }
}

#[cfg(feature = "bench-ort")]
fn run_onnx_encode(
    session: &mut ort::session::Session,
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
) -> Vec<f32> {
    let encoding = tokenizer.encode(text, true).unwrap();
    let ids = encoding.get_ids();
    let mask = encoding.get_attention_mask();
    let seq_len = ids.len();

    let input_ids: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    let attention_mask: Vec<i64> = mask.iter().map(|&x| x as i64).collect();

    let position_ids: Vec<i64> = (0..seq_len as i64).collect();

    let id_tensor =
        ort::value::Tensor::from_array((vec![1i64, seq_len as i64], input_ids)).unwrap();
    let mask_tensor =
        ort::value::Tensor::from_array((vec![1i64, seq_len as i64], attention_mask)).unwrap();
    let pos_tensor =
        ort::value::Tensor::from_array((vec![1i64, seq_len as i64], position_ids)).unwrap();

    let outputs = session
        .run(ort::inputs![id_tensor, mask_tensor, pos_tensor])
        .unwrap();

    // Last hidden state → last token → L2 normalize
    let output_view = outputs[0].try_extract_tensor::<f32>().unwrap();
    // Shape: [1, seq_len, hidden_size]
    let shape = output_view.0;
    let raw = output_view.1;
    let hidden_size = shape[2] as usize;
    let last_idx = seq_len - 1;
    let offset = last_idx * hidden_size;
    let mut embedding: Vec<f32> = raw[offset..offset + hidden_size].to_vec();

    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for v in &mut embedding {
            *v *= inv;
        }
    }
    embedding
}

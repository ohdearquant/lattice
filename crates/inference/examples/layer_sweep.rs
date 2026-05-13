//! Early-exit layer sweep: measure embedding quality at each layer exit point.
//!
//! Runs Qwen3-0.6B forward pass, captures embedding at each of the 28 layers,
//! and computes cosine similarity vs the full-model (layer 27) embedding.
//! This reveals the shallowest layer that produces acceptable embeddings.
//!
//! Usage:
//!   cargo run --release --features f16 --bin layer_sweep

use lattice_inference::QwenModel;
use std::time::Instant;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

fn main() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = format!("{home}/.lattice/models/qwen3-embedding-0.6b");
    let dir = std::path::Path::new(&model_dir);

    if !dir.join("model.safetensors").exists() {
        eprintln!("Model not found at {model_dir}");
        std::process::exit(1);
    }

    println!("Loading model...");
    let t0 = Instant::now();
    let model = QwenModel::from_directory(dir).unwrap();
    println!("Loaded in {:.0}ms\n", t0.elapsed().as_millis());

    let test_texts = [
        ("short_en", "hello world"),
        ("short_zh", "记得吃药"),
        (
            "medium",
            "The Qwen3 embedding model uses a decoder-only transformer architecture with grouped query attention and rotary position embeddings.",
        ),
        ("query", "what is quantum physics"),
        ("code", "fn main() { println!(\"hello\"); }"),
    ];

    // Also test retrieval quality: related pairs should have higher similarity than unrelated.
    let pairs = [
        ("记得吃药", "remember to take medicine", "quantum physics"),
        (
            "今天天气不错",
            "the weather is nice today",
            "machine learning",
        ),
        (
            "I want to eat hotpot",
            "我想吃火锅",
            "compiler optimization",
        ),
    ];

    println!("=== Per-Layer Cosine Similarity vs Layer 27 (full model) ===\n");
    println!(
        "{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "Layer", "short_en", "short_zh", "medium", "query", "code"
    );
    println!(
        "{:->6}  {:->10}  {:->10}  {:->10}  {:->10}  {:->10}",
        "", "", "", "", "", ""
    );

    // Get all-layer embeddings for each text.
    let all_embeddings: Vec<_> = test_texts
        .iter()
        .map(|(_, text)| model.encode_all_layers(text).unwrap())
        .collect();

    let num_layers = all_embeddings[0].len();

    for layer_idx in 0..num_layers {
        print!("{:>6}", layer_idx);
        for text_idx in 0..test_texts.len() {
            let layer_emb = &all_embeddings[text_idx][layer_idx];
            let full_emb = &all_embeddings[text_idx][num_layers - 1];
            let sim = cosine_sim(layer_emb, full_emb);
            print!("  {:>10.4}", sim);
        }
        println!();
    }

    // Find the earliest layer where all similarities > 0.95 and > 0.99.
    println!("\n=== Convergence Points ===\n");
    for threshold in [0.90, 0.95, 0.99] {
        let earliest = (0..num_layers).find(|&layer_idx| {
            (0..test_texts.len()).all(|text_idx| {
                let sim = cosine_sim(
                    &all_embeddings[text_idx][layer_idx],
                    &all_embeddings[text_idx][num_layers - 1],
                );
                sim >= threshold
            })
        });
        match earliest {
            Some(l) => println!(
                "  All texts > {threshold:.2}: layer {l} ({:.0}% of compute)",
                (l + 1) as f64 / num_layers as f64 * 100.0
            ),
            None => println!("  All texts > {threshold:.2}: never reached"),
        }
    }

    // Retrieval quality test at each layer.
    println!("\n=== Retrieval Quality: Related vs Unrelated Cosine Similarity ===\n");
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>10}",
        "Layer", "pair1_rel", "pair1_unrel", "pair2_rel", "delta_avg"
    );
    println!(
        "{:->6}  {:->12}  {:->12}  {:->12}  {:->10}",
        "", "", "", "", ""
    );

    let pair_embeddings: Vec<_> = pairs
        .iter()
        .map(|(a, b, unrel)| {
            (
                model.encode_all_layers(a).unwrap(),
                model.encode_all_layers(b).unwrap(),
                model.encode_all_layers(unrel).unwrap(),
            )
        })
        .collect();

    for layer_idx in 0..num_layers {
        let mut total_delta = 0.0f32;
        print!("{:>6}", layer_idx);
        for (i, (emb_a, emb_b, emb_unrel)) in pair_embeddings.iter().enumerate() {
            let related = cosine_sim(&emb_a[layer_idx], &emb_b[layer_idx]);
            let unrelated = cosine_sim(&emb_a[layer_idx], &emb_unrel[layer_idx]);
            total_delta += related - unrelated;
            if i < 2 {
                print!("  {:>12.4}  {:>12.4}", related, unrelated);
            }
        }
        println!("  {:>10.4}", total_delta / pairs.len() as f32);
    }

    println!("\n=== Timing ===\n");
    let t = Instant::now();
    let _ = model.encode_all_layers("hello world").unwrap();
    println!(
        "  encode_all_layers (short): {:.0}ms",
        t.elapsed().as_millis()
    );
    println!("  vs encode (short):         ~100ms");
    println!("  Overhead: one forward pass captures all 28 layer embeddings");
}

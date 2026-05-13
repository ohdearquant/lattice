//! End-to-end inference benchmark with real model weights (bge-small-en-v1.5).
//!
//! Requires model files at `~/.lattice/models/bge-small-en-v1.5/` or
//! `$LATTICE_INFERENCE_MODEL_DIR`. Download with:
//!
//! ```bash
//! mkdir -p ~/.lattice/models/bge-small-en-v1.5
//! curl -L https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/model.safetensors \
//!   -o ~/.lattice/models/bge-small-en-v1.5/model.safetensors
//! curl -L https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt \
//!   -o ~/.lattice/models/bge-small-en-v1.5/vocab.txt
//! ```
//!
//! Run: `cargo bench -p lattice-inference -- e2e`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lattice_inference::BertModel;
use std::path::PathBuf;
use std::time::Duration;

fn model_dir() -> Option<PathBuf> {
    // Try explicit env var first
    if let Ok(dir) = std::env::var("LATTICE_INFERENCE_MODEL_DIR") {
        let p = PathBuf::from(dir);
        if p.join("model.safetensors").exists() {
            return Some(p);
        }
    }
    // Try default cache location
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home).join(".lattice/models/bge-small-en-v1.5");
        if p.join("model.safetensors").exists() {
            return Some(p);
        }
    }
    None
}

fn e2e_encode(c: &mut Criterion) {
    let dir = match model_dir() {
        Some(d) => d,
        None => {
            eprintln!(
                "Skipping e2e benchmark: no model files found.\n\
                 Set LATTICE_INFERENCE_MODEL_DIR or place files at ~/.lattice/models/bge-small-en-v1.5/"
            );
            return;
        }
    };

    let model = BertModel::from_directory(&dir).expect("Failed to load model");

    // Warm up: first encode triggers mmap page faults
    let _ = model.encode("warmup");

    let texts: &[(&str, &str)] = &[
        ("short_5tok", "Hello world test"),
        (
            "medium_30tok",
            "The quick brown fox jumps over the lazy dog. \
             This sentence is designed to be roughly thirty tokens long for benchmarking.",
        ),
        (
            "long_100tok",
            "Artificial intelligence and machine learning have transformed the way we process \
             information and make decisions. From natural language processing to computer vision, \
             these technologies enable systems to understand, learn, and interact with the world \
             in increasingly sophisticated ways. The development of transformer architectures has \
             been particularly impactful, enabling models to capture long-range dependencies in \
             sequential data with unprecedented effectiveness and efficiency.",
        ),
    ];

    let mut group = c.benchmark_group("e2e_encode");
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));

    for (label, text) in texts {
        group.bench_with_input(BenchmarkId::new("bge_small", label), text, |b, text| {
            b.iter(|| std::hint::black_box(model.encode(std::hint::black_box(text)).unwrap()));
        });
    }
    group.finish();
}

fn e2e_encode_batch(c: &mut Criterion) {
    let dir = match model_dir() {
        Some(d) => d,
        None => return,
    };

    let model = BertModel::from_directory(&dir).expect("Failed to load model");
    let _ = model.encode("warmup");

    let sentences: Vec<&str> = vec![
        "Machine learning models process data efficiently.",
        "Natural language understanding requires deep context.",
        "Vector databases enable semantic search at scale.",
        "Embedding models compress text into dense representations.",
        "Information retrieval combines keyword and vector search.",
        "Transformer architectures revolutionized NLP tasks.",
        "Fine-tuning adapts pre-trained models to specific domains.",
        "Batch processing improves throughput for multiple inputs.",
    ];

    let mut group = c.benchmark_group("e2e_encode_batch");
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));

    for batch_size in [1, 4, 8] {
        let batch: Vec<&str> = sentences.iter().copied().take(batch_size).collect();
        group.bench_with_input(
            BenchmarkId::new("bge_small", format!("batch_{batch_size}")),
            &batch,
            |b, batch| {
                b.iter(|| {
                    std::hint::black_box(
                        model
                            .encode_batch(std::hint::black_box(batch.as_slice()))
                            .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, e2e_encode, e2e_encode_batch);
criterion_main!(benches);

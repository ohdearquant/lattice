//! Compare embedding drift between two JSON dump files (produced by `dump_parity_embeddings`).
//!
//! Usage:
//!   cargo run -p lattice-transport --example drift_compare --release -- \
//!       /tmp/emb_v030.json /tmp/emb_main.json
//!
//! For each model present in both files the tool reports:
//!   - Wasserstein distance (OT scalar)
//!   - max_displacement from the transport summary
//!   - mean_displacement from the transport summary
//!   - max pairwise (1 - cosine) across the 5 index-aligned vector pairs
//!     (ground-truth per-vector drift; OT is the distribution-level measure)

use std::collections::HashMap;

use lattice_transport::{DriftConfig, EmbeddingRecord, detect_drift_records};

/// Load a dump JSON file into a map of model_name -> list of embedding vectors.
fn load_dump(path: &str) -> HashMap<String, Vec<Vec<f32>>> {
    let raw =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    serde_json::from_str::<HashMap<String, Vec<Vec<f32>>>>(&raw)
        .unwrap_or_else(|e| panic!("failed to parse {path}: {e}"))
}

/// Cosine similarity between two equal-length f32 slices.
///
/// Returns the dot product divided by the product of L2 norms.  Returns 1.0
/// when both vectors are zero (identical degenerate case).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch in cosine");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm_a * norm_b;
    if denom < 1e-12 {
        1.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: drift_compare <baseline.json> <current.json>");
        std::process::exit(1);
    }
    let baseline_path = &args[1];
    let current_path = &args[2];

    let baseline = load_dump(baseline_path);
    let current = load_dump(current_path);

    let config = DriftConfig::default();

    // Collect model names present in both files, sorted for deterministic output.
    let mut models: Vec<&String> = baseline
        .keys()
        .filter(|k| current.contains_key(*k))
        .collect();
    models.sort();

    println!("\nDrift comparison: {baseline_path} (baseline) vs {current_path} (current)");
    println!(
        "{:<45} {:>12} {:>16} {:>16} {:>16}",
        "model", "wasserstein", "max_displacement", "mean_displacement", "max_1-cos"
    );
    println!("{}", "-".repeat(111));

    for model_name in &models {
        let base_vecs = &baseline[*model_name];
        let curr_vecs = &current[*model_name];

        // Build EmbeddingRecord slices.  EmbeddingRecord holds a &[f32] borrow,
        // so we construct the owned vecs first.
        let base_records: Vec<EmbeddingRecord<usize>> = base_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| EmbeddingRecord::uniform(i, v.as_slice()))
            .collect();
        let curr_records: Vec<EmbeddingRecord<usize>> = curr_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| EmbeddingRecord::uniform(i, v.as_slice()))
            .collect();

        let report = detect_drift_records(&base_records, &curr_records, &config)
            .unwrap_or_else(|e| panic!("drift detection failed for {model_name}: {e}"));

        // Pairwise (index-aligned) cosine drift: 1.0 - cosine(base_i, curr_i).
        let max_cos_drift: f32 = base_vecs
            .iter()
            .zip(curr_vecs.iter())
            .map(|(a, b)| 1.0 - cosine(a.as_slice(), b.as_slice()))
            .fold(0.0f32, f32::max);

        println!(
            "{:<45} {:>12.6e} {:>16.6e} {:>16.6e} {:>16.6e}",
            model_name,
            report.wasserstein_distance,
            report.summary.max_displacement,
            report.summary.mean_displacement,
            max_cos_drift,
        );
    }

    println!();
}

//! Compare embedding drift between two JSON dump files (produced by `dump_parity_embeddings`).
//!
//! Usage:
//!   cargo run -p lattice-transport --example drift_compare --release -- \
//!       /tmp/emb_v030.json /tmp/emb_main.json
//!
//! For each model present in both files the tool reports debiased Sinkhorn
//! divergence and max pairwise `1 - cosine` across index-aligned vectors.

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
/// Returns the dot product divided by the product of L2 norms. Returns 1.0
/// only when BOTH vectors collapse (L2 norm below the noise floor or
/// non-finite) — an identical degenerate case. Returns -1.0 (maximum "1 -
/// cosine" drift) when exactly ONE side collapses, matching the fail-closed
/// convention in `crates/embed/src/drift.rs`'s `cosine_f32`: a one-sided
/// collapse (e.g. a broken forward pass zeroing out the current embedding
/// while the baseline is non-degenerate) must read as maximal drift, not as
/// an identity match.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch in cosine");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let a_degenerate = !norm_a.is_finite() || norm_a < 1e-12;
    let b_degenerate = !norm_b.is_finite() || norm_b < 1e-12;
    match (a_degenerate, b_degenerate) {
        (true, true) => 1.0,
        (true, false) | (false, true) => -1.0,
        (false, false) => (dot / (norm_a * norm_b)).clamp(-1.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_zero_reads_as_identity() {
        assert_eq!(cosine(&[0.0, 0.0], &[0.0, 0.0]), 1.0);
    }

    #[test]
    fn one_sided_collapse_reads_as_maximal_drift_a_then_b() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), -1.0);
    }

    #[test]
    fn one_sided_collapse_reads_as_maximal_drift_b_then_a() {
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 0.0]), -1.0);
    }

    #[test]
    fn normal_pair_computes_true_cosine() {
        let value = cosine(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((value - 1.0).abs() < 1e-6, "expected ~1.0, got {value}");

        let value = cosine(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(value.abs() < 1e-6, "expected ~0.0, got {value}");
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
    println!("Sinkhorn divergence is debiased, so identical inputs read as zero.");
    println!(
        "max 1-cos uses f32 and saturates near 1.0; it cannot resolve below roughly 1e-7 and is not proof of identity."
    );
    println!(
        "{:<45} {:>20} {:>16}",
        "model", "sinkhorn_divergence", "max_1-cos"
    );
    println!("{}", "-".repeat(85));

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
        let sinkhorn_divergence = report.sinkhorn_divergence.unwrap_or_else(|| {
            panic!("drift detection did not compute Sinkhorn divergence for {model_name}")
        });

        // Pairwise (index-aligned) cosine drift: 1.0 - cosine(base_i, curr_i).
        let max_cos_drift: f32 = base_vecs
            .iter()
            .zip(curr_vecs.iter())
            .map(|(a, b)| 1.0 - cosine(a.as_slice(), b.as_slice()))
            .fold(0.0f32, f32::max);

        println!("{model_name:<45} {sinkhorn_divergence:>20.6e} {max_cos_drift:>16.6e}");
    }

    println!();
}

//! Embedding-drift gate against frozen lattice baselines.
//!
//! This gate detects changes within lattice's embedding geometry across versions.
//! `embed_parity_vs_hf` separately measures divergence from the Hugging Face reference.
//!
//! The primary gate is the shared per-vector comparison in [`lattice_embed::drift`].
//! A deliberate forward-path change must regenerate the fixtures with `embed-drift
//! --update-baseline` and version the fixture directory. Sinkhorn divergence remains a
//! secondary BGE-small sanity signal; raw entropic transport cost is not a drift threshold
//! because identical small point sets have a non-zero self-cost.
//!
//! Missing checkpoints remain visible skips for ordinary workspace tests. The dedicated gate
//! sets `LATTICE_DRIFT_GATE_ENFORCE=1`, which turns any missing checkpoint into a failure.
//!
//! Run:
//! ```bash
//! cargo test --release -p lattice-embed --test embed_drift_baseline -- --nocapture
//! ```

use std::path::PathBuf;

use lattice_embed::drift::{
    BaselineFixture, MAX_COSINE_DRIFT, ModelDriftOutcome, check_baseline, load_baselines,
};
use lattice_embed::{EmbeddingModel, EmbeddingService, NativeEmbeddingService};
use lattice_transport::{DriftConfig, EmbeddingRecord, detect_drift_records};

const MAX_SINKHORN_DIVERGENCE: f32 = 1.0;

fn baseline_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("embed_drift_baseline_v1")
}

async fn assert_sinkhorn_sanity(baseline: &BaselineFixture) {
    let model = EmbeddingModel::BgeSmallEnV15;
    let service = NativeEmbeddingService::with_model(model);
    let current = service
        .embed(&baseline.texts, model)
        .await
        .expect("embed call failed for Sinkhorn sanity check");
    let base_records = baseline
        .embeddings
        .iter()
        .enumerate()
        .map(|(index, vector)| EmbeddingRecord::uniform(index, vector.as_slice()))
        .collect::<Vec<_>>();
    let current_records = current
        .iter()
        .enumerate()
        .map(|(index, vector)| EmbeddingRecord::uniform(index, vector.as_slice()))
        .collect::<Vec<_>>();
    let report = detect_drift_records(&base_records, &current_records, &DriftConfig::default())
        .expect("detect_drift_records failed in drift gate test");

    eprintln!(
        "[bge-small drift gate] sinkhorn_divergence={:?}",
        report.sinkhorn_divergence
    );
    if let Some(divergence) = report.sinkhorn_divergence {
        assert!(
            divergence.is_finite(),
            "sinkhorn_divergence is non-finite ({divergence})"
        );
        assert!(
            divergence < MAX_SINKHORN_DIVERGENCE,
            "sinkhorn_divergence={divergence:.4e} exceeds sanity bound {MAX_SINKHORN_DIVERGENCE:.1e}"
        );
    }
}

#[tokio::test]
async fn embedding_drift_vs_frozen_baselines() {
    let baselines = load_baselines(&baseline_dir()).expect("failed to load drift baselines");
    assert!(!baselines.is_empty(), "no drift baseline fixtures found");
    let enforce = std::env::var("LATTICE_DRIFT_GATE_ENFORCE").is_ok();
    let mut checked = 0usize;
    let mut skipped = 0usize;
    let mut bge_checked = false;

    for baseline in &baselines {
        match check_baseline(baseline)
            .await
            .unwrap_or_else(|error| panic!("drift check failed for {}: {error}", baseline.model))
        {
            ModelDriftOutcome::Checked {
                max_one_minus_cos,
                worst_index,
            } => {
                checked += 1;
                bge_checked |= baseline.model == EmbeddingModel::BgeSmallEnV15.to_string();
                let worst_text = &baseline.texts[worst_index];
                eprintln!(
                    "[{} drift gate] max(1-cosine)={max_one_minus_cos:.4e} threshold={MAX_COSINE_DRIFT:.1e} worst_index={worst_index} text={worst_text:?}",
                    baseline.model
                );
                assert!(
                    max_one_minus_cos < MAX_COSINE_DRIFT,
                    "[{} drift gate] max 1-cosine {max_one_minus_cos:.6e} reached threshold {MAX_COSINE_DRIFT:.1e} at vector {worst_index}",
                    baseline.model
                );
            }
            ModelDriftOutcome::WeightsAbsent { model } => {
                skipped += 1;
                eprintln!("LATTICE_DRIFT_GATE_SKIPPED model={model} reason=missing_weights");
                assert!(
                    !enforce,
                    "drift gate weights missing for {model} despite LATTICE_DRIFT_GATE_ENFORCE"
                );
            }
            ModelDriftOutcome::NoBaseline { model } => {
                panic!("loaded fixture unexpectedly reported no baseline for {model}")
            }
        }
    }

    if bge_checked {
        let bge = baselines
            .iter()
            .find(|fixture| fixture.model == EmbeddingModel::BgeSmallEnV15.to_string())
            .expect("BGE-small fixture disappeared after successful check");
        assert_sinkhorn_sanity(bge).await;
    }
    eprintln!("Embedding drift summary: checked={checked} skipped={skipped}");
}

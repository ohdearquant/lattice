//! Embedding-drift gate against a frozen lattice baseline.
//!
//! **Purpose**: detect regressions in the BGE-small-en-v1.5 forward path (pooling,
//! tokenisation, SIMD normalisation) by comparing current embeddings against a set
//! of vectors committed at a known-good revision.  Unlike `embed_parity_vs_hf`,
//! which guards against divergence from HF's *reference implementation*, this test
//! guards against drift *within lattice itself* across versions.
//!
//! **Primary gate**: per-vector `1.0 - cosine(baseline_i, current_i) < 1e-3`.
//!
//!   Rationale: lattice embeddings are bit-for-bit reproducible across versions
//!   when no forward-path change is made (verified v0.3.0 → main).  `1e-3` is well
//!   above ULP noise yet tight enough to catch any structural forward-path change
//!   (the bf16 precision class adds ~0.00x PPL delta, Q4 quantisation ~0.1-0.3 PPL
//!   delta — both correspond to cosine drift far below 1e-3; material geometry
//!   shifts such as the RoPE pairing bug that caused 0.77 PPL regression register
//!   as cosine deltas >> 0.01).  Downstream consumers like khivedb that index
//!   lattice embeddings need to know when the embedding geometry has changed so
//!   they can re-index.  A deliberate forward-path change should regenerate the
//!   fixture via `crates/embed/examples/dump_parity_embeddings.rs` and bump
//!   `embed_drift_baseline_v1` → `embed_drift_baseline_v2`.
//!
//! **Secondary signal**: Sinkhorn divergence via `lattice_transport::detect_drift_records`.
//!   The debiased Sinkhorn divergence is ~0 for identical distributions; we assert
//!   it is finite and < 1.0 (loose sanity).  Raw Wasserstein is NOT used as a gate
//!   because the entropic-regularisation floor on small identical point-sets (n=5)
//!   produces a non-zero baseline even when all vectors are identical.
//!
//! **Skip guard**: when model weights are absent (local dev without `~/.lattice/`)
//!   the test prints a SKIP line and returns without failing.  CI provisions weights
//!   explicitly.
//!
//! **Run**:
//!   ```bash
//!   cargo test --release -p lattice-embed --test embed_drift_baseline -- --nocapture
//!   ```

use lattice_embed::{EmbeddingModel, EmbeddingService, NativeEmbeddingService};
use lattice_transport::{DriftConfig, EmbeddingRecord, detect_drift_records};
use serde::Deserialize;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Threshold
// ---------------------------------------------------------------------------

/// Maximum per-vector cosine drift allowed before the gate fires.
///
/// `1e-3` passes ULP/refactor noise and bf16 class precision variations (<0.00x
/// cosine delta) while catching structural changes like RoPE pairing bugs or
/// pooling regressions (>>0.01 cosine delta).
const MAX_COSINE_DRIFT: f32 = 1e-3;

/// Loose upper bound for the Sinkhorn divergence sanity assertion.
///
/// Identical distributions produce divergence ~0.  We gate loosely at 1.0 to
/// catch NaN/Inf pathologies without creating a fragile numerical threshold.
const MAX_SINKHORN_DIVERGENCE: f32 = 1.0;

// ---------------------------------------------------------------------------
// Fixture type
// ---------------------------------------------------------------------------

/// Self-contained baseline fixture: texts + frozen embeddings in one file.
#[derive(Deserialize)]
struct DriftBaseline {
    /// Model name (informational, for human readers).
    #[allow(dead_code)]
    model: String,
    /// The 5 corpus texts used to generate the baseline.
    texts: Vec<String>,
    /// Frozen embedding vectors [n_texts × dim].
    embeddings: Vec<Vec<f32>>,
}

fn load_baseline() -> DriftBaseline {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("embed_drift_baseline_v1")
        .join("bge_small_en_v15.json");
    assert!(
        path.exists(),
        "drift baseline fixture not found at {} — regenerate with \
         `cargo run -p lattice-embed --example dump_parity_embeddings` and commit",
        path.display()
    );
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read drift baseline {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("bad JSON in drift baseline: {e}"))
}

/// Cosine similarity between two equal-length f32 slices (pure scalar, no SIMD dep).
///
/// Returns 1.0 when both vectors are zero (identical degenerate case).
fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch in cosine_f32");
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

// ---------------------------------------------------------------------------
// Gate test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bge_small_drift_vs_frozen_baseline() {
    // --- Skip guard: no weights → skip gracefully, do not fail. -------
    let model_dir = PathBuf::from(std::env::var("HOME").expect("HOME env var not set"))
        .join(".lattice/models/bge-small-en-v1.5");

    if !model_dir.join("model.safetensors").exists() {
        eprintln!(
            "LATTICE_DRIFT_GATE_SKIPPED test=bge_small_drift_vs_frozen_baseline \
             reason=missing_weights path={}",
            model_dir.display()
        );
        return;
    }

    // --- Load frozen baseline. ----------------------------------------
    let baseline = load_baseline();
    assert_eq!(
        baseline.texts.len(),
        baseline.embeddings.len(),
        "fixture integrity: texts and embeddings counts differ"
    );
    let n = baseline.texts.len();
    eprintln!(
        "Loaded drift baseline: {n} texts × dim={}",
        baseline.embeddings[0].len()
    );

    // --- Embed the same texts with current code. ----------------------
    let model = EmbeddingModel::BgeSmallEnV15;
    let service = NativeEmbeddingService::with_model(model);

    let current_vecs = service
        .embed(&baseline.texts, model)
        .await
        .expect("embed call failed in drift gate test");

    assert_eq!(
        current_vecs.len(),
        n,
        "embed returned {} vectors, expected {n}",
        current_vecs.len()
    );

    // --- Per-vector cosine drift (PRIMARY GATE). ----------------------
    let mut max_drift: f32 = 0.0;
    let mut failures = 0usize;

    for i in 0..n {
        let cos = cosine_f32(&baseline.embeddings[i], &current_vecs[i]);
        let drift = 1.0 - cos;
        max_drift = max_drift.max(drift);

        if drift >= MAX_COSINE_DRIFT {
            failures += 1;
            eprintln!(
                "DRIFT_GATE_FAIL [bge-small] idx={i} text={:?}\n  \
                 1-cosine={drift:.6e}  (threshold {MAX_COSINE_DRIFT:.1e})",
                &baseline.texts[i],
            );
        } else {
            eprintln!(
                "  [bge-small] idx={i} text={:?}  1-cos={drift:.2e}",
                &baseline.texts[i],
            );
        }
    }

    eprintln!(
        "[bge-small drift gate] max(1-cosine)={max_drift:.4e}  threshold={MAX_COSINE_DRIFT:.1e}  \
         failures={failures}/{n}"
    );

    assert_eq!(
        failures, 0,
        "[bge-small drift gate] {failures}/{n} vectors exceeded the 1-cosine < {MAX_COSINE_DRIFT:.1e} \
         threshold — forward-path geometry has changed. If intentional, regenerate the baseline \
         fixture via `dump_parity_embeddings` and bump _v1 → _v2."
    );

    // --- Sinkhorn divergence (SECONDARY / informational). -------------
    //
    // Raw Wasserstein has a non-zero entropic-regularisation floor on identical
    // small point-sets (n=5), so it is NOT used as the primary gate.  Sinkhorn
    // divergence (debiased) is ~0 for identical distributions; we assert it is
    // finite and < MAX_SINKHORN_DIVERGENCE as a sanity check only.
    let base_records: Vec<EmbeddingRecord<usize>> = baseline
        .embeddings
        .iter()
        .enumerate()
        .map(|(i, v)| EmbeddingRecord::uniform(i, v.as_slice()))
        .collect();
    let curr_records: Vec<EmbeddingRecord<usize>> = current_vecs
        .iter()
        .enumerate()
        .map(|(i, v)| EmbeddingRecord::uniform(i, v.as_slice()))
        .collect();

    let config = DriftConfig::default();
    let report = detect_drift_records(&base_records, &curr_records, &config)
        .expect("detect_drift_records failed in drift gate test");

    eprintln!(
        "[bge-small drift gate] OT report: wasserstein={:.6e}  sinkhorn_divergence={:?}  \
         max_displacement={:.6e}  mean_displacement={:.6e}",
        report.wasserstein_distance,
        report.sinkhorn_divergence,
        report.summary.max_displacement,
        report.summary.mean_displacement,
    );

    if let Some(sd) = report.sinkhorn_divergence {
        assert!(
            sd.is_finite(),
            "sinkhorn_divergence is non-finite ({sd}): OT solver may have diverged"
        );
        assert!(
            sd < MAX_SINKHORN_DIVERGENCE,
            "sinkhorn_divergence={sd:.4e} exceeds loose sanity bound \
             ({MAX_SINKHORN_DIVERGENCE:.1e}) — unexpected large distribution shift"
        );
        eprintln!(
            "[bge-small drift gate] sinkhorn_divergence={sd:.4e} < {MAX_SINKHORN_DIVERGENCE:.1e} OK"
        );
    }
}

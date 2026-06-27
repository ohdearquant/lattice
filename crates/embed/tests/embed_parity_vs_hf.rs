//! HF reference embedding parity regression test.
//!
//! Loads pre-computed golden fixtures from
//! `crates/embed/tests/fixtures/embed_parity_v1/` and compares lattice's
//! computed embeddings against them using cosine similarity and element-wise
//! max-absolute-difference.
//!
//! **Purpose**: catch regressions in pooling, tokenization, or normalization
//! paths without requiring a live HF API call. The fixtures are committed and
//! represent the HF reference at the time of generation.
//!
//! **Tolerances** (constants at top of file — adjust here if needed):
//! - BGE, E5, MiniLM (full f32 inference): cosine ≥ 0.9990, max-abs-diff ≤ 5e-3
//! - Qwen3 (may have bf16 in path): cosine ≥ 0.9950, max-abs-diff ≤ 5e-3
//!
//! Note: max-abs-diff is observed at 1.8–3.5e-3 on current hardware (f32 → f64
//! cast noise). The threshold is set at 5e-3 to provide headroom without masking
//! regressions; the cosine guard (≥ 0.9990) is the primary quality signal.
//!
//! **How to regenerate fixtures** (run once, then commit the output):
//!   ```bash
//!   uv run --with transformers --with torch --with numpy \
//!       scripts/gen_embed_parity_goldens.py
//!   ```
//!
//! **Run this test**:
//!   ```bash
//!   cargo test -p lattice-embed --test embed_parity_vs_hf
//!   ```
//!   (also wired into `make ci`)

// ---------------------------------------------------------------------------
// Tolerance constants — edit here to adjust thresholds
// ---------------------------------------------------------------------------

/// Minimum cosine similarity for BGE and E5 (full f32 inference paths).
const COS_SIM_MIN_F32: f64 = 0.9990;

/// Minimum cosine similarity for Qwen3 (bf16 in forward pass).
const COS_SIM_MIN_QWEN: f64 = 0.9950;

/// Maximum element-wise absolute difference for BGE, E5, and MiniLM (full f32 inference).
const MAX_ABS_DIFF_F32: f64 = 5e-3;

/// Maximum element-wise absolute difference for Qwen3 (bf16 in forward pass).
const MAX_ABS_DIFF_QWEN: f64 = 5e-3;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

use lattice_embed::{EmbeddingModel, EmbeddingService, NativeEmbeddingService};
use serde::Deserialize;
use std::path::PathBuf;

/// Single (model, input) golden record as written by gen_embed_parity_goldens.py.
#[derive(Deserialize)]
struct Golden {
    /// HuggingFace model ID recorded for provenance / human readability.
    #[allow(dead_code)]
    model_id: String,
    pooling: String,
    prompt_prefix: String,
    input: String,
    /// Stored for cross-checking tokenizer output against HF reference (informational).
    #[allow(dead_code)]
    input_ids: Vec<u32>,
    embedding: Vec<f64>,
    embedding_dim: usize,
}

fn fixture_dir() -> PathBuf {
    // Resolved relative to the crate root at test time.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("embed_parity_v1")
}

fn load_fixture(filename: &str) -> Vec<Golden> {
    let path = fixture_dir().join(filename);
    assert!(
        path.exists(),
        "committed fixture not found at {path} — run scripts/gen_embed_parity_goldens.py and commit the output",
        path = path.display()
    );
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    serde_json::from_str(&data).unwrap_or_else(|e| panic!("bad JSON in {filename}: {e}"))
}

/// Handle the no-weights case: skip locally, but FAIL CLOSED in CI.
///
/// Locally and in the workspace-wide `cargo test` job the weights are usually
/// absent, so each parity test calls this and `return`s — a clean skip. The
/// dedicated release-cadence gate (`.github/workflows/embed-parity-release.yml`)
/// provisions the weights AND sets `LATTICE_PARITY_GATE_ENFORCE=1`; there a
/// missing model means provisioning failed and a silent skip would be pure
/// theater (a green gate that verified nothing). Under the enforce flag this
/// panics instead of returning, so the gate fails loudly. Mirrors the
/// `LATTICE_DRIFT_GATE_ENFORCE` guard in `embed_drift_baseline.rs`.
fn record_vector_weight_skip(test_name: &str, model_dir: &std::path::Path) {
    if std::env::var("LATTICE_PARITY_GATE_ENFORCE").is_ok() {
        panic!(
            "parity gate weights missing despite LATTICE_PARITY_GATE_ENFORCE=1 — \
             provisioning failed for {test_name} (expected weights at {}). A silent \
             skip here would make the release parity gate verify nothing.",
            model_dir.display()
        );
    }
    eprintln!(
        "LATTICE_VECTOR_PARITY_SKIPPED test={test_name} reason=missing_weights path={}",
        model_dir.display()
    );
}

/// Cosine similarity between two f32 vectors, returned as f64 for comparison.
fn cosine_sim(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y).sum();
    let norm_a: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&y| y.powi(2)).sum::<f64>().sqrt();
    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Element-wise max absolute difference between two f32 vectors.
fn max_abs_diff(a: &[f32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Per-model test helpers
// ---------------------------------------------------------------------------

/// Run the embed service for a single text, using the role implied by the golden's
/// prompt_prefix.  Returns the raw f32 embedding vector.
async fn embed_text(
    service: &NativeEmbeddingService,
    text: &str,
    model: EmbeddingModel,
) -> Vec<f32> {
    let texts = vec![text.to_string()];
    let mut vecs = service
        .embed(&texts, model)
        .await
        .unwrap_or_else(|e| panic!("embed failed for '{text}': {e}"));
    vecs.pop().expect("empty embed result")
}

/// Embed with the passage role (prepends model's document_instruction() if any).
async fn embed_passage_text(
    service: &NativeEmbeddingService,
    text: &str,
    model: EmbeddingModel,
) -> Vec<f32> {
    let texts = vec![text.to_string()];
    let mut vecs = service
        .embed_passage(&texts, model)
        .await
        .unwrap_or_else(|e| panic!("embed_passage failed for '{text}': {e}"));
    vecs.pop().expect("empty embed result")
}

// ---------------------------------------------------------------------------
// BGE-small-en-v1.5 parity test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bge_small_parity_vs_hf() {
    let goldens = load_fixture("bge_small_en_v15.json");

    let model_dir =
        PathBuf::from(std::env::var("HOME").unwrap()).join(".lattice/models/bge-small-en-v1.5");
    if !model_dir.join("model.safetensors").exists() {
        record_vector_weight_skip("bge_small_parity_vs_hf", &model_dir);
        return;
    }

    let model = EmbeddingModel::BgeSmallEnV15;
    let service = NativeEmbeddingService::with_model(model);

    let mut failures = 0;
    let mut min_cos = 1.0_f64;
    let mut max_diff = 0.0_f64;

    for golden in &goldens {
        // Golden was generated without any prompt prefix (BGE has none).
        // Use plain embed() to match.
        let lattice_vec = embed_text(&service, &golden.input, model).await;

        assert_eq!(
            lattice_vec.len(),
            golden.embedding_dim,
            "BGE dimension mismatch: got {}, want {}",
            lattice_vec.len(),
            golden.embedding_dim
        );

        let cos = cosine_sim(&lattice_vec, &golden.embedding);
        let diff = max_abs_diff(&lattice_vec, &golden.embedding);
        min_cos = min_cos.min(cos);
        max_diff = max_diff.max(diff);

        if cos < COS_SIM_MIN_F32 || diff > MAX_ABS_DIFF_F32 {
            failures += 1;
            eprintln!(
                "PARITY FAIL [bge-small] input={:?}\n  cosine={:.6}  (need ≥ {COS_SIM_MIN_F32})\n  max_abs_diff={diff:.2e}  (need ≤ {MAX_ABS_DIFF_F32})\n  pooling={}, prompt_prefix={}",
                golden.input, cos, golden.pooling, golden.prompt_prefix,
            );
        } else {
            println!(
                "  [bge-small] '{:.40}' cosine={:.6} max_diff={:.2e}",
                golden.input, cos, diff
            );
        }
    }

    println!(
        "[bge-small] aggregate: min_cosine={min_cos:.6} max_abs_diff={max_diff:.2e} failures={failures}/{}",
        goldens.len()
    );

    assert_eq!(
        failures,
        0,
        "[bge-small] {failures}/{} parity checks failed — see stderr",
        goldens.len()
    );
}

// ---------------------------------------------------------------------------
// multilingual-e5-small parity test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e5_small_parity_vs_hf() {
    let goldens = load_fixture("multilingual_e5_small.json");

    let model_dir =
        PathBuf::from(std::env::var("HOME").unwrap()).join(".lattice/models/multilingual-e5-small");
    if !model_dir.join("model.safetensors").exists() {
        record_vector_weight_skip("e5_small_parity_vs_hf", &model_dir);
        return;
    }

    let model = EmbeddingModel::MultilingualE5Small;
    let service = NativeEmbeddingService::with_model(model);

    let mut failures = 0;
    let mut min_cos = 1.0_f64;
    let mut max_diff = 0.0_f64;

    for golden in &goldens {
        // Golden was generated with "passage: " prefix via embed_passage().
        assert_eq!(
            golden.prompt_prefix, "passage: ",
            "E5 golden must use 'passage: ' prefix; got {:?}",
            golden.prompt_prefix
        );
        let lattice_vec = embed_passage_text(&service, &golden.input, model).await;

        assert_eq!(
            lattice_vec.len(),
            golden.embedding_dim,
            "E5 dimension mismatch: got {}, want {}",
            lattice_vec.len(),
            golden.embedding_dim
        );

        let cos = cosine_sim(&lattice_vec, &golden.embedding);
        let diff = max_abs_diff(&lattice_vec, &golden.embedding);
        min_cos = min_cos.min(cos);
        max_diff = max_diff.max(diff);

        if cos < COS_SIM_MIN_F32 || diff > MAX_ABS_DIFF_F32 {
            failures += 1;
            eprintln!(
                "PARITY FAIL [e5-small] input={:?}\n  cosine={:.6}  (need ≥ {COS_SIM_MIN_F32})\n  max_abs_diff={diff:.2e}  (need ≤ {MAX_ABS_DIFF_F32})\n  pooling={}, prompt_prefix={}",
                golden.input, cos, golden.pooling, golden.prompt_prefix,
            );
        } else {
            println!(
                "  [e5-small] '{:.40}' cosine={:.6} max_diff={:.2e}",
                golden.input, cos, diff
            );
        }
    }

    println!(
        "[e5-small] aggregate: min_cosine={min_cos:.6} max_abs_diff={max_diff:.2e} failures={failures}/{}",
        goldens.len()
    );

    assert_eq!(
        failures,
        0,
        "[e5-small] {failures}/{} parity checks failed — see stderr",
        goldens.len()
    );
}

// ---------------------------------------------------------------------------
// all-MiniLM-L6-v2 parity test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn all_minilm_l6_v2_parity_vs_hf() {
    let goldens = load_fixture("all_minilm_l6_v2.json");

    let model_dir =
        PathBuf::from(std::env::var("HOME").unwrap()).join(".lattice/models/all-minilm-l6-v2");
    if !model_dir.join("model.safetensors").exists() {
        record_vector_weight_skip("all_minilm_l6_v2_parity_vs_hf", &model_dir);
        return;
    }

    let model = EmbeddingModel::AllMiniLmL6V2;
    let service = NativeEmbeddingService::with_model(model);

    let mut failures = 0;
    let mut min_cos = 1.0_f64;
    let mut max_diff = 0.0_f64;

    for golden in &goldens {
        // Golden was generated without any prompt prefix (MiniLM has none).
        assert_eq!(
            golden.prompt_prefix, "",
            "all-MiniLM-L6-v2 golden must have empty prompt_prefix; got {:?}",
            golden.prompt_prefix
        );
        let lattice_vec = embed_text(&service, &golden.input, model).await;

        assert_eq!(
            lattice_vec.len(),
            golden.embedding_dim,
            "all-MiniLM-L6-v2 dimension mismatch: got {}, want {}",
            lattice_vec.len(),
            golden.embedding_dim
        );

        let cos = cosine_sim(&lattice_vec, &golden.embedding);
        let diff = max_abs_diff(&lattice_vec, &golden.embedding);
        min_cos = min_cos.min(cos);
        max_diff = max_diff.max(diff);

        if cos < COS_SIM_MIN_F32 || diff > MAX_ABS_DIFF_F32 {
            failures += 1;
            eprintln!(
                "PARITY FAIL [all-minilm-l6] input={:?}\n  cosine={:.6}  (need ≥ {COS_SIM_MIN_F32})\n  max_abs_diff={diff:.2e}  (need ≤ {MAX_ABS_DIFF_F32})\n  pooling={}, prompt_prefix={}",
                golden.input, cos, golden.pooling, golden.prompt_prefix,
            );
        } else {
            println!(
                "  [all-minilm-l6] '{:.40}' cosine={:.6} max_diff={:.2e}",
                golden.input, cos, diff
            );
        }
    }

    println!(
        "[all-minilm-l6] aggregate: min_cosine={min_cos:.6} max_abs_diff={max_diff:.2e} failures={failures}/{}",
        goldens.len()
    );

    assert_eq!(
        failures,
        0,
        "[all-minilm-l6] {failures}/{} parity checks failed — see stderr",
        goldens.len()
    );
}

// ---------------------------------------------------------------------------
// paraphrase-multilingual-MiniLM-L12-v2 parity test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn paraphrase_multilingual_minilm_l12_v2_parity_vs_hf() {
    let goldens = load_fixture("paraphrase_multilingual_minilm_l12_v2.json");

    let model_dir = PathBuf::from(std::env::var("HOME").unwrap())
        .join(".lattice/models/paraphrase-multilingual-minilm-l12-v2");
    if !model_dir.join("model.safetensors").exists() {
        record_vector_weight_skip(
            "paraphrase_multilingual_minilm_l12_v2_parity_vs_hf",
            &model_dir,
        );
        return;
    }

    let model = EmbeddingModel::ParaphraseMultilingualMiniLmL12V2;
    let service = NativeEmbeddingService::with_model(model);

    let mut failures = 0;
    let mut min_cos = 1.0_f64;
    let mut max_diff = 0.0_f64;

    for golden in &goldens {
        // Golden was generated without any prompt prefix (paraphrase-multilingual has none).
        assert_eq!(
            golden.prompt_prefix, "",
            "paraphrase-multilingual golden must have empty prompt_prefix; got {:?}",
            golden.prompt_prefix
        );
        let lattice_vec = embed_text(&service, &golden.input, model).await;

        assert_eq!(
            lattice_vec.len(),
            golden.embedding_dim,
            "paraphrase-multilingual dimension mismatch: got {}, want {}",
            lattice_vec.len(),
            golden.embedding_dim
        );

        let cos = cosine_sim(&lattice_vec, &golden.embedding);
        let diff = max_abs_diff(&lattice_vec, &golden.embedding);
        min_cos = min_cos.min(cos);
        max_diff = max_diff.max(diff);

        if cos < COS_SIM_MIN_F32 || diff > MAX_ABS_DIFF_F32 {
            failures += 1;
            eprintln!(
                "PARITY FAIL [paraphrase-multilingual-minilm-l12] input={:?}\n  cosine={:.6}  (need ≥ {COS_SIM_MIN_F32})\n  max_abs_diff={diff:.2e}  (need ≤ {MAX_ABS_DIFF_F32})\n  pooling={}, prompt_prefix={}",
                golden.input, cos, golden.pooling, golden.prompt_prefix,
            );
        } else {
            println!(
                "  [paraphrase-multilingual-minilm-l12] '{:.40}' cosine={:.6} max_diff={:.2e}",
                golden.input, cos, diff
            );
        }
    }

    println!(
        "[paraphrase-multilingual-minilm-l12] aggregate: min_cosine={min_cos:.6} max_abs_diff={max_diff:.2e} failures={failures}/{}",
        goldens.len()
    );

    assert_eq!(
        failures,
        0,
        "[paraphrase-multilingual-minilm-l12] {failures}/{} parity checks failed — see stderr",
        goldens.len()
    );
}

// ---------------------------------------------------------------------------
// Qwen3-Embedding-0.6B parity test
// ---------------------------------------------------------------------------

// TODO(lattice#103): Qwen3-Embedding forward-pass divergence (cosine 0.948 on
// whitespace input, 0.989 on "fox" even when tokens match HF exactly). Tracked
// at https://github.com/ohdearquant/lattice/issues/103. Run with `--ignored` to
// exercise. Remove this attribute once the bug is fixed.
#[tokio::test]
#[ignore = "Qwen3-Embedding forward-pass divergence — see lattice#103"]
async fn qwen3_embedding_0_6b_parity_vs_hf() {
    let goldens = load_fixture("qwen3_embedding_0_6b.json");

    let qwen_model_dir =
        PathBuf::from(std::env::var("HOME").unwrap()).join(".lattice/models/qwen3-embedding-0.6b");
    if !qwen_model_dir.join("model.safetensors").exists() {
        record_vector_weight_skip("qwen3_embedding_0_6b_parity_vs_hf", &qwen_model_dir);
        return;
    }

    // Set the env var so the service finds the model directory.
    // SAFETY: single-threaded test setup; no concurrent env reads during set.
    unsafe {
        std::env::set_var("LATTICE_QWEN_MODEL_DIR", &qwen_model_dir);
    }

    let model = EmbeddingModel::Qwen3Embedding0_6B;
    let service = NativeEmbeddingService::with_model(model);

    let mut failures = 0;
    let mut min_cos = 1.0_f64;
    let mut max_diff = 0.0_f64;

    for golden in &goldens {
        // Golden was generated without a prefix (document side, Qwen's
        // document_instruction() returns None).
        assert_eq!(
            golden.prompt_prefix, "",
            "Qwen golden must have empty prompt_prefix; got {:?}",
            golden.prompt_prefix
        );
        let lattice_vec = embed_text(&service, &golden.input, model).await;

        assert_eq!(
            lattice_vec.len(),
            golden.embedding_dim,
            "Qwen dimension mismatch: got {}, want {}",
            lattice_vec.len(),
            golden.embedding_dim
        );

        let cos = cosine_sim(&lattice_vec, &golden.embedding);
        let diff = max_abs_diff(&lattice_vec, &golden.embedding);
        min_cos = min_cos.min(cos);
        max_diff = max_diff.max(diff);

        if cos < COS_SIM_MIN_QWEN || diff > MAX_ABS_DIFF_QWEN {
            failures += 1;
            eprintln!(
                "PARITY FAIL [qwen3-0.6b] input={:?}\n  cosine={:.6}  (need ≥ {COS_SIM_MIN_QWEN})\n  max_abs_diff={diff:.2e}  (need ≤ {MAX_ABS_DIFF_QWEN})\n  pooling={}",
                golden.input, cos, golden.pooling,
            );
        } else {
            println!(
                "  [qwen3-0.6b] '{:.40}' cosine={:.6} max_diff={:.2e}",
                golden.input, cos, diff
            );
        }
    }

    println!(
        "[qwen3-0.6b] aggregate: min_cosine={min_cos:.6} max_abs_diff={max_diff:.2e} failures={failures}/{}",
        goldens.len()
    );

    assert_eq!(
        failures,
        0,
        "[qwen3-0.6b] {failures}/{} parity checks failed — see stderr",
        goldens.len()
    );
}

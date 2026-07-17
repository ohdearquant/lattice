//! ADR-069 S3a gate: real Qwen3.5-0.8B ViT forward pass (CPU reference) vs
//! the committed HF differential golden (cosine > 0.999 on the fixed golden
//! image). S3b (Metal port vs this CPU reference, under the machine GPU
//! lock) is a separate fast-follow gate.
//!
//! **Fail-closed contract** (mirrors `quarot_q4_composed_golden.rs`): with
//! `LATTICE_VISION_S3_MODEL_DIR` unset or pointing at a missing path, this
//! test prints a skip line and returns — most dev machines don't have the
//! 1.6 GB fp16 Qwen3.5-0.8B checkpoint on disk. Requires the `f16` cargo
//! feature (the checkpoint's `model.visual.*` tensors are stored as F16 in
//! the fp16 safetensors shard). When `LATTICE_VISION_S3_GATE_ENFORCE=1` is
//! set, a missing/misconfigured checkpoint panics instead of skipping.
//!
//! Run:
//! ```bash
//! LATTICE_VISION_S3_MODEL_DIR=~/.lattice/models/qwen3.5-0.8b \
//! cargo test --release -p lattice-inference --features f16 \
//!     --test vision_s3_vit_forward_test -- --nocapture
//! ```

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct TensorManifest {
    path: String,
    shape: Vec<usize>,
    num_elements: usize,
    sha256: String,
}

#[derive(Deserialize)]
struct Manifest {
    #[allow(dead_code)] // only read by the f16-gated `run_s3_gate`
    image_grid_thw: Vec<[u64; 3]>,
    vit_pre_merger: TensorManifest,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("vision")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn load_manifest(dir: &Path) -> Manifest {
    let bytes = std::fs::read(dir.join("manifest.json")).expect("reading manifest.json");
    serde_json::from_slice(&bytes).expect("parsing manifest.json")
}

fn load_f32_tensor(dir: &Path, tm: &TensorManifest) -> Vec<f32> {
    let bytes =
        std::fs::read(dir.join(&tm.path)).unwrap_or_else(|e| panic!("reading {}: {e}", tm.path));
    assert_eq!(
        bytes.len(),
        tm.num_elements * 4,
        "{}: byte length mismatch",
        tm.path
    );
    assert_eq!(
        sha256_hex(&bytes),
        tm.sha256,
        "{}: sha256 mismatch vs manifest.json",
        tm.path
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[allow(dead_code)] // only used by the f16-gated `run_s3_gate`
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum();
    let norm_a: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (norm_a * norm_b)
}

fn enforce() -> bool {
    std::env::var("LATTICE_VISION_S3_GATE_ENFORCE").as_deref() == Ok("1")
}

fn shellexpand_home(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return format!("{home}/{rest}");
    }
    path.to_string()
}

fn require_env_model_dir() -> Option<PathBuf> {
    const VAR: &str = "LATTICE_VISION_S3_MODEL_DIR";
    match std::env::var(VAR) {
        Ok(value) => {
            let path = PathBuf::from(shellexpand_home(&value));
            if path.exists() {
                Some(path)
            } else if enforce() {
                panic!(
                    "{VAR}={} does not exist, and LATTICE_VISION_S3_GATE_ENFORCE=1 — \
                     the S3a ViT gate must fail closed on a missing checkpoint",
                    path.display()
                );
            } else {
                eprintln!(
                    "LATTICE_VISION_S3_SKIPPED reason=missing_path {VAR}={}",
                    path.display()
                );
                None
            }
        }
        Err(_) if enforce() => panic!("{VAR} must be set when LATTICE_VISION_S3_GATE_ENFORCE=1"),
        Err(_) => {
            eprintln!("LATTICE_VISION_S3_SKIPPED reason=unset_env {VAR}");
            None
        }
    }
}

/// Fixture-only check: the committed pre-merger golden parses and matches
/// its recorded shape. Runs unconditionally, no checkpoint required.
#[test]
fn vit_pre_merger_golden_fixture_is_valid() {
    let dir = fixture_dir();
    let manifest = load_manifest(&dir);
    let pre = load_f32_tensor(&dir, &manifest.vit_pre_merger);
    assert_eq!(manifest.vit_pre_merger.shape, vec![256, 768]);
    assert_eq!(pre.len(), 256 * 768);
    assert!(pre.iter().all(|v| v.is_finite()));
}

#[cfg(feature = "f16")]
fn run_s3_gate(model_dir: &Path) {
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use lattice_inference::vision::checkpoint::load_qwen35_vision_weights;
    use lattice_inference::vision::qwen35_vit::{preprocess_qwen35_image, qwen35_vit_forward};

    let dir = fixture_dir();
    let manifest = load_manifest(&dir);
    let golden = load_f32_tensor(&dir, &manifest.vit_pre_merger);

    let [grid_t, grid_h, grid_w] = manifest.image_grid_thw[0];
    assert_eq!(
        (grid_t, grid_h, grid_w),
        (1, 16, 16),
        "fixture grid_thw drifted from the reviewed anchor"
    );

    let cfg = Qwen35Config::from_model_dir(model_dir).expect("0.8b config.json parses");
    let vision_cfg = cfg
        .vision_config
        .expect("Qwen3.5-0.8B checkpoint must carry a vision_config");

    let weights =
        load_qwen35_vision_weights(model_dir, &vision_cfg).expect("real vision weights must load");

    let image_bytes =
        std::fs::read(dir.join("golden_image.png")).expect("reading golden_image.png");
    let (pixel_values, grid) =
        preprocess_qwen35_image(&image_bytes, &vision_cfg, None).expect("preprocess golden image");
    assert_eq!(grid.num_patches(), 256);

    let out =
        qwen35_vit_forward(&weights, &vision_cfg, &pixel_values, grid).expect("qwen35 ViT forward");

    assert_eq!(
        out.len(),
        golden.len(),
        "pre-merger output length must match the golden"
    );
    assert!(
        out.iter().all(|v| v.is_finite()),
        "ViT output contains non-finite values"
    );

    let cos = cosine_similarity(&out, &golden);
    eprintln!("LATTICE_VISION_S3_COSINE cosine={cos:.8}");
    assert!(
        cos > 0.999,
        "ADR-069 S3a gate failed: cosine similarity {cos} vs committed golden must exceed 0.999"
    );

    // Mutation-sensitivity proof (CLAUDE.md "Regression Tests Must Be
    // Mutation-Sensitive"): perturbing a single loaded weight tensor must
    // drop the gate below threshold, proving this test actually exercises
    // the weights rather than trivially passing regardless of their values.
    let mut mutated_weights = weights.clone();
    for w in mutated_weights.blocks[0].qkv_weight.iter_mut() {
        *w *= -1.0;
    }
    let mutated_out = qwen35_vit_forward(&mutated_weights, &vision_cfg, &pixel_values, grid)
        .expect("mutated forward");
    let mutated_cos = cosine_similarity(&mutated_out, &golden);
    eprintln!("LATTICE_VISION_S3_MUTATION_COSINE cosine={mutated_cos:.8}");
    assert!(
        mutated_cos < 0.999,
        "mutation-sensitivity check failed: negating block[0].qkv_weight did not fail the \
         cosine>0.999 gate (baseline={cos}, mutated={mutated_cos}) — this test would pass even \
         with badly wrong weights"
    );
}

#[cfg(not(feature = "f16"))]
fn run_s3_gate(_model_dir: &Path) {
    if enforce() {
        panic!(
            "LATTICE_VISION_S3_GATE_ENFORCE=1 but the `f16` feature is not enabled — the real \
             fp16 checkpoint's model.visual.* tensors require it"
        );
    }
    eprintln!("LATTICE_VISION_S3_SKIPPED reason=f16_feature_disabled");
}

/// The S3a gate: real fp16 checkpoint -> `load_qwen35_vision_weights` ->
/// `qwen35_vit_forward` -> cosine > 0.999 vs the committed HF golden.
#[test]
fn qwen35_vit_forward_matches_hf_golden_cosine_gt_0999() {
    let Some(model_dir) = require_env_model_dir() else {
        return;
    };
    run_s3_gate(&model_dir);
}

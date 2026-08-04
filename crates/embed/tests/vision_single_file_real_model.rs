//! Manual real-model gate for the unindexed visual-checkpoint loader (#1381).
//!
//! This test is ignored because it maps and loads a multi-gigabyte Qwen3.5
//! checkpoint and runs two full CPU image embeddings. Run it explicitly:
//!
//! ```text
//! LATTICE_VISION_EMBED_MODEL_DIR=/path/to/qwen3.5-2b \
//! cargo test --release -p lattice-embed \
//!   --test vision_single_file_real_model -- --ignored --nocapture
//! ```

use lattice_embed::vision::{PoolingStrategy, VisionEmbeddingModel};
use std::path::{Path, PathBuf};

fn require_model_dir() -> PathBuf {
    let value = std::env::var("LATTICE_VISION_EMBED_MODEL_DIR")
        .expect("set LATTICE_VISION_EMBED_MODEL_DIR to an unindexed Qwen3.5 vision checkpoint");
    let path = PathBuf::from(value);
    assert!(
        path.is_dir(),
        "LATTICE_VISION_EMBED_MODEL_DIR={} is not a directory",
        path.display()
    );
    path
}

fn assert_single_file_layout(model_dir: &Path) {
    assert!(
        model_dir.join("model.safetensors").is_file(),
        "real-model gate requires model.safetensors"
    );
    assert!(
        !model_dir.join("model.safetensors.index.json").exists(),
        "real-model gate must exercise the unindexed path"
    );
    assert!(
        !model_dir.join("quantize_index.json").exists(),
        "real-model gate must exercise the f16/bf16 safetensors path"
    );
    let candidates = std::fs::read_dir(model_dir)
        .expect("inspect model directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect::<Vec<_>>();
    assert_eq!(
        candidates.len(),
        1,
        "real-model gate requires exactly one unindexed safetensors candidate"
    );
}

fn golden_image() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("vision")
        .join("golden_image.png");
    std::fs::read(&path)
        .unwrap_or_else(|err| panic!("failed to read golden image {}: {err}", path.display()))
}

#[test]
#[ignore = "requires a multi-gigabyte Qwen3.5 vision checkpoint; set LATTICE_VISION_EMBED_MODEL_DIR"]
fn qwen35_single_file_embedding_is_normalized_and_deterministic() {
    let model_dir = require_model_dir();
    assert_single_file_layout(&model_dir);

    let model = VisionEmbeddingModel::from_directory(&model_dir)
        .expect("an unindexed single-file Qwen3.5 vision checkpoint must load");
    assert_eq!(
        model.dimensions(),
        2048,
        "Qwen3.5-2B embedding dimension drifted"
    );

    let image = golden_image();
    let first = model
        .embed_image(&image, "", PoolingStrategy::MeanVisualTokens)
        .expect("first real image embedding succeeds");
    let second = model
        .embed_image(&image, "", PoolingStrategy::MeanVisualTokens)
        .expect("second real image embedding succeeds");

    assert_eq!(first.len(), model.dimensions());
    assert!(first.iter().all(|value| value.is_finite()));
    let norm = first.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1.0e-4, "embedding norm={norm}");
    assert_eq!(
        first, second,
        "identical image requests must produce deterministic embeddings"
    );
}

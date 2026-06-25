//! Dump parity embeddings for all four BERT-class fixture models.
//!
//! Reads the five `input` strings from each fixture JSON under
//! `crates/embed/tests/fixtures/embed_parity_v1/`, embeds them with
//! `NativeEmbeddingService`, and writes `{model_name: [[f32, ...], ...]}` JSON
//! to the path specified by `DUMP_OUT` (default: `/tmp/emb_dump.json`).
//!
//! Run:
//!   DUMP_OUT=/tmp/emb_main.json \
//!   cargo run -p lattice-embed --example dump_parity_embeddings --release
//!
//! The Qwen3-embedding-0.6b fixture is intentionally skipped (known forward
//! divergence, lattice#103).

use std::collections::HashMap;
use std::path::PathBuf;

use lattice_embed::{EmbeddingModel, EmbeddingService, NativeEmbeddingService};

/// Fixture file name, model variant, and the label used as the JSON key.
struct Fixture {
    file: &'static str,
    model: EmbeddingModel,
    label: &'static str,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        file: "bge_small_en_v15.json",
        model: EmbeddingModel::BgeSmallEnV15,
        label: "bge_small_en_v15",
    },
    Fixture {
        file: "all_minilm_l6_v2.json",
        model: EmbeddingModel::AllMiniLmL6V2,
        label: "all_minilm_l6_v2",
    },
    Fixture {
        file: "multilingual_e5_small.json",
        model: EmbeddingModel::MultilingualE5Small,
        label: "multilingual_e5_small",
    },
    Fixture {
        file: "paraphrase_multilingual_minilm_l12_v2.json",
        model: EmbeddingModel::ParaphraseMultilingualMiniLmL12V2,
        label: "paraphrase_multilingual_minilm_l12_v2",
    },
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_path: PathBuf = std::env::var("DUMP_OUT")
        .unwrap_or_else(|_| "/tmp/emb_dump.json".to_string())
        .into();

    // CARGO_MANIFEST_DIR points to crates/embed/ at build time.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let fixture_dir = PathBuf::from(manifest_dir)
        .join("tests")
        .join("fixtures")
        .join("embed_parity_v1");

    let mut results: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

    for fixture in FIXTURES {
        let fixture_path = fixture_dir.join(fixture.file);
        let raw = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", fixture_path.display()));

        let records: Vec<serde_json::Value> = serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("failed to parse {}: {e}", fixture.file));

        let inputs: Vec<String> = records
            .iter()
            .map(|r| {
                r["input"]
                    .as_str()
                    .unwrap_or_else(|| panic!("missing 'input' in {}", fixture.file))
                    .to_string()
            })
            .collect();

        println!(
            "Embedding {} texts with model {}...",
            inputs.len(),
            fixture.label
        );

        // One NativeEmbeddingService per model: each instance owns one loaded model.
        let service = NativeEmbeddingService::with_model(fixture.model);
        let embeddings = service
            .embed(&inputs, fixture.model)
            .await
            .unwrap_or_else(|e| panic!("embed failed for {}: {e}", fixture.label));

        println!(
            "  -> {} embeddings of dim {}",
            embeddings.len(),
            embeddings.first().map(Vec::len).unwrap_or(0)
        );

        results.insert(fixture.label.to_string(), embeddings);
    }

    let json = serde_json::to_string(&results).expect("serialization of embedding results failed");
    std::fs::write(&out_path, &json)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", out_path.display()));

    println!("Wrote {} bytes to {}", json.len(), out_path.display());
    Ok(())
}

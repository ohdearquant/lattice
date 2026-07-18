//! Regression for the BERT cross-encoder LoRA geometry bounds bug.
//!
//! `apply_lora` (crates/tune/src/lora/apply.rs) slices
//! `output[..lora.d_out]` after only a `debug_assert` -- compiled out in
//! release builds. Before the fix, `CrossEncoderModel::score_with_hook`
//! performed no BERT dimension validation, so a self-consistent adapter
//! declaring `d_out > hidden_size` (or a mismatched `d_in`) would panic
//! partway through scoring instead of being rejected up front. This test
//! builds a tiny synthetic BERT cross-encoder checkpoint (no network access,
//! no downloaded model) and drives it through the real
//! `CrossEncoderModel::score_with_hook` call site.

use std::path::Path;

use lattice_inference::error::InferenceError;
use lattice_inference::lora_hook::LoraHook;
use lattice_inference::model::CrossEncoderModel;

const HIDDEN_SIZE: usize = 4;
const INTERMEDIATE_SIZE: usize = 4;
const NUM_HIDDEN_LAYERS: usize = 1;
const NUM_ATTENTION_HEADS: usize = 1;
const MAX_POSITION_EMBEDDINGS: usize = 32;
const TYPE_VOCAB_SIZE: usize = 2;
// Must match the real bge-small-en-v1.5 tokenizer fixture's vocab size --
// `word_embeddings` is indexed directly by token id (no bounds check other
// than the slice itself), so an undersized vocab would panic for reasons
// unrelated to the LoRA geometry bug this test targets.
const VOCAB_SIZE: usize = 30522;

const TOKENIZER_JSON: &str = include_str!("fixtures/tokenizers/bge-small-en-v1.5/tokenizer.json");

/// A `LoraHook` whose `apply()` reproduces the exact bounds-panic shape of
/// the original bug (`output[..d_out]` on a too-small buffer), and whose
/// `validate_against_bert` mimics a real geometry check. This isolates the
/// call-site behavior under test (does `score_with_hook` call validation
/// *before* the row loop reaches `apply()`?) from
/// `lattice_tune::lora::LoraAdapter`'s own geometry-check correctness,
/// which is covered separately in `crates/tune/src/lora/mod.rs`.
struct FakeBertLoraHook {
    d_out: usize,
    d_in: usize,
    module: &'static str,
}

impl LoraHook for FakeBertLoraHook {
    fn apply(&self, _layer_idx: usize, module: &str, x: &[f32], output: &mut [f32]) {
        if module != self.module {
            return;
        }
        assert_eq!(
            x.len(),
            self.d_in,
            "row-apply must see the declared d_in width"
        );
        // Mirrors `apply_lora`'s `output[..lora.d_out]` slice: out-of-bounds
        // `d_out` panics here exactly as it would inside the real adapter,
        // if this hook is ever reached with a bad geometry.
        let slice = &mut output[..self.d_out];
        slice.fill(1.0);
    }

    fn validate_against_bert(
        &self,
        num_hidden_layers: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<(), String> {
        let (expected_d_in, expected_d_out) = match self.module {
            "query" | "key" | "value" | "attn_output" => (hidden_size, hidden_size),
            "ffn_intermediate" => (hidden_size, intermediate_size),
            "ffn_output" => (intermediate_size, hidden_size),
            m => return Err(format!("unrecognised module {m}")),
        };
        if NUM_HIDDEN_LAYERS > num_hidden_layers {
            return Err("layer index out of range".to_string());
        }
        if self.d_in != expected_d_in || self.d_out != expected_d_out {
            return Err(format!(
                "geometry mismatch: adapter has (d_in={}, d_out={}) but model expects \
                 (d_in={expected_d_in}, d_out={expected_d_out})",
                self.d_in, self.d_out
            ));
        }
        Ok(())
    }
}

fn write_f32_safetensors(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut payload: Vec<u8> = Vec::new();
    for (name, shape, values) in tensors {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            values.len(),
            expected_len,
            "tensor {name} shape/value length mismatch"
        );
        let start = payload.len();
        for &v in values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let end = payload.len();

        let mut entry = serde_json::Map::new();
        entry.insert("dtype".into(), serde_json::Value::String("F32".into()));
        entry.insert(
            "shape".into(),
            serde_json::Value::Array(shape.iter().map(|d| serde_json::Value::from(*d)).collect()),
        );
        entry.insert(
            "data_offsets".into(),
            serde_json::Value::Array(vec![start.into(), end.into()]),
        );
        header.insert((*name).to_string(), serde_json::Value::Object(entry));
    }

    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = Vec::with_capacity(8 + header_bytes.len() + payload.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&payload);

    std::fs::write(path, out).unwrap();
}

/// Deterministic, small, finite fill values -- no NaN/Inf, no dependence on
/// RNG. Actual numeric output of the forward pass is irrelevant to this
/// test; only "did it panic vs. return Err vs. return Ok" matters.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.01 * (((i + seed) % 7) as f32 - 3.0))
        .collect()
}

fn build_synthetic_cross_encoder_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();

    std::fs::write(dir.path().join("tokenizer.json"), TOKENIZER_JSON).unwrap();

    let config_json = format!(
        r#"{{
            "vocab_size": {VOCAB_SIZE},
            "hidden_size": {HIDDEN_SIZE},
            "num_hidden_layers": {NUM_HIDDEN_LAYERS},
            "num_attention_heads": {NUM_ATTENTION_HEADS},
            "intermediate_size": {INTERMEDIATE_SIZE},
            "max_position_embeddings": {MAX_POSITION_EMBEDDINGS},
            "type_vocab_size": {TYPE_VOCAB_SIZE},
            "layer_norm_eps": 1e-5
        }}"#
    );
    std::fs::write(dir.path().join("config.json"), config_json).unwrap();

    let h = HIDDEN_SIZE;
    let i = INTERMEDIATE_SIZE;
    let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
        (
            "embeddings.word_embeddings.weight",
            vec![VOCAB_SIZE, h],
            fill(VOCAB_SIZE * h, 1),
        ),
        (
            "embeddings.position_embeddings.weight",
            vec![MAX_POSITION_EMBEDDINGS, h],
            fill(MAX_POSITION_EMBEDDINGS * h, 2),
        ),
        (
            "embeddings.token_type_embeddings.weight",
            vec![TYPE_VOCAB_SIZE, h],
            fill(TYPE_VOCAB_SIZE * h, 3),
        ),
        ("embeddings.LayerNorm.weight", vec![h], vec![1.0; h]),
        ("embeddings.LayerNorm.bias", vec![h], vec![0.0; h]),
        (
            "encoder.layer.0.attention.self.query.weight",
            vec![h, h],
            fill(h * h, 4),
        ),
        (
            "encoder.layer.0.attention.self.query.bias",
            vec![h],
            fill(h, 5),
        ),
        (
            "encoder.layer.0.attention.self.key.weight",
            vec![h, h],
            fill(h * h, 6),
        ),
        (
            "encoder.layer.0.attention.self.key.bias",
            vec![h],
            fill(h, 7),
        ),
        (
            "encoder.layer.0.attention.self.value.weight",
            vec![h, h],
            fill(h * h, 8),
        ),
        (
            "encoder.layer.0.attention.self.value.bias",
            vec![h],
            fill(h, 9),
        ),
        (
            "encoder.layer.0.attention.output.dense.weight",
            vec![h, h],
            fill(h * h, 10),
        ),
        (
            "encoder.layer.0.attention.output.dense.bias",
            vec![h],
            fill(h, 11),
        ),
        (
            "encoder.layer.0.attention.output.LayerNorm.weight",
            vec![h],
            vec![1.0; h],
        ),
        (
            "encoder.layer.0.attention.output.LayerNorm.bias",
            vec![h],
            vec![0.0; h],
        ),
        (
            "encoder.layer.0.intermediate.dense.weight",
            vec![i, h],
            fill(i * h, 12),
        ),
        (
            "encoder.layer.0.intermediate.dense.bias",
            vec![i],
            fill(i, 13),
        ),
        (
            "encoder.layer.0.output.dense.weight",
            vec![h, i],
            fill(h * i, 14),
        ),
        ("encoder.layer.0.output.dense.bias", vec![h], fill(h, 15)),
        (
            "encoder.layer.0.output.LayerNorm.weight",
            vec![h],
            vec![1.0; h],
        ),
        (
            "encoder.layer.0.output.LayerNorm.bias",
            vec![h],
            vec![0.0; h],
        ),
        ("classifier.weight", vec![1, h], fill(h, 16)),
        ("classifier.bias", vec![1], vec![0.1]),
    ];
    write_f32_safetensors(&dir.path().join("model.safetensors"), &tensors);

    dir
}

#[test]
fn bert_cross_encoder_rejects_malformed_lora_d_out_without_panicking() {
    let dir = build_synthetic_cross_encoder_dir();
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();

    let bad_d_out = FakeBertLoraHook {
        d_out: HIDDEN_SIZE + 1, // > hidden_size: this is the original panic trigger
        d_in: HIDDEN_SIZE,
        module: "query",
    };

    let result = model.score_with_hook("what is rust", "rust is a language", &bad_d_out);
    let err = result.expect_err("oversized d_out must be rejected, not panic");
    assert!(
        matches!(err, InferenceError::InvalidInput(_)),
        "expected a recoverable InvalidInput error, got {err:?}"
    );
}

#[test]
fn bert_cross_encoder_rejects_malformed_lora_d_in_without_panicking() {
    let dir = build_synthetic_cross_encoder_dir();
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();

    let bad_d_in = FakeBertLoraHook {
        d_out: HIDDEN_SIZE,
        d_in: HIDDEN_SIZE + 1, // mismatched d_in: silent-wrong-math without this guard
        module: "query",
    };

    let result = model.score_with_hook("what is rust", "rust is a language", &bad_d_in);
    let err = result.expect_err("mismatched d_in must be rejected, not silently mis-scored");
    assert!(
        matches!(err, InferenceError::InvalidInput(_)),
        "expected a recoverable InvalidInput error, got {err:?}"
    );
}

#[test]
fn bert_cross_encoder_scores_ok_with_well_formed_lora_geometry() {
    let dir = build_synthetic_cross_encoder_dir();
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();

    let good = FakeBertLoraHook {
        d_out: HIDDEN_SIZE,
        d_in: HIDDEN_SIZE,
        module: "query",
    };

    let score = model
        .score_with_hook("what is rust", "rust is a language", &good)
        .expect("a well-formed adapter must not be rejected");
    assert!(
        (0.0..=1.0).contains(&score),
        "score {score} out of sigmoid range"
    );
}

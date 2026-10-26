#![cfg(feature = "bert")]

//! `BertModel::from_bytes_with_max_tokens` caps the input length the model
//! embeds at the caller's value, keeps the closing separator, never exceeds
//! the position-embedding table, and refuses a cap with no room for the two
//! special tokens. Runs on a tiny synthetic checkpoint built in memory, with
//! the real bge-small-en-v1.5 tokenizer, so it needs no downloaded model.

use lattice_inference::error::InferenceError;
use lattice_inference::model::bert::BertModel;

const HIDDEN_SIZE: usize = 4;
const INTERMEDIATE_SIZE: usize = 4;
const NUM_HIDDEN_LAYERS: usize = 1;
const NUM_ATTENTION_HEADS: usize = 1;
const MAX_POSITION_EMBEDDINGS: usize = 32;
const TYPE_VOCAB_SIZE: usize = 2;
// The tokenizer fixture's vocab size: word embeddings are indexed by token id.
const VOCAB_SIZE: usize = 30522;

const TOKENIZER_JSON: &str = include_str!("fixtures/tokenizers/bge-small-en-v1.5/tokenizer.json");

const SHORT_TEXT: &str = "hello world";
const LONG_TEXT: &str = "the quick brown fox jumps over the lazy dog while the small \
    cat watches from the warm window and the rain keeps falling on the quiet street \
    and the children run home before the evening bell rings across the old town square";

fn safetensors_bytes(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut payload: Vec<u8> = Vec::new();
    for (name, shape, values) in tensors {
        assert_eq!(
            values.len(),
            shape.iter().product::<usize>(),
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
    out
}

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.01 * (((i + seed) % 7) as f32 - 3.0))
        .collect()
}

fn config_json() -> String {
    format!(
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
    )
}

fn weights() -> Vec<u8> {
    let h = HIDDEN_SIZE;
    let i = INTERMEDIATE_SIZE;
    let p = "encoder.layer.0";
    let names = |suffix: &str| format!("{p}.{suffix}");
    let owned: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
        (
            "embeddings.word_embeddings.weight".into(),
            vec![VOCAB_SIZE, h],
            fill(VOCAB_SIZE * h, 1),
        ),
        (
            "embeddings.position_embeddings.weight".into(),
            vec![MAX_POSITION_EMBEDDINGS, h],
            fill(MAX_POSITION_EMBEDDINGS * h, 2),
        ),
        (
            "embeddings.token_type_embeddings.weight".into(),
            vec![TYPE_VOCAB_SIZE, h],
            fill(TYPE_VOCAB_SIZE * h, 3),
        ),
        ("embeddings.LayerNorm.weight".into(), vec![h], vec![1.0; h]),
        ("embeddings.LayerNorm.bias".into(), vec![h], vec![0.0; h]),
        (
            names("attention.self.query.weight"),
            vec![h, h],
            fill(h * h, 4),
        ),
        (names("attention.self.query.bias"), vec![h], fill(h, 5)),
        (
            names("attention.self.key.weight"),
            vec![h, h],
            fill(h * h, 6),
        ),
        (names("attention.self.key.bias"), vec![h], fill(h, 7)),
        (
            names("attention.self.value.weight"),
            vec![h, h],
            fill(h * h, 8),
        ),
        (names("attention.self.value.bias"), vec![h], fill(h, 9)),
        (
            names("attention.output.dense.weight"),
            vec![h, h],
            fill(h * h, 10),
        ),
        (names("attention.output.dense.bias"), vec![h], fill(h, 11)),
        (
            names("attention.output.LayerNorm.weight"),
            vec![h],
            vec![1.0; h],
        ),
        (
            names("attention.output.LayerNorm.bias"),
            vec![h],
            vec![0.0; h],
        ),
        (
            names("intermediate.dense.weight"),
            vec![i, h],
            fill(i * h, 12),
        ),
        (names("intermediate.dense.bias"), vec![i], fill(i, 13)),
        (names("output.dense.weight"), vec![h, i], fill(h * i, 14)),
        (names("output.dense.bias"), vec![h], fill(h, 15)),
        (names("output.LayerNorm.weight"), vec![h], vec![1.0; h]),
        (names("output.LayerNorm.bias"), vec![h], vec![0.0; h]),
    ];
    let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = owned
        .iter()
        .map(|(name, shape, values)| (name.as_str(), shape.clone(), values.clone()))
        .collect();
    safetensors_bytes(&tensors)
}

fn load_capped(max_tokens: usize) -> Result<BertModel, InferenceError> {
    BertModel::from_bytes_with_max_tokens(weights(), &config_json(), TOKENIZER_JSON, max_tokens)
}

fn load_default() -> BertModel {
    BertModel::from_bytes(weights(), &config_json(), TOKENIZER_JSON).unwrap()
}

#[test]
fn the_cap_limits_what_the_model_embeds() {
    let capped = load_capped(8).unwrap();
    assert_eq!(capped.tokenizer().max_seq_len(), 8);

    let counts = capped.embedding_token_counts(LONG_TEXT);
    assert!(
        counts.before_truncation > MAX_POSITION_EMBEDDINGS,
        "the long text must outrun both caps, got {} tokens",
        counts.before_truncation
    );
    assert_eq!(counts.embedded, 8);

    let uncapped = load_default();
    assert_eq!(uncapped.tokenizer().max_seq_len(), MAX_POSITION_EMBEDDINGS);
    assert_eq!(
        uncapped.embedding_token_counts(LONG_TEXT).embedded,
        MAX_POSITION_EMBEDDINGS
    );

    // A text under the cap embeds identically; one over it does not.
    assert_eq!(
        capped.encode(SHORT_TEXT).unwrap(),
        uncapped.encode(SHORT_TEXT).unwrap()
    );
    assert_ne!(
        capped.encode(LONG_TEXT).unwrap(),
        uncapped.encode(LONG_TEXT).unwrap()
    );
}

#[test]
fn a_truncated_input_keeps_its_closing_separator() {
    let capped = load_capped(8).unwrap();
    let sep = {
        let empty = capped.tokenizer().tokenize("");
        assert_eq!(empty.real_length, 2, "an empty text is [CLS] [SEP]");
        empty.input_ids[1]
    };
    let long = capped.tokenizer().tokenize(LONG_TEXT);
    assert_eq!(long.real_length, 8);
    assert_eq!(long.input_ids[7], sep);
}

#[test]
fn a_cap_above_the_position_table_is_lowered_to_it() {
    let model = load_capped(10_000).unwrap();
    assert_eq!(model.tokenizer().max_seq_len(), MAX_POSITION_EMBEDDINGS);
    assert_eq!(
        model.embedding_token_counts(LONG_TEXT).embedded,
        MAX_POSITION_EMBEDDINGS
    );
}

#[test]
fn a_cap_without_room_for_the_special_tokens_is_refused() {
    for max_tokens in [0, 1] {
        match load_capped(max_tokens) {
            Err(InferenceError::InvalidInput(_)) => {}
            Err(other) => panic!("max_tokens={max_tokens}: expected InvalidInput, got {other:?}"),
            Ok(_) => panic!("max_tokens={max_tokens}: expected InvalidInput, model loaded"),
        }
    }
    assert_eq!(load_capped(2).unwrap().tokenizer().max_seq_len(), 2);
}

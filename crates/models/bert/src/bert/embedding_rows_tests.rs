use super::BertModel;
use crate::cross_encoder::CrossEncoderModel;
use lattice_inference_core::InferenceError;
use lattice_weights::test_fixtures::bert_namespace as fixture;

use fixture::{HIDDEN, INTERMEDIATE, LAYERS, Tensor, checkpoint, safetensors_bytes};

const VOCAB: &str =
    "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nred\nblue\ngreen\ncat\ndog\nruns\nsleeps\n";

fn long_text() -> String {
    ["red", "blue", "green", "cat", "dog", "runs", "sleeps"]
        .iter()
        .cycle()
        .take(60)
        .copied()
        .collect::<Vec<_>>()
        .join(" ")
}

fn replace(tensors: &mut [Tensor], name: &str, rows: usize) {
    let tensor = tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
        .unwrap();
    *tensor = Tensor::new(name, &[rows, HIDDEN], vec![0.01; rows * HIDDEN]);
}

fn model_directory(
    tensors: &[Tensor],
    vocab: &str,
    vocab_size: usize,
    max_positions: usize,
) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("vocab.txt"), vocab).unwrap();
    let config = serde_json::json!({
        "vocab_size": vocab_size,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 2,
        "intermediate_size": INTERMEDIATE,
        "max_position_embeddings": max_positions,
        "type_vocab_size": 2,
        "layer_norm_eps": 1e-5,
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    std::fs::write(
        dir.path().join("model.safetensors"),
        safetensors_bytes(tensors),
    )
    .unwrap();
    dir
}

fn refusal(dir: &tempfile::TempDir, text: &str) -> String {
    match BertModel::from_directory(dir.path()) {
        Err(InferenceError::InvalidInput(message)) => message,
        Err(other) => panic!("expected InvalidInput, got {other}"),
        Ok(model) => {
            let result = model.encode(text);
            panic!("checkpoint loaded; encode returned {result:?}")
        }
    }
}

#[test]
fn bert_loads_and_encodes_when_tables_cover_the_reachable_range() {
    let dir = model_directory(&checkpoint("", true), VOCAB, 12, 32);
    let model = BertModel::from_directory(dir.path()).unwrap();
    assert_eq!(model.tokenizer().max_seq_len(), 32);
    let vector = model.encode(&long_text()).unwrap();
    assert_eq!(vector.len(), HIDDEN);
    assert!(vector.iter().all(|value| value.is_finite()));
}

#[test]
fn bert_accepts_extra_position_rows() {
    let dir = model_directory(&checkpoint("", true), VOCAB, 12, 16);
    let model = BertModel::from_directory(dir.path()).unwrap();
    assert_eq!(model.tokenizer().max_seq_len(), 16);
    assert_eq!(model.encode(&long_text()).unwrap().len(), HIDDEN);
}

#[test]
fn bert_refuses_position_table_shorter_than_the_encode_range() {
    let dir = model_directory(&checkpoint("", true), VOCAB, 12, 64);
    let message = refusal(&dir, &long_text());
    assert!(message.contains("position_embeddings"), "{message}");
    assert!(
        message.contains("32 rows") && message.contains("64 positions"),
        "{message}"
    );
}

#[test]
fn bert_refuses_word_table_shorter_than_the_tokenizer_vocabulary() {
    let vocab = format!("{VOCAB}extra\n");
    let dir = model_directory(&checkpoint("", true), &vocab, 12, 32);
    let message = refusal(&dir, "extra");
    assert!(message.contains("word_embeddings"), "{message}");
    assert!(
        message.contains("12 rows") && message.contains("13 ids"),
        "{message}"
    );
}

#[test]
fn bert_refuses_an_empty_token_type_table() {
    let mut tensors = checkpoint("", true);
    replace(&mut tensors, "embeddings.token_type_embeddings.weight", 0);
    let dir = model_directory(&tensors, VOCAB, 12, 32);
    let message = refusal(&dir, "red cat");
    assert!(message.contains("token_type_embeddings"), "{message}");
}

#[test]
fn cross_encoder_refuses_a_single_row_token_type_table() {
    let mut tensors = checkpoint("", true);
    replace(&mut tensors, "embeddings.token_type_embeddings.weight", 1);
    let dir = model_directory(&tensors, VOCAB, 12, 32);
    match CrossEncoderModel::from_directory(dir.path()) {
        Err(InferenceError::UnsupportedModel(message)) => {
            assert!(message.contains("token_type_embeddings"), "{message}");
            assert!(message.contains("found 1"), "{message}");
        }
        Err(other) => panic!("expected UnsupportedModel, got {other}"),
        Ok(model) => {
            let score = model.score("red cat", "blue dog");
            panic!("cross-encoder loaded; score returned {score}")
        }
    }
}

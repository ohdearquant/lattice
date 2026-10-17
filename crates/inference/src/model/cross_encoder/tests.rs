use super::*;
use serde_json::json;

const HIDDEN: usize = 4;

type Tensor = (String, Vec<usize>, Vec<f32>);

fn tensor(name: impl Into<String>, shape: &[usize], values: &[f32]) -> Tensor {
    assert_eq!(shape.iter().product::<usize>(), values.len());
    (name.into(), shape.to_vec(), values.to_vec())
}

/// A minimal, self-contained BERT-style checkpoint: no pooler tensors (direct
/// CLS classification), zero-gamma final LayerNorm so the pooled row is a fixed
/// bias vector independent of attention/input. Needs no files from the repo.
fn checkpoint() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("vocab.txt"),
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nquery\nshort\nlong\ndocument\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        json!({
            "vocab_size": 9,
            "hidden_size": HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": HIDDEN,
            "max_position_embeddings": 32,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-5
        })
        .to_string(),
    )
    .unwrap();

    let mut tensors = vec![
        tensor(
            "embeddings.word_embeddings.weight",
            &[9, HIDDEN],
            &[0.0; 36],
        ),
        tensor(
            "embeddings.position_embeddings.weight",
            &[32, HIDDEN],
            &[0.0; 128],
        ),
        tensor(
            "embeddings.token_type_embeddings.weight",
            &[2, HIDDEN],
            &[0.0; 8],
        ),
        tensor("embeddings.LayerNorm.weight", &[HIDDEN], &[1.0; HIDDEN]),
        tensor("embeddings.LayerNorm.bias", &[HIDDEN], &[0.0; HIDDEN]),
        tensor("classifier.weight", &[1, HIDDEN], &[0.8, -0.4, 0.6, -0.3]),
        tensor("classifier.bias", &[1], &[0.15]),
    ];
    for module in [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    ] {
        tensors.push(tensor(
            format!("encoder.layer.0.{module}.weight"),
            &[HIDDEN, HIDDEN],
            &[0.0; HIDDEN * HIDDEN],
        ));
        tensors.push(tensor(
            format!("encoder.layer.0.{module}.bias"),
            &[HIDDEN],
            &[0.0; HIDDEN],
        ));
    }
    tensors.push(tensor(
        "encoder.layer.0.attention.output.LayerNorm.weight",
        &[HIDDEN],
        &[1.0; HIDDEN],
    ));
    tensors.push(tensor(
        "encoder.layer.0.attention.output.LayerNorm.bias",
        &[HIDDEN],
        &[0.0; HIDDEN],
    ));
    // Zero gamma: every final hidden row is exactly beta, so the pooled CLS
    // vector below is fixed regardless of tokenization/attention.
    tensors.push(tensor(
        "encoder.layer.0.output.LayerNorm.weight",
        &[HIDDEN],
        &[0.0; HIDDEN],
    ));
    tensors.push(tensor(
        "encoder.layer.0.output.LayerNorm.bias",
        &[HIDDEN],
        &[0.25, -0.5, 0.75, 1.0],
    ));

    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = payload.len();
        payload.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        header.insert(
            name,
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(payload);
    std::fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
    dir
}

/// (a) score_batch must be bit-identical to N sequential score() calls.
#[test]
fn score_batch_is_bit_identical_to_sequential_score_calls() {
    let dir = checkpoint();
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();
    let documents = ["short", "long document", "", "query document pair"];

    let batch = model.score_batch("query", &documents);
    let sequential: Vec<f32> = documents
        .iter()
        .map(|doc| model.score("query", doc))
        .collect();

    assert_eq!(batch.len(), sequential.len());
    for (index, (&b, &s)) in batch.iter().zip(&sequential).enumerate() {
        assert_eq!(
            b.to_bits(),
            s.to_bits(),
            "document {index}: score_batch={b} score()={s}"
        );
    }
}

/// (b) an empty document list returns an empty vec.
#[test]
fn score_batch_empty_documents_returns_empty_vec() {
    let dir = checkpoint();
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();
    let documents: [&str; 0] = [];

    let scores = model.score_batch("query", &documents);

    assert!(scores.is_empty());
}

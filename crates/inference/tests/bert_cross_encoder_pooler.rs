//! Cross-encoder pooler regressions using a tiny, analytically controlled BERT.

use lattice_inference::CrossEncoderModel;
use lattice_inference::error::InferenceError;
use lattice_inference::lora_hook::NoopLoraHook;
use serde_json::json;

const HIDDEN: usize = 4;
const RAW_CLS: [f32; HIDDEN] = [0.25, -0.5, 0.75, 1.0];
const CLASSIFIER_WEIGHT: [f32; HIDDEN] = [0.8, -0.4, 0.6, -0.3];
const CLASSIFIER_BIAS: f32 = 0.15;
const POOLER_BIAS: [f32; HIDDEN] = [-0.7, 0.3, 1.1, -0.2];
const ZERO_WEIGHT: [f32; HIDDEN * HIDDEN] = [0.0; HIDDEN * HIDDEN];
const ASYMMETRIC_WEIGHT: [f32; HIDDEN * HIDDEN] = [
    0.25, -0.75, 0.5, 0.125, 1.0, 0.2, -0.3, 0.4, -0.5, 0.25, 0.7, -0.2, 0.3, -0.8, 0.15, 0.6,
];
const DOCUMENTS: &[&str] = &["short", "long document", ""];

type Tensor = (String, Vec<usize>, Vec<f32>);

fn tensor(name: impl Into<String>, shape: &[usize], values: &[f32]) -> Tensor {
    assert_eq!(shape.iter().product::<usize>(), values.len());
    (name.into(), shape.to_vec(), values.to_vec())
}

fn checkpoint(pooler_tensors: Vec<Tensor>) -> tempfile::TempDir {
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
        tensor("classifier.weight", &[1, HIDDEN], &CLASSIFIER_WEIGHT),
        tensor("classifier.bias", &[1], &[CLASSIFIER_BIAS]),
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
            &ZERO_WEIGHT,
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
    // Zero gamma makes every final hidden row exactly beta, independent of
    // attention, normalization rounding, token positions, and document length.
    tensors.push(tensor(
        "encoder.layer.0.output.LayerNorm.weight",
        &[HIDDEN],
        &[0.0; HIDDEN],
    ));
    tensors.push(tensor(
        "encoder.layer.0.output.LayerNorm.bias",
        &[HIDDEN],
        &RAW_CLS,
    ));
    tensors.extend(pooler_tensors);

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

fn pooler_tensors(weight: &[f32], bias: &[f32]) -> Vec<Tensor> {
    vec![
        tensor("pooler.dense.weight", &[HIDDEN, HIDDEN], weight),
        tensor("pooler.dense.bias", &[HIDDEN], bias),
    ]
}

fn classifier_probability(hidden: &[f64]) -> f64 {
    let logit = f64::from(CLASSIFIER_BIAS)
        + CLASSIFIER_WEIGHT
            .iter()
            .zip(hidden)
            .map(|(&weight, &value)| f64::from(weight) * value)
            .sum::<f64>();
    1.0 / (1.0 + (-logit).exp())
}

fn pooler_probability(weight: &[f32], bias: &[f32]) -> f64 {
    let pooled: Vec<f64> = weight
        .chunks_exact(HIDDEN)
        .zip(bias)
        .map(|(row, &bias)| {
            (row.iter()
                .zip(RAW_CLS)
                .map(|(&weight, value)| f64::from(weight) * f64::from(value))
                .sum::<f64>()
                + f64::from(bias))
            .tanh()
        })
        .collect();
    classifier_probability(&pooled)
}

fn assert_score(actual: f32, expected: f64) {
    assert!(actual.is_finite(), "non-finite score: {actual}");
    assert!(
        (f64::from(actual) - expected).abs() < 2e-6,
        "score {actual} differs from analytic reference {expected}"
    );
}

fn assert_batch(scores: &[f32], expected: f64) {
    assert_eq!(scores.len(), DOCUMENTS.len());
    for &score in scores {
        assert_score(score, expected);
    }
}

#[test]
fn zero_pooler_scalar_score_uses_tanh_bias() {
    let dir = checkpoint(pooler_tensors(&ZERO_WEIGHT, &POOLER_BIAS));
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();
    let expected = pooler_probability(&ZERO_WEIGHT, &POOLER_BIAS);
    assert_score(model.score("query", "short"), expected);
}

#[test]
fn zero_pooler_batch_score_uses_tanh_bias() {
    let dir = checkpoint(pooler_tensors(&ZERO_WEIGHT, &POOLER_BIAS));
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();
    let expected = pooler_probability(&ZERO_WEIGHT, &POOLER_BIAS);
    assert_batch(&model.score_batch("query", DOCUMENTS), expected);
}

#[test]
fn zero_pooler_hooked_score_uses_tanh_bias() {
    let dir = checkpoint(pooler_tensors(&ZERO_WEIGHT, &POOLER_BIAS));
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();
    let expected = pooler_probability(&ZERO_WEIGHT, &POOLER_BIAS);
    assert_score(
        model
            .score_with_hook("query", "short", &NoopLoraHook)
            .unwrap(),
        expected,
    );
}

#[test]
fn zero_pooler_hooked_batch_score_uses_tanh_bias() {
    let dir = checkpoint(pooler_tensors(&ZERO_WEIGHT, &POOLER_BIAS));
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();
    let expected = pooler_probability(&ZERO_WEIGHT, &POOLER_BIAS);
    assert_batch(
        &model
            .score_batch_with_hook("query", DOCUMENTS, &NoopLoraHook)
            .unwrap(),
        expected,
    );
}

#[test]
fn asymmetric_pooler_uses_output_rows_before_classifier() {
    let dir = checkpoint(pooler_tensors(&ASYMMETRIC_WEIGHT, &POOLER_BIAS));
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();
    let expected = pooler_probability(&ASYMMETRIC_WEIGHT, &POOLER_BIAS);
    let transposed: Vec<f32> = (0..HIDDEN * HIDDEN)
        .map(|index| ASYMMETRIC_WEIGHT[(index % HIDDEN) * HIDDEN + index / HIDDEN])
        .collect();
    let wrong_orientation = pooler_probability(&transposed, &POOLER_BIAS);
    let direct_cls = classifier_probability(&RAW_CLS.map(f64::from));
    assert!((expected - wrong_orientation).abs() > 0.01);
    assert!((expected - direct_cls).abs() > 0.01);
    assert_score(model.score("query", "short"), expected);
    assert_score(
        model
            .score_with_hook("query", "short", &NoopLoraHook)
            .unwrap(),
        expected,
    );
    assert_batch(&model.score_batch("query", DOCUMENTS), expected);
    assert_batch(
        &model
            .score_batch_with_hook("query", DOCUMENTS, &NoopLoraHook)
            .unwrap(),
        expected,
    );
}

#[test]
fn absent_pooler_preserves_direct_cls_scores() {
    let dir = checkpoint(Vec::new());
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();
    let expected = classifier_probability(&RAW_CLS.map(f64::from));
    assert_score(model.score("query", "short"), expected);
    assert_score(
        model
            .score_with_hook("query", "short", &NoopLoraHook)
            .unwrap(),
        expected,
    );
    assert_batch(&model.score_batch("query", DOCUMENTS), expected);
    assert_batch(
        &model
            .score_batch_with_hook("query", DOCUMENTS, &NoopLoraHook)
            .unwrap(),
        expected,
    );
}

#[test]
fn pooler_weight_without_bias_is_rejected() {
    let dir = checkpoint(vec![tensor(
        "pooler.dense.weight",
        &[HIDDEN, HIDDEN],
        &ASYMMETRIC_WEIGHT,
    )]);
    let result = CrossEncoderModel::from_directory(dir.path());
    assert!(result.is_err(), "partial pooler must not load successfully");
    assert!(result.err().unwrap().to_string().contains("pooler"));
}

#[test]
fn pooler_bias_without_weight_is_rejected() {
    let dir = checkpoint(vec![tensor("pooler.dense.bias", &[HIDDEN], &POOLER_BIAS)]);
    let result = CrossEncoderModel::from_directory(dir.path());
    assert!(result.is_err(), "partial pooler must not load successfully");
    assert!(result.err().unwrap().to_string().contains("pooler"));
}

#[test]
fn malformed_pooler_weight_shape_is_rejected() {
    let dir = checkpoint(vec![
        tensor(
            "pooler.dense.weight",
            &[HIDDEN * HIDDEN],
            &ASYMMETRIC_WEIGHT,
        ),
        tensor("pooler.dense.bias", &[HIDDEN], &POOLER_BIAS),
    ]);
    assert!(matches!(
        CrossEncoderModel::from_directory(dir.path()),
        Err(InferenceError::ShapeMismatch { name, .. }) if name == "pooler.dense.weight"
    ));
}

#[test]
fn malformed_pooler_bias_shape_is_rejected() {
    let dir = checkpoint(vec![
        tensor("pooler.dense.weight", &[HIDDEN, HIDDEN], &ASYMMETRIC_WEIGHT),
        tensor("pooler.dense.bias", &[1, HIDDEN], &POOLER_BIAS),
    ]);
    assert!(matches!(
        CrossEncoderModel::from_directory(dir.path()),
        Err(InferenceError::ShapeMismatch { name, .. }) if name == "pooler.dense.bias"
    ));
}

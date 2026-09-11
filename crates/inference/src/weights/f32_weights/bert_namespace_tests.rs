use super::SafetensorsFile;
use crate::attention::AttentionBuffers;
use crate::error::InferenceError;
use crate::forward::cpu::matmul_bt;
use crate::model::CrossEncoderModel;

const HIDDEN: usize = 4;
const INTERMEDIATE: usize = 6;
const LAYERS: usize = 2;
const VOCAB: &str =
    "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nred\nblue\ngreen\ncat\ndog\nruns\nsleeps\n";
const PAIRS: [(&str, &str); 3] = [
    ("red cat", "blue dog"),
    ("green dog runs", "red cat sleeps"),
    ("blue dog", "red cat"),
];

#[derive(Clone)]
struct Tensor {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl Tensor {
    fn new(name: &str, shape: &[usize], values: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len(), "{name}");
        Self {
            name: name.to_owned(),
            shape: shape.to_vec(),
            values,
        }
    }
}

fn fill(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 17 + seed * 13) % 37) as f32 - 18.0) * 0.013)
        .collect()
}

fn checkpoint(prefix: &str, pooler: bool) -> Vec<Tensor> {
    let mut tensors = vec![
        Tensor::new(
            "embeddings.word_embeddings.weight",
            &[12, HIDDEN],
            fill(12 * HIDDEN, 1),
        ),
        Tensor::new(
            "embeddings.position_embeddings.weight",
            &[32, HIDDEN],
            fill(32 * HIDDEN, 2),
        ),
        Tensor::new(
            "embeddings.token_type_embeddings.weight",
            &[2, HIDDEN],
            fill(2 * HIDDEN, 3),
        ),
        Tensor::new(
            "embeddings.LayerNorm.weight",
            &[HIDDEN],
            vec![0.91, 1.13, 0.87, 1.07],
        ),
        Tensor::new("embeddings.LayerNorm.bias", &[HIDDEN], fill(HIDDEN, 4)),
    ];
    for layer in 0..LAYERS {
        for (index, (suffix, shape)) in [
            ("attention.self.query.weight", vec![HIDDEN, HIDDEN]),
            ("attention.self.query.bias", vec![HIDDEN]),
            ("attention.self.key.weight", vec![HIDDEN, HIDDEN]),
            ("attention.self.key.bias", vec![HIDDEN]),
            ("attention.self.value.weight", vec![HIDDEN, HIDDEN]),
            ("attention.self.value.bias", vec![HIDDEN]),
            ("attention.output.dense.weight", vec![HIDDEN, HIDDEN]),
            ("attention.output.dense.bias", vec![HIDDEN]),
            ("attention.output.LayerNorm.weight", vec![HIDDEN]),
            ("attention.output.LayerNorm.bias", vec![HIDDEN]),
            ("intermediate.dense.weight", vec![INTERMEDIATE, HIDDEN]),
            ("intermediate.dense.bias", vec![INTERMEDIATE]),
            ("output.dense.weight", vec![HIDDEN, INTERMEDIATE]),
            ("output.dense.bias", vec![HIDDEN]),
            ("output.LayerNorm.weight", vec![HIDDEN]),
            ("output.LayerNorm.bias", vec![HIDDEN]),
        ]
        .into_iter()
        .enumerate()
        {
            let mut values = fill(shape.iter().product(), 5 + index + 19 * layer);
            if suffix.ends_with("LayerNorm.weight") {
                for value in &mut values {
                    *value += 1.0;
                }
            }
            tensors.push(Tensor::new(
                &format!("encoder.layer.{layer}.{suffix}"),
                &shape,
                values,
            ));
        }
    }
    if pooler {
        tensors.extend([
            Tensor::new(
                "pooler.dense.weight",
                &[HIDDEN, HIDDEN],
                vec![
                    0.31, -0.22, 0.07, 0.19, -0.11, 0.41, 0.23, -0.09, 0.17, 0.03, -0.37, 0.29,
                    0.13, -0.27, 0.11, 0.43,
                ],
            ),
            Tensor::new(
                "pooler.dense.bias",
                &[HIDDEN],
                vec![0.07, -0.13, 0.05, 0.11],
            ),
        ]);
    }
    for tensor in &mut tensors {
        tensor.name = format!("{prefix}{}", tensor.name);
    }
    tensors.extend([
        Tensor::new(
            "classifier.weight",
            &[1, HIDDEN],
            vec![0.43, -0.71, 0.29, 0.58],
        ),
        Tensor::new("classifier.bias", &[1], vec![-0.17]),
    ]);
    tensors
}

fn safetensors_bytes(tensors: &[Tensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for tensor in tensors {
        let start = payload.len();
        for value in &tensor.values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        assert!(
            header
                .insert(
                    tensor.name.clone(),
                    serde_json::json!({
                        "dtype": "F32",
                        "shape": tensor.shape,
                        "data_offsets": [start, payload.len()],
                    }),
                )
                .is_none(),
            "duplicate fixture tensor {}",
            tensor.name,
        );
    }
    let mut header_bytes = serde_json::to_vec(&header).unwrap();
    header_bytes.resize(header_bytes.len().next_multiple_of(8), b' ');
    let mut bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header_bytes);
    bytes.extend(payload);
    bytes
}

fn model_directory(tensors: &[Tensor]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("vocab.txt"), VOCAB).unwrap();
    let config = serde_json::json!({
        "vocab_size": 12,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 2,
        "intermediate_size": INTERMEDIATE,
        "max_position_embeddings": 32,
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

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn evaluate(tensors: &[Tensor]) -> Vec<(Vec<u32>, u32, u32)> {
    let dir = model_directory(tensors);
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();
    let file = SafetensorsFile::open(&dir.path().join("model.safetensors")).unwrap();
    let classifier = file.load_cross_encoder_weights(HIDDEN).unwrap();
    PAIRS
        .iter()
        .map(|&(query, document)| {
            let bert = model.bert();
            let input = bert.tokenizer().tokenize_pair(query, document);
            assert!(input.real_length > 3);
            assert!(input.token_type_ids[..input.real_length].contains(&1));
            let mut buffers = AttentionBuffers::new(input.real_length, HIDDEN, 2, INTERMEDIATE);
            let hidden = bert.forward_tokenized(&input, &mut buffers);
            let cls = &hidden[..HIDDEN];
            let (weight, bias) = bert.pooler_parameters();
            let mut pooled = vec![0.0; HIDDEN];
            let logit = if weight.is_empty() {
                classifier.logit(cls)
            } else {
                matmul_bt(cls, weight, &mut pooled, 1, HIDDEN, HIDDEN);
                for (value, bias) in pooled.iter_mut().zip(bias) {
                    *value = (*value + bias).tanh();
                }
                classifier.logit(&pooled)
            };
            assert!(logit.is_finite());
            let sigmoid = if logit >= 0.0 {
                1.0 / (1.0 + (-logit).exp())
            } else {
                let exp = logit.exp();
                exp / (1.0 + exp)
            };
            let score = model.score(query, document);
            assert!(score > 0.0 && score < 1.0);
            assert_eq!(score.to_bits(), sigmoid.to_bits());
            (bits(&hidden), logit.to_bits(), score.to_bits())
        })
        .collect()
}

fn rename(tensors: &mut [Tensor], old: &str, new: &str) {
    tensors
        .iter_mut()
        .find(|tensor| tensor.name == old)
        .unwrap()
        .name = new.to_owned();
}

fn assert_mixed(tensors: &[Tensor]) {
    let file = SafetensorsFile::from_bytes(safetensors_bytes(tensors)).unwrap();
    let error = file.load_bert_weights(LAYERS, HIDDEN).unwrap_err();
    assert!(
        matches!(&error, InferenceError::InvalidSafetensors(message)
            if message.contains("mixed BERT encoder namespaces")),
        "expected mixed namespace error, got {error:?}",
    );
}

#[test]
fn bert_namespace_flat_and_prefixed_logits_and_scores_are_bit_identical() {
    let flat = evaluate(&checkpoint("", true));
    assert!(flat.windows(2).any(|pair| pair[0].1 != pair[1].1));
    for prefix in [
        "bert.",
        "custom_encoder.",
        "model.backbone.",
        "wrapper.embeddings.",
        "wrapper.encoder.layer.",
        "wrapper.pooler.",
    ] {
        assert_eq!(flat, evaluate(&checkpoint(prefix, true)), "{prefix}");
    }
}

#[test]
fn bert_namespace_absent_pooler_preserves_direct_cls_scores() {
    let flat = evaluate(&checkpoint("", false));
    for prefix in ["bert.", "custom.encoder."] {
        let tensors = checkpoint(prefix, false);
        let file = SafetensorsFile::from_bytes(safetensors_bytes(&tensors)).unwrap();
        let weights = file.load_bert_weights(LAYERS, HIDDEN).unwrap();
        assert!(weights.pooler_weight.data.is_empty());
        assert!(weights.pooler_bias.data.is_empty());
        assert_eq!(flat, evaluate(&tensors), "{prefix}");
    }
    let with_pooler = evaluate(&checkpoint("", true));
    assert!(flat.iter().zip(with_pooler).any(|(a, b)| a.1 != b.1));
}

#[test]
fn bert_namespace_classifier_stays_top_level_even_with_a_namespaced_shadow() {
    let expected = evaluate(&checkpoint("", true));
    for prefix in [
        "bert.",
        "wrapper.embeddings.",
        "wrapper.encoder.layer.",
        "wrapper.pooler.",
    ] {
        let mut tensors = checkpoint(prefix, true);
        tensors.extend([
            Tensor::new(
                &format!("{prefix}classifier.weight"),
                &[1, HIDDEN],
                vec![9.0; HIDDEN],
            ),
            Tensor::new(&format!("{prefix}classifier.bias"), &[1], vec![12.0]),
        ]);
        let file = SafetensorsFile::from_bytes(safetensors_bytes(&tensors)).unwrap();
        let classifier = file.load_cross_encoder_weights(HIDDEN).unwrap();
        assert_eq!(
            bits(&classifier.classifier_weight),
            bits(&[0.43, -0.71, 0.29, 0.58])
        );
        assert_eq!(classifier.classifier_bias.to_bits(), (-0.17_f32).to_bits());
        assert_eq!(expected, evaluate(&tensors), "{prefix}");
    }
}

#[test]
fn bert_namespace_namespaced_only_classifier_is_rejected() {
    let mut tensors = checkpoint("bert.", true);
    rename(&mut tensors, "classifier.weight", "bert.classifier.weight");
    rename(&mut tensors, "classifier.bias", "bert.classifier.bias");
    let dir = model_directory(&tensors);
    let error = CrossEncoderModel::from_directory(dir.path()).err().unwrap();
    assert!(
        matches!(&error, InferenceError::MissingTensor(name) if name == "classifier.weight"),
        "{error:?}",
    );
}

#[test]
fn bert_namespace_rejects_complete_encoder_with_alternate_stray_tensor() {
    for (prefix, stray_prefix) in [("", "bert."), ("bert.", ""), ("bert.", "other.")] {
        let mut tensors = checkpoint(prefix, true);
        let mut stray = tensors
            .iter()
            .find(|tensor| tensor.name == format!("{prefix}embeddings.position_embeddings.weight"))
            .unwrap()
            .clone();
        stray.name = format!("{stray_prefix}embeddings.position_embeddings.weight");
        tensors.push(stray);
        assert_mixed(&tensors);
    }
}

#[test]
fn bert_namespace_rejects_two_complete_encoder_namespaces() {
    for (prefix, second) in [("", "bert."), ("bert.", "other.encoder.")] {
        let mut tensors = checkpoint(prefix, true);
        tensors.extend(
            checkpoint(second, true)
                .into_iter()
                .filter(|tensor| !tensor.name.starts_with("classifier.")),
        );
        assert_mixed(&tensors);
    }
}

#[test]
fn bert_namespace_rejects_split_embedding_and_layer_namespaces() {
    for suffix in [
        "embeddings.LayerNorm.bias",
        "encoder.layer.0.attention.self.query.weight",
        "encoder.layer.1.output.LayerNorm.bias",
    ] {
        let mut tensors = checkpoint("bert.", true);
        rename(&mut tensors, &format!("bert.{suffix}"), suffix);
        assert_mixed(&tensors);
    }
}

#[test]
fn bert_namespace_rejects_misplaced_and_split_pooler() {
    for misplaced in [
        &["pooler.dense.weight"][..],
        &["pooler.dense.bias"][..],
        &["pooler.dense.weight", "pooler.dense.bias"][..],
    ] {
        let mut tensors = checkpoint("bert.", true);
        for suffix in misplaced {
            rename(&mut tensors, &format!("bert.{suffix}"), suffix);
        }
        assert_mixed(&tensors);
    }
}

#[test]
fn bert_namespace_partial_checkpoint_reports_qualified_missing_tensor() {
    for prefix in ["", "bert.", "model.encoder."] {
        for suffix in [
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
            "embeddings.LayerNorm.bias",
            "encoder.layer.0.intermediate.dense.weight",
            "encoder.layer.1.attention.self.query.weight",
            "encoder.layer.1.output.LayerNorm.bias",
        ] {
            let mut tensors = checkpoint(prefix, true);
            let missing = format!("{prefix}{suffix}");
            tensors.retain(|tensor| tensor.name != missing);
            let file = SafetensorsFile::from_bytes(safetensors_bytes(&tensors)).unwrap();
            let error = file.load_bert_weights(LAYERS, HIDDEN).unwrap_err();
            assert!(
                matches!(&error, InferenceError::MissingTensor(name) if name == &missing),
                "missing {missing}: {error:?}",
            );
        }
    }
}

#[test]
fn bert_namespace_partial_pooler_keeps_existing_cross_encoder_rejection() {
    for prefix in ["", "bert."] {
        for suffix in ["pooler.dense.weight", "pooler.dense.bias"] {
            let mut tensors = checkpoint(prefix, true);
            tensors.retain(|tensor| tensor.name != format!("{prefix}{suffix}"));
            let dir = model_directory(&tensors);
            let error = CrossEncoderModel::from_directory(dir.path()).err().unwrap();
            assert!(
                matches!(&error, InferenceError::UnsupportedModel(message)
                    if message.contains("pooler requires both")),
                "{error:?}",
            );
        }
    }
}

use super::CrossEncoderModel;
use crate::attention::AttentionBuffers;
use crate::error::InferenceError;
use crate::forward::cpu::matmul_bt;
use crate::weights::SafetensorsFile;

use lattice_weights::test_fixtures::bert_namespace as fixture;

use fixture::{HIDDEN, INTERMEDIATE, LAYERS, Tensor, checkpoint, rename, safetensors_bytes};

const VOCAB: &str =
    "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nred\nblue\ngreen\ncat\ndog\nruns\nsleeps\n";
const PAIRS: [(&str, &str); 3] = [
    ("red cat", "blue dog"),
    ("green dog runs", "red cat sleeps"),
    ("blue dog", "red cat"),
];

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

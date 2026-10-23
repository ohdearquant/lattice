use super::SafetensorsFile;
use lattice_inference_core::InferenceError;

use crate::test_fixtures::bert_namespace as fixture;

use fixture::{HIDDEN, LAYERS, Tensor, checkpoint, rename, safetensors_bytes};

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

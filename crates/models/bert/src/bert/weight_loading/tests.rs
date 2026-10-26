use super::*;
use std::fs::File;
use std::io::Write;

use lattice_weights::test_fixtures::bert_namespace as fixture;

fn temp_path(name: &str) -> tempfile::TempPath {
    tempfile::Builder::new()
        .prefix(name)
        .suffix(".safetensors")
        .tempfile()
        .unwrap()
        .into_temp_path()
}

fn write_classifier_safetensors(
    path: &std::path::Path,
    weight_shape_header: &str,
    weight_values: &[f32],
    bias_value: f32,
    weight_end: usize,
) {
    let bias_end = weight_end + 4; // one f32
    let header = format!(
        r#"{{"classifier.weight":{{{weight_shape_header},"data_offsets":[0,{weight_end}]}},"classifier.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{weight_end},{bias_end}]}}}}"#
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in weight_values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend_from_slice(&bias_value.to_le_bytes());
    let mut file = File::create(path).expect("test setup: create safetensors");
    file.write_all(&bytes)
        .expect("test setup: write safetensors");
}

#[test]
fn test_load_cross_encoder_weights_2d_shape() {
    // classifier.weight [1, 3] — standard HF BertForSequenceClassification shape
    let path = temp_path("lattice_ce_weights_2d");
    write_classifier_safetensors(
        &path,
        r#""dtype":"F32","shape":[1,3]"#,
        &[1.0f32, 2.0, 3.0],
        0.5,
        12,
    );
    let st = SafetensorsFile::open(&path).unwrap();
    let weights = load_classifier_weights(&st, 3).unwrap();
    assert_eq!(weights.classifier_weight, vec![1.0f32, 2.0, 3.0]);
    assert!(
        (weights.classifier_bias - 0.5f32).abs() < 1e-6,
        "bias mismatch: {}",
        weights.classifier_bias
    );
}

#[test]
fn test_load_cross_encoder_weights_1d_shape() {
    // classifier.weight [3] — accepted as converted-checkpoint fallback
    let path = temp_path("lattice_ce_weights_1d");
    write_classifier_safetensors(
        &path,
        r#""dtype":"F32","shape":[3]"#,
        &[4.0f32, 5.0, 6.0],
        1.0,
        12,
    );
    let st = SafetensorsFile::open(&path).unwrap();
    let weights = load_classifier_weights(&st, 3).unwrap();
    assert_eq!(weights.classifier_weight, vec![4.0f32, 5.0, 6.0]);
    assert!(
        (weights.classifier_bias - 1.0f32).abs() < 1e-6,
        "bias mismatch: {}",
        weights.classifier_bias
    );
}

#[test]
fn test_load_cross_encoder_weights_shape_mismatch_error() {
    // classifier.weight [1, 4] with hidden_size=3 must yield ShapeMismatch
    let path = temp_path("lattice_ce_weights_mismatch");
    write_classifier_safetensors(
        &path,
        r#""dtype":"F32","shape":[1,4]"#,
        &[1.0f32, 2.0, 3.0, 4.0],
        0.0,
        16,
    );
    let st = SafetensorsFile::open(&path).unwrap();
    let err = load_classifier_weights(&st, 3).expect_err("shape mismatch should be an error");
    assert!(
        matches!(err, InferenceError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {err:?}"
    );
}

fn source_from(tensors: &[fixture::Tensor]) -> SafetensorsFile {
    SafetensorsFile::from_bytes(fixture::safetensors_bytes(tensors)).unwrap()
}

type ViewSnapshot = (Vec<usize>, Vec<u32>, *const f32);

fn one(view: &Tensor1D<'_>) -> ViewSnapshot {
    (
        vec![view.len],
        view.data.iter().map(|value| value.to_bits()).collect(),
        view.data.as_ptr(),
    )
}

fn two(view: &Tensor2D<'_>) -> ViewSnapshot {
    (
        vec![view.rows, view.cols],
        view.data.iter().map(|value| value.to_bits()).collect(),
        view.data.as_ptr(),
    )
}

fn snapshot(weights: &BertWeights<'_>) -> Vec<ViewSnapshot> {
    let mut views = vec![
        two(&weights.word_embeddings),
        two(&weights.position_embeddings),
        two(&weights.token_type_embeddings),
        one(&weights.embedding_layer_norm_weight),
        one(&weights.embedding_layer_norm_bias),
    ];
    for layer in &weights.layers {
        views.extend([
            two(&layer.query_weight),
            one(&layer.query_bias),
            two(&layer.key_weight),
            one(&layer.key_bias),
            two(&layer.value_weight),
            one(&layer.value_bias),
            two(&layer.attn_output_weight),
            one(&layer.attn_output_bias),
            one(&layer.attn_layer_norm_weight),
            one(&layer.attn_layer_norm_bias),
            two(&layer.ffn_intermediate_weight),
            one(&layer.ffn_intermediate_bias),
            two(&layer.ffn_output_weight),
            one(&layer.ffn_output_bias),
            one(&layer.ffn_layer_norm_weight),
            one(&layer.ffn_layer_norm_bias),
        ]);
    }
    views.extend([two(&weights.pooler_weight), one(&weights.pooler_bias)]);
    views
}

fn assert_missing(error: InferenceError, expected: &str) {
    let InferenceError::MissingTensor(name) = error else {
        panic!("expected MissingTensor, got {error:?}");
    };
    assert_eq!(name, expected);
}

fn assert_shape(
    error: InferenceError,
    expected_name: &str,
    expected_shape: &[usize],
    actual_shape: &[usize],
) {
    let InferenceError::ShapeMismatch {
        name,
        expected,
        actual,
    } = error
    else {
        panic!("expected ShapeMismatch, got {error:?}");
    };
    assert_eq!(name, expected_name);
    assert_eq!(expected.as_slice(), expected_shape);
    assert_eq!(actual.as_slice(), actual_shape);
}

#[allow(deprecated)]
fn bert_errors(source: &SafetensorsFile) -> [InferenceError; 2] {
    [
        source
            .load_bert_weights(fixture::LAYERS, fixture::HIDDEN)
            .unwrap_err(),
        load_weights(source, fixture::LAYERS, fixture::HIDDEN).unwrap_err(),
    ]
}

#[allow(deprecated)]
fn classifier_errors(source: &SafetensorsFile, hidden_size: usize) -> [InferenceError; 2] {
    [
        source.load_cross_encoder_weights(hidden_size).unwrap_err(),
        load_classifier_weights(source, hidden_size).unwrap_err(),
    ]
}

#[test]
#[allow(deprecated)]
fn bert_complete_borrowed_bundle_matches_legacy_across_namespaces_and_poolers() {
    for prefix in ["", "bert.", "wrapper.embeddings."] {
        for pooler in [false, true] {
            let source = source_from(&fixture::checkpoint(prefix, pooler));
            let old = source
                .load_bert_weights(fixture::LAYERS, fixture::HIDDEN)
                .unwrap();
            let new = load_weights(&source, fixture::LAYERS, fixture::HIDDEN).unwrap();
            assert_eq!(snapshot(&new), snapshot(&old));
            assert_eq!(new.layers.len(), 2);
            assert_eq!(
                (new.word_embeddings.rows, new.word_embeddings.cols),
                (12, 4)
            );
            assert_eq!(
                new.embedding_layer_norm_weight.data,
                &[0.91, 1.13, 0.87, 1.07]
            );
            let (data, _) = source
                .get_f32_tensor(&format!("{prefix}embeddings.LayerNorm.weight"))
                .unwrap();
            assert_eq!(new.embedding_layer_norm_weight.data.as_ptr(), data.as_ptr());
            if pooler {
                assert_eq!(new.pooler_bias.data, &[0.07, -0.13, 0.05, 0.11]);
                assert_eq!((new.pooler_weight.rows, new.pooler_weight.cols), (4, 4));
            } else {
                assert!(new.pooler_weight.data.is_empty());
                assert!(new.pooler_bias.data.is_empty());
                assert_eq!(
                    (
                        new.pooler_weight.rows,
                        new.pooler_weight.cols,
                        new.pooler_bias.len
                    ),
                    (0, 0, 0)
                );
            }
        }
    }
}

#[test]
fn bert_literal_fields_and_dimensions_select_every_layer_tensor() {
    let prefix = "literal.encoder.";
    let fields: [(&str, &[usize], &[f32]); 16] = [
        (
            "attention.self.query.weight",
            &[2, 2],
            &[11.0, -12.0, 13.0, -14.0],
        ),
        ("attention.self.query.bias", &[2], &[21.0, -22.0]),
        (
            "attention.self.key.weight",
            &[2, 2],
            &[31.0, -32.0, 33.0, -34.0],
        ),
        ("attention.self.key.bias", &[2], &[41.0, -42.0]),
        (
            "attention.self.value.weight",
            &[2, 2],
            &[51.0, -52.0, 53.0, -54.0],
        ),
        ("attention.self.value.bias", &[2], &[61.0, -62.0]),
        (
            "attention.output.dense.weight",
            &[2, 2],
            &[71.0, -72.0, 73.0, -74.0],
        ),
        ("attention.output.dense.bias", &[2], &[81.0, -82.0]),
        ("attention.output.LayerNorm.weight", &[2], &[91.0, 92.0]),
        ("attention.output.LayerNorm.bias", &[2], &[101.0, -102.0]),
        (
            "intermediate.dense.weight",
            &[3, 2],
            &[111.0, -112.0, 113.0, -114.0, 115.0, -116.0],
        ),
        ("intermediate.dense.bias", &[3], &[121.0, -122.0, 123.0]),
        (
            "output.dense.weight",
            &[2, 3],
            &[131.0, -132.0, 133.0, -134.0, 135.0, -136.0],
        ),
        ("output.dense.bias", &[2], &[141.0, -142.0]),
        ("output.LayerNorm.weight", &[2], &[151.0, 152.0]),
        ("output.LayerNorm.bias", &[2], &[161.0, -162.0]),
    ];
    let mut tensors = vec![
        fixture::Tensor::new(
            &format!("{prefix}embeddings.word_embeddings.weight"),
            &[3, 2],
            vec![0.5, -0.5, 1.0, -1.0, 1.5, -1.5],
        ),
        fixture::Tensor::new(
            &format!("{prefix}embeddings.position_embeddings.weight"),
            &[4, 2],
            vec![2.0, -2.0, 2.5, -2.5, 3.0, -3.0, 3.5, -3.5],
        ),
        fixture::Tensor::new(
            &format!("{prefix}embeddings.token_type_embeddings.weight"),
            &[2, 2],
            vec![4.0, -4.0, 4.5, -4.5],
        ),
        fixture::Tensor::new(
            &format!("{prefix}embeddings.LayerNorm.weight"),
            &[2],
            vec![1.25, 0.75],
        ),
        fixture::Tensor::new(
            &format!("{prefix}embeddings.LayerNorm.bias"),
            &[2],
            vec![-0.25, 0.5],
        ),
    ];
    for &(suffix, shape, values) in &fields {
        tensors.push(fixture::Tensor::new(
            &format!("{prefix}encoder.layer.0.{suffix}"),
            shape,
            values.to_vec(),
        ));
    }
    let source = source_from(&tensors);
    let weights = load_weights(&source, 1, 2).unwrap();
    assert_eq!(
        weights.word_embeddings.data,
        &[0.5, -0.5, 1.0, -1.0, 1.5, -1.5]
    );
    assert_eq!(
        (weights.word_embeddings.rows, weights.word_embeddings.cols),
        (3, 2)
    );
    assert_eq!(
        weights.position_embeddings.data,
        &[2.0, -2.0, 2.5, -2.5, 3.0, -3.0, 3.5, -3.5]
    );
    assert_eq!(
        (
            weights.position_embeddings.rows,
            weights.position_embeddings.cols
        ),
        (4, 2)
    );
    assert_eq!(weights.token_type_embeddings.data, &[4.0, -4.0, 4.5, -4.5]);
    assert_eq!(weights.embedding_layer_norm_weight.data, &[1.25, 0.75]);
    assert_eq!(weights.embedding_layer_norm_bias.data, &[-0.25, 0.5]);
    assert_eq!(weights.layers.len(), 1);
    let observed = snapshot(&weights);
    assert_eq!(observed.len(), 23);
    for ((suffix, shape, values), view) in fields.iter().zip(&observed[5..21]) {
        assert_eq!(view.0.as_slice(), *shape, "{suffix}");
        assert_eq!(
            view.1,
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{suffix}"
        );
        let (data, _) = source
            .get_f32_tensor(&format!("{prefix}encoder.layer.0.{suffix}"))
            .unwrap();
        assert_eq!(view.2, data.as_ptr(), "{suffix}");
    }
}

#[test]
fn bert_namespace_ignores_non_boundary_matches_and_classifier_shadows() {
    for (prefix, shadow) in [
        ("", "notembeddings.word_embeddings.weight"),
        ("bert.", "shadow.embeddings.classifier.weight"),
    ] {
        let mut tensors = fixture::checkpoint(prefix, true);
        tensors.push(fixture::Tensor::new(shadow, &[1], vec![7.0]));
        let source = source_from(&tensors);
        let weights = load_weights(&source, fixture::LAYERS, fixture::HIDDEN).unwrap();
        assert_eq!(
            weights.embedding_layer_norm_weight.data,
            &[0.91, 1.13, 0.87, 1.07]
        );
        assert_eq!(weights.layers.len(), 2);
    }
}

#[test]
fn bert_mixed_namespace_refusal_precedes_missing_tensor_reads() {
    let source = source_from(&[
        fixture::Tensor::new("bert.embeddings.LayerNorm.weight", &[4], vec![1.0; 4]),
        fixture::Tensor::new(
            "other.encoder.layer.0.attention.self.query.weight",
            &[4, 4],
            vec![2.0; 16],
        ),
    ]);
    for error in bert_errors(&source) {
        let InferenceError::InvalidSafetensors(message) = error else {
            panic!("expected namespace refusal before missing word embeddings, got {error:?}");
        };
        assert_eq!(
            message,
            "mixed BERT encoder namespaces: {\"bert.\", \"other.\"} (empty prefix is unprefixed)"
        );
    }
}

#[test]
fn bert_shape_lookup_order_precedes_embedding_shape_validation() {
    let source = source_from(&[
        fixture::Tensor::new(
            "bert.embeddings.word_embeddings.weight",
            &[1, 3],
            vec![1.0, 2.0, 3.0],
        ),
        fixture::Tensor::new(
            "bert.embeddings.token_type_embeddings.weight",
            &[2, 4],
            vec![1.0; 8],
        ),
    ]);
    for error in bert_errors(&source) {
        assert_missing(error, "bert.embeddings.position_embeddings.weight");
    }
}

#[test]
fn bert_global_and_all_layer_field_reads_preserve_literal_order() {
    let suffixes = [
        "attention.self.query.weight",
        "attention.self.query.bias",
        "attention.self.key.weight",
        "attention.self.key.bias",
        "attention.self.value.weight",
        "attention.self.value.bias",
        "attention.output.dense.weight",
        "attention.output.dense.bias",
        "attention.output.LayerNorm.weight",
        "attention.output.LayerNorm.bias",
        "intermediate.dense.weight",
        "intermediate.dense.bias",
        "output.dense.weight",
        "output.dense.bias",
        "output.LayerNorm.weight",
        "output.LayerNorm.bias",
    ];
    let mut pairs = vec![
        (
            "bert.embeddings.word_embeddings.weight".to_string(),
            "bert.embeddings.position_embeddings.weight".to_string(),
        ),
        (
            "bert.embeddings.position_embeddings.weight".to_string(),
            "bert.embeddings.token_type_embeddings.weight".to_string(),
        ),
        (
            "bert.embeddings.LayerNorm.weight".to_string(),
            "bert.embeddings.LayerNorm.bias".to_string(),
        ),
        (
            "bert.encoder.layer.0.attention.self.query.bias".to_string(),
            "bert.encoder.layer.1.attention.self.query.bias".to_string(),
        ),
    ];
    for pair in suffixes.windows(2) {
        pairs.push((
            format!("bert.encoder.layer.1.{}", pair[0]),
            format!("bert.encoder.layer.1.{}", pair[1]),
        ));
    }
    for (first, second) in pairs {
        let mut tensors = fixture::checkpoint("bert.", true);
        tensors.retain(|tensor| tensor.name != first && tensor.name != second);
        let source = source_from(&tensors);
        for error in bert_errors(&source) {
            assert_missing(error, &first);
        }
    }
}

#[test]
#[allow(deprecated)]
fn bert_optional_pooler_absence_and_partial_presence_keep_empty_views() {
    for (weight, bias) in [(false, false), (true, false), (false, true)] {
        let mut tensors = fixture::checkpoint("bert.", false);
        if weight {
            tensors.push(fixture::Tensor::new(
                "bert.pooler.dense.weight",
                &[4, 4],
                vec![0.125; 16],
            ));
        }
        if bias {
            tensors.push(fixture::Tensor::new(
                "bert.pooler.dense.bias",
                &[4],
                vec![0.25; 4],
            ));
        }
        let source = source_from(&tensors);
        for weights in [
            source.load_bert_weights(2, 4).unwrap(),
            load_weights(&source, 2, 4).unwrap(),
        ] {
            if weight {
                assert_eq!(weights.pooler_weight.data, &[0.125; 16]);
                assert_eq!(
                    (weights.pooler_weight.rows, weights.pooler_weight.cols),
                    (4, 4)
                );
            } else {
                assert!(weights.pooler_weight.data.is_empty());
                assert_eq!(
                    (weights.pooler_weight.rows, weights.pooler_weight.cols),
                    (0, 0)
                );
            }
            if bias {
                assert_eq!(weights.pooler_bias.data, &[0.25; 4]);
                assert_eq!(weights.pooler_bias.len, 4);
            } else {
                assert!(weights.pooler_bias.data.is_empty());
                assert_eq!(weights.pooler_bias.len, 0);
            }
        }
    }
}

#[test]
fn bert_present_pooler_errors_preserve_weight_before_bias_order() {
    for malformed_weight in [true, false] {
        let mut tensors = fixture::checkpoint("bert.", false);
        tensors.push(fixture::Tensor::new(
            "bert.pooler.dense.weight",
            if malformed_weight { &[4, 3] } else { &[4, 4] },
            vec![1.0; if malformed_weight { 12 } else { 16 }],
        ));
        tensors.push(fixture::Tensor::new(
            "bert.pooler.dense.bias",
            &[3],
            vec![2.0; 3],
        ));
        let source = source_from(&tensors);
        for error in bert_errors(&source) {
            if malformed_weight {
                assert_shape(error, "bert.pooler.dense.weight", &[4, 4], &[4, 3]);
            } else {
                assert_shape(error, "bert.pooler.dense.bias", &[4], &[3]);
            }
        }
    }
}

#[test]
#[allow(deprecated)]
fn classifier_rank_forms_keep_top_level_literals_and_owned_results() {
    for shape in [&[3][..], &[1, 3][..]] {
        let weights = {
            let source = source_from(&[
                fixture::Tensor::new("classifier.weight", shape, vec![1.5, -2.25, 3.125]),
                fixture::Tensor::new("classifier.bias", &[1], vec![0.5]),
                fixture::Tensor::new("bert.classifier.weight", &[1, 3], vec![9.0; 3]),
                fixture::Tensor::new("bert.classifier.bias", &[1], vec![8.0]),
            ]);
            let old = source.load_cross_encoder_weights(3).unwrap();
            let new = load_classifier_weights(&source, 3).unwrap();
            assert_eq!(new.classifier_weight, old.classifier_weight);
            assert_eq!(new.classifier_bias.to_bits(), old.classifier_bias.to_bits());
            assert_eq!(new.classifier_weight, vec![1.5, -2.25, 3.125]);
            assert_eq!(new.classifier_bias, 0.5);
            new
        };
        assert_eq!(weights.classifier_weight, vec![1.5, -2.25, 3.125]);
        assert_eq!(weights.classifier_bias, 0.5);
    }
}

#[test]
fn classifier_exact_rank_errors_precede_bias_and_preserve_bias_contract() {
    for shape in [&[1, 4][..], &[3, 1][..]] {
        let count = shape.iter().product();
        let source = source_from(&[fixture::Tensor::new(
            "classifier.weight",
            shape,
            vec![1.0; count],
        )]);
        for error in classifier_errors(&source, 3) {
            assert_shape(error, "classifier.weight", &[1, 3], shape);
        }
    }
    let source = source_from(&[
        fixture::Tensor::new("classifier.weight", &[3], vec![1.0, 2.0, 3.0]),
        fixture::Tensor::new("classifier.bias", &[2], vec![4.0, 5.0]),
    ]);
    for error in classifier_errors(&source, 3) {
        assert_shape(error, "classifier.bias", &[1], &[2]);
    }
    let source = source_from(&[fixture::Tensor::new(
        "classifier.weight",
        &[3],
        vec![1.0, 2.0, 3.0],
    )]);
    for error in classifier_errors(&source, 3) {
        assert_missing(error, "classifier.bias");
    }
}

#[test]
fn classifier_namespaced_shadows_cannot_replace_missing_top_level_weights() {
    let source = source_from(&[
        fixture::Tensor::new("bert.classifier.weight", &[1, 3], vec![1.0, 2.0, 3.0]),
        fixture::Tensor::new("bert.classifier.bias", &[1], vec![0.5]),
        fixture::Tensor::new("classifier.bias", &[2], vec![8.0, 9.0]),
    ]);
    for error in classifier_errors(&source, 3) {
        assert_missing(error, "classifier.weight");
    }
}

#[test]
fn bert_family_public_function_paths_preserve_borrowed_and_owned_signatures() {
    let load: for<'a> fn(
        &'a SafetensorsFile,
        usize,
        usize,
    ) -> Result<BertWeights<'a>, InferenceError> = crate::bert::weight_loading::load_weights;
    let classify: fn(&SafetensorsFile, usize) -> Result<CrossEncoderWeights, InferenceError> =
        crate::bert::weight_loading::load_classifier_weights;
    let source = source_from(&fixture::checkpoint("bert.", false));
    let weights = load(&source, 2, 4).unwrap();
    assert_eq!(
        weights.embedding_layer_norm_weight.data,
        &[0.91, 1.13, 0.87, 1.07]
    );
    let classifier = classify(&source, 4).unwrap();
    assert_eq!(classifier.classifier_weight, vec![0.43, -0.71, 0.29, 0.58]);
    assert_eq!(classifier.classifier_bias, -0.17);
}

#[test]
#[allow(deprecated)]
fn bert_zero_dimension_requests_preserve_legacy_preflight_and_errors() {
    let source = source_from(&fixture::checkpoint("bert.", true));
    for weights in [
        source.load_bert_weights(0, 4).unwrap(),
        load_weights(&source, 0, 4).unwrap(),
    ] {
        assert!(weights.layers.is_empty());
        assert_eq!(
            weights.embedding_layer_norm_weight.data,
            &[0.91, 1.13, 0.87, 1.07]
        );
        assert_eq!(weights.pooler_bias.data, &[0.07, -0.13, 0.05, 0.11]);
    }
    for error in [
        source.load_bert_weights(2, 0).unwrap_err(),
        load_weights(&source, 2, 0).unwrap_err(),
    ] {
        assert_shape(
            error,
            "bert.embeddings.word_embeddings.weight",
            &[12, 0],
            &[12, 4],
        );
    }
    let mut tensors = fixture::checkpoint("bert.", true);
    tensors.retain(|tensor| tensor.name != "bert.encoder.layer.0.intermediate.dense.weight");
    let source = source_from(&tensors);
    for error in [
        source.load_bert_weights(0, 4).unwrap_err(),
        load_weights(&source, 0, 4).unwrap_err(),
    ] {
        assert_missing(error, "bert.encoder.layer.0.intermediate.dense.weight");
    }
}

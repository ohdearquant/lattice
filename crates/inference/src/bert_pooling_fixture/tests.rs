use super::*;
use crate::model::BertModel;
use crate::pool::BertPooling;
use crate::weights::SafetensorsFile;

#[test]
fn checkpoint_is_deterministic_and_loads_all_twenty_one_tensors() {
    let parent = tempfile::tempdir().unwrap();
    let first = parent.path().join("first");
    let second = parent.path().join("second");
    write_checkpoint(&first).unwrap();
    write_checkpoint(&second).unwrap();
    let hashes = checkpoint_hashes(&first).unwrap();
    assert_eq!(hashes, checkpoint_hashes(&second).unwrap());
    assert_eq!(hashes.len(), 3);
    assert!(hashes.values().all(|hash| hash.len() == 64));
    assert_eq!(
        fs::metadata(first.join("model.safetensors")).unwrap().len(),
        2_470_632
    );
    let weights = SafetensorsFile::open(&first.join("model.safetensors")).unwrap();
    assert_eq!(weights.tensor_names().len(), TENSORS.len());
    for spec in &TENSORS {
        assert_eq!(weights.tensor_dtype(spec.name), Some("F32"));
        let (values, shape) = weights.get_f32_tensor(spec.name).unwrap();
        assert_eq!(shape, spec.shape);
        assert!(values.iter().all(|v| v.is_finite() && *v != 0.0));
    }
    let (tokens, _) = weights
        .get_f32_tensor("embeddings.word_embeddings.weight")
        .unwrap();
    for left in 0..10 {
        for right in left + 1..10 {
            let a = &tokens[left * DIMENSIONS..(left + 1) * DIMENSIONS];
            let b = &tokens[right * DIMENSIONS..(right + 1) * DIMENSIONS];
            assert!((1..DIMENSIONS).any(|column| a[0] * b[column] != b[0] * a[column]));
        }
    }
    let model = BertModel::from_directory(&first).unwrap();
    assert_eq!(model.dimensions(), DIMENSIONS);
    assert_eq!(model.config().num_hidden_layers, 1);
    assert_eq!(model.config().num_attention_heads, 6);
    assert_eq!(model.config().intermediate_size, 16);
    assert_eq!(model.config().max_position_embeddings, MAX_POSITIONS);
    for (input, ids) in
        INPUTS
            .into_iter()
            .zip([vec![2, 5, 3], vec![2, 5, 6, 3], vec![2, 7, 8, 9, 3]])
    {
        let actual = model.tokenizer().tokenize(input);
        assert_eq!(&actual.input_ids[..actual.real_length], ids);
        assert_eq!(actual.pre_truncation_len, ids.len());
    }
}

#[test]
fn checkpoint_makes_pooling_and_input_order_observable() {
    let parent = tempfile::tempdir().unwrap();
    let dir = parent.path().join("checkpoint");
    write_checkpoint(&dir).unwrap();
    let mut model = BertModel::from_directory(&dir).unwrap();
    let mut by_mode = Vec::new();
    for mode in [BertPooling::CLS, BertPooling::Mean] {
        model.set_pooling(mode);
        let vectors: Vec<_> = INPUTS
            .into_iter()
            .map(|text| model.encode(text).unwrap())
            .collect();
        for vector in &vectors {
            assert_eq!(vector.len(), DIMENSIONS);
            assert!(vector.iter().all(|v| v.is_finite()));
            let norm: f32 = vector.iter().map(|v| v * v).sum();
            assert!((norm - 1.0).abs() < 1e-5);
        }
        for left in 0..vectors.len() {
            for right in left + 1..vectors.len() {
                assert!(
                    vectors[left]
                        .iter()
                        .zip(&vectors[right])
                        .any(|(a, b)| (a - b).abs() >= 1e-3)
                );
            }
        }
        by_mode.push(vectors);
    }
    for input in 0..INPUTS.len() {
        let distance: f32 = by_mode[0][input]
            .iter()
            .zip(&by_mode[1][input])
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(distance.sqrt() >= 1e-3);
    }
}

#[test]
fn checkpoint_refuses_existing_destination_without_replacement() {
    let parent = tempfile::tempdir().unwrap();
    let dir = parent.path().join("checkpoint");
    write_checkpoint(&dir).unwrap();
    let before = checkpoint_hashes(&dir).unwrap();
    assert!(write_checkpoint(&dir).unwrap_err().contains("reserve"));
    assert_eq!(checkpoint_hashes(&dir).unwrap(), before);
    let file = parent.path().join("keep");
    fs::write(&file, b"unchanged").unwrap();
    assert!(write_checkpoint(&file).is_err());
    assert_eq!(fs::read(file).unwrap(), b"unchanged");
    assert!(checkpoint_hashes(&parent.path().join("absent")).is_err());
}

#[test]
fn checkpoint_failed_write_removes_only_owned_files() {
    let parent = tempfile::tempdir().unwrap();
    let keep = parent.path().join("keep");
    fs::write(&keep, b"parent-owned").unwrap();
    let dir = parent.path().join("checkpoint");
    let error = write_files(
        &dir,
        &[
            ("created", b"first".to_vec()),
            ("missing/child", b"second".to_vec()),
        ],
    )
    .unwrap_err();
    assert!(error.contains("create") && error.contains("missing/child"));
    assert!(!dir.exists());
    assert_eq!(fs::read(keep).unwrap(), b"parent-owned");
}

use super::*;
use crate::model::ModelConfig;
use std::path::Path;

fn write_shard(path: &Path, seed: f32) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, n) in [
        ("embed_tokens.weight", vec![6, 32], 192usize),
        ("norm.weight", vec![32], 32usize),
    ] {
        let start = data.len();
        for i in 0..n {
            let value = if name == "norm.weight" {
                1.0 + seed * 0.01
            } else {
                ((i % 17) as f32 - 8.0) * 0.03 + seed * 0.02
            };
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name.into(),
            serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
        );
    }
    let mut json = serde_json::to_vec(&header).unwrap();
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&json);
    file.extend_from_slice(&data);
    std::fs::write(path, file).unwrap();
}

fn checkpoint(dir: &Path, swapped: bool) {
    std::fs::write(dir.join("config.json"), br#"{"vocab_size":6,"hidden_size":32,"num_hidden_layers":0,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":32,"intermediate_size":32,"max_position_embeddings":8}"#).unwrap();
    std::fs::write(
        dir.join("vocab.txt"),
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\n",
    )
    .unwrap();
    std::fs::write(dir.join("inference_config.json"), br#"{"eos_token_id":3}"#).unwrap();
    write_shard(&dir.join("a.safetensors"), 1.0);
    write_shard(&dir.join("b.safetensors"), 2.0);
    let (a, b) = if swapped {
        ("b.safetensors", "a.safetensors")
    } else {
        ("a.safetensors", "b.safetensors")
    };
    let index = serde_json::json!({"weight_map":{"embed_tokens.weight":a,"norm.weight":b}});
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
}

fn service(dir: &Path, mode: IdentityMode) -> NativeEmbeddingService {
    let config = ModelConfig::try_new(EmbeddingModel::Qwen3Embedding0_6B, Some(32)).unwrap();
    let mut service = NativeEmbeddingService::with_model_config(config)
        .unwrap()
        .with_identity_mode(mode)
        .unwrap();
    service.model_directory = Some(dir.to_path_buf());
    service
}

async fn result(service: &NativeEmbeddingService) -> IdentifiedEmbeddings {
    service
        .embed_identified(
            &["a".into()],
            EmbeddingModel::Qwen3Embedding0_6B,
            EmbeddingRole::Generic,
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn qwen_index_mapping_and_copy_bind_identified_vectors() {
    #[cfg(target_os = "macos")]
    let _guard = lattice_metal_runtime::measurement::gpu_test_lock();
    let original = tempfile::tempdir().unwrap();
    let copy = tempfile::tempdir().unwrap();
    let swapped = tempfile::tempdir().unwrap();
    checkpoint(original.path(), false);
    checkpoint(copy.path(), false);
    checkpoint(swapped.path(), true);
    for name in ["a.safetensors", "b.safetensors"] {
        assert_eq!(
            std::fs::read(original.path().join(name)).unwrap(),
            std::fs::read(swapped.path().join(name)).unwrap()
        );
    }
    let first = result(&service(original.path(), IdentityMode::Full)).await;
    let same = result(&service(copy.path(), IdentityMode::Full)).await;
    let changed = result(&service(swapped.path(), IdentityMode::Full)).await;
    assert!(matches!(&first.identity, EmbeddingSpaceIdentity::Known(_)));
    assert!(first.identity.known_match(&same.identity));
    assert_eq!(first.embeddings, same.embeddings);
    assert!(!first.identity.known_match(&changed.identity));
    assert_eq!(first.embeddings[0].len(), 32);
    assert!(first.embeddings[0].iter().all(|value| value.is_finite()));
    assert_eq!(first.backends.len(), 1);
    assert!(first.backends[0].is_some());
    let header = result(&service(original.path(), IdentityMode::Header)).await;
    assert!(matches!(
        header.identity,
        EmbeddingSpaceIdentity::Unavailable
    ));
}

#[tokio::test]
async fn qwen_missing_shard_never_returns_identified_vector() {
    #[cfg(target_os = "macos")]
    let _guard = lattice_metal_runtime::measurement::gpu_test_lock();
    let dir = tempfile::tempdir().unwrap();
    checkpoint(dir.path(), false);
    std::fs::remove_file(dir.path().join("a.safetensors")).unwrap();
    let outcome = service(dir.path(), IdentityMode::Full)
        .embed_identified(
            &["a".into()],
            EmbeddingModel::Qwen3Embedding0_6B,
            EmbeddingRole::Generic,
        )
        .await;
    assert!(outcome.is_err());
}

#[tokio::test]
async fn qwen_single_file_takes_precedence_and_full_digest_detects_weight_changes() {
    #[cfg(target_os = "macos")]
    let _guard = lattice_metal_runtime::measurement::gpu_test_lock();
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let c = tempfile::tempdir().unwrap();
    for dir in [&a, &b, &c] {
        checkpoint(dir.path(), false);
    }
    write_shard(&a.path().join("model.safetensors"), 1.0);
    write_shard(&b.path().join("model.safetensors"), 1.0);
    write_shard(&c.path().join("model.safetensors"), 2.0);
    write_shard(&b.path().join("b.safetensors"), 7.0);
    let first = result(&service(a.path(), IdentityMode::Full)).await;
    let unused_changed = result(&service(b.path(), IdentityMode::Full)).await;
    let weights_changed = result(&service(c.path(), IdentityMode::Full)).await;
    assert!(first.identity.known_match(&unused_changed.identity));
    assert!(!first.identity.known_match(&weights_changed.identity));
}

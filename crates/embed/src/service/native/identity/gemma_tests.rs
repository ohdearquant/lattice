use super::*;
use crate::model::ModelConfig;
use std::path::Path;

#[path = "gemma_fixture.rs"]
mod fixture;

fn service(dir: &Path, width: Option<usize>, mode: IdentityMode) -> NativeEmbeddingService {
    let config = ModelConfig::try_new(EmbeddingModel::EmbeddingGemma2, width).unwrap();
    let mut service = NativeEmbeddingService::with_model_config(config)
        .unwrap()
        .with_identity_mode(mode)
        .unwrap();
    service.model_directory = Some(dir.to_path_buf());
    service
}

async fn embed(service: &NativeEmbeddingService, role: EmbeddingRole) -> IdentifiedEmbeddings {
    service
        .embed_identified(
            &["a b c".to_string()],
            EmbeddingModel::EmbeddingGemma2,
            role,
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn gemma_identified_vectors_bind_full_load_and_effective_width() {
    let dir = tempfile::tempdir().unwrap();
    fixture::write_checkpoint(dir.path());
    let model = service(dir.path(), None, IdentityMode::Full);
    let (generic, concurrent) = tokio::join!(
        embed(&model, EmbeddingRole::Generic),
        embed(&model, EmbeddingRole::Generic),
    );
    assert!(generic.identity.known_match(&concurrent.identity));
    assert_eq!(generic.embeddings, concurrent.embeddings);
    let query = embed(&model, EmbeddingRole::Query).await;
    let passage = embed(&model, EmbeddingRole::Passage).await;
    assert!(
        matches!(&generic.identity, EmbeddingSpaceIdentity::Known(id) if id.mode() == IdentityMode::Full)
    );
    assert!(generic.identity.known_match(&query.identity));
    assert!(generic.identity.known_match(&passage.identity));
    assert_eq!(generic.embeddings.len(), 1);
    assert_eq!(generic.embeddings[0].len(), 64);
    assert!(generic.embeddings[0].iter().all(|v| v.is_finite()));
    let explicit = embed(
        &service(dir.path(), Some(64), IdentityMode::Full),
        EmbeddingRole::Generic,
    )
    .await;
    assert!(generic.identity.known_match(&explicit.identity));
    assert_eq!(generic.embeddings, explicit.embeddings);
    let shorter = embed(
        &service(dir.path(), Some(32), IdentityMode::Full),
        EmbeddingRole::Generic,
    )
    .await;
    assert_eq!(shorter.embeddings[0].len(), 32);
    assert!(!generic.identity.known_match(&shorter.identity));
    let header = embed(
        &service(dir.path(), None, IdentityMode::Header),
        EmbeddingRole::Generic,
    )
    .await;
    assert!(matches!(
        header.identity,
        EmbeddingSpaceIdentity::Unavailable
    ));
    assert_eq!(header.embeddings, generic.embeddings);
}

#[tokio::test]
async fn gemma_full_identity_changes_for_each_consumed_input() {
    let baseline_dir = tempfile::tempdir().unwrap();
    fixture::write_checkpoint(baseline_dir.path());
    let baseline_service = service(baseline_dir.path(), None, IdentityMode::Full);
    let baseline = embed(&baseline_service, EmbeddingRole::Generic).await;
    for kind in ["config", "tokenizer", "weights"] {
        let dir = tempfile::tempdir().unwrap();
        let mut weights = fixture::write_checkpoint(dir.path());
        match kind {
            "config" | "tokenizer" => {
                let name = if kind == "config" {
                    "config.json"
                } else {
                    "tokenizer.json"
                };
                let path = dir.path().join(name);
                let mut bytes = std::fs::read(&path).unwrap();
                bytes.push(b' ');
                std::fs::write(path, bytes).unwrap();
            }
            "weights" => {
                let header_len = u64::from_le_bytes(weights[..8].try_into().unwrap()) as usize;
                weights[8 + header_len] ^= 1;
                std::fs::write(dir.path().join("model.safetensors"), weights).unwrap();
            }
            _ => unreachable!(),
        }
        let changed = embed(
            &service(dir.path(), None, IdentityMode::Full),
            EmbeddingRole::Generic,
        )
        .await;
        assert_eq!(changed.embeddings[0].len(), 64, "{kind}");
        assert!(!baseline.identity.known_match(&changed.identity), "{kind}");
    }
    let again = embed(&baseline_service, EmbeddingRole::Generic).await;
    assert!(baseline.identity.known_match(&again.identity));
    assert_eq!(baseline.embeddings, again.embeddings);
}

#[tokio::test]
async fn gemma_failed_load_returns_no_identified_batch() {
    let dir = tempfile::tempdir().unwrap();
    fixture::write_checkpoint(dir.path());
    std::fs::write(dir.path().join("tokenizer.json"), b"invalid").unwrap();
    let model = service(dir.path(), None, IdentityMode::Full);
    let result = model
        .embed_identified(
            &["a b c".to_string()],
            EmbeddingModel::EmbeddingGemma2,
            EmbeddingRole::Generic,
        )
        .await;
    assert!(result.is_err());
}

fn expected_full_digest(dir: &Path) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    fn framed(out: &mut Vec<u8>, bytes: &[u8]) {
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    let config = std::fs::read(dir.join("config.json")).unwrap();
    let tokenizer = std::fs::read(dir.join("tokenizer.json")).unwrap();
    let weights = std::fs::read(dir.join("model.safetensors")).unwrap();
    let header_len = u64::from_le_bytes(weights[..8].try_into().unwrap());
    let mut recipe = header_len.to_le_bytes().to_vec();
    recipe.extend_from_slice(&weights[8..8 + header_len as usize]);
    recipe.extend_from_slice(&Sha256::digest(&weights));
    let mut out = b"LATTICE_EMBED_SPACE\0".to_vec();
    out.extend_from_slice(&1u16.to_le_bytes());
    out.push(1);
    framed(&mut out, b"embeddinggemma2-native-text-v1");
    framed(&mut out, b"embeddinggemma2-native-text-v1");
    out.extend_from_slice(&3u32.to_le_bytes());
    for (tag, name, size, digest) in [
        (0u8, "config.json", config.len(), Sha256::digest(&config)),
        (
            1u8,
            "tokenizer.json",
            tokenizer.len(),
            Sha256::digest(&tokenizer),
        ),
        (
            2u8,
            "model.safetensors",
            weights.len(),
            Sha256::digest(&recipe),
        ),
    ] {
        out.push(tag);
        framed(&mut out, name.as_bytes());
        out.extend_from_slice(&(size as u64).to_le_bytes());
        out.extend_from_slice(&digest);
    }
    let mut entries: Vec<(&str, Vec<u8>)> = Vec::new();
    for (key, value) in [
        ("output.dimension", 64u64),
        ("input.max_tokens", 8192),
        ("input.bos_token_id", 2),
        ("input.eos_token_id", 1),
    ] {
        let mut encoded = vec![1];
        encoded.extend_from_slice(&value.to_le_bytes());
        entries.push((key, encoded));
    }
    for (key, value) in [
        ("pooling", "mean-all-tokens"),
        ("normalization", "l2-after-prefix-dimension"),
        ("input.truncation_side", "right-preserve-bos-eos"),
        ("input.program", "gemma-bpe-text-bos-eos-v1"),
        ("role.generic", ""),
        ("role.query", "task: search result | query: "),
        ("role.passage", "title: none | text: "),
        (
            "role.compatibility",
            "paired-query-passage;generic-unprefixed;text-only",
        ),
        ("precision", "f32"),
        ("backend", "cpu-runtime-dispatch"),
        ("backend.target_arch", std::env::consts::ARCH),
        ("backend.target_os", std::env::consts::OS),
        ("adapter", "none"),
        ("vision", "unsupported"),
    ] {
        let mut encoded = vec![3];
        framed(&mut encoded, value.as_bytes());
        entries.push((key, encoded));
    }
    let simd = lattice_compute::forward::cpu::simd_config();
    for (key, value) in [
        (
            "backend.x86_gemm",
            lattice_compute::__internal::x86_gemm_dispatch_enabled(),
        ),
        ("backend.avx2", simd.avx2_enabled),
        ("backend.fma", simd.fma_enabled),
        ("backend.avx512f", simd.avx512f_enabled),
        ("backend.neon", simd.neon_enabled),
    ] {
        entries.push((key, vec![0, u8::from(value)]));
    }
    entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (key, value) in entries {
        framed(&mut out, key.as_bytes());
        out.extend_from_slice(&value);
    }
    Sha256::digest(out).into()
}

#[tokio::test]
async fn gemma_identified_digest_matches_independent_load_recipe() {
    let dir = tempfile::tempdir().unwrap();
    fixture::write_checkpoint(dir.path());
    let expected = expected_full_digest(dir.path());
    let result = embed(
        &service(dir.path(), None, IdentityMode::Full),
        EmbeddingRole::Generic,
    )
    .await;
    let EmbeddingSpaceIdentity::Known(id) = result.identity else {
        panic!("expected Full identity");
    };
    assert_eq!(id.mode(), IdentityMode::Full);
    assert_eq!(id.digest(), expected);
}

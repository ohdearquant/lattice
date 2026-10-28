use super::{
    Artifact, ArtifactRole, EmbeddingModel, EmbeddingSpaceDescriptor, EmbeddingSpaceIdentity,
    IdentityMode, KnownEmbeddingSpaceId, PolicyEntry, PolicyValue, Sha256,
};
use lattice_model_embeddinggemma2::model::EmbeddingGemma2Model;
use sha2::Digest;

pub(in crate::service::native) fn identify_gemma(
    model: &EmbeddingGemma2Model,
    output_dim: Option<usize>,
    mode: IdentityMode,
) -> Result<EmbeddingSpaceIdentity, String> {
    if mode != IdentityMode::Full {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    }
    let Some(report) = model.load_report() else {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    };
    if report.weights().content_digest().is_none() {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    }
    let artifacts = vec![
        Artifact {
            role: ArtifactRole::Config,
            logical_identifier: "config.json".into(),
            file_size: report.config().size(),
            recipe_digest: report.config().digest(),
        },
        Artifact {
            role: ArtifactRole::Tokenizer,
            logical_identifier: "tokenizer.json".into(),
            file_size: report.tokenizer().size(),
            recipe_digest: report.tokenizer().digest(),
        },
        Artifact {
            role: ArtifactRole::Safetensors,
            logical_identifier: "model.safetensors".into(),
            file_size: report.weights().file_size(),
            recipe_digest: report.weights().recipe_digest(),
        },
    ];
    let config = model.config();
    let mut policy = Vec::new();
    let mut add = |key: &str, value| {
        policy.push(PolicyEntry {
            key: key.into(),
            value,
        });
    };
    for (key, value) in [
        ("output.dimension", output_dim.unwrap_or(model.dimensions())),
        (
            "input.max_tokens",
            model.max_tokens().ok_or("missing token limit")?,
        ),
        ("input.bos_token_id", config.bos_token_id as usize),
        ("input.eos_token_id", config.eos_token_id as usize),
    ] {
        add(key, PolicyValue::Unsigned(value as u64));
    }
    for (key, value) in [
        ("pooling", "mean-all-tokens"),
        ("normalization", "l2-after-prefix-dimension"),
        ("input.truncation_side", "right-preserve-bos-eos"),
        ("input.program", "gemma-bpe-text-bos-eos-v1"),
        ("role.generic", ""),
        (
            "role.query",
            EmbeddingModel::EmbeddingGemma2
                .query_instruction()
                .unwrap_or(""),
        ),
        (
            "role.passage",
            EmbeddingModel::EmbeddingGemma2
                .document_instruction()
                .unwrap_or(""),
        ),
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
        add(key, PolicyValue::String(value.into()));
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
        add(key, PolicyValue::Boolean(value));
    }
    let descriptor = EmbeddingSpaceDescriptor {
        mode,
        output_revision: "embeddinggemma2-native-text-v1".into(),
        producer_family: "embeddinggemma2-native-text-v1".into(),
        artifacts,
        policy,
    };
    let canonical = descriptor
        .canonical_bytes()
        .map_err(|error| error.to_string())?;
    Ok(EmbeddingSpaceIdentity::Known(
        KnownEmbeddingSpaceId::from_digest(mode, Sha256::digest(canonical).into()),
    ))
}

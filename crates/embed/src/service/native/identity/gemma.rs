use super::{
    Artifact, ArtifactRole, EmbeddingModel, EmbeddingSpaceDescriptor, EmbeddingSpaceIdentity,
    IdentityMode, PolicyEntry, PolicyValue,
};
use lattice_model_embeddinggemma2::model::EmbeddingGemma2Model;

pub(in crate::service::native) fn identify_gemma(
    model: &EmbeddingGemma2Model,
    output_dim: Option<usize>,
    mode: IdentityMode,
) -> Result<EmbeddingSpaceIdentity, String> {
    super::identity_from_descriptor(mode, || gemma_descriptor(model, output_dim, mode))
}

pub(super) fn gemma_descriptor(
    model: &EmbeddingGemma2Model,
    output_dim: Option<usize>,
    mode: IdentityMode,
) -> Result<Option<EmbeddingSpaceDescriptor>, String> {
    if mode != IdentityMode::Full {
        return Ok(None);
    }
    let Some(report) = model.load_report() else {
        return Ok(None);
    };
    if report.weights().content_digest().is_none() {
        return Ok(None);
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
        ("adapter", "none"),
        ("vision", "unsupported"),
    ] {
        add(key, PolicyValue::String(value.into()));
    }
    Ok(Some(EmbeddingSpaceDescriptor {
        mode,
        output_revision: "embeddinggemma2-native-text-v1".into(),
        producer_family: "embeddinggemma2-native-text-v1".into(),
        artifacts,
        policy,
    }))
}

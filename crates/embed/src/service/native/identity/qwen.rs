use super::{
    Artifact, ArtifactRole, EmbeddingModel, EmbeddingSpaceDescriptor, EmbeddingSpaceIdentity,
    IdentityMode, KnownEmbeddingSpaceId, PolicyEntry, PolicyValue, Sha256,
};
use lattice_model_qwen3_embedding::QwenModel;
use lattice_tokenizer::common::TokenizerPolicyValue;
use sha2::Digest;

pub(in crate::service::native) fn identify_qwen(
    model: &QwenModel,
    family: EmbeddingModel,
    mode: IdentityMode,
) -> Result<EmbeddingSpaceIdentity, String> {
    if mode != IdentityMode::Full {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    }
    let Some(report) = model.full_load_report() else {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    };
    let Some(config) = &report.config else {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    };
    let mut artifacts = vec![Artifact {
        role: ArtifactRole::Config,
        logical_identifier: "config.json".into(),
        file_size: config.size(),
        recipe_digest: config.digest(),
    }];
    if let Some(config) = &report.inference_config {
        artifacts.push(Artifact {
            role: ArtifactRole::Config,
            logical_identifier: "inference_config.json".into(),
            file_size: config.size(),
            recipe_digest: config.digest(),
        });
    }
    for (index, input) in report.tokenizer.inputs.iter().enumerate() {
        artifacts.push(Artifact {
            role: ArtifactRole::Tokenizer,
            logical_identifier: format!("{index}:{}", input.logical_identifier),
            file_size: input.bytes.len() as u64,
            recipe_digest: Sha256::digest(&input.bytes).into(),
        });
    }
    if report.tokenizer.inputs.is_empty() {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    }
    match (&report.single, &report.sharded) {
        (Some(single), None) if single.content_digest().is_some() => artifacts.push(Artifact {
            role: ArtifactRole::Safetensors,
            logical_identifier: "model.safetensors".into(),
            file_size: single.file_size(),
            recipe_digest: single.recipe_digest(),
        }),
        (None, Some(sharded)) if !sharded.shards.is_empty() => {
            artifacts.push(Artifact {
                role: ArtifactRole::SafetensorsIndex,
                logical_identifier: "model.safetensors.index.json".into(),
                file_size: sharded.index.size(),
                recipe_digest: sharded.index.digest(),
            });
            for (index, shard) in sharded.shards.values().enumerate() {
                if shard.content_digest().is_none() {
                    return Ok(EmbeddingSpaceIdentity::Unavailable);
                }
                artifacts.push(Artifact {
                    role: ArtifactRole::Safetensors,
                    logical_identifier: format!("shard.{index:04}"),
                    file_size: shard.file_size(),
                    recipe_digest: shard.recipe_digest(),
                });
            }
        }
        _ => return Ok(EmbeddingSpaceIdentity::Unavailable),
    }
    let mut policy = Vec::new();
    let mut add = |key: &str, value| {
        policy.push(PolicyEntry {
            key: key.into(),
            value,
        })
    };
    for (key, value) in [
        ("output.dimension", model.dimensions()),
        ("input.tokenizer_limit", model.tokenizer().max_seq_len()),
        ("input.rope_capacity", model.rope().max_positions()),
        (
            "input.eos_token_id",
            model.effective_inference_config().eos_token_id as usize,
        ),
    ] {
        add(key, PolicyValue::Unsigned(value as u64));
    }
    for (key, value) in [
        ("pooling", "last-token"),
        ("normalization", "l2-after-prefix-dimension"),
        ("input.program", "qwen-tokenizer-then-ensure-eos-v1"),
        ("tokenizer.format", report.tokenizer.format),
        ("role.generic", ""),
        ("role.query", family.query_instruction().unwrap_or("")),
        ("role.passage", family.document_instruction().unwrap_or("")),
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
    for (key, value) in &report.tokenizer.settings {
        let value = match value {
            TokenizerPolicyValue::Boolean(value) => PolicyValue::Boolean(*value),
            TokenizerPolicyValue::Unsigned(value) => PolicyValue::Unsigned(*value),
            TokenizerPolicyValue::String(value) => PolicyValue::String(value.clone()),
            TokenizerPolicyValue::F32(value) => PolicyValue::F32(*value),
        };
        add(&format!("tokenizer.effective.{key}"), value);
    }
    let descriptor = EmbeddingSpaceDescriptor {
        mode,
        output_revision: "qwen3-native-text-v1".into(),
        producer_family: "qwen3-native-text-v1".into(),
        artifacts,
        policy,
    };
    let bytes = descriptor.canonical_bytes().map_err(|e| e.to_string())?;
    Ok(EmbeddingSpaceIdentity::Known(
        KnownEmbeddingSpaceId::from_digest(mode, Sha256::digest(bytes).into()),
    ))
}

//! Declared native output identities bound to the successful load.

mod gemma;
mod qwen;

pub(super) use gemma::identify_gemma;
pub(super) use qwen::identify_qwen;

use super::{LoadedModel, NativeEmbeddingService, prepared_views};
use crate::error::{EmbedError, Result};
use crate::model::EmbeddingModel;
use crate::service::{EmbeddingRole, ValidatedTextBatch};
use lattice_arch::embedding_space::{
    Artifact, ArtifactRole, EmbeddingSpaceDescriptor, KnownEmbeddingSpaceId, PolicyEntry,
    PolicyValue,
};
pub use lattice_arch::embedding_space::{EmbeddingSpaceIdentity, IdentityMode};
use lattice_model_bert::BertModel;
use lattice_weights::consumed::WeightDigestMode;
use sha2::{Digest, Sha256};

/// Revision of BERT native output semantics not already distinguished by inputs.
const OUTPUT_REVISION: &str = "bert-native-output-v2";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IdentityFamily {
    BertSmall,
    BertFull,
    Qwen,
    Gemma,
    Unavailable,
}

pub(super) const fn identity_family(model: EmbeddingModel) -> IdentityFamily {
    match model {
        EmbeddingModel::BgeSmallEnV15 => IdentityFamily::BertSmall,
        EmbeddingModel::BgeBaseEnV15
        | EmbeddingModel::BgeLargeEnV15
        | EmbeddingModel::MultilingualE5Small
        | EmbeddingModel::MultilingualE5Base
        | EmbeddingModel::AllMiniLmL6V2
        | EmbeddingModel::ParaphraseMultilingualMiniLmL12V2 => IdentityFamily::BertFull,
        EmbeddingModel::Qwen3Embedding0_6B | EmbeddingModel::Qwen3Embedding4B => {
            IdentityFamily::Qwen
        }
        EmbeddingModel::EmbeddingGemma2 => IdentityFamily::Gemma,
        EmbeddingModel::TextEmbedding3Small => IdentityFamily::Unavailable,
    }
}

fn pooling_name(pooling: lattice_compute::pool::BertPooling) -> Option<&'static str> {
    match pooling {
        lattice_compute::pool::BertPooling::CLS => Some("cls"),
        lattice_compute::pool::BertPooling::Mean => Some("mean"),
        _ => None,
    }
}

/// Backend that produced an identified embedding, when known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingBackend {
    /// CPU transformer forward.
    Cpu,
    /// Metal transformer forward.
    Metal,
}

/// A newly computed batch and its identity from one model/policy snapshot.
#[derive(Clone, Debug)]
pub struct IdentifiedEmbeddings {
    /// Vectors produced by this call.
    pub embeddings: Vec<Vec<f32>>,
    /// Declared space; Unavailable never establishes a known match.
    pub identity: EmbeddingSpaceIdentity,
    /// Actual execution backend for each vector, or None when unreported.
    pub backends: Vec<Option<EmbeddingBackend>>,
}

pub(super) const fn weight_mode(mode: IdentityMode) -> WeightDigestMode {
    match mode {
        IdentityMode::Full => WeightDigestMode::Full,
        IdentityMode::Header => WeightDigestMode::Header,
    }
}

const fn identity_mode(mode: WeightDigestMode) -> IdentityMode {
    match mode {
        WeightDigestMode::Full => IdentityMode::Full,
        WeightDigestMode::Header => IdentityMode::Header,
    }
}

const _: () = {
    assert!(IdentityMode::Header.tag() == 0);
    assert!(IdentityMode::Full.tag() == 1);
    assert!(matches!(
        weight_mode(IdentityMode::Full),
        WeightDigestMode::Full
    ));
    assert!(matches!(
        weight_mode(IdentityMode::Header),
        WeightDigestMode::Header
    ));
    assert!(matches!(
        identity_mode(WeightDigestMode::Full),
        IdentityMode::Full
    ));
    assert!(matches!(
        identity_mode(WeightDigestMode::Header),
        IdentityMode::Header
    ));
};

impl NativeEmbeddingService {
    /// Select evidence before any lazy load is attempted.
    ///
    /// Full is the default. Explicit Header opt-in omits weight-data digests:
    /// header-preserving weight-value edits are undetected.
    pub fn with_identity_mode(mut self, mode: IdentityMode) -> Result<Self> {
        if self
            .initialization_started
            .load(std::sync::atomic::Ordering::Acquire)
            || self.model.get().is_some()
        {
            return Err(EmbedError::InvalidInput(
                "identity mode must be selected before loading".into(),
            ));
        }
        self.identity_mode = mode;
        Ok(self)
    }

    /// Load the BGE-small native producer from a supplied directory, lazily.
    ///
    /// Uses the ordinary tokenizer/config/weight interpretation and Full evidence.
    /// Recipe identity describes consumed contents, not the directory name.
    pub fn with_bge_small_directory(directory: impl Into<std::path::PathBuf>) -> Self {
        let mut service = Self::with_model(EmbeddingModel::BgeSmallEnV15);
        service.model_directory = Some(directory.into());
        service
    }

    /// Compute vectors and bind their declared identity to the same successful load.
    ///
    /// BGE, multilingual E5, MiniLM, Qwen3 and the native EmbeddingGemma 2 text producer declare recipes.
    /// Other producers, missing config origins and unreported preloaded models
    /// return Unavailable. BGE base/large, E5, MiniLM, Qwen3 and EmbeddingGemma 2 are available only in Full mode.
    /// Full-mode Qwen3 identified requests bypass legacy caches and report the actual backend per vector.
    /// Header-mode Qwen3 uses the ordinary cache and reports no per-vector backend.
    /// Execution backend is result metadata and is not part of these identity recipes.
    /// Generic has no prefix; Query/Passage share one paired text policy. Neither
    /// declared recipe covers image inputs.
    pub async fn embed_identified(
        &self,
        texts: &[String],
        model: EmbeddingModel,
        role: EmbeddingRole,
    ) -> Result<IdentifiedEmbeddings> {
        let texts = ValidatedTextBatch::new(texts)?;
        let prefixed = role
            .instruction(model)
            .map(|prefix| texts.to_owned_with_prefix(Some(prefix)));
        let views = prepared_views(&texts, prefixed.as_deref());
        self.check_prepared(&views, model)?;
        let loaded = self.ensure_model().await?;
        let identity = match loaded {
            LoadedModel::Bert { identity, .. }
                if matches!(
                    identity_family(model),
                    IdentityFamily::BertSmall | IdentityFamily::BertFull
                ) =>
            {
                identity.clone().map_err(EmbedError::InferenceFailed)?
            }
            LoadedModel::Qwen { identity, .. }
                if identity_family(model) == IdentityFamily::Qwen =>
            {
                identity.clone().map_err(EmbedError::InferenceFailed)?
            }
            LoadedModel::EmbeddingGemma2 { identity, .. }
                if identity_family(model) == IdentityFamily::Gemma =>
            {
                identity.clone().map_err(EmbedError::InferenceFailed)?
            }
            _ => EmbeddingSpaceIdentity::Unavailable,
        };
        let (embeddings, backends) = if let LoadedModel::Qwen { model, .. } = loaded
            && self.identity_mode == IdentityMode::Full
        {
            let mut embeddings = Vec::with_capacity(views.len());
            let mut backends = Vec::with_capacity(views.len());
            for text in &views {
                let (vector, backend) = model
                    .encode_identified_uncached(text)
                    .map_err(|e| EmbedError::InferenceFailed(e.to_string()))?;
                embeddings.push(vector);
                backends.push(Some(match backend {
                    lattice_model_qwen3_embedding::QwenEmbeddingBackend::Cpu => {
                        EmbeddingBackend::Cpu
                    }
                    lattice_model_qwen3_embedding::QwenEmbeddingBackend::Metal => {
                        EmbeddingBackend::Metal
                    }
                }));
            }
            (embeddings, backends)
        } else {
            let embeddings = loaded
                .encode_batch(&views)
                .map_err(EmbedError::InferenceFailed)?;
            let backend = if matches!(&identity, EmbeddingSpaceIdentity::Known(_))
                && matches!(
                    identity_family(model),
                    IdentityFamily::BertSmall | IdentityFamily::BertFull | IdentityFamily::Gemma
                ) {
                Some(EmbeddingBackend::Cpu)
            } else {
                None
            };
            let backends = vec![backend; embeddings.len()];
            (embeddings, backends)
        };
        Ok(IdentifiedEmbeddings {
            embeddings,
            identity,
            backends,
        })
    }
}

fn descriptor(model: &BertModel, mode: IdentityMode) -> Option<EmbeddingSpaceDescriptor> {
    let report = model.load_report()?;
    if report.weights().mode() != weight_mode(mode) {
        return None;
    }
    let mut artifacts = vec![
        Artifact {
            role: ArtifactRole::Config,
            logical_identifier: "config.json".into(),
            file_size: report.config().size(),
            recipe_digest: report.config().digest(),
        },
        Artifact {
            role: ArtifactRole::Safetensors,
            logical_identifier: "model.safetensors".into(),
            file_size: report.weights().file_size(),
            recipe_digest: report.weights().recipe_digest(),
        },
    ];
    artifacts.extend(report.tokenizer().iter().map(|(name, input)| Artifact {
        role: ArtifactRole::Tokenizer,
        logical_identifier: name.clone(),
        file_size: input.size(),
        recipe_digest: input.digest(),
    }));
    let config = model.config();
    let mut policy = Vec::new();
    let mut add = |key: &str, value| {
        policy.push(PolicyEntry {
            key: key.into(),
            value,
        })
    };
    for (key, value) in [
        ("config.vocab_size", config.vocab_size),
        ("config.hidden_size", config.hidden_size),
        ("config.num_hidden_layers", config.num_hidden_layers),
        ("config.num_attention_heads", config.num_attention_heads),
        ("config.intermediate_size", config.intermediate_size),
        (
            "config.max_position_embeddings",
            config.max_position_embeddings,
        ),
        ("config.type_vocab_size", config.type_vocab_size),
        (
            "input.max_tokens",
            model
                .tokenizer()
                .max_seq_len()
                .min(config.max_position_embeddings),
        ),
        ("output.dimension", model.dimensions()),
    ] {
        add(key, PolicyValue::Unsigned(value as u64));
    }
    add(
        "config.layer_norm_eps",
        PolicyValue::F32(config.layer_norm_eps),
    );
    let pooling = pooling_name(model.pooling())?;
    for (key, value) in [
        ("pooling", pooling),
        ("normalization", "l2"),
        ("tokenizer.format", report.tokenizer_format()),
        ("tokenizer.program", "lattice-tokenizer-v1"),
        ("input.truncation_side", "right"),
        ("input.program", "single-sequence-special-tokens-v1"),
        ("role.generic", ""),
        (
            "role.query",
            EmbeddingModel::BgeSmallEnV15
                .query_instruction()
                .unwrap_or(""),
        ),
        (
            "role.passage",
            EmbeddingModel::BgeSmallEnV15
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
    for (key, value) in report.tokenizer_settings() {
        use lattice_tokenizer::common::TokenizerPolicyValue as V;
        let value = match value {
            V::Boolean(value) => PolicyValue::Boolean(*value),
            V::Unsigned(value) => PolicyValue::Unsigned(*value),
            V::String(value) => PolicyValue::String(value.clone()),
            V::F32(value) => PolicyValue::F32(*value),
        };
        add(&format!("tokenizer.effective.{key}"), value);
    }
    Some(EmbeddingSpaceDescriptor {
        mode,
        output_revision: OUTPUT_REVISION.into(),
        producer_family: "bge-small-native-v1".into(),
        artifacts,
        policy,
    })
}

pub(super) fn identify(
    model: &BertModel,
    mode: IdentityMode,
) -> std::result::Result<EmbeddingSpaceIdentity, String> {
    let Some(descriptor) = descriptor(model, mode) else {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    };
    let canonical = descriptor
        .canonical_bytes()
        .map_err(|error| error.to_string())?;
    Ok(EmbeddingSpaceIdentity::Known(
        KnownEmbeddingSpaceId::from_digest(mode, Sha256::digest(canonical).into()),
    ))
}

pub(super) fn identify_bge_variant(
    model: &BertModel,
    mode: IdentityMode,
    family: EmbeddingModel,
) -> std::result::Result<EmbeddingSpaceIdentity, String> {
    if mode != IdentityMode::Full {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    }
    let Some(descriptor) = bge_variant_descriptor(model, mode, family) else {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    };
    let canonical = descriptor
        .canonical_bytes()
        .map_err(|error| error.to_string())?;
    Ok(EmbeddingSpaceIdentity::Known(
        KnownEmbeddingSpaceId::from_digest(mode, Sha256::digest(canonical).into()),
    ))
}

fn bge_variant_descriptor(
    model: &BertModel,
    mode: IdentityMode,
    family: EmbeddingModel,
) -> Option<EmbeddingSpaceDescriptor> {
    let family = match family {
        EmbeddingModel::BgeBaseEnV15 => "bge-base-native-v1",
        EmbeddingModel::BgeLargeEnV15 => "bge-large-native-v1",
        _ => return None,
    };
    let mut descriptor = descriptor(model, mode)?;
    descriptor.producer_family = family.into();
    Some(descriptor)
}

pub(super) fn identify_e5(
    model: &BertModel,
    mode: IdentityMode,
    family: EmbeddingModel,
) -> std::result::Result<EmbeddingSpaceIdentity, String> {
    if mode != IdentityMode::Full {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    }
    let Some(descriptor) = e5_descriptor(model, mode, family)? else {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    };
    let canonical = descriptor
        .canonical_bytes()
        .map_err(|error| error.to_string())?;
    Ok(EmbeddingSpaceIdentity::Known(
        KnownEmbeddingSpaceId::from_digest(mode, Sha256::digest(canonical).into()),
    ))
}

fn e5_descriptor(
    model: &BertModel,
    mode: IdentityMode,
    family: EmbeddingModel,
) -> std::result::Result<Option<EmbeddingSpaceDescriptor>, String> {
    let domain = match family {
        EmbeddingModel::MultilingualE5Small => "multilingual-e5-small-native-v1",
        EmbeddingModel::MultilingualE5Base => "multilingual-e5-base-native-v1",
        _ => return Ok(None),
    };
    let Some(mut descriptor) = descriptor(model, mode) else {
        return Ok(None);
    };
    descriptor.producer_family = domain.into();
    set_role_policy(&mut descriptor, family)?;
    Ok(Some(descriptor))
}

pub(super) fn identify_minilm(
    model: &BertModel,
    mode: IdentityMode,
    family: EmbeddingModel,
) -> std::result::Result<EmbeddingSpaceIdentity, String> {
    if mode != IdentityMode::Full {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    }
    let Some(descriptor) = minilm_descriptor(model, mode, family)? else {
        return Ok(EmbeddingSpaceIdentity::Unavailable);
    };
    let canonical = descriptor
        .canonical_bytes()
        .map_err(|error| error.to_string())?;
    Ok(EmbeddingSpaceIdentity::Known(
        KnownEmbeddingSpaceId::from_digest(mode, Sha256::digest(canonical).into()),
    ))
}

fn minilm_descriptor(
    model: &BertModel,
    mode: IdentityMode,
    family: EmbeddingModel,
) -> std::result::Result<Option<EmbeddingSpaceDescriptor>, String> {
    let domain = match family {
        EmbeddingModel::AllMiniLmL6V2 => "all-minilm-l6-v2-native-v1",
        EmbeddingModel::ParaphraseMultilingualMiniLmL12V2 => {
            "paraphrase-multilingual-minilm-l12-v2-native-v1"
        }
        _ => return Ok(None),
    };
    let Some(mut descriptor) = descriptor(model, mode) else {
        return Ok(None);
    };
    descriptor.producer_family = domain.into();
    set_role_policy(&mut descriptor, family)?;
    Ok(Some(descriptor))
}

fn set_role_policy(
    descriptor: &mut EmbeddingSpaceDescriptor,
    family: EmbeddingModel,
) -> std::result::Result<(), String> {
    for (key, value) in [
        ("role.query", family.query_instruction().unwrap_or("")),
        ("role.passage", family.document_instruction().unwrap_or("")),
    ] {
        let entry = descriptor
            .policy
            .iter_mut()
            .find(|entry| entry.key == key)
            .ok_or_else(|| format!("embedding identity missing {key} policy"))?;
        entry.value = PolicyValue::String(value.into());
    }
    Ok(())
}

#[cfg(test)]
fn backend_free(descriptor: &EmbeddingSpaceDescriptor) -> bool {
    descriptor
        .policy
        .iter()
        .all(|entry| entry.key != "backend" && !entry.key.starts_with("backend."))
}

#[cfg(test)]
mod gemma_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod qwen_tests;

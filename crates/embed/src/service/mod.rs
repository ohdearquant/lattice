//! Async embedding-service contract and native implementations.
//!
//! The trait defines generic, query, and passage embedding; native builds additionally expose
//! a lazy local-inference service and an LRU caching wrapper. See `docs/service.md` for the
//! lifecycle, prompt handling, validation rules, and cache behavior.

#[cfg(feature = "native")]
mod cached;
#[cfg(feature = "native")]
mod native;

#[cfg(test)]
mod tests;

use crate::error::{EmbedError, Result};
use crate::model::{EmbeddingModel, ModelConfig};
use async_trait::async_trait;

// Re-exports
#[cfg(feature = "native")]
pub use cached::CachedEmbeddingService;
#[cfg(feature = "native")]
pub use native::NativeEmbeddingService;

/// **Stable**: default maximum batch size to prevent OOM.
///
/// This limit prevents accidentally passing huge batches that could exhaust memory.
/// Can be overridden by using chunked calls if larger batches are needed.
pub const DEFAULT_MAX_BATCH_SIZE: usize = 1000;

/// **Stable**: maximum allowed text length in UTF-8 bytes.
///
/// This limit prevents OOM attacks via extremely large input texts.
/// 32KB is sufficient for most embedding use cases while preventing abuse.
///
/// Multibyte text can exceed this limit well before its `chars()` count does;
/// the guard checks `str::len()` (bytes), not character count.
pub const MAX_TEXT_BYTES: usize = 32768;

/// Deprecated alias for [`MAX_TEXT_BYTES`] — the old name implied character
/// count, but the guard has always counted UTF-8 bytes.
#[deprecated(
    since = "0.7.0",
    note = "use MAX_TEXT_BYTES; this limit counts UTF-8 bytes, not chars"
)]
pub const MAX_TEXT_CHARS: usize = MAX_TEXT_BYTES;

/// **Stable**: role of text in asymmetric retrieval.
///
/// Selects query, passage, or generic preparation and cache-key namespace.
/// See [`docs/service.md`](../../docs/service.md#trait-api-details) for retrieval-role semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingRole {
    /// Query / question text — may receive a query-side prompt prefix.
    Query,
    /// Document / passage text — may receive a passage-side prompt prefix.
    Passage,
    /// Generic text with no role-specific prefix (backwards-compatible default).
    Generic,
}

impl EmbeddingRole {
    /// Returns this role's short cache-key tag.
    #[inline]
    pub(crate) const fn cache_tag(self) -> &'static str {
        match self {
            EmbeddingRole::Query => "role:query",
            EmbeddingRole::Passage => "role:passage",
            EmbeddingRole::Generic => "role:generic",
        }
    }

    /// Returns the instruction this role prepends for `model`, if any.
    #[inline]
    pub(crate) const fn instruction(self, model: EmbeddingModel) -> Option<&'static str> {
        match self {
            EmbeddingRole::Query => model.query_instruction(),
            EmbeddingRole::Passage => model.document_instruction(),
            EmbeddingRole::Generic => None,
        }
    }
}

/// Enforces the published request bounds against caller-supplied text.
///
/// This is the contract check, so it runs on what the caller actually passed
/// and before any instruction is prepended. Guards that run downstream of
/// preparation are memory backstops and size themselves with
/// [`EmbeddingModel::max_instruction_bytes`]; they are not this check.
pub(crate) fn validate_texts(texts: &[String]) -> Result<()> {
    validate_texts_bounded(texts, MAX_TEXT_BYTES)
}

/// Shared body of the caller-text contract check and the prepared-text backstop.
///
/// `max_bytes` is [`MAX_TEXT_BYTES`] on caller text, and that value plus the
/// model's longest instruction on text that has already been prepared. The
/// reported `max` is `max_bytes` so an error names the bound that actually
/// rejected the input rather than a constant the caller cannot relate to.
pub(crate) fn validate_texts_bounded(texts: &[String], max_bytes: usize) -> Result<()> {
    if texts.is_empty() {
        return Err(EmbedError::InvalidInput("no texts provided".into()));
    }
    if texts.len() > DEFAULT_MAX_BATCH_SIZE {
        return Err(EmbedError::InvalidInput(format!(
            "batch size {} exceeds maximum {}",
            texts.len(),
            DEFAULT_MAX_BATCH_SIZE
        )));
    }
    for text in texts {
        if text.len() > max_bytes {
            return Err(EmbedError::TextTooLong {
                length: text.len(),
                max: max_bytes,
            });
        }
    }
    Ok(())
}

/// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
///
/// Async interface for producing one embedding per input text.
///
/// See [`docs/service.md`](../../docs/service.md#trait-api-details) for role handling and implementation requirements.
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    /// **Stable**: generate embeddings for multiple texts.
    ///
    /// Returns a vector of embeddings, one for each input text, in the same order.
    /// Applies no role-specific prompt prefix (equivalent to `Generic` role).
    /// Use [`EmbeddingService::embed_query`] / [`EmbeddingService::embed_passage`]
    /// for asymmetric retrieval models.
    async fn embed(&self, texts: &[String], model: EmbeddingModel) -> Result<Vec<Vec<f32>>>;

    /// **Stable**: generate an embedding for a single text.
    ///
    /// This is a convenience method that calls `embed` with a single-element slice.
    async fn embed_one(&self, text: &str, model: EmbeddingModel) -> Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let mut embeddings = self.embed(&texts, model).await?;
        embeddings
            .pop()
            .ok_or_else(|| EmbedError::Internal("no embedding generated".into()))
    }

    /// **Stable**: embed texts under a retrieval role, applying that role's instruction.
    ///
    /// The published length cap applies to the text the caller supplies, so this
    /// validates first and prepends the instruction second. Doing it the other
    /// way round charges the caller for bytes the service itself added, which
    /// rejects text that is within the documented limit.
    ///
    /// Implementors that enforce a text-length cap inside [`EmbeddingService::embed`]
    /// receive prepared text here, which is longer than the caller's by up to
    /// [`EmbeddingModel::max_instruction_bytes`]. Size that guard accordingly, or
    /// override this method to reach the backend without passing through it.
    ///
    /// See [`docs/service.md`](../../docs/service.md#trait-api-details) for role and cache behavior.
    async fn embed_with_role(
        &self,
        texts: &[String],
        model: EmbeddingModel,
        role: EmbeddingRole,
    ) -> Result<Vec<Vec<f32>>> {
        validate_texts(texts)?;
        let prepared = apply_prefix(texts, role.instruction(model));
        self.embed(&prepared, model).await
    }

    /// **Stable**: embed query texts after applying the model's query instruction.
    ///
    /// See [`docs/service.md`](../../docs/service.md#trait-api-details) for role and cache behavior.
    async fn embed_query(&self, texts: &[String], model: EmbeddingModel) -> Result<Vec<Vec<f32>>> {
        self.embed_with_role(texts, model, EmbeddingRole::Query)
            .await
    }

    /// **Stable**: embed passages after applying the model's document instruction.
    ///
    /// See [`docs/service.md`](../../docs/service.md#trait-api-details) for role and cache behavior.
    async fn embed_passage(
        &self,
        texts: &[String],
        model: EmbeddingModel,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_with_role(texts, model, EmbeddingRole::Passage)
            .await
    }

    /// **Unstable**: returns the effective configuration used for this model's cache keys.
    ///
    /// See [`docs/service.md`](../../docs/service.md#trait-api-details) for output-dimension behavior.
    fn model_config(&self, model: EmbeddingModel) -> ModelConfig {
        ModelConfig::new(model)
    }

    /// **Stable**: check if the service supports a given model.
    fn supports_model(&self, model: EmbeddingModel) -> bool;

    /// **Stable**: get the name/identifier of this service.
    fn name(&self) -> &'static str;
}

/// Prepends an optional prompt prefix to each text, cloning unchanged inputs when absent.
pub(crate) fn apply_prefix(texts: &[String], prefix: Option<&str>) -> Vec<String> {
    match prefix {
        None => texts.to_vec(),
        Some(p) => texts.iter().map(|t| format!("{p}{t}")).collect(),
    }
}

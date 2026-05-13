//! Native embedding service using lattice-inference (pure Rust, no C++ FFI).

use super::{DEFAULT_MAX_BATCH_SIZE, EmbeddingService, MAX_TEXT_CHARS};
use crate::error::{EmbedError, Result};
use crate::model::{EmbeddingModel, ModelConfig};
use async_trait::async_trait;
use lattice_inference::{BertModel, QwenModel};
use std::sync::{Arc, OnceLock};
use tracing::info;

/// Loaded model — either BERT-family (encoder) or Qwen (decoder).
enum LoadedModel {
    Bert(Arc<BertModel>),
    Qwen(Arc<QwenModel>),
}

// SA-161/162: Both BertModel and QwenModel derive Send + Sync automatically:
// BertModel has no interior mutability; QwenModel wraps mutable state in Mutex
// which is itself Send + Sync. The manual unsafe impls are therefore redundant
// and have been removed to prevent a stale "read-only" comment from misleading
// future readers.

impl LoadedModel {
    fn encode_batch(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<f32>>, String> {
        match self {
            LoadedModel::Bert(m) => m.encode_batch(texts).map_err(|e| e.to_string()),
            // For Qwen, use per-item encode() which checks the cache.
            LoadedModel::Qwen(m) => {
                let mut results = Vec::with_capacity(texts.len());
                for text in texts {
                    results.push(m.encode(text).map_err(|e| e.to_string())?);
                }
                Ok(results)
            }
        }
    }

    fn cache_size(&self) -> usize {
        match self {
            LoadedModel::Qwen(m) => m.cache_size(),
            _ => 0,
        }
    }
}

/// **Unstable**: model-loading API still evolving; signature may change as lattice-inference matures.
///
/// Pure Rust embedding service backed by lattice-inference.
///
/// Uses SIMD-accelerated matrix multiplication and safetensors weight loading.
/// No ONNX Runtime, no C++ FFI, no fastembed dependency.
///
/// Supports both encoder (BERT/BGE) and decoder (Qwen3) architectures.
///
/// # Cancellation Safety
///
/// Model loading uses `std::sync::OnceLock` + `spawn_blocking` instead of
/// `tokio::sync::OnceCell`. This is critical because `tokio::sync::OnceCell::
/// get_or_try_init` resets when the calling future is dropped (e.g., client
/// disconnect during MCP timeout). With `OnceLock`, the blocking task runs to
/// completion and stores the result regardless of async cancellation, so the
/// model only loads once per process lifetime.
pub struct NativeEmbeddingService {
    model: Arc<OnceLock<std::result::Result<LoadedModel, String>>>,
    model_config: ModelConfig,
}

impl Default for NativeEmbeddingService {
    fn default() -> Self {
        Self::new()
    }
}

const LATTICE_EMBED_DIM: &str = "LATTICE_EMBED_DIM";

fn model_config_from_env(model: EmbeddingModel) -> Result<ModelConfig> {
    let output_dim = match std::env::var(LATTICE_EMBED_DIM) {
        Ok(raw) if raw.trim().is_empty() => None,
        Ok(raw) => {
            let dim = raw.trim().parse::<usize>().map_err(|e| {
                EmbedError::InvalidInput(format!("invalid {LATTICE_EMBED_DIM}={raw:?}: {e}"))
            })?;
            Some(dim)
        }
        Err(std::env::VarError::NotPresent) => None,
        Err(e) => {
            return Err(EmbedError::InvalidInput(format!(
                "invalid {LATTICE_EMBED_DIM}: {e}"
            )));
        }
    };
    ModelConfig::try_new(model, output_dim)
}

impl NativeEmbeddingService {
    /// **Unstable**: constructor signature may change; use `EmbeddingService` trait for stable API.
    pub fn new() -> Self {
        Self {
            model: Arc::new(OnceLock::new()),
            model_config: ModelConfig::new(EmbeddingModel::default()),
        }
    }

    /// **Unstable**: constructor signature may change; use `EmbeddingService` trait for stable API.
    pub fn with_model(model_type: EmbeddingModel) -> Self {
        Self {
            model: Arc::new(OnceLock::new()),
            model_config: ModelConfig::new(model_type),
        }
    }

    /// **Unstable**: create with explicit model config (model + optional MRL truncation dim).
    pub fn with_model_config(model_config: ModelConfig) -> Result<Self> {
        model_config.validate()?;
        Ok(Self {
            model: Arc::new(OnceLock::new()),
            model_config,
        })
    }

    /// **Unstable**: create with model config read from `LATTICE_EMBED_DIM` env var.
    pub fn with_model_from_env(model_type: EmbeddingModel) -> Result<Self> {
        let config = model_config_from_env(model_type)?;
        Ok(Self {
            model: Arc::new(OnceLock::new()),
            model_config: config,
        })
    }

    /// **Unstable**: persistence API may be moved to a separate manager type.
    pub fn save_cache(&self) -> Result<usize> {
        let Some(Ok(model)) = self.model.get() else {
            return Ok(0);
        };
        match model {
            LoadedModel::Qwen(m) => {
                let model_name = self.model_config.model.to_string();
                let path = embedding_cache_path(&model_name, m.dimensions());
                m.cache_save(&path)
                    .map_err(|e| EmbedError::InferenceFailed(e.to_string()))
            }
            _ => Ok(0),
        }
    }

    /// **Unstable**: internal diagnostic; may be removed or moved to metrics.
    pub fn cache_size(&self) -> usize {
        self.model
            .get()
            .and_then(|r| r.as_ref().ok())
            .map(LoadedModel::cache_size)
            .unwrap_or(0)
    }

    /// Ensure the model is loaded (cancellation-safe).
    ///
    /// Uses `std::sync::OnceLock` so the model loading runs to completion
    /// inside `spawn_blocking` even if the calling async future is dropped
    /// (e.g., client disconnect during MCP timeout). The model loads exactly
    /// once per process lifetime.
    async fn ensure_model(&self) -> Result<&LoadedModel> {
        // Fast path: already loaded.
        if let Some(result) = self.model.get() {
            return result
                .as_ref()
                .map_err(|e| EmbedError::ModelInitialization(e.clone()));
        }

        // Slow path: load model on blocking thread.
        // Clone the Arc so spawn_blocking can store the result directly
        // in the OnceLock, surviving async cancellation.
        let model_lock = self.model.clone();
        let model_config = self.model_config;

        tokio::task::spawn_blocking(move || {
            // OnceLock::get_or_init blocks until init completes.
            // If another thread is already loading, this waits for it.
            // This is fine because we're on the blocking thread pool.
            model_lock.get_or_init(|| load_model_sync(model_config));
        })
        .await
        .map_err(|e| EmbedError::ModelInitialization(e.to_string()))?;

        self.model
            .get()
            .expect("set by spawn_blocking")
            .as_ref()
            .map_err(|e| EmbedError::ModelInitialization(e.clone()))
    }
}

/// Synchronous model loading (runs on blocking thread pool).
fn load_model_sync(model_config: ModelConfig) -> std::result::Result<LoadedModel, String> {
    match model_config.model {
        EmbeddingModel::BgeSmallEnV15
        | EmbeddingModel::BgeBaseEnV15
        | EmbeddingModel::BgeLargeEnV15
        | EmbeddingModel::MultilingualE5Small
        | EmbeddingModel::MultilingualE5Base
        | EmbeddingModel::AllMiniLmL6V2
        | EmbeddingModel::ParaphraseMultilingualMiniLmL12V2 => {
            let model_name = match model_config.model {
                EmbeddingModel::BgeSmallEnV15 => "bge-small-en-v1.5",
                EmbeddingModel::BgeBaseEnV15 => "bge-base-en-v1.5",
                EmbeddingModel::BgeLargeEnV15 => "bge-large-en-v1.5",
                EmbeddingModel::MultilingualE5Small => "multilingual-e5-small",
                EmbeddingModel::MultilingualE5Base => "multilingual-e5-base",
                EmbeddingModel::AllMiniLmL6V2 => "all-minilm-l6-v2",
                EmbeddingModel::ParaphraseMultilingualMiniLmL12V2 => {
                    "paraphrase-multilingual-minilm-l12-v2"
                }
                _ => unreachable!(),
            };
            info!(model = model_name, "loading native BERT embedding model");
            let bert = BertModel::from_pretrained(model_name).map_err(|e| e.to_string())?;
            Ok(LoadedModel::Bert(Arc::new(bert)))
        }
        EmbeddingModel::Qwen3Embedding0_6B | EmbeddingModel::Qwen3Embedding4B => {
            load_qwen_model(model_config)
        }
        other => Err(format!("unsupported model: {other:?}")),
    }
}

fn load_qwen_model(model_config: ModelConfig) -> std::result::Result<LoadedModel, String> {
    model_config.validate().map_err(|e| e.to_string())?;
    let model_type = model_config.model;
    let model_name = model_type.to_string();
    info!(
        model = %model_name,
        output_dim = ?model_config.output_dim,
        "loading Qwen embedding model"
    );
    let model_dir = qwen_model_dir(model_type).map_err(|e| e.to_string())?;
    let mut model = QwenModel::from_directory(&model_dir).map_err(|e| e.to_string())?;
    model.set_output_dim(model_config.output_dim);
    let cache_path = embedding_cache_path(&model_name, model.dimensions());
    match model.cache_load(&cache_path) {
        Ok(n) if n > 0 => {
            info!(entries = n, path = %cache_path.display(), "loaded embedding cache")
        }
        _ => {}
    }
    Ok(LoadedModel::Qwen(Arc::new(model)))
}

/// Path for persistent embedding cache: ~/.lattice/cache/embed_{model}_{dim}d.bin
fn embedding_cache_path(model: &str, dim: usize) -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    std::path::PathBuf::from(home)
        .join(".lattice")
        .join("cache")
        .join(format!("embed_{model}_{dim}d.bin"))
}

/// Locate Qwen3-Embedding model directory for the given model variant.
fn qwen_model_dir(model_type: EmbeddingModel) -> Result<std::path::PathBuf> {
    // Check env override first — applies to whichever Qwen model is loaded.
    if let Ok(dir) = std::env::var("LATTICE_QWEN_MODEL_DIR") {
        return Ok(std::path::PathBuf::from(dir));
    }

    let slug = match model_type {
        EmbeddingModel::Qwen3Embedding0_6B => "qwen3-embedding-0.6b",
        EmbeddingModel::Qwen3Embedding4B => "qwen3-embedding-4b",
        other => {
            return Err(EmbedError::ModelInitialization(format!(
                "not a Qwen model: {other}"
            )));
        }
    };

    let home = std::env::var("HOME")
        .map_err(|_| EmbedError::ModelInitialization("HOME not set".into()))?;
    let dir = std::path::PathBuf::from(home)
        .join(".lattice")
        .join("models")
        .join(slug);

    if dir.join("model.safetensors").exists() || dir.join("model.safetensors.index.json").exists() {
        Ok(dir)
    } else {
        Err(EmbedError::ModelInitialization(format!(
            "Qwen3 model not found at {}. Download from {}",
            dir.display(),
            model_type.model_id()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_path_contains_dim_in_filename() {
        let path = embedding_cache_path("qwen3-embedding-4b", 1024);
        let filename = path.file_name().unwrap().to_str().unwrap();
        assert_eq!(filename, "embed_qwen3-embedding-4b_1024d.bin");
    }

    #[test]
    fn test_cache_path_different_dims_produce_different_paths() {
        let path_1024 = embedding_cache_path("qwen3-embedding-4b", 1024);
        let path_2560 = embedding_cache_path("qwen3-embedding-4b", 2560);
        assert_ne!(path_1024, path_2560);
        assert!(path_1024.to_string_lossy().contains("1024d"));
        assert!(path_2560.to_string_lossy().contains("2560d"));
    }

    #[test]
    fn test_cache_path_model_slug_differentiates_variants() {
        let path_4b = embedding_cache_path("qwen3-embedding-4b", 2560);
        let path_06b = embedding_cache_path("qwen3-embedding-0.6b", 1024);
        assert_ne!(path_4b, path_06b);
        assert!(path_4b.to_string_lossy().contains("qwen3-embedding-4b"));
        assert!(path_06b.to_string_lossy().contains("qwen3-embedding-0.6b"));
    }

    #[test]
    fn test_cache_path_same_model_same_dim_same_path() {
        let p1 = embedding_cache_path("qwen3-embedding-4b", 1024);
        let p2 = embedding_cache_path("qwen3-embedding-4b", 1024);
        assert_eq!(p1, p2);
    }
}

#[async_trait]
impl EmbeddingService for NativeEmbeddingService {
    async fn embed(&self, texts: &[String], model: EmbeddingModel) -> Result<Vec<Vec<f32>>> {
        if model != self.model_config.model {
            return Err(EmbedError::InvalidInput(format!(
                "requested model {:?} but this service is loaded with {:?}",
                model, self.model_config.model
            )));
        }
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
            if text.len() > MAX_TEXT_CHARS {
                return Err(EmbedError::TextTooLong {
                    length: text.len(),
                    max: MAX_TEXT_CHARS,
                });
            }
        }

        let loaded = self.ensure_model().await?;
        let text_refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
        loaded
            .encode_batch(&text_refs)
            .map_err(EmbedError::InferenceFailed)
    }

    fn model_config(&self, model: EmbeddingModel) -> ModelConfig {
        if model == self.model_config.model {
            self.model_config
        } else {
            ModelConfig::new(model)
        }
    }

    fn supports_model(&self, model: EmbeddingModel) -> bool {
        model == self.model_config.model
    }

    fn name(&self) -> &'static str {
        "native-bert"
    }
}

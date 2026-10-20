use super::vision::VisionRuntime;
use crate::forward::metal_qwen35::MetalQwen35State;
use crate::model::qwen35::Qwen35Model;
use crate::model::qwen35_config::Qwen35Config;
use crate::model::serving_runtime::{QwenMetalRuntime, RuntimeFactory, ServingRuntime};
use crate::model_format::{self, ModelFormat};
use crate::serve::lora::{AdapterIndex, ResidencyLimits};
use crate::serve::metal_worker::serving_tokenizer;
use crate::serve::prepare::PreparationHandle;
use crate::serving_provider::{StandalonePostReady, StandaloneQwenLoader};
use crate::serving_runtime_contract::WorkerMetadata;
use crate::tokenizer::bpe::BpeTokenizer;
use std::sync::{Arc, RwLock};

/// Fallback when model configuration has no usable context length.
#[doc(hidden)]
pub const FALLBACK_MODEL_MAX_CONTEXT: usize = 4096;

/// Derive the context window from model configuration.
#[doc(hidden)]
pub fn model_context_from_config(cfg: Option<&Qwen35Config>) -> usize {
    cfg.map(|cfg| cfg.max_position_embeddings)
        .filter(|&length| length > 0)
        .unwrap_or(FALLBACK_MODEL_MAX_CONTEXT)
}

struct LoadedModel {
    metal: MetalQwen35State,
    tokenizer: BpeTokenizer,
    format: String,
    model_max_context: usize,
}

fn load_model(
    model_dir: &std::path::Path,
    tokenizer_path: &std::path::Path,
    format: ModelFormat,
) -> Result<LoadedModel, String> {
    let tokenizer = BpeTokenizer::from_tokenizer_json(tokenizer_path).map_err(|error| {
        format!(
            "tokenizer load failed ({}): {error}",
            tokenizer_path.display()
        )
    })?;

    match format {
        ModelFormat::Q4 => {
            let config = Qwen35Config::from_model_dir(model_dir)
                .map_err(|error| format!("config.json load failed: {error}"))?;
            let requested_context = model_context_from_config(Some(&config));
            let metal = MetalQwen35State::from_q4_dir(
                model_dir,
                tokenizer_path,
                &config,
                requested_context,
            )
            .map_err(|error| format!("Q4 model load failed: {error}"))?;
            let model_max_context = metal.max_context();
            Ok(LoadedModel {
                metal,
                tokenizer,
                format: "q4".to_string(),
                model_max_context,
            })
        }
        ModelFormat::Safetensors => {
            let model = Qwen35Model::from_safetensors(model_dir)
                .map_err(|error| format!("safetensors load failed: {error}"))?;
            let config = model.config().clone();
            let requested_context = model_context_from_config(Some(&config));
            let metal = MetalQwen35State::new(model.weights(), &config, requested_context)
                .map_err(|error| format!("Metal init failed: {error}"))?;
            let model_max_context = metal.max_context();
            Ok(LoadedModel {
                metal,
                tokenizer,
                format: "bf16".to_string(),
                model_max_context,
            })
        }
        ModelFormat::Unknown => Err(model_format::unrecognized_format_message(model_dir)),
        _ => Err(model_format::unloadable_format_message(model_dir)),
    }
}

pub(crate) fn standalone_loader() -> StandaloneQwenLoader {
    Box::new(|model_dir, tokenizer_path, format| {
        Box::new(move || {
            let LoadedModel {
                metal,
                tokenizer,
                format,
                model_max_context,
            } = load_model(&model_dir, &tokenizer_path, format)?;
            Ok((
                metal,
                tokenizer,
                WorkerMetadata {
                    format,
                    model_max_context,
                    context_window_policy:
                        crate::serving_runtime_contract::ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                },
            ))
        })
    })
}

pub(crate) fn vocabulary_after_ready(
    tokenizer_path: std::path::PathBuf,
    model_dir: std::path::PathBuf,
) -> StandalonePostReady {
    StandalonePostReady::new(move || {
        let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path).map_err(|error| {
            format!(
                "tokenizer load failed ({}): {error}",
                tokenizer_path.display()
            )
        })?;
        let vocab_size = Qwen35Config::from_model_dir(&model_dir)
            .map(|config| config.vocab_size)
            .map_err(|error| format!("config.json load failed: {error}"))?;
        Ok(Arc::new(tokenizer.vocab_bytes(vocab_size).map_err(
            |error| format!("tokenizer vocab_bytes failed: {error}"),
        )?))
    })
}

pub(crate) type QwenLoader =
    Box<dyn FnOnce() -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String> + Send>;

pub(crate) fn factory(loader: QwenLoader, vision: VisionRuntime) -> Box<dyn RuntimeFactory> {
    Box::new(QwenFactory { loader, vision })
}

struct QwenFactory {
    loader: QwenLoader,
    vision: VisionRuntime,
}

impl RuntimeFactory for QwenFactory {
    fn build(
        self: Box<Self>,
        index: Arc<RwLock<AdapterIndex>>,
        limits: ResidencyLimits,
    ) -> Result<(Box<dyn ServingRuntime>, WorkerMetadata, PreparationHandle), String> {
        let Self { loader, vision } = *self;
        let (state, tokenizer, metadata) = loader()?;
        let tokenizer = serving_tokenizer(tokenizer, metadata.model_max_context);
        let tokenizer = Arc::new(tokenizer);
        let preparation =
            PreparationHandle::qwen(Arc::clone(&tokenizer), metadata.model_max_context);
        let runtime =
            QwenMetalRuntime::new(state, tokenizer, vision, metadata.clone(), index, limits);
        Ok((Box::new(runtime), metadata, preparation))
    }
}

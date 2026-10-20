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

/// Served context window when no `--max-context` is given.
///
/// The Metal worker preallocates its KV cache for the whole served window at
/// load, so serving `max_position_embeddings` (262144 on Qwen3.5) would
/// reserve several GiB before the first request. A caller that needs a longer
/// window passes `--max-context`.
#[doc(hidden)]
pub const DEFAULT_SERVE_MAX_CONTEXT: usize = 32768;

/// Where the served context window came from.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeContextSource {
    /// `--max-context`, at or below the model's window.
    Flag,
    /// [`DEFAULT_SERVE_MAX_CONTEXT`], below the model's window.
    DefaultCap,
    /// The model's own window: it is no larger than the requested or default
    /// cap.
    Model,
}

impl ServeContextSource {
    /// Startup-line wording for this source.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Flag => "--max-context",
            Self::DefaultCap => "default cap, raise with --max-context",
            Self::Model => "model max_position_embeddings",
        }
    }
}

/// Parse a `--max-context` value. A malformed value or zero is refused rather
/// than replaced by the default.
#[doc(hidden)]
pub fn parse_max_context(raw: Option<&str>) -> Result<Option<usize>, String> {
    match raw {
        None => Ok(None),
        Some(raw) => match raw.parse::<usize>() {
            Ok(value) if value > 0 => Ok(Some(value)),
            _ => Err(format!(
                "--max-context: invalid value {raw:?} (expected a positive integer)"
            )),
        },
    }
}

/// Resolve the served context window from the model's window and an optional
/// `--max-context` request.
#[doc(hidden)]
pub fn resolve_serve_context(
    model_context: usize,
    requested: Option<usize>,
) -> (usize, ServeContextSource) {
    match requested {
        Some(requested) if requested <= model_context => (requested, ServeContextSource::Flag),
        Some(_) => (model_context, ServeContextSource::Model),
        None if model_context > DEFAULT_SERVE_MAX_CONTEXT => {
            (DEFAULT_SERVE_MAX_CONTEXT, ServeContextSource::DefaultCap)
        }
        None => (model_context, ServeContextSource::Model),
    }
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
    requested_context: usize,
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

pub(crate) fn standalone_loader(requested_context: usize) -> StandaloneQwenLoader {
    Box::new(|model_dir, tokenizer_path, format| {
        Box::new(move || {
            let LoadedModel {
                metal,
                tokenizer,
                format,
                model_max_context,
            } = load_model(&model_dir, &tokenizer_path, format, requested_context)?;
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

/// The worker tokenizer and request preparation, both sized from the one
/// served window in `metadata`. The worker's prompt admission reads the same
/// `metadata`.
fn serving_parts(
    tokenizer: BpeTokenizer,
    metadata: &WorkerMetadata,
) -> (Arc<BpeTokenizer>, PreparationHandle) {
    let tokenizer = Arc::new(serving_tokenizer(tokenizer, metadata.model_max_context));
    let preparation = PreparationHandle::qwen(Arc::clone(&tokenizer), metadata.model_max_context);
    (tokenizer, preparation)
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
        let (tokenizer, preparation) = serving_parts(tokenizer, &metadata);
        let runtime =
            QwenMetalRuntime::new(state, tokenizer, vision, metadata.clone(), index, limits);
        Ok((Box::new(runtime), metadata, preparation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::contract::{ChatRequest, GenerationDefaults};
    use crate::serving_runtime_contract::ContextWindowPolicy;
    use crate::serving_worker::check_prompt_fits_window;
    use std::collections::HashMap;

    #[test]
    fn served_window_sizes_the_clamp_the_tokenizer_and_the_refusal_alike() {
        let (window, source) = resolve_serve_context(262_144, None);
        assert_eq!(
            (window, source),
            (DEFAULT_SERVE_MAX_CONTEXT, ServeContextSource::DefaultCap)
        );
        let metadata = WorkerMetadata {
            format: "q4".to_string(),
            model_max_context: window,
            context_window_policy: ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        };
        let tokenizer = BpeTokenizer::from_vocab_and_merges(
            HashMap::from([("a".to_string(), 0), ("b".to_string(), 1)]),
            Vec::new(),
        )
        .expect("tiny tokenizer must construct");
        let (tokenizer, preparation) = serving_parts(tokenizer, &metadata);

        // The tokenizer cap only ever rises to the window, so prompts are
        // counted whole before admission.
        assert!(tokenizer.max_seq_len() >= window);

        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "ab"}],
            "max_tokens": 1_000_000
        }))
        .expect("request must parse");
        let (_, config) = preparation
            .prepare_standalone(&req, GenerationDefaults::standard(256), "m", false)
            .expect("request must prepare");
        assert_eq!(config.max_new_tokens, window - 1);

        // The clamped request fits exactly with an empty prompt and is
        // refused one prompt token later, at the same window.
        let policy = metadata.context_window_policy;
        assert!(check_prompt_fits_window(policy, metadata.model_max_context, 0, &config).is_ok());
        assert!(check_prompt_fits_window(policy, metadata.model_max_context, 1, &config).is_err());
    }

    #[test]
    fn max_context_parse_refuses_zero_and_garbage() {
        assert_eq!(parse_max_context(None), Ok(None));
        assert_eq!(parse_max_context(Some("8192")), Ok(Some(8192)));
        for raw in ["0", "-1", "8k", ""] {
            assert!(parse_max_context(Some(raw)).is_err(), "{raw:?} accepted");
        }
    }

    #[test]
    fn serve_context_defaults_to_the_cap_below_a_long_model_window() {
        assert_eq!(
            resolve_serve_context(262_144, None),
            (DEFAULT_SERVE_MAX_CONTEXT, ServeContextSource::DefaultCap)
        );
    }

    #[test]
    fn serve_context_keeps_a_model_window_at_or_below_the_cap() {
        assert_eq!(
            resolve_serve_context(4096, None),
            (4096, ServeContextSource::Model)
        );
        assert_eq!(
            resolve_serve_context(DEFAULT_SERVE_MAX_CONTEXT, None),
            (DEFAULT_SERVE_MAX_CONTEXT, ServeContextSource::Model)
        );
    }

    #[test]
    fn serve_context_honours_the_flag_up_to_the_model_window() {
        assert_eq!(
            resolve_serve_context(262_144, Some(8192)),
            (8192, ServeContextSource::Flag)
        );
        assert_eq!(
            resolve_serve_context(262_144, Some(262_144)),
            (262_144, ServeContextSource::Flag)
        );
        assert_eq!(
            resolve_serve_context(262_144, Some(1_000_000)),
            (262_144, ServeContextSource::Model)
        );
    }
}

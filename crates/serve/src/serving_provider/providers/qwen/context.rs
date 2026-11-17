//! The served context window: the model's own window, an operator's
//! `--max-context`, and the default cap that bounds it when neither is lower.

use lattice_inference::model::qwen35_config::Qwen35Config;

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

use crate::forward::metal_qwen35::ChatMessage;
use crate::generation::GenerateOutput;
use crate::serve::ApiError;
use crate::serve::lora::LoraSelection;

/// Which retained CPU generation operation a prepared prompt uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextGenerationEntry {
    Complete,
    StreamingWithCancel,
}

/// Input accepted by one serving runtime execution.
pub(crate) enum RuntimeInput<'a> {
    PreparedText {
        prompt: &'a str,
        entry: TextGenerationEntry,
    },
    ChatMessages {
        messages: &'a [ChatMessage],
        lora: &'a [LoraSelection],
    },
}

/// Selects the context-window formula enforced before Metal generation.
/// Each serve adapter supplies the policy matching its pre-worker contract.
// The Metal serving factory creates metadata with this policy.
#[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ContextWindowPolicy {
    /// Enforce `prompt_tokens + max_new_tokens <= model_max_context`.
    PromptAndMaxTokens,
    /// Enforce `prompt_tokens + max_new_tokens + reasoning_budget + 1
    /// <= model_max_context`.
    PromptAndDecodeWithDelimiter,
}

/// Everything a successful `MetalWorker::spawn` resolves to describe the
/// loaded model, beyond the client handle itself: the format string, the
/// actual KV context the loader allocated, and the adapter's window policy.
// The Metal serving factory creates this metadata.
#[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
#[derive(Debug, Clone)]
pub struct WorkerMetadata {
    pub format: String,
    pub model_max_context: usize,
    pub context_window_policy: ContextWindowPolicy,
}

/// Failure classification for serving runtime generation. Keeps the
/// `Rejected` vs. `Failed` distinction (#656 vs. #611) at the type level
/// instead of relying on message text or a string-prefix convention.
// Runtime implementations produce these failures for the serving worker.
#[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
#[derive(Debug)]
pub(crate) enum WorkerFailure {
    Rejected(ApiError),
    Failed(String),
    /// Remains distinct from `Failed` so the worker can report a blocked
    /// grammar constraint without inspecting the message text.
    ConstraintBlocked(String),
}

impl From<crate::error::InferenceError> for WorkerFailure {
    /// Classifies a generation-time inference error into the worker's
    /// failure shape. Grammar constraint failures retain their dedicated
    /// result; other variants remain generic failures.
    fn from(err: crate::error::InferenceError) -> Self {
        match err {
            crate::error::InferenceError::GrammarConstraintBlocked(message) => {
                WorkerFailure::ConstraintBlocked(message)
            }
            other => WorkerFailure::Failed(other.to_string()),
        }
    }
}

// Runtime implementations use this output when cancellation is observed.
#[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
pub(crate) fn cancelled_output() -> GenerateOutput {
    GenerateOutput {
        text: String::new(),
        token_ids: Vec::new(),
        prompt_tokens: 0,
        generated_tokens: 0,
        stopped: false,
        stop_reason: Some(crate::StopReason::Interrupt),
        token_logprobs: Vec::new(),
    }
}

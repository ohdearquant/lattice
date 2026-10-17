//! Gemma 4 E2B text on the CPU, confined to one serving worker.

use super::{ServingRuntime, WorkerFailure, cancelled_output};
use crate::generation::{GenerateConfig, GenerateOutput};
use crate::serve::ApiError;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::lora::{AdapterControlError, AdapterControlResult};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::metal_worker::AdapterCommand;
use crate::serve::prepare::lora_unsupported_backend;
use crate::serve::route::ServedRoute;
use crate::serving_cpu::GemmaCpuServing;
use crate::serving_runtime_contract::{RuntimeInput, TextGenerationEntry};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Runs each admitted job on the Gemma CPU session under the shared decoder
/// driver, on the worker thread, one job at a time.
///
/// Gemma 4 text has no adapter, vision or grammar path, so each of those is
/// refused by name before any generation work starts.
// The serving factory creates this CPU runtime for the worker.
#[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
pub(crate) struct GemmaCpuRuntime {
    serving: Arc<GemmaCpuServing>,
    vision: Arc<AtomicBool>,
}

impl GemmaCpuRuntime {
    // The serving factory uses this constructor when it builds the runtime.
    #[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
    pub(crate) fn new(serving: Arc<GemmaCpuServing>) -> Self {
        Self {
            serving,
            vision: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl ServingRuntime for GemmaCpuRuntime {
    fn execute(
        &mut self,
        input: RuntimeInput<'_>,
        cfg: &GenerateConfig,
        http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        let RuntimeInput::ChatMessages { messages, lora } = input else {
            return Err(WorkerFailure::Failed(
                "prepared text is not supported by the worker runtime".to_owned(),
            ));
        };
        if !lora.is_empty() {
            return Err(WorkerFailure::Rejected(lora_unsupported_backend()));
        }
        if cfg.grammar.is_some() {
            return Err(WorkerFailure::Rejected(ApiError::BadRequest {
                message: "grammar-constrained output is not supported for this model".to_string(),
                code: "unsupported_feature",
            }));
        }
        let (prompt, _prompt_len) = self
            .serving
            .render_within_window(messages, cfg)
            .map_err(WorkerFailure::Rejected)?;
        if should_cancel() {
            return Ok(cancelled_output());
        }
        let (output, evidence) = self.serving.generate_streaming(
            &prompt,
            cfg,
            |delta| on_token(delta, 0),
            should_cancel,
        )?;
        eprintln!(
            "{}",
            ServedRoute::GEMMA4_CPU.request_marker(http_stream, evidence)
        );
        Ok(output)
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn control(
        &mut self,
        _command: AdapterCommand,
    ) -> Result<AdapterControlResult, AdapterControlError> {
        Err(AdapterControlError::InvalidAdapter(
            "runtime LoRA adapters are not supported for this model".to_string(),
        ))
    }

    fn vision_supported(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.vision)
    }
}

/// Runs a prepared Gemma prompt on a request's blocking task.
pub(crate) struct GemmaPreparedCpuRuntime<'a> {
    serving: &'a GemmaCpuServing,
    vision: Arc<AtomicBool>,
}

impl<'a> GemmaPreparedCpuRuntime<'a> {
    pub(crate) fn new(serving: &'a GemmaCpuServing) -> Self {
        Self {
            serving,
            vision: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl ServingRuntime for GemmaPreparedCpuRuntime<'_> {
    fn execute(
        &mut self,
        input: RuntimeInput<'_>,
        cfg: &GenerateConfig,
        http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        let RuntimeInput::PreparedText { prompt, entry } = input else {
            return Err(WorkerFailure::Failed(
                "chat messages are not supported by the prepared CPU runtime".to_owned(),
            ));
        };
        let (output, evidence) = match entry {
            TextGenerationEntry::Complete => self
                .serving
                .generate(prompt, cfg)
                .map_err(|error| WorkerFailure::Failed(error.to_string()))?,
            TextGenerationEntry::StreamingWithCancel => self
                .serving
                .generate_streaming(prompt, cfg, |delta| on_token(delta, 0), should_cancel)
                .map_err(|error| WorkerFailure::Failed(error.to_string()))?,
        };
        eprintln!(
            "{}",
            ServedRoute::GEMMA4_CPU.request_marker(http_stream, evidence)
        );
        Ok(output)
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn control(
        &mut self,
        _command: AdapterCommand,
    ) -> Result<AdapterControlResult, AdapterControlError> {
        Err(AdapterControlError::InvalidAdapter(
            "runtime LoRA adapters are not supported for this model".to_string(),
        ))
    }

    fn vision_supported(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.vision)
    }
}

#[cfg(test)]
mod tests;

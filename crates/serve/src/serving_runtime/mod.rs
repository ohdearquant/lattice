//! Worker-local execution for model serving.

mod gemma_cpu;
mod qwen_cpu;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod qwen_metal;

use crate::serving_host::ServedOutput;
use crate::serving_runtime_contract::{RuntimeInput, WorkerMetadata};
pub(crate) use crate::serving_runtime_contract::{WorkerFailure, cancelled_output};
use crate::serving_worker::{ScoreOutcome, ScoreRow};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use gemma_cpu::GemmaCpuRuntime;
pub(crate) use gemma_cpu::GemmaPreparedCpuRuntime;
pub(crate) use qwen_cpu::QwenCpuRuntime;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use qwen_metal::QwenMetalRuntime;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::lora::{AdapterControlError, AdapterControlResult};
use crate::serve::lora::{AdapterIndex, LoraSelection, ResidencyLimits};
use crate::serve::prepare::PreparationHandle;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_worker::AdapterCommand;
use lattice_inference::forward::metal_qwen35::ChatMessage;
use lattice_inference::generation::{GenerateConfig, GenerateOutput};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};

/// The execution state stays on the thread that built it.
pub(crate) trait ServingRuntime {
    /// `stream` is whether the HTTP response is streamed. It does not change
    /// what is generated; a runtime that reports its route per request
    /// (`serve::route`) names it there.
    // The Metal worker invokes this method after its factory builds the runtime.
    #[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
    fn generate(
        &mut self,
        messages: &[ChatMessage],
        cfg: &GenerateConfig,
        lora: &[LoraSelection],
        stream: bool,
        explicitly_thinking: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        self.execute(
            RuntimeInput::ChatMessages {
                messages,
                lora,
                explicitly_thinking,
            },
            cfg,
            stream,
            on_token,
            should_cancel,
        )
    }

    fn execute(
        &mut self,
        input: RuntimeInput<'_>,
        cfg: &GenerateConfig,
        http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure>;

    /// [`Self::execute`] with a second sink for the text of the reply's reasoning field. The CPU
    /// host calls this for every request. A runtime whose replies have no reasoning field keeps
    /// this default, which runs `execute` and reports no reasoning; a runtime that overrides it
    /// passes each reasoning piece to `on_reasoning`, in generated order, from the thread that
    /// passes content to `on_token`, and stops when either sink returns false.
    fn execute_with_reasoning(
        &mut self,
        input: RuntimeInput<'_>,
        cfg: &GenerateConfig,
        http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        _on_reasoning: &mut dyn FnMut(&str) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<ServedOutput, WorkerFailure> {
        self.execute(input, cfg, http_stream, on_token, should_cancel)
            .map(ServedOutput::without_reasoning)
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn control(
        &mut self,
        command: AdapterCommand,
    ) -> Result<AdapterControlResult, AdapterControlError>;

    // The Metal worker reads this flag after its factory builds the runtime.
    #[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
    fn vision_supported(&self) -> Arc<AtomicBool>;

    /// Called once with every row of a scoring request before
    /// [`Self::score_option_rows`] scores them, so a runtime can prepare state
    /// the rows share. The rows may then be scored in several calls. The
    /// default prepares nothing. A runtime may ask `hung_up` while it
    /// prepares; `None` means the caller hung up and the preparation stopped.
    fn begin_option_rows(
        &mut self,
        rows: &[ScoreRow],
        hung_up: &mut dyn FnMut() -> bool,
    ) -> Option<Result<(), String>> {
        let _ = (rows, hung_up);
        Some(Ok(()))
    }

    /// Score option rows: for each row, return the raw logits at the last
    /// position of its ids for its label ids, in order, as if the row were
    /// prefilled from a fresh state. A runtime may ask `hung_up` while it
    /// scores; `None` means the caller hung up and the rows were not all
    /// scored. A runtime without an option scorer refuses every request.
    fn score_option_rows(
        &mut self,
        rows: &[ScoreRow],
        hung_up: &mut dyn FnMut() -> bool,
    ) -> Option<ScoreOutcome> {
        let _ = (rows, hung_up);
        Some(Err("this model does not score option rows".into()))
    }
}

/// Score `rows` on `runtime` the way every serving host does: prepare the
/// state the rows share once, then score one row at a time. `hung_up` is asked
/// before the preparation and before each row, and is handed to the runtime,
/// which may ask it inside either; `None` means the caller hung up.
pub(crate) fn score_rows_until_hang_up(
    runtime: &mut dyn ServingRuntime,
    rows: &[ScoreRow],
    hung_up: &mut dyn FnMut() -> bool,
) -> Option<ScoreOutcome> {
    if hung_up() {
        return None;
    }
    if let Err(error) = runtime.begin_option_rows(rows, hung_up)? {
        return Some(Err(error));
    }
    let mut logits = Vec::with_capacity(rows.len());
    for row in rows {
        if hung_up() {
            return None;
        }
        match runtime.score_option_rows(std::slice::from_ref(row), hung_up)? {
            Ok(mut scored) => logits.append(&mut scored),
            Err(error) => return Some(Err(error)),
        }
    }
    Some(Ok(logits))
}

/// A one-shot builder moved into the worker before any Metal state exists.
// The Metal worker calls this factory to create its runtime.
#[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
pub(crate) trait RuntimeFactory: Send + 'static {
    fn build(
        self: Box<Self>,
        index: Arc<RwLock<AdapterIndex>>,
        limits: ResidencyLimits,
    ) -> Result<(Box<dyn ServingRuntime>, WorkerMetadata, PreparationHandle), String>;
}

trait AmbiguousIfSend<A> {
    fn some_item() {}
}
impl<T: ?Sized> AmbiguousIfSend<()> for T {}
#[allow(dead_code)]
struct IsSend;
impl<T: ?Sized + Send> AmbiguousIfSend<IsSend> for T {}
const _: fn() = || {
    let _ = <dyn ServingRuntime as AmbiguousIfSend<_>>::some_item;
};

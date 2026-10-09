//! Worker-local execution for model serving.

mod gemma_cpu;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod qwen_metal;

use crate::serving_runtime_contract::WorkerMetadata;
pub(crate) use crate::serving_runtime_contract::{WorkerFailure, cancelled_output};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use gemma_cpu::GemmaCpuRuntime;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use qwen_metal::QwenMetalRuntime;

use crate::forward::metal_qwen35::ChatMessage;
use crate::generation::{GenerateConfig, GenerateOutput};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::lora::{AdapterControlError, AdapterControlResult};
use crate::serve::lora::{AdapterIndex, LoraSelection, ResidencyLimits};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::metal_worker::AdapterCommand;
use crate::serve::prepare::PreparationHandle;
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
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure>;

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn control(
        &mut self,
        command: AdapterCommand,
    ) -> Result<AdapterControlResult, AdapterControlError>;

    // The Metal worker reads this flag after its factory builds the runtime.
    #[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
    fn vision_supported(&self) -> Arc<AtomicBool>;
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

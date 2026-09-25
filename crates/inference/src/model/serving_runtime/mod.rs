//! Worker-local execution for model serving.

mod qwen_metal;

pub(crate) use qwen_metal::QwenMetalRuntime;

use crate::forward::metal_qwen35::ChatMessage;
use crate::generation::{GenerateConfig, GenerateOutput};
use crate::serve::lora::{
    AdapterControlError, AdapterControlResult, AdapterIndex, LoraSelection, ResidencyLimits,
};
use crate::serve::metal_worker::{AdapterCommand, WorkerFailure, WorkerMetadata};
use crate::serve::prepare::PreparationHandle;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};

/// The execution state stays on the thread that built it.
pub(crate) trait ServingRuntime {
    fn generate(
        &mut self,
        messages: &[ChatMessage],
        cfg: &GenerateConfig,
        lora: &[LoraSelection],
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure>;

    fn control(
        &mut self,
        command: AdapterCommand,
    ) -> Result<AdapterControlResult, AdapterControlError>;

    fn vision_supported(&self) -> Arc<AtomicBool>;
}

/// A one-shot builder moved into the worker before any Metal state exists.
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

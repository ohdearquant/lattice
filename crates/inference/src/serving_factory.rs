//! Select the serving family before building worker-local execution state.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::forward::metal_qwen35::MetalQwen35State;
use crate::model::serving_runtime::{RuntimeFactory, ServingRuntime};
use crate::serve::lora::{AdapterIndex, ResidencyLimits};
use crate::serve::prepare::PreparationHandle;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_provider::providers;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_provider::providers::qwen::vision::VisionRuntime;
use crate::serving_runtime_contract::WorkerMetadata;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::tokenizer::bpe::BpeTokenizer;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// A one-shot, thread-safe entry into worker-local model construction.
#[doc(hidden)]
pub struct ServingFactory {
    inner: Box<dyn RuntimeFactory>,
}

impl ServingFactory {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub(crate) fn new(inner: Box<dyn RuntimeFactory>) -> Self {
        Self { inner }
    }

    /// Bind the existing Qwen loader and lazy vision state.
    /// Loading still occurs on the worker thread.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub fn qwen_metal(
        loader: impl FnOnce() -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String>
        + Send
        + 'static,
        vision: VisionRuntime,
    ) -> Self {
        Self::new(providers::qwen::worker::factory(Box::new(loader), vision))
    }

    /// Bind a Gemma 4 E2B text checkpoint directory. The checkpoint is loaded
    /// on the worker thread, and its jobs run on the CPU under the shared
    /// decoder driver.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub fn gemma_cpu(model_dir: PathBuf) -> Self {
        Self::new(providers::gemma::worker::factory(model_dir))
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub(crate) fn legacy_qwen(
        loader: impl FnOnce() -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String>
        + Send
        + 'static,
    ) -> Self {
        Self::new(providers::qwen::worker::factory(
            Box::new(loader),
            VisionRuntime::unsupported(),
        ))
    }

    pub(crate) fn build(
        self,
        index: Arc<RwLock<AdapterIndex>>,
        limits: ResidencyLimits,
    ) -> Result<(Box<dyn ServingRuntime>, WorkerMetadata, PreparationHandle), String> {
        fn require_send_sync<T: Send + Sync>(_: &T) {}
        let (runtime, metadata, preparation) = self.inner.build(index, limits)?;
        require_send_sync(&preparation);
        Ok((runtime, metadata, preparation))
    }
}

#[cfg(all(test, target_os = "macos", feature = "metal-gpu"))]
mod tests;

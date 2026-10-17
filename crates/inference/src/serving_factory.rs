//! Select the serving family before building worker-local execution state.

use crate::forward::metal_qwen35::MetalQwen35State;
use crate::model::serving_runtime::{
    GemmaCpuRuntime, QwenMetalRuntime, RuntimeFactory, ServingRuntime,
};
use crate::serve::lora::{AdapterIndex, ResidencyLimits};
use crate::serve::metal_worker::{
    ContextWindowPolicy, VisionRuntime, WorkerMetadata, serving_tokenizer,
};
use crate::serve::prepare::PreparationHandle;
use crate::serving_cpu::GemmaCpuServing;
use crate::tokenizer::bpe::BpeTokenizer;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

type QwenLoader =
    Box<dyn FnOnce() -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String> + Send>;

/// A one-shot, thread-safe entry into worker-local model construction.
#[doc(hidden)]
pub struct ServingFactory {
    inner: Box<dyn RuntimeFactory>,
}

impl ServingFactory {
    /// Bind the existing Qwen loader and lazy vision state.
    /// Loading still occurs on the worker thread.
    pub fn qwen_metal(
        loader: impl FnOnce() -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String>
        + Send
        + 'static,
        vision: VisionRuntime,
    ) -> Self {
        Self {
            inner: Box::new(QwenFactory {
                loader: Box::new(loader),
                vision,
            }),
        }
    }

    /// Bind a Gemma 4 E2B text checkpoint directory. The checkpoint is loaded
    /// on the worker thread, and its jobs run on the CPU under the shared
    /// decoder driver.
    pub fn gemma_cpu(model_dir: PathBuf) -> Self {
        Self {
            inner: Box::new(GemmaFactory { model_dir }),
        }
    }

    pub(crate) fn legacy_qwen(
        loader: impl FnOnce() -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String>
        + Send
        + 'static,
    ) -> Self {
        Self::qwen_metal(loader, VisionRuntime::unsupported())
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

struct GemmaFactory {
    model_dir: PathBuf,
}

impl RuntimeFactory for GemmaFactory {
    fn build(
        self: Box<Self>,
        _index: Arc<RwLock<AdapterIndex>>,
        _limits: ResidencyLimits,
    ) -> Result<(Box<dyn ServingRuntime>, WorkerMetadata, PreparationHandle), String> {
        let serving = GemmaCpuServing::load(&self.model_dir)
            .map_err(|e| format!("Gemma 4 model load failed: {e}"))?;
        Ok(gemma_parts(serving))
    }
}

fn gemma_parts(
    serving: GemmaCpuServing,
) -> (Box<dyn ServingRuntime>, WorkerMetadata, PreparationHandle) {
    let metadata = WorkerMetadata {
        format: "safetensors".to_string(),
        model_max_context: serving.max_context(),
        context_window_policy: ContextWindowPolicy::PromptAndDecodeWithDelimiter,
    };
    let serving = Arc::new(serving);
    let preparation = PreparationHandle::gemma(Arc::clone(&serving));
    (
        Box::new(GemmaCpuRuntime::new(serving)),
        metadata,
        preparation,
    )
}

#[cfg(test)]
mod tests;

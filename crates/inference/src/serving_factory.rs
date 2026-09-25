//! Select the serving family before building worker-local execution state.

use crate::forward::metal_qwen35::MetalQwen35State;
use crate::model::serving_runtime::{QwenMetalRuntime, RuntimeFactory, ServingRuntime};
use crate::serve::lora::{AdapterIndex, ResidencyLimits};
use crate::serve::metal_worker::{VisionRuntime, WorkerMetadata, serving_tokenizer};
use crate::serve::prepare::PreparationHandle;
use crate::tokenizer::bpe::BpeTokenizer;
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

#[cfg(test)]
mod tests {
    use super::ServingFactory;
    use crate::serve::lora::{AdapterIndex, ResidencyLimits};
    use crate::serve::metal_worker::VisionRuntime;
    use std::sync::{Arc, RwLock};

    #[test]
    fn qwen_loader_error_is_returned_unchanged() {
        let factory = ServingFactory::qwen_metal(
            || Err("legacy load error".to_string()),
            VisionRuntime::unsupported(),
        );
        let error = factory
            .build(
                Arc::new(RwLock::new(AdapterIndex::default())),
                ResidencyLimits::default(),
            )
            .err()
            .unwrap();
        assert_eq!(error, "legacy load error");
    }
}

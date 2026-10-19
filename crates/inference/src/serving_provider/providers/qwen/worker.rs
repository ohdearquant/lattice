use super::vision::VisionRuntime;
use crate::forward::metal_qwen35::MetalQwen35State;
use crate::model::serving_runtime::{QwenMetalRuntime, RuntimeFactory, ServingRuntime};
use crate::serve::lora::{AdapterIndex, ResidencyLimits};
use crate::serve::metal_worker::{WorkerMetadata, serving_tokenizer};
use crate::serve::prepare::PreparationHandle;
use crate::tokenizer::bpe::BpeTokenizer;
use std::sync::{Arc, RwLock};

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
        let tokenizer = serving_tokenizer(tokenizer, metadata.model_max_context);
        let tokenizer = Arc::new(tokenizer);
        let preparation =
            PreparationHandle::qwen(Arc::clone(&tokenizer), metadata.model_max_context);
        let runtime =
            QwenMetalRuntime::new(state, tokenizer, vision, metadata.clone(), index, limits);
        Ok((Box::new(runtime), metadata, preparation))
    }
}

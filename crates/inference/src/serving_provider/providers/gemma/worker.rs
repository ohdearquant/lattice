use crate::model::serving_runtime::{GemmaCpuRuntime, RuntimeFactory, ServingRuntime};
use crate::serve::lora::{AdapterIndex, ResidencyLimits};
use crate::serve::metal_worker::ContextWindowPolicy;
use crate::serve::prepare::PreparationHandle;
use crate::serving_cpu::GemmaCpuServing;
use crate::serving_runtime_contract::WorkerMetadata;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

pub(crate) fn factory(model_dir: PathBuf) -> Box<dyn RuntimeFactory> {
    Box::new(GemmaFactory { model_dir })
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

pub(crate) fn gemma_parts(
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

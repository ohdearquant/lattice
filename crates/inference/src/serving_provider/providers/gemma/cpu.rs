use crate::model::serving_runtime::{GemmaPreparedCpuRuntime, ServingRuntime};
use crate::serving_cpu::GemmaCpuServing;
use crate::serving_cpu_host::{CpuRuntimeSource, SharedCpuHandle};
use crate::serving_preparation::PreparationHandle;
use crate::serving_provider::CheckpointEvidence;
use std::sync::Arc;

struct GemmaCpuSource {
    serving: Arc<GemmaCpuServing>,
}

impl CpuRuntimeSource for GemmaCpuSource {
    fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
        Box::new(GemmaPreparedCpuRuntime::new(&self.serving))
    }
}

pub(super) fn load(evidence: &CheckpointEvidence) -> Result<SharedCpuHandle, String> {
    let serving = super::load_cpu(&evidence.directory)
        .map_err(|error| format!("failed to load Gemma 4 model: {error}"))?;
    Ok(from_serving(Arc::new(serving)))
}

pub(crate) fn from_serving(serving: Arc<GemmaCpuServing>) -> SharedCpuHandle {
    let preparation = PreparationHandle::gemma(Arc::clone(&serving));
    let source: Arc<dyn CpuRuntimeSource> = Arc::new(GemmaCpuSource { serving });
    SharedCpuHandle::new(source, preparation)
}

#[cfg(test)]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GemmaCpuSource>();
};

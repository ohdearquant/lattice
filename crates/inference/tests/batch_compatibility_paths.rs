//! Compile-only witnesses for batch compatibility paths and item identity.

use lattice_compute::batch as owner;
use lattice_inference::batch as facade;

const _: fn(facade::BatchConfig) -> owner::config::BatchConfig = |value| value;
const _: fn(facade::FifoScheduler) -> owner::scheduler::FifoScheduler = |value| value;
const _: fn(facade::SchedulerDecision) -> owner::scheduler::SchedulerDecision = |value| value;
const _: fn(facade::AdapterKey) -> owner::sequence::AdapterKey = |value| value;
const _: fn(facade::FinishReason) -> owner::sequence::FinishReason = |value| value;
const _: fn(facade::SeqId) -> owner::sequence::SeqId = |value| value;
const _: fn(facade::Sequence) -> owner::sequence::Sequence = |value| value;
const _: fn(facade::SequenceManager) -> owner::sequence::SequenceManager = |value| value;
const _: fn(facade::SequenceState) -> owner::sequence::SequenceState = |value| value;
const _: fn(facade::BatchStepInput<'static>) -> owner::worker::BatchStepInput<'static> =
    |value| value;
const _: fn(facade::BatchStepOutput) -> owner::worker::BatchStepOutput = |value| value;
const _: fn(facade::BatchWorker) -> owner::worker::BatchWorker = |value| value;
const _: fn(facade::GdnStatePool) -> owner::worker::GdnStatePool = |value| value;
const _: fn(facade::InferenceRequest) -> owner::worker::InferenceRequest = |value| value;
const _: fn(facade::InferenceToken) -> owner::worker::InferenceToken = |value| value;
const _: fn(Box<dyn facade::Scheduler>) -> Box<dyn owner::scheduler::Scheduler> = |value| value;
const _: fn(
    Box<dyn facade::PagedKVCacheConfigExt>,
) -> Box<dyn owner::worker::PagedKVCacheConfigExt> = |value| value;

const _: fn(facade::config::BatchConfig) -> owner::BatchConfig = |value| value;
const _: fn(facade::scheduler::FifoScheduler) -> owner::FifoScheduler = |value| value;
const _: fn(facade::scheduler::SchedulerDecision) -> owner::SchedulerDecision = |value| value;
const _: fn(Box<dyn facade::scheduler::Scheduler>) -> Box<dyn owner::Scheduler> = |value| value;
const _: fn(facade::sequence::AdapterKey) -> owner::AdapterKey = |value| value;
const _: fn(facade::sequence::FinishReason) -> owner::FinishReason = |value| value;
const _: fn(facade::sequence::SeqId) -> owner::SeqId = |value| value;
const _: fn(facade::sequence::Sequence) -> owner::Sequence = |value| value;
const _: fn(facade::sequence::SequenceManager) -> owner::SequenceManager = |value| value;
const _: fn(facade::sequence::SequenceState) -> owner::SequenceState = |value| value;
const _: fn(facade::worker::BatchStepInput<'static>) -> owner::BatchStepInput<'static> =
    |value| value;
const _: fn(facade::worker::BatchStepOutput) -> owner::BatchStepOutput = |value| value;
const _: fn(facade::worker::BatchWorker) -> owner::BatchWorker = |value| value;
const _: fn(facade::worker::GdnStatePool) -> owner::GdnStatePool = |value| value;
const _: fn(facade::worker::InferenceRequest) -> owner::InferenceRequest = |value| value;
const _: fn(facade::worker::InferenceToken) -> owner::InferenceToken = |value| value;
const _: fn(
    Box<dyn facade::worker::PagedKVCacheConfigExt>,
) -> Box<dyn owner::PagedKVCacheConfigExt> = |value| value;

#[allow(deprecated)]
const _: fn(usize, usize, usize) -> facade::worker::GdnStatePool = facade::GdnStatePool::new;
#[allow(deprecated)]
const _: fn(
    facade::config::BatchConfig,
    lattice_inference::kv_cache::PagedKVCacheConfig,
    usize,
    usize,
    Option<u32>,
) -> facade::worker::BatchWorker = facade::BatchWorker::new;
const _: fn(&lattice_compute::kv_cache::PagedKVCacheConfig) -> usize =
    <lattice_inference::kv_cache::PagedKVCacheConfig as facade::worker::PagedKVCacheConfigExt>::floats_per_page_pub;
const _: fn(
    &lattice_inference::kv_cache::PagedKVCacheConfig,
) -> Result<usize, lattice_inference_core::InferenceError> =
    <lattice_compute::kv_cache::PagedKVCacheConfig as facade::PagedKVCacheConfigExt>::try_floats_per_page_pub;

#[cfg(all(test, feature = "test-utils"))]
use super::MetalWorkerClient;
use super::{EmbeddingState, GenerationDefaults, GrammarCache, ServeMetrics};
use std::sync::Arc;

#[cfg(all(test, feature = "test-utils"))]
pub(super) type DefaultServingHost = MetalWorkerClient;
#[cfg(not(all(test, feature = "test-utils")))]
pub(super) type DefaultServingHost = lattice_inference::serving_host::ServingHandle;

pub(super) trait IntoServingHost: Clone + Send + Sync + 'static {
    fn into_serving_host(self) -> lattice_inference::serving_host::ServingHandle;

    fn available_permits(&self) -> usize;

    fn supports_adapters(&self) -> bool;

    fn adapter_client_or(
        &self,
        refusal: lattice_inference::serve::ApiError,
    ) -> Result<&lattice_inference::serving_worker::WorkerClient, lattice_inference::serve::ApiError>;
}

impl IntoServingHost for lattice_inference::serving_host::ServingHandle {
    fn into_serving_host(self) -> lattice_inference::serving_host::ServingHandle {
        self
    }

    fn available_permits(&self) -> usize {
        lattice_inference::serving_host::ServingHandle::available_permits(self)
    }

    fn supports_adapters(&self) -> bool {
        lattice_inference::serving_host::ServingHandle::supports_adapters(self)
    }

    fn adapter_client_or(
        &self,
        refusal: lattice_inference::serve::ApiError,
    ) -> Result<&lattice_inference::serving_worker::WorkerClient, lattice_inference::serve::ApiError>
    {
        lattice_inference::serving_host::ServingHandle::adapter_client_or(self, refusal)
    }
}

#[cfg(all(test, feature = "test-utils"))]
impl IntoServingHost for MetalWorkerClient {
    fn into_serving_host(self) -> lattice_inference::serving_host::ServingHandle {
        lattice_inference::serving_host::ServingHandle::affine_standalone(self)
    }

    fn available_permits(&self) -> usize {
        MetalWorkerClient::available_permits(self)
    }

    fn supports_adapters(&self) -> bool {
        MetalWorkerClient::preparation(self)
            .is_none_or(lattice_inference::serve::prepare::PreparationHandle::supports_adapters)
    }

    fn adapter_client_or(
        &self,
        _refusal: lattice_inference::serve::ApiError,
    ) -> Result<&lattice_inference::serving_worker::WorkerClient, lattice_inference::serve::ApiError>
    {
        Ok(self)
    }
}

#[derive(Clone)]
pub(super) struct AppState<H = DefaultServingHost> {
    pub(super) jobs: H,
    pub(super) model_id: Arc<str>,
    pub(super) defaults: GenerationDefaults,
    /// Runtime context window reported by the selected provider. See
    /// `build_cfg` for request-level clamping.
    pub(super) model_max_context: usize,
    /// The shared worker's outstanding-job admission cap (issue #932),
    /// needed alongside `jobs.available_permits()` to compute the
    /// `/metrics` (#583) queue-depth/in-flight gauge as `max_pending -
    /// available_permits()`.
    pub(super) max_pending: usize,
    /// Process-wide Prometheus metrics registry (#583), `Arc`-shared so
    /// every clone of `AppState` (one per request) records into the same
    /// counters.
    pub(super) metrics: Arc<ServeMetrics>,
    /// The loaded model's tokenizer vocabulary, byte-decoded once at
    /// startup (structured-output v0 design note): `GrammarEngine::new`
    /// needs `vocab_bytes[i]` for every token id to compile a schema
    /// into a PDA + vocabulary partition. `Arc`-shared so every request
    /// clones a pointer, not the ~vocab_size-entry `Vec<Vec<u8>>` --
    /// `GrammarEngine::new` itself still takes ownership of a fresh
    /// clone per compile (no cross-request grammar cache in v0; see the
    /// structured-output v0 design note's stage split).
    pub(super) vocab_bytes: Arc<Vec<Vec<u8>>>,
    /// Bounded (32-entry, LRU) single-flight cache of compiled
    /// `GrammarEngine`s, keyed by `grammar_cache_key` on the
    /// canonicalized admitted schema JSON *and* the `vocab_bytes` this
    /// request was compiled against (structured-output v0 design note,
    /// stage 2). `Arc`-shared so every clone of `AppState` (one per
    /// request) hits the same cache.
    pub(super) grammar_cache: Arc<GrammarCache>,
    /// The optional embedding model loaded via `--embedding-model`
    /// (issue #584). `None` when the flag was not passed at startup, in
    /// which case `POST /v1/embeddings` answers 503
    /// `embedding_model_not_loaded` for every request rather than the
    /// route not existing at all -- the route is always registered so
    /// its absence is a documented, discoverable server state instead of
    /// a bare 404.
    pub(super) embedding: Option<Arc<EmbeddingState>>,
}

impl<H: IntoServingHost> AppState<H> {
    pub(super) fn into_serving_host(
        self,
    ) -> AppState<lattice_inference::serving_host::ServingHandle> {
        let Self {
            jobs,
            model_id,
            defaults,
            model_max_context,
            max_pending,
            metrics,
            vocab_bytes,
            grammar_cache,
            embedding,
        } = self;
        AppState {
            jobs: jobs.into_serving_host(),
            model_id,
            defaults,
            model_max_context,
            max_pending,
            metrics,
            vocab_bytes,
            grammar_cache,
            embedding,
        }
    }
}

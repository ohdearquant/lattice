//! Directory-aware provider selection for serving entry points.

use crate::error::InferenceError;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::forward::metal_qwen35::MetalQwen35State;
use crate::model_format::ModelFormat;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::lora::ResidencyLimits;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::prepare::PreparationHandle;
use crate::serve::route::{RouteRefusal, ServedBackend, ServedRoute};
use crate::serving_embeddings::{PooledEmbeddingHandle, TextEmbeddingHandle};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_factory::ServingFactory;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_runtime_contract::WorkerMetadata;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_worker::{AffineWorker, StartupError, WorkerClient, WorkerOwner};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::tokenizer::bpe::BpeTokenizer;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Semaphore;

pub mod providers;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub use providers::qwen::worker::{
    DEFAULT_SERVE_MAX_CONTEXT, FALLBACK_MODEL_MAX_CONTEXT, ServeContextSource,
    model_context_from_config, parse_max_context, resolve_serve_context,
};

/// A binary's serving route policy.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServingEntry {
    /// The `lattice serve` command.
    Lattice,
    /// The standalone `lattice_serve` binary.
    Standalone,
}

/// Checkpoint facts collected at the serving startup inspection stage.
#[doc(hidden)]
#[derive(Clone)]
pub struct CheckpointEvidence {
    directory: PathBuf,
    format: ModelFormat,
    model_type: Option<String>,
}

/// Options used when opening a `lattice serve` route.
#[doc(hidden)]
pub struct LatticeOpenOptions {
    /// Optional directory containing the Q4 tokenizer.
    pub tokenizer_dir: Option<PathBuf>,
    /// Maximum number of queued and in-flight worker requests.
    pub max_pending: usize,
    /// Maximum number of resident adapter identities.
    pub max_resident_adapters: usize,
    /// Maximum resident adapter tensor payload, in bytes.
    pub max_resident_adapter_bytes: usize,
    /// Whether to eagerly load available vision weights before worker startup.
    pub preload_vision: bool,
    /// Fixed context cap used by the `lattice` command.
    pub context_window: usize,
}

impl CheckpointEvidence {
    /// The checkpoint format observed during inspection.
    pub fn format(&self) -> ModelFormat {
        self.format
    }
}

/// Inspect a checkpoint directory without turning probe failures into errors.
#[doc(hidden)]
pub fn inspect_checkpoint(directory: &Path) -> CheckpointEvidence {
    let format = crate::model_format::detect_format(directory);
    let model_type = crate::model::config_file::read_config_json_bounded(
        &directory.join("config.json"),
        "config.json",
    )
    .ok()
    .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
    .and_then(|root| {
        root.get("model_type")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    });

    CheckpointEvidence {
        directory: directory.to_path_buf(),
        format,
        model_type,
    }
}

/// Why serving provider selection could not produce a route.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionError {
    /// More than one registered provider positively claimed the checkpoint.
    Ambiguous {
        /// The inspected checkpoint directory.
        directory: PathBuf,
        /// The provider names, in sorted order.
        claimants: Vec<&'static str>,
    },
    /// The selected provider refused this serving entry.
    Route {
        /// The inspected checkpoint directory.
        directory: PathBuf,
        /// The existing route refusal.
        refusal: RouteRefusal,
    },
}

impl SelectionError {
    /// Render the established route refusal or the sorted claimant list.
    pub fn message(&self) -> String {
        match self {
            Self::Ambiguous {
                directory,
                claimants,
            } => format!(
                "ambiguous serving providers for '{}': {}",
                directory.display(),
                claimants.join(", ")
            ),
            Self::Route { directory, refusal } => refusal.message(directory),
        }
    }
}

/// Why pooled embedding provider selection could not choose one provider.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingSelectionError {
    /// More than one registered provider claimed the checkpoint.
    Ambiguous {
        /// Provider names in deterministic sorted order.
        providers: Vec<&'static str>,
    },
}

impl std::fmt::Display for EmbeddingSelectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ambiguous { providers } => {
                write!(
                    formatter,
                    "ambiguous pooled embedding providers: {}",
                    providers.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for EmbeddingSelectionError {}

/// Opaque pooled embedding load plan selected from checkpoint evidence.
#[doc(hidden)]
pub struct PooledEmbeddingPlan {
    inner: Box<dyn PooledEmbeddingLoadPlan>,
}

impl PooledEmbeddingPlan {
    /// Loads the selected provider's pooled embedding service.
    pub fn load(self) -> Result<PooledEmbeddingHandle, InferenceError> {
        self.inner.load()
    }
}

/// Opaque text embedding load plan selected from checkpoint evidence.
#[doc(hidden)]
pub struct TextEmbeddingPlan {
    inner: Box<dyn TextEmbeddingLoadPlan>,
}

impl TextEmbeddingPlan {
    /// Loads the selected provider's text embedding service.
    pub fn load(self) -> Result<TextEmbeddingHandle, String> {
        self.inner.load()
    }
}

/// An implicit pooled embedding action selected alongside generation.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplicitPooledEmbeddingPlan {
    /// Checkpoint directory to try after generation loading succeeds.
    pub directory_to_try: Option<PathBuf>,
    /// Startup line to emit when pooled embeddings are disabled.
    pub disabled_line: Option<String>,
}

pub(crate) trait LatticeRoutePlan {
    fn selection_marker(&self) -> String;
    fn implicit_embeddings(&self, model: &str) -> ImplicitPooledEmbeddingPlan;
    fn open(
        self: Box<Self>,
        options: LatticeOpenOptions,
        generation_admission: Arc<Semaphore>,
    ) -> Result<crate::serving_host::ServingHandle, String>;
}

pub(crate) trait RegisteredProvider: Sync {
    fn generation(&self) -> Option<&dyn ServingProvider>;
    fn pooled_embeddings(&self) -> Option<&dyn PooledEmbeddingProvider>;
    fn text_embeddings(&self) -> Option<&dyn TextEmbeddingProvider>;
}

pub(crate) trait PooledEmbeddingProvider: Sync {
    fn name(&self) -> &'static str;
    fn claims(&self, evidence: &CheckpointEvidence) -> bool;
    fn plan(&self, directory: PathBuf) -> Box<dyn PooledEmbeddingLoadPlan>;
}

pub(crate) trait PooledEmbeddingLoadPlan {
    fn load(self: Box<Self>) -> Result<PooledEmbeddingHandle, InferenceError>;
}

pub(crate) trait TextEmbeddingProvider: Sync {
    fn name(&self) -> &'static str;
    fn claims(&self, evidence: &CheckpointEvidence) -> bool;
    fn plan(
        &self,
        directory: PathBuf,
        pooling_override: Option<String>,
    ) -> Box<dyn TextEmbeddingLoadPlan>;
}

pub(crate) trait TextEmbeddingLoadPlan {
    fn load(self: Box<Self>) -> Result<TextEmbeddingHandle, String>;
}

/// Plans a pooled embedding load from a checkpoint directory.
#[doc(hidden)]
pub fn plan_pooled_embeddings(
    directory: &Path,
) -> Result<PooledEmbeddingPlan, EmbeddingSelectionError> {
    let evidence = inspect_checkpoint(directory);
    let provider =
        select_pooled_with_registry(&evidence, &providers::PROVIDERS, &providers::QWEN_PROVIDER)?;
    Ok(PooledEmbeddingPlan {
        inner: provider.plan(evidence.directory),
    })
}

/// Plans a text embedding load from a checkpoint directory.
#[doc(hidden)]
pub fn plan_text_embeddings(
    directory: &Path,
    pooling_override: Option<String>,
) -> Result<TextEmbeddingPlan, EmbeddingSelectionError> {
    let evidence = inspect_checkpoint(directory);
    let provider =
        select_text_with_registry(&evidence, &providers::PROVIDERS, &providers::BERT_PROVIDER)?;
    Ok(TextEmbeddingPlan {
        inner: provider.plan(evidence.directory, pooling_override),
    })
}

/// Presence of standalone flags that a provider may reject.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub struct StandaloneOptionPresence {
    /// Whether `--preload-vision` appeared as a separate argument.
    pub preload_vision: bool,
    /// Whether `--tokenizer-dir` appeared as a separate argument.
    pub tokenizer_dir: bool,
    /// Whether `--max-resident-adapters` appeared as a separate argument.
    pub resident_count: bool,
    /// Whether `--max-resident-adapter-bytes` appeared as a separate argument.
    pub resident_bytes: bool,
    /// Whether the parsed reasoning budget is positive.
    pub positive_reasoning_budget: bool,
}

/// Options used to prepare a standalone load plan.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub struct StandaloneLoadOptions {
    /// Tokenizer file used by the existing Qwen worker loader.
    pub tokenizer_path: PathBuf,
    /// Whether Qwen vision weights should be loaded before worker startup.
    pub preload_vision: bool,
    /// `--max-context`: caps the served context window. `None` serves
    /// the smaller of the model window and the default serving cap.
    pub max_context: Option<usize>,
}

/// Compatibility loader builder for existing standalone factory callers.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub type StandaloneQwenLoader = Box<
    dyn FnOnce(
            PathBuf,
            PathBuf,
            ModelFormat,
        ) -> Box<
            dyn FnOnce() -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String>
                + Send
                + 'static,
        > + Send
        + 'static,
>;

/// Startup text selected with the serving provider.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub struct StandaloneStartupDescription {
    /// Route evidence printed before the loading line.
    pub route_marker: Option<String>,
    /// Format label used by the loading line.
    pub loading_label: &'static str,
}

/// A provider's staged standalone worker load and post-ready continuation.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub struct StandaloneLoadPlan {
    factory: ServingFactory,
    post_ready: StandalonePostReady,
    context_source: Option<ServeContextSource>,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
impl StandaloneLoadPlan {
    pub(crate) fn new(factory: ServingFactory, post_ready: StandalonePostReady) -> Self {
        Self {
            factory,
            post_ready,
            context_source: None,
        }
    }

    pub(crate) fn with_context_source(mut self, source: ServeContextSource) -> Self {
        self.context_source = Some(source);
        self
    }

    /// Open the worker and load the selected provider's model on its thread.
    pub fn open(
        self,
        max_pending: usize,
        residency_limits: ResidencyLimits,
    ) -> Result<StartedStandalone, String> {
        let (owner, jobs, metadata, preparation) = AffineWorker::spawn_with_vision(
            self.factory,
            max_pending,
            residency_limits,
        )
        .map_err(|error| match error {
            StartupError::Load(message) => message,
            StartupError::ThreadExited => "worker thread exited during model load".to_owned(),
            other => other.to_string(),
        })?;
        Ok(StartedStandalone {
            owner,
            jobs,
            metadata,
            preparation,
            post_ready: self.post_ready,
            context_source: self.context_source,
        })
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
type GrammarVocabularyLoad =
    Box<dyn FnOnce() -> Result<std::sync::Arc<Vec<Vec<u8>>>, String> + Send + 'static>;

/// One-shot work that follows the worker's ready line.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub struct StandalonePostReady {
    finish: GrammarVocabularyLoad,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
impl StandalonePostReady {
    pub(crate) fn new(
        finish: impl FnOnce() -> Result<std::sync::Arc<Vec<Vec<u8>>>, String> + Send + 'static,
    ) -> Self {
        Self {
            finish: Box::new(finish),
        }
    }

    /// Complete provider work and return the grammar vocabulary.
    pub fn finish(self) -> Result<std::sync::Arc<Vec<Vec<u8>>>, String> {
        (self.finish)()
    }
}

/// The standalone worker handles and post-ready continuation.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub struct StartedStandalone {
    /// Owner retained for the server lifetime.
    pub owner: WorkerOwner,
    /// Client stored in the standalone application state.
    pub jobs: WorkerClient,
    /// Metadata from the loaded model.
    pub metadata: WorkerMetadata,
    /// Preparation data returned by the worker factory.
    pub preparation: PreparationHandle,
    /// Provider-owned work performed after the ready line.
    pub post_ready: StandalonePostReady,
    /// Where the served context window came from, when the provider chose
    /// it. `None` means the model's own window.
    pub context_source: Option<ServeContextSource>,
}

/// The provider selected for one checkpoint and its admitted serving entry.
#[doc(hidden)]
pub struct SelectedProvider<'a> {
    #[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
    provider: &'a dyn ServingProvider,
    evidence: CheckpointEvidence,
    entry: ServingEntry,
    lattice_plan: Option<Box<dyn LatticeRoutePlan>>,
}

impl SelectedProvider<'_> {
    /// The route exposed by the selected provider's historical interface.
    pub fn legacy_route(&self) -> Option<ServedRoute> {
        self.provider
            .legacy_route(&self.evidence, self.entry)
            .ok()
            .flatten()
    }

    /// Describes the selected provider's implicit pooled embedding action.
    pub fn implicit_pooled_embeddings(&self, model: &str) -> ImplicitPooledEmbeddingPlan {
        self.provider
            .implicit_pooled_embeddings(&self.evidence, model)
    }

    /// Describes the selected provider's implicit pooled embedding action.
    #[doc(hidden)]
    pub fn implicit_embeddings(&self, model: &str) -> ImplicitPooledEmbeddingPlan {
        self.lattice_plan
            .as_ref()
            .map(|plan| plan.implicit_embeddings(model))
            .unwrap_or_else(|| {
                self.provider
                    .implicit_pooled_embeddings(&self.evidence, model)
            })
    }

    /// Returns the startup marker for an admitted `lattice serve` route.
    #[doc(hidden)]
    pub fn lattice_selection_marker(&self) -> Option<String> {
        self.lattice_plan
            .as_ref()
            .map(|plan| plan.selection_marker())
    }

    /// Load the selected provider's CPU route for `lattice serve`.
    pub fn load_lattice_cpu(&self) -> Result<crate::serving_cpu_host::SharedCpuHandle, String> {
        if self.entry != ServingEntry::Lattice {
            return Err("CPU loading requested for a non-lattice entry".to_owned());
        }
        if self
            .legacy_route()
            .is_none_or(|route| route.backend != ServedBackend::Cpu)
        {
            return Err("CPU loading requested for a non-CPU route".to_owned());
        }
        self.provider.load_lattice_cpu(&self.evidence)
    }

    /// Open the selected route for the `lattice serve` command.
    #[doc(hidden)]
    pub fn open_lattice(
        self,
        options: LatticeOpenOptions,
        generation_admission: Arc<Semaphore>,
    ) -> Result<crate::serving_host::ServingHandle, String> {
        if self.entry != ServingEntry::Lattice {
            return Err("lattice loading requested for a non-lattice entry".to_owned());
        }
        let Some(plan) = self.lattice_plan else {
            return Err(crate::model_format::unloadable_format_message(
                &self.evidence.directory,
            ));
        };
        plan.open(options, generation_admission)
    }

    /// Validate standalone flag presence through the selected provider.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub fn validate_standalone_options(
        &self,
        entry: ServingEntry,
        presence: &StandaloneOptionPresence,
    ) -> Result<(), String> {
        self.provider
            .validate_standalone_options(entry, self.legacy_route(), presence)
    }

    /// Describe the route before model loading starts.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub fn standalone_startup_description(&self) -> StandaloneStartupDescription {
        self.provider
            .standalone_startup_description(self.evidence.format, self.legacy_route())
    }

    /// Prepare the selected provider's staged standalone load plan.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub fn prepare_standalone_load(
        self,
        options: StandaloneLoadOptions,
    ) -> Result<StandaloneLoadPlan, String> {
        if self.entry != ServingEntry::Standalone {
            return Err("standalone load plan requested for a non-standalone entry".to_owned());
        }
        let route = self.legacy_route();
        self.provider
            .prepare_standalone_load(self.evidence, route, options)
    }

    /// Build the standalone worker factory through the selected provider.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub fn into_standalone_factory(
        self,
        entry: ServingEntry,
        evidence: CheckpointEvidence,
        options: StandaloneLoadOptions,
        qwen_loader: StandaloneQwenLoader,
    ) -> Result<ServingFactory, String> {
        if entry != ServingEntry::Standalone {
            return Err("standalone factory requested for a non-standalone entry".to_owned());
        }
        self.provider
            .standalone_factory(evidence, options, qwen_loader)
    }
}

pub(crate) trait ServingProvider: Sync {
    fn name(&self) -> &'static str;
    fn claims(&self, evidence: &CheckpointEvidence) -> bool;
    fn admit_lattice(
        &self,
        evidence: &CheckpointEvidence,
    ) -> Result<Box<dyn LatticeRoutePlan>, RouteRefusal>;
    fn legacy_route(
        &self,
        evidence: &CheckpointEvidence,
        entry: ServingEntry,
    ) -> Result<Option<ServedRoute>, RouteRefusal> {
        let _ = (evidence, entry);
        Ok(None)
    }

    fn implicit_pooled_embeddings(
        &self,
        evidence: &CheckpointEvidence,
        model: &str,
    ) -> ImplicitPooledEmbeddingPlan {
        let _ = model;
        ImplicitPooledEmbeddingPlan {
            directory_to_try: Some(evidence.directory.clone()),
            disabled_line: None,
        }
    }

    fn load_lattice_cpu(
        &self,
        _evidence: &CheckpointEvidence,
    ) -> Result<crate::serving_cpu_host::SharedCpuHandle, String> {
        Err("selected provider does not load a lattice CPU route".to_owned())
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn validate_standalone_options(
        &self,
        entry: ServingEntry,
        route: Option<ServedRoute>,
        presence: &StandaloneOptionPresence,
    ) -> Result<(), String>;

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn standalone_startup_description(
        &self,
        format: ModelFormat,
        route: Option<ServedRoute>,
    ) -> StandaloneStartupDescription;

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn prepare_standalone_load(
        &self,
        evidence: CheckpointEvidence,
        route: Option<ServedRoute>,
        options: StandaloneLoadOptions,
    ) -> Result<StandaloneLoadPlan, String>;

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn standalone_factory(
        &self,
        evidence: CheckpointEvidence,
        options: StandaloneLoadOptions,
        qwen_loader: StandaloneQwenLoader,
    ) -> Result<ServingFactory, String>;
}

pub(crate) struct ProviderRegistry<'a> {
    providers: &'a [&'a dyn RegisteredProvider],
    legacy_generation: &'a dyn ServingProvider,
}

/// Select a provider and admit one serving entry.
#[doc(hidden)]
pub fn select(
    evidence: CheckpointEvidence,
    entry: ServingEntry,
) -> Result<SelectedProvider<'static>, SelectionError> {
    let registry = ProviderRegistry {
        providers: &providers::PROVIDERS,
        legacy_generation: &providers::QWEN_PROVIDER,
    };
    select_with_registry(evidence, entry, &registry)
}

pub(crate) fn select_with_registry<'a>(
    evidence: CheckpointEvidence,
    entry: ServingEntry,
    registry: &ProviderRegistry<'a>,
) -> Result<SelectedProvider<'a>, SelectionError> {
    let provider = match unique_claim(
        registry
            .providers
            .iter()
            .filter_map(|provider| provider.generation()),
        |provider| provider.claims(&evidence),
        ServingProvider::name,
    ) {
        Ok(Some(provider)) => provider,
        Ok(None) => registry.legacy_generation,
        Err(claimants) => {
            return Err(SelectionError::Ambiguous {
                directory: evidence.directory.clone(),
                claimants,
            });
        }
    };
    let lattice_plan = match entry {
        ServingEntry::Lattice => {
            let plan =
                provider
                    .admit_lattice(&evidence)
                    .map_err(|refusal| SelectionError::Route {
                        directory: evidence.directory.clone(),
                        refusal,
                    })?;
            Some(plan)
        }
        ServingEntry::Standalone => {
            provider
                .legacy_route(&evidence, entry)
                .map_err(|refusal| SelectionError::Route {
                    directory: evidence.directory.clone(),
                    refusal,
                })?;
            None
        }
    };

    Ok(SelectedProvider {
        provider,
        evidence,
        entry,
        lattice_plan,
    })
}

fn unique_claim<'a, T: ?Sized>(
    providers: impl Iterator<Item = &'a T>,
    claims: impl Fn(&T) -> bool,
    name: impl Fn(&T) -> &'static str,
) -> Result<Option<&'a T>, Vec<&'static str>> {
    let claimants = providers
        .filter(|provider| claims(provider))
        .collect::<Vec<_>>();
    if claimants.len() > 1 {
        let mut names = claimants
            .iter()
            .map(|provider| name(provider))
            .collect::<Vec<_>>();
        names.sort_unstable();
        return Err(names);
    }
    Ok(claimants.into_iter().next())
}

fn select_pooled_with_registry<'a>(
    evidence: &CheckpointEvidence,
    providers: &'a [&'a dyn RegisteredProvider],
    legacy: &'a dyn PooledEmbeddingProvider,
) -> Result<&'a dyn PooledEmbeddingProvider, EmbeddingSelectionError> {
    match unique_claim(
        providers
            .iter()
            .filter_map(|provider| provider.pooled_embeddings()),
        |provider| provider.claims(evidence),
        PooledEmbeddingProvider::name,
    ) {
        Ok(Some(provider)) => Ok(provider),
        Ok(None) => Ok(legacy),
        Err(providers) => Err(EmbeddingSelectionError::Ambiguous { providers }),
    }
}

fn select_text_with_registry<'a>(
    evidence: &CheckpointEvidence,
    providers: &'a [&'a dyn RegisteredProvider],
    legacy: &'a dyn TextEmbeddingProvider,
) -> Result<&'a dyn TextEmbeddingProvider, EmbeddingSelectionError> {
    match unique_claim(
        providers
            .iter()
            .filter_map(|provider| provider.text_embeddings()),
        |provider| provider.claims(evidence),
        TextEmbeddingProvider::name,
    ) {
        Ok(Some(provider)) => Ok(provider),
        Ok(None) => Ok(legacy),
        Err(providers) => Err(EmbeddingSelectionError::Ambiguous { providers }),
    }
}

#[cfg(test)]
mod tests;

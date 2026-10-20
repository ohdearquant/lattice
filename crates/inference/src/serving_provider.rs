//! Directory-aware provider selection for serving entry points.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::forward::metal_qwen35::MetalQwen35State;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::model_format::ModelFamily;
use crate::model_format::ModelFormat;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::lora::ResidencyLimits;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::prepare::PreparationHandle;
use crate::serve::route::{RouteRefusal, ServedBackend, ServedRoute};
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
    /// The selected provider's legacy route projection refused this entry.
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

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) fn standalone_startup_description(
    format: ModelFormat,
    route: Option<ServedRoute>,
) -> StandaloneStartupDescription {
    let loading_label = if route.is_some_and(|route| route.family == ModelFamily::Gemma4) {
        "safetensors"
    } else {
        match format {
            ModelFormat::Q4 => "q4",
            ModelFormat::Safetensors => "bf16",
            ModelFormat::Unknown => "unknown",
            _ => "unknown",
        }
    };
    StandaloneStartupDescription {
        route_marker: route.map(|route| route.selection_marker(format)),
        loading_label,
    }
}

/// The provider selected for one checkpoint and its existing route projection.
#[doc(hidden)]
pub struct SelectedProvider<'a> {
    #[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
    provider: &'a dyn ServingProvider,
    evidence: CheckpointEvidence,
    entry: ServingEntry,
    legacy_route: Option<ServedRoute>,
}

impl SelectedProvider<'_> {
    /// The route the entry has historically exposed, if the entry defers it.
    pub fn legacy_route(&self) -> Option<ServedRoute> {
        self.legacy_route
    }

    /// Load the selected provider's CPU route for `lattice serve`.
    pub fn load_lattice_cpu(&self) -> Result<crate::serving_cpu_host::SharedCpuHandle, String> {
        if self.entry != ServingEntry::Lattice {
            return Err("CPU loading requested for a non-lattice entry".to_owned());
        }
        if self
            .legacy_route
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
        let Some(route) = self.legacy_route else {
            return Err(crate::model_format::unloadable_format_message(
                &self.evidence.directory,
            ));
        };
        match route.backend {
            ServedBackend::Cpu => {
                #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
                let _ = options;
                let cpu = self.load_lattice_cpu()?;
                Ok(crate::serving_host::ServingHandle::shared_cpu(
                    cpu,
                    generation_admission,
                ))
            }
            ServedBackend::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
                {
                    self.provider
                        .open_lattice_metal(self.evidence, options)
                        .map_err(|error| format!("failed to load Q4 model: {error}"))
                }
                #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
                {
                    let _ = (options, generation_admission);
                    Err(crate::model_format::metal_gpu_required_message(
                        &self.evidence.directory,
                    ))
                }
            }
        }
    }

    /// Validate standalone flag presence through the selected provider.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub fn validate_standalone_options(
        &self,
        entry: ServingEntry,
        presence: &StandaloneOptionPresence,
    ) -> Result<(), String> {
        self.provider
            .validate_standalone_options(entry, self.legacy_route, presence)
    }

    /// Describe the route before model loading starts.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub fn standalone_startup_description(&self) -> StandaloneStartupDescription {
        self.provider
            .standalone_startup_description(self.evidence.format, self.legacy_route)
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
        self.provider
            .prepare_standalone_load(self.evidence, self.legacy_route, options)
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
    fn legacy_route(
        &self,
        evidence: &CheckpointEvidence,
        entry: ServingEntry,
    ) -> Result<Option<ServedRoute>, RouteRefusal>;

    fn load_lattice_cpu(
        &self,
        _evidence: &CheckpointEvidence,
    ) -> Result<crate::serving_cpu_host::SharedCpuHandle, String> {
        Err("selected provider does not load a lattice CPU route".to_owned())
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn open_lattice_metal(
        &self,
        _evidence: CheckpointEvidence,
        _options: LatticeOpenOptions,
    ) -> Result<crate::serving_host::ServingHandle, String> {
        Err("selected provider does not load a lattice Metal route".to_owned())
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
    providers: &'a [&'a dyn ServingProvider],
    legacy: &'a dyn ServingProvider,
}

static LEGACY_PROVIDER: &dyn ServingProvider = &providers::QWEN_PROVIDER;

/// Select a provider and project the route for one serving entry.
#[doc(hidden)]
pub fn select(
    evidence: CheckpointEvidence,
    entry: ServingEntry,
) -> Result<SelectedProvider<'static>, SelectionError> {
    let registry = ProviderRegistry {
        providers: &providers::PROVIDERS,
        legacy: LEGACY_PROVIDER,
    };
    select_with_registry(evidence, entry, &registry)
}

pub(crate) fn select_with_registry<'a>(
    evidence: CheckpointEvidence,
    entry: ServingEntry,
    registry: &ProviderRegistry<'a>,
) -> Result<SelectedProvider<'a>, SelectionError> {
    let claimants = registry
        .providers
        .iter()
        .copied()
        .filter(|provider| provider.claims(&evidence))
        .collect::<Vec<_>>();

    if claimants.len() > 1 {
        let mut names = claimants
            .iter()
            .map(|provider| provider.name())
            .collect::<Vec<_>>();
        names.sort_unstable();
        return Err(SelectionError::Ambiguous {
            directory: evidence.directory.clone(),
            claimants: names,
        });
    }

    let provider = if claimants.len() == 1 {
        claimants[0]
    } else {
        registry.legacy
    };
    let legacy_route =
        provider
            .legacy_route(&evidence, entry)
            .map_err(|refusal| SelectionError::Route {
                directory: evidence.directory.clone(),
                refusal,
            })?;

    Ok(SelectedProvider {
        provider,
        evidence,
        entry,
        legacy_route,
    })
}

#[cfg(test)]
mod tests;

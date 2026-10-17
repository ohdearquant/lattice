//! Directory-aware provider selection for serving entry points.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::forward::metal_qwen35::MetalQwen35State;
use crate::model_format::ModelFormat;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::metal_worker::WorkerMetadata;
use crate::serve::route::{RouteRefusal, ServedBackend, ServedRoute};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_factory::ServingFactory;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::tokenizer::bpe::BpeTokenizer;
use std::path::{Path, PathBuf};

pub mod providers;

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

/// Options needed to construct the standalone serving factory.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub struct StandaloneLoadOptions {
    /// Tokenizer file used by the existing Qwen worker loader.
    pub tokenizer_path: PathBuf,
    /// Whether Qwen vision weights should be loaded before worker startup.
    pub preload_vision: bool,
}

/// Binary-owned worker loader builder retained at the standalone call site.
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
    fn validate_standalone_options(
        &self,
        entry: ServingEntry,
        route: Option<ServedRoute>,
        presence: &StandaloneOptionPresence,
    ) -> Result<(), String>;

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

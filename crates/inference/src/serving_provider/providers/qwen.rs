//! Qwen serving provider and typed CPU loader facade.

use crate::error::InferenceError;
use crate::model::qwen35::Qwen35Model;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::model::qwen35_config::Qwen35Config;
use crate::model_format::ModelFamily;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::metal_worker::VisionRuntime;
use crate::serve::route::{RouteRefusal, ServedRoute, select_route, select_standalone_route};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_factory::ServingFactory;
use crate::serving_provider::{CheckpointEvidence, ServingEntry, ServingProvider};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_provider::{
    StandaloneLoadOptions, StandaloneOptionPresence, StandaloneQwenLoader,
};
use std::path::Path;

pub(in crate::serving_provider) struct QwenProvider;

impl ServingProvider for QwenProvider {
    fn name(&self) -> &'static str {
        "qwen"
    }

    fn claims(&self, evidence: &CheckpointEvidence) -> bool {
        matches!(
            evidence.model_type.as_deref(),
            Some("qwen3_5" | "qwen3_5_moe")
        )
    }

    fn legacy_route(
        &self,
        evidence: &CheckpointEvidence,
        entry: ServingEntry,
    ) -> Result<Option<ServedRoute>, RouteRefusal> {
        match entry {
            ServingEntry::Lattice => select_route(evidence.format, ModelFamily::Qwen35).map(Some),
            ServingEntry::Standalone => {
                match select_standalone_route(evidence.format, ModelFamily::Qwen35) {
                    Ok(route) => Ok(Some(route)),
                    Err(RouteRefusal::UnrecognizedFormat) => Ok(None),
                    Err(refusal) => Err(refusal),
                }
            }
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn validate_standalone_options(
        &self,
        _entry: ServingEntry,
        _route: Option<ServedRoute>,
        _presence: &StandaloneOptionPresence,
    ) -> Result<(), String> {
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn standalone_factory(
        &self,
        evidence: CheckpointEvidence,
        options: StandaloneLoadOptions,
        qwen_loader: StandaloneQwenLoader,
    ) -> Result<ServingFactory, String> {
        qwen_standalone_factory(evidence, options, qwen_loader)
    }
}

/// Load a Qwen safetensors CPU model using the existing model loader.
#[doc(hidden)]
pub fn load_cpu(dir: &Path) -> Result<Qwen35Model, InferenceError> {
    Qwen35Model::from_safetensors(dir)
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(in crate::serving_provider) fn qwen_standalone_factory(
    evidence: CheckpointEvidence,
    options: StandaloneLoadOptions,
    qwen_loader: StandaloneQwenLoader,
) -> Result<ServingFactory, String> {
    let config = Qwen35Config::from_model_dir(&evidence.directory)
        .map_err(|error| format!("config.json load failed: {error}"))?;
    let mut vision_runtime = VisionRuntime::from_model_config(evidence.directory.clone(), &config);
    if options.preload_vision
        && let Err(error) = vision_runtime.preload()
    {
        eprintln!(
            "[lattice_serve] WARNING: --preload-vision failed, falling back to lazy \
             vision loading: {error}"
        );
    }
    let loader = qwen_loader(evidence.directory, options.tokenizer_path, evidence.format);
    Ok(ServingFactory::qwen_metal(loader, vision_runtime))
}

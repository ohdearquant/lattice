//! Gemma serving provider and typed CPU loader facade.

use crate::error::InferenceError;
use crate::model_format::ModelFamily;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::model_format::ModelFormat;
use crate::serve::route::{RouteRefusal, ServedRoute, select_route, select_standalone_route};
use crate::serving_cpu::GemmaCpuServing;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_factory::ServingFactory;
use crate::serving_provider::{CheckpointEvidence, ServingEntry, ServingProvider};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_provider::{
    StandaloneLoadOptions, StandaloneOptionPresence, StandaloneQwenLoader,
};
use std::path::Path;

pub(crate) mod cpu;
mod preparation;

#[cfg(test)]
pub(crate) use preparation::template_trim;
pub(crate) use preparation::{lora_unsupported_backend, prepare_gemma_chat_request};

pub(in crate::serving_provider) struct GemmaProvider;

impl ServingProvider for GemmaProvider {
    fn name(&self) -> &'static str {
        "gemma"
    }

    fn claims(&self, evidence: &CheckpointEvidence) -> bool {
        evidence.model_type.as_deref() == Some("gemma4")
    }

    fn legacy_route(
        &self,
        evidence: &CheckpointEvidence,
        entry: ServingEntry,
    ) -> Result<Option<ServedRoute>, RouteRefusal> {
        match entry {
            ServingEntry::Lattice => select_route(evidence.format, ModelFamily::Gemma4).map(Some),
            ServingEntry::Standalone => {
                match select_standalone_route(evidence.format, ModelFamily::Gemma4) {
                    Ok(route) => Ok(Some(route)),
                    Err(RouteRefusal::UnrecognizedFormat) => Ok(None),
                    Err(refusal) => Err(refusal),
                }
            }
        }
    }

    fn load_lattice_cpu(
        &self,
        evidence: &CheckpointEvidence,
    ) -> Result<crate::serving_cpu_host::SharedCpuHandle, String> {
        cpu::load(evidence)
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn validate_standalone_options(
        &self,
        entry: ServingEntry,
        route: Option<ServedRoute>,
        presence: &StandaloneOptionPresence,
    ) -> Result<(), String> {
        if entry != ServingEntry::Standalone
            || route.is_none_or(|route| route.family != ModelFamily::Gemma4)
        {
            return Ok(());
        }

        for (present, flag) in [
            (presence.preload_vision, "--preload-vision"),
            (presence.tokenizer_dir, "--tokenizer-dir"),
            (presence.resident_count, "--max-resident-adapters"),
            (presence.resident_bytes, "--max-resident-adapter-bytes"),
        ] {
            if present {
                return Err(format!(
                    "unsupported_feature: {flag} is not supported for Gemma 4 checkpoints"
                ));
            }
        }
        if presence.positive_reasoning_budget {
            return Err(
                "unsupported_feature: --reasoning-budget is not supported for Gemma 4 checkpoints"
                    .to_owned(),
            );
        }
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn standalone_factory(
        &self,
        evidence: CheckpointEvidence,
        options: StandaloneLoadOptions,
        qwen_loader: StandaloneQwenLoader,
    ) -> Result<ServingFactory, String> {
        if evidence.format == ModelFormat::Safetensors {
            Ok(ServingFactory::gemma_cpu(evidence.directory))
        } else {
            super::qwen::qwen_standalone_factory(evidence, options, qwen_loader)
        }
    }
}

/// Load Gemma's CPU serving resources using the existing serving loader.
#[doc(hidden)]
pub fn load_cpu(dir: &Path) -> Result<GemmaCpuServing, InferenceError> {
    GemmaCpuServing::load(dir)
}

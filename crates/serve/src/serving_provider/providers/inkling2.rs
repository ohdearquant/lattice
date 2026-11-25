//! Inkling 2 serving provider: text generation on the CPU for `lattice serve`.
//!
//! A checkpoint whose `config.json` has a top-level `model_type` of `inkling2_loop` is claimed
//! here. A safetensors directory is served on the CPU; every other format is refused as
//! unrecognized, and the standalone `lattice_serve` binary refuses the checkpoint by name.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::route::INKLING2_STANDALONE_UNSUPPORTED_CODE;
use crate::serve::route::{RouteRefusal, ServedRoute, select_route};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_factory::ServingFactory;
use crate::serving_host::ServingHandle;
use crate::serving_provider::{
    CheckpointEvidence, ImplicitPooledEmbeddingPlan, LatticeOpenOptions, LatticeRoutePlan,
    RegisteredProvider, ServingEntry, ServingProvider,
};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_provider::{
    StandaloneLoadOptions, StandaloneLoadPlan, StandaloneOptionPresence, StandaloneQwenLoader,
    StandaloneStartupDescription,
};
use lattice_inference::model_format::ModelFamily;
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use lattice_inference::model_format::ModelFormat;
use std::sync::Arc;

pub(crate) mod cpu;
mod preparation;
#[cfg(test)]
mod reasoning_tests;
#[cfg(test)]
mod tests;

/// The top-level `config.json` `model_type` of an Inkling 2 checkpoint.
pub(crate) const INKLING2_MODEL_TYPE: &str = "inkling2_loop";

pub(in crate::serving_provider) struct Inkling2Provider;

struct Inkling2LatticeRoutePlan {
    evidence: CheckpointEvidence,
}

fn embeddings_disabled(model: &str) -> ImplicitPooledEmbeddingPlan {
    ImplicitPooledEmbeddingPlan {
        directory_to_try: None,
        disabled_line: Some(format!(
            "Embeddings disabled ({model}): not available for Inkling 2 checkpoints."
        )),
    }
}

impl LatticeRoutePlan for Inkling2LatticeRoutePlan {
    fn selection_marker(&self) -> String {
        ServedRoute::INKLING2_CPU.selection_marker(self.evidence.format)
    }

    fn implicit_embeddings(&self, model: &str) -> ImplicitPooledEmbeddingPlan {
        embeddings_disabled(model)
    }

    fn open(
        self: Box<Self>,
        options: LatticeOpenOptions,
        generation_admission: Arc<tokio::sync::Semaphore>,
    ) -> Result<ServingHandle, String> {
        let cpu = cpu::load(&self.evidence, options.max_context)?;
        Ok(ServingHandle::shared_cpu(cpu, generation_admission))
    }
}

impl ServingProvider for Inkling2Provider {
    fn name(&self) -> &'static str {
        "inkling2"
    }

    fn claims(&self, evidence: &CheckpointEvidence) -> bool {
        evidence.model_type.as_deref() == Some(INKLING2_MODEL_TYPE)
    }

    fn admit_lattice(
        &self,
        evidence: &CheckpointEvidence,
    ) -> Result<Box<dyn LatticeRoutePlan>, RouteRefusal> {
        select_route(evidence.format, ModelFamily::Inkling2)?;
        Ok(Box::new(Inkling2LatticeRoutePlan {
            evidence: evidence.clone(),
        }))
    }

    fn legacy_route(
        &self,
        evidence: &CheckpointEvidence,
        entry: ServingEntry,
    ) -> Result<Option<ServedRoute>, RouteRefusal> {
        match entry {
            ServingEntry::Lattice => select_route(evidence.format, ModelFamily::Inkling2).map(Some),
            ServingEntry::Standalone => Err(RouteRefusal::Inkling2StandaloneUnsupported),
        }
    }

    fn implicit_pooled_embeddings(
        &self,
        _evidence: &CheckpointEvidence,
        model: &str,
    ) -> ImplicitPooledEmbeddingPlan {
        embeddings_disabled(model)
    }

    fn load_lattice_cpu(
        &self,
        evidence: &CheckpointEvidence,
    ) -> Result<crate::serving_cpu_host::SharedCpuHandle, String> {
        cpu::load(evidence, None)
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn validate_standalone_options(
        &self,
        entry: ServingEntry,
        _route: Option<ServedRoute>,
        _presence: &StandaloneOptionPresence,
    ) -> Result<(), String> {
        match entry {
            ServingEntry::Lattice => Ok(()),
            ServingEntry::Standalone => Err(standalone_refusal()),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn standalone_startup_description(
        &self,
        format: ModelFormat,
        route: Option<ServedRoute>,
    ) -> StandaloneStartupDescription {
        StandaloneStartupDescription {
            route_marker: route.map(|route| route.selection_marker(format)),
            loading_label: match format {
                ModelFormat::Safetensors => "safetensors",
                ModelFormat::Q4 => "q4",
                _ => "unknown",
            },
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn prepare_standalone_load(
        &self,
        _evidence: CheckpointEvidence,
        _route: Option<ServedRoute>,
        _options: StandaloneLoadOptions,
    ) -> Result<StandaloneLoadPlan, String> {
        Err(standalone_refusal())
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn standalone_factory(
        &self,
        _evidence: CheckpointEvidence,
        _options: StandaloneLoadOptions,
        _qwen_loader: StandaloneQwenLoader,
    ) -> Result<ServingFactory, String> {
        Err(standalone_refusal())
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn standalone_refusal() -> String {
    format!(
        "{INKLING2_STANDALONE_UNSUPPORTED_CODE}: Inkling 2 checkpoints are served by `lattice \
         serve` on the CPU backend, not by the standalone lattice_serve binary"
    )
}

impl RegisteredProvider for Inkling2Provider {
    fn generation(&self) -> Option<&dyn ServingProvider> {
        Some(self)
    }

    fn pooled_embeddings(&self) -> Option<&dyn crate::serving_provider::PooledEmbeddingProvider> {
        None
    }

    fn text_embeddings(&self) -> Option<&dyn crate::serving_provider::TextEmbeddingProvider> {
        None
    }
}

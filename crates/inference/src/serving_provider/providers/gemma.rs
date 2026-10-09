//! Gemma serving provider and typed CPU loader facade.

use crate::error::InferenceError;
use crate::model_format::ModelFamily;
use crate::serve::route::{RouteRefusal, ServedRoute, select_route, select_standalone_route};
use crate::serving_cpu::GemmaCpuServing;
use crate::serving_provider::{CheckpointEvidence, ServingEntry, ServingProvider};
use std::path::Path;

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
}

/// Load Gemma's CPU serving resources using the existing serving loader.
#[doc(hidden)]
pub fn load_cpu(dir: &Path) -> Result<GemmaCpuServing, InferenceError> {
    GemmaCpuServing::load(dir)
}

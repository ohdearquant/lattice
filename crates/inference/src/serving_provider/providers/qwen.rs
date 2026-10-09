//! Qwen serving provider and typed CPU loader facade.

use crate::error::InferenceError;
use crate::model::qwen35::Qwen35Model;
use crate::model_format::ModelFamily;
use crate::serve::route::{RouteRefusal, ServedRoute, select_route, select_standalone_route};
use crate::serving_provider::{CheckpointEvidence, ServingEntry, ServingProvider};
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
}

/// Load a Qwen safetensors CPU model using the existing model loader.
#[doc(hidden)]
pub fn load_cpu(dir: &Path) -> Result<Qwen35Model, InferenceError> {
    Qwen35Model::from_safetensors(dir)
}

use crate::serving_provider::{
    CheckpointEvidence, RegisteredProvider, TextEmbeddingLoadPlan, TextEmbeddingProvider,
};
use std::path::PathBuf;

pub(crate) mod embedding;

pub(in crate::serving_provider) struct BertProvider;

impl RegisteredProvider for BertProvider {
    fn generation(&self) -> Option<&dyn crate::serving_provider::ServingProvider> {
        None
    }

    fn pooled_embeddings(&self) -> Option<&dyn crate::serving_provider::PooledEmbeddingProvider> {
        None
    }

    fn text_embeddings(&self) -> Option<&dyn TextEmbeddingProvider> {
        Some(self)
    }
}

impl TextEmbeddingProvider for BertProvider {
    fn name(&self) -> &'static str {
        "bert"
    }

    fn claims(&self, evidence: &CheckpointEvidence) -> bool {
        evidence.model_type.as_deref() == Some("bert")
    }

    fn plan(
        &self,
        directory: PathBuf,
        pooling_override: Option<String>,
    ) -> Box<dyn TextEmbeddingLoadPlan> {
        Box::new(embedding::BertTextEmbeddingPlan {
            directory,
            pooling_override,
        })
    }
}

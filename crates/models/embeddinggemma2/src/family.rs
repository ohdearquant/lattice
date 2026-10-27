//! EmbeddingGemma 2 family registration.

use crate::encoder::EmbeddingGemma2Encoder;
use crate::model::EmbeddingGemma2Model;
use lattice_arch::encoder::Encoder;
use lattice_arch::family::{FamilyDescriptor, LoadRequest};
use lattice_inference_core::InferenceError;
use std::sync::Arc;

fn load_embeddinggemma2(request: &LoadRequest) -> Result<Arc<dyn Encoder>, InferenceError> {
    let model = EmbeddingGemma2Model::from_model_dir(request.model_dir())?;
    Ok(Arc::new(EmbeddingGemma2Encoder::new(
        Arc::new(model),
        None,
    )?))
}

/// EmbeddingGemma 2 text embeddings on the CPU at the native width. Resolved by family name: it
/// claims no `model_type`.
pub static EMBEDDING_GEMMA2: FamilyDescriptor =
    FamilyDescriptor::encoder("embeddinggemma2", &[], load_embeddinggemma2);

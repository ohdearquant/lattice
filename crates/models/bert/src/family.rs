//! BERT-family text encoder registration.

use crate::BertModel;
use crate::encoder::BertEncoder;
use lattice_arch::encoder::Encoder;
use lattice_arch::family::{FamilyDescriptor, LoadRequest};
use lattice_inference_core::InferenceError;
use std::sync::Arc;

fn load_bert(request: &LoadRequest) -> Result<Arc<dyn Encoder>, InferenceError> {
    let model = BertModel::from_directory(request.model_dir())?;
    Ok(Arc::new(BertEncoder::new(Arc::new(model))?))
}

/// BERT-family text embeddings on the CPU: `model_type` `bert`.
pub static BERT: FamilyDescriptor = FamilyDescriptor::encoder("bert", &["bert"], load_bert);

//! Qwen3.5-2B text generation model.
//!
//! Hybrid architecture: 18 GatedDeltaNet (linear attention) + 6 standard GQA layers.
//! Pattern: [linear, linear, linear, full] x 6 = 24 layers.
//!
//! Key features:
//! - Autoregressive text generation with temperature/top-k/top-p sampling
//! - Tied embeddings (lm_head = embed_tokens^T)
//! - Partial RoPE (25% of head_dim) on full-attention layers
//! - GatedDeltaNet recurrent state for linear layers, KV cache for full layers

mod cache;
mod debug;
mod detokenize;
mod forward;
mod generation;
mod loading;
mod model;
mod moe;
mod norm;
mod sampling;
mod weights;

#[cfg(test)]
mod tests;

pub use model::Qwen35Model;
pub use weights::ModelWeights;

pub(crate) use cache::{ForwardScratch, KvCache, resize};
pub(crate) use detokenize::decode_tokens;
pub(crate) use norm::qwen35_rms_norm;
pub(crate) use sampling::sample_token;
pub(crate) use weights::{
    AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
    FullAttentionLayerWeights,
};

#[cfg(test)]
pub use detokenize::bytes_to_unicode;
#[cfg(test)]
pub use generation::should_stop_token;
#[cfg(test)]
pub use loading::qwen_required_tensor_names;

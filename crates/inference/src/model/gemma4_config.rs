//! Gemma 4 configuration facade over `lattice-model-config`.
//!
//! Every public item keeps the path it had before the configuration moved into its own crate. The
//! re-exports are named rather than globbed so the facade never widens the supported API.

pub(crate) use lattice_model_config::gemma4_config::resolve_stop_token_ids;
#[doc(inline)]
pub use lattice_model_config::gemma4_config::{
    GEMMA4_EXPECTED_DTYPE, Gemma4Config, Gemma4LayerType,
};

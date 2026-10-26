//! Generation facade over `lattice-arch`.
//!
//! Every public item keeps the path it had before the generation types moved into `lattice-arch`
//! (ADR-103 Amendment 21). The re-exports are named rather than globbed so the facade never widens
//! the supported API. The decode-policy items below them are crate-internal glue.

#[doc(inline)]
pub use lattice_arch::generation::{GenerateConfig, GenerateOutput, TokenLogprob, TopLogprob};

pub(crate) use lattice_arch::__internal::{
    DecodePolicy, StepOutcome, StopCheckOutcome, truncate_token_logprobs_to_retained_text,
};

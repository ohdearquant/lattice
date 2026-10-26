//! Sampling facade over `lattice-arch`.
//!
//! Every public item keeps the path it had before sampling moved into `lattice-arch` (ADR-103
//! Amendment 21). The re-exports are named rather than globbed so the facade never widens the
//! supported API. The helpers below them are crate-internal glue, each imported under the
//! configurations whose code calls it.

#[doc(inline)]
pub use lattice_arch::sampling::{Candidate, CandidateSet, Sampler, SamplingConfig};

#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(crate) use lattice_arch::__internal::{
    argmax_f32_first_wins, temperature_degenerate, uniform_f32_from_u64,
};
pub(crate) use lattice_arch::__internal::{compute_step_logprobs, sample_full_logits};
#[cfg(test)]
pub(crate) use lattice_arch::__internal::{penalized_logit, xorshift64_next};

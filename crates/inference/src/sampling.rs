//! Sampling facade over `lattice-arch`.
//!
//! Every public item keeps the path it had before sampling moved into `lattice-arch` (ADR-103
//! Amendment 21). The re-exports are named rather than globbed so the facade never widens the
//! supported API.

#[doc(inline)]
pub use lattice_arch::sampling::{Candidate, CandidateSet, Sampler, SamplingConfig};

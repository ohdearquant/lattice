#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(crate) use lattice_arch::__internal::{
    argmax_f32_first_wins, temperature_degenerate, uniform_f32_from_u64,
};
pub(crate) use lattice_arch::__internal::{compute_step_logprobs, sample_full_logits};
#[cfg(test)]
pub(crate) use lattice_arch::__internal::{penalized_logit, xorshift64_next};
pub(crate) use lattice_arch::sampling::{Candidate, CandidateSet, Sampler};

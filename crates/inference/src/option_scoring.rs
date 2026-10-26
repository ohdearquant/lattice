//! Option-letter scoring, re-exported from `lattice-compute`.
//!
//! Named exports keep the existing public path; the letter-id validation helper is public in compute
//! and imported here as `pub(crate)` so the facade's public listing is unchanged.

pub(crate) use lattice_compute::option_scoring::check_letter_ids;
#[doc(inline)]
pub use lattice_compute::option_scoring::{
    DecisionLabelTable, LETTER_LABELS, OptionScores, option_scores_from_logits,
    resolve_decision_labels, resolve_option_letters,
};

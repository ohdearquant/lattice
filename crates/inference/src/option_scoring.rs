//! Option-letter scoring, re-exported from `lattice-compute`.
//!
//! Named exports keep the existing public path.

#[doc(inline)]
pub use lattice_compute::option_scoring::{
    DecisionLabelTable, LETTER_LABELS, OptionScores, option_scores_from_logits,
    resolve_decision_labels, resolve_option_letters,
};

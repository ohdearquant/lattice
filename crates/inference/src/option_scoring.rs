//! Option-letter scoring, re-exported from `lattice-compute`.
//!
//! Named exports keep the existing public path.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-compute` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-compute` directly.

#[doc(inline)]
pub use lattice_compute::option_scoring::{
    DecisionLabelTable, LETTER_LABELS, OptionScores, option_scores_from_logits,
    resolve_decision_labels, resolve_option_letters,
};

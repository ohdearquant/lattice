//! Test support for the checkpoint-gated unit tests in [`crate::model`].
//!
//! [`require_checkpoint_dir`] lives in `lattice_model_io::test_support`, so the model crates
//! that leave this one can share it. This module keeps the crate-local path the unit tests call,
//! and its tests keep the source-policy guards over this crate's own tree.

pub(crate) use lattice_model_io::test_support::require_checkpoint_dir;

#[cfg(test)]
mod tests;

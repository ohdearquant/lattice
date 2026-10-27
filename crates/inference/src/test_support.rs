//! Test support for the checkpoint-gated unit tests in [`crate::model`].
//!
//! `require_checkpoint_dir` lives in `lattice_model_io::test_support`, where the model crates call
//! it. This module's tests keep the source-policy guards over this crate's own tree.

#[cfg(test)]
mod tests;

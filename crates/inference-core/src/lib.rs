//! Shared base types for Lattice inference crates.
//!
//! This crate holds [`InferenceError`], the error type returned across the
//! inference, tokenizer, and model-loading boundaries.

mod error;

pub use error::InferenceError;

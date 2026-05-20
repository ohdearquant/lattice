//! Grammar-constrained generation engine (ADR-046).
//!
//! Implements XGrammar-style structured output for lattice-inference.
//!
//! # Architecture
//!
//! ```text
//! GrammarSpec  ──compile──►  CompiledGrammar  ──partition──►  VocabPartition
//!   (JsonSchema            (byte-level PDA      (bitmask table
//!    or GBNF)               rules + alts)        per state × token)
//!                                    │
//!                                    ▼
//!                          GrammarEngine::new()
//!                                    │
//!                              generate loop:
//!                      engine.mask_logits(state, logits)
//!                      token = sampler.sample(logits)
//!                      engine.advance(state, token_id)
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use lattice_inference::grammar::{GrammarEngine, GrammarSpec};
//! use lattice_inference::generate::GenerateConfig;
//!
//! let spec = GrammarSpec::json_schema_str(r#"{"type":"object","properties":{"name":{"type":"string"}}}"#)?;
//! let vocab_bytes: Vec<Vec<u8>> = tokenizer.vocab_bytes();
//! let engine = Arc::new(GrammarEngine::new(&spec, vocab_bytes)?);
//!
//! let config = GenerateConfig {
//!     grammar: Some(Arc::clone(&engine)),
//!     ..Default::default()
//! };
//! ```
//!
//! # Supported grammar formats
//!
//! - **JSON Schema** (primary): `GrammarSpec::JsonSchema(serde_json::Value)` —
//!   supports `object`, `array`, `string` (with `enum`), `number`, `integer`,
//!   `boolean`, `null`, `anyOf`/`oneOf`, `$ref` (local only).
//! - **GBNF** (secondary): `GrammarSpec::Gbnf(String)` — llama.cpp-compatible
//!   subset supporting literals, character classes, alternation, repetition.

pub mod engine;
pub mod gbnf;
pub mod json_schema;
pub mod pda;
pub mod spec;
pub mod vocab_partition;

// Re-exports for the primary public API.
pub use engine::{GrammarEngine, GrammarError};
pub use spec::GrammarSpec;

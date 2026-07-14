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
//! use lattice_inference::model::GenerateConfig;
//!
//! let spec = GrammarSpec::json_schema_str(r#"{"type":"object","properties":{"name":{"type":"string"}}}"#)?;
//! let vocab_bytes: Vec<Vec<u8>> = tokenizer.vocab_bytes(model.config().vocab_size)?;
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
//!
//! # Stability and known limitations
//!
//! Grammar-constrained generation is **BETA**. It is opt-in via
//! `GenerateConfig::grammar` and disabled by default.
//!
//! The mixed-required/optional JSON-Schema rejection and the no-rewind-backtracker
//! over-accept/over-reject class this BETA originally shipped with are fixed
//! ([#355], [#353] — closed via [#380], [#468], [#471], [#472]). Correctness work in
//! this area is ongoing: residual and newer findings in the JSON-Schema compiler and
//! the PDA/engine runtime are tracked live at [#310] and [#322] respectively — check
//! those issues for current status rather than assuming this comment stays up to date.
//!
//! [#353]: https://github.com/ohdearquant/lattice/issues/353
//! [#355]: https://github.com/ohdearquant/lattice/issues/355
//! [#380]: https://github.com/ohdearquant/lattice/pull/380
//! [#468]: https://github.com/ohdearquant/lattice/pull/468
//! [#471]: https://github.com/ohdearquant/lattice/pull/471
//! [#472]: https://github.com/ohdearquant/lattice/pull/472
//! [#310]: https://github.com/ohdearquant/lattice/issues/310
//! [#322]: https://github.com/ohdearquant/lattice/issues/322

pub mod engine;
pub mod gbnf;
pub mod json_schema;
pub mod pda;
pub mod spec;
pub mod vocab_partition;

// Re-exports for the primary public API.
pub use engine::{GrammarEngine, GrammarError};
pub use spec::GrammarSpec;

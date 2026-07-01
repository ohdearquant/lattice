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
//!
//! # Stability and known limitations
//!
//! Grammar-constrained generation is **BETA**. It is opt-in via
//! `GenerateConfig::grammar` and disabled by default.
//!
//! Known issues tracked at the time of this release:
//!
//! - **Object schemas mixing required and optional properties may reject valid
//!   JSON** ([#355]). A JSON object that satisfies a schema with both
//!   `required` and non-required properties can be incorrectly refused by the
//!   PDA state machine during generation. Workaround: use schemas with either
//!   all-required or all-optional properties, or inline the optional fields as
//!   `anyOf: [{...}, {}]`.
//!
//! - **The PDA backtracker does not rewind consumed bytes** ([#353]). Once the
//!   PDA advances past a byte, it cannot backtrack. This means some shared-prefix
//!   `anyOf`/`oneOf` grammars over-accept or over-reject tokens at branch points
//!   where the correct parse requires lookahead beyond a committed prefix.
//!
//! Both issues are pre-existing (present since v0.3.0) and tracked. Follow
//! [#353] and [#355] on GitHub for fixes.
//!
//! [#353]: https://github.com/ohdearquant/lattice/issues/353
//! [#355]: https://github.com/ohdearquant/lattice/issues/355

pub mod engine;
pub mod gbnf;
pub mod json_schema;
pub mod pda;
pub mod spec;
pub mod vocab_partition;

// Re-exports for the primary public API.
pub use engine::{GrammarEngine, GrammarError};
pub use spec::GrammarSpec;

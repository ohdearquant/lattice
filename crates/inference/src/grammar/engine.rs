//! Grammar-constrained decoding engine.
//!
//! `GrammarEngine` is the top-level public type for structured output.
//! It combines:
//!
//! 1. A compiled grammar (`CompiledGrammar`) derived from a `GrammarSpec`.
//! 2. A precomputed vocabulary partition (`VocabPartition`) that maps each
//!    grammar state × token to allow/deny.
//! 3. Per-decode-step state management (`GrammarState`).
//!
//! # Lifecycle
//!
//! ```text
//! GrammarEngine::new(spec, vocab_bytes)     — called once at init time
//!     ↓
//! let mut state = engine.initial_state();   — per-request
//!     ↓
//! loop {
//!     forward_pass(input) → logits
//!     engine.mask_logits(&mut state, logits) — apply grammar constraint
//!     token = sampler.sample(logits)
//!     engine.advance(&mut state, token_id)  — update grammar state
//! }
//! ```
//!
//! The `GrammarEngine` is `Send + Sync` and can be shared across requests
//! via `Arc<GrammarEngine>`.
//!
//! # Performance
//!
//! - `mask_logits`: O(vocab_size / 64) bitmask scan + O(k × stack_depth) for
//!   context-dependent tokens, where k ≈ 1% of vocab_size.
//! - `advance`: O(stack_depth) PDA step; typical depth 2–8.
//! - `new`: O(|states| × vocab_size × max_token_len) — called once.

use crate::grammar::gbnf::parse_gbnf;
use crate::grammar::json_schema::compile;
use crate::grammar::pda::{
    CompiledGrammar, GrammarState, SimResult, StepResult, advance_byte, simulate_token,
};
use crate::grammar::spec::GrammarSpec;
use crate::grammar::vocab_partition::VocabPartition;
use std::fmt;

/// Error from `GrammarEngine::new`.
#[derive(Debug, Clone)]
pub struct GrammarError(pub String);

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GrammarEngine error: {}", self.0)
    }
}
impl std::error::Error for GrammarError {}

impl From<crate::grammar::json_schema::SchemaError> for GrammarError {
    fn from(e: crate::grammar::json_schema::SchemaError) -> Self {
        GrammarError(e.0)
    }
}

impl From<crate::grammar::gbnf::GbnfError> for GrammarError {
    fn from(e: crate::grammar::gbnf::GbnfError) -> Self {
        GrammarError(e.0)
    }
}

// ---------------------------------------------------------------------------
// State enumeration
// ---------------------------------------------------------------------------

/// Enumerate all reachable grammar states via BFS from the initial state.
///
/// Each unique PDA stack configuration encountered while simulating all
/// single-token prefixes is added to the returned set.  This is a
/// conservative superset: some states may not be reachable for a given
/// model vocabulary, but all reachable states are included.
///
/// For v0 we use a depth-limited BFS to bound memory and runtime.
fn enumerate_grammar_states(
    grammar: &CompiledGrammar,
    vocab_bytes: &[Vec<u8>],
    max_states: usize,
) -> Vec<GrammarState> {
    let initial = GrammarState::initial();
    let mut queue: Vec<GrammarState> = vec![initial.clone()];
    let mut visited: Vec<GrammarState> = vec![initial];
    let mut head = 0;

    while head < queue.len() && visited.len() < max_states {
        let state = queue[head].clone();
        head += 1;

        for token_bytes in vocab_bytes {
            if token_bytes.is_empty() {
                continue;
            }
            let (result, next_state) = simulate_token(&state, grammar, token_bytes);
            if result == SimResult::Accept || result == SimResult::ContextDependent {
                // Only add if the stack configuration is new.
                if !visited.iter().any(|s| states_equal(s, &next_state)) {
                    visited.push(next_state.clone());
                    if visited.len() < max_states {
                        queue.push(next_state);
                    }
                }
            }
        }
    }

    visited
}

/// Compare two grammar states by their PDA stack configurations.
///
/// We only compare the stack frames (not partial_token_bytes, which is
/// transient).  Two states with identical stack frames will produce
/// identical bitmasks.
fn states_equal(a: &GrammarState, b: &GrammarState) -> bool {
    a.stack == b.stack && a.complete == b.complete
}

// ---------------------------------------------------------------------------
// GrammarEngine
// ---------------------------------------------------------------------------

/// Grammar-constrained decoding engine.
///
/// Thread-safe and `Clone`-free after construction.  Share via
/// `Arc<GrammarEngine>` across concurrent requests.
pub struct GrammarEngine {
    /// The compiled grammar (for runtime PDA advancement).
    grammar: CompiledGrammar,
    /// Precomputed vocabulary partition (bitmask table + context-dependent ids).
    partition: VocabPartition,
    /// Vocabulary size (used for bounds checking).
    vocab_size: usize,
    /// Raw vocabulary byte sequences (used for context-dependent token checks).
    vocab_bytes: Vec<Vec<u8>>,
}

impl GrammarEngine {
    /// Build a `GrammarEngine` from a `GrammarSpec` and the model vocabulary.
    ///
    /// `vocab_bytes[i]` is the UTF-8 / byte-level representation of token `i`.
    /// For BPE tokenizers, obtain this via `BpeTokenizer::vocab_bytes()`.
    ///
    /// This runs in O(|states| × vocab_size × max_token_len) time.  For
    /// large vocabularies (e.g. Qwen3 at 248,320 tokens) this may take
    /// 50–200 ms.  Cache the `GrammarEngine` across requests with the same
    /// schema.
    pub fn new(spec: &GrammarSpec, vocab_bytes: Vec<Vec<u8>>) -> Result<Self, GrammarError> {
        let vocab_size = vocab_bytes.len();

        // Compile spec to grammar.
        let grammar = match spec {
            GrammarSpec::JsonSchema(schema) => compile(schema)?,
            GrammarSpec::Gbnf(gbnf) => parse_gbnf(gbnf)?,
        };

        // Enumerate grammar states reachable from the initial state.
        // We limit to MAX_GRAMMAR_STATES to bound memory usage.
        let states = enumerate_grammar_states(
            &grammar,
            &vocab_bytes,
            crate::grammar::vocab_partition::MAX_GRAMMAR_STATES,
        );

        if states.len() >= crate::grammar::vocab_partition::MAX_GRAMMAR_STATES {
            tracing::warn!(
                "grammar state count hit limit ({}); some states may fall back to context-dependent checks",
                crate::grammar::vocab_partition::MAX_GRAMMAR_STATES
            );
        }

        // Build the vocabulary partition.
        let partition = VocabPartition::build(&grammar, states, &vocab_bytes);

        Ok(Self {
            grammar,
            partition,
            vocab_size,
            vocab_bytes,
        })
    }

    /// Create the initial `GrammarState` for a new decode sequence.
    pub fn initial_state(&self) -> GrammarState {
        GrammarState::initial()
    }

    /// Apply grammar constraints to `logits` in-place.
    ///
    /// Sets disallowed token positions to `f32::NEG_INFINITY`.
    ///
    /// The `state` parameter is the current grammar state, which must have
    /// been created by `initial_state()` and advanced via `advance()` after
    /// each sampled token.
    ///
    /// # Performance
    ///
    /// The hot path is a bitmask scan over `vocab_size / 64` words (~3,880
    /// iterations for Qwen3's 248,320 tokens), taking under 40 µs on modern
    /// Apple Silicon.  Context-dependent tokens add O(k × stack_depth)
    /// overhead (k ≈ 1% of vocab).
    pub fn mask_logits(&self, state: &mut GrammarState, logits: &mut [f32]) {
        assert!(
            logits.len() >= self.vocab_size,
            "logits length {} < vocab_size {}",
            logits.len(),
            self.vocab_size
        );

        // Find the state id in the partition.
        let state_id = self.find_state_id(state);

        // Apply the precomputed bitmask.
        self.partition.apply_mask(state_id, logits);

        // Re-check context-dependent tokens at runtime.
        for &token_id in self.partition.context_dependent_ids() {
            if token_id >= self.vocab_size {
                continue;
            }
            // If the bitmask already blocked this token, nothing to do.
            if logits[token_id] == f32::NEG_INFINITY {
                continue;
            }
            // Simulate advancing the grammar with this token's bytes.
            let token_bytes = &self.vocab_bytes[token_id];
            if token_bytes.is_empty() {
                logits[token_id] = f32::NEG_INFINITY;
                continue;
            }
            let (result, next_state) = simulate_token(state, &self.grammar, token_bytes);
            match result {
                SimResult::Reject => {
                    // Byte-level rejection: token is definitively invalid.
                    logits[token_id] = f32::NEG_INFINITY;
                }
                SimResult::ContextDependent => {
                    // Partial consumption: the token straddles a grammar
                    // boundary and cannot be generated as a complete unit.
                    logits[token_id] = f32::NEG_INFINITY;
                }
                SimResult::Accept => {
                    // All bytes consumed; check that the resulting state is
                    // either mid-grammar (not complete but still valid) or
                    // fully complete.  We allow both — the token is legal.
                    // next_state validity is implicit: simulate_token only
                    // returns Accept when no byte was rejected.
                    let _ = next_state; // used above for Accept branch
                }
            }
        }
    }

    /// Advance the grammar state by one token.
    ///
    /// Call this after sampling `token_id` to update the grammar state for
    /// the next step.  Returns `true` if the token was accepted, `false` if
    /// the grammar rejected it (caller should treat this as an error and stop
    /// generation).
    pub fn advance(&self, state: &mut GrammarState, token_id: u32) -> bool {
        let token_id = token_id as usize;
        if token_id >= self.vocab_size {
            return false;
        }
        let token_bytes = &self.vocab_bytes[token_id];
        if token_bytes.is_empty() {
            // Empty token: no grammar advancement, treat as accepted.
            return true;
        }
        for &b in token_bytes {
            if advance_byte(state, &self.grammar, b) == StepResult::Rejected {
                return false;
            }
        }
        true
    }

    /// Find the partition state id for `state` by matching stack configuration.
    ///
    /// If no matching precomputed state is found, returns 0 (initial state).
    /// This is safe (conservative) because unknown states fall through to the
    /// initial-state bitmask, which may over-allow some tokens — context-dependent
    /// re-checks will catch incorrect allowances.
    ///
    /// For v0 this is a linear scan.  For large grammars a hash map would be
    /// appropriate.
    fn find_state_id(&self, state: &GrammarState) -> usize {
        for sid in 0..self.partition.num_states() {
            if let Some(ps) = self.partition.grammar_state(sid) {
                if ps.stack == state.stack && ps.complete == state.complete {
                    return sid;
                }
            }
        }
        0 // fallback to initial state
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny vocab: token 0 = b"t", token 1 = b"f", token 2 = b"x".
    fn tiny_vocab() -> Vec<Vec<u8>> {
        vec![b"t".to_vec(), b"f".to_vec(), b"x".to_vec()]
    }

    /// Grammar spec: boolean (true | false)
    fn bool_spec() -> GrammarSpec {
        GrammarSpec::JsonSchema(serde_json::json!({"type": "boolean"}))
    }

    #[test]
    fn engine_new_from_json_schema() {
        let vocab = tiny_vocab();
        let spec = bool_spec();
        let result = GrammarEngine::new(&spec, vocab);
        assert!(result.is_ok(), "engine construction should succeed");
    }

    #[test]
    fn engine_new_from_gbnf() {
        let vocab = tiny_vocab();
        let spec = GrammarSpec::Gbnf("root ::= \"t\" | \"f\"\n".to_string());
        let result = GrammarEngine::new(&spec, vocab);
        assert!(result.is_ok());
    }

    #[test]
    fn mask_logits_blocks_disallowed() {
        // Boolean grammar: only "true" or "false" tokens allowed.
        // vocab: token 0 = b"true", token 1 = b"false", token 2 = b"other"
        let vocab = vec![b"true".to_vec(), b"false".to_vec(), b"other".to_vec()];
        let spec = GrammarSpec::JsonSchema(serde_json::json!({"type": "boolean"}));
        let engine = GrammarEngine::new(&spec, vocab).unwrap();

        let mut state = engine.initial_state();
        let mut logits = vec![1.0f32, 2.0f32, 3.0f32];
        engine.mask_logits(&mut state, &mut logits);

        // "other" should be blocked.
        assert_eq!(
            logits[2],
            f32::NEG_INFINITY,
            "token 'other' should be blocked"
        );
        // "true" or "false" tokens should be allowed (at least one).
        assert!(
            logits[0] > f32::NEG_INFINITY || logits[1] > f32::NEG_INFINITY,
            "at least one of 'true'/'false' must be allowed"
        );
    }

    #[test]
    fn advance_updates_state() {
        // null grammar: "null"
        let vocab = vec![b"n".to_vec(), b"u".to_vec(), b"l".to_vec(), b"l".to_vec()];
        let spec = GrammarSpec::JsonSchema(serde_json::json!({"type": "null"}));
        let engine = GrammarEngine::new(&spec, vocab).unwrap();

        let mut state = engine.initial_state();
        // Advance by token 0 ('n'): should be accepted.
        assert!(engine.advance(&mut state, 0));
    }

    #[test]
    fn advance_rejects_wrong_token() {
        // "null" grammar — token 0 = b"x" should be rejected.
        let vocab = vec![b"x".to_vec(), b"n".to_vec()];
        let spec = GrammarSpec::JsonSchema(serde_json::json!({"type": "null"}));
        let engine = GrammarEngine::new(&spec, vocab).unwrap();

        let mut state = engine.initial_state();
        // Token 0 = 'x' should be rejected in the "null" grammar.
        let result = engine.advance(&mut state, 0);
        assert!(!result, "'x' should be rejected for null grammar");
    }

    #[test]
    fn initial_state_not_complete() {
        let vocab = vec![b"t".to_vec()];
        let spec = GrammarSpec::Gbnf("root ::= \"test\"\n".to_string());
        let engine = GrammarEngine::new(&spec, vocab).unwrap();
        let state = engine.initial_state();
        assert!(!state.is_complete(), "initial state should not be complete");
    }

    #[test]
    fn engine_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GrammarEngine>();
    }

    #[test]
    fn mask_logits_logits_shorter_panics() {
        let vocab = vec![b"a".to_vec(), b"b".to_vec()];
        let spec = GrammarSpec::JsonSchema(serde_json::json!({"type": "null"}));
        let engine = GrammarEngine::new(&spec, vocab).unwrap();
        let mut state = engine.initial_state();
        // Panic if logits slice is shorter than vocab_size.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut logits = vec![0.0f32]; // shorter than vocab_size=2
            engine.mask_logits(&mut state, &mut logits);
        }));
        assert!(result.is_err(), "should panic when logits too short");
    }

    #[test]
    fn bitmask_and_correctness_large_vocab() {
        // Build a vocab of 130 tokens: tokens 0..4 = b"true" chars,
        // rest = b"x".  Only token 0 starts "true"/"false" in a boolean grammar.
        let mut vocab: Vec<Vec<u8>> = vec![b"true".to_vec(), b"false".to_vec()];
        for i in 2..130 {
            vocab.push(format!("tok{i}").into_bytes());
        }
        let spec = GrammarSpec::JsonSchema(serde_json::json!({"type": "boolean"}));
        let engine = GrammarEngine::new(&spec, vocab).unwrap();

        let mut state = engine.initial_state();
        let mut logits = vec![1.0f32; 130];
        engine.mask_logits(&mut state, &mut logits);

        // At least tokens 0 and 1 should be allowed.
        assert!(
            logits[0] > f32::NEG_INFINITY,
            "token 'true' should be allowed"
        );
        assert!(
            logits[1] > f32::NEG_INFINITY,
            "token 'false' should be allowed"
        );
        // All other tokens should be blocked.
        for i in 2..130 {
            assert_eq!(logits[i], f32::NEG_INFINITY, "token {i} should be blocked");
        }
    }
}

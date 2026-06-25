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
        match self.find_state_id(state) {
            Some(state_id) => {
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
                    let (result, _) = simulate_token(state, &self.grammar, token_bytes);
                    match result {
                        // Byte-level rejection, or partial consumption (the token
                        // straddles a grammar boundary and cannot be generated as a
                        // complete unit): block.
                        SimResult::Reject | SimResult::ContextDependent => {
                            logits[token_id] = f32::NEG_INFINITY;
                        }
                        // All bytes consumed with no rejection: the token is legal.
                        SimResult::Accept => {}
                    }
                }
            }
            None => {
                // The grammar exceeded `MAX_GRAMMAR_STATES`, so the BFS state
                // enumeration was truncated and this runtime state has no
                // precomputed mask. Falling back to the initial-state mask would
                // be unsound: tokens valid only at position 0 (e.g. an opening
                // quote) would be left allowed at a deep state, and single-byte
                // tokens are not in the context-dependent recheck set, so nothing
                // would correct them. Compute the exact mask by simulating every
                // token against the actual state instead. This is the universal
                // algorithm the precomputed table caches; it is sound and live
                // (a fully fail-closed "block everything" fallback would be sound
                // but would stall generation by leaving no legal token).
                self.mask_by_simulation(state, logits);
            }
        }
    }

    /// Compute the grammar mask for `state` directly by simulating every token.
    ///
    /// Used as the exact fallback for runtime states that are not present in the
    /// precomputed partition (grammars exceeding `MAX_GRAMMAR_STATES`). Blocks
    /// every token whose byte sequence does not fully advance the PDA from
    /// `state`. Cost: O(vocab_size × token_len); only invoked on the unknown-state
    /// path, never for grammars within the state cap.
    fn mask_by_simulation(&self, state: &GrammarState, logits: &mut [f32]) {
        for token_id in 0..self.vocab_size {
            if logits[token_id] == f32::NEG_INFINITY {
                continue;
            }
            let token_bytes = &self.vocab_bytes[token_id];
            if token_bytes.is_empty() {
                logits[token_id] = f32::NEG_INFINITY;
                continue;
            }
            let (result, _) = simulate_token(state, &self.grammar, token_bytes);
            if result != SimResult::Accept {
                logits[token_id] = f32::NEG_INFINITY;
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
    /// Returns `None` when no matching precomputed state exists. This happens
    /// only for grammars that exceed `MAX_GRAMMAR_STATES`, where the BFS state
    /// enumeration was truncated. Callers must handle `None` by computing the
    /// mask directly (see `mask_by_simulation`) rather than assuming the
    /// initial state, which would be unsound.
    ///
    /// For v0 this is a linear scan.  For large grammars a hash map would be
    /// appropriate.
    fn find_state_id(&self, state: &GrammarState) -> Option<usize> {
        for sid in 0..self.partition.num_states() {
            if let Some(ps) = self.partition.grammar_state(sid) {
                if ps.stack == state.stack && ps.complete == state.complete {
                    return Some(sid);
                }
            }
        }
        None
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

    /// Regression (issue #343): a left-recursive GBNF grammar (`root ::= root`)
    /// must not hang. Before the PDA depth cap, `enumerate_grammar_states` grew
    /// the PDA stack without bound inside a single `simulate_token` call. The
    /// depth cap turns it into a bounded dead grammar; construction returns.
    #[test]
    fn cyclic_gbnf_does_not_hang() {
        let vocab = tiny_vocab();
        let spec = GrammarSpec::Gbnf("root ::= root\n".to_string());
        let result = GrammarEngine::new(&spec, vocab);
        // Bounded construction (Ok with a dead grammar) is the contract — the
        // point is that it terminates rather than hanging or OOMing.
        assert!(result.is_ok(), "cyclic GBNF should construct, not hang");
    }

    /// Regression (issue #343): a JSON-Schema `$ref` cycle compiles to a cyclic
    /// grammar and previously hung at construction via the same PDA mechanism.
    #[test]
    fn cyclic_ref_schema_does_not_hang() {
        let vocab = tiny_vocab();
        let spec = GrammarSpec::JsonSchema(serde_json::json!({
            "$ref": "#/$defs/Node",
            "$defs": { "Node": { "$ref": "#/$defs/Node" } }
        }));
        let result = GrammarEngine::new(&spec, vocab);
        assert!(
            result.is_ok(),
            "cyclic $ref schema should construct, not hang"
        );
    }

    /// Regression (issue #343): an array schema with an absurd `maxItems`
    /// (near `u64::MAX`) previously overflowed the stack in `build_bounded_tail`.
    /// It must now be rejected with a typed error at the parse boundary.
    #[test]
    fn array_maxitems_overflow_rejected() {
        let vocab = tiny_vocab();
        let spec = GrammarSpec::JsonSchema(serde_json::json!({
            "type": "array",
            "items": { "type": "boolean" },
            "maxItems": 18446744073709551615u64
        }));
        let err = match GrammarEngine::new(&spec, vocab) {
            Ok(_) => panic!("absurd maxItems must be rejected, not overflow the stack"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("maxItems"),
            "error should name the offending field: {err}"
        );
    }

    /// Regression (issue #343): the `minItems` sibling — a near-`u64::MAX`
    /// `minItems` drives the required-item loop ~1.8e19 times. Reject it too.
    #[test]
    fn array_minitems_overflow_rejected() {
        let vocab = tiny_vocab();
        let spec = GrammarSpec::JsonSchema(serde_json::json!({
            "type": "array",
            "items": { "type": "boolean" },
            "minItems": 18446744073709551615u64
        }));
        let err = match GrammarEngine::new(&spec, vocab) {
            Ok(_) => panic!("absurd minItems must be rejected, not hang"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("minItems"),
            "error should name the offending field: {err}"
        );
    }

    /// Regression (issue #343, codex finding B): a long *acyclic* `$defs`
    /// reference chain (`N0→N1→…→Nk`, each a distinct unseen ref) recurses
    /// `compile_schema` once per link. Without the depth cap this overflowed the
    /// stack at compile time, independent of the PDA cycle fix. A 2000-link
    /// chain (well past MAX_SCHEMA_DEPTH=512) must reject cleanly.
    #[test]
    fn deep_ref_chain_rejected() {
        let n = 2000usize;
        let mut defs = serde_json::Map::new();
        for i in 0..n {
            let target = if i + 1 < n {
                serde_json::json!({ "$ref": format!("#/$defs/N{}", i + 1) })
            } else {
                serde_json::json!({ "type": "boolean" })
            };
            defs.insert(format!("N{i}"), target);
        }
        let schema = serde_json::json!({ "$ref": "#/$defs/N0", "$defs": defs });
        let spec = GrammarSpec::JsonSchema(schema);
        let err = match GrammarEngine::new(&spec, tiny_vocab()) {
            Ok(_) => panic!("deep $ref chain must be rejected, not overflow the stack"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("depth"), "error should mention depth: {err}");
    }

    /// A short acyclic `$ref` chain (well under the cap) must still compile.
    #[test]
    fn shallow_ref_chain_accepted() {
        let n = 64usize;
        let mut defs = serde_json::Map::new();
        for i in 0..n {
            let target = if i + 1 < n {
                serde_json::json!({ "$ref": format!("#/$defs/N{}", i + 1) })
            } else {
                serde_json::json!({ "type": "boolean" })
            };
            defs.insert(format!("N{i}"), target);
        }
        let schema = serde_json::json!({ "$ref": "#/$defs/N0", "$defs": defs });
        let result = GrammarEngine::new(&GrammarSpec::JsonSchema(schema), tiny_vocab());
        assert!(result.is_ok(), "a 64-link $ref chain should compile");
    }

    /// Boundary (issue #343, codex finding D): `maxItems` one past the cap is
    /// rejected; a small in-range `maxItems` still compiles.
    #[test]
    fn array_maxitems_boundary() {
        let over = GrammarSpec::JsonSchema(serde_json::json!({
            "type": "array", "items": { "type": "boolean" }, "maxItems": 4097
        }));
        assert!(
            GrammarEngine::new(&over, tiny_vocab()).is_err(),
            "maxItems just past the cap must be rejected"
        );
        let ok = GrammarSpec::JsonSchema(serde_json::json!({
            "type": "array", "items": { "type": "boolean" }, "maxItems": 8
        }));
        assert!(
            GrammarEngine::new(&ok, tiny_vocab()).is_ok(),
            "a small in-range maxItems must still compile"
        );
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
    fn unknown_state_fallback_is_sound() {
        // Soundness regression: when a grammar exceeds MAX_GRAMMAR_STATES the
        // BFS enumeration is truncated, so a deep runtime state is not in the
        // partition. `find_state_id` must NOT silently fall back to the
        // initial-state mask (which over-allows tokens that are valid only at
        // position 0). It must compute the exact mask by simulation.
        //
        // Grammar: a literal chain `"` + 300×`a` + `"` (>256 PDA states).
        // Vocab: token 0 = b"\"" (quote), token 1 = b"a".
        // At a deep mid-chain state the only valid next token is `a`; the
        // closing quote `"` is invalid there but IS valid at state 0.
        use crate::grammar::pda::{CompiledGrammar, Rule, Symbol};

        let mut chain = vec![Symbol::Terminal(b'"')];
        chain.extend(std::iter::repeat_n(Symbol::Terminal(b'a'), 300));
        chain.push(Symbol::Terminal(b'"'));
        let grammar = CompiledGrammar {
            rules: vec![Rule {
                name: "root".to_string(),
                alts: vec![chain],
            }],
        };
        let vocab = vec![b"\"".to_vec(), b"a".to_vec()];

        let spec = GrammarSpec::Gbnf("root ::= \"placeholder\"\n".to_string());
        // Build engine then swap in the hand-built grammar + partition so the
        // test does not depend on json_schema const-compilation internals.
        let mut engine = GrammarEngine::new(&spec, vocab.clone()).unwrap();
        let states = enumerate_grammar_states(
            &grammar,
            &vocab,
            crate::grammar::vocab_partition::MAX_GRAMMAR_STATES,
        );
        // Enumeration must have hit the cap, otherwise the test is vacuous.
        assert_eq!(
            states.len(),
            crate::grammar::vocab_partition::MAX_GRAMMAR_STATES,
            "grammar must exceed the state cap for this regression to bite"
        );
        engine.partition = VocabPartition::build(&grammar, states, &vocab);
        engine.grammar = grammar;

        // Drive into a deep state beyond the enumerated cap: quote + 270 a's.
        let mut state = engine.initial_state();
        assert!(engine.advance(&mut state, 0), "opening quote accepted");
        for _ in 0..270 {
            assert!(engine.advance(&mut state, 1), "mid-chain 'a' accepted");
        }

        // The deep state must NOT be in the enumerated partition — otherwise
        // this test exercises the precomputed fast path, not the simulation
        // fallback it is meant to cover.
        assert!(
            engine.find_state_id(&state).is_none(),
            "deep state must be unknown to the capped partition (else the \
             simulation fallback is not the path under test)"
        );

        let mut logits = vec![1.0f32, 1.0f32];
        engine.mask_logits(&mut state, &mut logits);

        assert!(
            logits[1] > f32::NEG_INFINITY,
            "valid mid-chain token 'a' must remain allowed at the deep state"
        );
        assert_eq!(
            logits[0],
            f32::NEG_INFINITY,
            "invalid closing-quote token must be blocked at the deep state \
             (state-0 fallback would wrongly allow it)"
        );
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

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
use crate::grammar::trie::ByteTrie;
use crate::grammar::vocab_partition::VocabPartition;
use std::fmt;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Profiling instrumentation (issue #734 diagnostics)
// ---------------------------------------------------------------------------
//
// Thread-local, opt-in counters used by `gramperf_profile` (crates/inference/
// src/bin/gramperf_profile.rs) to break decode-step cost down into the
// precomputed-bitmask path, the context-dependent recheck loop, and the
// `mask_by_simulation` fallback, without touching any call site outside this
// file. Disabled by default (`mask_profiling_enabled()` short-circuits on a
// single `Cell<bool>` read), so production decode pays no cost beyond that
// one branch when profiling is off.

thread_local! {
    static MASK_PROFILING_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static MASK_PROFILE: std::cell::RefCell<MaskProfile> =
        const { std::cell::RefCell::new(MaskProfile::new()) };
    static BUILD_PROFILE: std::cell::RefCell<BuildProfile> =
        const { std::cell::RefCell::new(BuildProfile::new()) };
}

/// Aggregated per-decode-step grammar-masking cost, accumulated across every
/// `mask_logits`/`advance` call since the last [`enable_mask_profiling`].
#[derive(Debug, Clone, Copy, Default)]
pub struct MaskProfile {
    /// Precomputed-bitmask path: `find_state_id` (hit) + `VocabPartition::apply_mask`.
    pub precomputed_calls: u64,
    pub precomputed_ns: u64,
    /// Context-dependent token recheck loop (runs only on the precomputed-hit path).
    pub context_recheck_calls: u64,
    pub context_recheck_ns: u64,
    /// `mask_by_simulation` fallback: `find_state_id` (miss) + full-vocab simulation.
    pub fallback_calls: u64,
    pub fallback_ns: u64,
    /// `GrammarEngine::advance` (PDA byte-stepping), tracked separately so it
    /// is not misattributed to either masking path in the report.
    pub advance_calls: u64,
    pub advance_ns: u64,
}

impl MaskProfile {
    const fn new() -> Self {
        Self {
            precomputed_calls: 0,
            precomputed_ns: 0,
            context_recheck_calls: 0,
            context_recheck_ns: 0,
            fallback_calls: 0,
            fallback_ns: 0,
            advance_calls: 0,
            advance_ns: 0,
        }
    }
}

/// One-time `GrammarEngine::new` cost breakdown, overwritten on every call
/// (unconditional — construction is a once-per-schema event, so the two
/// extra `Instant::now()` calls are immaterial next to the multi-second
/// build itself).
#[derive(Debug, Clone, Copy, Default)]
pub struct BuildProfile {
    pub bfs_ns: u64,
    pub partition_build_ns: u64,
    /// `enumerate_grammar_states`'s returned state count, which is *itself*
    /// capped at the `max_states` argument passed to it — see
    /// `probe_reachable_states` for the uncapped count.
    pub reachable_states: usize,
    pub capped_states: usize,
}

impl BuildProfile {
    const fn new() -> Self {
        Self {
            bfs_ns: 0,
            partition_build_ns: 0,
            reachable_states: 0,
            capped_states: 0,
        }
    }
}

/// Enable per-decode-step mask profiling on the current thread and reset the
/// accumulator. Call [`take_mask_profile`] after the measured run(s).
pub fn enable_mask_profiling() {
    MASK_PROFILING_ENABLED.with(|e| e.set(true));
    MASK_PROFILE.with(|p| *p.borrow_mut() = MaskProfile::new());
}

/// Disable mask profiling and return the accumulated [`MaskProfile`].
pub fn take_mask_profile() -> MaskProfile {
    MASK_PROFILING_ENABLED.with(|e| e.set(false));
    MASK_PROFILE.with(|p| *p.borrow())
}

fn mask_profiling_enabled() -> bool {
    MASK_PROFILING_ENABLED.with(std::cell::Cell::get)
}

/// Return the [`BuildProfile`] captured by the most recent `GrammarEngine::new`
/// call on this thread.
pub fn last_build_profile() -> BuildProfile {
    BUILD_PROFILE.with(|p| *p.borrow())
}

/// Profiling-only probe: run `enumerate_grammar_states` with an arbitrary
/// `max_states` ceiling (independent of `MAX_GRAMMAR_STATES`) and return how
/// many states it found. Because `enumerate_grammar_states` stops growing
/// `visited` once it reaches `max_states`, `GrammarEngine::new`'s own BFS
/// output is *already* capped when a grammar exceeds the production limit —
/// this is the only way to see the true reachable-state count past the cap.
/// Never called from production code paths; used solely by
/// `gramperf_profile` (issue #734).
pub fn probe_reachable_states(
    spec: &GrammarSpec,
    vocab_bytes: &[Vec<u8>],
    max_states: usize,
) -> Result<usize, GrammarError> {
    let grammar = match spec {
        GrammarSpec::JsonSchema(schema) => compile(schema)?,
        GrammarSpec::Gbnf(gbnf) => parse_gbnf(gbnf)?,
    };
    Ok(enumerate_grammar_states(&grammar, vocab_bytes, max_states).len())
}

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
    /// Set when state enumeration in `new` hit `MAX_GRAMMAR_STATES` before
    /// exhausting the reachable state graph: some runtime states then have
    /// no precomputed mask and fall back to `mask_by_trie`. See
    /// `exceeds_state_budget`.
    state_limit_exceeded: bool,
    /// Byte trie over `vocab_bytes`, used by `mask_by_trie` for states with
    /// no precomputed mask. Built lazily on first use (`OnceLock`) so
    /// grammars that never exceed `MAX_GRAMMAR_STATES` — the common case —
    /// never pay the build cost.
    trie: OnceLock<ByteTrie>,
    /// Nanoseconds spent building `trie`, captured the one time
    /// `trie.get_or_init` actually runs the builder. Zero until the trie
    /// has been built at least once.
    trie_build_ns: AtomicU64,
}

impl GrammarEngine {
    /// Build a `GrammarEngine` from a `GrammarSpec` and the model vocabulary.
    ///
    /// `vocab_bytes[i]` is the UTF-8 / byte-level representation of token `i`.
    /// For BPE tokenizers, obtain this via `BpeTokenizer::vocab_bytes(model_vocab_size)`.
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
        let bfs_t0 = std::time::Instant::now();
        let states = enumerate_grammar_states(
            &grammar,
            &vocab_bytes,
            crate::grammar::vocab_partition::MAX_GRAMMAR_STATES,
        );
        let bfs_ns = bfs_t0.elapsed().as_nanos() as u64;

        let state_limit_exceeded =
            states.len() >= crate::grammar::vocab_partition::MAX_GRAMMAR_STATES;
        if state_limit_exceeded {
            tracing::warn!(
                "grammar state count hit limit ({}); some states may fall back to context-dependent checks",
                crate::grammar::vocab_partition::MAX_GRAMMAR_STATES
            );
        }
        let reachable_states = states.len();

        // Build the vocabulary partition.
        let partition_t0 = std::time::Instant::now();
        let partition = VocabPartition::build(&grammar, states, &vocab_bytes);
        let partition_build_ns = partition_t0.elapsed().as_nanos() as u64;

        BUILD_PROFILE.with(|p| {
            *p.borrow_mut() = BuildProfile {
                bfs_ns,
                partition_build_ns,
                reachable_states,
                capped_states: crate::grammar::vocab_partition::MAX_GRAMMAR_STATES,
            }
        });

        Ok(Self {
            grammar,
            partition,
            vocab_size,
            vocab_bytes,
            state_limit_exceeded,
            trie: OnceLock::new(),
            trie_build_ns: AtomicU64::new(0),
        })
    }

    /// Nanoseconds spent building the byte trie used by `mask_by_trie`, or 0
    /// if the trie has not been built yet (grammar never hit an over-cap
    /// state, or none has been masked yet). Diagnostic accessor for
    /// self-measurement harnesses.
    pub fn trie_build_ns(&self) -> u64 {
        self.trie_build_ns.load(Ordering::Relaxed)
    }

    /// True when state enumeration hit `MAX_GRAMMAR_STATES` (256) before
    /// covering the whole reachable state graph. Once true, some runtime
    /// states have no precomputed mask and every step in that state pays
    /// `mask_by_simulation`'s full-vocab scan instead of a bitmask lookup --
    /// an unbounded-latency mode unacceptable for an unrestricted serve API.
    /// Callers that need a bounded-latency guarantee (e.g. HTTP strict
    /// structured-output admission) should reject the schema instead of
    /// using an engine that reports `true` here.
    pub fn exceeds_state_budget(&self) -> bool {
        self.state_limit_exceeded
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

        let profiling = mask_profiling_enabled();
        let find_t0 = profiling.then(std::time::Instant::now);
        let found = self.find_state_id(state);
        let find_ns = find_t0.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);

        // Find the state id in the partition.
        match found {
            Some(state_id) => {
                // Apply the precomputed bitmask.
                let t0 = profiling.then(std::time::Instant::now);
                self.partition.apply_mask(state_id, logits);
                if let Some(t0) = t0 {
                    let ns = find_ns + t0.elapsed().as_nanos() as u64;
                    MASK_PROFILE.with(|p| {
                        let mut p = p.borrow_mut();
                        p.precomputed_calls += 1;
                        p.precomputed_ns += ns;
                    });
                }

                // Re-check context-dependent tokens at runtime.
                let t1 = profiling.then(std::time::Instant::now);
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
                if let Some(t1) = t1 {
                    let ns = t1.elapsed().as_nanos() as u64;
                    MASK_PROFILE.with(|p| {
                        let mut p = p.borrow_mut();
                        p.context_recheck_calls += 1;
                        p.context_recheck_ns += ns;
                    });
                }
            }
            None => {
                // The grammar exceeded `MAX_GRAMMAR_STATES`, so the BFS state
                // enumeration was truncated and this runtime state has no
                // precomputed mask. Falling back to the initial-state mask would
                // be unsound: tokens valid only at position 0 (e.g. an opening
                // quote) would be left allowed at a deep state, and single-byte
                // tokens are not in the context-dependent recheck set, so nothing
                // would correct them. Compute the exact mask via the byte trie
                // instead — same contract as `mask_by_simulation` (only cheaper:
                // a rejected byte prunes every token sharing that prefix in one
                // step instead of re-walking each one independently).
                // `mask_by_simulation` stays available as the oracle for the
                // differential tests below and as a manual fallback.
                let t0 = profiling.then(std::time::Instant::now);
                self.mask_by_trie(state, logits);
                if let Some(t0) = t0 {
                    let ns = find_ns + t0.elapsed().as_nanos() as u64;
                    MASK_PROFILE.with(|p| {
                        let mut p = p.borrow_mut();
                        p.fallback_calls += 1;
                        p.fallback_ns += ns;
                    });
                }
            }
        }
    }

    /// Compute the grammar mask for `state` directly by simulating every token.
    ///
    /// This was the over-cap fallback in `mask_logits` before the byte-trie
    /// path (`mask_by_trie`) replaced it on the hot path (issue #734: this
    /// full-vocab independent simulation is O(vocab_size × token_len) and
    /// dominates decode-step wall time for schemas that exceed
    /// `MAX_GRAMMAR_STATES`). It defines the mask contract `mask_by_trie`
    /// must reproduce exactly and stays public as the oracle for the
    /// differential tests below and as a slow-path manual escape hatch.
    pub fn mask_by_simulation(&self, state: &GrammarState, logits: &mut [f32]) {
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

    /// Compute the grammar mask for `state` via the byte trie.
    ///
    /// Same contract as `mask_by_simulation` (see [`crate::grammar::trie`]),
    /// used as the over-cap fallback in `mask_logits` in place of the
    /// full-vocab independent simulation. Builds the trie on first use and
    /// reuses it for every subsequent call against this engine.
    fn mask_by_trie(&self, state: &GrammarState, logits: &mut [f32]) {
        let trie = self.trie.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let built = ByteTrie::build(&self.vocab_bytes);
            self.trie_build_ns
                .store(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            built
        });
        trie.mask(state, &self.grammar, self.vocab_size, logits);
    }

    /// Advance the grammar state by one token.
    ///
    /// Call this after sampling `token_id` to update the grammar state for
    /// the next step.  Returns `true` if the token was accepted, `false` if
    /// the grammar rejected it (caller should treat this as an error and stop
    /// generation).
    pub fn advance(&self, state: &mut GrammarState, token_id: u32) -> bool {
        let profiling = mask_profiling_enabled();
        let t0 = profiling.then(std::time::Instant::now);
        let result = self.advance_inner(state, token_id);
        if let Some(t0) = t0 {
            let ns = t0.elapsed().as_nanos() as u64;
            MASK_PROFILE.with(|p| {
                let mut p = p.borrow_mut();
                p.advance_calls += 1;
                p.advance_ns += ns;
            });
        }
        result
    }

    fn advance_inner(&self, state: &mut GrammarState, token_id: u32) -> bool {
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

    /// Return whether `state` is accepting and no non-empty vocabulary token
    /// can legally advance it.
    pub(crate) fn is_complete_without_continuation(&self, state: &GrammarState) -> bool {
        if !state.is_complete() {
            return false;
        }

        let has_continuation = match self.find_state_id(state) {
            Some(state_id) => self.partition.any_allowed_token(state_id, |token_id| {
                simulate_token(state, &self.grammar, &self.vocab_bytes[token_id]).0
                    == SimResult::Accept
            }),
            None => self.vocab_bytes.iter().any(|token_bytes| {
                !token_bytes.is_empty()
                    && simulate_token(state, &self.grammar, token_bytes).0 == SimResult::Accept
            }),
        };
        !has_continuation
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
            if let Some(ps) = self.partition.grammar_state(sid)
                && ps.stack == state.stack
                && ps.complete == state.complete
            {
                return Some(sid);
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

    /// Regression (issue #343, finding B): a long *acyclic* `$defs`
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

    /// Boundary (issue #343, finding D): `maxItems` one past the cap is
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
    fn complete_state_without_continuation_is_terminal() {
        let vocab = vec![b"a".to_vec()];
        let spec = GrammarSpec::Gbnf("root ::= \"a\"\n".to_string());
        let engine = GrammarEngine::new(&spec, vocab).unwrap();
        let mut state = engine.initial_state();

        assert!(engine.advance(&mut state, 0));
        assert!(engine.is_complete_without_continuation(&state));
    }

    #[test]
    fn complete_state_with_continuation_is_not_terminal() {
        let vocab = vec![b"a".to_vec()];
        let spec = GrammarSpec::Gbnf("root ::= \"a\"+\n".to_string());
        let engine = GrammarEngine::new(&spec, vocab).unwrap();
        let mut state = engine.initial_state();

        assert!(engine.advance(&mut state, 0));
        assert!(state.is_complete());
        assert!(!engine.is_complete_without_continuation(&state));
        assert!(engine.advance(&mut state, 0));
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

    // -----------------------------------------------------------------
    // Trie-mask differential tests (issue #734 fix)
    //
    // `mask_by_trie` must reproduce `mask_by_simulation`'s mask bit for
    // bit. These tests harvest a corpus of real grammar states from a
    // generation trajectory against the exact schema shape
    // `gramperf_profile` profiles (crates/inference/src/bin/
    // gramperf_profile.rs), then compare the two algorithms directly.
    // -----------------------------------------------------------------

    /// The #734-shape schema from `gramperf_profile`: 4 nested object
    /// levels, 3 array fields, 6 six-member string enums. Reused verbatim
    /// (not approximated) so this corpus exercises the same structural
    /// over-cap behavior the profiling report measured.
    fn trie_diff_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "object",
                                    "properties": {
                                        "level4": {
                                            "type": "object",
                                            "properties": {
                                                "status": {"type": "string", "enum": ["active", "inactive", "pending", "archived", "deleted", "draft"]},
                                                "value": {"type": "integer"}
                                            },
                                            "required": ["status", "value"]
                                        }
                                    },
                                    "required": ["level4"]
                                },
                                "category": {"type": "string", "enum": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]}
                            },
                            "required": ["level3", "category"]
                        },
                        "tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["level2", "tags"]
                },
                "items": {"type": "array", "items": {"type": "integer"}},
                "flags": {"type": "array", "items": {"type": "boolean"}},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent", "critical", "none"]},
                "region": {"type": "string", "enum": ["us", "eu", "apac", "latam", "mea", "other"]},
                "mode": {"type": "string", "enum": ["sync", "async", "batch", "stream", "manual", "auto"]},
                "role": {"type": "string", "enum": ["admin", "user", "guest", "owner", "viewer", "editor"]}
            },
            "required": ["level1", "items", "flags", "priority", "region", "mode", "role"]
        })
    }

    /// Synthetic vocabulary sized for fast unit tests (not the real 248K
    /// tokenizer vocab): every single byte (so the trie branches fully at
    /// every depth, matching the real vocab's near-total root fan-out)
    /// plus multi-byte literal fragments drawn from the schema's property
    /// names, enum members, and JSON keywords, so the trie also exercises
    /// prefix-shared, straight-line multi-byte runs.
    fn trie_diff_vocab() -> Vec<Vec<u8>> {
        let mut vocab: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        let fragments: &[&str] = &[
            "\"level1\"",
            "\"level2\"",
            "\"level3\"",
            "\"level4\"",
            "\"status\"",
            "\"value\"",
            "\"category\"",
            "\"tags\"",
            "\"items\"",
            "\"flags\"",
            "\"priority\"",
            "\"region\"",
            "\"mode\"",
            "\"role\"",
            "\"active\"",
            "\"inactive\"",
            "\"pending\"",
            "\"archived\"",
            "\"deleted\"",
            "\"draft\"",
            "\"alpha\"",
            "\"beta\"",
            "\"gamma\"",
            "\"delta\"",
            "\"epsilon\"",
            "\"zeta\"",
            "\"low\"",
            "\"medium\"",
            "\"high\"",
            "\"urgent\"",
            "\"critical\"",
            "\"none\"",
            "\"us\"",
            "\"eu\"",
            "\"apac\"",
            "\"latam\"",
            "\"mea\"",
            "\"other\"",
            "\"sync\"",
            "\"async\"",
            "\"batch\"",
            "\"stream\"",
            "\"manual\"",
            "\"auto\"",
            "\"admin\"",
            "\"user\"",
            "\"guest\"",
            "\"owner\"",
            "\"viewer\"",
            "\"editor\"",
            "true",
            "false",
            "null",
        ];
        for f in fragments {
            vocab.push(f.as_bytes().to_vec());
        }
        vocab
    }

    /// A minified, schema-valid JSON instance. `serde_json::Value::Object`
    /// is a `BTreeMap` in this workspace (no `preserve_order` feature), so
    /// `compile_object`'s property order — and therefore the required-key
    /// order the PDA enforces — is alphabetical-by-key, not `properties`
    /// declaration order or `required`-array order; this instance's key
    /// order at every nesting level is alphabetical to match (top:
    /// flags, items, level1, mode, priority, region, role; level2:
    /// category, level3 — note `category` sorts before `level3`). Walking
    /// this byte-by-byte through the PDA is the "generation trajectory"
    /// the corpus is harvested from: its byte positions span the initial
    /// state (byte 0), deep nesting (into level1/level2/level3/level4),
    /// inside-string (mid `"active`), inside-enum-literal (mid `"alpha`,
    /// `"low`, `"sync`, ...), post-number (right after `42`, `1`, `2`,
    /// `3`), and near-complete (the last few closing-brace bytes) — the
    /// edge-state categories the correctness bar requires, without needing
    /// a separate hand-built state per category.
    const TRIE_DIFF_INSTANCE: &str = r#"{"flags":[true,false],"items":[1,2,3],"level1":{"level2":{"category":"alpha","level3":{"level4":{"status":"active","value":42}}},"tags":["x","y"]},"mode":"sync","priority":"low","region":"us","role":"admin"}"#;

    /// Harvest a corpus of distinct grammar states by advancing `engine`
    /// byte-by-byte through [`TRIE_DIFF_INSTANCE`], deduping by PDA
    /// identity (`states_equal`, same identity `find_state_id` uses).
    /// Panics with the failing byte position if the instance is rejected —
    /// a silent partial trajectory would silently shrink the corpus
    /// instead of failing the test that depends on its size.
    fn harvest_trajectory_states(engine: &GrammarEngine) -> Vec<GrammarState> {
        let mut state = engine.initial_state();
        let mut corpus = vec![state.clone()];
        for (i, &b) in TRIE_DIFF_INSTANCE.as_bytes().iter().enumerate() {
            let step = advance_byte(&mut state, &engine.grammar, b);
            assert_eq!(
                step,
                StepResult::Accepted,
                "trajectory instance rejected at byte {i} ({:?}); fixture is out of \
                 sync with the schema",
                b as char
            );
            if !corpus.iter().any(|s| states_equal(s, &state)) {
                corpus.push(state.clone());
            }
        }
        assert!(
            state.is_complete(),
            "trajectory instance must fully complete the grammar"
        );
        corpus
    }

    #[test]
    fn trie_mask_byte_identical_to_oracle_over_corpus() {
        let vocab = trie_diff_vocab();
        let spec = GrammarSpec::JsonSchema(trie_diff_schema());
        let engine = GrammarEngine::new(&spec, vocab.clone()).unwrap();
        assert!(
            engine.exceeds_state_budget(),
            "fixture must exceed the state cap for this differential to exercise \
             the fallback path (the point of this test)"
        );

        let corpus = harvest_trajectory_states(&engine);
        assert!(
            corpus.len() >= 20,
            "need at least 20 distinct harvested states, got {}",
            corpus.len()
        );

        let mut over_cap_states = 0usize;
        for (i, state) in corpus.iter().enumerate() {
            let mut oracle_logits = vec![0.0f32; vocab.len()];
            let mut trie_logits = vec![0.0f32; vocab.len()];
            engine.mask_by_simulation(state, &mut oracle_logits);
            engine.mask_by_trie(state, &mut trie_logits);
            if engine.find_state_id(state).is_none() {
                over_cap_states += 1;
            }
            for tok in 0..vocab.len() {
                assert_eq!(
                    trie_logits[tok],
                    oracle_logits[tok],
                    "state #{i}: token {tok} ({:?}) mismatch — trie={} oracle={}",
                    String::from_utf8_lossy(&vocab[tok]),
                    trie_logits[tok],
                    oracle_logits[tok]
                );
            }
        }
        assert!(
            over_cap_states >= 15,
            "corpus should mostly cover the over-cap fallback path (the profiling \
             report measured a 98% fallback rate on this schema shape); got \
             {over_cap_states}/{}",
            corpus.len()
        );
    }

    #[test]
    fn trie_mask_never_over_accepts_vs_oracle() {
        // P0, reported separately from the byte-identical test above: this
        // isolates the over-accept direction (trie allows a token the
        // oracle rejects) so a prune-logic regression that only
        // over-accepts is caught and reported distinctly from an
        // under-accept regression.
        let vocab = trie_diff_vocab();
        let spec = GrammarSpec::JsonSchema(trie_diff_schema());
        let engine = GrammarEngine::new(&spec, vocab.clone()).unwrap();
        let corpus = harvest_trajectory_states(&engine);

        let mut over_accepts: Vec<(usize, Vec<u8>)> = Vec::new();
        for state in &corpus {
            let mut oracle_logits = vec![0.0f32; vocab.len()];
            let mut trie_logits = vec![0.0f32; vocab.len()];
            engine.mask_by_simulation(state, &mut oracle_logits);
            engine.mask_by_trie(state, &mut trie_logits);
            for tok in 0..vocab.len() {
                let oracle_blocked = oracle_logits[tok] == f32::NEG_INFINITY;
                let trie_allowed = trie_logits[tok] != f32::NEG_INFINITY;
                if oracle_blocked && trie_allowed {
                    over_accepts.push((tok, vocab[tok].clone()));
                }
            }
        }
        assert!(
            over_accepts.is_empty(),
            "trie over-accepted {} token(s) the oracle rejects (P0 soundness \
             violation): {:?}",
            over_accepts.len(),
            over_accepts
                .iter()
                .take(5)
                .map(|(id, bytes)| (id, String::from_utf8_lossy(bytes).to_string()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn trie_routes_over_cap_states_through_mask_logits() {
        // End-to-end wiring check: the public `mask_logits` entry point
        // (not the private helpers) must actually route over-cap states
        // through the trie and produce the same result as calling the
        // oracle directly — guards against the routing edit in
        // `mask_logits` silently going stale if the fallback call site
        // changes again.
        let vocab = trie_diff_vocab();
        let spec = GrammarSpec::JsonSchema(trie_diff_schema());
        let engine = GrammarEngine::new(&spec, vocab.clone()).unwrap();
        let corpus = harvest_trajectory_states(&engine);
        let deep_state = corpus
            .iter()
            .find(|s| engine.find_state_id(s).is_none())
            .expect("corpus must contain at least one over-cap state");

        let mut via_mask_logits = vec![0.0f32; vocab.len()];
        let mut oracle_logits = vec![0.0f32; vocab.len()];
        let mut state_for_public_call = deep_state.clone();
        engine.mask_logits(&mut state_for_public_call, &mut via_mask_logits);
        engine.mask_by_simulation(deep_state, &mut oracle_logits);

        assert_eq!(
            via_mask_logits, oracle_logits,
            "mask_logits on an over-cap state must match the oracle mask exactly"
        );
    }

    /// Same differential as `trie_mask_byte_identical_to_oracle_over_corpus` /
    /// `trie_mask_never_over_accepts_vs_oracle`, but over the REAL model
    /// tokenizer vocabulary instead of the synthetic `trie_diff_vocab`. The
    /// synthetic vocab (256 single bytes + ~50 fragments) is a stand-in that
    /// exercises trie branching and prefix sharing but is roughly three
    /// orders of magnitude smaller than a real ~248K-token vocab, so it
    /// cannot rule out a size- or content-dependent trie bug that only
    /// surfaces at production scale. `#[ignore]`d because it loads a real
    /// tokenizer and simulates the oracle over the full vocab (slow, and
    /// requires a model checkout on disk); run explicitly with a `real_vocab`
    /// test filter.
    #[test]
    #[ignore = "loads a real tokenizer + full vocab oracle simulation; run with `real_vocab` filter"]
    fn trie_mask_byte_identical_to_oracle_real_vocab() {
        let home = std::env::var("HOME").expect("HOME must be set");
        let tokenizer_dir_str = std::env::var("LATTICE_TOKENIZER_DIR")
            .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
        let tokenizer_dir = std::path::Path::new(&tokenizer_dir_str);
        let config_path = tokenizer_dir.join("config.json");
        let tokenizer_path = tokenizer_dir.join("tokenizer.json");
        if !config_path.exists() || !tokenizer_path.exists() {
            panic!(
                "real-vocab differential test skipped: no model checkout at \
                 {tokenizer_dir_str} (expected config.json + tokenizer.json); \
                 set LATTICE_TOKENIZER_DIR to point at one"
            );
        }

        let cfg = crate::model::qwen35_config::Qwen35Config::from_model_dir(tokenizer_dir)
            .expect("config.json load");
        let tokenizer = crate::tokenizer::BpeTokenizer::from_tokenizer_json(&tokenizer_path)
            .expect("tokenizer.json load");
        let vocab = tokenizer
            .vocab_bytes(cfg.vocab_size)
            .expect("vocab_bytes over real tokenizer");
        assert!(
            vocab.len() > 100_000,
            "expected a production-scale vocab (~248K tokens for Qwen3.5), got {}",
            vocab.len()
        );

        let spec = GrammarSpec::JsonSchema(trie_diff_schema());
        let engine = GrammarEngine::new(&spec, vocab.clone()).unwrap();
        assert!(
            engine.exceeds_state_budget(),
            "fixture must exceed the state cap for this differential to exercise \
             the fallback path (the point of this test) — got a real vocab of {} \
             tokens",
            vocab.len()
        );

        let corpus = harvest_trajectory_states(&engine);
        assert!(
            corpus.len() >= 20,
            "need at least 20 distinct harvested states, got {}",
            corpus.len()
        );

        let mut over_cap_states = 0usize;
        let mut mismatches: Vec<(usize, usize)> = Vec::new();
        let mut over_accepts: Vec<(usize, usize)> = Vec::new();
        for (i, state) in corpus.iter().enumerate() {
            let mut oracle_logits = vec![0.0f32; vocab.len()];
            let mut trie_logits = vec![0.0f32; vocab.len()];
            engine.mask_by_simulation(state, &mut oracle_logits);
            engine.mask_by_trie(state, &mut trie_logits);
            if engine.find_state_id(state).is_none() {
                over_cap_states += 1;
            }
            for tok in 0..vocab.len() {
                if trie_logits[tok] != oracle_logits[tok] {
                    mismatches.push((i, tok));
                }
                let oracle_blocked = oracle_logits[tok] == f32::NEG_INFINITY;
                let trie_allowed = trie_logits[tok] != f32::NEG_INFINITY;
                if oracle_blocked && trie_allowed {
                    over_accepts.push((i, tok));
                }
            }
        }

        assert!(
            over_cap_states >= 10,
            "corpus should mostly cover the over-cap fallback path; got \
             {over_cap_states}/{}",
            corpus.len()
        );

        // Reported separately from the byte-identical check below so a
        // prune-logic regression that only over-accepts (P0 soundness) is
        // distinguishable from a broader mismatch (which could also include
        // under-accepts).
        assert!(
            over_accepts.is_empty(),
            "trie over-accepted {} token(s) the oracle rejects on the real \
             vocab (P0 soundness violation): {:?}",
            over_accepts.len(),
            over_accepts
                .iter()
                .take(5)
                .map(|(state_idx, tok)| (
                    state_idx,
                    tok,
                    String::from_utf8_lossy(&vocab[*tok]).to_string()
                ))
                .collect::<Vec<_>>()
        );

        assert!(
            mismatches.is_empty(),
            "trie mask diverged from oracle mask on {} (state, token) pair(s) \
             over the real vocab: {:?}",
            mismatches.len(),
            mismatches
                .iter()
                .take(5)
                .map(|(state_idx, tok)| (
                    state_idx,
                    tok,
                    String::from_utf8_lossy(&vocab[*tok]).to_string()
                ))
                .collect::<Vec<_>>()
        );

        eprintln!(
            "real_vocab differential: vocab_size={} corpus_states={} \
             over_cap_states={over_cap_states} over_accepts=0 mismatches=0",
            vocab.len(),
            corpus.len(),
        );
    }
}

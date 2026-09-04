//! Vocabulary partitioning for XGrammar-style constrained decoding.
//!
//! # Background (XGrammar, MLSys 2025)
//!
//! For each grammar state, tokens are classified as:
//!
//! - **Context-independent**: whether the token is legal depends only on the
//!   current grammar state, not on partially-accumulated bytes within the
//!   token.  These are precomputed into a bitmask table indexed by
//!   `(grammar_state, token_id)` — one bit per token.
//!
//! - **Context-dependent**: legality requires inspecting the runtime PDA
//!   stack — typically tokens that straddle a grammar boundary mid-byte
//!   sequence.  These are identified during bitmask precomputation and
//!   checked at decode time.
//!
//! # Bitmask layout
//!
//! ```text
//! masks: Vec<u64>
//! masks[state * mask_stride + word] encodes 64 tokens:
//!   bit j of masks[state * mask_stride + word] = token (word * 64 + j) is allowed
//! mask_stride = ceil(vocab_size / 64)
//! ```
//!
//! # Usage
//!
//! 1. `VocabPartition::build(grammar, grammar_states, vocab_bytes)` — called
//!    once at `GrammarEngine::new` time.  It stores the enumerated grammar
//!    states and one uninitialised row cell per state; every per-state row
//!    (mask + context-dependent list) is built on first visit, so
//!    construction is O(|states|) and no (state, token) pair is simulated
//!    up front.
//! 2. `VocabPartition::apply_mask(state_id, logits)` — called per decode
//!    step; builds the state's row on first use.
//! 3. `VocabPartition::context_dependent_ids_for_state(state_id)` — returns
//!    the token ids that need runtime PDA inspection in the current state,
//!    from that state's own list.  An out-of-range `state_id` returns an
//!    empty list (fail-closed): the caller has no precomputed mask for it
//!    and must never fall back to another state's or the global union's
//!    candidates.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use crate::grammar::pda::{CompiledGrammar, GrammarState, SimResult, simulate_token};

/// Maximum number of grammar states for v0.
/// A grammar with more states triggers a warning at build time.
pub const MAX_GRAMMAR_STATES: usize = 256;

/// Precomputed vocabulary partition for a grammar.
///
/// `state_count` is the number of distinct grammar states tracked.  For
/// most JSON schemas this is the number of unique PDA stack configurations
/// reachable from the initial state — typically under 100.
///
/// Per-state rows are built lazily on first visit (`build_state_row`), so
/// only states a decode run actually reaches cost anything.
pub struct VocabPartition {
    /// Shared grammar, handed to the lazy row builder at first-visit time.
    /// Owned jointly with `GrammarEngine` (see `build`) so rows can be built
    /// from `&self` without borrowing the engine.
    grammar: Arc<CompiledGrammar>,
    /// Shared vocabulary, one byte sequence per token. Owned jointly with
    /// `GrammarEngine` for the same reason. The vocabulary is large (a
    /// 248,320-token BPE vocab is hundreds of MiB of byte sequences) and is
    /// never duplicated here.
    vocab: Arc<Vec<Vec<u8>>>,
    mask_stride: usize,
    vocab_size: usize,
    /// One cell per effective (enumerated) grammar state. The cell holds
    /// that state's precomputed row once any decode step has visited it.
    /// States that are never visited never pay for their row.
    rows: Vec<OnceLock<StateRow>>,
    /// Grammar states indexed by `state_id`. `num_states()` and
    /// `grammar_state()` read only this list and must not force a row
    /// build: the engine's per-step state lookup iterates over all of them
    /// on every decode step, so touching a row from there would rebuild the
    /// eager all-states cost on the first decode step.
    states: Vec<GrammarState>,
    /// Sorted union of every state's per-state context-dependent list,
    /// computed lazily on first `context_dependent_ids()` call. Purely
    /// informational — exposed for callers that want the whole-vocabulary
    /// picture; the decode recheck loop uses `context_dependent_ids_for_state`,
    /// never this.
    context_dependent: OnceLock<Vec<usize>>,
}

/// Precomputed data for one grammar state, built on first visit.
///
/// `mask` encodes which tokens are allowed (or optimistically allowed and
/// pending the runtime context-dependent recheck) in this state.
/// `context_dependent` lists the tokens that need runtime PDA inspection;
/// every built row carries its own list, so a decode step in state `s`
/// never consults any other state's candidates.
struct StateRow {
    mask: Vec<u64>,
    context_dependent: Vec<usize>,
}

impl VocabPartition {
    /// Build the vocabulary partition.
    ///
    /// `grammar_states` are the grammar states to track.  `vocab_bytes[i]`
    /// is the byte sequence for token `i`.
    ///
    /// Per-state precomputation is **lazy**: this call only stores the
    /// enumerated states (capped at `MAX_GRAMMAR_STATES`) and one
    /// uninitialised row cell per state.  A state's mask and
    /// context-dependent list are simulated on first visit
    /// (`build_state_row`), so construction no longer pays
    /// O(|states| × |vocab| × |token_length|) and only states a decode run
    /// actually reaches ever cost anything.
    ///
    /// `grammar` and `vocab_bytes` are shared (not cloned) with the caller
    /// — `GrammarEngine` keeps its own handles to the same allocation.
    pub fn build(
        grammar: &Arc<CompiledGrammar>,
        grammar_states: Vec<GrammarState>,
        vocab_bytes: &Arc<Vec<Vec<u8>>>,
    ) -> Self {
        let vocab_size = vocab_bytes.len();
        let mask_stride = vocab_size.div_ceil(64);
        let num_states = grammar_states.len();

        if num_states > MAX_GRAMMAR_STATES {
            tracing::warn!(
                "grammar has {} states (max {}); first {} will be precomputed",
                num_states,
                MAX_GRAMMAR_STATES,
                MAX_GRAMMAR_STATES
            );
        }

        let effective_states = num_states.min(MAX_GRAMMAR_STATES);
        let mut rows = Vec::with_capacity(effective_states);
        for _ in 0..effective_states {
            rows.push(OnceLock::new());
        }

        Self {
            grammar: Arc::clone(grammar),
            vocab: Arc::clone(vocab_bytes),
            mask_stride,
            vocab_size,
            rows,
            states: grammar_states,
            context_dependent: OnceLock::new(),
        }
    }

    /// Simulate every (this state, token) pair into one row.
    ///
    /// Called once per state, on first visit, from `state_row` via
    /// `OnceLock::get_or_init`.
    fn build_state_row(
        grammar_state: &GrammarState,
        grammar: &CompiledGrammar,
        vocab_bytes: &[Vec<u8>],
    ) -> StateRow {
        let vocab_size = vocab_bytes.len();
        let mask_stride = vocab_size.div_ceil(64);
        let mut mask = vec![0u64; mask_stride];
        let mut state_context_dependent: Vec<usize> = Vec::new();

        for (token_id, token_bytes) in vocab_bytes.iter().enumerate() {
            // Skip empty tokens.
            if token_bytes.is_empty() {
                continue;
            }

            let (sim_result, _) = simulate_token(grammar_state, grammar, token_bytes);
            match sim_result {
                SimResult::Accept => {
                    // Set bit for this token in state's mask.
                    let word = token_id / 64;
                    let bit = token_id % 64;
                    mask[word] |= 1u64 << bit;
                }
                SimResult::ContextDependent => {
                    // Mark as context-dependent for this state.
                    state_context_dependent.push(token_id);
                    // Also set the bit optimistically (runtime check will verify).
                    let word = token_id / 64;
                    let bit = token_id % 64;
                    mask[word] |= 1u64 << bit;
                }
                SimResult::Reject => {
                    // Bit remains 0 (token disallowed).
                }
            }
        }
        state_context_dependent.shrink_to_fit();

        StateRow {
            mask,
            context_dependent: state_context_dependent,
        }
    }

    /// Resolve `state_id`'s row, building it on first visit.
    ///
    /// `get_or_init` hands back `&StateRow` from `&self`, so no public
    /// signature changes and no lock is held after initialisation.
    fn state_row(&self, state_id: usize) -> &StateRow {
        let cell = &self.rows[state_id];
        cell.get_or_init(|| {
            Self::build_state_row(
                &self.states[state_id],
                self.grammar.as_ref(),
                self.vocab.as_ref(),
            )
        })
    }

    /// Apply the precomputed bitmask for `state_id` to `logits` in-place.
    ///
    /// Sets disallowed token positions to `f32::NEG_INFINITY`.
    /// Cost: O(vocab_size / 64) word-level iterations.
    pub fn apply_mask(&self, state_id: usize, logits: &mut [f32]) {
        debug_assert!(
            logits.len() >= self.vocab_size,
            "logits slice shorter than vocab_size"
        );
        if state_id >= self.states.len().min(MAX_GRAMMAR_STATES) {
            // Unknown state: block all tokens (fail-closed).
            for l in logits[..self.vocab_size].iter_mut() {
                *l = f32::NEG_INFINITY;
            }
            return;
        }

        let row = self.state_row(state_id);
        for word_idx in 0..self.mask_stride {
            let mask_word = row.mask[word_idx];
            let base_token = word_idx * 64;
            if mask_word == u64::MAX {
                // All 64 tokens in this word allowed — skip inner loop.
                continue;
            }
            if mask_word == 0 {
                // All 64 disallowed — fast fill.
                let end = (base_token + 64).min(self.vocab_size);
                for l in logits[base_token..end].iter_mut() {
                    *l = f32::NEG_INFINITY;
                }
                continue;
            }
            // Mixed word: check each bit.
            for bit in 0..64u32 {
                let token_idx = base_token + bit as usize;
                if token_idx >= self.vocab_size {
                    break;
                }
                if mask_word & (1u64 << bit) == 0 {
                    logits[token_idx] = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// Returns the token ids that are context-dependent for at least one
    /// state: the sorted union of every tracked state's per-state list.
    /// These require runtime PDA stack inspection before finalising the
    /// mask.
    ///
    /// Informational access to the whole-vocabulary picture; the decode
    /// recheck loop uses `context_dependent_ids_for_state`, not this.
    /// Computed lazily on first call (which builds every state's row), so a
    /// partition whose states are never all visited does not pay for a
    /// full-scan union it does not need.
    pub fn context_dependent_ids(&self) -> &[usize] {
        self.context_dependent.get_or_init(|| {
            let mut union: HashSet<usize> = HashSet::new();
            for state_id in 0..self.num_states() {
                union.extend(self.state_row(state_id).context_dependent.iter().copied());
            }
            let mut v: Vec<usize> = union.into_iter().collect();
            v.sort_unstable();
            v
        })
    }

    /// Returns the token ids that need runtime PDA inspection in `state_id`,
    /// from that state's own list (built on first visit).
    ///
    /// An out-of-range `state_id` returns an empty slice — fail-closed. No
    /// other state's or the global union's candidates may stand in for an
    /// unknown state: `apply_mask` already fail-closes an unknown state to
    /// all-`NEG_INFINITY`, so a decode step cannot rely on this method
    /// alone, and handing out stale candidates would only add runtime
    /// rechecks of tokens that are already blocked.
    pub(crate) fn context_dependent_ids_for_state(&self, state_id: usize) -> &[usize] {
        if state_id >= self.num_states() {
            return &[];
        }
        &self.state_row(state_id).context_dependent
    }

    /// Returns the number of precomputed grammar states.
    pub fn num_states(&self) -> usize {
        self.states.len().min(MAX_GRAMMAR_STATES)
    }

    /// Return the `GrammarState` for a given `state_id`.
    pub fn grammar_state(&self, state_id: usize) -> Option<&GrammarState> {
        self.states.get(state_id)
    }

    /// Return whether any token allowed by the precomputed mask satisfies
    /// `predicate`.
    pub(crate) fn any_allowed_token(
        &self,
        state_id: usize,
        mut predicate: impl FnMut(usize) -> bool,
    ) -> bool {
        if state_id >= self.states.len().min(MAX_GRAMMAR_STATES) {
            return false;
        }

        let row = self.state_row(state_id);
        for word_idx in 0..self.mask_stride {
            let mut mask_word = row.mask[word_idx];
            while mask_word != 0 {
                let bit = mask_word.trailing_zeros() as usize;
                let token_id = word_idx * 64 + bit;
                if token_id < self.vocab_size && predicate(token_id) {
                    return true;
                }
                mask_word &= mask_word - 1;
            }
        }
        false
    }

    /// Whether `state_id`'s row has been built (initialised). Test hook for
    /// the laziness contract.
    #[cfg(test)]
    pub(crate) fn row_is_built(&self, state_id: usize) -> bool {
        self.rows
            .get(state_id)
            .is_some_and(|cell| cell.get().is_some())
    }

    /// Pre-seed the global union as empty so a test can prove the decode
    /// path never consults it. Idempotent and effective only while the
    /// union is still uninitialised — which holds for a freshly built
    /// partition that has only run `apply_mask`/`mask_logits` (both of
    /// which read the per-state lists, not the union).
    #[cfg(test)]
    pub(crate) fn force_context_union_empty_for_test(&self) {
        self.context_dependent.get_or_init(Vec::new);
    }

    /// Owned copy of `state_id`'s per-state context-dependent list, built
    /// on first visit. Test seam for the engine-level recheck-loop tests:
    /// the recheck loop iterates exactly this list, so an engine whose
    /// per-state list differs from the one a mutation would have produced
    /// rechecks a different candidate set.
    #[cfg(test)]
    pub(crate) fn ctx_list_for_state(&self, state_id: usize) -> Vec<usize> {
        self.context_dependent_ids_for_state(state_id).to_vec()
    }

    /// Install an explicit row for `state_id` while its cell is still
    /// uninitialised. Test seam so an engine-level test can drive the
    /// recheck loop with a controlled per-state list (e.g. emptied)
    /// without re-simulating the grammar. Panics in tests if the row was
    /// already built, which would silently void the intended mutation.
    #[cfg(test)]
    pub(crate) fn set_row_for_test(
        &self,
        state_id: usize,
        mask: Vec<u64>,
        context_dependent: Vec<usize>,
    ) {
        let cell = self
            .rows
            .get(state_id)
            .unwrap_or_else(|| panic!("set_row_for_test: no cell for state {state_id}"));
        cell.set(StateRow {
            mask,
            context_dependent,
        })
        .unwrap_or_else(|_| panic!("set_row_for_test: row {state_id} already initialised"));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::pda::{
        CompiledGrammar, GrammarBuilder, GrammarState, Rule, StepResult, Symbol, advance_byte,
        simulate_token,
    };

    /// Grammar: root = 'a' | 'b'
    fn or_grammar() -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        b.add_rule(
            "root",
            vec![vec![Symbol::Terminal(b'a')], vec![Symbol::Terminal(b'b')]],
        );
        b.build()
    }

    /// Two-token vocabulary: token 0 = b"a", token 1 = b"b".
    fn ab_vocab() -> Vec<Vec<u8>> {
        vec![b"a".to_vec(), b"b".to_vec()]
    }

    /// Three-token vocabulary: token 0 = b"a", token 1 = b"b", token 2 = b"c".
    fn abc_vocab() -> Vec<Vec<u8>> {
        vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]
    }

    #[test]
    fn build_basic_mask() {
        let grammar = or_grammar();
        let states = vec![GrammarState::initial()];
        let vocab = ab_vocab();
        let partition = VocabPartition::build(&Arc::new(grammar), states, &Arc::new(vocab));
        assert_eq!(partition.num_states(), 1);
    }

    #[test]
    fn apply_mask_allows_correct_tokens() {
        let grammar = or_grammar();
        let states = vec![GrammarState::initial()];
        let vocab = abc_vocab();
        let partition = VocabPartition::build(&Arc::new(grammar), states, &Arc::new(vocab));

        let mut logits = vec![1.0f32, 2.0f32, 3.0f32];
        partition.apply_mask(0, &mut logits);

        // Tokens 0 ('a') and 1 ('b') are allowed; token 2 ('c') is blocked.
        assert!(logits[0] > f32::NEG_INFINITY, "token 'a' should be allowed");
        assert!(logits[1] > f32::NEG_INFINITY, "token 'b' should be allowed");
        assert_eq!(logits[2], f32::NEG_INFINITY, "token 'c' should be blocked");
    }

    #[test]
    fn apply_mask_unknown_state_blocks_all() {
        let grammar = or_grammar();
        let states = vec![GrammarState::initial()];
        let vocab = ab_vocab();
        let partition = VocabPartition::build(&Arc::new(grammar), states, &Arc::new(vocab));

        let mut logits = vec![1.0f32, 2.0f32];
        // State 99 doesn't exist.
        partition.apply_mask(99, &mut logits);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn mask_all_zeros_fills_neg_inf() {
        // Grammar that accepts nothing: empty root.
        let grammar = CompiledGrammar {
            rules: vec![Rule {
                name: "root".to_string(),
                alts: vec![],
            }],
        };
        let states = vec![GrammarState::initial()];
        let vocab = ab_vocab();
        let partition = VocabPartition::build(&Arc::new(grammar), states, &Arc::new(vocab));

        let mut logits = vec![1.0f32, 2.0f32];
        partition.apply_mask(0, &mut logits);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn mask_all_ones_preserves_logits() {
        // Grammar: root = . (any byte) — all single-byte tokens allowed.
        let mut builder = GrammarBuilder::new();
        builder.add_rule("root", vec![vec![Symbol::AnyByte]]);
        let grammar = builder.build();

        let states = vec![GrammarState::initial()];
        let vocab = abc_vocab();
        let partition = VocabPartition::build(&Arc::new(grammar), states, &Arc::new(vocab));

        let mut logits = vec![1.0f32, 2.0f32, 3.0f32];
        partition.apply_mask(0, &mut logits);
        // No tokens should be blocked.
        for &l in &logits {
            assert!(l > f32::NEG_INFINITY);
        }
    }

    #[test]
    fn bitmask_and_correctness() {
        // Verify the bit-counting logic with a vocab of exactly 65 tokens
        // (two full 64-bit words plus one extra token).
        let grammar = or_grammar();
        // Build vocab: token 0 = b"a", 1 = b"b", 2..64 = b"c" repeated.
        let mut vocab: Vec<Vec<u8>> = vec![b"a".to_vec(), b"b".to_vec()];
        vocab.extend((2..65).map(|_| b"c".to_vec()));
        assert_eq!(vocab.len(), 65);

        let states = vec![GrammarState::initial()];
        let partition = VocabPartition::build(&Arc::new(grammar), states, &Arc::new(vocab));

        let mut logits = vec![1.0f32; 65];
        partition.apply_mask(0, &mut logits);

        // Only tokens 0 and 1 should be allowed.
        assert!(logits[0] > f32::NEG_INFINITY, "token 0 allowed");
        assert!(logits[1] > f32::NEG_INFINITY, "token 1 allowed");
        for i in 2..65 {
            assert_eq!(logits[i], f32::NEG_INFINITY, "token {i} blocked");
        }
    }

    #[test]
    fn empty_token_skipped() {
        let grammar = or_grammar();
        // vocab has an empty token at index 1.
        let vocab = vec![b"a".to_vec(), vec![], b"b".to_vec()];
        let states = vec![GrammarState::initial()];
        let partition = VocabPartition::build(&Arc::new(grammar), states, &Arc::new(vocab));

        let mut logits = vec![1.0f32; 3];
        partition.apply_mask(0, &mut logits);
        // Token 0 ('a') allowed, token 1 (empty) skipped = not allowed, token 2 ('b') allowed.
        assert!(logits[0] > f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY); // empty token not set
        assert!(logits[2] > f32::NEG_INFINITY);
    }

    #[test]
    fn context_dependent_ids_are_partitioned_by_state() {
        let mut builder = GrammarBuilder::new();
        builder.add_rule(
            "root",
            vec![b"abcd".iter().copied().map(Symbol::Terminal).collect()],
        );
        let grammar = builder.build();

        let state0 = GrammarState::initial();
        let mut state1 = state0.clone();
        assert_eq!(
            advance_byte(&mut state1, &grammar, b'a'),
            StepResult::Accepted
        );
        let mut state2 = state1.clone();
        assert_eq!(
            advance_byte(&mut state2, &grammar, b'b'),
            StepResult::Accepted
        );
        let vocab = vec![b"ax".to_vec(), b"bx".to_vec(), b"cx".to_vec()];
        let partition = VocabPartition::build(
            &Arc::new(grammar),
            vec![state0, state1, state2],
            &Arc::new(vocab),
        );

        assert_eq!(partition.context_dependent_ids(), &[0, 1, 2]);
        assert_eq!(partition.context_dependent_ids_for_state(0), &[0]);
        assert_eq!(partition.context_dependent_ids_for_state(1), &[1]);
        assert_eq!(partition.context_dependent_ids_for_state(2), &[2]);
        assert!(
            partition
                .context_dependent_ids_for_state(usize::MAX)
                .is_empty(),
            "unknown states must fail closed with an empty list, never the global union"
        );
    }

    #[test]
    fn out_of_range_state_ids_are_fail_closed() {
        // Pins the out-of-range contract: the per-state query must return an
        // empty list (never another state's candidates, never the global
        // union) and apply_mask must block every token.
        let grammar = or_grammar();
        let states = vec![GrammarState::initial()];
        let vocab = abc_vocab();
        let partition = VocabPartition::build(&Arc::new(grammar), states, &Arc::new(vocab));

        assert!(
            partition.context_dependent_ids_for_state(99).is_empty(),
            "unknown states must fail closed with an empty list"
        );

        let mut logits = vec![1.0f32, 2.0f32, 3.0f32];
        partition.apply_mask(99, &mut logits);
        for l in &logits {
            assert_eq!(*l, f32::NEG_INFINITY, "unknown state blocks all tokens");
        }
    }

    #[test]
    fn state_lookup_does_not_force_row_build() {
        // `num_states()` and `grammar_state()` must read only the stored
        // state list. The engine's per-step state lookup calls both for
        // every state on every decode step, so forcing a row build from
        // either would rebuild the eager all-states cost on the first
        // decode step and defeat the laziness.
        let grammar = or_grammar();
        let states = vec![GrammarState::initial()];
        let vocab = abc_vocab();
        let vocab_len = vocab.len();
        let partition = VocabPartition::build(&Arc::new(grammar), states, &Arc::new(vocab));

        for sid in 0..partition.num_states() {
            assert!(partition.grammar_state(sid).is_some(), "stored state list");
        }
        assert!(!partition.row_is_built(0), "lookup must not build rows");

        // A real visit still builds the row (control: the hook can
        // distinguish visited from unvisited).
        let mut logits = vec![1.0f32; vocab_len];
        partition.apply_mask(0, &mut logits);
        assert!(
            partition.row_is_built(0),
            "apply_mask builds the visited row"
        );
    }

    #[test]
    fn build_is_lazy_no_eager_simulation() {
        // The whole point of the redesign: `build` stores states but
        // simulates nothing, so no row is initialised after construction,
        // and the global union is not computed until asked for.
        let grammar = or_grammar();
        let states = vec![GrammarState::initial()];
        let vocab = abc_vocab();
        let partition = VocabPartition::build(&Arc::new(grammar), states, &Arc::new(vocab));

        assert!(
            !partition.row_is_built(0),
            "no row built at construction time"
        );
        assert!(
            partition.context_dependent.get().is_none(),
            "union not computed until context_dependent_ids() is called"
        );
    }

    /// Sanity: `simulate_token` classification used by the tests. A
    /// single valid boundary byte is Accept; a valid-then-invalid multi-byte
    /// token is ContextDependent.
    #[test]
    fn sim_classification_helper() {
        let mut builder = GrammarBuilder::new();
        builder.add_rule(
            "root",
            vec![b"ab".iter().copied().map(Symbol::Terminal).collect()],
        );
        let grammar = builder.build();
        let state0 = GrammarState::initial();
        assert_eq!(simulate_token(&state0, &grammar, b"a").0, SimResult::Accept);
        assert_eq!(
            simulate_token(&state0, &grammar, b"ax").0,
            SimResult::ContextDependent
        );
        assert_eq!(simulate_token(&state0, &grammar, b"b").0, SimResult::Reject);
    }
}

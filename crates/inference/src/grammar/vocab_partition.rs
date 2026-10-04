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
//! 1. `VocabPartition::build(grammar, grammar_states, vocab_bytes)` — called once
//!    at `GrammarEngine::new` time (which uses `build_from_trie` with the trie
//!    it keeps for the runtime fallback). It builds a byte trie over the
//!    vocabulary and walks it once per grammar state, so a rejected byte
//!    prunes every token sharing that prefix instead of simulating each
//!    (state, token) pair separately.
//! 2. `VocabPartition::apply_mask(state_id, logits)` — called per decode step.
//! 3. `VocabPartition::context_dependent_ids_for_state(state_id)` — returns
//!    the token ids that need runtime PDA inspection in the current state,
//!    when a state-local list was stored; a state whose list was withheld
//!    under the aggregate capacity budget falls back to the global union
//!    across every state.

use crate::grammar::pda::{CompiledGrammar, GrammarState};
use crate::grammar::trie::ByteTrie;

/// Maximum number of grammar states for v0.
/// A grammar with more states triggers a warning at build time.
pub const MAX_GRAMMAR_STATES: usize = 256;

/// Precomputed vocabulary partition for a grammar.
///
/// `state_count` is the number of distinct grammar states tracked.  For
/// most JSON schemas this is the number of unique PDA stack configurations
/// reachable from the initial state — typically under 100.
pub struct VocabPartition {
    /// Bitmask table.  `masks[s * mask_stride + w]` has bit `t % 64` set if
    /// token `w * 64 + t % 64` is allowed in grammar state `s`.
    masks: Vec<u64>,
    mask_stride: usize,
    vocab_size: usize,
    /// Grammar states indexed by `state_id`.
    states: Vec<GrammarState>,
    /// Token ids that are context-dependent for at least one grammar state.
    context_dependent: Vec<usize>,
    /// Context-dependent token ids indexed by precomputed grammar state.
    ///
    /// `None` uses the conservative global union because storing that state's
    /// local set would exceed the aggregate memory budget.
    context_dependent_by_state: Vec<Option<Vec<usize>>>,
}

impl VocabPartition {
    /// Build the vocabulary partition by walking a byte trie over the
    /// vocabulary once per grammar state.
    ///
    /// `grammar_states` are the grammar states to precompute masks for.
    /// `vocab_bytes[i]` is the byte sequence for token `i`.
    ///
    /// Builds a [`ByteTrie`] over `vocab_bytes` and delegates to
    /// [`Self::build_from_trie`]. Callers that already hold a trie for this
    /// vocabulary (`GrammarEngine::new`) use that entry point directly so the
    /// trie is built once.
    pub fn build(
        grammar: &CompiledGrammar,
        grammar_states: Vec<GrammarState>,
        vocab_bytes: &[Vec<u8>],
    ) -> Self {
        let trie = ByteTrie::build(vocab_bytes);
        Self::build_from_trie(grammar, grammar_states, vocab_bytes.len(), &trie)
    }

    /// Build the vocabulary partition from a prebuilt `trie` over a
    /// vocabulary of `vocab_size` tokens.
    ///
    /// For each precomputed state, one [`ByteTrie::classify`] walk visits the
    /// vocabulary's shared byte prefixes and classifies every non-empty token
    /// the way `simulate_token` would: a rejected first byte rejects the token,
    /// a rejection after the first byte makes it context-dependent, and a
    /// fully accepted token is allowed. Empty tokens are always blocked (an
    /// empty token emits no bytes, so allowing one would let decoding make no
    /// progress), as in the previous per-token builder, although
    /// `simulate_token` returns `Accept` for an empty slice. Cost is
    /// O(|states| × trie nodes) PDA steps
    /// plus the size of each state's context-dependent set; a rejected byte
    /// prunes every token sharing that prefix at once. `trie` must have been
    /// built from the same vocabulary of `vocab_size` tokens.
    pub(crate) fn build_from_trie(
        grammar: &CompiledGrammar,
        grammar_states: Vec<GrammarState>,
        vocab_size: usize,
        trie: &ByteTrie,
    ) -> Self {
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
        let mut masks = vec![0u64; effective_states * mask_stride];
        // Bitset over token ids: the union of every state's context-dependent
        // ids, read back in increasing id order.
        let mut ctx_dep_bits = vec![0u64; mask_stride];
        let mut context_dependent_by_state = Vec::with_capacity(effective_states);
        // Keep the aggregate length of the new state-local lists no larger
        // than the existing mask table (measured in entries, not allocator
        // capacity, so the decision is deterministic). A dense adversarial
        // grammar can classify every token as context-dependent in every
        // state; storing all of those ids would otherwise cost 64x the masks
        // on 64-bit hosts.
        // Falling back to the global union preserves exact masking semantics.
        let context_entry_budget =
            masks.len().saturating_mul(std::mem::size_of::<u64>()) / std::mem::size_of::<usize>();
        let mut context_entries_stored = 0usize;

        for (state_id, grammar_state) in grammar_states[..effective_states].iter().enumerate() {
            let state_mask = &mut masks[state_id * mask_stride..(state_id + 1) * mask_stride];
            let mut state_context_dependent = Vec::new();
            trie.classify(
                grammar_state,
                grammar,
                state_mask,
                &mut state_context_dependent,
            );
            // The walk emits ids in trie order; the stored list is in
            // increasing token id order.
            state_context_dependent.sort_unstable();
            for &token_id in &state_context_dependent {
                // Also set the bit optimistically (runtime check will verify).
                state_mask[token_id / 64] |= 1u64 << (token_id % 64);
                ctx_dep_bits[token_id / 64] |= 1u64 << (token_id % 64);
            }
            let stored_len = state_context_dependent.len();
            if context_entries_stored.saturating_add(stored_len) <= context_entry_budget {
                context_entries_stored += stored_len;
                state_context_dependent.shrink_to_fit();
                context_dependent_by_state.push(Some(state_context_dependent));
            } else {
                context_dependent_by_state.push(None);
            }
        }

        let mut context_dependent = Vec::new();
        for (word_idx, &word) in ctx_dep_bits.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                context_dependent.push(word_idx * 64 + word.trailing_zeros() as usize);
                word &= word - 1;
            }
        }

        Self {
            masks,
            mask_stride,
            vocab_size,
            states: grammar_states,
            context_dependent,
            context_dependent_by_state,
        }
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

        let mask_base = state_id * self.mask_stride;
        for word_idx in 0..self.mask_stride {
            let mask_word = self.masks[mask_base + word_idx];
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

    /// Returns the token ids that are context-dependent for at least one state.
    /// These require runtime PDA stack inspection before finalising the mask.
    pub fn context_dependent_ids(&self) -> &[usize] {
        &self.context_dependent
    }

    /// Returns the token ids that need runtime PDA inspection in `state_id`.
    ///
    /// An unknown state falls back to the conservative global union rather than
    /// skipping runtime checks.
    pub(crate) fn context_dependent_ids_for_state(&self, state_id: usize) -> &[usize] {
        self.context_dependent_by_state
            .get(state_id)
            .and_then(Option::as_deref)
            .unwrap_or(&self.context_dependent)
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

        let mask_base = state_id * self.mask_stride;
        for word_idx in 0..self.mask_stride {
            let mut mask_word = self.masks[mask_base + word_idx];
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
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::gbnf::parse_gbnf;
    use crate::grammar::json_schema::compile;
    use crate::grammar::pda::{
        CompiledGrammar, GrammarBuilder, GrammarState, Rule, SimResult, StepResult, Symbol,
        advance_byte, initial_grammar_state, simulate_token,
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
        let partition = VocabPartition::build(&grammar, states, &vocab);
        assert_eq!(partition.num_states(), 1);
    }

    #[test]
    fn apply_mask_allows_correct_tokens() {
        let grammar = or_grammar();
        let states = vec![GrammarState::initial()];
        let vocab = abc_vocab();
        let partition = VocabPartition::build(&grammar, states, &vocab);

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
        let partition = VocabPartition::build(&grammar, states, &vocab);

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
        let partition = VocabPartition::build(&grammar, states, &vocab);

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
        let partition = VocabPartition::build(&grammar, states, &vocab);

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
        let partition = VocabPartition::build(&grammar, states, &vocab);

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
        let partition = VocabPartition::build(&grammar, states, &vocab);

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
        let partition = VocabPartition::build(&grammar, vec![state0, state1, state2], &vocab);

        assert_eq!(partition.context_dependent_ids(), &[0, 1, 2]);
        assert_eq!(partition.context_dependent_ids_for_state(0), &[0]);
        assert_eq!(partition.context_dependent_ids_for_state(1), &[1]);
        assert_eq!(partition.context_dependent_ids_for_state(2), &[2]);
        assert_eq!(
            partition.context_dependent_ids_for_state(usize::MAX),
            &[0, 1, 2],
            "unknown states must use the conservative global union"
        );
    }

    #[test]
    fn dense_state_lists_fall_back_within_mask_sized_budget() {
        let mut builder = GrammarBuilder::new();
        builder.add_rule(
            "root",
            vec![b"aaaa".iter().copied().map(Symbol::Terminal).collect()],
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
            advance_byte(&mut state2, &grammar, b'a'),
            StepResult::Accepted
        );
        let vocab = vec![b"ax".to_vec(); 128];
        let partition = VocabPartition::build(&grammar, vec![state0, state1, state2], &vocab);

        assert_eq!(partition.context_dependent_ids().len(), 128);
        assert!(
            partition
                .context_dependent_by_state
                .iter()
                .all(Option::is_none),
            "dense local lists must use the global fallback instead of exceeding the mask-sized \
             storage budget"
        );
        for state_id in 0..3 {
            assert_eq!(
                partition.context_dependent_ids_for_state(state_id),
                partition.context_dependent_ids()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Equivalence with the per-token simulation the trie walk replaced
    // -----------------------------------------------------------------------

    /// The pre-trie construction, kept as the oracle: simulates every
    /// (state, token) pair independently with `simulate_token`.
    fn build_by_simulation(
        grammar: &CompiledGrammar,
        grammar_states: Vec<GrammarState>,
        vocab_bytes: &[Vec<u8>],
    ) -> VocabPartition {
        let vocab_size = vocab_bytes.len();
        let mask_stride = vocab_size.div_ceil(64);
        let num_states = grammar_states.len();
        let effective_states = num_states.min(MAX_GRAMMAR_STATES);
        let mut masks = vec![0u64; effective_states * mask_stride];
        let mut ctx_dep_set = std::collections::HashSet::new();
        let mut context_dependent_by_state = Vec::with_capacity(effective_states);
        let context_entry_budget =
            masks.len().saturating_mul(std::mem::size_of::<u64>()) / std::mem::size_of::<usize>();
        let mut context_entries_stored = 0usize;

        for (state_id, grammar_state) in grammar_states[..effective_states].iter().enumerate() {
            let mut state_context_dependent = Vec::new();
            for (token_id, token_bytes) in vocab_bytes.iter().enumerate() {
                // Empty tokens are always blocked, although `simulate_token`
                // returns `Accept` for an empty slice.
                if token_bytes.is_empty() {
                    continue;
                }
                let (sim_result, _) = simulate_token(grammar_state, grammar, token_bytes);
                match sim_result {
                    SimResult::Accept => {
                        masks[state_id * mask_stride + token_id / 64] |= 1u64 << (token_id % 64);
                    }
                    SimResult::ContextDependent => {
                        ctx_dep_set.insert(token_id);
                        state_context_dependent.push(token_id);
                        masks[state_id * mask_stride + token_id / 64] |= 1u64 << (token_id % 64);
                    }
                    SimResult::Reject => {}
                }
            }
            let stored_len = state_context_dependent.len();
            if context_entries_stored.saturating_add(stored_len) <= context_entry_budget {
                context_entries_stored += stored_len;
                state_context_dependent.shrink_to_fit();
                context_dependent_by_state.push(Some(state_context_dependent));
            } else {
                context_dependent_by_state.push(None);
            }
        }

        let mut context_dependent: Vec<usize> = ctx_dep_set.into_iter().collect();
        context_dependent.sort_unstable();

        VocabPartition {
            masks,
            mask_stride,
            vocab_size,
            states: grammar_states,
            context_dependent,
            context_dependent_by_state,
        }
    }

    /// Breadth-first reachable states, expanded by simulating every token
    /// (same rule as the engine's enumeration): the states keep the byte
    /// history `simulate_token` leaves behind, as they do in production.
    fn reachable_states(
        grammar: &CompiledGrammar,
        vocab: &[Vec<u8>],
        max_states: usize,
    ) -> Vec<GrammarState> {
        let initial = initial_grammar_state(grammar);
        let mut queue = vec![initial.clone()];
        let mut visited = vec![initial];
        let mut head = 0;
        while head < queue.len() && visited.len() < max_states {
            let state = queue[head].clone();
            head += 1;
            for token in vocab.iter().filter(|t| !t.is_empty()) {
                let (result, next) = simulate_token(&state, grammar, token);
                if result != SimResult::Reject
                    && !visited
                        .iter()
                        .any(|s| s.stack == next.stack && s.complete == next.complete)
                {
                    visited.push(next.clone());
                    if visited.len() < max_states {
                        queue.push(next);
                    }
                }
            }
        }
        visited
    }

    fn assert_partitions_equal(got: &VocabPartition, want: &VocabPartition, label: &str) {
        assert_eq!(got.vocab_size, want.vocab_size, "{label}: vocab_size");
        assert_eq!(got.mask_stride, want.mask_stride, "{label}: mask_stride");
        assert_eq!(got.states.len(), want.states.len(), "{label}: state count");
        for (i, (g, w)) in got.states.iter().zip(&want.states).enumerate() {
            assert_eq!(g.stack, w.stack, "{label}: state {i} stack");
            assert_eq!(g.complete, w.complete, "{label}: state {i} complete");
            assert_eq!(
                g.partial_token_bytes, w.partial_token_bytes,
                "{label}: state {i} partial_token_bytes"
            );
        }
        assert_eq!(got.masks, want.masks, "{label}: masks");
        assert_eq!(
            got.context_dependent, want.context_dependent,
            "{label}: global context_dependent"
        );
        assert_eq!(
            got.context_dependent_by_state, want.context_dependent_by_state,
            "{label}: context_dependent_by_state"
        );
    }

    /// Builds with both constructions (and through a prebuilt trie) and
    /// requires every field to match, including each stored state's stack,
    /// completion flag and partial bytes.
    fn assert_trie_build_matches_oracle(
        grammar: &CompiledGrammar,
        states: Vec<GrammarState>,
        vocab: &[Vec<u8>],
        label: &str,
    ) -> VocabPartition {
        let want = build_by_simulation(grammar, states.clone(), vocab);
        let got = VocabPartition::build(grammar, states.clone(), vocab);
        assert_partitions_equal(&got, &want, label);
        let trie = ByteTrie::build(vocab);
        let from_trie = VocabPartition::build_from_trie(grammar, states, vocab.len(), &trie);
        assert_partitions_equal(&from_trie, &want, label);
        got
    }

    /// All 256 single bytes, JSON-shaped multi-byte tokens sharing prefixes,
    /// two ids on one byte sequence, and empty tokens (always blocked).
    fn equivalence_vocab() -> Vec<Vec<u8>> {
        let mut vocab: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        for frag in [
            "\"name\"",
            "\"nam",
            "\"na",
            "\"age\"",
            "\"a",
            "\":",
            "\": ",
            "\":1",
            ",\"",
            ",\"n",
            "{\"",
            "{\"name\":",
            "}",
            "},",
            "[]",
            "[1,2]",
            "[\"",
            "true",
            "tru",
            "false",
            "null",
            "12",
            "123",
            "-1",
            "1.5",
            "ab",
            "ac",
            "ad",
            "abc",
            "abd",
            "abcd",
            "ax",
            "abx",
            "red",
            "re",
            "green",
            "\"red\"",
            "\"green\"",
        ] {
            vocab.push(frag.as_bytes().to_vec());
        }
        vocab.push(Vec::new());
        // Duplicate byte sequences under distinct ids.
        vocab.push(b"\"name\"".to_vec());
        vocab.push(b"a".to_vec());
        vocab.push(Vec::new());
        vocab
    }

    fn optional_properties_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
                "active": {"type": "boolean"},
                "score": {"type": "number"},
                "title": {"type": "string"},
                "city": {"type": "string"},
                "zip": {"type": "string"},
                "phone": {"type": "string"},
                "note": {"type": "string"}
            }
        })
    }

    fn enum_and_array_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "color": {"type": "string", "enum": ["red", "green", "blue"]},
                "ids": {"type": "array", "items": {"type": "integer"}},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["color", "ids"]
        })
    }

    #[test]
    fn trie_build_matches_oracle_for_optional_properties_schema() {
        let grammar = compile(&optional_properties_schema()).unwrap();
        let vocab = equivalence_vocab();
        let states = reachable_states(&grammar, &vocab, MAX_GRAMMAR_STATES);
        assert!(
            states.len() > 10,
            "schema must reach many states, got {}",
            states.len()
        );
        let partition =
            assert_trie_build_matches_oracle(&grammar, states, &vocab, "optional properties");
        assert!(
            !partition.context_dependent_ids().is_empty(),
            "fixture must exercise context-dependent tokens"
        );
    }

    #[test]
    fn trie_build_matches_oracle_for_enum_and_array_schema() {
        let grammar = compile(&enum_and_array_schema()).unwrap();
        let vocab = equivalence_vocab();
        let states = reachable_states(&grammar, &vocab, MAX_GRAMMAR_STATES);
        assert_trie_build_matches_oracle(&grammar, states, &vocab, "enum and array");
    }

    #[test]
    fn trie_build_matches_oracle_for_gbnf_alternatives_sharing_a_first_byte() {
        let grammar = parse_gbnf("root ::= \"ab\" | \"ac\" | \"abcd\" | \"ad\" \"x\"\n").unwrap();
        let vocab = equivalence_vocab();
        let states = reachable_states(&grammar, &vocab, MAX_GRAMMAR_STATES);
        assert!(states.len() > 1);
        assert_trie_build_matches_oracle(&grammar, states, &vocab, "gbnf shared first byte");
    }

    #[test]
    fn trie_build_matches_oracle_when_tokens_straddle_a_boundary() {
        // "ax" and "abx" get past their first byte and are then rejected, so
        // they are context-dependent where "x" alone is rejected outright.
        let mut builder = GrammarBuilder::new();
        builder.add_rule(
            "root",
            vec![b"abcd".iter().copied().map(Symbol::Terminal).collect()],
        );
        let grammar = builder.build();
        let vocab = equivalence_vocab();
        let states = reachable_states(&grammar, &vocab, MAX_GRAMMAR_STATES);
        let partition = assert_trie_build_matches_oracle(&grammar, states, &vocab, "straddle");
        let ax = vocab.iter().position(|t| t == b"ax").unwrap();
        let abx = vocab.iter().position(|t| t == b"abx").unwrap();
        assert!(partition.context_dependent_ids().contains(&ax));
        assert!(partition.context_dependent_ids().contains(&abx));
        assert!(partition.context_dependent_ids_for_state(0).contains(&ax));
    }

    #[test]
    fn trie_build_matches_oracle_when_the_context_entry_budget_is_exceeded() {
        let mut builder = GrammarBuilder::new();
        builder.add_rule(
            "root",
            vec![b"aaaa".iter().copied().map(Symbol::Terminal).collect()],
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
            advance_byte(&mut state2, &grammar, b'a'),
            StepResult::Accepted
        );
        let states = vec![state0, state1, state2];

        // 3 states x 2 mask words = a 6-entry budget. Four context-dependent
        // ids per state fit once and then overflow, so one `Some` precedes
        // the `None`s.
        let mut vocab = vec![b"ax".to_vec(); 4];
        vocab.extend(std::iter::repeat_n(b"c".to_vec(), 66));
        let mixed = assert_trie_build_matches_oracle(&grammar, states.clone(), &vocab, "mixed");
        assert!(mixed.context_dependent_by_state[0].is_some());
        assert!(mixed.context_dependent_by_state[1].is_none());

        // Every state's list alone is over budget.
        let dense = vec![b"ax".to_vec(); 128];
        let all_none = assert_trie_build_matches_oracle(&grammar, states, &dense, "dense");
        assert!(
            all_none
                .context_dependent_by_state
                .iter()
                .all(Option::is_none)
        );
    }

    #[test]
    fn trie_build_matches_oracle_past_the_state_cap() {
        let mut chain = Vec::new();
        chain.extend(std::iter::repeat_n(Symbol::Terminal(b'a'), 300));
        let grammar = CompiledGrammar {
            rules: vec![Rule {
                name: "root".to_string(),
                alts: vec![chain],
            }],
        };
        let mut states = vec![GrammarState::initial()];
        while states.len() < MAX_GRAMMAR_STATES + 14 {
            let mut next = states.last().unwrap().clone();
            assert_eq!(
                advance_byte(&mut next, &grammar, b'a'),
                StepResult::Accepted
            );
            states.push(next);
        }
        assert!(states.len() > MAX_GRAMMAR_STATES);
        let vocab = vec![
            b"a".to_vec(),
            b"aa".to_vec(),
            b"ab".to_vec(),
            b"b".to_vec(),
            Vec::new(),
            b"a".to_vec(),
        ];
        let partition = assert_trie_build_matches_oracle(&grammar, states, &vocab, "state cap");
        assert_eq!(partition.num_states(), MAX_GRAMMAR_STATES);
        assert_eq!(
            partition.context_dependent_by_state.len(),
            MAX_GRAMMAR_STATES
        );
    }

    /// Empty tokens are always blocked, as in the previous per-token builder
    /// (an empty token emits no bytes), even though `simulate_token` returns
    /// `Accept` for an empty slice; the oracle skips them the same way.
    #[test]
    fn trie_build_blocks_empty_tokens_like_the_previous_builder() {
        let grammar = or_grammar();
        let states = vec![GrammarState::initial()];
        for vocab in [vec![Vec::new(); 97], Vec::new()] {
            let partition =
                assert_trie_build_matches_oracle(&grammar, states.clone(), &vocab, "empty vocab");
            assert!(partition.masks.iter().all(|&w| w == 0));
            assert!(partition.context_dependent_ids().is_empty());
            assert_eq!(partition.context_dependent_by_state, vec![Some(Vec::new())]);
        }
    }

    /// A vocabulary token's byte length must not drive the native call depth
    /// of the classification walk: a 200_000-byte token is classified on a
    /// thread with a 256 KiB stack, which a recursive walk overflows.
    #[test]
    fn trie_build_classifies_a_very_long_token_on_a_small_stack() {
        const LONG: usize = 200_000;
        // A flat chain keeps the PDA stack shallow however many bytes are
        // accepted (a `*` repetition recurses and would hit `MAX_PDA_DEPTH`).
        let grammar = CompiledGrammar {
            rules: vec![Rule {
                name: "root".to_string(),
                alts: vec![vec![Symbol::Terminal(b'a'); LONG]],
            }],
        };
        let mut vocab = vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"aa".to_vec(),
            b"ab".to_vec(),
            b"ba".to_vec(),
        ];
        // Accepted at every byte: classified through accepted edges only.
        let accepted_id = vocab.len();
        vocab.push(vec![b'a'; LONG]);
        // Rejected on its second byte: the whole long chain below the
        // rejected edge is collected as context-dependent.
        let context_id = vocab.len();
        let mut straddle = b"ab".to_vec();
        straddle.extend(std::iter::repeat_n(b'c', LONG));
        vocab.push(straddle);
        let states = vec![initial_grammar_state(&grammar)];

        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(256 * 1024)
                .spawn_scoped(scope, || {
                    let partition = assert_trie_build_matches_oracle(
                        &grammar,
                        states.clone(),
                        &vocab,
                        "long token",
                    );
                    assert!(
                        partition.masks[accepted_id / 64] & (1u64 << (accepted_id % 64)) != 0,
                        "the all-accepted long token must be allowed"
                    );
                    assert!(
                        !partition
                            .context_dependent_ids_for_state(0)
                            .contains(&accepted_id)
                    );
                    assert!(
                        partition
                            .context_dependent_ids_for_state(0)
                            .contains(&context_id),
                        "the long token rejected after its first byte is context-dependent"
                    );
                })
                .unwrap()
                .join()
                .unwrap();
        });
    }
}

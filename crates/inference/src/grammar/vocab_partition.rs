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
    /// Set when classifying some (state, token) pair exceeded the matcher's
    /// stack limits. Such a pair is neither allowed nor rejected by the
    /// grammar, so the engine refuses to build on a partition that has one.
    stack_limit_exceeded: bool,
}

impl VocabPartition {
    /// Build the vocabulary partition by walking a byte trie over the
    /// vocabulary once per grammar state.
    ///
    /// `grammar_states` are the grammar states to precompute masks for.
    /// `vocab_bytes[i]` is the byte sequence for token `i`.
    ///
    /// Builds a [`ByteTrie`] over `vocab_bytes` and delegates to
    /// `Self::build_from_trie`. Callers that already hold a trie for this
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
        let mut stack_limit_exceeded = false;

        for (state_id, grammar_state) in grammar_states[..effective_states].iter().enumerate() {
            let state_mask = &mut masks[state_id * mask_stride..(state_id + 1) * mask_stride];
            let mut state_context_dependent = Vec::new();
            stack_limit_exceeded |= trie.classify(
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
            stack_limit_exceeded,
        }
    }

    /// Returns true when some (state, token) pair could not be classified
    /// because the matcher exceeded its stack limits during the build.
    ///
    /// A set flag means the mask is not a grammar verdict: an unclassified
    /// token has its bit unset in `apply_mask`, which blocks it exactly like a
    /// token the grammar rejects. `GrammarEngine::new` refuses to build on a
    /// partition with this flag set; a caller that uses the partition directly
    /// must check it before trusting a mask.
    pub fn stack_limit_exceeded(&self) -> bool {
        self.stack_limit_exceeded
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
mod tests;

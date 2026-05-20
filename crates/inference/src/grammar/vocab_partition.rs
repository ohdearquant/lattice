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
//!    at `GrammarEngine::new` time.
//! 2. `VocabPartition::apply_mask(state_id, logits)` — called per decode step.
//! 3. `VocabPartition::context_dependent_ids()` — returns token ids that need
//!    runtime PDA inspection.

use crate::grammar::pda::{CompiledGrammar, GrammarState, SimResult, simulate_token};

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
}

impl VocabPartition {
    /// Build the vocabulary partition by simulating every (state, token) pair.
    ///
    /// `grammar_states` are the grammar states to precompute masks for.
    /// `vocab_bytes[i]` is the byte sequence for token `i`.
    ///
    /// This runs in O(|states| × |vocab| × |token_length|) time and is
    /// called once at `GrammarEngine::new` time.
    pub fn build(
        grammar: &CompiledGrammar,
        grammar_states: Vec<GrammarState>,
        vocab_bytes: &[Vec<u8>],
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
        let mut masks = vec![0u64; effective_states * mask_stride];
        let mut ctx_dep_set = std::collections::HashSet::new();

        for (state_id, grammar_state) in grammar_states[..effective_states].iter().enumerate() {
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
                        masks[state_id * mask_stride + word] |= 1u64 << bit;
                    }
                    SimResult::ContextDependent => {
                        // Mark as context-dependent.
                        ctx_dep_set.insert(token_id);
                        // Also set the bit optimistically (runtime check will verify).
                        let word = token_id / 64;
                        let bit = token_id % 64;
                        masks[state_id * mask_stride + word] |= 1u64 << bit;
                    }
                    SimResult::Reject => {
                        // Bit remains 0 (token disallowed).
                    }
                }
            }
        }

        let mut context_dependent: Vec<usize> = ctx_dep_set.into_iter().collect();
        context_dependent.sort_unstable();

        Self {
            masks,
            mask_stride,
            vocab_size,
            states: grammar_states,
            context_dependent,
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

    /// Returns the number of precomputed grammar states.
    pub fn num_states(&self) -> usize {
        self.states.len().min(MAX_GRAMMAR_STATES)
    }

    /// Return the `GrammarState` for a given `state_id`.
    pub fn grammar_state(&self, state_id: usize) -> Option<&GrammarState> {
        self.states.get(state_id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::pda::{Alt, CompiledGrammar, GrammarBuilder, GrammarState, Rule, Symbol};

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
}

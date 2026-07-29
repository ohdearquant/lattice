//! Trie-accelerated vocabulary masking for grammar states outside the
//! precomputed `VocabPartition` (issue #734).
//!
//! # Motivation
//!
//! `GrammarEngine::mask_by_simulation` computes the mask for a runtime state
//! that has no precomputed bitmask by running `simulate_token` independently
//! for every vocabulary token: O(vocab_size × token_len) byte-walks, almost
//! all of which reject on the very first byte. A byte trie built once over
//! `vocab_bytes` lets the PDA walk the *shared prefix structure* of the
//! vocabulary instead: a single rejected byte transition at a trie node
//! prunes every token beneath it in one step, since they all share that
//! prefix and would reject identically.
//!
//! # Mask contract
//!
//! [`ByteTrie::mask`] reproduces `mask_by_simulation`'s contract bit for bit:
//! a token is allowed iff `simulate_token(state, grammar, token_bytes)`
//! would return `SimResult::Accept` — i.e. every byte of the token is
//! accepted by the PDA in order, with no rejection at any position. Both
//! `SimResult::Reject` (first-byte rejection) and `SimResult::ContextDependent`
//! (rejection after a partial prefix) block the token identically: a DFS
//! walk that fails to reach a token's terminal trie node covers both cases
//! without needing to distinguish them.
//!
//! # DFS state handling
//!
//! Trie construction is independent of any grammar (it depends only on
//! `vocab_bytes`), so the same trie is reused across every over-cap
//! `mask_logits` call for a given engine. Masking DFS-walks the trie
//! carrying a live `GrammarState`, calling `advance_byte` once per trie
//! edge:
//!
//! - A rejected edge prunes the whole subtree below it (no recursion).
//! - An accepted edge whose destination node has terminal token ids marks
//!   those tokens allowed (the path from root to that node — and therefore
//!   every byte of that token — was accepted).
//!
//! `GrammarState` clones are O(stack depth) (pda.rs), typically 2-8 frames,
//! but a naive "clone before every child" DFS would pay that cost at every
//! trie edge, including long straight-line runs (nodes with exactly one
//! child) where no other branch needs the pre-advance state. The DFS here
//! clones only when a node has more than one child (clone for every child
//! but the last; the last child reuses the caller's owned state by move),
//! so straight-line chains — the common case for byte-level BPE tokens
//! after the first few branching bytes — advance in place with zero clones.
//!
//! That O(stack depth) bound depends on the walk starting from a state
//! whose `partial_token_bytes` is empty, not a clone of the live decode
//! state's. `advance_byte` unconditionally appends to that field, and the
//! live state passed into `mask` has one entry per byte generated so far
//! in the whole decode — cloning it as the walk root would make every
//! DFS clone O(stack depth + bytes generated so far) instead. See
//! [`walk_root_state`] for the fix.

use crate::grammar::pda::{CompiledGrammar, GrammarState, StepResult, advance_byte};

// Reused across every `ByteTrie::mask` call on this thread so the hot
// over-cap path doesn't heap-allocate a fresh `vocab_size / 64`-word bitvec
// every time (issue #734 follow-up: this buffer was previously a
// `vec![0u64; stride]` per call, ~31KB of allocator traffic at 248K-token
// vocab). `ByteTrie::mask` takes `&self`, so a `RefCell` field can't hand
// out a mutably-borrowed buffer across calls without risking a
// double-borrow if `mask` were ever re-entered on the same thread; it
// isn't (the DFS never calls back into `mask`), so a plain thread-local
// scratch buffer is sound and avoids that hazard entirely — concurrent
// callers on other threads each get their own buffer, so there's no data
// race to reason about either.
thread_local! {
    static MASK_SCRATCH: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// One node in the byte trie. Children are kept as a small sorted-free
/// association list rather than a fixed 256-entry table: vocab tries are
/// deep and sparse (most nodes have very few children), so a `Vec` avoids
/// paying 256 pointers per node.
struct TrieNode {
    /// `(byte, child_node_index)`, insertion order (not sorted — node
    /// fan-out is small enough that a linear scan beats maintaining order).
    children: Vec<(u8, u32)>,
    /// Token ids whose full byte sequence ends exactly at this node.
    /// Almost always empty or length 1; multiple ids only when the
    /// vocabulary contains duplicate byte sequences under different ids.
    terminals: Vec<u32>,
}

/// A byte trie over a model vocabulary, built once and reused for every
/// masking call against that vocabulary.
pub struct ByteTrie {
    /// Arena of nodes; index 0 is always the root (zero bytes consumed).
    nodes: Vec<TrieNode>,
}

impl ByteTrie {
    /// Build a trie over `vocab_bytes`. Empty-byte tokens are not inserted
    /// (they are unconditionally blocked, matching `mask_by_simulation`'s
    /// explicit empty-token handling) — they simply never get a set mask
    /// bit, which is the correct "blocked" default.
    pub fn build(vocab_bytes: &[Vec<u8>]) -> Self {
        let mut nodes = vec![TrieNode {
            children: Vec::new(),
            terminals: Vec::new(),
        }];

        for (token_id, bytes) in vocab_bytes.iter().enumerate() {
            if bytes.is_empty() {
                continue;
            }
            let mut cur = 0u32;
            for &b in bytes {
                cur = match nodes[cur as usize]
                    .children
                    .iter()
                    .find(|&&(cb, _)| cb == b)
                {
                    Some(&(_, child)) => child,
                    None => {
                        let new_idx = nodes.len() as u32;
                        nodes.push(TrieNode {
                            children: Vec::new(),
                            terminals: Vec::new(),
                        });
                        nodes[cur as usize].children.push((b, new_idx));
                        new_idx
                    }
                };
            }
            nodes[cur as usize].terminals.push(token_id as u32);
        }

        Self { nodes }
    }

    /// Compute the grammar mask for `state` by DFS-walking the trie, and
    /// apply it to `logits` in place (same contract as
    /// `GrammarEngine::mask_by_simulation`: sets disallowed positions to
    /// `f32::NEG_INFINITY`, leaves allowed positions' original values
    /// untouched).
    pub fn mask(
        &self,
        state: &GrammarState,
        grammar: &CompiledGrammar,
        vocab_size: usize,
        logits: &mut [f32],
    ) {
        let mask_stride = vocab_size.div_ceil(64);
        MASK_SCRATCH.with(|scratch| {
            let mut allowed = scratch.borrow_mut();
            allowed.clear();
            allowed.resize(mask_stride, 0u64);
            mark_allowed(
                &self.nodes,
                0,
                walk_root_state(state),
                grammar,
                &mut allowed,
            );
            apply_allowed_mask(&allowed, vocab_size, logits);
        });
    }
}

/// Build the DFS walk's root state from the live decode-step `state`.
///
/// Carries a clone of `stack` and `complete` (the only fields `advance_byte`
/// and `is_accepting` ever read) but starts `partial_token_bytes` empty
/// instead of cloning the live state's history. `advance_byte` (pda.rs)
/// unconditionally *appends* to `partial_token_bytes` and nothing in PDA
/// matching ever reads it — it exists only for external context-dependent
/// inspection of a state returned to a caller, which the trie walk's
/// throwaway internal states are not. Cloning the live field here would
/// make every non-last-child DFS clone (`mark_allowed`) pay for the entire
/// generated-so-far byte history instead of O(stack depth), since
/// `partial_token_bytes` grows once per accepted byte for the whole
/// decode, not just within one trie walk.
fn walk_root_state(state: &GrammarState) -> GrammarState {
    GrammarState {
        stack: state.stack.clone(),
        partial_token_bytes: Vec::new(),
        complete: state.complete,
    }
}

/// DFS from `node_idx` carrying an owned live `state`. Marks every
/// terminal token reachable via an all-accepted byte path as allowed in
/// `allowed`.
fn mark_allowed(
    nodes: &[TrieNode],
    node_idx: u32,
    state: GrammarState,
    grammar: &CompiledGrammar,
    allowed: &mut [u64],
) {
    let node = &nodes[node_idx as usize];
    for &token_id in &node.terminals {
        let idx = token_id as usize;
        allowed[idx / 64] |= 1u64 << (idx % 64);
    }

    let children = &node.children;
    let Some((&last, rest)) = children.split_last() else {
        return;
    };

    // Every non-last child needs its own state: cloning here (rather than
    // unconditionally on every recursive call) is what keeps straight-line
    // single-child runs — the common case past the first few bytes of a
    // token — clone-free.
    for &(byte, child_idx) in rest {
        let mut child_state = state.clone();
        if advance_byte(&mut child_state, grammar, byte) == StepResult::Accepted {
            mark_allowed(nodes, child_idx, child_state, grammar, allowed);
        }
    }

    // Last child reuses (moves) the caller's owned state — no clone.
    let (byte, child_idx) = last;
    let mut last_state = state;
    if advance_byte(&mut last_state, grammar, byte) == StepResult::Accepted {
        mark_allowed(nodes, child_idx, last_state, grammar, allowed);
    }
}

/// Apply a bitmask (1 = allowed) to `logits`, blocking every position whose
/// bit is unset. Word-scanning structure mirrors
/// `VocabPartition::apply_mask` (skip all-allowed words, fast-fill
/// all-blocked words).
fn apply_allowed_mask(allowed: &[u64], vocab_size: usize, logits: &mut [f32]) {
    let mask_stride = allowed.len();
    for word_idx in 0..mask_stride {
        let word = allowed[word_idx];
        let base_token = word_idx * 64;
        if word == u64::MAX {
            continue;
        }
        if word == 0 {
            let end = (base_token + 64).min(vocab_size);
            for l in logits[base_token..end].iter_mut() {
                *l = f32::NEG_INFINITY;
            }
            continue;
        }
        for bit in 0..64u32 {
            let token_idx = base_token + bit as usize;
            if token_idx >= vocab_size {
                break;
            }
            if word & (1u64 << bit) == 0 {
                logits[token_idx] = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::pda::{GrammarBuilder, Symbol};

    /// Grammar: root = 'a' | 'b'
    fn or_grammar() -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        b.add_rule(
            "root",
            vec![vec![Symbol::Terminal(b'a')], vec![Symbol::Terminal(b'b')]],
        );
        b.build()
    }

    #[test]
    fn trie_build_shares_prefixes() {
        let vocab = vec![b"ab".to_vec(), b"ac".to_vec(), b"b".to_vec()];
        let trie = ByteTrie::build(&vocab);
        // root -> 'a' -> {'b','c'} ; root -> 'b'. Node count: root, a, ab,
        // ac, b(root's) = 5.
        assert_eq!(trie.nodes.len(), 5);
    }

    #[test]
    fn trie_mask_matches_simple_grammar() {
        let vocab = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        let grammar = or_grammar();
        let trie = ByteTrie::build(&vocab);
        let state = GrammarState::initial();
        let mut logits = vec![1.0f32, 2.0f32, 3.0f32];
        trie.mask(&state, &grammar, vocab.len(), &mut logits);
        assert!(logits[0] > f32::NEG_INFINITY, "'a' allowed");
        assert!(logits[1] > f32::NEG_INFINITY, "'b' allowed");
        assert_eq!(logits[2], f32::NEG_INFINITY, "'c' blocked");
    }

    #[test]
    fn trie_mask_empty_token_blocked() {
        let vocab = vec![b"a".to_vec(), vec![]];
        let grammar = or_grammar();
        let trie = ByteTrie::build(&vocab);
        let state = GrammarState::initial();
        let mut logits = vec![1.0f32, 1.0f32];
        trie.mask(&state, &grammar, vocab.len(), &mut logits);
        assert!(logits[0] > f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY, "empty token always blocked");
    }

    /// Regression test for the over-cap trie-mask hot path (issue #734
    /// follow-up): `mark_allowed`'s DFS clones must never carry the live
    /// decode state's `partial_token_bytes` history. `advance_byte`
    /// (pda.rs) appends to that field on every accepted byte for the
    /// *whole* decode, and nothing in PDA matching (`try_advance_stack`,
    /// `is_accepting`) ever reads it back. If the walk root cloned it,
    /// every non-last-child DFS clone in `mark_allowed` would copy the
    /// entire generated-so-far history instead of paying only for the
    /// stack.
    ///
    /// Mutation-sensitive: reverting `walk_root_state` to `state.clone()`
    /// makes this fail immediately, since `root.partial_token_bytes` would
    /// equal the 64KB history instead of being empty. Verified by hand:
    /// temporarily replacing the body with `state.clone()` and re-running
    /// this test fails on the `is_empty()` assertion; restoring the fix
    /// makes it pass again.
    #[test]
    fn trie_walk_root_state_drops_partial_token_bytes() {
        let mut state = GrammarState::initial();
        // Representative of a long-running decode: `partial_token_bytes`
        // grows by one entry per accepted byte across the whole
        // generation, not just within one trie walk.
        state.partial_token_bytes = vec![b'x'; 64 * 1024];

        let root = walk_root_state(&state);

        assert!(
            root.partial_token_bytes.is_empty(),
            "walk root must start with empty partial_token_bytes regardless \
             of the live state's history length (got {} bytes) — a full \
             state.clone() here reintroduces O(generation-so-far) DFS clones",
            root.partial_token_bytes.len()
        );
        assert_eq!(root.stack, state.stack, "walk root must preserve stack");
        assert_eq!(
            root.complete, state.complete,
            "walk root must preserve the complete flag"
        );
    }

    /// End-to-end companion to the unit test above: the mask output must
    /// not depend on how much byte history the live state carries in
    /// `partial_token_bytes`, since the trie walk never reads that field
    /// through a cloned live state.
    #[test]
    fn trie_mask_byte_identical_regardless_of_partial_token_bytes_history() {
        let vocab = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        let grammar = or_grammar();
        let trie = ByteTrie::build(&vocab);

        let clean_state = GrammarState::initial();
        let mut clean_logits = vec![1.0f32, 2.0f32, 3.0f32];
        trie.mask(&clean_state, &grammar, vocab.len(), &mut clean_logits);

        let mut heavy_state = GrammarState::initial();
        heavy_state.partial_token_bytes = vec![b'x'; 64 * 1024];
        let mut heavy_logits = vec![1.0f32, 2.0f32, 3.0f32];
        trie.mask(&heavy_state, &grammar, vocab.len(), &mut heavy_logits);

        assert_eq!(
            clean_logits, heavy_logits,
            "mask output must be independent of partial_token_bytes history"
        );
    }
}

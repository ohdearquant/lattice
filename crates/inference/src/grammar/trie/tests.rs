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
    trie.mask(&state, &grammar, vocab.len(), &mut logits)
        .expect("fixture stays inside the stack limits");
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
    trie.mask(&state, &grammar, vocab.len(), &mut logits)
        .expect("fixture stays inside the stack limits");
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
    assert_eq!(root.stacks, state.stacks, "walk root must preserve stacks");
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
    trie.mask(&clean_state, &grammar, vocab.len(), &mut clean_logits)
        .expect("fixture stays inside the stack limits");

    let mut heavy_state = GrammarState::initial();
    heavy_state.partial_token_bytes = vec![b'x'; 64 * 1024];
    let mut heavy_logits = vec![1.0f32, 2.0f32, 3.0f32];
    trie.mask(&heavy_state, &grammar, vocab.len(), &mut heavy_logits)
        .expect("fixture stays inside the stack limits");

    assert_eq!(
        clean_logits, heavy_logits,
        "mask output must be independent of partial_token_bytes history"
    );
}

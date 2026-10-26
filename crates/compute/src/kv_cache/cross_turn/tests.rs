use super::*;

fn metadata() -> CrossTurnPrefixMetadata {
    CrossTurnPrefixMetadata {
        model_fingerprint: 1,
        tokenizer_fingerprint: 2,
        adapter_id: AdapterId::BASE,
        vocab_size: 1000,
        max_cache_len: 4096,
        kv_f16: false,
        rope_theta_bits: 0,
        partial_rotary_factor_bits: None,
        layer_pattern_hash: 42,
        chat_template_version: 1,
    }
}

fn entry(
    token_ids: Vec<u32>,
    represented_len: usize,
    gdn_snapshot_len: usize,
) -> CrossTurnPrefixEntry {
    CrossTurnPrefixEntry {
        slot_id: CrossTurnSlotId::DEFAULT,
        metadata: metadata(),
        token_ids,
        represented_len,
        kv: KvPrefixHandle {
            represented_len,
            num_full_attention_layers: 6,
            kv_dim: 512,
            max_cache_len: 4096,
            kv_f16: false,
        },
        gdn_snapshot_len,
    }
}

#[test]
fn longest_common_prefix_full_hit() {
    assert_eq!(longest_common_token_prefix(&[1, 2, 3], &[1, 2, 3]), 3);
}

#[test]
fn longest_common_prefix_empty_hit() {
    assert_eq!(longest_common_token_prefix(&[], &[1, 2, 3]), 0);
    assert_eq!(longest_common_token_prefix(&[1, 2, 3], &[]), 0);
    assert_eq!(longest_common_token_prefix(&[9], &[1, 2, 3]), 0);
}

#[test]
fn longest_common_prefix_first_divergence() {
    assert_eq!(longest_common_token_prefix(&[1, 2, 3], &[1, 2, 9]), 2);
    assert_eq!(longest_common_token_prefix(&[1, 9, 3], &[1, 2, 3]), 1);
}

#[test]
fn longest_common_prefix_shorter_old() {
    assert_eq!(longest_common_token_prefix(&[1, 2], &[1, 2, 3, 4]), 2);
}

#[test]
fn longest_common_prefix_shorter_new() {
    assert_eq!(longest_common_token_prefix(&[1, 2, 3, 4], &[1, 2]), 2);
}

#[test]
fn plan_no_entry_is_full_refill() {
    let plan = plan_prefix_reuse(None, &metadata(), &[1, 2, 3], &[]);
    assert_eq!(plan.mode, PrefixReuseMode::FullRefill);
    assert_eq!(plan.suffix_start, 0);
    assert_eq!(plan.suffix_len, 3);
}

#[test]
fn plan_metadata_mismatch_is_full_refill() {
    let e = entry(vec![1, 2, 3], 3, 3);
    let mut other = metadata();
    other.model_fingerprint = 999;
    let plan = plan_prefix_reuse(Some(&e), &other, &[1, 2, 3, 4], &[]);
    assert_eq!(plan.mode, PrefixReuseMode::FullRefill);
    assert_eq!(plan.reusable_len, 0);
}

#[test]
fn plan_empty_new_prompt_is_full_refill() {
    let e = entry(vec![1, 2, 3], 3, 3);
    let plan = plan_prefix_reuse(Some(&e), &metadata(), &[], &[]);
    assert_eq!(plan.mode, PrefixReuseMode::FullRefill);
}

#[test]
fn plan_zero_shared_prefix_is_full_refill() {
    let e = entry(vec![1, 2, 3], 3, 3);
    let plan = plan_prefix_reuse(Some(&e), &metadata(), &[9, 8, 7], &[]);
    assert_eq!(plan.mode, PrefixReuseMode::FullRefill);
    assert_eq!(plan.shared_token_prefix_len, 0);
}

#[test]
fn plan_exact_append() {
    let e = entry(vec![1, 2, 3], 3, 3);
    let plan = plan_prefix_reuse(Some(&e), &metadata(), &[1, 2, 3, 4, 5], &[]);
    assert_eq!(plan.mode, PrefixReuseMode::ExactAppend);
    assert_eq!(plan.shared_token_prefix_len, 3);
    assert_eq!(plan.reusable_len, 3);
    assert_eq!(plan.suffix_start, 3);
    assert_eq!(plan.suffix_len, 2);
    assert_eq!(plan.old_represented_len, 3);
}

// An exact-equal prompt retry has shared == represented_len but a
// zero-length suffix. Without the `new_prompt_ids.len() > shared` guard
// this produces `ExactAppend` with `suffix_len == 0`, which the Metal
// integration's `forward_prefill_from` rejects as an empty-suffix error
// instead of treating as a valid (degenerate) reuse. The planner must
// fall back to `FullRefill` instead.
//
// Mutation sensitivity: dropping the `new_prompt_ids.len() > shared`
// conjunct makes this test fail, because the plan would again be
// `ExactAppend` with `suffix_len == 0`.
#[test]
fn plan_exact_equal_prompt_is_full_refill() {
    let e = entry(vec![1, 2, 3], 3, 3);
    let plan = plan_prefix_reuse(Some(&e), &metadata(), &[1, 2, 3], &[]);
    assert_eq!(plan.mode, PrefixReuseMode::FullRefill);
    assert_eq!(plan.shared_token_prefix_len, 3);
    assert_eq!(plan.old_represented_len, 3);
    assert_eq!(plan.suffix_len, 3);
}

#[test]
fn plan_exact_append_requires_gdn_snapshot_at_boundary() {
    // GDN snapshot lags represented_len -> not a safe ExactAppend even
    // though the token prefix matches exactly.
    let e = entry(vec![1, 2, 3], 3, 2);
    let plan = plan_prefix_reuse(Some(&e), &metadata(), &[1, 2, 3, 4], &[]);
    assert_eq!(plan.mode, PrefixReuseMode::FullRefill);
}

#[test]
fn plan_mid_history_divergence_without_checkpoint_is_full_refill() {
    // Shared prefix (2) is shorter than represented_len (3): edited history.
    let e = entry(vec![1, 2, 3], 3, 3);
    let plan = plan_prefix_reuse(Some(&e), &metadata(), &[1, 2, 9, 9], &[]);
    assert_eq!(plan.mode, PrefixReuseMode::FullRefill);
    assert_eq!(plan.shared_token_prefix_len, 2);
    assert_eq!(plan.old_represented_len, 3);
}

#[test]
fn plan_mid_history_divergence_shorter_new_prompt_is_full_refill() {
    let e = entry(vec![1, 2, 3, 4, 5], 5, 5);
    let plan = plan_prefix_reuse(Some(&e), &metadata(), &[1, 2, 3], &[]);
    assert_eq!(plan.mode, PrefixReuseMode::FullRefill);
    assert_eq!(plan.shared_token_prefix_len, 3);
}

#[test]
fn plan_sparse_checkpoint_replay_only_when_valid() {
    let e = entry(vec![1, 2, 3, 4, 5], 5, 5);
    // Divergence after token 3; a checkpoint at 3 exists -> replay from 3.
    let plan = plan_prefix_reuse(Some(&e), &metadata(), &[1, 2, 3, 9, 9], &[3]);
    assert_eq!(
        plan.mode,
        PrefixReuseMode::ReplayFromCheckpoint { checkpoint_len: 3 }
    );
    assert_eq!(plan.reusable_len, 3);
    assert_eq!(plan.suffix_start, 3);
    assert_eq!(plan.suffix_len, 2);
}

#[test]
fn plan_sparse_checkpoint_beyond_shared_prefix_is_ignored() {
    let e = entry(vec![1, 2, 3, 4, 5], 5, 5);
    // Checkpoint at 4 is past the shared prefix (3) -> cannot be used.
    let plan = plan_prefix_reuse(Some(&e), &metadata(), &[1, 2, 3, 9, 9], &[4]);
    assert_eq!(plan.mode, PrefixReuseMode::FullRefill);
}

#[test]
fn plan_picks_the_deepest_valid_checkpoint() {
    let e = entry(vec![1, 2, 3, 4, 5], 5, 5);
    let plan = plan_prefix_reuse(Some(&e), &metadata(), &[1, 2, 3, 9, 9], &[1, 3]);
    assert_eq!(
        plan.mode,
        PrefixReuseMode::ReplayFromCheckpoint { checkpoint_len: 3 }
    );
}

// A new prompt that ends exactly on the only retained checkpoint shares
// its whole length with that checkpoint, so replaying from it would
// leave an empty suffix, which `forward_prefill_from` rejects. The
// planner must skip that checkpoint and fall back to `FullRefill`.
//
// Mutation sensitivity: dropping the `len < new_prompt_ids.len()`
// conjunct makes this test fail, because the plan would again be
// `ReplayFromCheckpoint { checkpoint_len: 2 }` with `suffix_len == 0`.
#[test]
fn plan_checkpoint_at_new_prompt_end_is_full_refill() {
    let e = entry(vec![1, 2, 3, 4], 4, 4);
    let plan = plan_prefix_reuse(Some(&e), &metadata(), &[1, 2], &[2]);
    assert_eq!(plan.mode, PrefixReuseMode::FullRefill);
    assert_eq!(plan.shared_token_prefix_len, 2);
    assert_eq!(plan.reusable_len, 0);
    assert_eq!(plan.suffix_start, 0);
    assert_eq!(plan.suffix_len, 2);
}

// With a shallower checkpoint also retained, the one at the new prompt's
// end is skipped and the deepest checkpoint strictly inside the prompt
// wins, leaving a non-empty suffix.
#[test]
fn plan_checkpoint_at_new_prompt_end_uses_shallower_checkpoint() {
    let e = entry(vec![1, 2, 3, 4], 4, 4);
    let plan = plan_prefix_reuse(Some(&e), &metadata(), &[1, 2], &[1, 2]);
    assert_eq!(
        plan.mode,
        PrefixReuseMode::ReplayFromCheckpoint { checkpoint_len: 1 }
    );
    assert_eq!(plan.shared_token_prefix_len, 2);
    assert_eq!(plan.reusable_len, 1);
    assert_eq!(plan.suffix_start, 1);
    assert_eq!(plan.suffix_len, 1);
}

// #590: the survival predicate is the single source of truth for which
// ring checkpoints carry over across a save. Mutation sensitivity: each
// case below fails if one of the three conjuncts is dropped.
#[test]
fn checkpoint_survival_requires_all_three_conditions() {
    // Survives: positive, within the shared prefix, below the new boundary.
    assert!(checkpoint_survives_save(3, 5, 8));
    // Boundary-inclusive on the shared prefix.
    assert!(checkpoint_survives_save(5, 5, 8));
    // Zero-length boundary never survives.
    assert!(!checkpoint_survives_save(0, 5, 8));
    // Past the divergence point: KV rows above lcp were overwritten.
    assert!(!checkpoint_survives_save(6, 5, 8));
    // Equal to the new boundary: duplicate of the entry's own snapshot.
    assert!(!checkpoint_survives_save(8, 8, 8));
    // Degenerate save at len 0 retains nothing.
    assert!(!checkpoint_survives_save(1, 5, 0));
}

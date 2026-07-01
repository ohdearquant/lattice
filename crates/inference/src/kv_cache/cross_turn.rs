//! Cross-turn KV prefix cache planning (#462).
//!
//! Pure token-level planning for reusing a previous turn's KV/GDN state
//! across chat turns. This module owns no GPU buffers and no live model
//! state — it only decides, from token IDs and metadata, whether a cached
//! prefix can be reused and where the new suffix begins. The Metal-specific
//! runtime (live KV buffers, GDN snapshots) lives in
//! `crate::forward::metal_qwen35`.
//!
//! See design.md step 4 (ADR: latest-boundary GDN snapshot + live KV logical
//! reuse for v1). Reuse is only ever claimed for an exact token prefix that
//! is already fully represented by retained state; any divergence falls
//! back to `PrefixReuseMode::FullRefill`.

use crate::kv_cache::AdapterId;

/// Identifies one cross-turn cache slot. Distinct slots never share state,
/// so distinct conversations (or clients) using distinct slot IDs cannot
/// read one another's KV/GDN state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CrossTurnSlotId(pub u64);

impl CrossTurnSlotId {
    /// The single-client / local-use default slot.
    pub const DEFAULT: Self = Self(0);

    /// Build a slot ID from a caller-supplied value (e.g. a hashed session key).
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
}

/// Everything that must match between the cached entry and the current
/// request for reuse to be considered. Any field mismatch invalidates the
/// entry — the cached KV/GDN state is not state-equivalent to a re-prefill
/// under a different model, tokenizer, adapter, RoPE, or KV layout.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrossTurnPrefixMetadata {
    pub model_fingerprint: u64,
    pub tokenizer_fingerprint: u64,
    pub adapter_id: AdapterId,
    pub vocab_size: usize,
    pub max_cache_len: usize,
    pub kv_f16: bool,
    pub rope_theta_bits: u64,
    pub partial_rotary_factor_bits: Option<u32>,
    pub layer_pattern_hash: u64,
    pub chat_template_version: u32,
}

/// Logical handle to the live full-attention KV buffers backing a cached
/// prefix. Does not own the buffers — the Metal state keeps them in place
/// and only truncates/advances the logical `seq_len`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPrefixHandle {
    pub represented_len: usize,
    pub num_full_attention_layers: usize,
    pub kv_dim: usize,
    pub max_cache_len: usize,
    pub kv_f16: bool,
}

/// One retained cross-turn prefix: the exact tokens, and the length to
/// which model state (KV + GDN) is known to represent them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossTurnPrefixEntry {
    pub slot_id: CrossTurnSlotId,
    pub metadata: CrossTurnPrefixMetadata,
    pub token_ids: Vec<u32>,
    pub represented_len: usize,
    pub kv: KvPrefixHandle,
    pub gdn_snapshot_len: usize,
}

/// How the next turn should build on (or discard) the cached entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixReuseMode {
    /// No usable overlap: reset state and prefill the whole prompt.
    FullRefill,
    /// The cached entry is an exact prefix of the new prompt and its GDN
    /// snapshot is taken at that same boundary — reuse it verbatim.
    ExactAppend,
    /// Reuse only up to an earlier exact GDN checkpoint, then replay/prefill
    /// forward. v2: no v1 caller currently supplies checkpoints, so this
    /// variant is never produced by the current Metal integration.
    ReplayFromCheckpoint { checkpoint_len: usize },
}

/// The result of planning: what to restore, and where the new suffix starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixRestorePlan {
    pub mode: PrefixReuseMode,
    pub shared_token_prefix_len: usize,
    pub reusable_len: usize,
    pub suffix_start: usize,
    pub suffix_len: usize,
    pub old_represented_len: usize,
}

/// Length of the longest common prefix of two token ID slices.
pub fn longest_common_token_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Decide how (or whether) to reuse `entry` for `new_prompt_ids`.
///
/// See the module-level rules: reuse is only claimed for an exact prefix
/// match against fully represented state (`ExactAppend`), or against an
/// exact earlier GDN checkpoint boundary present in `sparse_checkpoint_lens`
/// (`ReplayFromCheckpoint`). Everything else falls back to `FullRefill` —
/// decline-beats-fabricate for the token-identity invariant.
///
/// `ExactAppend` additionally requires a non-empty suffix
/// (`new_prompt_ids.len() > shared`, #516 round-1 remediation D5): an
/// exact-equal retry of the represented prompt has no divergent suffix to
/// prefill, and the Metal integration's `forward_prefill_from` treats an
/// empty suffix as an internal invariant violation, not a valid no-op. That
/// case falls back to `FullRefill` here rather than surfacing as an error.
pub fn plan_prefix_reuse(
    entry: Option<&CrossTurnPrefixEntry>,
    metadata: &CrossTurnPrefixMetadata,
    new_prompt_ids: &[u32],
    sparse_checkpoint_lens: &[usize],
) -> PrefixRestorePlan {
    let full_refill = |shared: usize, old_represented_len: usize| PrefixRestorePlan {
        mode: PrefixReuseMode::FullRefill,
        shared_token_prefix_len: shared,
        reusable_len: 0,
        suffix_start: 0,
        suffix_len: new_prompt_ids.len(),
        old_represented_len,
    };

    let Some(entry) = entry else {
        return full_refill(0, 0);
    };

    if &entry.metadata != metadata || new_prompt_ids.is_empty() {
        return full_refill(0, entry.represented_len);
    }

    let shared = longest_common_token_prefix(&entry.token_ids, new_prompt_ids);
    if shared == 0 {
        return full_refill(0, entry.represented_len);
    }

    if shared == entry.represented_len
        && entry.gdn_snapshot_len == entry.represented_len
        && new_prompt_ids.len() > shared
    {
        return PrefixRestorePlan {
            mode: PrefixReuseMode::ExactAppend,
            shared_token_prefix_len: shared,
            reusable_len: shared,
            suffix_start: shared,
            suffix_len: new_prompt_ids.len() - shared,
            old_represented_len: entry.represented_len,
        };
    }

    // Mid-history reuse: only valid at an exact, already-owned checkpoint
    // boundary at or below the shared prefix length.
    if let Some(&checkpoint_len) = sparse_checkpoint_lens
        .iter()
        .filter(|&&len| len > 0 && len <= shared)
        .max()
    {
        return PrefixRestorePlan {
            mode: PrefixReuseMode::ReplayFromCheckpoint { checkpoint_len },
            shared_token_prefix_len: shared,
            reusable_len: checkpoint_len,
            suffix_start: checkpoint_len,
            suffix_len: new_prompt_ids.len() - checkpoint_len,
            old_represented_len: entry.represented_len,
        };
    }

    full_refill(shared, entry.represented_len)
}

#[cfg(test)]
mod tests {
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

    // #516 round-1 remediation D5 (finding 4): an exact-equal prompt retry
    // has shared == represented_len but a zero-length suffix. Before D5 this
    // produced `ExactAppend` with `suffix_len == 0`, which the Metal
    // integration's `forward_prefill_from` rejects as an empty-suffix error
    // instead of treating as a valid (degenerate) reuse. The planner must
    // fall back to `FullRefill` instead.
    //
    // Mutation sensitivity: dropping the `new_prompt_ids.len() > shared`
    // conjunct (reverting to the pre-D5 condition) makes this test fail,
    // because the plan would again be `ExactAppend` with `suffix_len == 0`.
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
}

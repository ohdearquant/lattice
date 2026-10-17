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
//! reuse for v1). Three reuse modes exist: an exact token prefix that is
//! already fully represented by retained state plans
//! `PrefixReuseMode::ExactAppend`; a mid-history divergence at or after a
//! retained sparse GDN checkpoint boundary plans
//! `PrefixReuseMode::ReplayFromCheckpoint` (#590, deepest valid boundary
//! wins); divergence with no usable checkpoint falls back to
//! `PrefixReuseMode::FullRefill`.

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
#[non_exhaustive]
pub enum PrefixReuseMode {
    /// No usable overlap: reset state and prefill the whole prompt.
    FullRefill,
    /// The cached entry is an exact prefix of the new prompt and its GDN
    /// snapshot is taken at that same boundary — reuse it verbatim.
    ExactAppend,
    /// Reuse only up to an earlier exact GDN checkpoint, then replay/prefill
    /// forward. Produced when the Metal integration retains sparse GDN
    /// checkpoints at prior turn boundaries (#590) and the new prompt
    /// diverges mid-history at or after one of them.
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
/// Both reuse modes additionally require a non-empty suffix. `ExactAppend`
/// requires `new_prompt_ids.len() > shared`: an exact-equal retry of the
/// represented prompt has no divergent suffix to prefill.
/// `ReplayFromCheckpoint` only considers checkpoints with
/// `checkpoint_len < new_prompt_ids.len()`: a new prompt that ends exactly
/// on a checkpoint boundary would otherwise replay an empty suffix, so the
/// deepest checkpoint strictly inside the new prompt wins instead. The Metal
/// integration's `forward_prefill_from` treats an empty suffix as an
/// internal invariant violation, not a valid no-op, so when no reuse mode
/// leaves a non-empty suffix the plan falls back to `FullRefill` here rather
/// than surfacing as an error.
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
    // boundary at or below the shared prefix length, and strictly inside the
    // new prompt so the replayed suffix is non-empty.
    if let Some(&checkpoint_len) = sparse_checkpoint_lens
        .iter()
        .filter(|&&len| len > 0 && len <= shared && len < new_prompt_ids.len())
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

/// Whether a retained sparse checkpoint at `len` remains valid when the
/// slot's entry is replaced by a save at `new_represented_len` whose token
/// history shares `common_prefix_len` tokens with the old entry's history
/// (#590).
///
/// A checkpoint survives iff:
/// - `len > 0` — a zero-length boundary is just `FullRefill`, never worth
///   retaining a snapshot for;
/// - `len <= common_prefix_len` — the live KV rows below `len` still
///   represent exactly the tokens the checkpoint's GDN snapshot was taken
///   against (rows at or above the divergence point were overwritten by
///   the new suffix prefill);
/// - `len < new_represented_len` — the new entry's own boundary snapshot
///   already covers `len == new_represented_len`; a duplicate would waste
///   a ring slot.
pub fn checkpoint_survives_save(
    len: usize,
    common_prefix_len: usize,
    new_represented_len: usize,
) -> bool {
    len > 0 && len <= common_prefix_len && len < new_represented_len
}

#[cfg(test)]
mod tests;

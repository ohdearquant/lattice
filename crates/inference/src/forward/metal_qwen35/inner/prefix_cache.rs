use super::*;

// ---------------------------------------------------------------------------
// Cross-turn KV prefix cache (#462)
// ---------------------------------------------------------------------------

/// Per-call statistics for a cache-aware generate/chat call.
#[derive(Debug, Clone)]
pub struct CrossTurnCacheStats {
    pub slot_id: crate::kv_cache::CrossTurnSlotId,
    pub prompt_tokens: usize,
    pub reused_tokens: usize,
    pub prefetched_tokens: usize,
    pub mode: crate::kv_cache::PrefixReuseMode,
}

/// [`GenerateOutput`] plus cross-turn cache stats for the call that produced it.
#[derive(Debug, Clone)]
pub struct CachedGenerateOutput {
    pub output: GenerateOutput,
    pub cache: CrossTurnCacheStats,
}

/// [`ChatCompletionOutput`] plus cross-turn cache stats for the call that produced it.
#[derive(Debug, Clone)]
pub struct CachedChatCompletionOutput {
    pub output: ChatCompletionOutput,
    pub cache: CrossTurnCacheStats,
}

/// One sparse GDN checkpoint retained at an earlier turn boundary
/// (#590): the token length it represents plus the exact GDN
/// recurrent-state snapshot taken at that boundary. Valid only while
/// the live KV rows below `len` still hold the prefix the snapshot was
/// taken against — the same lifetime as the owning entry, enforced by
/// storing checkpoints inside `MetalCrossTurnPrefixEntry` and pruning
/// on every save via `checkpoint_survives_save`.
#[derive(Debug, Clone)]
pub(super) struct MetalGdnCheckpoint {
    pub(super) len: usize,
    pub(super) snapshot: crate::attention::gdn::GdnSnapshot,
}

/// Maximum sparse checkpoints retained per entry (#590). Each GDN
/// snapshot is ~19MiB on Qwen3.5-0.8B (#516's measurement), so the
/// worst-case cross-turn memory is `CAP + 1` snapshots (~76MiB): the
/// ring plus the entry's own boundary snapshot. Eviction keeps the
/// DEEPEST boundaries — in append-only chat flow boundaries grow
/// monotonically, so this equals keeping the most recent turns; a
/// deeper checkpoint also saves the most re-prefill work when a late
/// turn is edited or regenerated. The cost of an evicted checkpoint is
/// only a longer replay prefill, never wrong output.
const CROSS_TURN_GDN_CHECKPOINT_CAP: usize = 3;

/// One retained cross-turn entry: the generic (Metal-agnostic) planning
/// record plus the exact GDN recurrent-state snapshot taken at the same
/// token boundary. The live full-attention KV buffers are NOT copied here
/// — they stay in `self.session.kv_cache` and are only logically
/// truncated/advanced (see `restore_cross_turn_prefix`).
#[derive(Debug, Clone)]
pub(super) struct MetalCrossTurnPrefixEntry {
    pub(super) generic: crate::kv_cache::CrossTurnPrefixEntry,
    pub(super) gdn_snapshot: crate::attention::gdn::GdnSnapshot,
    /// Sparse GDN checkpoints at earlier turn boundaries, ascending by
    /// `len`, at most [`CROSS_TURN_GDN_CHECKPOINT_CAP`] (#590). These
    /// enable `PrefixReuseMode::ReplayFromCheckpoint` when a new prompt
    /// diverges mid-history instead of falling all the way back to
    /// `FullRefill`.
    pub(super) checkpoints: Vec<MetalGdnCheckpoint>,
}

/// Worker-local, single-live-entry cross-turn prefix cache (#516).
///
/// Ownership invariant: at most ONE `MetalCrossTurnPrefixEntry` is ever
/// retained, bounded by design (never an unbounded per-slot map). The
/// entry is valid **iff** the live `self.session.kv_cache` rows and
/// `self.session.position` were left by exactly the generation call that
/// saved it. Any other mutation of live KV/GDN state — a different
/// slot's generation, the plain (non-cache-aware) generate path, a
/// `reset_state()` call, or a LoRA adapter load/unload — must invalidate
/// the entry before that mutation is allowed to proceed, because the
/// entry does not own private copies of the full-attention KV buffers
/// (see `MetalCrossTurnPrefixEntry` above): it can only ever describe
/// the ONE live buffer this process currently has. `get`/`take` both
/// enforce the slot match themselves so a caller can never accidentally
/// restore a different slot's entry.
#[derive(Debug, Default)]
pub(super) struct MetalCrossTurnPrefixCache {
    pub(super) entry: Option<MetalCrossTurnPrefixEntry>,
}

impl MetalCrossTurnPrefixCache {
    /// Returns the retained entry only if it belongs to `slot_id`. A
    /// different (or absent) slot is always a miss — never a partial or
    /// stale match — so the planner falls back to `FullRefill`.
    pub(super) fn get(
        &self,
        slot_id: crate::kv_cache::CrossTurnSlotId,
    ) -> Option<&MetalCrossTurnPrefixEntry> {
        self.entry.as_ref().filter(|e| e.generic.slot_id == slot_id)
    }

    /// Replace whatever is currently retained (if anything, for any
    /// slot) with `entry`. At most one GDN snapshot is ever alive.
    fn insert(&mut self, entry: MetalCrossTurnPrefixEntry) {
        self.entry = Some(entry);
    }

    /// Remove and return the retained entry iff it belongs to
    /// `slot_id`, leaving the cache empty either way it existed for
    /// that slot. Used by `restore_cross_turn_prefix` so an
    /// `ExactAppend` restore consumes its GDN snapshot instead of
    /// cloning it, and so the cache is provably empty for the
    /// remainder of the call until a fresh save succeeds.
    fn take(
        &mut self,
        slot_id: crate::kv_cache::CrossTurnSlotId,
    ) -> Option<MetalCrossTurnPrefixEntry> {
        if self
            .entry
            .as_ref()
            .is_some_and(|e| e.generic.slot_id == slot_id)
        {
            self.entry.take()
        } else {
            None
        }
    }

    pub(super) fn remove(&mut self, slot_id: crate::kv_cache::CrossTurnSlotId) {
        if self
            .entry
            .as_ref()
            .is_some_and(|e| e.generic.slot_id == slot_id)
        {
            self.entry = None;
        }
    }

    pub(super) fn clear(&mut self) {
        self.entry = None;
    }
}

impl MetalQwen35State {
    /// Drop all retained cross-turn cache slots. Call this after any
    /// error surfaces from a `*_with_prefix_cache` entry point (or from
    /// the caller's own error handling) so a later turn cannot restore
    /// from state that may no longer match the live KV/GDN buffers.
    pub fn clear_cross_turn_prefix_cache(&mut self) {
        self.cross_turn_prefix_cache.clear();
    }

    /// Everything that invalidates cross-turn reuse if it changes
    /// between calls: model identity, tokenizer, adapter, RoPE/KV
    /// layout, and chat template. Coarse-grained on purpose — a false
    /// invalidation only costs a full re-prefill, never wrong output.
    pub(super) fn cross_turn_metadata(
        &self,
        tokenizer: &BpeTokenizer,
    ) -> crate::kv_cache::CrossTurnPrefixMetadata {
        use rustc_hash::FxHasher;
        use std::hash::{Hash, Hasher};

        let cfg = &self.engine.config;

        let mut layer_hasher = FxHasher::default();
        for lt in &cfg.layer_types {
            (*lt as u8).hash(&mut layer_hasher);
        }
        let layer_pattern_hash = layer_hasher.finish();

        let mut model_hasher = FxHasher::default();
        cfg.hidden_size.hash(&mut model_hasher);
        cfg.num_hidden_layers.hash(&mut model_hasher);
        cfg.vocab_size.hash(&mut model_hasher);
        cfg.rope_theta.to_bits().hash(&mut model_hasher);
        cfg.mtp_num_hidden_layers.hash(&mut model_hasher);
        layer_pattern_hash.hash(&mut model_hasher);
        let model_fingerprint = model_hasher.finish();

        let mut tok_hasher = FxHasher::default();
        tokenizer.vocab_size().hash(&mut tok_hasher);
        tokenizer
            .special_token_id("<|im_end|>")
            .hash(&mut tok_hasher);
        // Tokenizer identity only -- resolved unconditionally regardless
        // of whether *this* request has an active reasoning budget.
        // reasoning_budget is a generation-time policy choice, not a
        // prefix-defining input: two requests with identical prompts and
        // tokenizer state must hash to the same fingerprint whether or
        // not either one sets a budget, so a turn that happens to omit
        // (or vary) reasoning_budget can still reuse the other's cached
        // prefix instead of forcing FullRefill. The close-marker
        // VALIDATION that a budget is actually enforceable still runs
        // separately, per-request, in `resolve_reasoning_close_token`.
        // Hash EVERY id rendering the marker, not just the first tier hit.
        // Two tokenizers that agree on the winning id but differ in their
        // remaining aliases are not the same tokenizer, and folding only
        // one id in would hash them identically. Sorted because the added
        // -token walk is over a HashMap, and a cache key should not depend
        // on iteration order.
        let mut close_marker_ids = tok::token_ids_for_content(tokenizer, REASONING_CLOSE_MARKER);
        close_marker_ids.sort_unstable();
        close_marker_ids.hash(&mut tok_hasher);
        let tokenizer_fingerprint = tok_hasher.finish();

        let adapter_id = match &self.lora {
            None => crate::kv_cache::AdapterId::BASE,
            Some(adapter) => {
                let mut h = FxHasher::default();
                adapter.max_rank.hash(&mut h);
                adapter.scale.to_bits().hash(&mut h);
                adapter.projections.len().hash(&mut h);
                crate::kv_cache::AdapterId::new(h.finish())
            }
        };

        crate::kv_cache::CrossTurnPrefixMetadata {
            model_fingerprint,
            tokenizer_fingerprint,
            adapter_id,
            vocab_size: cfg.vocab_size,
            max_cache_len: self.session.kv_cache.max_cache_len,
            kv_f16: self.use_kv_f16,
            rope_theta_bits: cfg.rope_theta.to_bits(),
            partial_rotary_factor_bits: Some(cfg.partial_rotary_factor.to_bits()),
            layer_pattern_hash,
            chat_template_version: 1,
        }
    }

    /// Pure planning step: delegates to [`crate::kv_cache::plan_prefix_reuse`]
    /// against whatever is currently cached for `slot_id`, supplying the
    /// entry's retained sparse GDN checkpoint boundaries (#590) so the
    /// planner can produce `PrefixReuseMode::ReplayFromCheckpoint` for
    /// mid-history divergence instead of always falling back to
    /// `FullRefill`.
    pub(super) fn plan_cross_turn_reuse(
        &self,
        slot_id: crate::kv_cache::CrossTurnSlotId,
        metadata: &crate::kv_cache::CrossTurnPrefixMetadata,
        prompt_ids: &[u32],
    ) -> crate::kv_cache::PrefixRestorePlan {
        let entry = self.cross_turn_prefix_cache.get(slot_id);
        let checkpoint_lens: Vec<usize> = entry
            .map(|e| e.checkpoints.iter().map(|c| c.len).collect())
            .unwrap_or_default();
        crate::kv_cache::plan_prefix_reuse(
            entry.map(|e| &e.generic),
            metadata,
            prompt_ids,
            &checkpoint_lens,
        )
    }

    /// Pure preflight over the suffix a [`crate::kv_cache::PrefixRestorePlan`]
    /// selects — every suffix token id within `vocab_size`, the suffix
    /// non-empty, and `plan.suffix_start + suffix.len()` within
    /// `max_cache_len`. Must run BEFORE `restore_cross_turn_prefix`/
    /// `reset_state` in `generate_streaming_with_prefix_cache_and_cancel_inner`
    /// (#835): those calls `take()` the cache slot's entry (ExactAppend /
    /// ReplayFromCheckpoint) or clear live state (FullRefill), and the
    /// public wrapper's error-recovery arm additionally evicts the slot's
    /// cache entry outright on ANY `Err` from `_inner`. Rejecting an
    /// invalid suffix here, before any of that runs, leaves a warmed slot
    /// untouched instead of destroying it for a request that never
    /// touched live KV/GDN state.
    ///
    /// Mirrors `forward_prefill_from`'s own checks byte-for-byte — same
    /// `InferenceError` variant and message shape — so this is
    /// observationally a pure timing fix, not a new error taxonomy.
    /// `forward_prefill_from`'s checks are left in place unchanged
    /// (defense in depth for its other, non-prefix-cache callers).
    pub(super) fn plan_prefix_request(
        &self,
        prompt_ids: &[u32],
        plan: &crate::kv_cache::PrefixRestorePlan,
    ) -> Result<(), crate::error::InferenceError> {
        use crate::error::InferenceError;

        let vocab = self.engine.config.vocab_size;
        let suffix = &prompt_ids[plan.suffix_start..];
        for (t, &id) in suffix.iter().enumerate() {
            if (id as usize) >= vocab {
                return Err(InferenceError::InvalidInput(format!(
                    "forward_prefill_from: token_ids[{t}]={id} out of range: vocab_size is {vocab}"
                )));
            }
        }
        if suffix.is_empty() {
            return Err(InferenceError::PrefixCache(
                "forward_prefill_from: cannot prefill an empty suffix without a saved \
                     logits source"
                    .into(),
            ));
        }
        if plan.suffix_start + suffix.len() > self.session.kv_cache.max_cache_len {
            return Err(InferenceError::InvalidInput(format!(
                "forward_prefill_from: start_pos {} + len {} exceeds max_cache_len {}",
                plan.suffix_start,
                suffix.len(),
                self.session.kv_cache.max_cache_len
            )));
        }
        Ok(())
    }

    /// Restore live state to the plan's reusable boundary.
    ///
    /// `ExactAppend` and `ReplayFromCheckpoint` (#590) both consume the
    /// cached entry and return it, so the caller can carry its
    /// checkpoint ring (and its own boundary snapshot) into the
    /// end-of-generation save. `FullRefill` is handled by the caller
    /// (`self.reset_state()`) per design.md step 5 and returns `None`.
    pub(super) fn restore_cross_turn_prefix(
        &mut self,
        slot_id: crate::kv_cache::CrossTurnSlotId,
        plan: &crate::kv_cache::PrefixRestorePlan,
    ) -> Result<Option<MetalCrossTurnPrefixEntry>, crate::error::InferenceError> {
        use crate::error::InferenceError;
        use crate::kv_cache::PrefixReuseMode;

        match plan.mode {
            PrefixReuseMode::ExactAppend => {
                // Take (not clone): from this point on the cache is
                // EMPTY for this slot until a fresh save succeeds at
                // the end of generation. Any early return between here
                // and that save (error `?`, caller-interrupt) leaves
                // the cache empty rather than retaining a stale entry
                // — fail-closed by construction (#516). This also
                // removes a ~19MiB GDN
                // snapshot clone per restore.
                let entry = self.cross_turn_prefix_cache.take(slot_id).ok_or_else(|| {
                    InferenceError::PrefixCache(
                        "restore_cross_turn_prefix: ExactAppend plan but no cached entry \
                             for this slot"
                            .into(),
                    )
                })?;
                self.restore_gdn_states_checked(&entry.gdn_snapshot)?;
                self.session.set_position(plan.reusable_len);
                Ok(Some(entry))
            }
            PrefixReuseMode::ReplayFromCheckpoint { checkpoint_len } => {
                // Same take-not-clone fail-closed discipline as
                // ExactAppend. The GDN state is restored from the ring
                // checkpoint at exactly `checkpoint_len`; the live KV
                // rows below that boundary are reused in place (they
                // still hold the checkpoint's prefix — the planner only
                // selects checkpoints at or below the shared token
                // prefix), and the caller prefills the divergent suffix
                // forward from `checkpoint_len` via
                // `forward_prefill_from`, overwriting rows at and above
                // the divergence point.
                let entry = self.cross_turn_prefix_cache.take(slot_id).ok_or_else(|| {
                    InferenceError::PrefixCache(
                        "restore_cross_turn_prefix: ReplayFromCheckpoint plan but no cached \
                             entry for this slot"
                            .into(),
                    )
                })?;
                let checkpoint = entry
                    .checkpoints
                    .iter()
                    .find(|c| c.len == checkpoint_len)
                    .ok_or_else(|| {
                        InferenceError::PrefixCache(format!(
                            "restore_cross_turn_prefix: plan selected checkpoint_len \
                                 {checkpoint_len} but no retained snapshot exists at that \
                                 boundary"
                        ))
                    })?;
                self.restore_gdn_states_checked(&checkpoint.snapshot)?;
                self.session.set_position(checkpoint_len);
                Ok(Some(entry))
            }
            PrefixReuseMode::FullRefill => {
                self.reset_state();
                Ok(None)
            }
        }
    }

    /// Prefill `token_ids` starting at absolute position `start_pos`,
    /// writing KV/RoPE rows and advancing GDN state from that position —
    /// the suffix-prefill primitive the design requires. Unlike
    /// `forward_prefill`/`forward_prefill_impl` (which always start at
    /// position 0, including their `len == 1` fast path), every branch
    /// here honors `start_pos`, so a single reused-prefix token forwards
    /// at its true absolute position.
    pub(crate) fn forward_prefill_from(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        all_positions: bool,
    ) -> Result<Vec<f32>, crate::error::InferenceError> {
        use crate::error::InferenceError;

        let vocab = self.engine.config.vocab_size;
        for (t, &id) in token_ids.iter().enumerate() {
            if (id as usize) >= vocab {
                return Err(InferenceError::InvalidInput(format!(
                    "forward_prefill_from: token_ids[{t}]={id} out of range: vocab_size is {vocab}"
                )));
            }
        }
        if token_ids.is_empty() {
            return Err(InferenceError::PrefixCache(
                "forward_prefill_from: cannot prefill an empty suffix without a saved \
                     logits source"
                    .into(),
            ));
        }
        if token_ids.len() == 1 {
            return self.try_forward_step(token_ids[0], start_pos);
        }
        self.check_forward_range_capacity(start_pos, token_ids.len(), false)?;
        if self.lora.is_some() {
            if all_positions {
                return Err(InferenceError::PrefixCache(
                    "forward_prefill_from: all-position output is not supported with an \
                         active LoRA adapter"
                        .into(),
                ));
            }
            if self.engine.has_moe_layer {
                // The batched schedulers have no MoE schedule; per-token steps do.
                let mut last_logits = Vec::new();
                for (i, &id) in token_ids.iter().enumerate() {
                    last_logits = self.forward_step(id, start_pos + i);
                }
                return Ok(last_logits);
            }
        }
        // Every branch below is batched (the single-token and LoRA-on-MoE fast paths
        // returned above via per-token forward steps, which support MoE), so the
        // MoE rejection applies unconditionally from here (#1448).
        self.reject_moe_batched("forward_prefill_from")?;
        let max_prefill = self.session.max_prefill;
        if token_ids.len() <= max_prefill {
            // Only/last chunk — its logits are always the caller's answer.
            return Ok(self.forward_prefill_batched_chunk(
                token_ids,
                start_pos,
                all_positions,
                true,
                false,
            ));
        }
        if all_positions {
            let mut all_logits = Vec::with_capacity(token_ids.len() * vocab);
            let mut pos = start_pos;
            for chunk in token_ids.chunks(max_prefill) {
                // Perplexity needs every chunk's per-position logits.
                all_logits
                    .extend(self.forward_prefill_batched_chunk(chunk, pos, true, true, false));
                pos += chunk.len();
            }
            Ok(all_logits)
        } else {
            let total = token_ids.len();
            let mut last_logits = Vec::new();
            let mut pos = start_pos;
            for chunk in token_ids.chunks(max_prefill) {
                // Only the LAST chunk's tail is ever kept — see
                // forward_prefill_batched_chunk's doc comment (Experiment B).
                let is_last = pos + chunk.len() == start_pos + total;
                last_logits = self.forward_prefill_batched_chunk(chunk, pos, false, is_last, false);
                pos += chunk.len();
            }
            Ok(last_logits)
        }
    }

    /// Save `represented_token_ids` as the new cross-turn entry for
    /// `slot_id`. Requires that live KV state already represents exactly
    /// those tokens (`kv_cache.seq_len == represented_token_ids.len()`) —
    /// callers must prefill/restore before saving, never after a partial
    /// or speculative step.
    pub(super) fn save_cross_turn_prefix(
        &mut self,
        slot_id: crate::kv_cache::CrossTurnSlotId,
        metadata: crate::kv_cache::CrossTurnPrefixMetadata,
        represented_token_ids: Vec<u32>,
        consumed_entry: Option<MetalCrossTurnPrefixEntry>,
    ) -> Result<(), crate::error::InferenceError> {
        use crate::error::InferenceError;

        let represented_len = represented_token_ids.len();
        if represented_len != self.session.kv_cache.seq_len {
            return Err(InferenceError::PrefixCache(format!(
                "save_cross_turn_prefix: represented_len {represented_len} != kv seq_len {}",
                self.session.kv_cache.seq_len
            )));
        }
        let gdn_snapshot = self.snapshot_gdn_states_checked()?;
        let kv = crate::kv_cache::KvPrefixHandle {
            represented_len,
            num_full_attention_layers: self.session.kv_cache.k_bufs.len(),
            kv_dim: self.session.kv_cache.kv_dim,
            max_cache_len: self.session.kv_cache.max_cache_len,
            kv_f16: self.use_kv_f16,
        };

        // #590: carry surviving sparse checkpoints from the entry this
        // generation consumed (ExactAppend / ReplayFromCheckpoint), and
        // promote that entry's own boundary snapshot into the ring. A
        // checkpoint survives only while the live KV rows below its
        // boundary still hold the tokens it was taken against — the
        // pure predicate `checkpoint_survives_save` encodes that rule
        // against the common token prefix. `FullRefill` saves pass
        // `None`: `reset_state` rebuilt every KV row from scratch, so
        // no prior snapshot is byte-consistent with the live buffers
        // (chunk boundaries differ between the original and the
        // re-prefill), and the ring starts empty by design.
        let mut checkpoints: Vec<MetalGdnCheckpoint> = Vec::new();
        if let Some(old) = consumed_entry {
            let common = crate::kv_cache::longest_common_token_prefix(
                &old.generic.token_ids,
                &represented_token_ids,
            );
            for cp in old.checkpoints {
                if crate::kv_cache::checkpoint_survives_save(cp.len, common, represented_len) {
                    checkpoints.push(cp);
                }
            }
            if old.generic.gdn_snapshot_len == old.generic.represented_len
                && crate::kv_cache::checkpoint_survives_save(
                    old.generic.represented_len,
                    common,
                    represented_len,
                )
            {
                checkpoints.push(MetalGdnCheckpoint {
                    len: old.generic.represented_len,
                    snapshot: old.gdn_snapshot,
                });
            }
            checkpoints.sort_by_key(|c| c.len);
            checkpoints.dedup_by_key(|c| c.len);
            // Evict the shallowest boundaries first — see
            // CROSS_TURN_GDN_CHECKPOINT_CAP for the tradeoff.
            if checkpoints.len() > CROSS_TURN_GDN_CHECKPOINT_CAP {
                let excess = checkpoints.len() - CROSS_TURN_GDN_CHECKPOINT_CAP;
                checkpoints.drain(..excess);
            }
        }

        let generic = crate::kv_cache::CrossTurnPrefixEntry {
            slot_id,
            metadata,
            token_ids: represented_token_ids,
            represented_len,
            kv,
            gdn_snapshot_len: represented_len,
        };
        self.cross_turn_prefix_cache
            .insert(MetalCrossTurnPrefixEntry {
                generic,
                gdn_snapshot,
                checkpoints,
            });
        Ok(())
    }

    /// Best-effort cache save: a bookkeeping failure must not fail an
    /// otherwise-successful generation. On failure the slot is cleared
    /// rather than left holding a possibly-stale entry (fail-closed).
    pub(super) fn save_cross_turn_prefix_or_clear(
        &mut self,
        slot_id: crate::kv_cache::CrossTurnSlotId,
        metadata: crate::kv_cache::CrossTurnPrefixMetadata,
        represented_token_ids: Vec<u32>,
        consumed_entry: Option<MetalCrossTurnPrefixEntry>,
    ) {
        if self
            .save_cross_turn_prefix(slot_id, metadata, represented_token_ids, consumed_entry)
            .is_err()
        {
            self.cross_turn_prefix_cache.remove(slot_id);
        }
    }

    /// [`Self::snapshot_gdn_states`] has no failure mode today, but is
    /// wrapped in `Result` so cache callers have one fallible surface to
    /// widen behind if that ever changes.
    fn snapshot_gdn_states_checked(
        &self,
    ) -> Result<crate::attention::gdn::GdnSnapshot, crate::error::InferenceError> {
        use crate::speculative::MtpTargetVerifier;
        Ok(self.snapshot_gdn_states())
    }

    /// Fallible sibling of [`Self::restore_gdn_states`] for callers that
    /// can propagate a malformed-snapshot error instead of using the
    /// infallible trait entry point's logged, no-write rejection.
    fn restore_gdn_states_checked(
        &mut self,
        snapshot: &crate::attention::gdn::GdnSnapshot,
    ) -> Result<(), crate::error::InferenceError> {
        self.validate_gdn_restore_snapshot(snapshot)?;
        self.restore_gdn_states_validated(snapshot);
        Ok(())
    }
}

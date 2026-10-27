#[cfg(feature = "gdn-state-counters")]
use super::super::{
    GdnStateCopyKind, GdnStateTrafficCounters, GdnStateTrafficReport, GdnStateTrafficScope,
};
use super::MetalQwen35State;
use std::sync::atomic::{AtomicU64, Ordering};

/// Dispatch/readback-site counters for the Metal path-proof probe.
///
/// `&self`-compatible (interior mutability) because the dispatch helpers that
/// increment these run inside `&self` methods sharing one command encoder.
#[derive(Default)]
pub struct PathProofCounters {
    pub prefill_kv_batch: AtomicU64,
    pub prefill_attn_batched: AtomicU64,
    pub prefill_hidden_readback: AtomicU64,
    pub prefill_full_vocab_logit_readback: AtomicU64,
    pub prefill_compact_candidate_logit_readback: AtomicU64,
    pub decode_kv_copy: AtomicU64,
    pub decode_attn_direct: AtomicU64,
    pub decode_attn_split_partial: AtomicU64,
    pub decode_attn_split_reduce: AtomicU64,
    pub decode_hidden_readback: AtomicU64,
    pub decode_full_vocab_logit_readback: AtomicU64,
    pub decode_compact_candidate_logit_readback: AtomicU64,
    /// Device-to-host bytes moved by the logit readbacks counted above.
    ///
    /// These are not redundant with the counts, for two independent reasons. A
    /// compact-eligible request that takes the full-vocab path moves
    /// `vocab_size * 4` bytes instead of `k * 8`, a ratio near 1187 at a 151936
    /// vocabulary and `k = 64`, while both spellings increment some counter by
    /// exactly 1. And `prefill_full_vocab_logit_readback` is incremented by two
    /// sites that move different amounts: the perplexity path reads
    /// `n * vocab_size` values, the ordinary path reads `vocab_size`. The event
    /// count reports 1 for both and destroys the factor of `n`, with no
    /// substitution and no adversary involved.
    pub prefill_full_vocab_logit_readback_bytes: AtomicU64,
    pub prefill_compact_candidate_logit_readback_bytes: AtomicU64,
    pub decode_full_vocab_logit_readback_bytes: AtomicU64,
    pub decode_compact_candidate_logit_readback_bytes: AtomicU64,
}

impl PathProofCounters {
    pub(super) fn reset(&self) {
        self.prefill_kv_batch.store(0, Ordering::Relaxed);
        self.prefill_attn_batched.store(0, Ordering::Relaxed);
        self.prefill_hidden_readback.store(0, Ordering::Relaxed);
        self.prefill_full_vocab_logit_readback
            .store(0, Ordering::Relaxed);
        self.prefill_compact_candidate_logit_readback
            .store(0, Ordering::Relaxed);
        self.decode_kv_copy.store(0, Ordering::Relaxed);
        self.decode_attn_direct.store(0, Ordering::Relaxed);
        self.decode_attn_split_partial.store(0, Ordering::Relaxed);
        self.decode_attn_split_reduce.store(0, Ordering::Relaxed);
        self.decode_hidden_readback.store(0, Ordering::Relaxed);
        self.decode_full_vocab_logit_readback
            .store(0, Ordering::Relaxed);
        self.decode_compact_candidate_logit_readback
            .store(0, Ordering::Relaxed);
        self.prefill_full_vocab_logit_readback_bytes
            .store(0, Ordering::Relaxed);
        self.prefill_compact_candidate_logit_readback_bytes
            .store(0, Ordering::Relaxed);
        self.decode_full_vocab_logit_readback_bytes
            .store(0, Ordering::Relaxed);
        self.decode_compact_candidate_logit_readback_bytes
            .store(0, Ordering::Relaxed);
    }
}

/// Snapshot of [`PathProofCounters`] plus the KV-cache precision mode in effect,
/// suitable for formatting into the `[METAL_PATH_PROOF]` stderr marker.
#[derive(Debug, Clone, Copy, Default)]
pub struct PathProofSnapshot {
    pub prefill_kv_batch: u64,
    pub prefill_attn_batched: u64,
    pub decode_kv_copy: u64,
    pub decode_attn_direct: u64,
    pub decode_attn_split_partial: u64,
    pub decode_attn_split_reduce: u64,
    pub kv_f16: bool,
}

/// Explicit hidden-readback counts recorded by the Metal path-proof probe.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HiddenReadbackPathProofSnapshot {
    pub decode: u64,
    pub prefill: u64,
}

/// Explicit vocabulary-logit-readback counts recorded by the Metal path-proof
/// probe, split by shape: a full-vocab readback (`vocab_size` f32 values) versus
/// a compact-candidate readback (the top-k shortlist only). These are separate
/// dispatch sites in [`MetalQwen35State`] from the hidden-state readback counted
/// by [`HiddenReadbackPathProofSnapshot`] — a request eligible for the compact
/// path that instead takes the full-vocab path changes this snapshot's shape
/// without changing that one's.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogitReadbackPathProofSnapshot {
    pub decode_full_vocab: u64,
    pub decode_compact_candidate: u64,
    pub prefill_full_vocab: u64,
    pub prefill_compact_candidate: u64,
    /// Device-to-host bytes for each of the four readback shapes above.
    ///
    /// ADR-090 D7 asks for transfers *and* bytes around the selected request (the
    /// bare label `D7` is reused elsewhere in this file for an unrelated item, so
    /// the ADR is named here rather than assumed); the counts
    /// above are the transfers. A shape check can be satisfied by a substitution
    /// that moves three orders of magnitude more data under a different field
    /// name, which is what these make assertable.
    pub decode_full_vocab_bytes: u64,
    pub decode_compact_candidate_bytes: u64,
    pub prefill_full_vocab_bytes: u64,
    pub prefill_compact_candidate_bytes: u64,
}

// -----------------------------------------------------------------------
// Layer importance scoring — calibration harness
// -----------------------------------------------------------------------

/// Per-layer hidden-state snapshot for calibration scoring.
pub(super) struct LayerTrace {
    layer_idx: usize,
    input: Vec<f32>,
    output: Vec<f32>,
}

/// Layer importance score derived from hidden-state cosine similarity.
#[derive(Debug, Clone)]
pub struct LayerImportanceScore {
    pub layer_idx: usize,
    pub layer_type: lattice_arch::qwen35_config::LayerType,
    /// Mean cosine similarity of input vs output hidden state across calibration prompts.
    /// High cosine → small change → safer prune candidate.
    pub mean_cosine: f32,
    /// Importance proxy: `1.0 - mean_cosine`.
    pub importance: f32,
}

/// Pruning plan returned by [`MetalQwen35State::score_layer_importance`].
#[derive(Debug, Clone)]
pub struct LayerPruningPlan {
    /// Layers sorted by `mean_cosine` descending (highest cosine first = best prune candidate).
    pub scores: Vec<LayerImportanceScore>,
    /// Recommended per-layer mask (`false` = prune, `true` = keep).
    pub recommended_mask: Vec<bool>,
    /// Set when `prune_layers` exceeds 20 % of the total layer count.
    pub warning: Option<String>,
}

/// Numerically stable cosine similarity using f64 accumulators.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa == 0.0 || bb == 0.0 {
        return 0.0;
    }
    (dot / (aa.sqrt() * bb.sqrt())) as f32
}

/// Per-phase timing accumulator for `forward_step_inner`.
#[derive(Default)]
pub(super) struct StepProfile {
    pub(super) embedding_us: u128,
    pub(super) projection_us: u128,
    pub(super) gdn_recurrence_us: u128,
    pub(super) gqa_attention_us: u128,
    pub(super) mlp_us: u128,
    pub(super) final_us: u128,
    // GPU-time fields populated by LATTICE_DECODE_PROFILE per-group command-buffer split.
    // True GPU execution time measured as Instant::now() around commit()+wait_until_completed()
    // for serialized per-group command buffers. Zero when LATTICE_DECODE_PROFILE is not set.
    pub(super) gpu_gdn_us: u128,
    pub(super) gpu_gqa_us: u128,
    pub(super) gpu_lm_head_us: u128,
    // 4-category de-confounded fields: separates shared MLP from layer-specific mixer/attn.
    pub(super) gpu_gdn_mixer_us: u128,
    pub(super) gpu_gqa_attn_us: u128,
    pub(super) gpu_mlp_us: u128,
}

#[cfg(test)]
thread_local! {
    /// Test-only override for `MetalQwen35State::encode_moe_ffn`'s
    /// CPU-routed top-k expert selection. When set, the real
    /// logits/softmax computation still runs (harmless extra CPU work,
    /// compiled out entirely in non-test builds) but its result is
    /// discarded in favor of this list, so a test can drive a specific,
    /// deterministic sequence of `ExpertSlotCache::resolve` calls
    /// through the SAME real encoded command buffer `forward_step` uses
    /// in production — unlike a helper that calls `resolve()` directly
    /// outside any encoder, this exercises the actual within-token GPU
    /// dispatch ordering the eviction invariant depends on. Cleared
    /// with `None` between tests/tokens by the caller.
    pub(super) static FORCED_MOE_EXPERTS_FOR_TEST: std::cell::RefCell<Option<Vec<usize>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_forced_moe_experts_for_test(experts: Option<Vec<usize>>) {
    FORCED_MOE_EXPERTS_FOR_TEST.with(|c| *c.borrow_mut() = experts);
}

#[cfg(test)]
thread_local! {
    /// Test-only override for `MetalQwen35State::encode_moe_ffn`'s
    /// routed-expert prefetch fan-out (#682 Stage 2): `None` (the
    /// production default) prefetches via rayon; `Some(false)` forces
    /// the identical pre-assigned plan through a plain serial
    /// iterator instead, so a test can assert the two fan-out
    /// strategies produce byte-identical decode output from the same
    /// checkpoint and token sequence. `Some(true)` forces rayon
    /// explicitly (same as unset, provided for symmetry). Cleared
    /// with `None` between tests by the caller.
    pub(super) static MOE_PREFETCH_PARALLEL_FOR_TEST: std::cell::RefCell<Option<bool>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_moe_prefetch_parallel_for_test(parallel: Option<bool>) {
    MOE_PREFETCH_PARALLEL_FOR_TEST.with(|c| *c.borrow_mut() = parallel);
}

#[cfg(test)]
thread_local! {
    /// Test-only ordering-proof hook, paired with
    /// `MOE_PREFETCH_STEP2_DONE_TX_FOR_TEST` below: when set, applies
    /// ONLY to the gate_up cache's dequant spawn for the next
    /// `encode_moe_ffn` call (down's is unaffected — proving overlap
    /// for one cache is sufficient evidence of the mechanism). Moved
    /// (via `.take()`) into `ExpertSlotCache::spawn_dequant` on the
    /// calling thread, same reasoning as the panic-injection hook
    /// above. Cleared to `None` after being taken.
    pub(super) static MOE_PREFETCH_ORDERING_GATE_FOR_TEST:
        std::cell::RefCell<Option<crate::forward::moe_expert_cache::PrefetchOrderingGate>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_moe_prefetch_ordering_gate_for_test(
    gate: Option<crate::forward::moe_expert_cache::PrefetchOrderingGate>,
) {
    MOE_PREFETCH_ORDERING_GATE_FOR_TEST.with(|c| *c.borrow_mut() = gate);
}

#[cfg(test)]
thread_local! {
    /// Test-only: when set, `encode_moe_ffn` sends on it immediately
    /// after Step 2's shared-expert GEMVs are fully encoded (before
    /// joining the routed-expert dequant spawn(s)), so a test can
    /// prove Step 2 encoding completed while a still-gated dequant
    /// spawn is deliberately blocked on
    /// `MOE_PREFETCH_ORDERING_GATE_FOR_TEST`'s `release_rx`. Cleared
    /// to `None` after being taken.
    pub(super) static MOE_PREFETCH_STEP2_DONE_TX_FOR_TEST:
        std::cell::RefCell<Option<std::sync::mpsc::Sender<()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_moe_prefetch_step2_done_tx_for_test(tx: Option<std::sync::mpsc::Sender<()>>) {
    MOE_PREFETCH_STEP2_DONE_TX_FOR_TEST.with(|c| *c.borrow_mut() = tx);
}

/// One recorded top-k routing decision: which experts `encode_moe_ffn`'s
/// CPU router selected for one token at one MoE layer, and their
/// (post-renormalization) gate weights, in `selected_ids[i]` /
/// `gate_weights[i]` correspondence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MoeRoutingTraceRecord {
    pub layer_idx: usize,
    pub token_idx: usize,
    pub selected_ids: Vec<usize>,
    pub gate_weights: Vec<f32>,
}

thread_local! {
    /// Optional per-token routing-divergence trace collector for
    /// `encode_moe_ffn`'s CPU top-k selection (#682 Stage 4 instrumentation
    /// seam). Same arm/drain-via-free-function shape as
    /// `FORCED_MOE_EXPERTS_FOR_TEST` above, but NOT `#[cfg(test)]`-gated:
    /// production harnesses (e.g. `eval_perplexity`) arm it outside of
    /// tests to record per-token top-k expert-id sets and gate weights,
    /// for computing set-agreement (Jaccard) between a quantized arm and
    /// a full-precision reference. `None` (the default) makes every
    /// `encode_moe_ffn` call a single branch-on-None with no allocation.
    pub(super) static MOE_ROUTING_TRACE: std::cell::RefCell<Option<Vec<MoeRoutingTraceRecord>>> =
        const { std::cell::RefCell::new(None) };
}

/// Arm the routing-trace collector on this thread: every subsequent
/// `encode_moe_ffn` call appends one [`MoeRoutingTraceRecord`], until
/// [`take_moe_routing_trace`] disarms it. Re-arming an already-armed
/// collector discards any records buffered since the previous arm.
pub fn arm_moe_routing_trace() {
    MOE_ROUTING_TRACE.with(|c| *c.borrow_mut() = Some(Vec::new()));
}

/// Disarm the collector and return every record buffered since the last
/// [`arm_moe_routing_trace`] call (empty if it was never armed).
pub fn take_moe_routing_trace() -> Vec<MoeRoutingTraceRecord> {
    MOE_DEVICE_ROUTE_PENDING.with(|c| c.borrow_mut().clear());
    MOE_ROUTING_TRACE.with(|c| c.borrow_mut().take().unwrap_or_default())
}

thread_local! {
    /// Device-route buffers (ADR-053 D3') the armed trace still has to publish,
    /// as `(layer_idx, token_idx, top_k, route)` in encode order. The router
    /// kernel writes them inside the token's command buffer, so they are read
    /// only by [`publish_moe_device_routes`] after that buffer completes.
    pub(super) static MOE_DEVICE_ROUTE_PENDING:
        std::cell::RefCell<Vec<(usize, usize, usize, metal::Buffer)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Move every pending device route into the armed routing trace, with its
/// original layer index, token position, rank order and normalized weights.
///
/// # Safety
/// The command buffers that wrote the pending route buffers have completed.
pub(super) unsafe fn publish_moe_device_routes() {
    let pending = MOE_DEVICE_ROUTE_PENDING.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if pending.is_empty() {
        return;
    }
    MOE_ROUTING_TRACE.with(|c| {
        let mut armed = c.borrow_mut();
        let Some(trace) = armed.as_mut() else {
            return;
        };
        for (layer_idx, token_idx, top_k, route) in pending {
            // SAFETY: the route buffer is StorageModeShared and holds at least
            // `5 * top_k + 1` u32 words; its writer completed (caller contract).
            let words =
                unsafe { std::slice::from_raw_parts(route.contents() as *const u32, 2 * top_k) };
            trace.push(MoeRoutingTraceRecord {
                layer_idx,
                token_idx,
                selected_ids: words[..top_k].iter().map(|&id| id as usize).collect(),
                gate_weights: words[top_k..].iter().map(|&w| f32::from_bits(w)).collect(),
            });
        }
    });
}

#[cfg(test)]
thread_local! {
    /// Test-only router-run counters `(host, device)`: the synchronized CPU
    /// router in `encode_moe_ffn` bumps the first, the device-route encoder the
    /// second, once per MoE layer per token.
    pub(super) static MOE_ROUTER_RUNS_FOR_TEST: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
}

#[cfg(test)]
pub(crate) fn take_moe_router_runs_for_test() -> (usize, usize) {
    MOE_ROUTER_RUNS_FOR_TEST.with(|c| c.replace((0, 0)))
}

/// Write `records` as newline-delimited JSON, one [`MoeRoutingTraceRecord`]
/// per line, so an external harness can diff a trace against a reference run.
pub fn dump_moe_routing_trace_jsonl(
    records: &[MoeRoutingTraceRecord],
    path: &std::path::Path,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for record in records {
        serde_json::to_writer(&mut out, record).map_err(std::io::Error::other)?;
        out.write_all(b"\n")?;
    }
    out.flush()
}

impl MetalQwen35State {
    /// Zeroes the Metal path-proof counters.
    ///
    /// Call before a one-shot `generate`/`generate_streaming` run so
    /// [`Self::path_proof_snapshot`], [`Self::hidden_readback_path_proof_snapshot`],
    /// and [`Self::logit_readback_path_proof_snapshot`] afterward reflect only
    /// that run's dispatches, explicit hidden readbacks, and logit readbacks.
    pub fn reset_path_proof_counters(&self) {
        self.path_proof.reset();
    }

    /// Snapshots the Metal attention/KV-cache path-proof counters (issue #239)
    /// without resetting them, alongside the KV-cache precision mode in effect.
    pub fn path_proof_snapshot(&self) -> PathProofSnapshot {
        PathProofSnapshot {
            prefill_kv_batch: self.path_proof.prefill_kv_batch.load(Ordering::Relaxed),
            prefill_attn_batched: self.path_proof.prefill_attn_batched.load(Ordering::Relaxed),
            decode_kv_copy: self.path_proof.decode_kv_copy.load(Ordering::Relaxed),
            decode_attn_direct: self.path_proof.decode_attn_direct.load(Ordering::Relaxed),
            decode_attn_split_partial: self
                .path_proof
                .decode_attn_split_partial
                .load(Ordering::Relaxed),
            decode_attn_split_reduce: self
                .path_proof
                .decode_attn_split_reduce
                .load(Ordering::Relaxed),
            kv_f16: self.use_kv_f16,
        }
    }

    /// Snapshots explicit hidden-readback path-proof counters without resetting them.
    pub fn hidden_readback_path_proof_snapshot(&self) -> HiddenReadbackPathProofSnapshot {
        HiddenReadbackPathProofSnapshot {
            decode: self
                .path_proof
                .decode_hidden_readback
                .load(Ordering::Relaxed),
            prefill: self
                .path_proof
                .prefill_hidden_readback
                .load(Ordering::Relaxed),
        }
    }

    /// Snapshots explicit vocabulary-logit-readback path-proof counters (full-vocab
    /// vs compact-candidate, decode and prefill) without resetting them. See
    /// [`LogitReadbackPathProofSnapshot`] for why this is a separate instrument from
    /// [`Self::hidden_readback_path_proof_snapshot`].
    pub fn logit_readback_path_proof_snapshot(&self) -> LogitReadbackPathProofSnapshot {
        LogitReadbackPathProofSnapshot {
            decode_full_vocab: self
                .path_proof
                .decode_full_vocab_logit_readback
                .load(Ordering::Relaxed),
            decode_compact_candidate: self
                .path_proof
                .decode_compact_candidate_logit_readback
                .load(Ordering::Relaxed),
            prefill_full_vocab: self
                .path_proof
                .prefill_full_vocab_logit_readback
                .load(Ordering::Relaxed),
            prefill_compact_candidate: self
                .path_proof
                .prefill_compact_candidate_logit_readback
                .load(Ordering::Relaxed),
            decode_full_vocab_bytes: self
                .path_proof
                .decode_full_vocab_logit_readback_bytes
                .load(Ordering::Relaxed),
            decode_compact_candidate_bytes: self
                .path_proof
                .decode_compact_candidate_logit_readback_bytes
                .load(Ordering::Relaxed),
            prefill_full_vocab_bytes: self
                .path_proof
                .prefill_full_vocab_logit_readback_bytes
                .load(Ordering::Relaxed),
            prefill_compact_candidate_bytes: self
                .path_proof
                .prefill_compact_candidate_logit_readback_bytes
                .load(Ordering::Relaxed),
        }
    }

    /// Resets GDN state-traffic counters to zero without changing the shape.
    ///
    /// No-op when counters are inactive (feature off, or
    /// `LATTICE_GDN_STATE_COUNTERS` unset at construction time).
    #[cfg(feature = "gdn-state-counters")]
    pub fn reset_gdn_state_traffic(&mut self) {
        if let Some(ref mut counters) = self.session.gdn_state_traffic {
            counters.reset();
        }
    }

    /// Returns the current GDN state-traffic report without resetting it.
    ///
    /// Returns `None` when counters are inactive.
    #[cfg(feature = "gdn-state-counters")]
    pub fn gdn_state_traffic_report(&self) -> Option<GdnStateTrafficReport> {
        self.session
            .gdn_state_traffic
            .as_ref()
            .map(GdnStateTrafficCounters::report)
    }

    /// Returns the current GDN state-traffic report and resets the counters,
    /// so callers can print one row per decode step without subtracting totals.
    ///
    /// Returns `None` when counters are inactive.
    #[cfg(feature = "gdn-state-counters")]
    pub fn take_gdn_state_traffic_report(&mut self) -> Option<GdnStateTrafficReport> {
        let report = self
            .session
            .gdn_state_traffic
            .as_ref()
            .map(GdnStateTrafficCounters::report);
        if let Some(ref mut counters) = self.session.gdn_state_traffic {
            counters.reset();
        }
        report
    }

    #[cfg(feature = "gdn-state-counters")]
    pub(super) fn record_gdn_layer_access(
        &mut self,
        scope: GdnStateTrafficScope,
        layer_count: usize,
    ) {
        if let Some(ref mut counters) = self.session.gdn_state_traffic {
            counters.record_layer_access(scope, layer_count);
        }
    }

    #[cfg(feature = "gdn-state-counters")]
    pub(super) fn record_gdn_copy(&mut self, scope: GdnStateTrafficScope, kind: GdnStateCopyKind) {
        if let Some(ref mut counters) = self.session.gdn_state_traffic {
            counters.record_copy(scope, kind);
        }
    }

    // ===================================================================
    // Layer importance scoring
    // ===================================================================

    /// Run a prefix-masked forward pass capturing the last-token hidden state
    /// before and after each active layer for importance scoring.
    ///
    /// For N active layers, performs N sequential model passes where pass `i`
    /// activates only layers `[active_0 .. active_i]`.  Each pass processes all
    /// tokens in `token_ids` via `forward_step_inner`.  Because preceding layers
    /// are identical across passes, `output[i]` is the exact post-layer-i hidden
    /// state from the full-model run.
    ///
    /// Intended for offline calibration only.
    /// Cost: O(N²) in active layers — performs N passes of increasing length
    /// (pass i activates layers 0..=i, total layer-token work ≈ N²/2).
    /// Acceptable for calibration-only usage where N ≤ 64.
    pub(super) fn forward_prefill_layer_traces_last_token(
        &mut self,
        token_ids: &[u32],
    ) -> Result<Vec<LayerTrace>, lattice_inference_core::InferenceError> {
        if token_ids.is_empty() {
            return Err(lattice_inference_core::InferenceError::Inference(
                "forward_prefill_layer_traces_last_token: empty token sequence".to_string(),
            ));
        }
        // `reset_state` below zeroes `kv_cache.seq_len`, so this call's own
        // `forward_step_inner` loop drives `position` from 0..token_ids.len() — a
        // caller-supplied length reaching this public entry point via
        // `score_layer_importance`'s `calibration_prompts`. Validate against the
        // session's RoPE/KV capacity before any pass dispatches, same as
        // `check_forward_step_capacity` does for `forward_step`.
        self.check_forward_range_capacity(0, token_ids.len(), false)?;
        let orig_cfg = self.engine.config.clone();
        let hidden = orig_cfg.hidden_size;
        let n = token_ids.len();

        let active_indices: Vec<usize> = (0..orig_cfg.num_hidden_layers)
            .filter(|&i| orig_cfg.is_layer_active(i))
            .collect();

        // Input to the first active layer = embedding of the last token.
        let last_id = *token_ids.last().unwrap();
        assert!(
            (last_id as usize) < orig_cfg.vocab_size,
            "token_id {last_id} >= vocab_size {}",
            orig_cfg.vocab_size
        );
        // SAFETY: last_id < vocab_size above; out holds hidden f32 values.
        let first_input: Vec<f32> = unsafe {
            let mut out = vec![0.0f32; hidden];
            self.engine
                .embed_tokens
                .write_row(last_id as usize, &mut out);
            out
        };

        let mut traces: Vec<LayerTrace> = Vec::with_capacity(active_indices.len());
        let mut prev_output: Vec<f32> = first_input;

        for (active_pos, &layer_i) in active_indices.iter().enumerate() {
            // Mask: activate only layers active_indices[0..=active_pos].
            let mut trace_mask = vec![false; orig_cfg.num_hidden_layers];
            for j in 0..=active_pos {
                trace_mask[active_indices[j]] = true;
            }
            self.engine.config.layer_mask = trace_mask;
            self.reset_state();

            // Process all tokens; capture pre-final hidden for the last one.
            for (pos, &id) in token_ids.iter().enumerate() {
                let is_last = pos == n - 1;
                let _ = self.forward_step_inner(
                    id,
                    pos,
                    is_last,
                    crate::forward::signpost::Scope::NotDecode,
                );
            }

            // last_pre_final_hidden = post-last-active-layer hidden for the last token.
            let output_hidden = self.session.last_pre_final_hidden.clone();
            let input_hidden = prev_output.clone();
            prev_output = output_hidden.clone();

            traces.push(LayerTrace {
                layer_idx: layer_i,
                input: input_hidden,
                output: output_hidden,
            });
        }

        // Restore original config and leave model in a clean state.
        self.engine.config.layer_mask = orig_cfg.layer_mask;
        self.reset_state();

        Ok(traces)
    }

    /// Score each active layer's importance by measuring how much it changes the
    /// last-token hidden state across `calibration_prompts`.
    ///
    /// Layers with high mean cosine similarity change the hidden state least and
    /// are the safest prune candidates.  Returns a [`LayerPruningPlan`] containing
    /// per-layer scores (sorted highest cosine first), a type-balanced recommended
    /// mask for `prune_layers` removals, and an optional Gromov warning when
    /// `prune_layers` exceeds 20 % of the total layer count.
    pub fn score_layer_importance(
        &mut self,
        calibration_prompts: &[Vec<u32>],
        prune_layers: usize,
    ) -> Result<LayerPruningPlan, lattice_inference_core::InferenceError> {
        use lattice_arch::qwen35_config::LayerType;
        let cfg = self.engine.config.clone();

        let mut sums = vec![0.0f64; cfg.num_hidden_layers];
        let mut counts = vec![0usize; cfg.num_hidden_layers];

        for prompt in calibration_prompts {
            let traces = self.forward_prefill_layer_traces_last_token(prompt)?;
            for trace in &traces {
                let cos = cosine_similarity(&trace.input, &trace.output);
                sums[trace.layer_idx] += cos as f64;
                counts[trace.layer_idx] += 1;
            }
        }

        let mut scores: Vec<LayerImportanceScore> = (0..cfg.num_hidden_layers)
            .filter(|&i| cfg.is_layer_active(i))
            .map(|i| {
                let mean_cosine = if counts[i] == 0 {
                    0.0
                } else {
                    (sums[i] / counts[i] as f64) as f32
                };
                LayerImportanceScore {
                    layer_idx: i,
                    layer_type: cfg.layer_types[i],
                    mean_cosine,
                    importance: 1.0 - mean_cosine,
                }
            })
            .collect();

        // Highest cosine first = lowest importance = best prune candidate.
        scores.sort_by(|a, b| b.mean_cosine.total_cmp(&a.mean_cosine));

        // Type-balanced quota: ~1 GQA removed per 3 GDN.
        let gqa_quota = (prune_layers + 2) / 4;
        let gdn_quota = prune_layers.saturating_sub(gqa_quota);
        let mut gdn_left = gdn_quota;
        let mut gqa_left = gqa_quota;
        let mut recommended_mask = cfg.layer_mask.clone();

        for score in &scores {
            match score.layer_type {
                LayerType::LinearAttention if gdn_left > 0 => {
                    recommended_mask[score.layer_idx] = false;
                    gdn_left -= 1;
                }
                LayerType::FullAttention if gqa_left > 0 => {
                    recommended_mask[score.layer_idx] = false;
                    gqa_left -= 1;
                }
                _ => {}
            }
            if gdn_left == 0 && gqa_left == 0 {
                break;
            }
        }

        let warning = if prune_layers * 5 > cfg.num_hidden_layers {
            Some(format!(
                "Gromov warning: requested prune_layers={prune_layers} exceeds 20% of {} layers",
                cfg.num_hidden_layers
            ))
        } else {
            None
        };

        Ok(LayerPruningPlan {
            scores,
            recommended_mask,
            warning,
        })
    }
}

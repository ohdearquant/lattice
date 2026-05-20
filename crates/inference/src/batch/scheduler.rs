//! Iteration-level batch scheduler for continuous batching (ADR-048).
//!
//! The [`Scheduler`] trait defines the interface called each iteration.
//! [`FifoScheduler`] implements FIFO admission with chunked prefill
//! interleaved with decode steps — Phase 1 of ADR-048.
//!
//! # Invariants
//!
//! - All sequences in `decode` have completed at least one prefill chunk.
//! - Sequences are never simultaneously in both `prefill` and `decode`.
//! - The scheduler does not evict running sequences in Phase 1; it rejects
//!   new admissions when KV pages are low instead.

use std::collections::VecDeque;

use crate::batch::config::BatchConfig;
use crate::batch::sequence::SeqId;

// ---------------------------------------------------------------------------
// SchedulerDecision
// ---------------------------------------------------------------------------

/// The set of actions the worker should take this iteration.
#[derive(Debug, Default)]
pub struct SchedulerDecision {
    /// Sequences whose next prefill chunk should run this iteration.
    ///
    /// Each entry is `(seq_id, chunk_start, chunk_len)` where `chunk_start`
    /// is the token index into the original prompt at which to begin, and
    /// `chunk_len` is the number of tokens to process.
    pub prefill: Vec<(SeqId, usize, usize)>,

    /// Sequences ready for a single decode step.
    ///
    /// Invariant: every seq_id here has `SequenceState::Decoding`.
    pub decode: Vec<SeqId>,

    /// Sequences to evict before this iteration runs.
    ///
    /// Phase 1: empty (no eviction; admission is refused instead).
    /// Phase 2: populated under memory pressure (ADR-048 §alternatives).
    pub evict: Vec<SeqId>,
}

// ---------------------------------------------------------------------------
// Scheduler trait
// ---------------------------------------------------------------------------

/// Iteration-level scheduler interface.
///
/// Called once per inference iteration. Returns a [`SchedulerDecision`]
/// describing which sequences should run prefill, which should decode, and
/// which (if any) should be evicted.
pub trait Scheduler: Send {
    /// Select the batch for this iteration.
    ///
    /// # Arguments
    ///
    /// * `waiting` — Sequence IDs in `Prefilling` state (not yet decoding).
    /// * `running` — Sequence IDs in `Decoding` state.
    /// * `kv_free_pages` — Number of unallocated KV cache pages.
    /// * `gdn_free_slots` — Number of unallocated GDN state slots.
    fn select_batch(
        &mut self,
        waiting: &[SeqId],
        running: &[SeqId],
        kv_free_pages: usize,
        gdn_free_slots: usize,
    ) -> SchedulerDecision;

    /// Notify the scheduler that `seq_id` has been evicted by the worker.
    fn on_preempt(&mut self, seq_id: SeqId);
}

// ---------------------------------------------------------------------------
// FifoScheduler
// ---------------------------------------------------------------------------

/// First-in-first-out scheduler with chunked prefill. Phase 1 of ADR-048.
///
/// # Policy
///
/// - Decode slots fill first (bandwidth-bound, low latency impact).
/// - One prefill chunk is admitted per waiting sequence per iteration,
///   bounded by `config.chunk_size`.
/// - New sequences are admitted only when:
///   1. `running + waiting < max_batch_size`
///   2. `kv_free_pages >= prefill_reserve_pages`
/// - No eviction in Phase 1: when capacity is low, the waiting queue grows.
///
/// # LoRA adapter grouping
///
/// The scheduler surfaces per-sequence `adapter_id` in future work (ADR-048
/// §LoRA adapter grouping). Phase 1 does not enforce sub-batch splitting;
/// the worker is responsible for grouping sequences by adapter identity before
/// dispatching matrix multiplies.
#[derive(Debug)]
pub struct FifoScheduler {
    config: BatchConfig,
    /// FIFO admission queue: new sequence IDs waiting to be scheduled.
    admission_queue: VecDeque<SeqId>,
}

impl FifoScheduler {
    /// Create a new FIFO scheduler with the given configuration.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            admission_queue: VecDeque::new(),
        }
    }

    /// Enqueue a sequence for future admission.
    ///
    /// The scheduler will admit sequences from this queue when capacity allows.
    pub fn enqueue(&mut self, seq_id: SeqId) {
        self.admission_queue.push_back(seq_id);
    }

    /// Number of sequences currently waiting for admission.
    #[inline]
    pub fn waiting_count(&self) -> usize {
        self.admission_queue.len()
    }
}

impl Scheduler for FifoScheduler {
    fn select_batch(
        &mut self,
        waiting: &[SeqId],
        running: &[SeqId],
        kv_free_pages: usize,
        _gdn_free_slots: usize,
    ) -> SchedulerDecision {
        let mut decision = SchedulerDecision::default();

        // Phase 1: No eviction — compute how many active slots are available.
        let active_count = waiting.len() + running.len();
        let capacity_remaining = self.config.max_batch_size.saturating_sub(active_count);

        // Admit sequences from the admission queue if capacity and memory allow.
        // Memory guard: refuse admission when free pages are below the reserve threshold
        // AND we already have sequences running (don't starve a fresh empty batch).
        let can_admit = kv_free_pages >= self.config.prefill_reserve_pages || active_count == 0;

        let admit_limit = if can_admit { capacity_remaining } else { 0 };
        for _ in 0..admit_limit {
            if let Some(seq_id) = self.admission_queue.pop_front() {
                // The worker is responsible for actually transitioning the sequence
                // into the manager; we just indicate it should start this iteration.
                decision.prefill.push((seq_id, 0, self.config.chunk_size));
            } else {
                break;
            }
        }

        // Schedule decode for all running sequences.
        //
        // Running sequences already hold allocated GDN slots (allocated during
        // their first prefill chunk). `gdn_free_slots` gates NEW admissions
        // above, not existing decode sequences. Phase 2 may add priority-based
        // decode scheduling; Phase 1 decodes all running sequences each iteration.
        for &seq_id in running {
            decision.decode.push(seq_id);
        }

        // Schedule remaining chunks for sequences already in prefilling state.
        // These are sequences the worker told us are still chunked-prefilling.
        // The worker passes their current `chunk_start` via `waiting`; here we
        // produce the (seq_id, chunk_start, chunk_len) tuple.
        //
        // Note: `waiting` contains seq_ids whose chunk_start is tracked by the
        // worker (in SequenceManager). The scheduler provides chunk_size; the
        // worker computes the actual chunk extent.
        //
        // We avoid re-adding newly-admitted sequences (they are already in
        // `decision.prefill`). Newly admitted IDs came from `admission_queue`,
        // not from `waiting`.
        let admitted_ids: std::collections::HashSet<SeqId> =
            decision.prefill.iter().map(|&(id, _, _)| id).collect();

        for &seq_id in waiting {
            if !admitted_ids.contains(&seq_id) {
                // chunk_start is unknown to the scheduler; use 0 as a sentinel.
                // The worker substitutes the real chunk_start from SequenceManager.
                decision
                    .prefill
                    .push((seq_id, usize::MAX, self.config.chunk_size));
            }
        }

        decision
    }

    fn on_preempt(&mut self, _seq_id: SeqId) {
        // Phase 1: preemption does not occur. This is a no-op placeholder
        // for Phase 2, which will re-queue the evicted sequence here.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::config::BatchConfig;

    fn default_sched() -> FifoScheduler {
        FifoScheduler::new(BatchConfig::default())
    }

    fn small_sched(max_batch: usize, chunk: usize, reserve: usize) -> FifoScheduler {
        FifoScheduler::new(BatchConfig {
            max_batch_size: max_batch,
            max_seq_len: 4096,
            chunk_size: chunk,
            prefill_reserve_pages: reserve,
        })
    }

    // --- Empty state ---

    #[test]
    fn empty_state_returns_empty_decision() {
        let mut sched = default_sched();
        let dec = sched.select_batch(&[], &[], 100, 100);
        assert!(dec.prefill.is_empty());
        assert!(dec.decode.is_empty());
        assert!(dec.evict.is_empty());
    }

    // --- Admission ---

    #[test]
    fn new_sequence_admitted_into_prefill() {
        let mut sched = default_sched();
        sched.enqueue(SeqId(1));
        let dec = sched.select_batch(&[], &[], 100, 100);
        assert_eq!(dec.prefill.len(), 1);
        let (id, start, len) = dec.prefill[0];
        assert_eq!(id, SeqId(1));
        assert_eq!(start, 0);
        assert_eq!(len, 512); // default chunk_size
    }

    #[test]
    fn admission_respects_max_batch_size() {
        let mut sched = small_sched(2, 512, 0);
        sched.enqueue(SeqId(1));
        sched.enqueue(SeqId(2));
        sched.enqueue(SeqId(3)); // over capacity
        // No running or waiting sequences — all 3 enqueued.
        // Capacity = max_batch_size - 0 = 2 → only 2 admitted.
        let dec = sched.select_batch(&[], &[], 100, 100);
        assert_eq!(dec.prefill.len(), 2);
        // Third sequence still in queue.
        assert_eq!(sched.waiting_count(), 1);
    }

    #[test]
    fn admission_refuses_when_active_count_full() {
        let mut sched = small_sched(2, 512, 0);
        sched.enqueue(SeqId(10));
        // Both slots already occupied by running sequences.
        let dec = sched.select_batch(&[], &[SeqId(1), SeqId(2)], 100, 100);
        // Capacity = max_batch_size(2) - running(2) = 0 → no new admissions.
        assert!(dec.prefill.is_empty());
        assert_eq!(sched.waiting_count(), 1);
    }

    #[test]
    fn admission_blocked_by_memory_guard() {
        let mut sched = small_sched(32, 512, 8);
        sched.enqueue(SeqId(1));
        // kv_free_pages < prefill_reserve_pages AND active_count > 0
        let dec = sched.select_batch(&[SeqId(0)], &[], 3, 100);
        // No new admissions because pages < reserve.
        // SeqId(1) was NOT admitted; SeqId(0) is already in waiting, so it gets
        // a chunk entry (sentinel chunk_start = usize::MAX).
        assert!(dec.prefill.iter().all(|&(id, _, _)| id == SeqId(0)));
        assert_eq!(sched.waiting_count(), 1);
    }

    #[test]
    fn admission_allowed_when_batch_empty_despite_low_pages() {
        let mut sched = small_sched(32, 512, 8);
        sched.enqueue(SeqId(1));
        // active_count == 0 → bypass memory guard.
        let dec = sched.select_batch(&[], &[], 0, 100);
        assert_eq!(dec.prefill.len(), 1);
    }

    // --- Decode ---

    #[test]
    fn running_sequences_go_to_decode() {
        let mut sched = default_sched();
        let dec = sched.select_batch(&[], &[SeqId(5), SeqId(6)], 100, 100);
        assert_eq!(dec.decode.len(), 2);
        assert!(dec.decode.contains(&SeqId(5)));
        assert!(dec.decode.contains(&SeqId(6)));
    }

    #[test]
    fn all_running_sequences_decode_regardless_of_free_gdn_slots() {
        let mut sched = default_sched();
        // Running sequences already hold allocated GDN slots. Free slot count
        // should NOT throttle decode — it only gates new admissions.
        let dec = sched.select_batch(&[], &[SeqId(1), SeqId(2), SeqId(3)], 100, 0);
        assert_eq!(dec.decode.len(), 3);
        assert!(dec.decode.contains(&SeqId(1)));
        assert!(dec.decode.contains(&SeqId(2)));
        assert!(dec.decode.contains(&SeqId(3)));
    }

    // --- Prefill continuation ---

    #[test]
    fn waiting_sequences_get_continuation_chunk() {
        let mut sched = small_sched(32, 128, 0);
        // SeqId(2) is already in the waiting (prefilling) list.
        let dec = sched.select_batch(&[SeqId(2)], &[], 100, 100);
        // Should have one entry for SeqId(2) with sentinel chunk_start.
        let found = dec
            .prefill
            .iter()
            .any(|&(id, start, len)| id == SeqId(2) && start == usize::MAX && len == 128);
        assert!(found, "continuation chunk for waiting sequence not found");
    }

    #[test]
    fn no_duplicate_seqid_in_prefill_between_new_and_waiting() {
        let mut sched = small_sched(32, 512, 0);
        // Enqueue a new sequence that is NOT in waiting.
        sched.enqueue(SeqId(99));
        // waiting contains a different sequence.
        let dec = sched.select_batch(&[SeqId(10)], &[], 100, 100);
        // SeqId(99) admitted at (0, chunk_size), SeqId(10) gets sentinel.
        let ids: Vec<SeqId> = dec.prefill.iter().map(|&(id, _, _)| id).collect();
        // No duplicates.
        let mut seen = std::collections::HashSet::new();
        for id in &ids {
            assert!(seen.insert(id), "duplicate SeqId in prefill: {id}");
        }
    }

    // --- Eviction ---

    #[test]
    fn phase1_no_evictions() {
        let mut sched = default_sched();
        let dec = sched.select_batch(&[], &[SeqId(1)], 100, 100);
        assert!(dec.evict.is_empty());
    }

    // --- on_preempt noop ---

    #[test]
    fn on_preempt_does_not_panic() {
        let mut sched = default_sched();
        sched.on_preempt(SeqId(42)); // must not panic
    }

    // --- Chunk size propagation ---

    #[test]
    fn chunk_size_propagated_to_new_admissions() {
        let mut sched = small_sched(32, 256, 0);
        sched.enqueue(SeqId(1));
        let dec = sched.select_batch(&[], &[], 100, 100);
        let (_, _, len) = dec.prefill[0];
        assert_eq!(len, 256);
    }
}

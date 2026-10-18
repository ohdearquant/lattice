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
    /// is the token index into the original prompt at which to begin, or
    /// `usize::MAX` to use the cursor from sequence state. `FifoScheduler`
    /// uses zero for new admissions and `usize::MAX` for continuations.
    /// `chunk_len` is the maximum number of prompt tokens to process.
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
    /// * `waiting` — Already admitted sequence IDs in `Prefilling` state,
    ///   each holding a GDN slot. Excludes IDs still in the admission queue.
    /// * `running` — Admitted sequence IDs in `Decoding` state, holding GDN slots.
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
/// - One prefill chunk is scheduled per admitted prefilling sequence per iteration,
///   bounded by `config.chunk_size`.
/// - New sequences are admitted only when:
///   1. `running + waiting < max_batch_size`
///   2. `kv_free_pages >= prefill_reserve_pages` or no sequences are active
///   3. a GDN state slot is free (one new admission consumes exactly one slot)
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
        gdn_free_slots: usize,
    ) -> SchedulerDecision {
        let mut decision = SchedulerDecision::default();

        // Phase 1: No eviction — compute how many active slots are available.
        let active_count = waiting.len() + running.len();
        let capacity_remaining = self.config.max_batch_size.saturating_sub(active_count);

        // Admit sequences from the admission queue if capacity and memory allow.
        // Memory guard: refuse admission when free pages are below the reserve threshold
        // AND we already have sequences running (don't starve a fresh empty batch).
        let can_admit = kv_free_pages >= self.config.prefill_reserve_pages || active_count == 0;

        // Each newly admitted sequence consumes exactly one GDN state slot on its
        // first prefill chunk (BatchWorker::step allocates from the pool as it
        // processes decision.prefill), so admission can never outrun the free
        // slot count regardless of how much batch/page capacity remains.
        let admit_limit = if can_admit {
            capacity_remaining.min(gdn_free_slots)
        } else {
            0
        };
        for _ in 0..admit_limit {
            if let Some(seq_id) = self.admission_queue.pop_front() {
                // The worker already tracks the request and allocates its GDN slot
                // when this first prefill chunk executes.
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

        // Continuations already hold slots, so admission pressure must not
        // prevent them from advancing and eventually releasing those slots.
        let admitted_ids: std::collections::HashSet<SeqId> =
            decision.prefill.iter().map(|&(id, _, _)| id).collect();

        for &seq_id in waiting {
            if !admitted_ids.contains(&seq_id) {
                // chunk_start is unknown to the scheduler; use usize::MAX as a sentinel.
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
mod tests;

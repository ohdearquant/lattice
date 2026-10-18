//! Per-sequence state tracking for the continuous batching engine.
//!
//! [`Sequence`] is the unit of work: it holds the token buffer, a reference to
//! KV pages (via [`crate::kv_cache::PageTable`]), and the GDN recurrent state
//! for linear-attention layers (ADR-048 §GDN state batching).
//!
//! [`SequenceManager`] owns all live sequences and their KV page tables.
//! It is the external owner referenced in `paged.rs:247-251` — `PagedKVCache`
//! tracks a single `PageTable`; for multi-sequence batching we manage one
//! `PageTable` per sequence here.

use std::collections::HashMap;

use crate::kv_cache::PageTable;
use crate::sampling::SamplingConfig;

// ---------------------------------------------------------------------------
// SeqId
// ---------------------------------------------------------------------------

/// Stable identifier for a sequence across its lifetime in the scheduler.
///
/// Monotonically increasing; never reused within a process lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SeqId(pub u64);

impl SeqId {
    /// Next identifier from a counter.
    #[inline]
    pub fn next(counter: &mut u64) -> Self {
        let id = *counter;
        *counter += 1;
        Self(id)
    }
}

impl std::fmt::Display for SeqId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "seq:{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// SequenceState
// ---------------------------------------------------------------------------

/// Lifecycle state of a sequence in the continuous batch.
///
/// Transitions:
/// ```text
/// Prefilling → Decoding → Finished
///                      ↑
///           Prefilling (chunked: stays Prefilling until all chunks done)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    /// Prompt tokens are being processed in chunks; not yet producing output.
    Prefilling {
        /// Index into `prompt_ids` of the next token to be processed.
        chunk_start: usize,
    },
    /// All prompt tokens have been processed; generating output tokens one-by-one.
    Decoding,
    /// Generation is complete (EOS reached, max_new_tokens hit, or preempted).
    Finished(FinishReason),
}

/// Why a sequence stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Model produced the EOS token.
    Eos,
    /// `max_new_tokens` budget exhausted.
    MaxLength,
    /// Scheduler evicted the sequence under memory pressure.
    Preempted,
}

// ---------------------------------------------------------------------------
// AdapterKey
// ---------------------------------------------------------------------------

/// Identifier for a LoRA adapter.
///
/// `None` in [`Sequence::adapter_id`] means the base model is used.
/// Sequences with the same `AdapterKey` can share a sub-batch matrix multiply;
/// sequences with different keys must run as separate sub-batches (ADR-048
/// §LoRA adapter grouping).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdapterKey(pub String);

// ---------------------------------------------------------------------------
// Sequence
// ---------------------------------------------------------------------------

/// All per-sequence state required by the continuous batching engine.
///
/// Owned by [`SequenceManager`].  The scheduler references sequences by
/// [`SeqId`]; only the manager directly mutates sequence state.
#[derive(Debug)]
pub struct Sequence {
    /// Stable identifier.
    pub id: SeqId,

    /// Original prompt token IDs.  Immutable after construction.
    pub prompt_ids: Vec<u32>,

    /// Tokens generated so far (not including prompt).
    pub generated_ids: Vec<u32>,

    /// Sampling configuration for this sequence.
    pub sampling: SamplingConfig,

    /// LoRA adapter to use, or `None` for base model.
    pub adapter_id: Option<AdapterKey>,

    /// Maximum tokens to generate (hard cap enforced by the engine).
    pub max_new_tokens: usize,

    /// Current lifecycle state.
    pub state: SequenceState,
}

impl Sequence {
    /// Create a new sequence in `Prefilling` state starting at token 0.
    pub fn new(
        id: SeqId,
        prompt_ids: Vec<u32>,
        sampling: SamplingConfig,
        adapter_id: Option<AdapterKey>,
        max_new_tokens: usize,
    ) -> Self {
        Self {
            id,
            prompt_ids,
            generated_ids: Vec::new(),
            sampling,
            adapter_id,
            max_new_tokens,
            state: SequenceState::Prefilling { chunk_start: 0 },
        }
    }

    /// Total tokens processed so far (prompt tokens already prefilled + generated).
    #[inline]
    pub fn position(&self) -> usize {
        match self.state {
            SequenceState::Prefilling { chunk_start } => chunk_start,
            SequenceState::Decoding | SequenceState::Finished(_) => {
                self.prompt_ids.len() + self.generated_ids.len()
            }
        }
    }

    /// Whether this sequence is still active (not finished).
    #[inline]
    pub fn is_active(&self) -> bool {
        !matches!(self.state, SequenceState::Finished(_))
    }

    /// Advance the prefill cursor by `chunk_len` tokens.
    ///
    /// If the cursor reaches the end of the prompt, transitions to `Decoding`.
    pub fn advance_prefill(&mut self, chunk_len: usize) {
        if let SequenceState::Prefilling { chunk_start } = self.state {
            let new_start = chunk_start + chunk_len;
            if new_start >= self.prompt_ids.len() {
                self.state = SequenceState::Decoding;
            } else {
                self.state = SequenceState::Prefilling {
                    chunk_start: new_start,
                };
            }
        }
    }

    /// Append a newly generated token and check termination conditions.
    ///
    /// Returns `true` if the sequence should be finished after this token.
    pub fn push_token(&mut self, token_id: u32, eos_token_id: Option<u32>) -> bool {
        self.generated_ids.push(token_id);
        let done_eos = eos_token_id == Some(token_id);
        let done_len = self.generated_ids.len() >= self.max_new_tokens;
        if done_eos {
            self.state = SequenceState::Finished(FinishReason::Eos);
            return true;
        }
        if done_len {
            self.state = SequenceState::Finished(FinishReason::MaxLength);
            return true;
        }
        false
    }

    /// Mark the sequence as preempted (evicted under memory pressure).
    pub fn preempt(&mut self) {
        self.state = SequenceState::Finished(FinishReason::Preempted);
    }
}

// ---------------------------------------------------------------------------
// SequenceManager
// ---------------------------------------------------------------------------

/// Owns all live sequences and their per-sequence KV page tables.
///
/// This is the external multi-sequence page-table owner described in ADR-047
/// (`paged.rs:247-251`): `PagedKVCache` tracks one `PageTable`; we maintain a
/// separate `PageTable` per sequence here and interact with a shared
/// `PagePool` (via `PagedKVCache` accessor methods exposed on `PageTable`).
///
/// Design note: page *allocation* from the global pool is still coordinated by
/// the caller (the worker/scheduler), which has visibility into total free pages.
/// `SequenceManager` only tracks the logical mapping for each sequence.
#[derive(Debug, Default)]
pub struct SequenceManager {
    sequences: HashMap<SeqId, Sequence>,
    /// Per-sequence KV page tables.  Key matches `sequences`.
    page_tables: HashMap<SeqId, PageTable>,
    /// Monotonically increasing counter for new [`SeqId`]s.
    next_id: u64,
}

impl SequenceManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a new [`SeqId`] without adding a sequence.
    pub fn next_id(&mut self) -> SeqId {
        SeqId::next(&mut self.next_id)
    }

    /// Add a sequence with a fresh, empty page table.
    ///
    /// `page_size` must match the `PagePool` page size.
    pub fn add(&mut self, seq: Sequence, page_size: usize) {
        let id = seq.id;
        self.page_tables.insert(id, PageTable::new(page_size));
        self.sequences.insert(id, seq);
    }

    /// Remove a sequence and its page table.
    ///
    /// Returns the page table so the caller can return physical pages to the
    /// `PagePool`.
    pub fn remove(&mut self, id: SeqId) -> Option<(Sequence, PageTable)> {
        match (self.sequences.remove(&id), self.page_tables.remove(&id)) {
            (Some(seq), Some(table)) => Some((seq, table)),
            _ => None,
        }
    }

    /// Immutable access to a sequence.
    #[inline]
    pub fn get(&self, id: SeqId) -> Option<&Sequence> {
        self.sequences.get(&id)
    }

    /// Mutable access to a sequence.
    #[inline]
    pub fn get_mut(&mut self, id: SeqId) -> Option<&mut Sequence> {
        self.sequences.get_mut(&id)
    }

    /// Immutable access to a sequence's page table.
    #[inline]
    pub fn page_table(&self, id: SeqId) -> Option<&PageTable> {
        self.page_tables.get(&id)
    }

    /// Mutable access to a sequence's page table.
    #[inline]
    pub fn page_table_mut(&mut self, id: SeqId) -> Option<&mut PageTable> {
        self.page_tables.get_mut(&id)
    }

    /// IDs of all sequences in a given state (by discriminant match).
    pub fn ids_in_state_prefilling(&self) -> Vec<SeqId> {
        self.sequences
            .values()
            .filter(|s| matches!(s.state, SequenceState::Prefilling { .. }))
            .map(|s| s.id)
            .collect()
    }

    /// IDs of all sequences in `Decoding` state.
    pub fn ids_decoding(&self) -> Vec<SeqId> {
        self.sequences
            .values()
            .filter(|s| s.state == SequenceState::Decoding)
            .map(|s| s.id)
            .collect()
    }

    /// IDs of all sequences in `Finished` state.
    pub fn ids_finished(&self) -> Vec<SeqId> {
        self.sequences
            .values()
            .filter(|s| matches!(s.state, SequenceState::Finished(_)))
            .map(|s| s.id)
            .collect()
    }

    /// Total number of live sequences (any state).
    #[inline]
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// True when no sequences are tracked.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
}

#[cfg(test)]
mod tests;

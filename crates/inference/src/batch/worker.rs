//! Continuous batching loop and GDN state pool (ADR-048).
//!
//! [`GdnStatePool`] pre-allocates per-sequence GDN recurrent state buffers.
//! On Apple Silicon these are plain `Vec<f32>` allocations (no Metal buffers
//! yet; Metal integration is Phase 2). Each slot holds the full multi-layer
//! GDN state for one concurrent sequence.
//!
//! [`BatchWorker`] drives the iteration loop:
//! 1. Drain evictions notified by the scheduler.
//! 2. Admit newly enqueued sequences into [`SequenceManager`].
//! 3. Call [`Scheduler::select_batch`] with current state counts.
//! 4. Execute one prefill chunk per prefilling sequence.
//! 5. Execute one decode step per decoding sequence.
//! 6. Evict finished sequences and return their KV pages to the pool.
//!
//! The actual forward pass (transformer layers, KV cache writes, logit
//! projection) is provided by a caller-supplied [`ForwardFn`] closure, keeping
//! this module independent of model architecture.

use std::collections::{HashMap, VecDeque};

use crate::batch::config::BatchConfig;
use crate::batch::scheduler::{FifoScheduler, Scheduler};
use crate::batch::sequence::{
    AdapterKey, FinishReason, SeqId, Sequence, SequenceManager, SequenceState,
};
use crate::kv_cache::{PagePool, PagedKVCacheConfig};
use crate::sampling::{Sampler, SamplingConfig};

// ---------------------------------------------------------------------------
// GdnStatePool
// ---------------------------------------------------------------------------

/// Per-sequence GDN recurrent state slot (CPU buffers, Phase 1).
///
/// Each slot holds `num_gdn_layers` pairs of (s_matrix, conv_buffer) that
/// parallel the `GatedDeltaNetState` type without taking a direct dependency
/// on the model-specific config.
///
/// Memory layout per slot:
/// - s_matrices: `num_gdn_layers × heads × key_dim × value_dim` f32 values
/// - conv_buffers: `num_gdn_layers × conv_dim × (kernel_size - 1)` f32 values
#[derive(Debug)]
pub struct GdnStatePool {
    /// Pre-allocated s_matrix buffers indexed by slot.
    s_matrices: Vec<Vec<f32>>,
    /// Pre-allocated conv buffers indexed by slot.
    conv_buffers: Vec<Vec<f32>>,
    /// Available (unused) slot indices.
    free_slots: VecDeque<usize>,
    /// Mapping from sequence ID to allocated slot.
    seq_to_slot: HashMap<SeqId, usize>,
    /// Total capacity (number of slots).
    capacity: usize,
}

impl GdnStatePool {
    /// Allocate a pool with `capacity` slots.
    ///
    /// `s_floats_per_slot` and `conv_floats_per_slot` are the flat buffer
    /// sizes for the full model (all GDN layers combined):
    /// - `s_floats_per_slot` = Σ_layers (num_heads × key_dim × value_dim)
    /// - `conv_floats_per_slot` = Σ_layers (conv_dim × (kernel_size - 1))
    pub fn new(capacity: usize, s_floats_per_slot: usize, conv_floats_per_slot: usize) -> Self {
        let s_matrices = (0..capacity)
            .map(|_| vec![0.0f32; s_floats_per_slot])
            .collect();
        let conv_buffers = (0..capacity)
            .map(|_| vec![0.0f32; conv_floats_per_slot])
            .collect();
        let free_slots = (0..capacity).collect();
        Self {
            s_matrices,
            conv_buffers,
            free_slots,
            seq_to_slot: HashMap::new(),
            capacity,
        }
    }

    /// Allocate a slot for a sequence. Returns `None` when the pool is full.
    pub fn alloc(&mut self, seq_id: SeqId) -> Option<usize> {
        let slot = self.free_slots.pop_front()?;
        self.seq_to_slot.insert(seq_id, slot);
        Some(slot)
    }

    /// Release the slot for a sequence back to the pool, zeroing the buffers.
    pub fn free(&mut self, seq_id: SeqId) {
        if let Some(slot) = self.seq_to_slot.remove(&seq_id) {
            self.s_matrices[slot].fill(0.0);
            self.conv_buffers[slot].fill(0.0);
            self.free_slots.push_back(slot);
        }
    }

    /// Slot index for a given sequence, if allocated.
    #[inline]
    pub fn slot_of(&self, seq_id: SeqId) -> Option<usize> {
        self.seq_to_slot.get(&seq_id).copied()
    }

    /// Mutable access to the s_matrix buffer for a slot.
    #[inline]
    pub fn s_matrix_mut(&mut self, slot: usize) -> &mut [f32] {
        &mut self.s_matrices[slot]
    }

    /// Mutable access to the conv buffer for a slot.
    #[inline]
    pub fn conv_buffer_mut(&mut self, slot: usize) -> &mut [f32] {
        &mut self.conv_buffers[slot]
    }

    /// Number of free slots.
    #[inline]
    pub fn free_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Total pool capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// ForwardFn trait alias
// ---------------------------------------------------------------------------

/// Input descriptor for one sequence in a batch step.
#[derive(Debug)]
pub struct BatchStepInput<'a> {
    pub seq_id: SeqId,
    /// Token IDs to process this step (chunk for prefill, single token for decode).
    pub token_ids: &'a [u32],
    /// Starting position in the full sequence (for RoPE and causal masking).
    pub start_pos: usize,
    /// Index into the GDN state pool for this sequence.
    pub gdn_slot: usize,
    /// LoRA adapter to use, if any.
    pub adapter_id: Option<&'a AdapterKey>,
}

/// Output from one forward step for a single sequence.
#[derive(Debug)]
pub struct BatchStepOutput {
    pub seq_id: SeqId,
    /// Logits for the last token, length = vocab_size.
    pub logits: Vec<f32>,
}

// ---------------------------------------------------------------------------
// InferenceRequest / InferenceToken
// ---------------------------------------------------------------------------

/// Submitted by a caller to request text generation.
#[derive(Debug)]
pub struct InferenceRequest {
    /// Prompt token IDs.
    pub prompt_ids: Vec<u32>,
    /// Sampling configuration.
    pub sampling: SamplingConfig,
    /// None = base model; Some = LoRA adapter key.
    pub lora_adapter: Option<AdapterKey>,
    /// Hard cap on generated tokens.
    pub max_new_tokens: usize,
}

/// A single generated token streamed back to the caller.
#[derive(Debug, Clone)]
pub struct InferenceToken {
    pub seq_id: SeqId,
    pub token_id: u32,
    pub finished: bool,
    pub finish_reason: Option<FinishReason>,
}

// ---------------------------------------------------------------------------
// BatchWorker
// ---------------------------------------------------------------------------

/// Drives the continuous batching iteration loop (Phase 1).
///
/// The worker owns:
/// - A [`SequenceManager`] for per-sequence token state and KV page tables.
/// - A [`GdnStatePool`] for per-sequence GDN recurrent state.
/// - A shared [`PagePool`] for KV page allocation.
/// - A [`FifoScheduler`] for admission and batch selection.
///
/// On each call to [`BatchWorker::step`], it runs one full iteration:
/// select batch → run prefill chunks → run decode steps → evict finished.
///
/// The actual model forward pass is injected via a closure to keep this module
/// independent of model architecture (CPU vs Metal, Qwen3.5 vs future models).
pub struct BatchWorker {
    config: BatchConfig,
    seq_manager: SequenceManager,
    gdn_pool: GdnStatePool,
    kv_pool: PagePool,
    scheduler: FifoScheduler,
    /// Per-sequence samplers (one Sampler per sequence, indexed by SeqId).
    samplers: HashMap<SeqId, Sampler>,
    /// EOS token id for the current model.
    eos_token_id: Option<u32>,
    /// Generated tokens ready to stream out, ordered by iteration.
    output_buffer: VecDeque<InferenceToken>,
}

impl BatchWorker {
    /// Create a new batch worker.
    ///
    /// # Arguments
    ///
    /// * `config` — Batching parameters (max_batch_size, chunk_size, etc.).
    /// * `kv_pool_config` — Paged KV cache config; the worker creates its own
    ///   `PagePool` from this.
    /// * `s_floats_per_slot` — GDN s_matrix floats per concurrent sequence.
    /// * `conv_floats_per_slot` — GDN conv buffer floats per concurrent sequence.
    /// * `eos_token_id` — Token ID that signals end-of-sequence, if any.
    pub fn new(
        config: BatchConfig,
        kv_pool_config: PagedKVCacheConfig,
        s_floats_per_slot: usize,
        conv_floats_per_slot: usize,
        eos_token_id: Option<u32>,
    ) -> Self {
        let floats_per_page = kv_pool_config.floats_per_page_pub();
        let kv_pool = PagePool::new(kv_pool_config.max_pages, floats_per_page);
        let gdn_pool = GdnStatePool::new(
            config.max_batch_size,
            s_floats_per_slot,
            conv_floats_per_slot,
        );
        let scheduler = FifoScheduler::new(config.clone());
        Self {
            config,
            seq_manager: SequenceManager::new(),
            gdn_pool,
            kv_pool,
            scheduler,
            samplers: HashMap::new(),
            eos_token_id,
            output_buffer: VecDeque::new(),
        }
    }

    /// Submit a new inference request.
    ///
    /// Returns the [`SeqId`] assigned to this request. The actual forward pass
    /// will begin on the next call to [`step`].
    ///
    /// Returns `None` if the request is invalid (empty prompt, exceeds
    /// `max_seq_len`).
    pub fn submit(&mut self, request: InferenceRequest) -> Option<SeqId> {
        if request.prompt_ids.is_empty() {
            return None;
        }
        let total_len = request.prompt_ids.len() + request.max_new_tokens;
        if total_len > self.config.max_seq_len {
            return None;
        }

        let id = self.seq_manager.next_id();
        let seq = Sequence::new(
            id,
            request.prompt_ids,
            request.sampling.clone(),
            request.lora_adapter,
            request.max_new_tokens,
        );
        let page_size = self.kv_pool_page_size();
        self.seq_manager.add(seq, page_size);
        let sampler = Sampler::new(request.sampling);
        self.samplers.insert(id, sampler);
        self.scheduler.enqueue(id);
        Some(id)
    }

    /// Run one continuous batching iteration.
    ///
    /// # Arguments
    ///
    /// * `forward_fn` — Called once per sequence per iteration with a
    ///   [`BatchStepInput`]. Must return logits for the last token.
    ///
    /// # Returns
    ///
    /// A slice of [`InferenceToken`] events produced this iteration (may be
    /// empty if all active sequences are still prefilling their first chunk).
    pub fn step(
        &mut self,
        mut forward_fn: impl FnMut(BatchStepInput<'_>, &mut GdnStatePool) -> Vec<f32>,
    ) -> Vec<InferenceToken> {
        // 1. Collect current waiting / running IDs.
        let mut waiting = self.seq_manager.ids_in_state_prefilling();
        let mut running = self.seq_manager.ids_decoding();

        // Sort for deterministic ordering.
        waiting.sort();
        running.sort();

        // 2. Ask the scheduler for the iteration batch.
        let kv_free = self.kv_pool.free_count();
        let gdn_free = self.gdn_pool.free_count();
        let decision = self
            .scheduler
            .select_batch(&waiting, &running, kv_free, gdn_free);

        // 3. Evict any sequences marked for eviction (Phase 1: always empty).
        for seq_id in &decision.evict {
            self.evict_sequence(*seq_id, FinishReason::Preempted);
            self.scheduler.on_preempt(*seq_id);
        }

        // 4. Process prefill entries.
        //    Entries from newly admitted sequences have chunk_start = 0.
        //    Entries for sequences already in the manager have chunk_start = usize::MAX
        //    (sentinel from FifoScheduler) — we substitute the real value from the manager.
        for (seq_id, sentinel_start, chunk_len) in &decision.prefill {
            let seq_id = *seq_id;

            // Ensure a GDN slot exists for this sequence.
            if self.gdn_pool.slot_of(seq_id).is_none() && self.gdn_pool.alloc(seq_id).is_none() {
                // No GDN slot available: skip this sequence this iteration.
                continue;
            }

            // Determine the real chunk start from the sequence manager.
            let (real_start, real_len) = {
                let Some(seq) = self.seq_manager.get(seq_id) else {
                    continue;
                };
                let start = if *sentinel_start == usize::MAX {
                    // Continuation chunk: read from sequence state.
                    match seq.state {
                        SequenceState::Prefilling { chunk_start } => chunk_start,
                        _ => continue, // already decoding or finished
                    }
                } else {
                    *sentinel_start
                };
                let prompt_len = seq.prompt_ids.len();
                let remaining = prompt_len.saturating_sub(start);
                let len = remaining.min(*chunk_len);
                (start, len)
            };

            if real_len == 0 {
                continue;
            }

            // Run forward for the chunk.
            let Some(gdn_slot) = self.gdn_pool.slot_of(seq_id) else {
                continue;
            };

            let logits = {
                let Some(seq) = self.seq_manager.get(seq_id) else {
                    continue;
                };
                let chunk = &seq.prompt_ids[real_start..real_start + real_len];
                let adapter = seq.adapter_id.as_ref();
                let input = BatchStepInput {
                    seq_id,
                    token_ids: chunk,
                    start_pos: real_start,
                    gdn_slot,
                    adapter_id: adapter,
                };
                forward_fn(input, &mut self.gdn_pool)
            };

            // Advance the prefill cursor.
            if let Some(seq) = self.seq_manager.get_mut(seq_id) {
                seq.advance_prefill(real_len);
            }

            // If this chunk finishes the prefill, sample the first decode token.
            let just_finished_prefill = self
                .seq_manager
                .get(seq_id)
                .is_some_and(|s| s.state == SequenceState::Decoding);

            if just_finished_prefill {
                let token_id = {
                    let Some(sampler) = self.samplers.get_mut(&seq_id) else {
                        continue;
                    };
                    sampler.sample(&logits)
                };
                let done = self
                    .seq_manager
                    .get_mut(seq_id)
                    .is_some_and(|s| s.push_token(token_id, self.eos_token_id));
                let finish_reason = if done {
                    self.seq_manager.get(seq_id).and_then(|s| match s.state {
                        SequenceState::Finished(r) => Some(r),
                        _ => None,
                    })
                } else {
                    None
                };
                self.output_buffer.push_back(InferenceToken {
                    seq_id,
                    token_id,
                    finished: done,
                    finish_reason,
                });
            }
        }

        // 5. Process decode steps.
        for seq_id in &decision.decode {
            let seq_id = *seq_id;

            let Some(gdn_slot) = self.gdn_pool.slot_of(seq_id) else {
                continue;
            };

            let logits = {
                let Some(seq) = self.seq_manager.get(seq_id) else {
                    continue;
                };
                let Some(&last_token) = seq.generated_ids.last() else {
                    continue; // should not happen in Decoding state
                };
                let start_pos = seq.position().saturating_sub(1);
                let token_buf = std::slice::from_ref(&last_token);
                let adapter = seq.adapter_id.as_ref();
                let input = BatchStepInput {
                    seq_id,
                    token_ids: token_buf,
                    start_pos,
                    gdn_slot,
                    adapter_id: adapter,
                };
                forward_fn(input, &mut self.gdn_pool)
            };

            let token_id = {
                let Some(sampler) = self.samplers.get_mut(&seq_id) else {
                    continue;
                };
                sampler.sample(&logits)
            };

            let done = self
                .seq_manager
                .get_mut(seq_id)
                .is_some_and(|s| s.push_token(token_id, self.eos_token_id));
            let finish_reason = if done {
                self.seq_manager.get(seq_id).and_then(|s| match s.state {
                    SequenceState::Finished(r) => Some(r),
                    _ => None,
                })
            } else {
                None
            };
            self.output_buffer.push_back(InferenceToken {
                seq_id,
                token_id,
                finished: done,
                finish_reason,
            });
        }

        // 6. Evict finished sequences.
        let finished_ids: Vec<SeqId> = self.seq_manager.ids_finished();
        for seq_id in finished_ids {
            self.evict_sequence(seq_id, FinishReason::Eos); // reason already set in state
        }

        // 7. Drain and return this iteration's output tokens.
        self.output_buffer.drain(..).collect()
    }

    /// Number of active sequences (prefilling + decoding).
    #[inline]
    pub fn active_count(&self) -> usize {
        self.seq_manager.len()
    }

    /// True when no sequences are being processed and the admission queue is empty.
    #[inline]
    pub fn is_idle(&self) -> bool {
        self.seq_manager.is_empty() && self.scheduler.waiting_count() == 0
    }

    // --- Internal helpers ---

    fn evict_sequence(&mut self, seq_id: SeqId, _reason: FinishReason) {
        if let Some((_, table)) = self.seq_manager.remove(seq_id) {
            // Return all KV pages to the pool.
            for &phys in table.physical_pages() {
                self.kv_pool.free(phys);
            }
        }
        self.gdn_pool.free(seq_id);
        self.samplers.remove(&seq_id);
    }

    /// Page size used by the KV pool (needed when creating PageTables).
    fn kv_pool_page_size(&self) -> usize {
        // The PagePool doesn't directly expose page_size; we track it via config.
        // In Phase 1 the KV pool page size defaults to 256 tokens.
        // If needed, store it explicitly in the worker.
        256
    }
}

// ---------------------------------------------------------------------------
// PagedKVCacheConfig extension — expose floats_per_page publicly for worker
// ---------------------------------------------------------------------------

/// Extension trait to expose `floats_per_page` calculation to `BatchWorker`.
///
/// `PagedKVCacheConfig::floats_per_page` is `pub(crate)` in paged.rs; this
/// trait provides the equivalent calculation for the worker.
pub trait PagedKVCacheConfigExt {
    fn floats_per_page_pub(&self) -> usize;
}

impl PagedKVCacheConfigExt for crate::kv_cache::PagedKVCacheConfig {
    fn floats_per_page_pub(&self) -> usize {
        self.num_layers * 2 * self.page_size * self.kv_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::config::BatchConfig;
    use crate::kv_cache::{EvictionPolicy, PagedKVCacheConfig};
    use crate::sampling::SamplingConfig;

    fn test_kv_config() -> PagedKVCacheConfig {
        PagedKVCacheConfig {
            page_size: 4,
            max_pages: 64,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            eviction: EvictionPolicy::None,
        }
    }

    fn test_worker() -> BatchWorker {
        let config = BatchConfig {
            max_batch_size: 4,
            max_seq_len: 64,
            chunk_size: 8,
            prefill_reserve_pages: 2,
        };
        BatchWorker::new(
            config,
            test_kv_config(),
            16,      // s_floats_per_slot (tiny for tests)
            8,       // conv_floats_per_slot
            Some(2), // eos = token 2
        )
    }

    fn dummy_logits(winner: u32, vocab: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; vocab];
        if (winner as usize) < vocab {
            v[winner as usize] = 10.0;
        }
        v
    }

    // --- GdnStatePool ---

    #[test]
    fn gdn_pool_alloc_and_free() {
        let mut pool = GdnStatePool::new(2, 4, 2);
        assert_eq!(pool.free_count(), 2);
        let s0 = pool.alloc(SeqId(0));
        assert!(s0.is_some());
        assert_eq!(pool.free_count(), 1);
        let s1 = pool.alloc(SeqId(1));
        assert!(s1.is_some());
        assert_eq!(pool.free_count(), 0);
        // Pool full — third alloc fails.
        assert!(pool.alloc(SeqId(2)).is_none());
        // Free slot 0.
        pool.free(SeqId(0));
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn gdn_pool_free_zeroes_buffers() {
        let mut pool = GdnStatePool::new(1, 4, 2);
        let slot = pool.alloc(SeqId(0)).unwrap();
        pool.s_matrix_mut(slot).fill(1.0);
        pool.conv_buffer_mut(slot).fill(2.0);
        pool.free(SeqId(0));
        // Re-allocate the same slot.
        let slot2 = pool.alloc(SeqId(1)).unwrap();
        assert_eq!(slot2, slot);
        assert!(pool.s_matrix_mut(slot2).iter().all(|&x| x == 0.0));
        assert!(pool.conv_buffer_mut(slot2).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn gdn_pool_slot_of() {
        let mut pool = GdnStatePool::new(2, 4, 2);
        pool.alloc(SeqId(5)).unwrap();
        assert!(pool.slot_of(SeqId(5)).is_some());
        assert!(pool.slot_of(SeqId(99)).is_none());
    }

    // --- BatchWorker submission ---

    #[test]
    fn submit_valid_request_returns_seq_id() {
        let mut worker = test_worker();
        let req = InferenceRequest {
            prompt_ids: vec![1, 2, 3],
            sampling: SamplingConfig::greedy(),
            lora_adapter: None,
            max_new_tokens: 4,
        };
        let id = worker.submit(req);
        assert!(id.is_some());
    }

    #[test]
    fn submit_empty_prompt_returns_none() {
        let mut worker = test_worker();
        let req = InferenceRequest {
            prompt_ids: vec![],
            sampling: SamplingConfig::greedy(),
            lora_adapter: None,
            max_new_tokens: 4,
        };
        assert!(worker.submit(req).is_none());
    }

    #[test]
    fn submit_too_long_returns_none() {
        let mut worker = test_worker(); // max_seq_len = 64
        let req = InferenceRequest {
            prompt_ids: vec![0u32; 60],
            sampling: SamplingConfig::greedy(),
            lora_adapter: None,
            max_new_tokens: 10, // 60 + 10 = 70 > 64
        };
        assert!(worker.submit(req).is_none());
    }

    // --- BatchWorker step: sequence lifecycle ---

    #[test]
    fn step_no_requests_returns_empty() {
        let mut worker = test_worker();
        let out = worker.step(|_input, _pool| vec![0.0f32; 8]);
        assert!(out.is_empty());
    }

    #[test]
    fn step_single_request_prefill_then_decode() {
        let mut worker = test_worker(); // chunk_size = 8, max_new_tokens cap
        // Prompt of 4 tokens (< chunk_size: one chunk covers it all).
        worker.submit(InferenceRequest {
            prompt_ids: vec![10, 11, 12, 13],
            sampling: SamplingConfig::greedy(),
            lora_adapter: None,
            max_new_tokens: 2,
        });

        // First step: prefill should complete (4 tokens < chunk_size=8).
        // forward_fn returns logits that vote for token 5.
        let step1 = worker.step(|_input, _pool| dummy_logits(5, 8));
        // After prefill completes, first decode token (5) is emitted.
        assert_eq!(step1.len(), 1);
        assert_eq!(step1[0].token_id, 5);
        assert!(!step1[0].finished);

        // Second step: decode step 1 → token 6.
        let step2 = worker.step(|_input, _pool| dummy_logits(6, 8));
        assert_eq!(step2.len(), 1);
        assert_eq!(step2[0].token_id, 6);
        // max_new_tokens = 2, second generated token → finished
        assert!(step2[0].finished);
        assert_eq!(step2[0].finish_reason, Some(FinishReason::MaxLength));
    }

    #[test]
    fn step_eos_finishes_sequence() {
        let mut worker = test_worker(); // eos = token 2
        worker.submit(InferenceRequest {
            prompt_ids: vec![10],
            sampling: SamplingConfig::greedy(),
            lora_adapter: None,
            max_new_tokens: 10,
        });
        // Prefill → emit token 2 (EOS).
        let out = worker.step(|_input, _pool| dummy_logits(2, 8));
        assert_eq!(out.len(), 1);
        assert!(out[0].finished);
        assert_eq!(out[0].finish_reason, Some(FinishReason::Eos));
    }

    #[test]
    fn step_evicts_finished_sequence() {
        let mut worker = test_worker();
        worker.submit(InferenceRequest {
            prompt_ids: vec![10],
            sampling: SamplingConfig::greedy(),
            lora_adapter: None,
            max_new_tokens: 1,
        });
        // After one step (prefill+first token), max_new_tokens=1 → finished.
        worker.step(|_input, _pool| dummy_logits(5, 8));
        // Sequence should be evicted; worker is idle.
        assert!(worker.is_idle());
    }

    #[test]
    fn step_chunked_prefill_multiple_chunks() {
        let mut worker = BatchWorker::new(
            BatchConfig {
                max_batch_size: 4,
                max_seq_len: 128,
                chunk_size: 3, // small chunk to force multiple prefill steps
                prefill_reserve_pages: 0,
            },
            test_kv_config(),
            16,
            8,
            Some(99), // eos = 99
        );
        // Prompt of 7 tokens requires ceil(7/3) = 3 prefill steps.
        worker.submit(InferenceRequest {
            prompt_ids: vec![1, 2, 3, 4, 5, 6, 7],
            sampling: SamplingConfig::greedy(),
            lora_adapter: None,
            max_new_tokens: 2,
        });

        // Step 1: chunk [0..3] → still prefilling (4 tokens remain).
        let s1 = worker.step(|_input, _pool| dummy_logits(10, 12));
        assert!(s1.is_empty(), "no output yet during prefill");

        // Step 2: chunk [3..6] → still prefilling (1 token remain).
        let s2 = worker.step(|_input, _pool| dummy_logits(10, 12));
        assert!(s2.is_empty());

        // Step 3: chunk [6..7] → prefill complete, first token emitted.
        let s3 = worker.step(|_input, _pool| dummy_logits(10, 12));
        assert_eq!(s3.len(), 1);
        assert_eq!(s3[0].token_id, 10);
    }

    #[test]
    fn step_multiple_concurrent_sequences() {
        let mut worker = test_worker();
        worker.submit(InferenceRequest {
            prompt_ids: vec![1],
            sampling: SamplingConfig::greedy(),
            lora_adapter: None,
            max_new_tokens: 1,
        });
        worker.submit(InferenceRequest {
            prompt_ids: vec![2],
            sampling: SamplingConfig::greedy(),
            lora_adapter: None,
            max_new_tokens: 1,
        });
        // Both sequences admitted simultaneously; both should prefill this step.
        let out = worker.step(|_input, _pool| dummy_logits(7, 8));
        // Both emit their first (and only, max_new_tokens=1) token.
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|t| t.token_id == 7));
        assert!(out.iter().all(|t| t.finished));
        assert!(worker.is_idle());
    }

    // --- PagedKVCacheConfigExt ---

    #[test]
    fn floats_per_page_ext() {
        let cfg = test_kv_config(); // page=4, layers=2, kv_heads=2, head_dim=4
        // kv_dim = 2*4 = 8; floats = 2 * 2 * 4 * 8 = 128
        assert_eq!(cfg.floats_per_page_pub(), 128);
    }
}

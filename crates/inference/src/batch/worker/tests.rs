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
    BatchWorker::try_new(
        config,
        test_kv_config(),
        16,      // s_floats_per_slot (tiny for tests)
        8,       // conv_floats_per_slot
        Some(2), // eos = token 2
    )
    .expect("valid test worker config must succeed")
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
    let mut pool = GdnStatePool::try_new(2, 4, 2).expect("valid pool geometry");
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
fn gdn_pool_alloc_duplicate_with_spare_capacity_returns_existing_slot() {
    let mut pool = GdnStatePool::try_new(2, 4, 2).expect("valid pool geometry");
    let slot = pool.alloc(SeqId(7)).expect("first allocation fits");
    pool.s_matrix_mut(slot).fill(1.0);
    pool.conv_buffer_mut(slot).fill(2.0);
    assert_eq!(pool.free_count(), 1);

    let duplicate = pool.alloc(SeqId(7));

    assert_eq!(duplicate, Some(slot));
    assert_eq!(pool.slot_of(SeqId(7)), Some(slot));
    assert_eq!(pool.free_count(), 1);
    assert!(pool.s_matrix_mut(slot).iter().all(|&x| x == 1.0));
    assert!(pool.conv_buffer_mut(slot).iter().all(|&x| x == 2.0));
}

#[test]
fn gdn_pool_alloc_duplicate_on_full_pool_returns_existing_slot() {
    let mut pool = GdnStatePool::try_new(1, 4, 2).expect("valid pool geometry");
    let slot = pool.alloc(SeqId(7)).expect("first allocation fits");
    assert_eq!(pool.free_count(), 0);

    let duplicate = pool.alloc(SeqId(7));

    assert_eq!(duplicate, Some(slot));
    assert_eq!(pool.slot_of(SeqId(7)), Some(slot));
    assert_eq!(pool.free_count(), 0);
}

#[test]
fn gdn_pool_all_slots_reusable_after_free() {
    let mut pool = GdnStatePool::try_new(2, 4, 2).expect("valid pool geometry");
    let first = pool.alloc(SeqId(7)).expect("first allocation fits");
    let duplicate = pool.alloc(SeqId(7)).expect("duplicate allocation succeeds");
    assert_eq!(duplicate, first);

    pool.free(SeqId(7));

    assert_eq!(pool.free_count(), pool.capacity());
    let mut slots = [
        pool.alloc(SeqId(8)).expect("first freed slot is reusable"),
        pool.alloc(SeqId(9)).expect("second freed slot is reusable"),
    ];
    slots.sort_unstable();
    assert_eq!(slots, [0, 1]);
    assert_eq!(pool.free_count(), 0);
}

#[test]
fn gdn_pool_free_zeroes_buffers() {
    let mut pool = GdnStatePool::try_new(1, 4, 2).expect("valid pool geometry");
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
    let mut pool = GdnStatePool::try_new(2, 4, 2).expect("valid pool geometry");
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

/// `prompt_ids.len() + max_new_tokens` must not overflow `usize`: before the fix,
/// `max_new_tokens = usize::MAX` wrapped the unchecked `+` to a small value in
/// release builds (bypassing the `total_len > max_seq_len` guard and admitting an
/// absurd budget) or panicked in debug. `checked_add` must reject this cleanly on
/// both profiles. Mutation-sensitive: reverting `checked_add` back to `+` makes
/// this test panic on overflow in a debug build (the profile tests run under).
#[test]
fn submit_max_new_tokens_near_usize_max_returns_none() {
    let mut worker = test_worker();
    let req = InferenceRequest {
        prompt_ids: vec![1, 2, 3],
        sampling: SamplingConfig::greedy(),
        lora_adapter: None,
        max_new_tokens: usize::MAX,
    };
    assert!(worker.submit(req).is_none());
}

/// Regression guard for #387 batch path: `submit` must seed the sampler's
/// repetition-penalty history with the prompt tokens so the FIRST sampled
/// token already penalizes any token that appeared in the prompt.
///
/// Mutation-sensitive: removing `sampler.seed_history(&request.prompt_ids)`
/// from `submit()` leaves `recent_tokens` empty at the first sample call.
/// The greedy fast-path shortcut then fires (`penalty > 1.0 &&
/// !recent_tokens.contains(&raw_best)` is true for empty history), returning
/// the un-penalized argmax (token 3) instead of the penalized winner (token 4).
#[test]
fn submit_seeds_prompt_history_for_repetition_penalty() {
    let mut worker = test_worker();
    worker.submit(InferenceRequest {
        prompt_ids: vec![3], // token 3 is in the prompt; must be penalized
        sampling: SamplingConfig {
            min_p: 0.0,
            temperature: 0.0, // greedy fast path
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 2.0,
        },
        lora_adapter: None,
        max_new_tokens: 1,
    });

    // logits: token 3 raw-wins (9.0 > 5.5) but loses after penalty (4.5 < 5.5).
    let mut logits = vec![0.0f32; 8];
    logits[3] = 9.0; // prompt token — raw winner, must be penalized
    logits[4] = 5.5; // penalty-adjusted winner

    let out = worker.step(|_input, _pool| logits.clone());
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].token_id, 4,
        "prompt token 3 must be penalized (9.0/2.0=4.5 < 5.5) so token 4 wins; \
             without seed_history in submit(), token 3 is not penalized and wins raw \
             (mutation: remove seed_history call from submit())"
    );
}

// --- BatchWorker step: sequence lifecycle ---

#[test]
fn step_no_requests_returns_empty() {
    let mut worker = test_worker();
    let out = worker.step(|_input, _pool| vec![0.0f32; 8]);
    assert!(out.is_empty());
}

#[test]
fn step_zero_token_budget_emits_nothing() {
    let mut worker = test_worker();
    assert!(
        worker
            .submit(InferenceRequest {
                prompt_ids: vec![10, 11, 12],
                sampling: SamplingConfig::greedy(),
                lora_adapter: None,
                max_new_tokens: 0,
            })
            .is_some()
    );

    let mut forward_called = false;
    let out = worker.step(|_input, _pool| {
        forward_called = true;
        dummy_logits(5, 8)
    });

    assert!(
        out.is_empty(),
        "a zero-token budget must emit no token event"
    );
    assert!(
        !forward_called,
        "a zero-token budget must finish before prefill or sampling"
    );
    assert!(worker.is_idle());
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
    let mut worker = BatchWorker::try_new(
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
    )
    .expect("valid chunked-prefill worker config must succeed");
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

// --- Overflow hardening tests (#460) ---

#[test]
fn batch_worker_try_new_overflow_floats_per_page_returns_invalid_input() {
    let config = BatchConfig {
        max_batch_size: 1,
        max_seq_len: 8,
        chunk_size: 1,
        prefill_reserve_pages: 0,
    };
    let kv_pool_config = PagedKVCacheConfig {
        page_size: 8,
        max_pages: 1,
        num_layers: usize::MAX / 16 + 2,
        num_kv_heads: 1,
        head_dim: 1,
        eviction: EvictionPolicy::None,
    };
    let r = BatchWorker::try_new(config, kv_pool_config, 1, 1, None);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "expected InvalidInput on worker KV page overflow"
    );
}

#[test]
fn batch_worker_try_new_overflow_page_pool_capacity_returns_invalid_input() {
    let config = BatchConfig {
        max_batch_size: 1,
        max_seq_len: 8,
        chunk_size: 1,
        prefill_reserve_pages: 0,
    };
    let kv_pool_config = PagedKVCacheConfig {
        page_size: 1,
        max_pages: usize::MAX,
        num_layers: 1,
        num_kv_heads: 1,
        head_dim: 1,
        eviction: EvictionPolicy::None,
    };
    let r = BatchWorker::try_new(config, kv_pool_config, 1, 1, None);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "expected InvalidInput on worker PagePool capacity overflow"
    );
}

#[test]
fn batch_worker_try_new_rejects_invalid_batch_config() {
    let invalid_configs = [
        (
            "max_batch_size",
            BatchConfig {
                max_batch_size: 0,
                ..BatchConfig::default()
            },
        ),
        (
            "max_seq_len",
            BatchConfig {
                max_seq_len: 0,
                ..BatchConfig::default()
            },
        ),
        (
            "chunk_size",
            BatchConfig {
                chunk_size: 0,
                ..BatchConfig::default()
            },
        ),
        (
            "chunk_size",
            BatchConfig {
                chunk_size: 513,
                ..BatchConfig::default()
            },
        ),
    ];

    for (field, config) in invalid_configs {
        let result = BatchWorker::try_new(config, test_kv_config(), 1, 1, None);
        let Err(InferenceError::InvalidInput(message)) = result else {
            panic!("expected InvalidInput for invalid {field}");
        };
        assert!(
            message.contains(field),
            "error for invalid {field} must identify the field, got: {message}"
        );
    }
}

// --- GdnStatePool isize::MAX boundary test (Fix 2) ---

#[test]
fn gdn_state_pool_try_new_isize_max_byte_bound_returns_invalid_input() {
    // s_floats_per_slot such that s_floats * 4 > isize::MAX; element count
    // itself fits usize. The GdnStatePool::try_new guard must catch this
    // before any Vec is allocated; without it vec![0.0f32; s_floats] panics.
    let s_floats = (isize::MAX as usize / std::mem::size_of::<f32>()) + 1;
    let r = GdnStatePool::try_new(1, s_floats, 1);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "expected InvalidInput when s_floats byte size exceeds isize::MAX, got {r:?}"
    );
}

#[test]
fn gdn_state_pool_try_new_outer_vec_capacity_bound_returns_invalid_input() {
    // capacity * size_of::<Vec<f32>>() exceeds isize::MAX while capacity *
    // size_of::<usize>() (the old guard width) does NOT; per-slot float
    // counts are 0 so the s/conv guards pass. Without the outer-vector guard
    // `new`'s collect::<Vec<Vec<f32>>>() panics with capacity overflow.
    let capacity = (isize::MAX as usize / std::mem::size_of::<Vec<f32>>()) + 1;
    let r = GdnStatePool::try_new(capacity, 0, 0);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "expected InvalidInput when outer-vector byte size exceeds isize::MAX, got {r:?}"
    );
}

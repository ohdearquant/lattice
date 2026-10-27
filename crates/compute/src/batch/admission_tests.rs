use super::*;
use crate::kv_cache::EvictionPolicy;

#[derive(Debug, PartialEq, Eq)]
struct ForwardCall {
    id: SeqId,
    start: usize,
    tokens: Vec<u32>,
    slot: usize,
}

fn worker(capacity: usize, chunk_size: usize, eos: Option<u32>) -> BatchWorker {
    BatchWorker::try_new(
        BatchConfig {
            max_batch_size: capacity,
            max_seq_len: 32,
            chunk_size,
            prefill_reserve_pages: 2,
        },
        PagedKVCacheConfig {
            page_size: 256,
            max_pages: 4,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 1,
            eviction: EvictionPolicy::None,
        },
        2,
        1,
        eos,
    )
    .unwrap()
}

fn submit(worker: &mut BatchWorker, prompt: &[u32], budget: usize) -> SeqId {
    worker
        .submit(InferenceRequest {
            prompt_ids: prompt.to_vec(),
            sampling: SamplingConfig::greedy(),
            lora_adapter: None,
            max_new_tokens: budget,
        })
        .unwrap()
}

fn record(input: &BatchStepInput<'_>, pool: &GdnStatePool, trace: &mut Vec<ForwardCall>) {
    assert_eq!(pool.slot_of(input.seq_id), Some(input.gdn_slot));
    assert!(input.gdn_slot < pool.capacity());
    assert_eq!(pool.seq_to_slot.len() + pool.free_count(), pool.capacity());
    trace.push(ForwardCall {
        id: input.seq_id,
        start: input.start_pos,
        tokens: input.token_ids.to_vec(),
        slot: input.gdn_slot,
    });
}

fn logits(winner: usize) -> Vec<f32> {
    let mut values = vec![-1.0; 8];
    values[winner] = 1.0;
    values
}

fn step(worker: &mut BatchWorker, trace: &mut Vec<ForwardCall>) -> Vec<InferenceToken> {
    worker.step(|input, pool| {
        record(&input, pool, trace);
        logits(7)
    })
}

fn events(tokens: &[InferenceToken]) -> Vec<(SeqId, u32, bool, Option<FinishReason>)> {
    tokens
        .iter()
        .map(|token| {
            (
                token.seq_id,
                token.token_id,
                token.finished,
                token.finish_reason,
            )
        })
        .collect()
}

fn calls(trace: &[ForwardCall]) -> Vec<(SeqId, usize, Vec<u32>)> {
    trace
        .iter()
        .map(|call| (call.id, call.start, call.tokens.clone()))
        .collect()
}

fn assert_drained(worker: &mut BatchWorker) {
    assert_eq!(worker.active_count(), 0);
    assert_eq!(worker.scheduler.waiting_count(), 0);
    assert_eq!(worker.gdn_pool.free_count(), worker.gdn_pool.capacity());
    assert!(worker.gdn_pool.seq_to_slot.is_empty());
    assert!(worker.samplers.is_empty());
    assert!(worker.output_buffer.is_empty());
    assert!(worker.is_idle());
    assert!(
        worker
            .step(|_, _| panic!("drained worker must not run forward"))
            .is_empty()
    );
    assert_eq!(worker.gdn_pool.free_count(), worker.gdn_pool.capacity());
    assert!(worker.is_idle());
}

#[test]
fn capacity_one_reuses_slots_after_each_completion() {
    let mut worker = worker(1, 2, None);
    let mut completed = Vec::new();
    for prompt in [[1, 2], [3, 4], [5, 6], [1, 3]] {
        let id = submit(&mut worker, &prompt, 1);
        assert_eq!(worker.scheduler.waiting_count(), 1);
        assert_eq!(worker.gdn_pool.free_count(), 1);
        let mut trace = Vec::new();
        let tokens = worker.step(|input, pool| {
            record(&input, pool, &mut trace);
            assert_eq!(pool.free_count(), 0);
            assert!(pool.s_matrix_mut(input.gdn_slot).iter().all(|&v| v == 0.0));
            assert!(
                pool.conv_buffer_mut(input.gdn_slot)
                    .iter()
                    .all(|&v| v == 0.0)
            );
            pool.s_matrix_mut(input.gdn_slot).fill(3.0);
            pool.conv_buffer_mut(input.gdn_slot).fill(4.0);
            logits(7)
        });
        assert_eq!(
            events(&tokens),
            [(id, 7, true, Some(FinishReason::MaxLength))]
        );
        assert_eq!(
            trace,
            [ForwardCall {
                id,
                start: 0,
                tokens: prompt.to_vec(),
                slot: 0,
            }]
        );
        completed.push(id);
        assert_drained(&mut worker);
    }
    assert!(completed.windows(2).all(|pair| pair[0] < pair[1]));
}

fn run_fifo_population(population: usize) {
    let mut worker = worker(2, 2, None);
    let ids: Vec<_> = (0..population)
        .map(|index| submit(&mut worker, &[1 + index as u32], 2))
        .collect();
    let mut trace = Vec::new();
    let mut expected_calls = Vec::new();
    let mut completed = Vec::new();
    for first in (0..population).step_by(2) {
        let end = (first + 2).min(population);
        let admitted = &ids[first..end];
        let tokens = step(&mut worker, &mut trace);
        assert_eq!(
            events(&tokens),
            admitted
                .iter()
                .map(|&id| (id, 7, false, None))
                .collect::<Vec<_>>()
        );
        assert_eq!(worker.scheduler.waiting_count(), population - end);
        assert_eq!(worker.gdn_pool.free_count(), 2 - admitted.len());
        for &queued in &ids[end..] {
            assert_eq!(worker.gdn_pool.slot_of(queued), None);
            assert_eq!(
                worker.seq_manager.get(queued).unwrap().state,
                SequenceState::Prefilling { chunk_start: 0 }
            );
        }
        let slots: Vec<_> = admitted
            .iter()
            .map(|&id| worker.gdn_pool.slot_of(id).unwrap())
            .collect();
        assert_eq!(
            slots.iter().collect::<std::collections::HashSet<_>>().len(),
            admitted.len()
        );
        for index in first..end {
            expected_calls.push((ids[index], 0, vec![1 + index as u32]));
        }
        assert_eq!(calls(&trace), expected_calls);

        let tokens = step(&mut worker, &mut trace);
        assert_eq!(
            events(&tokens),
            admitted
                .iter()
                .map(|&id| (id, 7, true, Some(FinishReason::MaxLength)))
                .collect::<Vec<_>>()
        );
        for (&id, &slot) in admitted.iter().zip(&slots) {
            expected_calls.push((id, 1, vec![7]));
            let own_calls: Vec<_> = trace.iter().filter(|call| call.id == id).collect();
            assert_eq!(own_calls.len(), 2);
            assert!(own_calls.iter().all(|call| call.slot == slot));
            completed.push(id);
        }
        assert_eq!(calls(&trace), expected_calls);
        assert_eq!(worker.scheduler.waiting_count(), population - end);
        assert_eq!(worker.gdn_pool.free_count(), 2);
        assert_eq!(worker.active_count(), population - end);
    }
    assert_eq!(completed, ids);
    assert_drained(&mut worker);
}

#[test]
fn capacity_sized_queue_admits_each_request_once() {
    run_fifo_population(2);
}

#[test]
fn over_capacity_queue_preserves_fifo_without_duplicate_prefill() {
    run_fifo_population(5);
}

#[test]
fn chunked_prefill_interleaves_with_decode_at_exact_positions() {
    let mut worker = worker(2, 2, None);
    let long = submit(&mut worker, &[1, 2, 3, 4, 5], 2);
    let short = submit(&mut worker, &[6], 4);
    let mut trace = Vec::new();

    assert_eq!(
        events(&step(&mut worker, &mut trace)),
        [(short, 7, false, None)]
    );
    assert_eq!(worker.scheduler.waiting_count(), 0);
    assert_eq!(worker.gdn_pool.free_count(), 0);
    assert_eq!(
        events(&step(&mut worker, &mut trace)),
        [(short, 7, false, None)]
    );
    assert_eq!(worker.gdn_pool.free_count(), 0);
    assert_eq!(
        events(&step(&mut worker, &mut trace)),
        [(long, 7, false, None), (short, 7, false, None)]
    );
    assert_eq!(worker.gdn_pool.free_count(), 0);
    assert_eq!(
        events(&step(&mut worker, &mut trace)),
        [
            (long, 7, true, Some(FinishReason::MaxLength)),
            (short, 7, true, Some(FinishReason::MaxLength)),
        ]
    );
    assert_eq!(
        calls(&trace),
        [
            (long, 0, vec![1, 2]),
            (short, 0, vec![6]),
            (long, 2, vec![3, 4]),
            (short, 1, vec![7]),
            (long, 4, vec![5]),
            (short, 2, vec![7]),
            (long, 5, vec![7]),
            (short, 3, vec![7]),
        ]
    );
    assert!(
        trace
            .iter()
            .all(|call| call.slot == usize::from(call.id == short))
    );
    assert_drained(&mut worker);
}

#[test]
fn low_page_reserve_defers_queued_work_until_pages_return() {
    let mut worker = worker(2, 2, None);
    let active = submit(&mut worker, &[1], 4);
    let mut trace = Vec::new();
    assert_eq!(
        events(&step(&mut worker, &mut trace)),
        [(active, 7, false, None)]
    );
    for _ in 0..3 {
        let page = worker.kv_pool.alloc().unwrap();
        worker
            .seq_manager
            .page_table_mut(active)
            .unwrap()
            .push_page(page);
    }
    assert_eq!(worker.kv_pool.free_count(), 1);
    let queued = submit(&mut worker, &[2], 1);
    assert_eq!(
        events(&step(&mut worker, &mut trace)),
        [(active, 7, false, None)]
    );
    assert_eq!(worker.scheduler.waiting_count(), 1);
    assert_eq!(worker.gdn_pool.free_count(), 1);
    assert_eq!(worker.gdn_pool.slot_of(queued), None);
    assert_eq!(worker.kv_pool.free_count(), 1);
    assert_eq!(
        worker.seq_manager.get(queued).unwrap().state,
        SequenceState::Prefilling { chunk_start: 0 }
    );

    let returned = worker
        .seq_manager
        .page_table_mut(active)
        .unwrap()
        .pop_page()
        .unwrap();
    worker.kv_pool.free(returned);
    assert_eq!(worker.kv_pool.free_count(), 2);
    assert_eq!(
        events(&step(&mut worker, &mut trace)),
        [
            (queued, 7, true, Some(FinishReason::MaxLength)),
            (active, 7, false, None),
        ]
    );
    assert_eq!(worker.scheduler.waiting_count(), 0);
    assert_eq!(worker.gdn_pool.free_count(), 1);
    assert_eq!(worker.kv_pool.free_count(), 2);
    assert_eq!(
        events(&step(&mut worker, &mut trace)),
        [(active, 7, true, Some(FinishReason::MaxLength))]
    );
    assert_eq!(
        calls(&trace),
        [
            (active, 0, vec![1]),
            (active, 1, vec![7]),
            (queued, 0, vec![2]),
            (active, 2, vec![7]),
            (active, 3, vec![7]),
        ]
    );
    assert_eq!(worker.kv_pool.free_count(), 4);
    assert_drained(&mut worker);
}

#[test]
fn eos_and_length_finishes_release_admission_slots() {
    let mut worker = worker(2, 2, Some(2));
    let eos = submit(&mut worker, &[1], 1);
    let length = submit(&mut worker, &[3], 2);
    let mut trace = Vec::new();
    let first = worker.step(|input, pool| {
        let winner = if input.seq_id == eos { 2 } else { 7 };
        record(&input, pool, &mut trace);
        logits(winner)
    });
    assert_eq!(
        events(&first),
        [
            (eos, 2, true, Some(FinishReason::Eos)),
            (length, 7, false, None)
        ]
    );
    assert_eq!(worker.scheduler.waiting_count(), 0);
    assert_eq!(worker.gdn_pool.free_count(), 1);
    assert_eq!(worker.gdn_pool.slot_of(eos), None);
    assert_eq!(
        events(&step(&mut worker, &mut trace)),
        [(length, 7, true, Some(FinishReason::MaxLength))]
    );
    assert_eq!(
        calls(&trace),
        [
            (eos, 0, vec![1]),
            (length, 0, vec![3]),
            (length, 1, vec![7])
        ]
    );
    assert_drained(&mut worker);
}

#[test]
fn zero_budget_never_enters_admission_or_forward() {
    let mut worker = worker(1, 2, None);
    let id = submit(&mut worker, &[1, 2], 0);
    assert_eq!(worker.scheduler.waiting_count(), 0);
    assert_eq!(worker.gdn_pool.free_count(), 1);
    assert_eq!(worker.gdn_pool.slot_of(id), None);
    assert!(!worker.samplers.contains_key(&id));
    assert_eq!(
        worker.seq_manager.get(id).unwrap().state,
        SequenceState::Finished(FinishReason::MaxLength)
    );
    assert!(
        worker
            .step(|_, _| panic!("zero budget must not run forward"))
            .is_empty()
    );
    assert_drained(&mut worker);
}

#[test]
fn missing_queued_sequence_does_not_allocate_a_slot() {
    let mut worker = worker(1, 2, None);
    let missing = submit(&mut worker, &[1], 1);
    worker.evict_sequence(missing, FinishReason::Preempted);
    assert_eq!(worker.scheduler.waiting_count(), 1);
    assert!(worker.seq_manager.get(missing).is_none());
    assert!(
        worker
            .step(|_, _| panic!("missing sequence must not run forward"))
            .is_empty()
    );
    assert_eq!(worker.gdn_pool.slot_of(missing), None);
    assert_drained(&mut worker);
    let next = submit(&mut worker, &[3], 1);
    let mut trace = Vec::new();
    assert_eq!(
        events(&step(&mut worker, &mut trace)),
        [(next, 7, true, Some(FinishReason::MaxLength))]
    );
    assert_eq!(calls(&trace), [(next, 0, vec![3])]);
    assert_drained(&mut worker);
}

#[test]
fn decoding_queued_sequence_is_not_prefilled() {
    let mut worker = worker(2, 2, None);
    let id = submit(&mut worker, &[1], 4);
    let sequence = worker.seq_manager.get_mut(id).unwrap();
    sequence.state = SequenceState::Decoding;
    sequence.generated_ids.push(7);
    assert_eq!(worker.scheduler.waiting_count(), 1);
    assert!(
        worker
            .step(|_, _| panic!("unadmitted decoding sequence must not run forward"))
            .is_empty()
    );
    assert_eq!(worker.scheduler.waiting_count(), 0);
    assert_eq!(worker.gdn_pool.free_count(), 2);
    assert_eq!(worker.gdn_pool.slot_of(id), None);
    let sequence = worker.seq_manager.get(id).unwrap();
    assert_eq!(sequence.state, SequenceState::Decoding);
    assert_eq!(sequence.generated_ids, [7]);
    worker.evict_sequence(id, FinishReason::Preempted);
    assert_drained(&mut worker);
}

#[test]
fn finished_queued_sequence_is_not_prefilled() {
    let mut worker = worker(1, 2, None);
    let id = submit(&mut worker, &[1], 1);
    worker.seq_manager.get_mut(id).unwrap().state =
        SequenceState::Finished(FinishReason::Preempted);
    assert_eq!(worker.scheduler.waiting_count(), 1);
    assert!(
        worker
            .step(|_, _| panic!("finished sequence must not run forward"))
            .is_empty()
    );
    assert_eq!(worker.gdn_pool.slot_of(id), None);
    assert_drained(&mut worker);
}

#[test]
fn empty_queued_prompt_does_not_allocate_a_slot() {
    let mut worker = worker(1, 2, None);
    let id = submit(&mut worker, &[1], 1);
    worker.seq_manager.get_mut(id).unwrap().prompt_ids.clear();
    assert_eq!(worker.scheduler.waiting_count(), 1);
    assert!(
        worker
            .step(|_, _| panic!("empty prefill extent must not run forward"))
            .is_empty()
    );
    assert_eq!(worker.scheduler.waiting_count(), 0);
    assert_eq!(worker.gdn_pool.free_count(), 1);
    assert_eq!(worker.gdn_pool.slot_of(id), None);
    assert_eq!(
        worker.seq_manager.get(id).unwrap().state,
        SequenceState::Prefilling { chunk_start: 0 }
    );
    worker.evict_sequence(id, FinishReason::Preempted);
    assert_drained(&mut worker);
}

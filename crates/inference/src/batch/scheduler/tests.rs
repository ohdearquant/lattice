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

#[test]
fn admission_capped_by_free_gdn_slots() {
    let mut sched = small_sched(32, 512, 0);
    for i in 1..=5 {
        sched.enqueue(SeqId(i));
    }
    // Batch size (32) and pages (100, plenty) would allow all 5, but only
    // 3 GDN state slots are free — admission must not exceed that.
    let dec = sched.select_batch(&[], &[], 100, 3);
    assert_eq!(dec.prefill.len(), 3);
    let ids: std::collections::HashSet<SeqId> = dec.prefill.iter().map(|&(id, _, _)| id).collect();
    assert_eq!(ids, [SeqId(1), SeqId(2), SeqId(3)].into_iter().collect());
    assert_eq!(sched.waiting_count(), 2);
}

#[test]
fn zero_free_gdn_slots_admits_none_but_running_still_decodes() {
    let mut sched = small_sched(32, 512, 0);
    sched.enqueue(SeqId(3));
    // No GDN slots free at all: the new admission must be refused, while
    // already-running sequences (which already hold their own slots) keep
    // decoding uninterrupted.
    let dec = sched.select_batch(&[], &[SeqId(1), SeqId(2)], 100, 0);
    assert!(dec.prefill.is_empty());
    assert_eq!(dec.decode.len(), 2);
    assert!(dec.decode.contains(&SeqId(1)));
    assert!(dec.decode.contains(&SeqId(2)));
    assert_eq!(sched.waiting_count(), 1);
}

#[test]
fn admission_unaffected_when_gdn_slots_plentiful() {
    // Same shape as `admission_respects_max_batch_size`, but pinning down
    // that a free-slot count at or above capacity changes nothing: the
    // max_batch_size bound alone still decides how many are admitted.
    let mut sched = small_sched(2, 512, 0);
    sched.enqueue(SeqId(1));
    sched.enqueue(SeqId(2));
    sched.enqueue(SeqId(3));
    let dec = sched.select_batch(&[], &[], 100, 2);
    assert_eq!(dec.prefill.len(), 2);
    assert_eq!(sched.waiting_count(), 1);
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

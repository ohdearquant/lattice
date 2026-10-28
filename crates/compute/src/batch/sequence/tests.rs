use super::*;

fn make_seq(id: u64, prompt_len: usize) -> Sequence {
    Sequence::new(
        SeqId(id),
        vec![0u32; prompt_len],
        SamplingConfig::default(),
        None,
        64,
    )
}

// --- SeqId ---

#[test]
fn seq_id_counter_increments() {
    let mut counter = 0u64;
    let a = SeqId::next(&mut counter);
    let b = SeqId::next(&mut counter);
    assert_eq!(a.0, 0);
    assert_eq!(b.0, 1);
}

#[test]
fn seq_id_display() {
    assert_eq!(SeqId(7).to_string(), "seq:7");
}

// --- Sequence lifecycle ---

#[test]
fn new_sequence_is_prefilling_at_zero() {
    let seq = make_seq(0, 10);
    assert_eq!(seq.state, SequenceState::Prefilling { chunk_start: 0 });
    assert!(seq.is_active());
}

#[test]
fn advance_prefill_partial_chunk() {
    let mut seq = make_seq(0, 20);
    seq.advance_prefill(10);
    assert_eq!(seq.state, SequenceState::Prefilling { chunk_start: 10 });
    assert!(seq.is_active());
}

#[test]
fn advance_prefill_complete_transitions_to_decoding() {
    let mut seq = make_seq(0, 10);
    seq.advance_prefill(10);
    assert_eq!(seq.state, SequenceState::Decoding);
    assert!(seq.is_active());
}

#[test]
fn advance_prefill_overshoot_transitions_to_decoding() {
    let mut seq = make_seq(0, 10);
    seq.advance_prefill(20); // chunk larger than prompt
    assert_eq!(seq.state, SequenceState::Decoding);
}

#[test]
fn push_token_normal() {
    let mut seq = make_seq(0, 1);
    seq.state = SequenceState::Decoding;
    let done = seq.push_token(42, None);
    assert!(!done);
    assert_eq!(seq.generated_ids, vec![42]);
    assert_eq!(seq.state, SequenceState::Decoding);
}

#[test]
fn push_token_eos_finishes_sequence() {
    let mut seq = make_seq(0, 1);
    seq.state = SequenceState::Decoding;
    let done = seq.push_token(2, Some(2)); // EOS = token 2
    assert!(done);
    assert_eq!(seq.state, SequenceState::Finished(FinishReason::Eos));
    assert!(!seq.is_active());
}

#[test]
fn push_token_max_length_finishes_sequence() {
    let mut seq = Sequence::new(
        SeqId(0),
        vec![0u32; 1],
        SamplingConfig::default(),
        None,
        2, // max_new_tokens = 2
    );
    seq.state = SequenceState::Decoding;
    assert!(!seq.push_token(10, None));
    let done = seq.push_token(11, None);
    assert!(done);
    assert_eq!(seq.state, SequenceState::Finished(FinishReason::MaxLength));
}

#[test]
fn preempt_marks_sequence_finished() {
    let mut seq = make_seq(0, 5);
    seq.state = SequenceState::Decoding;
    seq.preempt();
    assert_eq!(seq.state, SequenceState::Finished(FinishReason::Preempted));
    assert!(!seq.is_active());
}

#[test]
fn position_prefilling_returns_chunk_start() {
    let mut seq = make_seq(0, 20);
    seq.state = SequenceState::Prefilling { chunk_start: 7 };
    assert_eq!(seq.position(), 7);
}

#[test]
fn position_decoding_returns_prompt_plus_generated() {
    let mut seq = make_seq(0, 5); // prompt_len = 5
    seq.state = SequenceState::Decoding;
    seq.generated_ids = vec![1, 2, 3];
    assert_eq!(seq.position(), 8); // 5 + 3
}

// --- SequenceManager ---

#[test]
fn manager_add_and_get() {
    let mut mgr = SequenceManager::new();
    let seq = make_seq(0, 5);
    mgr.add(seq, 256);
    assert_eq!(mgr.len(), 1);
    assert!(mgr.get(SeqId(0)).is_some());
    assert!(mgr.page_table(SeqId(0)).is_some());
}

#[test]
fn manager_remove_returns_seq_and_table() {
    let mut mgr = SequenceManager::new();
    mgr.add(make_seq(0, 5), 256);
    let result = mgr.remove(SeqId(0));
    assert!(result.is_some());
    let (seq, table) = result.unwrap();
    assert_eq!(seq.id, SeqId(0));
    assert_eq!(table.seq_len(), 0);
    assert!(mgr.is_empty());
}

#[test]
fn manager_ids_in_state_prefilling() {
    let mut mgr = SequenceManager::new();
    mgr.add(make_seq(0, 10), 256); // Prefilling
    let mut seq1 = make_seq(1, 10);
    seq1.state = SequenceState::Decoding;
    mgr.add(seq1, 256);
    let prefilling = mgr.ids_in_state_prefilling();
    assert_eq!(prefilling, vec![SeqId(0)]);
}

#[test]
fn manager_ids_decoding() {
    let mut mgr = SequenceManager::new();
    let mut seq = make_seq(0, 5);
    seq.state = SequenceState::Decoding;
    mgr.add(seq, 256);
    mgr.add(make_seq(1, 10), 256); // Prefilling
    let decoding = mgr.ids_decoding();
    assert_eq!(decoding, vec![SeqId(0)]);
}

#[test]
fn manager_ids_finished() {
    let mut mgr = SequenceManager::new();
    let mut seq = make_seq(0, 5);
    seq.state = SequenceState::Finished(FinishReason::Eos);
    mgr.add(seq, 256);
    mgr.add(make_seq(1, 10), 256);
    let finished = mgr.ids_finished();
    assert_eq!(finished, vec![SeqId(0)]);
}

#[test]
fn manager_next_id_is_unique() {
    let mut mgr = SequenceManager::new();
    let a = mgr.next_id();
    let b = mgr.next_id();
    assert_ne!(a, b);
}

#[test]
fn manager_remove_nonexistent_returns_none() {
    let mut mgr = SequenceManager::new();
    assert!(mgr.remove(SeqId(99)).is_none());
}

#[test]
fn manager_get_mut_allows_state_update() {
    let mut mgr = SequenceManager::new();
    mgr.add(make_seq(0, 5), 256);
    let seq = mgr.get_mut(SeqId(0)).unwrap();
    seq.state = SequenceState::Decoding;
    assert_eq!(mgr.get(SeqId(0)).unwrap().state, SequenceState::Decoding);
}

//! Allocation counts for warmed, small native BERT batches on the calling thread.
#![cfg(all(
    target_arch = "x86_64",
    not(target_os = "macos"),
    not(feature = "x86-gemm")
))]

use lattice_model_bert::BertModel;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

#[path = "support/bert_checkpoint.rs"]
mod checkpoint_support;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AllocationCounts {
    alloc: u64,
    zeroed: u64,
    realloc: u64,
}

impl AllocationCounts {
    const ZERO: Self = Self {
        alloc: 0,
        zeroed: 0,
        realloc: 0,
    };

    fn total(self) -> u64 {
        assert!(self.alloc < u64::MAX && self.zeroed < u64::MAX && self.realloc < u64::MAX);
        self.alloc
            .checked_add(self.zeroed)
            .and_then(|value| value.checked_add(self.realloc))
            .expect("allocation counts must fit u64")
    }
}

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static COUNTS: Cell<AllocationCounts> = const { Cell::new(AllocationCounts::ZERO) };
}

fn record(update: impl FnOnce(&mut AllocationCounts)) {
    let _ = COUNTING.try_with(|active| {
        if active.get() {
            let _ = COUNTS.try_with(|slot| {
                let mut counts = slot.get();
                update(&mut counts);
                slot.set(counts);
            });
        }
    });
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: every operation forwards the original pointer/layout contract to System;
// constant-initialized thread-local counters do not allocate or unwind.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(|counts| counts.alloc = counts.alloc.saturating_add(1));
        // SAFETY: the caller supplied a valid allocation layout, forwarded unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(|counts| counts.zeroed = counts.zeroed.saturating_add(1));
        // SAFETY: System receives the caller's unchanged layout and supplies zeroed memory.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(|counts| counts.realloc = counts.realloc.saturating_add(1));
        // SAFETY: the live System pointer, original layout and new size are forwarded unchanged.
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller supplies a live System allocation and its original layout.
        unsafe { System.dealloc(pointer, layout) }
    }
}

struct CountingScope;

impl Drop for CountingScope {
    fn drop(&mut self) {
        let _ = COUNTING.try_with(|active| active.set(false));
    }
}

fn count_allocations<T>(run: impl FnOnce() -> T) -> (T, AllocationCounts) {
    COUNTING.with(|active| assert!(!active.get(), "allocation scopes must not nest"));
    COUNTS.with(|slot| slot.set(AllocationCounts::ZERO));
    COUNTING.with(|active| active.set(true));
    let scope = CountingScope;
    let result = run();
    drop(scope);
    (result, COUNTS.with(Cell::get))
}

fn assert_layer_independent_allocations(batch: usize) {
    let one_layer = checkpoint_support::tiny_bert_checkpoint(1, 2);
    let three_layers = checkpoint_support::tiny_bert_checkpoint(3, 2);
    let models = [
        BertModel::from_directory(one_layer.path()).unwrap(),
        BertModel::from_directory(three_layers.path()).unwrap(),
    ];
    let texts = &checkpoint_support::MIXED_INPUTS[..batch];
    for model in &models {
        assert_eq!(model.config().hidden_size, 4);
        assert_eq!(model.config().intermediate_size, 4);
        assert_eq!(model.config().num_attention_heads, 2);
        assert!(
            model
                .tokenizer()
                .tokenize_batch(texts)
                .iter()
                .all(|input| input.real_length > 0 && input.real_length <= 32)
        );
        for _ in 0..2 {
            drop(black_box(model.encode_batch(texts).unwrap()));
        }
    }

    let mut samples = [[AllocationCounts::ZERO; 2]; 3];
    for sample in &mut samples {
        for (slot, model) in sample.iter_mut().zip(&models) {
            let (result, counts) =
                count_allocations(|| black_box(model.encode_batch(black_box(texts))));
            let outputs = result.unwrap();
            assert_eq!(outputs.len(), batch);
            assert!(outputs.iter().all(|values| values.len() == 4
                && values.iter().all(|value| value.is_finite())));
            *slot = counts;
            drop(black_box(outputs));
        }
    }
    for (index, counts) in samples.iter().enumerate() {
        println!(
            "batch={batch} sample={index} layers1={:?} layers3={:?} delta={}",
            counts[0],
            counts[1],
            i128::from(counts[1].total()) - i128::from(counts[0].total())
        );
    }
    for counts in &samples {
        assert_eq!(
            *counts, samples[0],
            "batch={batch}: allocation counts must repeat"
        );
        assert!(
            counts[0].total() > 0 && counts[1].total() > 0,
            "batch={batch}: the counter must observe the encode allocations"
        );
        assert_eq!(
            counts[1].total(),
            counts[0].total(),
            "batch={batch}: attention scratch allocations must not grow with layer count"
        );
    }
}

#[test]
fn single_batch_has_layer_independent_allocations() {
    assert_layer_independent_allocations(1);
}

#[test]
fn packed_batch3_has_layer_independent_allocations() {
    assert_layer_independent_allocations(3);
}

#[test]
fn packed_batch17_has_layer_independent_allocations() {
    assert_layer_independent_allocations(17);
}

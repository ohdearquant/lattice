//! Allocation-size checks for warmed CPU embedding cache misses.
#![cfg(all(
    target_arch = "x86_64",
    not(target_os = "macos"),
    not(feature = "x86-gemm")
))]

use lattice_model_qwen3_embedding::{QwenEmbeddingBackend, QwenModel};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

#[path = "support/embedding_checkpoint.rs"]
mod support;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counts {
    alloc: u64,
    zeroed: u64,
    realloc: u64,
    target: u64,
}

impl Counts {
    const ZERO: Self = Self {
        alloc: 0,
        zeroed: 0,
        realloc: 0,
        target: 0,
    };

    fn total(self) -> u64 {
        assert!(
            [self.alloc, self.zeroed, self.realloc, self.target]
                .into_iter()
                .all(|value| value < u64::MAX)
        );
        self.alloc
            .checked_add(self.zeroed)
            .and_then(|value| value.checked_add(self.realloc))
            .expect("allocation counts must fit u64")
    }
}

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static TARGET_SIZE: Cell<usize> = const { Cell::new(0) };
    static COUNTS: Cell<Counts> = const { Cell::new(Counts::ZERO) };
}

fn record(size: usize, update: impl FnOnce(&mut Counts)) {
    let _ = ACTIVE.try_with(|active| {
        if active.get() {
            let _ = TARGET_SIZE.try_with(|target| {
                let _ = COUNTS.try_with(|slot| {
                    let mut counts = slot.get();
                    update(&mut counts);
                    if size == target.get() {
                        counts.target = counts.target.saturating_add(1);
                    }
                    slot.set(counts);
                });
            });
        }
    });
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: every operation delegates the original allocation contract to System.
// Constant-initialized TLS bookkeeping neither allocates nor unwinds.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's valid layout is forwarded unchanged to System.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size(), |counts| {
                counts.alloc = counts.alloc.saturating_add(1)
            });
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: System receives the unchanged layout and supplies zeroed memory.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record(layout.size(), |counts| {
                counts.zeroed = counts.zeroed.saturating_add(1)
            });
        }
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the live System pointer, original layout and requested size are unchanged.
        let result = unsafe { System.realloc(pointer, layout, new_size) };
        if !result.is_null() {
            record(new_size, |counts| {
                counts.realloc = counts.realloc.saturating_add(1)
            });
        }
        result
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller supplies a live System allocation and its original layout.
        unsafe { System.dealloc(pointer, layout) }
    }
}

struct CountingScope;

impl Drop for CountingScope {
    fn drop(&mut self) {
        let _ = ACTIVE.try_with(|active| active.set(false));
    }
}

fn measured<T>(target: usize, run: impl FnOnce() -> T) -> (T, Counts) {
    ACTIVE.with(|active| assert!(!active.get(), "measurement must not nest"));
    TARGET_SIZE.with(|slot| slot.set(target));
    COUNTS.with(|slot| slot.set(Counts::ZERO));
    ACTIVE.with(|active| active.set(true));
    let scope = CountingScope;
    let result = run();
    drop(scope);
    (result, COUNTS.with(Cell::get))
}

#[derive(Clone, Copy, Debug)]
enum Route {
    Serial,
    Identified,
    Packed,
}

fn run(model: &QwenModel, route: Route) -> Vec<Vec<f32>> {
    match route {
        Route::Serial => vec![model.encode(support::TEXTS[0]).unwrap()],
        Route::Identified => {
            let (output, backend) = model.encode_identified_uncached(support::TEXTS[0]).unwrap();
            assert_eq!(backend, QwenEmbeddingBackend::Cpu);
            vec![output]
        }
        Route::Packed => model.encode_batch(&support::TEXTS).unwrap(),
    }
}

fn measure(model: &QwenModel, route: Route, target: usize) -> (Vec<Vec<f32>>, Counts) {
    match route {
        Route::Serial => {
            let (result, counts) = measured(target, || {
                black_box(model.encode(black_box(support::TEXTS[0])))
            });
            (vec![result.unwrap()], counts)
        }
        Route::Identified => {
            let (result, counts) = measured(target, || {
                black_box(model.encode_identified_uncached(black_box(support::TEXTS[0])))
            });
            let (output, backend) = result.unwrap();
            assert_eq!(backend, QwenEmbeddingBackend::Cpu);
            (vec![output], counts)
        }
        Route::Packed => {
            let (result, counts) = measured(target, || {
                black_box(model.encode_batch(black_box(&support::TEXTS)))
            });
            (result.unwrap(), counts)
        }
    }
}

fn assert_no_full_hidden_copy(route: Route) {
    let checkpoint = support::checkpoint();
    let model = QwenModel::from_directory(checkpoint.path()).unwrap();
    support::assert_fixture(&model);
    let (tokens, rows, cache_entries) = match route {
        Route::Serial => (support::LENGTHS[0], 1, 1),
        Route::Identified => (support::LENGTHS[0], 1, 0),
        Route::Packed => (support::LENGTHS.iter().sum(), 3, 3),
    };
    let target = tokens * support::HIDDEN * size_of::<f32>();
    assert_eq!(target, if rows == 3 { 4224 } else { 2176 });
    for _ in 0..2 {
        model.cache_clear();
        drop(black_box(run(&model, route)));
    }

    let mut samples = [Counts::ZERO; 3];
    for slot in &mut samples {
        model.cache_clear();
        assert_eq!(model.cache_size(), 0);
        let (outputs, counts) = measure(&model, route, target);
        assert_eq!(model.cache_size(), cache_entries);
        assert_eq!(outputs.len(), rows);
        for output in &outputs {
            assert_eq!(output.len(), support::HIDDEN);
            assert!(output.iter().all(|value| value.is_finite()));
            assert!(output.iter().any(|value| *value != 0.0));
        }
        *slot = counts;
        drop(black_box(outputs));
    }
    for (sample, counts) in samples.iter().enumerate() {
        println!(
            "route={route:?} sample={sample} tokens={tokens} hidden={} target_bytes={target} counts={counts:?}",
            support::HIDDEN
        );
    }
    for counts in samples {
        assert_eq!(
            counts.target, samples[0].target,
            "{route:?}: target allocation counts must repeat"
        );
    }
    for counts in samples {
        assert!(
            counts.total() > 0,
            "{route:?}: counter must observe allocations"
        );
        assert_eq!(
            counts.target, 0,
            "{route:?}: encode must not allocate the {target}-byte full hidden matrix"
        );
    }
}

#[test]
fn serial_encode_avoids_full_hidden_allocation() {
    assert_no_full_hidden_copy(Route::Serial);
}

#[test]
fn identified_cpu_encode_avoids_full_hidden_allocation() {
    assert_no_full_hidden_copy(Route::Identified);
}

#[test]
fn packed_cpu_encode_avoids_full_hidden_allocation() {
    assert_no_full_hidden_copy(Route::Packed);
}

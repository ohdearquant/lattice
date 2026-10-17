use criterion::black_box;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Counting allocator — wraps the system allocator to track per-call statistics.
// Shared only by the benchmark and its isolated control target, never the library.
// ---------------------------------------------------------------------------

struct CountingAlloc;

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static DEALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static REALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES_ALLOCATED: AtomicU64 = AtomicU64::new(0);
static BYTES_DEALLOCATED: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

// SAFETY: Every operation delegates the original allocation contract to System.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES_ALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: GlobalAlloc requires the caller to supply a valid nonzero layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES_DEALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: The caller supplies our System allocation and its original layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        REALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES_ALLOCATED.fetch_add(new_size as u64, Ordering::Relaxed);
        BYTES_DEALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: The caller supplies a live System allocation, its layout and valid new size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[allow(dead_code)]
pub(crate) struct AllocationSnapshot {
    pub(crate) alloc_calls: u64,
    pub(crate) realloc_calls: u64,
    pub(crate) bytes_allocated: u64,
}

#[allow(dead_code)]
impl AllocationSnapshot {
    pub(crate) fn capture() -> Self {
        Self {
            alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
            realloc_calls: REALLOC_CALLS.load(Ordering::Relaxed),
            bytes_allocated: BYTES_ALLOCATED.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn delta_since(start: &AllocationSnapshot) -> AllocationDelta {
        AllocationDelta {
            alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed) - start.alloc_calls,
            realloc_calls: REALLOC_CALLS.load(Ordering::Relaxed) - start.realloc_calls,
            bytes_allocated: BYTES_ALLOCATED.load(Ordering::Relaxed) - start.bytes_allocated,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)]
pub(crate) struct AllocationDelta {
    pub(crate) alloc_calls: u64,
    pub(crate) realloc_calls: u64,
    pub(crate) bytes_allocated: u64,
}

// The executor a counted sample runs on: one rayon thread, never the ambient pool.
//
// The counting allocator is process-wide, so a sampled call counts the allocations of every
// rayon worker that runs any part of it. Which worker that is, and therefore which lazily
// sized per-worker structure grows during which sample, is a scheduling decision: on an
// 8-thread pool the retained samples of one call can disagree by an allocation or a few
// hundred bytes even after a warm-up. `ThreadPool::install` puts the sampled call on a pool
// with one thread, and rayon resolves the pool from the current worker, so the parallel
// iterators the sampled code calls (the split `matmul_bt` for one) also run there, one thread
// at a time, instead of spreading over the ambient pool. Worker identity, split count, job
// order and touched state are then the same in every sample. Only counted samples are
// affected; the caller's timed work never enters this pool.
//
// The pool is built once and lived in for the process, so per-pool and per-worker state that
// grows on first use is paid for by the first sampled call, and its own warm-up, not by a
// retained sample. It is built before the first snapshot of that call, because building it
// allocates.
static COUNTED_SAMPLE_EXECUTOR: OnceLock<rayon::ThreadPool> = OnceLock::new();

#[allow(dead_code)]
pub(crate) fn counted_sample_executor() -> &'static rayon::ThreadPool {
    COUNTED_SAMPLE_EXECUTOR.get_or_init(|| {
        let pool = match rayon::ThreadPoolBuilder::new().num_threads(1).build() {
            Ok(pool) => pool,
            Err(error) => panic!(
                "could not build the one-thread executor for counted samples, and the ambient \
                 pool cannot be used instead because its per-worker allocations differ between \
                 samples: {error}"
            ),
        };
        // `build` returns once the worker thread has been spawned, not once it has finished
        // starting, and that start-up allocates. Entering the pool here runs it to completion
        // before the first sampled call, so no retained sample can be the one that pays for
        // per-worker state set up on first entry. Nothing is printed here: the caller's own
        // measurement must not carry an unexpected stderr line.
        pool.install(|| {});
        pool
    })
}

// Runs one discarded warm-up call and the three retained samples on the one-thread executor.
//
// The executor is entered with `ThreadPool::install`, which requires `Send`, rather than
// adopted with `ThreadPoolBuilder::use_current_thread`, which requires nothing: adopting the
// calling thread binds that thread to this one-thread pool for the rest of the process
// (rayon leaks the adopted worker, and no public API clears it), and the calling thread is
// the thread the caller's own timed work runs on, so every later rayon call there would
// silently move onto the one-thread pool. Entering the executor leaves the calling thread on
// its own pool. Pool construction happens in `counted_sample_executor`, before any snapshot.
pub(crate) fn sample_allocations(
    run_fn: &mut (impl FnMut() -> AllocationDelta + Send),
) -> [AllocationDelta; 3] {
    counted_sample_executor().install(|| {
        assert_eq!(
            rayon::current_num_threads(),
            1,
            "counted samples must run on the one-thread executor, not on the ambient pool"
        );
        assert_eq!(
            rayon::current_thread_index(),
            Some(0),
            "counted samples must run on the executor's own worker thread"
        );
        let _ = run_fn();
        [run_fn(), run_fn(), run_fn()]
    })
}

// Builds the stderr heading for one allocation report. The suite is the caller's
// Criterion group, so a report can never be filed under another bench's name; an
// empty `label` collapses to `<suite>/<phase>` for benches whose only axis is the
// phase.
pub(crate) fn allocation_report_heading(suite: &str, label: &str, phase: &str) -> String {
    if label.is_empty() {
        format!("{suite}/{phase}")
    } else {
        format!("{suite}/{label}/{phase}")
    }
}

// Asserts that a zeroing allocation is visible to `CountingAlloc`. `GlobalAlloc`'s
// `alloc_zeroed` is not overridden above, so zeroed allocations are only counted
// while its default implementation routes through `alloc`. Overriding it later to
// recover the platform calloc path would stop counting a whole category, and every
// total would drop -- which reads as an improvement. This runs on the bench's own
// path rather than as a `#[test]`, because this target is declared `harness = false`
// and libtest never runs here.
pub(crate) fn assert_zeroed_allocation_is_counted() {
    let before = AllocationSnapshot::capture();
    let zeroed: Vec<u8> = vec![0; 4096];
    black_box(&zeroed);
    let delta = AllocationSnapshot::delta_since(&before);
    assert!(
        delta.alloc_calls > 0,
        "zeroed allocation was not observed by CountingAlloc: a 4 KiB vec![0; _] moved \
         alloc_calls by {} and bytes_allocated by {}; allocation totals in this bench \
         are undercounting every calloc-shaped allocation",
        delta.alloc_calls,
        delta.bytes_allocated,
    );
}

// Print allocation counts without asserting zero — use for "before" snapshots.
#[allow(dead_code)]
pub(crate) fn allocation_count_print(
    suite: &str,
    label: &str,
    phase: &str,
    tokens: usize,
    run_fn: &mut (impl FnMut() -> AllocationDelta + Send),
) {
    assert_zeroed_allocation_is_counted();
    let heading = allocation_report_heading(suite, label, phase);
    let samples = sample_allocations(run_fn);
    let s0 = samples[0];
    for (i, &s) in samples.iter().enumerate().skip(1) {
        if s != s0 {
            panic!(
                "allocation count is non-deterministic at run {i} for {heading}: \
                 run0=({},{},{}) run{i}=({},{},{})",
                s0.alloc_calls,
                s0.realloc_calls,
                s0.bytes_allocated,
                s.alloc_calls,
                s.realloc_calls,
                s.bytes_allocated,
            );
        }
    }
    let n = tokens as f64;
    eprintln!(
        "\n{heading}\n\
         tokens={tokens}\n\
         alloc_calls_total={}\n\
         realloc_calls_total={}\n\
         bytes_allocated_total={}\n\
         allocations_per_token={:.2}\n\
         reallocations_per_token={:.2}\n\
         bytes_allocated_per_token={:.0}",
        s0.alloc_calls,
        s0.realloc_calls,
        s0.bytes_allocated,
        s0.alloc_calls as f64 / n,
        s0.realloc_calls as f64 / n,
        s0.bytes_allocated as f64 / n,
    );
}

#[allow(dead_code)]
pub(crate) fn allocation_count_report(
    suite: &str,
    label: &str,
    phase: &str,
    tokens: usize,
    run_fn: &mut (impl FnMut() -> AllocationDelta + Send),
) {
    assert_zeroed_allocation_is_counted();
    let heading = allocation_report_heading(suite, label, phase);
    let samples = sample_allocations(run_fn);

    let s0 = samples[0];
    for (i, &s) in samples.iter().enumerate().skip(1) {
        if s != s0 {
            panic!(
                "allocation count is non-deterministic at run {i} for {heading}: \
                 run0=({},{},{}) run{i}=({},{},{})",
                s0.alloc_calls,
                s0.realloc_calls,
                s0.bytes_allocated,
                s.alloc_calls,
                s.realloc_calls,
                s.bytes_allocated,
            );
        }
    }

    if s0.alloc_calls != 0 || s0.realloc_calls != 0 || s0.bytes_allocated != 0 {
        panic!(
            "allocation gate failed for {heading}: \
             alloc_calls_total={} realloc_calls_total={} bytes_allocated_total={}",
            s0.alloc_calls, s0.realloc_calls, s0.bytes_allocated,
        );
    }

    let n = tokens as f64;
    eprintln!(
        "\n{heading}\n\
         tokens={tokens}\n\
         alloc_calls_total={}\n\
         realloc_calls_total={}\n\
         bytes_allocated_total={}\n\
         allocations_per_token={:.2}\n\
         reallocations_per_token={:.2}\n\
         bytes_allocated_per_token={:.0}",
        s0.alloc_calls,
        s0.realloc_calls,
        s0.bytes_allocated,
        s0.alloc_calls as f64 / n,
        s0.realloc_calls as f64 / n,
        s0.bytes_allocated as f64 / n,
    );
}

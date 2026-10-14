//! ADR-099 F1 allocation exception: with `x86-gemm`, a multi-row `matmul_bt` allocates at most
//! once per call, and a single-row call (the existing kernel) not at all. One test function, so
//! no other test thread allocates while the counter runs.
#![cfg(all(feature = "x86-gemm", target_arch = "x86_64", not(target_os = "macos")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use lattice_inference::forward::cpu::matmul_bt;

struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded unchanged to the system allocator; the counter is a side
// effect only.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarded with the caller's layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr came from System.alloc with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations_per_call(m: usize, k: usize, n: usize, calls: usize) -> (usize, Vec<f32>) {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let b: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32 - 3.0) * 0.02).collect();
    let mut c = vec![0.0f32; m * n];
    matmul_bt(&a, &b, &mut c, m, k, n);
    ALLOCATIONS.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    for _ in 0..calls {
        matmul_bt(&a, &b, &mut c, m, k, n);
    }
    COUNTING.store(false, Ordering::SeqCst);
    (ALLOCATIONS.load(Ordering::SeqCst), c)
}

fn scalar(m: usize, k: usize, n: usize) -> Vec<f32> {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let b: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32 - 3.0) * 0.02).collect();
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for p in 0..k {
                s += a[i * k + p] as f64 * b[j * k + p] as f64;
            }
            c[i * n + j] = s as f32;
        }
    }
    c
}

#[test]
fn gemm_allocates_at_most_once_per_call() {
    let calls = 8;
    // (m, k, n): BERT-small shapes at batch 1 and 8, a single-row control, and a large product
    // that takes the parallel path on a multi-thread pool.
    for &(m, k, n) in &[
        (22, 384, 384),
        (22, 384, 1536),
        (22, 1536, 384),
        (176, 384, 1152),
        (512, 384, 1536),
        (1, 384, 384),
    ] {
        let (count, c) = allocations_per_call(m, k, n, calls);
        let reference = scalar(m, k, n);
        let max_err = c
            .iter()
            .zip(&reference)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        println!(
            "shape m={m} k={k} n={n} threads={} allocations={count} calls={calls} max_err={max_err:e}",
            rayon::current_num_threads()
        );
        assert!(max_err < 1e-4, "m={m} k={k} n={n}: max error {max_err}");
        if m == 1 {
            assert_eq!(count, 0, "single-row call allocated {count} times");
        } else {
            assert!(
                count <= calls,
                "m={m} k={k} n={n}: {count} allocations over {calls} calls"
            );
        }
    }
}

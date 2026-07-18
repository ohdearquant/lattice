//! Structural regression: `LoraAdapter::apply` must not heap-allocate on the
//! per-token-row hot path (`lattice_inference::lora_hook::apply_lora_rows`
//! calls it once per row — 6 x layers x tokens times for a BERT forward
//! pass), whether or not the layer/module has an adapter.
//!
//! Installs a counting global allocator for this test binary only (isolated
//! from other integration tests, which run as separate processes) and
//! asserts zero allocation-call deltas across both the missing-adapter fast
//! path and the matched, rank-driven path.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use lattice_tune::lora::{LoraAdapter, LoraConfig, LoraLayer};

struct CountingAlloc;

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static DEALLOC_CALLS: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn alloc_calls() -> u64 {
    ALLOC_CALLS.load(Ordering::Relaxed)
}

fn dealloc_calls() -> u64 {
    DEALLOC_CALLS.load(Ordering::Relaxed)
}

fn make_adapter() -> LoraAdapter {
    let config = LoraConfig {
        rank: 8,
        alpha: 16.0,
        target_modules: vec!["query".into(), "value".into()],
    };

    let mut layers = HashMap::new();
    for layer_idx in 0..12 {
        for module in ["query", "value"] {
            layers.insert(
                (layer_idx, module.to_string()),
                LoraLayer {
                    a: vec![0.01; 8 * 16],
                    b: vec![0.02; 16 * 8],
                    d_in: 16,
                    d_out: 16,
                    rank: 8,
                },
            );
        }
    }

    LoraAdapter::new(config, layers).expect("valid adapter config")
}

#[test]
fn apply_matched_and_missing_hot_path_allocates_nothing() {
    let adapter = make_adapter();
    let x = vec![0.1f32; 16];

    // Warm up: touch every code path once, and allocate the row output
    // buffer itself, before the measured window starts — the buffer is
    // reused (never resized) across every row below, and any one-time
    // lazy-static setup in the allocator machinery happens here too, not
    // inside the measured window.
    let mut output = vec![0.0f32; 16];
    adapter.apply(0, "query", &x, &mut output);
    adapter.apply(0, "key", &x, &mut output); // no adapter for "key"

    let alloc_before = alloc_calls();
    let dealloc_before = dealloc_calls();

    // Simulate `apply_lora_rows` driving 512 token rows through 6 hooked
    // projections across 12 layers, mixing matched ("query"/"value") and
    // missing ("key"/"attn_output") modules the way a real BERT forward
    // pass would.
    let modules = ["query", "key", "value", "attn_output"];
    for _token in 0..512 {
        for layer_idx in 0..12 {
            for module in modules {
                adapter.apply(layer_idx, module, &x, &mut output);
            }
        }
    }

    let alloc_after = alloc_calls();
    let dealloc_after = dealloc_calls();

    assert_eq!(
        alloc_after - alloc_before,
        0,
        "LoraAdapter::apply performed {} heap allocations across 512 * 12 * 4 \
         hooked rows; the per-row hot path must be allocation-free",
        alloc_after - alloc_before
    );
    assert_eq!(
        dealloc_after - dealloc_before,
        0,
        "LoraAdapter::apply performed {} heap deallocations across 512 * 12 * 4 \
         hooked rows; the per-row hot path must be allocation-free",
        dealloc_after - dealloc_before
    );
}

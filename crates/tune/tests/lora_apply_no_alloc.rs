//! Structural regression: `LoraAdapter::apply` must not heap-allocate on the
//! per-token-row hot path (`lattice_inference::lora_hook::apply_lora_rows`
//! calls it once per row — 6 x layers x tokens times for a BERT forward
//! pass), whether or not the layer/module has an adapter — for ranks within
//! `apply::STACK_RANK_CAPACITY` (128), the only case this claim covers.
//! `apply_lora`'s own doc comment states the contract as rank-conditional:
//! above that cap it falls back to a heap `Vec` for the `A @ x` intermediate
//! and stays correct, just not allocation-free. `LoraAdapter::validate_against_bert`
//! deliberately permits `rank > min(d_in, d_out)` (redundant but valid; see
//! `blend_lora_adapters`, which produces exactly this shape), so a rank above
//! 128 is reachable through the public API, not just a hypothetical.
//!
//! Installs a counting global allocator for this test binary only (isolated
//! from other integration tests, which run as separate processes) and
//! asserts zero allocation-call deltas across both the missing-adapter fast
//! path and the matched, rank-driven path — plus a companion test below that
//! records the heap branch honestly instead of leaving it unmeasured.
//!
//! Counters are thread-local (#1272): the test body runs on a harness-spawned
//! thread while other harness threads (result reporting, output capture,
//! timing) remain live and may allocate during the measured window. A
//! process-wide counter attributes that unrelated activity to
//! `LoraAdapter::apply`, producing an intermittent nonzero delta with no
//! code change — the same commit both failed and passed in CI. Scoping the
//! counters to the measuring thread removes that cross-thread noise instead
//! of tolerating it with a fudge-factor threshold.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashMap;

use lattice_tune::lora::{LoraAdapter, LoraConfig, LoraLayer};

struct CountingAlloc;

thread_local! {
    static ALLOC_CALLS: Cell<u64> = const { Cell::new(0) };
    static DEALLOC_CALLS: Cell<u64> = const { Cell::new(0) };
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.with(|c| c.set(c.get() + 1));
        // SAFETY: `System` imposes the same contract on `alloc` that this
        // impl's caller has already satisfied, and `layout` reaches it
        // unchanged. Counting is side-effect free with respect to that
        // contract, so this delegation is sound exactly when the call into
        // this allocator was.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_CALLS.with(|c| c.set(c.get() + 1));
        // SAFETY: every pointer this allocator hands out comes from `System`,
        // so a `ptr`/`layout` pair the caller validly passes here is one
        // `System` allocated under that same layout. Both are forwarded
        // unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_CALLS.with(|c| c.set(c.get() + 1));
        // SAFETY: same forwarding invariant as `dealloc` for `ptr`/`layout`,
        // and `new_size` is passed through untouched for `System` to validate
        // against its own contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn alloc_calls() -> u64 {
    ALLOC_CALLS.with(Cell::get)
}

fn dealloc_calls() -> u64 {
    DEALLOC_CALLS.with(Cell::get)
}

fn make_adapter() -> LoraAdapter {
    let config = LoraConfig {
        rank: 8,
        alpha: 16.0,
        target_modules: vec!["query".into(), "value".into()],
        dtype: "f32".into(),
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

/// Companion to the test above: a rank above the stack-scratch cap (128, per
/// `apply::STACK_RANK_CAPACITY`) takes the documented heap-fallback branch.
/// This records that honestly (asserting the allocation actually happens)
/// rather than silently leaving it unmeasured — a renamed or generalized
/// "allocation-free" claim without this would still prove nothing about the
/// over-cap path. `rank = 200` matches the value `apply.rs`'s own internal
/// unit test uses for the same fallback.
#[test]
fn apply_rank_above_stack_capacity_falls_back_to_heap_allocation() {
    let rank = 200;
    let config = LoraConfig {
        rank,
        alpha: rank as f32,
        target_modules: vec!["query".into()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0, "query".to_string()),
        LoraLayer {
            a: vec![0.01; rank * 16],
            b: vec![0.02; 16 * rank],
            d_in: 16,
            d_out: 16,
            rank,
        },
    );
    let adapter = LoraAdapter::new(config, layers).expect("valid adapter config");
    let x = vec![0.1f32; 16];
    let mut output = vec![0.0f32; 16];

    // Warm up once, matching the pattern above, so only the measured calls
    // below are attributed.
    adapter.apply(0, "query", &x, &mut output);

    let alloc_before = alloc_calls();
    adapter.apply(0, "query", &x, &mut output);
    let alloc_after = alloc_calls();

    assert!(
        alloc_after > alloc_before,
        "rank {rank} (above STACK_RANK_CAPACITY=128) is documented to take the \
         heap-fallback branch for its A @ x scratch buffer, but no allocation \
         was observed — either the fallback stopped firing or this test no \
         longer exercises it"
    );
}

/// Pins the one thing this file's counter silently depends on: that a ZEROED
/// allocation reaches `CountingAlloc::alloc` and is counted.
///
/// `CountingAlloc` overrides `alloc`, `dealloc` and `realloc`, and does not
/// override `alloc_zeroed`. It is counted today only because `GlobalAlloc`'s
/// default `alloc_zeroed` calls `self.alloc` and then zeroes the block. That
/// is a property of the standard library's default method, not of anything
/// written here, and the whole dependency is invisible at this file's call
/// sites — nothing in the assertions above mentions zeroing.
///
/// The regression it guards is quiet in the dangerous direction. Adding an
/// `alloc_zeroed` override that forwards to `System.alloc_zeroed` without
/// touching the counter would not fail to compile and would not fail any
/// no-alloc assertion in this file, because those assert that a count did
/// NOT rise. An allocation that stops being counted makes them pass harder.
///
/// This matters beyond this file: a zeroed vector is the ordinary shape of a
/// large scratch buffer (`vec![0.0f32; n]` compiles to the zeroed path for
/// any element whose zero is all-zero bits), so any later measurement reusing
/// this allocator to prove a large buffer was NOT materialized is resting on
/// the same default.
#[test]
fn zeroed_allocations_are_counted() {
    // Control first, and it runs in the same test rather than being described
    // in a comment: if an ordinary allocation is not counted, the counter is
    // dead and the zeroed reading below would be meaningless rather than
    // reassuring.
    let before = alloc_calls();
    let ordinary: Vec<u8> = Vec::with_capacity(4096);
    let ordinary_delta = alloc_calls() - before;
    std::hint::black_box(&ordinary);
    assert!(
        ordinary_delta > 0,
        "control failed: an ordinary Vec::with_capacity was not counted, so \
         this test can say nothing about zeroed allocations"
    );

    // `vec![0u8; n]` and `vec![0.0f32; n]` both lower to the zeroed path.
    let before = alloc_calls();
    let zeroed_bytes: Vec<u8> = vec![0u8; 4096];
    let zeroed_bytes_delta = alloc_calls() - before;
    std::hint::black_box(&zeroed_bytes);

    let before = alloc_calls();
    let zeroed_floats: Vec<f32> = vec![0.0f32; 4096];
    let zeroed_floats_delta = alloc_calls() - before;
    std::hint::black_box(&zeroed_floats);

    assert!(
        zeroed_bytes_delta > 0,
        "vec![0u8; 4096] allocated without being counted: CountingAlloc no \
         longer sees the zeroed path"
    );
    assert!(
        zeroed_floats_delta > 0,
        "vec![0.0f32; 4096] allocated without being counted: CountingAlloc no \
         longer sees the zeroed path"
    );

    // Calling `alloc_zeroed` directly removes any doubt about what `vec!`
    // lowers to on this toolchain, which is the actual invariant at risk.
    let layout = Layout::from_size_align(4096, 8).expect("valid layout");
    let before = alloc_calls();
    // SAFETY: `layout` has a non-zero size and a valid power-of-two align, so
    // it satisfies `alloc_zeroed`'s contract. The returned pointer is freed
    // below through the same allocator under the identical layout.
    let ptr = unsafe { GLOBAL.alloc_zeroed(layout) };
    let direct_delta = alloc_calls() - before;
    assert!(
        !ptr.is_null(),
        "alloc_zeroed returned null for a 4096-byte layout"
    );
    // SAFETY: `ptr` came from `GLOBAL.alloc_zeroed` under this same `layout`
    // on the line above and has not been freed or reallocated since.
    unsafe { GLOBAL.dealloc(ptr, layout) };

    assert!(
        direct_delta > 0,
        "GlobalAlloc::alloc_zeroed did not reach CountingAlloc::alloc, so \
         zeroed allocations are invisible to every counter in this file"
    );
}

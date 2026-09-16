//! Allocation-counting instrument for the Metal Qwen3.5 forward pass.
//!
//! This target never times anything. It exists to answer a different
//! question than the timing benchmarks in this crate: how many Rust-side
//! heap allocations happen while the GPU forward pass and its logits
//! readback actually run, as opposed to while the surrounding call merely
//! submits work. `harness = false` and the absence of any Criterion
//! dependency below are both deliberate -- a counting target must never be
//! readable as a timing arm, so it is kept out of Criterion's
//! `--save-baseline` / `--baseline` pairing entirely.
//!
//! # Env
//! - `LATTICE_MODEL_DIR` -- path to a Q4-quantized model directory (default
//!   `~/.lattice/models/qwen3.5-0.8b-q4`). Must contain `config.json`.
//! - `LATTICE_TOKENIZER_DIR` -- path to a directory with `tokenizer.json`
//!   (default `~/.lattice/models/qwen3.5-0.8b`).
//!
//! # Run
//! ```text
//! cargo run --release -p lattice-inference --features metal-gpu,f16 --example bench_metal_forward_allocation
//! ```
//!
//! # CI note
//! Gated on `#[cfg(all(target_os = "macos", feature = "metal-gpu"))]` and
//! checkpoint-directory existence, so no GPU calls occur in CI.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

// ---------------------------------------------------------------------------
// Phase-gated counting allocator. Global allocators are per-binary and can
// only be registered by the final artifact, so this bench defines its own
// rather than sharing the library-internal tracking module in
// `src/quant/quarot/forward_equivalence.rs` -- the phase-marker shape below
// follows that module's `pre_admission_allocation_tracking` pattern.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Inactive,
    Active,
}

#[derive(Clone, Copy)]
struct Counts {
    alloc_calls: u64,
    realloc_calls: u64,
    bytes_allocated: u64,
}

std::thread_local! {
    static PHASE: Cell<Phase> = const { Cell::new(Phase::Inactive) };
    static COUNTS: Cell<Counts> = const {
        Cell::new(Counts {
            alloc_calls: 0,
            realloc_calls: 0,
            bytes_allocated: 0,
        })
    };
}

struct PhaseCountingAlloc;

#[global_allocator]
static GLOBAL: PhaseCountingAlloc = PhaseCountingAlloc;

fn record(bytes: u64, is_realloc: bool) {
    let _ = PHASE.try_with(|phase| {
        if phase.get() != Phase::Active {
            return;
        }
        let _ = COUNTS.try_with(|cell| {
            let mut counts = cell.get();
            if is_realloc {
                counts.realloc_calls = counts.realloc_calls.saturating_add(1);
            } else {
                counts.alloc_calls = counts.alloc_calls.saturating_add(1);
            }
            counts.bytes_allocated = counts.bytes_allocated.saturating_add(bytes);
            cell.set(counts);
        });
    });
}

// SAFETY: allocation requests are observed without modification and then
// forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for PhaseCountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size() as u64, false);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size() as u64, false);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size as u64, true);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        PHASE.with(|p| p.set(Phase::Inactive));
    }
}

// Brackets the phase inside the actual GPU worker call this bench measures,
// never around the code that merely arranges to call it -- the row this
// target exists for asks specifically about allocations during real forward
// execution, not enqueue-time bookkeeping.
fn start() -> Guard {
    PHASE.with(|p| {
        assert_eq!(
            p.get(),
            Phase::Inactive,
            "allocation counting already active"
        );
        p.set(Phase::Active);
    });
    COUNTS.with(|c| {
        c.set(Counts {
            alloc_calls: 0,
            realloc_calls: 0,
            bytes_allocated: 0,
        });
    });
    Guard
}

fn snapshot() -> Counts {
    COUNTS.with(Cell::get)
}

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("SKIP bench_metal_forward_allocation: requires macOS + metal-gpu feature");
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    run();
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use std::path::PathBuf;

    fn model_dir() -> Option<PathBuf> {
        if let Ok(v) = std::env::var("LATTICE_MODEL_DIR") {
            Some(PathBuf::from(v))
        } else {
            let home = std::env::var("HOME").ok()?;
            Some(PathBuf::from(format!(
                "{home}/.lattice/models/qwen3.5-0.8b-q4"
            )))
        }
    }

    fn tokenizer_dir() -> Option<PathBuf> {
        let dir = if let Ok(v) = std::env::var("LATTICE_TOKENIZER_DIR") {
            PathBuf::from(v)
        } else {
            let home = std::env::var("HOME").ok()?;
            PathBuf::from(format!("{home}/.lattice/models/qwen3.5-0.8b"))
        };
        if dir.join("tokenizer.json").exists() {
            Some(dir)
        } else {
            None
        }
    }

    let Some(dir) = model_dir() else {
        eprintln!(
            "SKIP bench_metal_forward_allocation: model checkpoint not found \
             (set LATTICE_MODEL_DIR)"
        );
        return;
    };
    let Some(tok_dir) = tokenizer_dir() else {
        eprintln!("SKIP bench_metal_forward_allocation: tokenizer.json not found");
        return;
    };

    // `from_model_dir` fails closed on a missing or unreadable config.json, so it
    // doubles as the checkpoint-presence probe. A separate `exists()` check is
    // both redundant and the lexical shape examples_no_preset_fallback.rs guards.
    let cfg = match Qwen35Config::from_model_dir(&dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP bench_metal_forward_allocation: model config unavailable: {e}");
            return;
        }
    };

    let tok_path = tok_dir.join("tokenizer.json");

    // Machine-wide GPU serialization (crates/inference/src/measurement.rs):
    // bound in this same function, before construction, per the shared
    // construction-site contract every Metal measurement harness follows.
    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();

    let mut state = match MetalQwen35State::from_q4_dir(&dir, &tok_path, &cfg, 4096) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP bench_metal_forward_allocation: Metal Q4 init failed: {e}");
            return;
        }
    };
    state.reset_state();

    // Warm-up, outside the counted phase: first-call allocations (lazy
    // buffer growth, one-time Metal pipeline construction) are steady-state
    // noise this instrument does not care about.
    let _ = state.forward_step(0, 0);

    let guard = start();
    let logits = state.forward_step(0, 1);
    drop(guard);
    drop(logits);

    let counts = snapshot();
    println!(
        "bench_metal_forward_allocation: alloc_calls={} realloc_calls={} bytes_allocated={}",
        counts.alloc_calls, counts.realloc_calls, counts.bytes_allocated
    );
}

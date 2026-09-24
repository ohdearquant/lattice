//! ADR-090 row D: the allocation half of the pair evidence.
//!
//! **What this is.** One arm of a base-vs-head comparison. A `#[global_allocator]`
//! is per final BINARY, so an allocation A/B cannot be done inside one test
//! binary: this target is run at the base ref, then at the head ref, and the two
//! printed records are diffed. Nothing here asserts a cross-ref delta, because
//! nothing here can see one. What it asserts is that the instrument is alive; the
//! comparison is the caller's.
//!
//! **The base arm is an OVERLAY, and the sentence above hides that.** This file
//! does not exist at any ref it will be compared against -- it was written for the
//! comparison -- so "run it at the base ref" is not a checkout, and a reader who
//! takes it literally finds no such target and cannot reproduce the numbers. The
//! two lines appended to the manifest are the same `[[test]]` block this branch
//! adds, and they are appended rather than patched so the recipe does not depend
//! on the base manifest's shape. Pick a `WT` outside the repository.
//!
//! Run these FROM the checkout that contains this file -- the head side. That is
//! not a detail: `cp` below reads the file out of the current checkout, so running
//! the recipe from a checkout that predates this target fails at the copy with
//! "No such file or directory". Measured, by running it from `main`.
//!
//! ```text
//! BASE=<the base ref>
//! WT=../decoder-alloc-ab-base
//! git worktree add --detach "$WT" "$BASE"
//! cp crates/inference/tests/decoder_allocation_arm.rs "$WT/crates/inference/tests/"
//! printf '\n[[test]]\nname = "decoder_allocation_arm"\ntest = false\n' >> "$WT/crates/inference/Cargo.toml"
//! (cd "$WT" && cargo test -p lattice-inference --test decoder_allocation_arm \
//!    --features f16 -- --nocapture --test-threads=1)
//! ```
//!
//! Run the same command in the ordinary checkout for the head arm, and diff the
//! two printed record blocks. Remove the overlay worktree afterwards; it carries
//! an untracked file and a modified manifest by construction, so it will never
//! read clean and must not be mistaken for a working branch.
//!
//! **Why its own target.** A global allocator applies to the whole binary it is
//! linked into. Putting one in an existing test file would silently instrument
//! every test beside it and change their allocation behaviour.
//!
//! **Counters are thread-local, not process-wide**, inherited from
//! `crates/tune/tests/lora_apply_no_alloc.rs` rather than rediscovered. The test
//! body runs on a harness-spawned thread while other harness threads allocate
//! during the measured window; a process-wide counter attributes that to the
//! subject and produces an intermittent delta with no code change.
//!
//! **What D1 actually forbids** is materializing an owned vocabulary-sized vector
//! to cross the decoder interface. So the assertion is NOT "zero allocations" --
//! `generate()` legitimately allocates -- and the record carries a large-allocation
//! histogram, not just a call count.
//!
//! **A pre-existing large allocation that must NOT be read as a driver cost.**
//! `prefill_tokens_batched_for_generate` (`crates/inference/src/forward/batch_prefill.rs`)
//! returns an owned vocabulary-sized `Vec<f32>`. It predates this work and is
//! present on BOTH arms. It is the reason the record prints a histogram the
//! reader subtracts across refs rather than a pass/fail this file decides.
//!
//! **Run it** (not run by `cargo test --workspace`; the target is declared
//! `test = false` in `crates/inference/Cargo.toml`, a manifest declaration a
//! reader can grep rather than a runtime branch that renders as a pass):
//!
//! ```bash
//! LATTICE_CPU_GREEDY_MODEL_DIR=/abs/path/to/qwen3.5-0.8b \
//!   cargo test -p lattice-inference --test decoder_allocation_arm \
//!   --features f16 -- --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` is load-bearing: two measurement bodies on two threads have
//! two independent thread-local counters, which is correct, but they would
//! contend for the same CPU and the byte totals would interleave in the printed
//! record in an order the reader cannot reconstruct.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// Allocations at or above this many bytes are recorded separately.
///
/// A vocabulary-sized `f32` buffer for this model family is on the order of
/// 600 KB (about 151k entries, 4 bytes each). The threshold sits well below
/// that and well above the incidental `String`/`Vec` traffic of a decode loop,
/// so a materialized logits-width buffer lands in the histogram and ordinary
/// bookkeeping does not.
///
/// It is a named constant rather than a value read off the model because
/// `Qwen35Model` exposes no public vocabulary accessor. Deriving it from a
/// private field would make this file depend on internals it has no other
/// reason to know; stating it, and stating why, keeps the number auditable.
const LARGE_ALLOC_BYTES: usize = 256 * 1024;

struct CountingAlloc;

thread_local! {
    static ALLOC_CALLS: Cell<u64> = const { Cell::new(0) };
    static ALLOC_BYTES: Cell<u64> = const { Cell::new(0) };
    static LARGE_CALLS: Cell<u64> = const { Cell::new(0) };
    static MAX_BYTES: Cell<usize> = const { Cell::new(0) };
}

fn note(size: usize) {
    ALLOC_CALLS.with(|c| c.set(c.get() + 1));
    ALLOC_BYTES.with(|c| c.set(c.get() + size as u64));
    if size >= LARGE_ALLOC_BYTES {
        LARGE_CALLS.with(|c| c.set(c.get() + 1));
    }
    MAX_BYTES.with(|c| {
        if size > c.get() {
            c.set(size);
        }
    });
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        // SAFETY: `System` imposes the same contract on `alloc` that this impl's
        // caller has already satisfied, and `layout` reaches it unchanged.
        // Recording is side-effect free with respect to that contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: every pointer this allocator hands out comes from `System`, so
        // a `ptr`/`layout` pair the caller validly passes here is one `System`
        // allocated under that same layout. Both are forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        // SAFETY: same forwarding invariant as `dealloc` for `ptr`/`layout`, and
        // `new_size` is passed through untouched for `System` to validate.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

// `alloc_zeroed` is deliberately NOT overridden. `GlobalAlloc`'s default
// implementation calls `self.alloc` and then zeroes the block, so zeroed
// allocations reach `note` through the override above. That is a property of the
// standard library rather than of this file, which is exactly why
// `counter_sees_zeroed_allocations` below pins it: a `vec![0.0f32; n]` -- the
// ordinary shape of the vocabulary-sized buffer this arm exists to detect --
// takes the zeroed path, and an override forwarding to `System.alloc_zeroed`
// without recording would make the subject invisible while every assertion here
// still passed.

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[derive(Debug, Clone, Copy)]
struct Record {
    calls: u64,
    bytes: u64,
    large_calls: u64,
    max_bytes: usize,
}

fn reset() {
    ALLOC_CALLS.with(|c| c.set(0));
    ALLOC_BYTES.with(|c| c.set(0));
    LARGE_CALLS.with(|c| c.set(0));
    MAX_BYTES.with(|c| c.set(0));
}

fn snapshot() -> Record {
    Record {
        calls: ALLOC_CALLS.with(Cell::get),
        bytes: ALLOC_BYTES.with(Cell::get),
        large_calls: LARGE_CALLS.with(Cell::get),
        max_bytes: MAX_BYTES.with(Cell::get),
    }
}

/// Checkpoint-free controls, in their own module so CI can name them.
///
/// The module boundary is the point: a `--skip <measurement>` filter would run
/// any test added later that DOES need a checkpoint, and fail in CI. Naming the
/// module runs exactly what is in it.
mod controls {
    /// The instrument's own liveness, with the control in the same body.
    ///
    /// An ordinary allocation must be counted first. Without that, a zero from the
    /// zeroed-path checks below would be indistinguishable from a dead counter, and
    /// a dead counter is the failure this whole target would report as "no
    /// allocations", which reads like good news.
    #[test]
    fn counter_sees_zeroed_allocations() {
        super::reset();
        let ordinary: Vec<u8> = Vec::with_capacity(4096);
        let after_ordinary = super::snapshot();
        std::hint::black_box(&ordinary);
        assert!(
            after_ordinary.calls > 0,
            "control failed: an ordinary Vec::with_capacity was not counted, so this \
             target can say nothing about anything"
        );

        super::reset();
        let zeroed: Vec<f32> = vec![0.0f32; 4096];
        let after_zeroed = super::snapshot();
        std::hint::black_box(&zeroed);
        assert!(
            after_zeroed.calls > 0,
            "vec![0.0f32; 4096] allocated without being counted: the default \
             alloc_zeroed no longer reaches CountingAlloc::alloc, so every \
             vocabulary-sized zeroed buffer is invisible to this arm"
        );
    }

    /// The detector must be able to express the thing it is looking for.
    ///
    /// A histogram that has never registered a large allocation proves nothing about
    /// a run in which none appears. This deliberately materializes a buffer of the
    /// width D1 forbids across the interface and shows the check catching it.
    #[test]
    fn vocabulary_sized_allocation_is_detected() {
        const VOCAB_ISH: usize = 151_936;
        let threshold = super::LARGE_ALLOC_BYTES;

        super::reset();
        let before = super::snapshot();
        assert_eq!(
            before.large_calls, 0,
            "a reset record already carries a large allocation"
        );

        let logits_width: Vec<f32> = vec![0.0f32; VOCAB_ISH];
        let after = super::snapshot();
        std::hint::black_box(&logits_width);

        assert!(
            after.large_calls > before.large_calls,
            "a {VOCAB_ISH}-entry f32 buffer ({} bytes) did not register as a large \
             allocation against a {threshold}-byte threshold",
            VOCAB_ISH * 4
        );
        assert!(
            after.max_bytes >= VOCAB_ISH * 4,
            "max_bytes {} is below the buffer just allocated ({} bytes)",
            after.max_bytes,
            VOCAB_ISH * 4
        );
    }
}

/// The measurement. Prints two machine-readable records per case -- a `record`
/// line for `generate` and a `record_streaming` line for `generate_streaming`
/// with a no-op callback, in the same column layout -- and asserts only what is
/// true at any ref; the base-vs-head comparison is the caller's.
///
/// Every `generate` measurement runs, and every `record` line is printed, before
/// the first `generate_streaming` call, so the `record` lines stay comparable with
/// runs that predate the streaming records.
#[test]
fn measure_generate_allocations() {
    use lattice_inference::GenerateConfig;
    use lattice_inference::model::qwen35::Qwen35Model;

    let model_dir = std::env::var("LATTICE_CPU_GREEDY_MODEL_DIR")
        .or_else(|_| std::env::var("LATTICE_MODEL_DIR"))
        .unwrap_or_else(|_| {
            panic!(
                "neither LATTICE_CPU_GREEDY_MODEL_DIR nor LATTICE_MODEL_DIR is set. \
                 This target enforces by default rather than skipping: a skip on a \
                 machine without the checkpoint renders as a pass, and an allocation \
                 arm that silently did not run is the failure it exists to prevent."
            )
        });
    let model_dir = std::path::PathBuf::from(model_dir);
    assert!(
        model_dir.is_dir(),
        "checkpoint {model_dir:?} is not a directory"
    );

    // The prompts come from the greedy golden's own fixture so the measured path
    // is the one the correctness gate pins, rather than a second set of prompts
    // that could exercise a different branch.
    let fixture =
        include_str!("fixtures/cpu_pre_migration_greedy_v1/qwen35_0_8b_cpu_greedy_tokens.json");
    let golden: serde_json::Value = serde_json::from_str(fixture).expect("golden fixture parses");
    let cases = golden["cases"]
        .as_array()
        .expect("golden fixture declares cases");

    let model = Qwen35Model::from_safetensors(&model_dir)
        .unwrap_or_else(|e| panic!("loading {model_dir:?} failed: {e}"));

    println!("# decoder_allocation_arm records; large threshold {LARGE_ALLOC_BYTES} bytes");
    println!("# case\tcalls\tbytes\tlarge_calls\tmax_bytes");

    let config_for = |case: &serde_json::Value| {
        let mut cfg = GenerateConfig::default();
        cfg.max_new_tokens = golden["max_new_tokens"].as_u64().expect("max_new_tokens") as usize;
        cfg.temperature = golden["generation"]["temperature"]
            .as_f64()
            .expect("temperature") as f32;
        cfg.repetition_penalty = golden["generation"]["repetition_penalty"]
            .as_f64()
            .expect("repetition_penalty") as f32;
        cfg.seed = golden["generation"]["seed"].as_u64();
        cfg.enable_thinking = golden["generation"]["enable_thinking"]
            .as_bool()
            .expect("enable_thinking");
        cfg.reasoning_budget = case["reasoning_budget"].as_u64().map(|v| v as usize);
        cfg
    };

    let mut any = 0usize;
    for case in cases {
        let name = case["name"].as_str().expect("case has a name");
        let prompt = case["prompt"].as_str().expect("case has a prompt");
        let cfg = config_for(case);

        // Warm once OUTSIDE the measured window. The first generation pulls in
        // lazily-initialized state that belongs to neither arm, and attributing
        // it to whichever case ran first would show up as a per-case difference
        // that has nothing to do with the diff under test.
        let _ = model.generate(prompt, &cfg);

        reset();
        let output = model
            .generate(prompt, &cfg)
            .unwrap_or_else(|e| panic!("case {name}: generation failed: {e}"));
        let rec = snapshot();
        std::hint::black_box(&output);

        assert!(
            rec.calls > 0,
            "case {name}: generate() recorded zero allocations, which is not a \
             result but a dead counter"
        );
        println!(
            "record\t{name}\t{}\t{}\t{}\t{}",
            rec.calls, rec.bytes, rec.large_calls, rec.max_bytes
        );
        any += 1;
    }

    println!("# record_streaming: generate_streaming with a no-op callback; columns as above");

    let mut any_streaming = 0usize;
    for case in cases {
        let name = case["name"].as_str().expect("case has a name");
        let prompt = case["prompt"].as_str().expect("case has a prompt");
        let cfg = config_for(case);

        // Warmed separately: the streaming entry is a different call path from
        // `generate`, so the `generate` warm-up above does not stand in for it.
        let _ = model.generate_streaming(prompt, &cfg, |_delta: &str| {});

        reset();
        let output = model
            .generate_streaming(prompt, &cfg, |_delta: &str| {})
            .unwrap_or_else(|e| panic!("case {name}: streaming generation failed: {e}"));
        let rec = snapshot();
        std::hint::black_box(&output);

        assert!(
            rec.calls > 0,
            "case {name}: generate_streaming() recorded zero allocations, which is \
             not a result but a dead counter"
        );
        println!(
            "record_streaming\t{name}\t{}\t{}\t{}\t{}",
            rec.calls, rec.bytes, rec.large_calls, rec.max_bytes
        );
        any_streaming += 1;
    }

    assert!(
        any > 0,
        "the golden fixture declared no cases, so nothing was measured"
    );
    assert_eq!(
        any_streaming, any,
        "generate_streaming measured a different number of cases than generate"
    );
    println!("# measured {any} case(s)");
}

# Implementation report: `lm_head` Criterion bench group (#547)

## What changed

- Added a new Criterion bench target, `lm_head_bench`, registered in
  `crates/inference/Cargo.toml` with no `required-features`, so it always
  compiles under default features and prints a clean skip when the real
  measurement path isn't available.
- Added `crates/inference/benches/lm_head_bench.rs`. It defines a Criterion
  group named `lm_head`, parameterized over:
  - quant path: `q8_tied_f16_wide_gemv`, `q4_gemv`
  - route: `full`, `block_argmax`, `block_topk_k8`, `block_topk_k16`,
    `block_topk_k40`, `block_topk_k64`

  giving benchmark IDs like `lm_head/q8_tied_f16_wide_gemv/block_topk_k40`.
- Added a `bench-internals`-gated `bench_support` module inside
  `crates/inference/src/forward/metal_qwen35.rs` (nested in the crate-private
  `inner` module, so it can reach `encode_final_head`,
  `read_topk_candidates`, and the other private dispatch helpers without
  loosening their visibility). It exposes `LmHeadBenchFixture`, which:
  - `prepare_hidden_for_bench` runs one real `forward_step` so
    `session.activations.hidden` holds a live hidden state before timing.
  - `run_once` sets `session.compact_route`/`compact_topk` for the requested
    route, opens one command buffer, calls the real `encode_final_head`
    (the same entry point the production decode path and
    `LATTICE_DECODE_PROFILE` use), commits, waits, and reads back a minimal
    marker (one `f32` bit-pattern for the full route, or the first candidate
    token id for the block routes) to force GPU completion.

No Metal shader, dispatch geometry, or route-selection logic changed — this
only adds a measurement harness around the existing kernels.

## Running it

```bash
# default features: compiles, prints a skip, exits 0 (this is a PASS)
cargo bench -p lattice-inference --bench lm_head_bench -- "lm_head"

# real measurement on macOS with a checkpoint present
cargo bench -p lattice-inference --features metal-gpu,f16,bench-internals --bench lm_head_bench -- "lm_head"
```

## Skip-when-absent behavior

- Default features (no `metal-gpu`/`f16`/`bench-internals`, or non-macOS):
  the bench function itself is compiled out and replaced with a stub that
  prints `SKIP: lm_head benchmark requires macOS + features
  metal-gpu,f16,bench-internals` and returns. No panic, no failure.
- Real-feature build with no checkpoint found: each quant path is probed
  independently. A missing Q8 or Q4 checkpoint (or, for Q4, a missing
  tokenizer) prints `SKIP: lm_head/<quant> checkpoint not found; searched:
  <paths>` and that quant path is skipped while the other still runs if its
  checkpoint is present.
- Checkpoint discovery reuses the existing conventions already used
  elsewhere in this bench suite, rather than inventing a new one:
  `LATTICE_INFERENCE_MODEL_DIR` (Q8), `LATTICE_MODEL_DIR` /
  `LATTICE_TOKENIZER_DIR` (Q4), and `LATTICE_MODEL_CACHE` /
  `~/.lattice/models/...` as the fallback root.
- A checkpoint that's present but fails to load (bad config, corrupt
  tensors) is treated as a real setup error (`Err`, surfaced via `expect` in
  the bench body), not a silent skip — that distinction matters so a broken
  local checkpoint doesn't get misreported as "no checkpoint".

## Verification performed this session

- `cargo check --workspace` — passes.
- `cargo check -p lattice-inference --bench lm_head_bench` (default
  features) — compiles clean.
- `cargo check -p lattice-inference --features metal-gpu,f16,bench-internals
  --bench lm_head_bench` — compiles clean.
- `cargo bench -p lattice-inference --bench lm_head_bench -- "lm_head"`
  (default features) — printed the feature skip and exited 0.
- `cargo bench -p lattice-inference --features
  metal-gpu,f16,bench-internals --bench lm_head_bench -- "lm_head"` — ran
  end-to-end against the real local Q8 and Q4-QuaRot 0.8B checkpoints,
  producing timings for all 12 quant/route combinations.
- `cargo clippy --workspace -- -D warnings` — clean.
- `cargo clippy -p lattice-inference --features metal-gpu,f16,bench-internals
  --bench lm_head_bench -- -D warnings` — clean.
- `cargo fmt --all -- --check` — clean.

No speedup numbers are claimed here — this change only adds the measurement
harness.

## Follow-up noticed

- `scripts/bench-compare.sh` and `make bench-ci`/`make bench-gate` still
  target `elementwise_cpu_bench` only, by design (this bench needs a local
  checkpoint and Metal GPU, so it can't run in the default CI A/B path). If
  `lm_head` needs standard `make bench-compare` treatment later, that script
  would need an env-overridable bench target/feature list rather than
  changing its CPU-only default.
- `session.compact_topk`/`compact_route` are reused directly by
  `bench_support::run_once` for fidelity with the production dispatch path,
  but the block-route readback in this bench uses the route's own `local_k`
  rather than `session.compact_topk`, since `encode_final_head`'s block
  branch doesn't consult `compact_topk` — worth keeping in mind if a future
  change makes the block path start reading it.

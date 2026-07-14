# Metal context-budget admission

## Changes

- Added a fallible `MetalQwen35State::generate_with_speculation` entry point that checks the prompt plus requested decode cap before the first Metal forward step.
- Unified direct, streaming, prefix-cache, multimodal, and LoRA-mixture admission on the shared typed context-budget error. Prompt-over-window requests no longer return successful `KvFull` completions.
- Reused the direct generator preflight in `generate_with_lora_mixture` before adapter unload, CPU blending, Metal upload, cache invalidation, or generation-state reset.
- Updated context-budget diagnostics to report the effective decode cap and the reasoning budget when present.
- Added device-required Metal regressions for exact-fit and one-past-window admission, speculative generation, and LoRA adapter/cache preservation.

## Verification

- `cargo fmt --all -- --check`
- `cargo clippy -p lattice-inference --all-targets --features metal-gpu -- -D warnings`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test -p lattice-inference`: 1,806 library tests passed, 18 ignored; all binary, integration, and doc-test targets passed.
- `LATTICE_METAL_TEST_ENFORCE=1 cargo test -p lattice-inference --features metal-gpu context_ -- --nocapture --test-threads=1`: 22 passed, including all three new Metal regressions.

Mutation verification:

- Removing the shared direct/LoRA context check made `metal_generation_entrypoints_enforce_context_admission` fail because a 32-token prompt plus one requested token returned `Ok`.
- Moving LoRA-mixture admission below adapter blend/upload made `lora_mixture_context_rejection_preserves_adapter_and_cache` fail because the warm prefix-cache entry was cleared.
- Both guards were restored and both focused tests passed.

## Performance

ADR-058 benchmarking is not applicable: admission runs once per request before decoding, and the Criterion suites (`elementwise_cpu_bench` and embedding SIMD) do not instantiate `MetalQwen35State`. No benchmark was run and no measured performance claim is made.
